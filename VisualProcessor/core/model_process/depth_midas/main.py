#!/VisualProcessor/core/model_process/.model_process_venv python3
"""
Production-ready depth extraction using MiDaS.

Design decisions and behavior (short):
- We assume FrameManager.get(idx) returns an RGB uint8 HxWx3 image (this matches other modules).
  If your FrameManager returns BGR images, set --frames-bgr to True.
- Outputs:
  * <rs_path>/core_depth_midas/depth.npz  -- compressed NPZ containing:
      - depth_maps: float32 array (N, out_h, out_w) with per-sampled-frame depth (NaN for failures)
      - frame_indices: int32 (N,)
      - version, model_name, created_at, total_frames, sample_step
  * optionally per-frame full-resolution .npy files under <rs_path>/core_depth_midas/maps/ (if --save-full-res)
- We default to MiDaS_small (fast) but loader is isolated so model swap is straightforward.
"""

import argparse
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional

import numpy as np      # type: ignore
import cv2              # type: ignore
import torch            # type: ignore
from PIL import Image   # type: ignore

_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if _path not in sys.path:
    sys.path.append(_path)

from utils.frame_manager import FrameManager
from utils.logger import get_logger
from utils.utilites import load_metadata

VERSION = "2.0"
NAME = "core_depth_midas"
LOGGER = get_logger(NAME)


def load_midas(model_family: str = "intel-isl/MiDaS", model_name: str = "MiDaS_small", device: str = "cuda"):
    """
    Load MiDaS model and transforms.

    Returns:
        model, transform_callable
    Notes:
        - We use torch.hub to keep parity with original code, wrapped in try/except.
        - model is moved to device and set to eval().
    """
    LOGGER.info(f"{NAME} | load_midas | model_family={model_family} model_name={model_name} device={device}")
    try:
        model = (
            torch.hub.load(
                model_family,
                model_name,
                pretrained=True,
                trust_repo=True,
                verbose=False,
            )
            .to(device)
            .eval()
        )
    except Exception as e:
        LOGGER.error(f"{NAME} | load_midas | Failed to load model {model_family}/{model_name}: {e}")
        raise

    try:
        transforms_module = torch.hub.load(model_family, "transforms", trust_repo=True, verbose=False)
        # common MiDaS public API exposes different transforms for model sizes:
        # small_transform for MiDaS_small, default_transform for other variants.
        if hasattr(transforms_module, "small_transform"):
            transforms = transforms_module.small_transform
        elif hasattr(transforms_module, "default_transform"):
            transforms = transforms_module.default_transform
        else:
            transforms = transforms_module
    except Exception as e:
        LOGGER.error(f"{NAME} | load_midas | Failed to load transforms from {model_family}: {e}")
        raise

    LOGGER.info(f"{NAME} | load_midas | Loaded model and transforms successfully")
    return model, transforms


# -------------------------
# Core processing function
# -------------------------
def compute_depth_maps_batch(
    frame_manager: FrameManager,
    frame_indices: List[int],
    midas_model,
    midas_transform,
    device: str,
    out_size: Tuple[int, int],
    batch_size: int = 1,
    frames_are_bgr: bool = False,
    save_full_res: bool = False,
    maps_dir: Optional[str] = None,
) -> np.ndarray:
    """
    Compute depth maps for a list of frame indices.

    Returns:
        depth_maps: np.ndarray shape (N, out_h, out_w), dtype=float32
    Implementation notes:
    - We run MiDaS in batches when batch_size > 1 (midas_transform produces a tensor per PIL image).
    - We interpolate model predictions to the requested out_size with bicubic interpolation.
    - For failing frames we leave NaN in the output array.
    - If save_full_res==True and maps_dir provided, we save full-resolution depth (as float32 .npy)
      for each processed frame alongside the main NPZ artifact.
    """

    n = len(frame_indices)
    out_h, out_w = out_size
    LOGGER.info(f"{NAME} | compute_depth_maps_batch | frames={n} out_size={(out_h, out_w)} batch_size={batch_size}")

    # Preallocate with NaNs to indicate missing frames or failures.
    depth_maps = np.full((n, out_h, out_w), np.nan, dtype=np.float32)

    # If saving full-res, make sure maps_dir exists
    if save_full_res and maps_dir:
        os.makedirs(maps_dir, exist_ok=True)

    # Process in batches
    try:
        with torch.no_grad():
            for start in range(0, n, batch_size):
                batch_inds = frame_indices[start : start + batch_size]
                imgs = []
                orig_sizes = []

                # Load and transform frames
                for idx in batch_inds:
                    frame = frame_manager.get(idx)  # assume RGB by default
                    if frames_are_bgr:
                        # convert BGR -> RGB if requested (legacy compatibility)
                        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    # track original size if we optionally save full-res
                    orig_sizes.append(frame.shape[:2])  # (h, w)
                    tensor = midas_transform(frame)  # transform -> tensor expected by model
                    # Normalize shape:
                    # allow [3,H,W] or [1,3,H,W] → always [3,H,W]
                    if tensor.dim() == 4 and tensor.shape[0] == 1:
                        tensor = tensor.squeeze(0)
                    imgs.append(tensor)

                if not imgs:
                    continue

                batch_tensor = torch.stack(imgs).to(device)

                # Model forward
                pred = midas_model(batch_tensor)
                # MiDaS output shape may be (B, H', W') or (B, 1, H', W') depending on model variant.
                # Normalize shape to (B, H', W')
                if pred.dim() == 4 and pred.shape[1] == 1:
                    pred = pred.squeeze(1)

                # We need to resize predictions to desired out_size.
                # Use torch.nn.functional.interpolate for batch tensor.
                pred_resized = torch.nn.functional.interpolate(
                    pred.unsqueeze(1),
                    size=(out_h, out_w),
                    mode="bicubic",
                    align_corners=False,
                ).squeeze(1)

                pred_resized = pred_resized.cpu().numpy().astype(np.float32)  # (B, out_h, out_w)

                # Optionally save full resolution maps per-frame (before resize)
                if save_full_res and maps_dir:
                    for i_local, idx in enumerate(batch_inds):
                        full_pred = pred[i_local]  # may be smaller than original; we resample to orig_sizes
                        # Upsample full_pred to original frame size for best fidelity when saving full-res.
                        orig_h, orig_w = orig_sizes[i_local]
                        full_up = torch.nn.functional.interpolate(
                            full_pred.unsqueeze(0).unsqueeze(0),
                            size=(orig_h, orig_w),
                            mode="bicubic",
                            align_corners=False,
                        ).squeeze().cpu().numpy().astype(np.float32)
                        filename = os.path.join(maps_dir, f"depth_{int(idx):06d}.npy")
                        np.save(filename, full_up)

                # Store resized maps into the preallocated array
                for i_local in range(pred_resized.shape[0]):
                    depth_maps[start + i_local] = pred_resized[i_local]

                if (start // batch_size) % 10 == 0:
                    LOGGER.info(f"{NAME} | compute_depth_maps_batch | processed {min(start + batch_size, n)}/{n} frames")

    except Exception as e:
        LOGGER.error(f"{NAME} | compute_depth_maps_batch | error during processing: {e}")
        raise

    return depth_maps


# -------------------------
# CLI entrypoint
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Production MiDaS depth extractor")
    parser.add_argument("--frames-dir", type=str, required=True, help="Path to frames directory")
    parser.add_argument("--rs-path", type=str, required=True, help="Path to VisualProcessor result_store")
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda", help="Preferred device")
    parser.add_argument("--out-width", type=int, default=256, help="Output width of saved depth maps (downsampled) to store in NPZ")
    parser.add_argument("--out-height", type=int, default=256, help="Output height of saved depth maps (downsampled) to store in NPZ")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for MiDaS inference (memory sensitive)")
    parser.add_argument("--frames-bgr", action="store_true", help="Set if FrameManager returns BGR images instead of RGB")
    parser.add_argument("--save-full-res", action="store_true", help="Also save full-resolution per-frame .npy depth maps under maps/")
    parser.add_argument("--model-name", type=str, default="MiDaS_small", help="MiDaS model name to load from intel-isl/MiDaS")
    args = parser.parse_args()

    # Choose device: prefer cuda if available and requested
    device = args.device if (args.device == "cpu" or torch.cuda.is_available()) else "cpu"
    LOGGER.info(f"{NAME} | main | device selected: {device}")

    # Load model
    midas_model, midas_transform = load_midas(model_name=args.model_name, device=device)

    # Load metadata
    meta = load_metadata(os.path.join(args.frames_dir, "metadata.json"), NAME)
    total_frames = int(meta.get("total_frames", 0))

    # Strict sampling contract: Segmenter must provide per-provider indices in metadata[NAME].frame_indices.
    block = meta.get(NAME)
    if not isinstance(block, dict) or "frame_indices" not in block:
        raise RuntimeError(
            f"{NAME} | metadata missing '{NAME}.frame_indices'. "
            "Segmenter must provide per-provider frame_indices. No fallback is allowed."
        )
    frame_indices_raw = block.get("frame_indices")
    if not isinstance(frame_indices_raw, list) or not frame_indices_raw:
        raise RuntimeError(f"{NAME} | metadata '{NAME}.frame_indices' is empty/invalid.")
    frame_indices = [int(x) for x in frame_indices_raw]
    LOGGER.info(f"{NAME} | main | sampled frames: {len(frame_indices)} / total={total_frames}")

    # Initialize FrameManager
    frame_manager = FrameManager(
        frames_dir=args.frames_dir,
        chunk_size=meta.get("chunk_size", 32),
        cache_size=meta.get("cache_size", 2),
    )
    LOGGER.info(f"{NAME} | main | FrameManager initialized (chunk_size={meta.get('chunk_size', 32)}, cache_size={meta.get('cache_size', 2)})")

    # Prepare output dirs
    core_dir = os.path.join(args.rs_path, NAME)
    maps_dir = os.path.join(core_dir, "maps")
    os.makedirs(core_dir, exist_ok=True)
    if args.save_full_res:
        os.makedirs(maps_dir, exist_ok=True)

    # Compute depth maps (batched)
    out_size = (args.out_height, args.out_width)  # (H, W)
    try:
        depth_maps = compute_depth_maps_batch(
            frame_manager=frame_manager,
            frame_indices=frame_indices,
            midas_model=midas_model,
            midas_transform=midas_transform,
            device=device,
            out_size=out_size,
            batch_size=max(1, int(args.batch_size)),
            frames_are_bgr=args.frames_bgr,
            save_full_res=args.save_full_res,
            maps_dir=maps_dir if args.save_full_res else None,
        )
    finally:
        # Always close frame manager and free GPU memory
        try:
            frame_manager.close()
        except Exception:
            pass
        # explicit cleanup
        try:
            del midas_model
            torch.cuda.empty_cache()
        except Exception:
            pass

    out_path = os.path.join(core_dir, "depth.npz")
    created_at = datetime.utcnow().isoformat()

    meta_out: Dict[str, Any] = {
        "producer": NAME,
        "producer_version": VERSION,
        "created_at": created_at,
        "status": "ok" if len(frame_indices) > 0 else "empty",
        "empty_reason": None if len(frame_indices) > 0 else "no_frames",
        "model_name": args.model_name,
        "total_frames": int(total_frames),
        "out_width": int(args.out_width),
        "out_height": int(args.out_height),
        "batch_size": int(args.batch_size),
        "device": str(device),
    }
    for k in ["platform_id", "video_id", "run_id", "sampling_policy_version", "config_hash"]:
        if k in meta:
            meta_out[k] = meta.get(k)

    np.savez_compressed(
        out_path,
        # legacy fields (kept)
        version=VERSION,
        model_name=args.model_name,
        created_at=created_at,
        total_frames=total_frames,
        frame_indices=np.array(frame_indices, dtype=np.int32),
        depth_maps=depth_maps,  # shape (N, out_h, out_w), dtype float32
        # canonical meta (required by artifact_validator)
        meta=np.asarray(meta_out, dtype=object),
    )

    LOGGER.info(f"{NAME} | main | Saved NPZ artifact: {out_path} | created_at={created_at}")
    if args.save_full_res:
        LOGGER.info(f"{NAME} | main | Per-frame full-resolution maps saved under: {maps_dir}")


if __name__ == "__main__":
    main()
