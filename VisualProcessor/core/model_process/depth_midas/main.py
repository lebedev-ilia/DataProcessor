#!/VisualProcessor/core/model_process/.model_process_venv python3
"""
Production-ready depth extraction (MiDaS family) via Triton.

Design decisions and behavior (short):
- We assume FrameManager.get(idx) returns an RGB uint8 HxWx3 image (this matches other modules).
  If your FrameManager returns BGR images, set --frames-bgr to True.
- Outputs:
  * <rs_path>/core_depth_midas/depth.npz  -- compressed NPZ containing:
      - depth_maps: float32 array (N, out_h, out_w)
      - frame_indices: int32 (N,)
      - version, model_name, created_at, total_frames
- Triton-only: preprocessing lives in Triton; no torch.hub / no local torch engine.
"""

import argparse
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Tuple, Optional

import numpy as np      # type: ignore
import cv2              # type: ignore

_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if _path not in sys.path:
    sys.path.append(_path)
# repo root (needed for dp_triton)
_root = os.path.dirname(_path)
if _root not in sys.path:
    sys.path.append(_root)

from utils.frame_manager import FrameManager
from utils.logger import get_logger
from utils.utilites import load_metadata
from utils.meta_builder import apply_models_meta, model_used

VERSION = "2.0"
NAME = "core_depth_midas"
SCHEMA_VERSION = "core_depth_midas_npz_v1"
LOGGER = get_logger(NAME)


def _load_triton_spec_via_model_manager(model_spec_name: str) -> dict:
    """
    Resolve Triton model spec via dp_models.ModelManager (no-network, reproducible).
    Returns dict with keys:
      - client: TritonHttpClient
      - rp: runtime_params
      - models_used_entry: dict (model_used)
    """
    from dp_models import get_global_model_manager  # type: ignore

    mm = get_global_model_manager()
    rm = mm.get(model_name=str(model_spec_name))
    rp = rm.spec.runtime_params or {}
    handle = rm.handle or {}
    client = None
    if isinstance(handle, dict):
        client = handle.get("client")
    if client is None:
        raise RuntimeError(f"{NAME} | ModelManager returned empty Triton client handle for: {model_spec_name}")
    if not isinstance(rp, dict) or not rp:
        raise RuntimeError(f"{NAME} | ModelManager returned empty runtime_params for: {model_spec_name}")
    return {"client": client, "rp": rp, "models_used_entry": rm.models_used_entry}


def _preset_to_input_size(preset: str) -> int:
    p = str(preset or "").strip().lower()
    if p in ("midas_256", "256"):
        return 256
    if p in ("midas_384", "384"):
        return 384
    if p in ("midas_512", "512"):
        return 512
    raise ValueError(f"{NAME} | unknown triton_preprocess_preset: {preset!r}")


def _prep_batch_rgb_uint8(frames: List[np.ndarray], *, input_size: int, frames_are_bgr: bool) -> np.ndarray:
    """
    Minimal client-side formatting for Triton (NOT full preprocessing):
    - ensure RGB
    - resize to (S,S)
    - keep UINT8 NHWC (baseline GPU contract)

    Full preprocessing (normalize/layout conversion to model FP32 NCHW) lives in Triton ensemble.
    """
    s = int(input_size)
    if s <= 0:
        raise ValueError(f"{NAME} | invalid input_size={input_size}")
    out: List[np.ndarray] = []
    for fr in frames:
        if frames_are_bgr:
            fr = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        # Resize (square) for preset
        fr_r = cv2.resize(fr, (s, s), interpolation=cv2.INTER_AREA)
        # Keep UINT8 NHWC
        out.append(np.asarray(fr_r, dtype=np.uint8))
    if not out:
        return np.zeros((0, s, s, 3), dtype=np.uint8)
    return np.stack(out, axis=0).astype(np.uint8)


# -------------------------
# CLI entrypoint
# -------------------------
def main():
    parser = argparse.ArgumentParser(description="Production MiDaS depth extractor")
    parser.add_argument("--frames-dir", type=str, required=True, help="Path to frames directory")
    parser.add_argument("--rs-path", type=str, required=True, help="Path to VisualProcessor result_store")
    # Triton-only policy (prod): local torch/onnx engines are removed.
    parser.add_argument("--runtime", type=str, default="triton", choices=["triton"], help="Runtime (prod: triton only)")
    parser.add_argument("--triton-http-url", type=str, default=None)
    # Preferred: resolve Triton params via ModelManager specs (recommended; overrides explicit triton_* args when provided).
    parser.add_argument("--triton-model-spec", type=str, default=None, help="dp_models spec name (e.g., midas_256_triton)")
    parser.add_argument("--triton-model-name", type=str, default=None)
    parser.add_argument("--triton-model-version", type=str, default=None)
    parser.add_argument("--triton-input-name", type=str, default="INPUT__0")
    parser.add_argument("--triton-output-name", type=str, default="OUTPUT__0")
    # Triton ensemble expects UINT8 NHWC input.
    parser.add_argument("--triton-datatype", type=str, default="UINT8")
    parser.add_argument(
        "--triton-preprocess-preset",
        type=str,
        default="midas_384",
        choices=["midas_256", "midas_384", "midas_512"],
        help="Input preset (square size) for Triton depth model.",
    )
    parser.add_argument("--model-version", type=str, default="unknown")
    parser.add_argument("--weights-digest", type=str, default="unknown")
    parser.add_argument("--precision", type=str, default="fp32")
    parser.add_argument("--out-width", type=int, default=384, help="Output width of saved depth maps (downsampled) to store in NPZ")
    parser.add_argument("--out-height", type=int, default=384, help="Output height of saved depth maps (downsampled) to store in NPZ")
    parser.add_argument("--batch-size", type=int, required=True, help="Batch size (must be provided by scheduler/orchestrator)")
    parser.add_argument("--frames-bgr", action="store_true", help="Set if FrameManager returns BGR images instead of RGB")
    args = parser.parse_args()

    # Triton-only mode: device is not observable from client reliably; we assume GPU-backed Triton.
    runtime = str(args.runtime or "triton").strip().lower()
    if runtime != "triton":
        raise RuntimeError(f"{NAME} | runtime must be triton (no-fallback), got: {runtime}")

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
    if len(frame_indices) <= 0:
        raise RuntimeError(f"{NAME} | empty frame_indices is not allowed (no-fallback)")

    # Initialize FrameManager
    frame_manager = FrameManager(
        frames_dir=args.frames_dir,
        chunk_size=meta.get("chunk_size", 32),
        cache_size=meta.get("cache_size", 2),
    )
    LOGGER.info(f"{NAME} | main | FrameManager initialized (chunk_size={meta.get('chunk_size', 32)}, cache_size={meta.get('cache_size', 2)})")

    batch_size = int(args.batch_size)
    if batch_size <= 0:
        raise RuntimeError(f"{NAME} | --batch-size must be > 0 (scheduler-controlled); got {batch_size}")

    # Prepare output dir
    core_dir = os.path.join(args.rs_path, NAME)
    os.makedirs(core_dir, exist_ok=True)

    # Triton client (repo-local)
    from dp_triton import TritonHttpClient, TritonError  # type: ignore

    mm_entry = None
    if args.triton_model_spec:
        mm_entry = _load_triton_spec_via_model_manager(str(args.triton_model_spec))
        client = mm_entry["client"]
        rp = mm_entry["rp"]
        # Override explicit args from runtime_params (single source-of-truth).
        args.triton_http_url = str(rp.get("triton_http_url") or args.triton_http_url or "")
        args.triton_model_name = str(rp.get("triton_model_name") or args.triton_model_name or "")
        args.triton_model_version = str(rp.get("triton_model_version") or "") or None
        args.triton_input_name = str(rp.get("triton_input_name") or args.triton_input_name)
        args.triton_output_name = str(rp.get("triton_output_name") or args.triton_output_name)
        args.triton_datatype = str(rp.get("triton_input_datatype") or args.triton_datatype)
    else:
        if not args.triton_http_url or not str(args.triton_http_url).strip():
            raise RuntimeError(f"{NAME} | runtime=triton requires --triton-http-url or --triton-model-spec (no-fallback)")
        if not args.triton_model_name or not str(args.triton_model_name).strip():
            raise RuntimeError(f"{NAME} | runtime=triton requires --triton-model-name or --triton-model-spec (no-fallback)")
        client = TritonHttpClient(base_url=str(args.triton_http_url), timeout_sec=10.0)
        if not client.ready():
            raise TritonError(f"{NAME} | Triton is not ready at {args.triton_http_url}", error_code="triton_unavailable")

    input_size = _preset_to_input_size(str(args.triton_preprocess_preset))
    out_size = (int(args.out_height), int(args.out_width))  # (H,W)

    # Compute depth maps (batched)
    try:
        n = len(frame_indices)
        out_h, out_w = out_size
        depth_maps = np.full((n, out_h, out_w), np.nan, dtype=np.float32)

        for start in range(0, n, batch_size):
            batch_ids = frame_indices[start : start + batch_size]
            frames = [frame_manager.get(i) for i in batch_ids]
            inp = _prep_batch_rgb_uint8(frames, input_size=input_size, frames_are_bgr=bool(args.frames_bgr))
            try:
                res = client.infer(
                    model_name=str(args.triton_model_name),
                    model_version=str(args.triton_model_version) if args.triton_model_version else None,
                    input_name=str(args.triton_input_name),
                    input_tensor=inp,
                    output_name=str(args.triton_output_name),
                    datatype=str(args.triton_datatype),
                )
            except Exception as e:
                raise RuntimeError(f"{NAME} | Triton infer failed: {e}") from e

            out = np.asarray(res.output, dtype=np.float32)
            # Expect (B,1,h,w) or (B,h,w)
            if out.ndim == 4 and out.shape[1] == 1:
                out = out[:, 0, :, :]
            if out.ndim != 3:
                raise RuntimeError(f"{NAME} | Triton output has invalid shape: {out.shape}")
            if out.shape[0] != len(batch_ids):
                raise RuntimeError(f"{NAME} | Triton output batch mismatch: got {out.shape[0]} expected {len(batch_ids)}")

            for i_local in range(out.shape[0]):
                m = out[i_local]
                dm = cv2.resize(m, (out_w, out_h), interpolation=cv2.INTER_CUBIC).astype(np.float32)
                if not np.isfinite(dm).any():
                    raise RuntimeError(f"{NAME} | invalid depth map produced (NaN/empty) at frame_idx={batch_ids[i_local]}")
                depth_maps[start + i_local] = dm
    finally:
        # Always close frame manager and free GPU memory
        try:
            frame_manager.close()
        except Exception:
            pass

    out_path = os.path.join(core_dir, "depth.npz")
    created_at = datetime.utcnow().isoformat()

    meta_out: Dict[str, Any] = {
        "producer": NAME,
        "producer_version": VERSION,
        "schema_version": SCHEMA_VERSION,
        "created_at": created_at,
        "status": "ok",
        "empty_reason": None,
        "model_name": str(args.triton_model_name),
        "total_frames": int(total_frames),
        "out_width": int(args.out_width),
        "out_height": int(args.out_height),
        "batch_size": int(batch_size),
        "runtime": "triton-gpu",
        "device": "cuda",
        "triton_preprocess_preset": str(args.triton_preprocess_preset),
    }
    required_run_keys = ["platform_id", "video_id", "run_id", "sampling_policy_version", "config_hash"]
    missing = [k for k in required_run_keys if not meta.get(k)]
    if missing:
        raise RuntimeError(f"{NAME} | frames metadata missing required run identity keys: {missing}")
    for k in required_run_keys:
        meta_out[k] = meta.get(k)

    # PR-3: model system baseline
    meta_out = apply_models_meta(
        meta_out,
        models_used=[
            model_used(
                model_name=str(args.triton_model_name),
                model_version=str(args.model_version or "unknown"),
                weights_digest=str(args.weights_digest or "unknown"),
                runtime="triton-gpu",
                engine="onnx",
                precision=str(args.precision or "unknown"),
                device="cuda",
            )
        ],
    )

    np.savez_compressed(
        out_path,
        # legacy fields (kept)
        version=VERSION,
        model_name=str(args.triton_model_name),
        created_at=created_at,
        total_frames=total_frames,
        frame_indices=np.array(frame_indices, dtype=np.int32),
        depth_maps=depth_maps,  # shape (N, out_h, out_w), dtype float32
        # canonical meta (required by artifact_validator)
        meta=np.asarray(meta_out, dtype=object),
    )

    LOGGER.info(f"{NAME} | main | Saved NPZ artifact: {out_path} | created_at={created_at}")


if __name__ == "__main__":
    main()
