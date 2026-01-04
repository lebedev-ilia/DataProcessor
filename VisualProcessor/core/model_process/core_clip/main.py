import argparse
import os
import sys
from datetime import datetime
from typing import List, Tuple, Optional

import numpy as np      # type: ignore
import torch            # type: ignore
from PIL import Image   # type: ignore

_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if _path not in sys.path:
    sys.path.append(_path)

from utils.frame_manager import FrameManager
from utils.batching import auto_batch_size
from utils.logger import get_logger
from utils.resource_probe import pick_device
from utils.utilites import load_metadata


NAME = "core_clip"
VERSION = "2.0"
SCHEMA_VERSION = "core_clip_npz_v1"
LOGGER = get_logger(NAME)

SHOT_QUALITY_PROMPTS: List[str] = [
    "cinematic shot, high-quality professional footage",
    "professional low-light cinematic footage",
    "good smartphone video quality",
    "poor smartphone video quality, grainy, noisy",
    "webcam low resolution footage",
    "screen recording of display",
    "cctv surveillance camera footage low quality",
]


def _require_frame_indices(meta: dict, name: str) -> List[int]:
    """
    Strict contract: frame sampling is owned by Segmenter/DataProcessor.
    Providers MUST use metadata[name].frame_indices and MUST NOT fallback.
    """
    block = meta.get(name)
    if not isinstance(block, dict) or "frame_indices" not in block:
        raise RuntimeError(
            f"{name} | metadata missing '{name}.frame_indices'. "
            "Segmenter must provide per-provider frame_indices. No fallback is allowed."
        )
    frame_indices = block.get("frame_indices")
    if not isinstance(frame_indices, list) or not frame_indices:
        raise RuntimeError(f"{name} | metadata '{name}.frame_indices' is empty/invalid.")
    return [int(x) for x in frame_indices]


def compute_text_embeddings(
    model,
    device: str,
    prompts: List[str],
) -> np.ndarray:
    import clip  # type: ignore

    text = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        emb = model.encode_text(text)
        emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-9)
    return emb.detach().cpu().numpy().astype(np.float32)


def init_clip(model_name: str, preferred_device: str = "auto") -> Tuple[torch.nn.Module, callable, str]:
    import clip # type: ignore

    device = pick_device(preferred_device)
    model, preprocess = clip.load(model_name, device=device)

    model.eval()

    LOGGER.info(
        f"{NAME} | CLIP initialized | model: {model_name} | device: {device}"
    )

    return model, preprocess, device


def compute_clip_embeddings(
    frame_manager: FrameManager,
    frame_indices: List[int],
    model_name: str,
    batch_size: int,
) -> np.ndarray:

    if not frame_indices:
        LOGGER.warning(f"{NAME} | No frame indices provided")
        return np.zeros((0, 0), dtype=np.float32)

    model, preprocess, device = init_clip(model_name)

    n_frames = len(frame_indices)

    embeddings_out = None
    embed_dim = None

    try:
        with torch.no_grad():
            for start in range(0, n_frames, batch_size):
                batch_ids = frame_indices[start : start + batch_size]

                images = []
                for idx in batch_ids:
                    frame = frame_manager.get(idx)
                    img = Image.fromarray(frame)
                    images.append(preprocess(img))

                batch_tensor = torch.stack(images).to(device)
                
                emb = model.encode_image(batch_tensor)

                # L2 normalization (standard CLIP practice)
                emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-9)

                emb_np = emb.cpu().numpy().astype(np.float32)

                if embeddings_out is None:
                    embed_dim = emb_np.shape[1]
                    embeddings_out = np.zeros((n_frames, embed_dim), dtype=np.float32)

                embeddings_out[start : start + len(batch_ids)] = emb_np

                if start % (batch_size * 10) == 0:
                    LOGGER.info(
                        f"{NAME} | processed {start + len(batch_ids)}/{n_frames}"
                    )
    finally:
        del model
        torch.cuda.empty_cache()

    return embeddings_out


def main():
    parser = argparse.ArgumentParser(description="Production CLIP per-frame embedding extractor")
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--rs-path", required=True)
    parser.add_argument("--model-name", default="ViT-B/32")
    parser.add_argument("--batch-size", type=str, default="32", help="Integer batch size or 'auto' (or 0)")
    args = parser.parse_args()

    meta_path = os.path.join(args.frames_dir, "metadata.json")
    meta = load_metadata(meta_path, NAME)

    total_frames = int(meta["total_frames"])

    frame_indices = _require_frame_indices(meta, NAME)
    LOGGER.info(f"{NAME} | sampled frames: {len(frame_indices)} / total={total_frames}")

    frame_manager = FrameManager(
        frames_dir=args.frames_dir,
        chunk_size=meta.get("chunk_size", 32),
        cache_size=meta.get("cache_size", 2),
    )

    # Init CLIP once and reuse for both image + text embeddings.
    model, preprocess, device = init_clip(args.model_name, preferred_device="auto")
    # Safe default; may be overridden by auto selection below.
    batch_size = 32
    try:
        # Resolve batch size (strictly: number or 'auto'/0)
        bs_raw = (args.batch_size or "").strip().lower()
        if bs_raw in ("auto", "0", ""):
            decision = auto_batch_size(
                device=device,
                frame_shape=(int(frame_manager.height), int(frame_manager.width), int(frame_manager.channels)),
                model_hint="clip",
                max_batch_cap=64,
                reserve_ratio=0.25,
                cpu_default=1,
            )
            batch_size = int(decision.batch_size)
            LOGGER.info(
                f"{NAME} | auto batch_size={batch_size} | reason={decision.reason} | "
                f"free={decision.free_bytes} total={decision.total_bytes} per_sample_est={decision.per_sample_bytes_est}"
            )
        else:
            batch_size = max(1, int(bs_raw))

        # --- image embeddings ---
        n_frames = len(frame_indices)
        embeddings_out = None
        embed_dim = None
        with torch.no_grad():
            for start in range(0, n_frames, batch_size):
                batch_ids = frame_indices[start : start + batch_size]
                images = []
                for idx in batch_ids:
                    frame = frame_manager.get(idx)
                    img = Image.fromarray(frame)
                    images.append(preprocess(img))
                batch_tensor = torch.stack(images).to(device)
                emb = model.encode_image(batch_tensor)
                emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-9)
                emb_np = emb.cpu().numpy().astype(np.float32)
                if embeddings_out is None:
                    embed_dim = int(emb_np.shape[1])
                    embeddings_out = np.zeros((n_frames, embed_dim), dtype=np.float32)
                embeddings_out[start : start + len(batch_ids)] = emb_np
                if start % (batch_size * 10) == 0:
                    LOGGER.info(f"{NAME} | processed {start + len(batch_ids)}/{n_frames}")

        embeddings = embeddings_out if embeddings_out is not None else np.zeros((0, 0), dtype=np.float32)

        # --- text embeddings for downstream modules (shot_quality) ---
        shot_quality_text_embeddings = compute_text_embeddings(
            model=model,
            device=device,
            prompts=SHOT_QUALITY_PROMPTS,
    )

    finally:
        del model
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    LOGGER.info(
        f"{NAME} | embeddings computed | shape: {embeddings.shape}"
    )

    frame_manager.close()

    out_dir = os.path.join(args.rs_path, NAME)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "embeddings.npz")

    created_at = datetime.utcnow().isoformat()
    meta_out = {
        "producer": NAME,
        "producer_version": VERSION,
        "schema_version": SCHEMA_VERSION,
        "created_at": created_at,
        "status": "ok" if len(frame_indices) > 0 else "empty",
        "empty_reason": None if len(frame_indices) > 0 else "no_frames",
        "model_name": args.model_name,
        "total_frames": int(total_frames),
        "batch_size": int(batch_size),
    }
    required_run_keys = ["platform_id", "video_id", "run_id", "sampling_policy_version", "config_hash"]
    missing = [k for k in required_run_keys if not meta.get(k)]
    if missing:
        raise RuntimeError(f"{NAME} | frames metadata missing required run identity keys: {missing}")
    for k in required_run_keys:
            meta_out[k] = meta.get(k)

    np.savez_compressed(
        out_path,
        # legacy fields (kept)
        version=VERSION,
        created_at=created_at,
        model_name=args.model_name,
        total_frames=total_frames,
        frame_indices=np.array(frame_indices, dtype=np.int32),
        frame_embeddings=embeddings,
        # downstream contract for shot_quality
        shot_quality_prompts=np.array(SHOT_QUALITY_PROMPTS, dtype=object),
        shot_quality_text_embeddings=shot_quality_text_embeddings,
        # canonical meta (required by artifact_validator)
        meta=np.asarray(meta_out, dtype=object),
    )

    LOGGER.info(f"{NAME} | Saved result: {out_path}")


if __name__ == "__main__":
    main()