#!/usr/bin/env python3
"""
core_optical_flow

Core provider: RAFT optical flow (torchvision) → lightweight motion curves for downstream modules.

Contract:
- Segmenter provides `metadata["core_optical_flow"]["frame_indices"]` (union-domain indices).
- This provider MUST NOT invent sampling.
- Frames from FrameManager.get(idx) are RGB uint8 (HxWx3).

Output:
- <rs_path>/core_optical_flow/flow.npz
  Keys:
    - frame_indices: int32 (N,)
    - motion_px_mean: float32 (N,)        # mean magnitude in pixels (0 for first frame)
    - motion_px_per_sec_mean: float32 (N,)# motion_px_mean / dt_seconds (NaN for first frame)
    - dt_seconds: float32 (N,)            # NaN for first frame
    - meta: object(dict)                  # at least {producer, created_at}
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np  # type: ignore

try:
    import torch  # type: ignore
    import torch.nn.functional as F  # type: ignore
    import torchvision.models.optical_flow as tv_flow  # type: ignore
except Exception as e:  # pragma: no cover
    torch = None  # type: ignore
    F = None  # type: ignore
    tv_flow = None  # type: ignore
    _TORCH_IMPORT_ERR = e

_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if _path not in sys.path:
    sys.path.append(_path)

from utils.frame_manager import FrameManager
from utils.logger import get_logger
from utils.utilites import load_metadata

NAME = "core_optical_flow"
VERSION = "2.0"
LOGGER = get_logger(NAME)


def _require_frame_indices(meta: dict, name: str) -> List[int]:
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


def _get_union_timestamps_sec(frame_manager: FrameManager) -> Optional[np.ndarray]:
    meta = getattr(frame_manager, "meta", None)
    if not isinstance(meta, dict):
        return None
    ts = meta.get("union_timestamps_sec")
    if not isinstance(ts, list) or not ts:
        return None
    try:
        return np.asarray(ts, dtype=np.float32)
    except Exception:
        return None


def _resize_torch(img: torch.Tensor, max_dim: int) -> torch.Tensor:
    # img: (3,H,W) float32 in [0,1]
    _, h, w = img.shape
    if max(h, w) <= max_dim:
        return img
    if h >= w:
        new_h = int(max_dim)
        new_w = int(round(w * (max_dim / float(h))))
    else:
        new_w = int(max_dim)
        new_h = int(round(h * (max_dim / float(w))))
    new_h = max(new_h, 8)
    new_w = max(new_w, 8)
    out = F.interpolate(img.unsqueeze(0), size=(new_h, new_w), mode="bilinear", align_corners=False).squeeze(0)
    return out


def _pad_to_8(img: torch.Tensor) -> Tuple[torch.Tensor, Tuple[int, int]]:
    # RAFT expects H,W divisible by 8
    _, h, w = img.shape
    pad_h = (8 - (h % 8)) % 8
    pad_w = (8 - (w % 8)) % 8
    if pad_h == 0 and pad_w == 0:
        return img, (h, w)
    img = F.pad(img, (0, pad_w, 0, pad_h), mode="constant", value=0.0)
    return img, (h, w)


def _flow_mean_magnitude(flow: torch.Tensor) -> float:
    # flow: (2,H,W)
    mag = torch.sqrt(flow[0] ** 2 + flow[1] ** 2)
    return float(mag.mean().item())


def main() -> None:
    parser = argparse.ArgumentParser(description="core_optical_flow (RAFT/torchvision) motion curve extractor")
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--rs-path", required=True)
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--model", type=str, choices=["small", "large"], default="small")
    parser.add_argument("--max-dim", type=int, default=256, help="Max side for RAFT input (trade speed vs accuracy)")
    args = parser.parse_args()

    if torch is None or tv_flow is None or F is None:  # pragma: no cover
        raise RuntimeError(f"{NAME} | torch/torchvision not available: {_TORCH_IMPORT_ERR}")

    meta = load_metadata(os.path.join(args.frames_dir, "metadata.json"), NAME)
    total_frames = int(meta.get("total_frames", 0))
    frame_indices = _require_frame_indices(meta, NAME)
    LOGGER.info(f"{NAME} | sampled frames: {len(frame_indices)} / total={total_frames}")

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        device = "cpu"
    dev = torch.device(device)
    LOGGER.info(f"{NAME} | device={device} model={args.model} max_dim={args.max_dim}")

    if args.model == "large":
        model = tv_flow.raft_large(weights=tv_flow.Raft_Large_Weights.DEFAULT, progress=True).to(dev)
    else:
        model = tv_flow.raft_small(weights=tv_flow.Raft_Small_Weights.DEFAULT, progress=True).to(dev)
    model.eval()

    frame_manager = FrameManager(
        frames_dir=args.frames_dir,
        chunk_size=meta.get("chunk_size", 32),
        cache_size=meta.get("cache_size", 2),
    )

    union_ts = _get_union_timestamps_sec(frame_manager)
    idx_np = np.asarray(frame_indices, dtype=np.int32)
    n = int(idx_np.size)

    motion_px = np.zeros((n,), dtype=np.float32)
    dt_seconds = np.full((n,), np.nan, dtype=np.float32)
    motion_px_per_sec = np.full((n,), np.nan, dtype=np.float32)

    try:
        with torch.no_grad():
            prev_img: Optional[torch.Tensor] = None
            prev_t: Optional[float] = None
            for i, fi in enumerate(idx_np.tolist()):
                frame_rgb = frame_manager.get(int(fi))  # RGB uint8
                img = torch.from_numpy(frame_rgb).to(dev).permute(2, 0, 1).float() / 255.0
                img = _resize_torch(img, int(args.max_dim))
                img, _ = _pad_to_8(img)

                if prev_img is None:
                    prev_img = img
                    prev_t = float(union_ts[int(fi)]) if union_ts is not None and int(fi) < union_ts.size else None
                    continue

                # dt
                cur_t = float(union_ts[int(fi)]) if union_ts is not None and int(fi) < union_ts.size else None
                if prev_t is not None and cur_t is not None:
                    dt = max(cur_t - prev_t, 1e-6)
                else:
                    fps = float(getattr(frame_manager, "fps", 30.0) or 30.0)
                    dt = 1.0 / max(fps, 1e-6)

                flows = model(prev_img.unsqueeze(0), img.unsqueeze(0))
                flow = flows[-1].squeeze(0)  # (2,H,W)
                mag_mean = _flow_mean_magnitude(flow)

                motion_px[i] = float(mag_mean)
                dt_seconds[i] = float(dt)
                motion_px_per_sec[i] = float(mag_mean / dt)

                prev_img = img
                prev_t = cur_t if cur_t is not None else (prev_t + dt if prev_t is not None else None)

                if i % 50 == 0:
                    LOGGER.info(f"{NAME} | processed {i+1}/{n}")
    finally:
        try:
            frame_manager.close()
        except Exception:
            pass

    out_dir = os.path.join(args.rs_path, NAME)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "flow.npz")

    meta_out: Dict[str, Any] = {
        "producer": NAME,
        "producer_version": VERSION,
        "created_at": datetime.utcnow().isoformat(),
        "status": "ok" if n > 0 else "empty",
        "empty_reason": None if n > 0 else "no_frames",
        "model_name": f"raft_{args.model}",
        "device": str(device),
        "max_dim": int(args.max_dim),
        "total_frames": int(total_frames),
    }
    # attach run context if present in frames_dir metadata (Segmenter writes these fields)
    for k in ["platform_id", "video_id", "run_id", "sampling_policy_version", "config_hash"]:
        if k in meta:
            meta_out[k] = meta.get(k)

    np.savez_compressed(
        out_path,
        frame_indices=idx_np,
        motion_px_mean=motion_px,
        motion_px_per_sec_mean=motion_px_per_sec,
        dt_seconds=dt_seconds,
        meta=np.asarray(meta_out, dtype=object),
    )

    LOGGER.info(f"{NAME} | Saved result: {out_path}")


if __name__ == "__main__":
    main()


