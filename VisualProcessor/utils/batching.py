from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence, Tuple, Union

from utils.resource_probe import get_cuda_mem_info


@dataclass(frozen=True)
class AutoBatchDecision:
    device: str
    batch_size: int
    reason: str
    free_bytes: Optional[int] = None
    total_bytes: Optional[int] = None
    per_sample_bytes_est: Optional[int] = None
    budget_bytes: Optional[int] = None


def estimate_image_bytes(
    h: int,
    w: int,
    channels: int = 3,
    dtype_bytes: int = 4,
    safety_mult: float = 8.0,
) -> int:
    """
    Very rough upper-bound estimate of *working* memory per sample.

    Notes:
    - We intentionally over-estimate to avoid OOMs.
    - This is NOT just input tensor bytes; it also tries to account for activations,
      intermediate buffers, and framework overhead via safety_mult.
    """
    hh = max(1, int(h))
    ww = max(1, int(w))
    cc = max(1, int(channels))
    db = max(1, int(dtype_bytes))

    base = hh * ww * cc * db
    est = int(math.ceil(float(base) * float(max(1.0, safety_mult))))
    return max(est, base)


def _model_safety_mult(model_hint: str) -> float:
    """
    Model-specific (very rough) safety multipliers.
    """
    h = (model_hint or "").strip().lower()
    if h in ("clip", "core_clip"):
        return 10.0
    if h in ("midas", "depth", "core_depth_midas"):
        return 18.0
    if h in ("yolo", "ultralytics", "object_detections", "core_object_detections"):
        return 12.0
    return 12.0


def auto_batch_size(
    device: str,
    frame_shape: Union[Sequence[int], Tuple[int, int], Tuple[int, int, int]],
    model_hint: str = "generic",
    max_batch_cap: int = 64,
    reserve_ratio: float = 0.25,
    cpu_default: int = 1,
) -> AutoBatchDecision:
    """
    Pick a conservative batch size based on current free CUDA memory.

    Conventions:
    - If device != "cuda" or CUDA probe unavailable -> return cpu_default.
    - frame_shape can be (H,W) or (H,W,C).
    """
    dev = (device or "").strip().lower()
    cap = max(1, int(max_batch_cap))

    # Normalize shape
    h = w = 0
    c = 3
    try:
        if len(frame_shape) >= 2:
            h = int(frame_shape[0])
            w = int(frame_shape[1])
        if len(frame_shape) >= 3:
            c = int(frame_shape[2])
    except Exception:
        h, w, c = 0, 0, 3

    if dev != "cuda":
        bs = max(1, int(cpu_default))
        return AutoBatchDecision(device=dev or "cpu", batch_size=min(bs, cap), reason="non-cuda-device")

    mem = get_cuda_mem_info()
    if mem is None or mem.free_bytes <= 0:
        bs = max(1, int(cpu_default))
        return AutoBatchDecision(device="cuda", batch_size=min(bs, cap), reason="cuda-mem-probe-unavailable")

    # Budget for this process:
    rr = float(reserve_ratio)
    rr = min(max(rr, 0.0), 0.9)
    budget = int(mem.free_bytes * (1.0 - rr))
    budget = max(0, budget)

    # Estimate bytes per sample (assume float32 working tensors).
    per_sample = estimate_image_bytes(
        h=h,
        w=w,
        channels=c,
        dtype_bytes=4,
        safety_mult=_model_safety_mult(model_hint),
    )
    if per_sample <= 0:
        return AutoBatchDecision(
            device="cuda",
            batch_size=1,
            reason="invalid-per-sample-estimate",
            free_bytes=int(mem.free_bytes),
            total_bytes=int(mem.total_bytes),
            per_sample_bytes_est=int(per_sample),
            budget_bytes=int(budget),
        )

    bs = int(budget // per_sample) if budget > 0 else 0
    bs = max(1, min(bs, cap))

    return AutoBatchDecision(
        device="cuda",
        batch_size=int(bs),
        reason="cuda-mem-auto",
        free_bytes=int(mem.free_bytes),
        total_bytes=int(mem.total_bytes),
        per_sample_bytes_est=int(per_sample),
        budget_bytes=int(budget),
    )


