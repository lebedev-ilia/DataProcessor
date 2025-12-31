from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass(frozen=True)
class CudaMemInfo:
    free_bytes: int
    total_bytes: int


def get_cuda_mem_info() -> Optional[CudaMemInfo]:
    """
    Best-effort GPU memory probe.
    Returns None if torch/cuda is not available.
    """
    try:
        import torch  # type: ignore
    except Exception:
        return None
    try:
        if not torch.cuda.is_available():
            return None
        free_b, total_b = torch.cuda.mem_get_info()
        return CudaMemInfo(free_bytes=int(free_b), total_bytes=int(total_b))
    except Exception:
        return None


def normalize_device(preferred: str) -> str:
    """
    Normalize device string to {"cpu","cuda"} when possible.
    """
    p = (preferred or "").strip().lower()
    if p in ("cuda", "gpu"):
        return "cuda"
    return "cpu"


def pick_device(preferred: str = "auto") -> str:
    """
    Pick device with best-effort torch probe. Never raises.
    """
    p = (preferred or "").strip().lower()
    if p == "cpu":
        return "cpu"
    if p == "cuda":
        try:
            import torch  # type: ignore

            return "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            return "cpu"
    # auto
    try:
        import torch  # type: ignore

        return "cuda" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"


