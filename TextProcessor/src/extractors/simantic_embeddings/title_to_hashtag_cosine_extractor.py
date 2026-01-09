from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from src.core.base_extractor import BaseExtractor
from src.core.metrics import system_snapshot, process_memory_bytes
from src.core.path_utils import default_artifacts_dir


def _latest(artifacts_dir: Path, pattern: str) -> Optional[Path]:
    files = glob.glob(str(artifacts_dir / pattern))
    if not files:
        return None
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return Path(files[0])


def _l2n(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n > 0:
        return v / n
    return v


class TitleToHashtagCosineExtractor(BaseExtractor):
    VERSION = "1.0.0"

    def __init__(self, artifacts_dir: str | None = None) -> None:
        self.artifacts_dir = Path(artifacts_dir).expanduser().resolve() if artifacts_dir else default_artifacts_dir()

    def _load_vector(self, pattern: str) -> Optional[np.ndarray]:
        p = _latest(self.artifacts_dir, pattern)
        if p is None:
            return None
        try:
            v = np.load(p)
            v = np.asarray(v, dtype=np.float32).reshape(-1)
            return v
        except Exception:
            return None

    def extract(self, doc: Any) -> Dict[str, Any]:
        import time

        t0 = time.perf_counter()
        sys_before = system_snapshot()
        mem_before = process_memory_bytes()

        title_vec = self._load_vector("title_embedding_*.npy")
        hash_vec = self._load_vector("hashtag_embedding_*.npy")

        metrics: Dict[str, Any] = {}
        error: Optional[str] = None

        if title_vec is None:
            error = "title_embedding_not_found"
        elif hash_vec is None:
            error = "hashtag_embedding_not_found"
        else:
            a = _l2n(title_vec)
            b = _l2n(hash_vec)
            sim = float(np.dot(a, b))
            metrics["title_to_hashtag_cosine"] = sim

        sys_after = system_snapshot()
        mem_after = process_memory_bytes()
        total_s = time.perf_counter() - t0

        return {
            "device": "cpu",
            "version": self.VERSION,
            "system": {
                "pre_init": sys_before,
                "post_init": sys_before,
                "post_process": sys_after,
                "peaks": {
                    "ram_peak_mb": int(max(mem_before, mem_after) / 1024 / 1024),
                    "gpu_peak_mb": 0,
                },
            },
            "timings_s": {"total": round(total_s, 3)},
            "result": {"title_to_hashtag": metrics},
            "error": error,
        }


