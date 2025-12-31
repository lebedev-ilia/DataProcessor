from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from src.core.base_extractor import BaseExtractor
from src.core.metrics import system_snapshot, process_memory_bytes


def _latest(glob_pattern: str) -> Optional[Path]:
    files = glob.glob(glob_pattern)
    if not files:
        return None
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return Path(files[0])


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)
    # assume already normalized; still guard
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    sim = (a_n @ b_n.T)[0, 0]
    return float(sim)


class CosineMetricsExtractor(BaseExtractor):
    VERSION = "1.0.0"

    def __init__(self, artifacts_dir: str = "/home/ilya/Рабочий стол/DataProcessor/TextProcessor/.artifacts") -> None:
        self.artifacts_dir = Path(artifacts_dir)

    def _load_vector(self, pattern: str) -> Optional[np.ndarray]:
        p = _latest(str(self.artifacts_dir / pattern))
        if p is None:
            return None
        try:
            v = np.load(p)
            v = np.asarray(v, dtype=np.float32)
            return v
        except Exception:
            return None

    def _load_matrix(self, pattern: str) -> Optional[np.ndarray]:
        p = _latest(str(self.artifacts_dir / pattern))
        if p is None:
            return None
        try:
            m = np.load(p)
            m = np.asarray(m, dtype=np.float32)
            if m.ndim == 1:
                m = m.reshape(1, -1)
            return m
        except Exception:
            return None

    def extract(self, doc: Any) -> Dict[str, Any]:
        import time

        t0 = time.perf_counter()
        sys_before = system_snapshot()
        mem_before = process_memory_bytes()

        result: Dict[str, Any] = {}

        # Load title/description vectors
        title = self._load_vector("title_embedding_*.npy")
        desc = self._load_vector("description_embedding_*.npy")

        # Transcript aggregated mean (prefer combined → whisper → youtube)
        # select first available transcript aggregate
        transcript = None
        for pat in [
            "transcript_combined_agg_mean_*.npy",
            "transcript_whisper_agg_mean_*.npy",
            "transcript_youtube_auto_agg_mean_*.npy",
        ]:
            transcript = self._load_vector(pat)
            if transcript is not None:
                break

        # Comments matrix
        comments = self._load_matrix("comments_embeddings_*.npy")

        if title is not None and desc is not None:
            result["title_description_cosine"] = _cosine(title, desc)

        if title is not None and transcript is not None:
            result["title_transcript_cosine"] = _cosine(title, transcript)

        if desc is not None and transcript is not None:
            result["description_transcript_cosine"] = _cosine(desc, transcript)

        if transcript is not None and comments is not None and comments.size > 0:
            t = transcript.reshape(1, -1)
            # cos for each comment vs transcript
            t_norm = t / (np.linalg.norm(t, axis=1, keepdims=True) + 1e-9)
            c_norm = comments / (np.linalg.norm(comments, axis=1, keepdims=True) + 1e-9)
            sims = (c_norm @ t_norm.T).reshape(-1)
            result["transcript_comments_cosine_mean"] = float(np.mean(sims))
            result["transcript_comments_cosine_median"] = float(np.median(sims))

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
            "result": {"cosine_metrics": result},
            "error": None,
        }


