from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.core.base_extractor import BaseExtractor
from src.core.metrics import system_snapshot, process_memory_bytes


def _latest(path: Path, pattern: str) -> Optional[Path]:
    files = glob.glob(str(path / pattern))
    if not files:
        return None
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return Path(files[0])


class EmbeddingStatsExtractor(BaseExtractor):
    """
    18. embedding_variance_across_chunks: L2-норма дисперсии + top-k компонентных дисперсий
    19. embedding_topic_mix_entropy: энтропия усреднённого топик-распределения (если доступно)
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        artifacts_dir: str = "/home/ilya/Рабочий стол/DataProcessor/TextProcessor/.artifacts",
        cache_dir: str = "/home/ilya/Рабочий стол/DataProcessor/TextProcessor/.cache/transcript_embed",
        topk: int = 5,
    ) -> None:
        self.artifacts_dir = Path(artifacts_dir)
        self.cache_dir = Path(cache_dir)
        self.topk = topk

    def _load_chunks(self) -> Tuple[Optional[np.ndarray], Optional[Path]]:
        # prefer whisper, then youtube_auto
        for src in ["whisper", "youtube_auto"]:
            p = _latest(self.artifacts_dir, f"transcript_{src}_embedding_*.npy")
            if p is not None:
                try:
                    m = np.load(p)
                    m = np.asarray(m, dtype=np.float32)
                    if m.ndim == 1:
                        m = m.reshape(1, -1)
                    return m, p
                except Exception:
                    continue
        return None, None

    def _load_topic_probs(self, chunks_artifact: Optional[Path]) -> Optional[np.ndarray]:
        if chunks_artifact is None:
            return None
        # derive hash from file name pattern transcript_{src}_embedding_{hash}.npy
        try:
            name = chunks_artifact.name
            h = name.split("_")[-1].split(".")[0]
            meta_path = self.cache_dir / f"{h}.json"
            if not meta_path.exists():
                return None
            data = json.loads(meta_path.read_text())
            probs = data.get("topic_probs")
            if probs is None:
                return None
            arr = np.asarray(probs, dtype=np.float32)
            if arr.ndim != 2:
                return None
            return arr
        except Exception:
            return None

    def _variance_across_chunks(self, chunks: np.ndarray) -> Dict[str, Any]:
        if chunks.size == 0:
            return {"l2_variance": 0.0, "topk_variances": []}
        var_vec = np.var(chunks, axis=0)
        l2_variance = float(np.linalg.norm(var_vec))
        topk = np.sort(var_vec)[-self.topk :]
        return {"l2_variance": l2_variance, "topk_variances": [float(x) for x in topk.tolist()]}

    def _topic_mix_entropy(self, topic_probs: Optional[np.ndarray]) -> Dict[str, Any]:
        if topic_probs is None or topic_probs.size == 0:
            return {"topic_entropy": None, "error": "topic_probs_not_found"}
        avg = np.mean(topic_probs, axis=0)
        eps = 1e-12
        entropy = -float(np.sum(avg * np.log(avg + eps)))
        return {"topic_entropy": entropy}

    def extract(self, doc: Any) -> Dict[str, Any]:
        import time

        t0 = time.perf_counter()
        sys_before = system_snapshot()
        mem_before = process_memory_bytes()

        chunks, chunks_path = self._load_chunks()
        variance_block: Dict[str, Any]
        topic_block: Dict[str, Any]

        if chunks is not None:
            variance_block = self._variance_across_chunks(chunks)
            topic_probs = self._load_topic_probs(chunks_path)
            topic_block = self._topic_mix_entropy(topic_probs)
        else:
            variance_block = {"l2_variance": None, "topk_variances": []}
            topic_block = {"topic_entropy": None, "error": "chunks_not_found"}

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
            "result": {
                "embedding_variance_across_chunks": variance_block,
                "embedding_topic_mix_entropy": topic_block,
            },
            "error": None,
        }


