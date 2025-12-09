from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from src.core.base_extractor import BaseExtractor
from src.core.metrics import system_snapshot, process_memory_bytes


class TitleEmbeddingClusterEntropyExtractor(BaseExtractor):
    VERSION = "1.0.0"

    def __init__(
        self,
        clusters_path: str = "/home/ilya/Рабочий стол/DataProcessor/TextProcessor/models/title_clusters.npy",
        top_k: int = 5,
        temperature: float = 0.1,
        artifacts_dir: str = "/home/ilya/Рабочий стол/DataProcessor/TextProcessor/.artifacts",
    ) -> None:
        self.clusters_path = Path(clusters_path)
        self.top_k = int(top_k)
        self.temperature = float(temperature)
        self.artifacts_dir = Path(artifacts_dir)
        self._centroids: Optional[np.ndarray] = None

    def _load_centroids(self) -> Optional[np.ndarray]:
        if self._centroids is not None:
            return self._centroids
        try:
            centroids = np.load(self.clusters_path)
            centroids = np.asarray(centroids, dtype=np.float32)
            if centroids.ndim != 2:
                return None
            # L2-normalize centroids for cosine
            norms = np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-9
            self._centroids = centroids / norms
            return self._centroids
        except Exception:
            return None

    @staticmethod
    def _latest_title_embedding(artifacts_dir: Path) -> Optional[Path]:
        files = glob.glob(str(artifacts_dir / "title_embedding_*.npy"))
        if not files:
            return None
        files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return Path(files[0])

    @staticmethod
    def _l2n(v: np.ndarray) -> np.ndarray:
        n = float(np.linalg.norm(v))
        if n > 0:
            return v / n
        return v

    @staticmethod
    def _entropy(p: np.ndarray) -> float:
        return float(-np.sum(p * np.log(p + 1e-9)))

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        z = x / max(self.temperature, 1e-6)
        z = z - np.max(z)
        e = np.exp(z)
        d = np.sum(e) + 1e-9
        return (e / d).astype(np.float32)

    def extract(self, doc: Any) -> Dict[str, Any]:
        import time

        t0 = time.perf_counter()
        sys_before = system_snapshot()
        mem_before = process_memory_bytes()

        centroids = self._load_centroids()
        if centroids is None:
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
                "result": {"title_embedding_cluster_entropy": {"error": "clusters_not_loaded"}},
                "error": None,
            }

        title_path = self._latest_title_embedding(self.artifacts_dir)
        if title_path is None:
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
                "result": {"title_embedding_cluster_entropy": {"error": "title_embedding_not_found"}},
                "error": None,
            }

        title = np.load(title_path).astype(np.float32).reshape(-1)
        title = self._l2n(title)

        # cosine similarities: title vs centroids
        sims = (centroids @ title.reshape(-1, 1)).reshape(-1)
        k = min(self.top_k, sims.shape[0])
        topk_idx = np.argsort(sims)[-k:]
        probs = self._softmax(sims[topk_idx])
        ent = self._entropy(probs)

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
                "title_embedding_cluster_entropy": {
                    "entropy": float(ent),
                    "distinct_clusters_topk": int(len(np.unique(topk_idx))),
                    "top_k": int(k),
                    "temperature": float(self.temperature),
                    "clusters_path": str(self.clusters_path),
                }
            },
            "error": None,
        }


