from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np

from src.core.base_extractor import BaseExtractor
from src.core.metrics import system_snapshot, process_memory_bytes
from src.core.path_utils import default_artifacts_dir, textprocessor_root

try:
    import faiss  # type: ignore
except Exception:
    faiss = None  # type: ignore


def _latest(artifacts_dir: Path, pattern: str) -> Optional[Path]:
    paths = glob.glob(str(artifacts_dir / pattern))
    if not paths:
        return None
    paths.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return Path(paths[0])


def _l2_normalize(x: np.ndarray, axis: int = 1, eps: float = 1e-9) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    n = np.maximum(n, eps)
    return x / n


class SemanticClusterExtractor(BaseExtractor):
    VERSION = "1.0.0"

    def __init__(
        self,
        artifacts_dir: str | None = None,
        cluster_model_path: str | None = None,
        pca_model_path: str | None = None,
        use_hdbscan: bool = False,
        source: str = "title",  # which embedding to use by default: title|description|hashtag
    ) -> None:
        base_models = textprocessor_root() / "models"
        self.artifacts_dir = Path(artifacts_dir).expanduser().resolve() if artifacts_dir else default_artifacts_dir()
        self.cluster_model_path = Path(cluster_model_path).expanduser().resolve() if cluster_model_path else (base_models / "centroids.npy")
        self.pca_model_path = Path(pca_model_path).expanduser().resolve() if pca_model_path else (base_models / "pca.npy")
        self.use_hdbscan = use_hdbscan
        self.source = source

        self._centroids: Optional[np.ndarray] = None
        self._pca: Optional[np.ndarray] = None
        self._faiss_index = None

        self._load_models()

    def _load_models(self) -> None:
        # Load PCA: shape (orig_dim, reduced_dim)
        try:
            self._pca = np.load(self.pca_model_path)
        except Exception:
            self._pca = None
        # Load centroids: shape (n_clusters, reduced_dim)
        try:
            centroids = np.load(self.cluster_model_path)
            centroids = np.asarray(centroids, dtype=np.float32)
            centroids = _l2_normalize(centroids, axis=1)
            self._centroids = centroids
        except Exception:
            self._centroids = None
        # FAISS index (if available)
        if self._centroids is not None and faiss is not None:
            dim = self._centroids.shape[1]
            index = faiss.IndexFlatIP(dim)
            index.add(self._centroids.astype("float32"))
            self._faiss_index = index

    def _pick_embedding(self) -> Tuple[Optional[np.ndarray], str]:
        # Choose embedding file based on preferred source; fallback to others
        order = []
        if self.source == "title":
            order = ["title_embedding_*.npy", "description_embedding_*.npy", "hashtag_embedding_*.npy"]
        elif self.source == "description":
            order = ["description_embedding_*.npy", "title_embedding_*.npy", "hashtag_embedding_*.npy"]
        else:
            order = ["hashtag_embedding_*.npy", "title_embedding_*.npy", "description_embedding_*.npy"]

        for pat in order:
            p = _latest(self.artifacts_dir, pat)
            if p is None:
                continue
            try:
                v = np.load(p)
                v = np.asarray(v, dtype=np.float32).reshape(-1)
                return v, pat.split("_")[0]  # return detected source name
            except Exception:
                continue
        return None, ""

    def extract(self, doc: Any) -> Dict[str, Any]:
        import time

        t0 = time.perf_counter()
        sys_before = system_snapshot()
        mem_before = process_memory_bytes()

        out: Dict[str, Any] = {}

        # Preconditions
        if self._pca is None or self._centroids is None:
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
                "result": {"semantic_cluster": {"error": "models_not_loaded"}},
                "error": None,
            }

        vec, detected = self._pick_embedding()
        if vec is None:
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
                "result": {"semantic_cluster": {"error": "embedding_not_found"}},
                "error": None,
            }

        # Project via PCA and normalize
        reduced = vec @ self._pca  # (reduced_dim,)
        reduced = _l2_normalize(reduced.reshape(1, -1), axis=1)

        # Nearest centroid by cosine similarity
        if self._faiss_index is not None:
            scores, idx = self._faiss_index.search(reduced.astype("float32"), 1)
            sim = float(scores[0, 0])
            cid = int(idx[0, 0])
        else:
            sims = (reduced @ self._centroids.T).reshape(-1)  # type: ignore[arg-type]
            cid = int(np.argmax(sims))
            sim = float(sims[cid])

        dist = 1.0 - sim
        if self.use_hdbscan and cid == -1:
            # noise handling
            dist = None

        out = {
            "source": detected or self.source,
            "semantic_cluster_id": cid,
            "semantic_cluster_similarity": round(sim, 4),
            "semantic_cluster_distance": None if dist is None else round(dist, 4),
        }

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
            "result": {"semantic_cluster": out},
            "error": None,
        }


