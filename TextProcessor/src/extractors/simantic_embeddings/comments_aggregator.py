"""
CommentsAggregationExtractor — агрегирует эмбеддинги комментариев (уже посчитанные) по стратегиям:
- weighted mean (веса: likes × authority × recency, если заданы)
- component-wise median

Совместим со структурой проекта:
- читает список комментариев из VideoDocument
- ищет артефакт эмбеддингов, сохранённый CommentsEmbedder: .artifacts/comments_embeddings_{hash}.npy
- сохраняет агрегаты в .artifacts и возвращает только метаданные и пути
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from src.core.base_extractor import BaseExtractor
from src.core.metrics import system_snapshot, process_memory_bytes
from src.core.text_utils import normalize_whitespace
from src.schemas.models import VideoDocument


class CommentsAggregationExtractor(BaseExtractor):
    VERSION = "1.0.0"
    DEFAULT_EMBED_DIM = 384

    def __init__(
        self,
        artifacts_dir: str | None = None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ) -> None:
        from src.core.path_utils import default_artifacts_dir  # local import to avoid cycles

        self.artifacts_dir = Path(artifacts_dir).expanduser().resolve() if artifacts_dir else default_artifacts_dir()
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name

        # metrics
        self._init_metrics: Dict[str, Any] = {
            "pre_init": system_snapshot(),
            "post_init": system_snapshot(),
            "ram_peak_bytes": process_memory_bytes(),
        }

    @staticmethod
    def _hash_list(texts: List[str], model_name: str) -> str:
        payload = (model_name + "||" + "\n".join(texts)).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _load_comment_embeddings(self, comments: List[str]) -> Optional[np.ndarray]:
        h = self._hash_list(comments, self.model_name)
        emb_path = self.artifacts_dir / f"comments_embeddings_{h}.npy"
        if not emb_path.exists():
            return None
        try:
            arr = np.load(emb_path)
            return np.asarray(arr, dtype=np.float32)
        except Exception:
            return None

    @staticmethod
    def _aggregate_weighted_mean(embs: np.ndarray, likes: Optional[List[float]], authority: Optional[List[float]], recency: Optional[List[float]], *, default_dim: int) -> Dict[str, Any]:
        if embs.size == 0:
            return {"embedding": np.zeros((default_dim,), dtype=np.float32), "count": 0, "std": 0.0}
        n, _ = embs.shape
        w = np.ones(n, dtype=np.float32)
        if likes is not None and len(likes) == n:
            w *= np.clip(np.asarray(likes, dtype=np.float32), 0.1, None)
        if authority is not None and len(authority) == n:
            w *= np.clip(np.asarray(authority, dtype=np.float32), 0.1, None)
        if recency is not None and len(recency) == n:
            w *= np.clip(np.asarray(recency, dtype=np.float32), 0.1, None)
        w_sum = float(np.sum(w))
        if w_sum <= 0:
            w = np.ones(n, dtype=np.float32)
            w_sum = float(n)
        w /= w_sum
        vec = np.average(embs, axis=0, weights=w)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return {"embedding": vec.astype(np.float32), "count": int(n), "std": float(np.std(embs))}

    @staticmethod
    def _aggregate_median(embs: np.ndarray, *, default_dim: int) -> Dict[str, Any]:
        if embs.size == 0:
            return {"embedding": np.zeros((default_dim,), dtype=np.float32), "count": 0, "std": 0.0}
        vec = np.median(embs, axis=0)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return {"embedding": vec.astype(np.float32), "count": int(embs.shape[0]), "std": float(np.std(embs))}

    def extract(self, doc: VideoDocument) -> Dict[str, Any]:
        import time

        t0 = time.perf_counter()
        sys_before = system_snapshot()
        mem_before = process_memory_bytes()

        comments_texts = [normalize_whitespace(c.text) for c in (doc.comments or []) if normalize_whitespace(c.text)]
        embs = self._load_comment_embeddings(comments_texts) if comments_texts else None

        if embs is None or embs.size == 0:
            sys_after = system_snapshot()
            mem_after = process_memory_bytes()
            total_s = time.perf_counter() - t0
            return {
                "device": "cpu",
                "version": self.VERSION,
                "system": {
                    "pre_init": self._init_metrics.get("pre_init"),
                    "post_init": self._init_metrics.get("post_init"),
                    "post_process": sys_after,
                    "peaks": {
                        "ram_peak_mb": int(max(self._init_metrics.get("ram_peak_bytes", 0), mem_before, mem_after) / 1024 / 1024),
                        "gpu_peak_mb": 0,
                    },
                },
                "timings_s": {"total": round(total_s, 3)},
                "result": {"comments_aggregates": {}},
                "error": None,
            }

        # Опциональные веса — если где-то в документе присутствуют
        likes = getattr(doc, "comments_likes", None)
        authority = getattr(doc, "comments_authority", None)
        recency = getattr(doc, "comments_recency", None)

        dim = int(embs.shape[1]) if isinstance(embs, np.ndarray) and embs.ndim == 2 and embs.shape[1] > 0 else self.DEFAULT_EMBED_DIM

        # weighted mean
        t_agg0 = time.perf_counter()
        mean_res = self._aggregate_weighted_mean(embs, likes, authority, recency, default_dim=dim)
        mean_s = time.perf_counter() - t_agg0

        # median
        t_agg1 = time.perf_counter()
        med_res = self._aggregate_median(embs, default_dim=dim)
        median_s = time.perf_counter() - t_agg1

        # save artifacts
        h = self._hash_list(comments_texts, self.model_name)
        mean_path = self.artifacts_dir / f"comments_agg_mean_{h}.npy"
        med_path = self.artifacts_dir / f"comments_agg_median_{h}.npy"
        tmp_mean = mean_path.with_suffix(".tmp.npy")
        tmp_median = med_path.with_suffix(".tmp.npy")
        np.save(tmp_mean, mean_res["embedding"]) ; tmp_mean.replace(mean_path)
        np.save(tmp_median, med_res["embedding"]) ; tmp_median.replace(med_path)

        sys_after = system_snapshot()
        mem_after = process_memory_bytes()
        total_s = time.perf_counter() - t0

        return {
            "device": "cpu",
            "version": self.VERSION,
            "system": {
                "pre_init": self._init_metrics.get("pre_init"),
                "post_init": self._init_metrics.get("post_init"),
                "post_process": sys_after,
                "peaks": {
                    "ram_peak_mb": int(max(self._init_metrics.get("ram_peak_bytes", 0), mem_before, mem_after) / 1024 / 1024),
                    "gpu_peak_mb": 0,
                },
            },
            "timings_s": {"mean": round(mean_s, 3), "median": round(median_s, 3), "total": round(total_s, 3)},
            "result": {
                "comments_aggregates": {
                    "weighted_mean": {"path": str(mean_path.resolve()), "count": int(mean_res["count"]), "std": float(mean_res["std"])},
                    "median": {"path": str(med_path.resolve()), "count": int(med_res["count"]), "std": float(med_res["std"])},
                }
            },
            "error": None,
        }


