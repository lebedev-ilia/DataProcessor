from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from src.core.base_extractor import BaseExtractor
from src.core.metrics import system_snapshot, process_memory_bytes

try:
    import faiss  # type: ignore
except Exception:
    faiss = None  # type: ignore


def _l2n(x: np.ndarray, axis: int = 1, eps: float = 1e-10) -> np.ndarray:
    n = np.linalg.norm(x, axis=axis, keepdims=True)
    n = np.maximum(n, eps)
    return x / n


class TopKSimilarCorpusTitlesExtractor(BaseExtractor):
    VERSION = "1.0.0"

    def __init__(
        self,
        corpus_embeddings_path: str = "",
        corpus_ids_path: str = "",
        k: int = 5,
    ) -> None:
        self.k = k
        self._index: Any = None
        self._ids: List[Any] = []
        self._dim: Optional[int] = None

        self._load_corpus(corpus_embeddings_path, corpus_ids_path)

    def _load_corpus(self, emb_path: str, ids_path: str) -> None:
        try:
            emb = np.load(emb_path)
            emb = np.asarray(emb, dtype=np.float32)
            if emb.ndim != 2:
                raise ValueError("corpus_embeddings must be 2D")
            emb = _l2n(emb, axis=1)
            self._dim = emb.shape[1]

            ids: List[Any] = []
            with open(ids_path, "r", encoding="utf-8") as f:
                ids = json.load(f)
            if not isinstance(ids, list) or len(ids) != emb.shape[0]:
                raise ValueError("corpus_ids must be a list with same length as embeddings")
            self._ids = ids

            if faiss is not None:
                index = faiss.IndexHNSWFlat(self._dim, 32)
                index.hnsw.efConstruction = 200
                index.add(emb.astype(np.float32))
                self._index = index
            else:
                # fallback: store in memory for np cosine search
                self._index = emb
        except Exception:
            self._index = None
            self._ids = []
            self._dim = None

    def _search_np(self, query: np.ndarray, corpus: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        sims = (_l2n(query, axis=1) @ _l2n(corpus, axis=1).T)
        k = min(k, sims.shape[1])
        idx = np.argsort(-sims, axis=1)[:, :k]
        scr = np.take_along_axis(sims, idx, axis=1)
        return idx, scr

    def extract(self, doc: Any) -> Dict[str, Any]:
        import time

        t0 = time.perf_counter()
        sys_before = system_snapshot()
        mem_before = process_memory_bytes()

        result: Dict[str, Any] = {}
        error: Optional[str] = None

        # Preconditions
        if self._index is None or not self._ids:
            error = "corpus_not_loaded"
        else:
            # load latest title embedding
            try:
                # resolve latest artifact
                from glob import glob
                import os
                artifacts_dir = Path("/home/ilya/Рабочий стол/DataProcessor/TextProcessor/.artifacts")
                files = glob(str(artifacts_dir / "title_embedding_*.npy"))
                if not files:
                    raise FileNotFoundError
                files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                title = np.load(files[0]).astype(np.float32).reshape(1, -1)
                title = _l2n(title, axis=1)

                if faiss is not None and isinstance(self._index, faiss.Index):  # type: ignore[arg-type]
                    # cosine via inner product on normalized vectors
                    scores, indices = self._index.search(title.astype(np.float32), min(self.k, len(self._ids)))
                    top_ids = [self._ids[i] for i in indices[0].tolist()]
                    top_scores = scores[0].astype(float).tolist()
                else:
                    idx, scr = self._search_np(title, self._index, self.k)  # type: ignore[arg-type]
                    top_ids = [self._ids[i] for i in idx[0].tolist()]
                    top_scores = scr[0].astype(float).tolist()

                result = {"topk_similar_ids": top_ids, "topk_similar_scores": top_scores}
            except Exception:
                error = "title_embedding_not_found"

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
            "result": {"topk_similar_corpus_titles": result},
            "error": error,
        }


