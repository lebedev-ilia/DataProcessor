"""
TitleEmbedder - извлекает L2-нормализованные эмбеддинги для заголовков (titles)
и одновременно предоставляет L2-нормы необработанных векторов (title_embedding_norm).

Особенности:
- Батчинг
- Локальный кеш по SHA256(content + model_name) — сохраняются и векторы, и нормы
- GPU (cuda) поддержка, опционально fp16
- Сохранение/загрузка кеша на диск (atomic save)
- Возвращает numpy массивы:
    - embeddings: shape (N, D) — L2-нормированные векторы
    - norms: shape (N,) — L2-нормы raw (unnormalized) векторов
"""

import os
import hashlib
from pathlib import Path
from typing import List, Optional, Tuple, Any, Dict

import numpy as np
import torch

try:
    from sentence_transformers import SentenceTransformer
except Exception as e:
    raise ImportError(
        "Requires sentence-transformers. Install with: pip install sentence-transformers"
    ) from e


from src.core.base_extractor import BaseExtractor  # noqa: E402
from src.core.text_utils import normalize_whitespace  # noqa: E402
from src.schemas.models import VideoDocument  # noqa: E402
from src.core.metrics import system_snapshot, process_memory_bytes  # noqa: E402
from src.core.model_registry import get_model  # noqa: E402


class TitleEmbedder(BaseExtractor):
    VERSION = "1.0.0"
    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-large",
        cache_dir: str = "/home/ilya/Рабочий стол/DataProcessor/TextProcessor/.cache/embed_cache",
        device: Optional[str] = None,   # "cuda" or "cpu" or None(auto)
        fp16: bool = True,
        batch_size: int = 128,
        artifacts_dir: str = "/home/ilya/Рабочий стол/DataProcessor/TextProcessor/.artifacts",
    ):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.batch_size = batch_size
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # metrics: init snapshots
        init_sys_before = system_snapshot()
        init_mem_before = process_memory_bytes()

        self.fp16 = fp16 and ("cuda" in self.device)

        self._load_model()

        init_sys_after = system_snapshot()
        init_mem_after = process_memory_bytes()
        self._init_metrics: Dict[str, Any] = {
            "pre_init": init_sys_before,
            "post_init": init_sys_after,
            "ram_peak_bytes": max(init_mem_before, init_mem_after),
        }

    def _load_model(self):
        # Use shared registry to reuse model instance
        self.model = get_model(self.model_name, self.device, self.fp16)

    # release_resources removed: models are shared via registry and persist

    @staticmethod
    def _hash_text(text: str, model_name: str) -> str:
        normalized = " ".join(text.strip().split())
        payload = (model_name + "||" + normalized).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _cache_path_vector(self, h: str) -> Path:
        return self.cache_dir / f"{h}.npy"

    def _cache_path_norm(self, h: str) -> Path:
        return self.cache_dir / f"{h}.norm.npy"

    def _load_vector_from_cache(self, h: str) -> Optional[np.ndarray]:
        p = self._cache_path_vector(h)
        if p.exists():
            try:
                arr = np.load(p)
                # ensure float32
                return arr.astype(np.float32)
            except Exception:
                try:
                    p.unlink()
                except Exception:
                    pass
        return None

    def _load_norm_from_cache(self, h: str) -> Optional[float]:
        p = self._cache_path_norm(h)
        if p.exists():
            try:
                arr = np.load(p)
                return float(arr.item())
            except Exception:
                try:
                    p.unlink()
                except Exception:
                    pass
        return None

    def _save_vector_to_cache(self, h: str, vector: np.ndarray):
        p = self._cache_path_vector(h)
        tmp = p.with_suffix(".tmp.npy")
        # ensure dtype float32
        to_save = np.asarray(vector, dtype=np.float32)
        np.save(tmp, to_save)
        tmp.replace(p)

    def _save_norm_to_cache(self, h: str, val: float):
        p = self._cache_path_norm(h)
        tmp = p.with_suffix(".tmp.npy")
        np.save(tmp, np.array(val, dtype=np.float32))
        tmp.replace(p)

    def embed_titles(
        self,
        titles: List[str],
        use_cache: bool = True,
    ) -> np.ndarray:
        """
        Возвращает L2-нормализованные эмбеддинги для списка заголовков.
        (Совместимая версия — без явного возвращения норм).
        """
        embeddings, _ = self.embed_titles_with_norms(titles, use_cache=use_cache, return_norms=False)
        return embeddings

    def embed_titles_with_norms(
        self,
        titles: List[str],
        use_cache: bool = True,
        return_norms: bool = True,
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Возвращает (embeddings, norms)
        embeddings: np.ndarray shape (N, D) — L2-normalized embeddings
        norms: np.ndarray shape (N,) — L2 norms of raw/unnormalized embeddings (if return_norms=True)
        """
        if not isinstance(titles, (list, tuple)):
            raise ValueError("titles must be a list of strings")

        n_total = len(titles)
        embeddings: List[Optional[np.ndarray]] = [None] * n_total
        norms: List[Optional[float]] = [None] * n_total
        to_compute_indices = []
        hashes = []

        # 1) try load both vector and norm from cache (preferred)
        for i, t in enumerate(titles):
            h = self._hash_text(t, self.model_name)
            hashes.append(h)
            if use_cache:
                vec = self._load_vector_from_cache(h)
                nrm = self._load_norm_from_cache(h)
                if vec is not None and nrm is not None:
                    embeddings[i] = vec
                    norms[i] = float(nrm)
                else:
                    # if vector exists but norm missing, re-compute (so treat as missing)
                    to_compute_indices.append(i)
            else:
                to_compute_indices.append(i)

        # 2) compute missing ones in batches: we will request raw vectors (normalize_embeddings=False)
        if len(to_compute_indices) > 0:
            texts_to_compute = [titles[i] for i in to_compute_indices]
            computed_raw_batches = []
            m = len(texts_to_compute)
            for start in range(0, m, self.batch_size):
                end = min(m, start + self.batch_size)
                batch = texts_to_compute[start:end]
                with torch.no_grad():
                    # IMPORTANT: request raw vectors so we can compute raw norms
                    raw = self.model.encode(
                        batch,
                        show_progress_bar=False,
                        convert_to_numpy=True,
                        normalize_embeddings=False,
                    )
                raw = np.asarray(raw, dtype=np.float32)
                computed_raw_batches.append(raw)

            raw_all = np.vstack(computed_raw_batches)
            # compute raw norms
            raw_norms = np.linalg.norm(raw_all, axis=1)
            # avoid zero norms
            raw_norms_safe = raw_norms.copy()
            raw_norms_safe[raw_norms_safe == 0] = 1.0
            # normalized vectors
            normalized_vectors = raw_all / raw_norms_safe.reshape(-1, 1)

            # assign back and cache both normalized vector and raw norm
            j = 0
            for idx in to_compute_indices:
                vec = normalized_vectors[j]
                nrm = float(raw_norms[j])
                embeddings[idx] = vec
                norms[idx] = nrm
                if use_cache:
                    try:
                        self._save_vector_to_cache(hashes[idx], vec)
                        self._save_norm_to_cache(hashes[idx], nrm)
                    except Exception:
                        # swallow caching errors
                        pass
                j += 1

        # 3) all filled — stack and final safety normalization for embeddings
        emb_stack = np.vstack([e for e in embeddings]).astype(np.float32)
        # safety L2 normalize (in case cached vectors were not normalized)
        emb_norms = np.linalg.norm(emb_stack, axis=1, keepdims=True)
        emb_norms[emb_norms == 0] = 1.0
        emb_stack = emb_stack / emb_norms

        if return_norms:
            norms_arr = np.array([float(x) for x in norms], dtype=np.float32)
            return emb_stack, norms_arr
        else:
            return emb_stack, None


    def extract(self, doc: VideoDocument) -> Dict[str, Any]:
        """
        Реализация интерфейса BaseExtractor с метриками и сохранением артефактов.
        Возвращает словарь с полями device, version, timings, system, result, error.
        """
        import time

        started = time.perf_counter()
        sys_before = system_snapshot()
        mem_before = process_memory_bytes()
        error: Optional[str] = None

        # processing block: encode
        t0 = time.perf_counter()
        title = normalize_whitespace(doc.title)
        embeddings, norms = self.embed_titles_with_norms([title], use_cache=True, return_norms=True)
        encode_s = (time.perf_counter() - t0)

        vec = embeddings[0]
        norm_val = float(norms[0]) if norms is not None else float(np.linalg.norm(vec))

        # save artifacts to .npy and return only metadata
        h = self._hash_text(title, self.model_name)
        emb_path = self.artifacts_dir / f"title_embedding_{h}.npy"
        try:
            np.save(emb_path, vec.astype(np.float32))
            # write meta with model info
            meta_path = emb_path.with_suffix(".meta.json")
            import json as _json
            meta = {"model": self.model_name}
            meta_path.write_text(_json.dumps(meta, ensure_ascii=False, indent=2))
        except Exception as e:
            error = f"artifact_save_error: {e}"

        sys_after = system_snapshot()
        mem_after = process_memory_bytes()
        total_s = (time.perf_counter() - started)

        def _gpu_used_mb(snap: Any) -> int:
            try:
                g = (snap or {}).get("gpu") or {}
                arr = g.get("gpus") or []
                return max([int(x.get("memory_used_mb", 0)) for x in arr] or [0])
            except Exception:
                return 0

        gpu_peak_mb = max(
            _gpu_used_mb(self._init_metrics.get("pre_init")),
            _gpu_used_mb(self._init_metrics.get("post_init")),
            _gpu_used_mb(sys_after),
        )

        result: Dict[str, Any] = {
            "device": self.device,
            "version": self.VERSION,
            "model_version": self.model_name,
            "system": {
                "pre_init": self._init_metrics.get("pre_init"),
                "post_init": self._init_metrics.get("post_init"),
                "post_process": sys_after,
                "peaks": {
                    "ram_peak_mb": int(max(self._init_metrics.get("ram_peak_bytes", 0), mem_before, mem_after) / 1024 / 1024),
                    "gpu_peak_mb": int(gpu_peak_mb),
                },
            },
            "timings_s": {
                "encode": round(encode_s, 3),
                "total": round(total_s, 3),
            },
            "result": {
                "title_embedding": {
                    "shape": list(vec.shape),
                    "dtype": str(vec.dtype),
                    "path": str(emb_path.resolve()),
                    "l2_norm": float(np.linalg.norm(vec)),
                },
                "title_embedding_norm": norm_val,
            },
            "error": error,
        }

        return result


# Example usage
# if __name__ == "__main__":
#    titles = [
#        "Что такое искусственный интеллект и как он работает?",
#        "5 простых трюков для ускорения Python кода",
#    ]
#    embedder = TitleEmbedder(
#        model_name="sentence-transformers/all-mpnet-base-v2",
#        cache_dir="./embed_cache",
#        fp16=False,           # True → попробуйте на современной NVidia GPU
#        batch_size=64,
#    )
#    embs, norms = embedder.embed_titles_with_norms(titles, use_cache=True, return_norms=True)
#    print("embeddings shape:", embs.shape)     # (2, dim)
#    print("first embedding (norm):", np.linalg.norm(embs[0]))
#    print("raw norms:", norms)                 # norms of raw vectors (before normalization)
