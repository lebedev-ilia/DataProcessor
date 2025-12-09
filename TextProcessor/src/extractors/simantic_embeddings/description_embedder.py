"""
DescriptionEmbedder — извлекает L2-нормализованные эмбеддинги для описаний (description)
и одновременно вычисляет L2-нормы необработанных векторов (description_embedding_norm).

Особенности:
- Поддержка длинных описаний (chunk-and-aggregate)
- Attention-weighted pooling по длине чанка
- Батчинг
- GPU (cuda) поддержка, fp16 опционально
- Кеш по SHA256(content + model_name)
- Сохранение артефактов и метрик
"""

import os
import hashlib
from pathlib import Path
from typing import List, Optional, Tuple, Any, Dict

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from src.core.model_registry import get_model

from src.core.base_extractor import BaseExtractor  # noqa
from src.core.text_utils import normalize_whitespace  # noqa
from src.schemas.models import VideoDocument  # noqa
from src.core.metrics import system_snapshot, process_memory_bytes  # noqa


class DescriptionEmbedder(BaseExtractor):
    VERSION = "1.0.0"

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-large",
        cache_dir: str = "/home/ilya/Рабочий стол/DataProcessor/TextProcessor/.cache/embed_cache",
        device: Optional[str] = None,
        fp16: bool = True,
        batch_size: int = 32,
        artifacts_dir: str = "/home/ilya/Рабочий стол/DataProcessor/TextProcessor/.artifacts",
        max_chunk_tokens: int = 512,
    ):
        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self.batch_size = batch_size
        self.max_chunk_tokens = max_chunk_tokens

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        self.fp16 = fp16 and ("cuda" in self.device)

        # --- инициализация модели ---
        init_sys_before = system_snapshot()
        init_mem_before = process_memory_bytes()

        self._load_model()

        init_sys_after = system_snapshot()
        init_mem_after = process_memory_bytes()
        self._init_metrics: Dict[str, Any] = {
            "pre_init": init_sys_before,
            "post_init": init_sys_after,
            "ram_peak_bytes": max(init_mem_before, init_mem_after),
        }

    def _load_model(self):
        self.model = get_model(self.model_name, self.device, self.fp16)

    # release_resources removed: models are shared via registry and persist

    # -------------------- КЕШ --------------------
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
                return arr.astype(np.float32)
            except Exception:
                p.unlink(missing_ok=True)
        return None

    def _load_norm_from_cache(self, h: str) -> Optional[float]:
        p = self._cache_path_norm(h)
        if p.exists():
            try:
                arr = np.load(p)
                return float(arr.item())
            except Exception:
                p.unlink(missing_ok=True)
        return None

    def _save_vector_to_cache(self, h: str, vec: np.ndarray):
        p = self._cache_path_vector(h)
        tmp = p.with_suffix(".tmp.npy")
        np.save(tmp, np.asarray(vec, dtype=np.float32))
        tmp.replace(p)

    def _save_norm_to_cache(self, h: str, val: float):
        p = self._cache_path_norm(h)
        tmp = p.with_suffix(".tmp.npy")
        np.save(tmp, np.array(val, dtype=np.float32))
        tmp.replace(p)

    # -------------------- ЛОГИКА --------------------
    def _chunk_text(self, text: str) -> List[str]:
        words = text.split()
        if len(words) <= self.max_chunk_tokens:
            return [text]
        chunks = []
        for i in range(0, len(words), self.max_chunk_tokens):
            chunks.append(" ".join(words[i : i + self.max_chunk_tokens]))
        return chunks

    def _weighted_pooling(self, embeddings: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
        weighted_sum = (embeddings * weights.unsqueeze(-1)).sum(dim=0)
        denom = weights.sum() + 1e-9
        return weighted_sum / denom

    # -------------------- ОСНОВНОЙ МЕТОД --------------------
    def extract(self, doc: VideoDocument) -> Dict[str, Any]:
        import time

        start = time.perf_counter()
        sys_before = system_snapshot()
        mem_before = process_memory_bytes()
        error: Optional[str] = None

        text = normalize_whitespace(doc.description or "")
        if not text:
            # Create zero vector artifact to keep schema consistent
            h = self._hash_text("", self.model_name)
            emb_path = self.artifacts_dir / f"description_embedding_{h}.npy"
            zero_vec = np.zeros(384, dtype=np.float32)
            try:
                np.save(emb_path, zero_vec)
            except Exception as e:
                error = f"artifact_save_error: {e}"
            sys_after = system_snapshot()
            mem_after = process_memory_bytes()
            total_s = (time.perf_counter() - start)

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

            return {
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
                "timings_s": {"total": round(total_s, 3)},
                "result": {
                    "description_embedding": {
                        "shape": list(zero_vec.shape),
                        "dtype": str(zero_vec.dtype),
                        "path": str(emb_path.resolve()),
                        "l2_norm": float(np.linalg.norm(zero_vec)),
                    },
                    "description_embedding_norm": 0.0,
                },
                "error": "empty description",
            }

        h = self._hash_text(text, self.model_name)
        vec_cached = self._load_vector_from_cache(h)
        norm_cached = self._load_norm_from_cache(h)
        if vec_cached is not None and norm_cached is not None:
            # ensure artifact exists and return structured result
            emb_path = self.artifacts_dir / f"description_embedding_{h}.npy"
            try:
                if not emb_path.exists():
                    np.save(emb_path, np.asarray(vec_cached, dtype=np.float32))
            except Exception as e:
                error = f"artifact_save_error: {e}"
            sys_after = system_snapshot()
            mem_after = process_memory_bytes()
            total_s = (time.perf_counter() - start)

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

            return {
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
                "timings_s": {"total": round(total_s, 3)},
                "result": {
                    "description_embedding": {
                        "shape": list(np.asarray(vec_cached).shape),
                        "dtype": str(np.asarray(vec_cached).dtype),
                        "path": str(emb_path.resolve()),
                        "l2_norm": float(np.linalg.norm(np.asarray(vec_cached))),
                    },
                    "description_embedding_norm": float(norm_cached),
                },
                "error": None,
            }

        # --- chunk and embed ---
        chunks = self._chunk_text(text)
        with torch.no_grad():
            embeds = self.model.encode(
                chunks,
                batch_size=self.batch_size,
                convert_to_tensor=True,
                normalize_embeddings=False,
                show_progress_bar=False,
            )

        # --- attention-weighted pooling (по длине чанка) ---
        weights = torch.tensor([len(c.split()) for c in chunks], dtype=torch.float32, device=embeds.device)
        weights = weights / (weights.sum() + 1e-9)
        pooled = self._weighted_pooling(embeds, weights)
        pooled_norm = torch.nn.functional.normalize(pooled, p=2, dim=0)
        pooled_np = pooled_norm.cpu().numpy().astype(np.float32)
        norm_val = float(torch.linalg.norm(pooled))

        # --- кеш и артефакты ---
        try:
            self._save_vector_to_cache(h, pooled_np)
            self._save_norm_to_cache(h, norm_val)
        except Exception as e:
            error = f"cache_save_error: {e}"

        emb_path = self.artifacts_dir / f"description_embedding_{h}.npy"
        try:
            np.save(emb_path, pooled_np)
            # write meta with model info
            meta_path = emb_path.with_suffix(".meta.json")
            import json as _json
            meta = {"model": self.model_name}
            meta_path.write_text(_json.dumps(meta, ensure_ascii=False, indent=2))
        except Exception as e:
            error = f"artifact_save_error: {e}"

        sys_after = system_snapshot()
        mem_after = process_memory_bytes()
        total_s = (time.perf_counter() - start)

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
            "timings_s": {"total": round(total_s, 3)},
            "result": {
                "description_embedding": {
                    "shape": list(pooled_np.shape),
                    "dtype": str(pooled_np.dtype),
                    "path": str(emb_path.resolve()),
                    "l2_norm": float(np.linalg.norm(pooled_np)),
                },
                "description_embedding_norm": norm_val,
            },
            "error": error,
        }
        return result
