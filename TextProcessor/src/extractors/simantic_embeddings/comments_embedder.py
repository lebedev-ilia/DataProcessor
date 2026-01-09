"""
CommentsEmbedder — извлекает L2-нормализованные эмбеддинги для комментариев.

- Использует общий ModelRegistry (SentenceTransformer переиспользуется)
- Батчинг encode, inference_mode
- Сохраняет артефакт одним массивом (N, D) в .artifacts
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.core.base_extractor import BaseExtractor
from src.core.metrics import system_snapshot, process_memory_bytes
from src.core.model_registry import get_model
from src.core.path_utils import default_artifacts_dir
from src.core.text_utils import normalize_whitespace
from src.schemas.models import VideoDocument


class CommentsEmbedder(BaseExtractor):
    VERSION = "1.0.0"

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        artifacts_dir: Optional[str] = None,
        device: Optional[str] = "cpu",
        fp16: bool = True,
        batch_size: int = 64,
    ) -> None:
        self.model_name = model_name
        self.artifacts_dir = Path(artifacts_dir).expanduser().resolve() if artifacts_dir else default_artifacts_dir()
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        self.device = str(device or "cpu")
        self.fp16 = fp16 and ("cuda" in self.device)
        self.batch_size = batch_size

        # metrics: init snapshots
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

    def _load_model(self) -> None:
        self.model = get_model(self.model_name, self.device, self.fp16)

    @staticmethod
    def _hash_list(texts: List[str], model_name: str) -> str:
        payload = (model_name + "||" + "\n".join(texts)).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        out_batches: List[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            with torch.no_grad():
                raw = self.model.encode(
                    batch,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=False,
                )
            raw = np.asarray(raw, dtype=np.float32)
            # l2 normalize
            norms = np.linalg.norm(raw, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            normed = raw / norms
            out_batches.append(normed)
        return np.vstack(out_batches)

    def extract(self, doc: VideoDocument) -> Dict[str, Any]:
        import time

        t0 = time.perf_counter()
        sys_before = system_snapshot()
        mem_before = process_memory_bytes()
        error: Optional[str] = None

        # gather comments
        comments_texts = [normalize_whitespace(c.text) for c in (doc.comments or []) if normalize_whitespace(c.text)]

        if not comments_texts:
            sys_after = system_snapshot()
            mem_after = process_memory_bytes()
            total_s = time.perf_counter() - t0
            return {
                "device": self.device,
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
                "result": {
                    "comments_embeddings": {
                        "shape": [0, 0],
                        "dtype": "float32",
                        "path": "",
                        "count": 0,
                    }
                },
                "error": None,
            }

        # encode
        t_enc = time.perf_counter()
        embs = self._encode_texts(comments_texts)
        encode_s = time.perf_counter() - t_enc

        # save artifact
        h = self._hash_list(comments_texts, self.model_name)
        emb_path = self.artifacts_dir / f"comments_embeddings_{h}.npy"
        try:
            tmp = emb_path.with_suffix(".tmp.npy")
            np.save(tmp, embs.astype(np.float32))
            tmp.replace(emb_path)
            # write meta with model info
            meta_path = emb_path.with_suffix(".meta.json")
            import json as _json
            meta = {"model": self.model_name}
            meta_path.write_text(_json.dumps(meta, ensure_ascii=False, indent=2))
        except Exception as e:
            error = f"artifact_save_error: {e}"

        sys_after = system_snapshot()
        mem_after = process_memory_bytes()
        total_s = time.perf_counter() - t0

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
                    "gpu_peak_mb": 0,
                },
            },
            "timings_s": {"encode": round(encode_s, 3), "total": round(total_s, 3)},
            "result": {
                "comments_embeddings": {
                    "shape": list(embs.shape),
                    "dtype": str(embs.dtype),
                    "path": str(emb_path.resolve()),
                    "count": int(embs.shape[0]),
                }
            },
            "error": error,
        }

        return result


