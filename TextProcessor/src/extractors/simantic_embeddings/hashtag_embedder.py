from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from sentence_transformers import SentenceTransformer  # noqa: F401

from src.core.base_extractor import BaseExtractor
from src.core.metrics import system_snapshot, process_memory_bytes
from src.core.model_registry import get_model
from src.schemas.models import VideoDocument


class HashtagEmbedder(BaseExtractor):
    VERSION = "1.0.0"

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-large",
        artifacts_dir: str = "/home/ilya/Рабочий стол/DataProcessor/TextProcessor/.artifacts",
        device: Optional[str] = None,
        fp16: bool = True,
        batch_size: int = 128,
    ) -> None:
        self.model_name = model_name
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.fp16 = fp16 and ("cuda" in self.device)
        self.batch_size = batch_size
        self._model = get_model(self.model_name, self.device, self.fp16)

    @staticmethod
    def _hash_tags(tags: List[str], model_name: str) -> str:
        payload = (model_name + "||" + " ".join(tags)).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)
        out: List[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            with torch.no_grad():
                raw = self._model.encode(
                    batch,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=False,
                )
            raw = np.asarray(raw, dtype=np.float32)
            norms = np.linalg.norm(raw, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            out.append(raw / norms)
        return np.vstack(out)

    def extract(self, doc: VideoDocument) -> Dict[str, Any]:
        import time

        t0 = time.perf_counter()
        sys_before = system_snapshot()
        mem_before = process_memory_bytes()

        tags = getattr(doc, "hashtags", None)
        if not tags or not isinstance(tags, list):
            # can still return empty output
            tags = []

        # embed per-tag then compute weighted average by frequency
        embs = self._encode_texts(tags)
        if embs.size == 0:
            # empty
            sys_after = system_snapshot()
            mem_after = process_memory_bytes()
            total_s = time.perf_counter() - t0
            return {
                "device": self.device,
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
                "result": {"hashtag_embedding": {"path": "", "count": 0, "l2_norm": 0.0}},
                "error": None,
            }

        # average with equal weights (tags are unique from TagsExtractor)
        mean_vec = embs.mean(axis=0)
        nrm = np.linalg.norm(mean_vec)
        if nrm > 0:
            mean_vec = mean_vec / nrm

        h = self._hash_tags(tags, self.model_name)
        out_path = self.artifacts_dir / f"hashtag_embedding_{h}.npy"
        tmp = out_path.with_suffix(".tmp.npy")
        np.save(tmp, mean_vec.astype(np.float32))
        tmp.replace(out_path)
        # write meta with model info
        meta_path = out_path.with_suffix(".meta.json")
        import json as _json
        meta = {"model": self.model_name}
        meta_path.write_text(_json.dumps(meta, ensure_ascii=False, indent=2))

        sys_after = system_snapshot()
        mem_after = process_memory_bytes()
        total_s = time.perf_counter() - t0

        return {
            "device": self.device,
            "version": self.VERSION,
            "model_version": self.model_name,
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
                "hashtag_embedding": {
                    "path": str(out_path.resolve()),
                    "count": int(embs.shape[0]),
                    "l2_norm": float(np.linalg.norm(mean_vec)),
                }
            },
            "error": None,
        }


