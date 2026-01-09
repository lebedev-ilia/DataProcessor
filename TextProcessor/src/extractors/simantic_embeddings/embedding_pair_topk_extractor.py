from __future__ import annotations

import glob
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from src.core.base_extractor import BaseExtractor
from src.core.metrics import system_snapshot, process_memory_bytes
from src.core.path_utils import default_artifacts_dir, default_cache_dir

try:
    from sentence_transformers import CrossEncoder  # type: ignore
except Exception:
    CrossEncoder = None


def _latest(pattern: str) -> Optional[Path]:
    files = glob.glob(pattern)
    if not files:
        return None
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return Path(files[0])


def _cosine_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    if a.ndim == 1:
        a = a.reshape(1, -1)
    if b.ndim == 1:
        b = b.reshape(1, -1)
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-9)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
    return a_n @ b_n.T


class EmbeddingPairTopKExtractor(BaseExtractor):
    VERSION = "1.0.0"

    def __init__(
        self,
        artifacts_dir: str | None = None,
        cache_dir: str | None = None,
        top_k: int = 5,
        use_cross_encoder: bool = True,
        temperature: float = 0.1,
        device: Optional[str] = "cpu",
    ) -> None:
        self.artifacts_dir = Path(artifacts_dir).expanduser().resolve() if artifacts_dir else default_artifacts_dir()
        self.cache_dir = Path(cache_dir).expanduser().resolve() if cache_dir else (default_cache_dir() / "transcript_embed")
        self.top_k = top_k
        self.temperature = temperature
        self.device = str(device or "cpu")
        # Privacy policy: TranscriptChunkEmbedder does not store raw chunk texts → cross-encoder rerank is disabled by default.
        self.use_cross_encoder = False
        self.cross_model = None

        self._ensure_cross_model()

    def _ensure_cross_model(self) -> None:
        if self.use_cross_encoder and self.cross_model is None and CrossEncoder is not None:
            model_name = "cross-encoder/ms-marco-MiniLM-L-6-v2"
            self.cross_model = CrossEncoder(model_name, device=self.device)

    def _retrieve_topk(self, query: np.ndarray, corpus: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
        # Lazy import FAISS to avoid crashing on incompatible NumPy builds
        try:
            import faiss  # type: ignore
            dim = corpus.shape[1]
            index = faiss.IndexFlatIP(dim)
            faiss.normalize_L2(corpus)
            faiss.normalize_L2(query)
            index.add(corpus)
            scores, indices = index.search(query, min(k, corpus.shape[0]))
            return indices, scores
        except Exception:
            pass
        sims = _cosine_matrix(query, corpus)
        k = min(k, sims.shape[1])
        idx = np.argsort(-sims, axis=1)[:, :k]
        scr = np.take_along_axis(sims, idx, axis=1)
        return idx, scr

    def _load_vector(self, pattern: str) -> Optional[np.ndarray]:
        p = _latest(str(self.artifacts_dir / pattern))
        if p is None:
            return None
        try:
            v = np.load(p)
            return np.asarray(v, dtype=np.float32)
        except Exception:
            return None

    def _load_transcript_chunks(self) -> Tuple[Optional[np.ndarray], None]:
        # Prefer whisper → youtube
        for source in ["whisper", "youtube_auto"]:
            p = _latest(str(self.artifacts_dir / f"transcript_{source}_embedding_*.npy"))
            if p is None:
                continue
            try:
                m = np.load(p)
                m = np.asarray(m, dtype=np.float32)
                return m, None
            except Exception:
                continue
        return None, None

    def extract(self, doc: Any) -> Dict[str, Any]:
        import time

        t0 = time.perf_counter()
        sys_before = system_snapshot()
        mem_before = process_memory_bytes()

        results: Dict[str, Any] = {}

        title = self._load_vector("title_embedding_*.npy")
        desc = self._load_vector("description_embedding_*.npy")
        chunks, chunk_texts = self._load_transcript_chunks()

        if title is not None and desc is not None:
            td = float(_cosine_matrix(title, desc)[0, 0])
            results["title_description_cosine"] = td

        if title is not None and chunks is not None and chunks.size > 0:
            q = title.reshape(1, -1)
            idx, scr = self._retrieve_topk(q, chunks, self.top_k)
            results["title_transcript_topk_cosine"] = scr.flatten().astype(float).tolist()

            if self.use_cross_encoder and chunk_texts:
                title_text = getattr(doc, "title", "")
                flat_idx = idx.flatten().tolist()
                pairs = [[title_text, chunk_texts[i]] for i in flat_idx if 0 <= i < len(chunk_texts)]
                if pairs:
                    with torch.no_grad():
                        logits = self.cross_model.predict(pairs)  # type: ignore[attr-defined]
                    logits = np.asarray(logits, dtype=np.float32).reshape(-1)
                    # sanitize logits
                    if not np.isfinite(logits).all():
                        logits = np.nan_to_num(logits, nan=0.0, posinf=0.0, neginf=0.0)
                    tau = max(self.temperature, 1e-6)
                    z = logits / tau
                    z -= np.max(z)
                    exp_z = np.exp(z)
                    denom = float(np.sum(exp_z))
                    if denom <= 0.0 or not np.isfinite(denom):
                        probs = np.full_like(exp_z, 1.0 / max(len(exp_z), 1), dtype=float)
                    else:
                        probs = (exp_z / (denom + 1e-9)).astype(float)
                    # final sanitize
                    if not np.isfinite(probs).all():
                        probs = np.full_like(exp_z, 1.0 / max(len(exp_z), 1), dtype=float)
                    results["title_transcript_topk_cross"] = probs.tolist()
                else:
                    results["title_transcript_topk_cross"] = []

        sys_after = system_snapshot()
        mem_after = process_memory_bytes()
        total_s = time.perf_counter() - t0

        return {
            "device": self.device,
            "version": self.VERSION,
            "model_version": (
                "cross-encoder/ms-marco-MiniLM-L-6-v2" if (self.use_cross_encoder and self.cross_model is not None) else None
            ),
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
            "result": {"embedding_pair_topk_scores": results},
            "error": None,
        }


