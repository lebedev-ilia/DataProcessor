from __future__ import annotations

import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from src.core.base_extractor import BaseExtractor
from src.core.metrics import system_snapshot, process_memory_bytes
from src.core.model_registry import get_model
from src.core.path_utils import default_artifacts_dir
from src.core.text_utils import normalize_whitespace
from src.schemas.models import VideoDocument


class QAEmbeddingPairsExtractor(BaseExtractor):
    """
    Извлекает вопросоподобные фразы из транскрипта и считает их эмбеддинги.
    """

    VERSION = "1.0.0"
    QUESTION_REGEX = re.compile(r".*\b(кто|что|где|когда|почему|зачем|как)\b.*\?", re.IGNORECASE)

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        artifacts_dir: str | None = None,
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
        self._model = get_model(self.model_name, self.device, self.fp16)

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        # Простое разбиение по знакам конца предложения
        parts = re.split(r"[.!?]+\s+", text)
        # убрать пустые и тримминг
        return [normalize_whitespace(p) for p in parts if normalize_whitespace(p)]

    def _encode(self, sentences: List[str]) -> np.ndarray:
        if not sentences:
            return np.zeros((0, 0), dtype=np.float32)
        out: List[np.ndarray] = []
        for i in range(0, len(sentences), self.batch_size):
            batch = sentences[i : i + self.batch_size]
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
        import json

        t0 = time.perf_counter()
        sys_before = system_snapshot()
        mem_before = process_memory_bytes()

        # Собираем источники: title, description, transcript, comments
        sources: Dict[str, List[str]] = {"title": [], "description": [], "transcript": [], "comments": []}

        # title
        if getattr(doc, "title", None):
            sources["title"].append(normalize_whitespace(str(doc.title)))
        # description
        if getattr(doc, "description", None):
            sources["description"].append(normalize_whitespace(str(doc.description)))
        # transcript
        tr = getattr(doc, "transcripts", {}) or {}
        if isinstance(tr, dict):
            tx = []
            if tr.get("whisper"):
                tx.append(str(tr.get("whisper")))
            if tr.get("youtube_auto"):
                tx.append(str(tr.get("youtube_auto")))
            if tx:
                sources["transcript"].append(normalize_whitespace(" ".join(tx)))
        # comments
        cmts = getattr(doc, "comments", []) or []
        for c in cmts:
            try:
                txt = normalize_whitespace(str(c.text))
                if txt:
                    sources["comments"].append(txt)
            except Exception:
                continue

        # Извлекаем вопросы по каждому источнику
        candidate_texts: List[str] = []
        candidate_sources: List[str] = []
        per_source_counts: Dict[str, int] = {"title": 0, "description": 0, "transcript": 0, "comments": 0}

        # title/description/transcript: режем на предложения
        for key in ["title", "description", "transcript"]:
            for block in sources[key]:
                sentences = self._split_sentences(block)
                for s in sentences:
                    if self.QUESTION_REGEX.match(s):
                        candidate_texts.append(s)
                        candidate_sources.append(key)
                        per_source_counts[key] += 1
        # comments: каждая строка как отдельное предложение
        for cm in sources["comments"]:
            if self.QUESTION_REGEX.match(cm):
                candidate_texts.append(cm)
                candidate_sources.append("comments")
                per_source_counts["comments"] += 1

        embs = self._encode(candidate_texts)
        num_q = len(candidate_texts)

        # Сохраняем как артефакт матрицу вопросов (N×D) и мета по источникам
        out_path = self.artifacts_dir / "qa_question_embeddings.npy"
        tmp = out_path.with_suffix(".tmp.npy")
        np.save(tmp, embs.astype(np.float32))
        tmp.replace(out_path)
        meta = {
            # Privacy-safe: do not persist raw question texts by default.
            "question_hashes": [str(abs(hash(t))) for t in candidate_texts],
            "sources": candidate_sources,
            "per_source_counts": per_source_counts,
            "model": self.model_name,
        }
        meta_path = self.artifacts_dir / "qa_question_embeddings_meta.json"
        meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))

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
                "qa_embeddings": {
                    "path": str(out_path.resolve()),
                    "meta_path": str(meta_path.resolve()),
                    "num_questions": int(num_q),
                    "embedding_dim": int(embs.shape[1]) if embs.size > 0 else 0,
                    "per_source_counts": per_source_counts,
                }
            },
            "error": None,
        }


