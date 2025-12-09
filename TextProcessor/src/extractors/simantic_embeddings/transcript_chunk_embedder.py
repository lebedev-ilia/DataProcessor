"""
TranscriptChunkEmbedder — извлекает эмбеддинги по чанкам из транскрипта.

Особенности (согласованы со стилем проекта):
- Разбиение транскрипта на чанки (по предложениям с overlap);
- L2-нормализация эмбеддингов;
- Кеширование векторов и меты (атомарная запись *.tmp.npy → .npy);
- Метрики: pre_init/post_init/post_process, peaks.ram_peak_mb и peaks.gpu_peak_mb;
- Timings в секундах: timings_s { total };
- Возвращает пути к сохранённым артефактам в result.
"""

from __future__ import annotations

import json
import hashlib
from pathlib import Path
from typing import List, Dict, Any, Optional

import numpy as np
import torch

try:
    import nltk
    from nltk.tokenize import sent_tokenize
except Exception as e:  # pragma: no cover
    raise ImportError("Requires nltk. Install with: pip install nltk") from e

from sentence_transformers import SentenceTransformer
from src.core.model_registry import get_model

from src.core.base_extractor import BaseExtractor
from src.core.metrics import system_snapshot, process_memory_bytes
from src.schemas.models import VideoDocument
from src.core.text_utils import normalize_whitespace


class TranscriptChunkEmbedder(BaseExtractor):
    VERSION = "1.0.0"

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-large",
        cache_dir: str = "/home/ilya/Рабочий стол/DataProcessor/TextProcessor/.cache/transcript_embed",
        artifacts_dir: str = "/home/ilya/Рабочий стол/DataProcessor/TextProcessor/.artifacts",
        device: Optional[str] = None,
        fp16: bool = True,
        batch_size: int = 64,
        max_chunk_tokens: int = 3,
        overlap_ratio: float = 0.15,
    ) -> None:
        # Ensure punkt is available but don't spam downloads
        try:
            nltk.data.find("tokenizers/punkt")
        except Exception:
            try:
                nltk.download("punkt", quiet=True)
            except Exception as e:
                print(f"Error downloading nltk punkt: {e}")
                raise e

        try:
            nltk.data.find("tokenizers/punkt_tab")
        except Exception:
            try:
                nltk.download("punkt_tab", quiet=True)
            except Exception as e:
                print(f"Error downloading nltk punkt_tab: {e}")
                raise e

        self.model_name = model_name
        self.cache_dir = Path(cache_dir)
        self.artifacts_dir = Path(artifacts_dir)
        self.batch_size = batch_size
        self.max_chunk_tokens = max_chunk_tokens
        self.overlap_ratio = overlap_ratio
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)

        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.fp16 = fp16 and ("cuda" in self.device)

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
        try:
            self.model = get_model(self.model_name, self.device, self.fp16)
        except Exception as e:
            # Auto-fallback to CPU on CUDA OOM or similar GPU init failures
            print(f"Error loading model on {self.device}: {e}. Auto-fallback to CPU.")
            msg = str(e)
            if ("CUDA out of memory" in msg or "CUDA" in msg) and self.device and "cuda" in self.device:
                self.device = "cpu"
                self.fp16 = False
                # reduce batch size defensively on CPU
                self.batch_size = min(self.batch_size, 16)
                self.model = get_model(self.model_name, self.device, self.fp16)
            else:
                raise

    # release_resources removed: models are shared via registry and persist

    @staticmethod
    def _hash_text(text: str, model_name: str) -> str:
        payload = (model_name + "||" + text.strip()).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _split_into_chunks(self, text: str) -> List[str]:
        sents = sent_tokenize(text)
        chunks: List[str] = []
        current_tokens: List[str] = []
        token_count = 0

        for sent in sents:
            words = sent.split()
            if token_count + len(words) > self.max_chunk_tokens:
                if current_tokens:
                    chunks.append(" ".join(current_tokens))
                overlap = int(self.max_chunk_tokens * self.overlap_ratio)
                if overlap > 0:
                    current_tokens = current_tokens[-overlap:]
                else:
                    current_tokens = []
                token_count = len(current_tokens)
            current_tokens.extend(words)
            token_count += len(words)

        if current_tokens:
            chunks.append(" ".join(current_tokens))
        return chunks

    def _encode_chunks(self, chunks: List[str]) -> np.ndarray:
        all_embeddings: List[np.ndarray] = []
        for i in range(0, len(chunks), self.batch_size):
            batch = chunks[i : i + self.batch_size]
            with torch.no_grad():
                raw = self.model.encode(
                    batch,
                    show_progress_bar=False,
                    convert_to_numpy=True,
                    normalize_embeddings=False,
                )
            raw = np.asarray(raw, dtype=np.float32)
            norms = np.linalg.norm(raw, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            normed = raw / norms
            all_embeddings.append(normed)
        return np.vstack(all_embeddings) if all_embeddings else np.zeros((0, 0), dtype=np.float32)

    def _gpu_used_mb(self, snap: Any) -> int:
        try:
            g = (snap or {}).get("gpu") or {}
            arr = g.get("gpus") or []
            return max([int(x.get("memory_used_mb", 0)) for x in arr] or [0])
        except Exception:
            return 0

    def extract(self, doc: VideoDocument) -> Dict[str, Any]:
        import time

        started = time.perf_counter()
        sys_before = system_snapshot()
        mem_before = process_memory_bytes()
        error: Optional[str] = None

        # Gather transcripts by source without merging; process each separately
        texts_by_source: Dict[str, str] = {}
        try:
            transcripts_dict = getattr(doc, "transcripts", {}) or {}
            if isinstance(transcripts_dict, dict):
                if transcripts_dict.get("whisper"):
                    texts_by_source["whisper"] = normalize_whitespace(transcripts_dict.get("whisper", ""))
                if transcripts_dict.get("youtube_auto"):
                    texts_by_source["youtube_auto"] = normalize_whitespace(transcripts_dict.get("youtube_auto", ""))
            else:
                print("Transcripts field is not a dict on VideoDocument.")
        except Exception as e:
            print(f"Error getting transcript text: {e}. Returning empty transcript.")

        if not any(t.strip() for t in texts_by_source.values()):
            return {
                "device": self.device,
                "version": self.VERSION,
                "system": {
                    "pre_init": self._init_metrics.get("pre_init"),
                    "post_init": self._init_metrics.get("post_init"),
                    "post_process": sys_before,
                    "peaks": {
                        "ram_peak_mb": int(max(self._init_metrics.get("ram_peak_bytes", 0), mem_before) / 1024 / 1024),
                        "gpu_peak_mb": int(self._gpu_used_mb(sys_before)),
                    },
                },
                "timings_s": {"total": 0.0},
                "result": {"transcript_chunks_by_source": {}},
                "error": "empty transcript",
            }

        results_by_source: Dict[str, Any] = {}

        # Process each available source independently
        for source_key, source_text in texts_by_source.items():
            if not source_text.strip():
                continue

            h = self._hash_text(source_text, self.model_name)
            # store vectors as artifact, meta in cache
            artifacts_vec_path = self.artifacts_dir / f"transcript_{source_key}_embedding_{h}.npy"
            cache_vec_path = self.cache_dir / f"{h}.npy"  # legacy fallback if existed earlier
            cache_meta_path = self.cache_dir / f"{h}.json"

            # try cache
            cached_ok = False
            if cache_meta_path.exists():
                try:
                    meta = json.loads(cache_meta_path.read_text())
                    # ensure artifact exists; if not, try to rebuild from legacy cache vec
                    if not artifacts_vec_path.exists() and cache_vec_path.exists():
                        vectors = np.load(cache_vec_path)
                        tmp_vec = artifacts_vec_path.with_suffix(".tmp.npy")
                        np.save(tmp_vec, np.asarray(vectors, dtype=np.float32))
                        tmp_vec.replace(artifacts_vec_path)
                    results_by_source[source_key] = {
                        "embeddings_path": str(artifacts_vec_path.resolve()) if artifacts_vec_path.exists() else "",
                        "meta_path": str(cache_meta_path.resolve()),
                        "n_chunks": int(meta.get("n_chunks", len(meta.get("chunks", [])))),
                        "embedding_dim": int(meta.get("embedding_dim", 0)),
                    }
                    cached_ok = True
                except Exception:
                    cached_ok = False

            if not cached_ok:
                chunks = self._split_into_chunks(source_text)
                embeddings = self._encode_chunks(chunks)

                # save vectors to artifacts (atomic tmp → final)
                tmp_vec = artifacts_vec_path.with_suffix(".tmp.npy")
                np.save(tmp_vec, embeddings.astype(np.float32))
                tmp_vec.replace(artifacts_vec_path)

                meta = {
                    "source": source_key,
                    "chunks": chunks,
                    "hash": h,
                    "model": self.model_name,
                    "device": self.device,
                    "n_chunks": len(chunks),
                    "embedding_dim": int(embeddings.shape[1]) if embeddings.size > 0 else 0,
                }
                cache_meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2))

                results_by_source[source_key] = {
                    "embeddings_path": str(artifacts_vec_path.resolve()),
                    "meta_path": str(cache_meta_path.resolve()),
                    "n_chunks": len(chunks),
                    "embedding_dim": int(embeddings.shape[1]) if embeddings.size > 0 else 0,
                }

        sys_after = system_snapshot()
        mem_after = process_memory_bytes()
        total_s = (time.perf_counter() - started)

        gpu_peak_mb = max(
            self._gpu_used_mb(self._init_metrics.get("pre_init")),
            self._gpu_used_mb(self._init_metrics.get("post_init")),
            self._gpu_used_mb(sys_after),
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
            "result": {"transcript_chunks_by_source": results_by_source},
            "error": None,
        }


