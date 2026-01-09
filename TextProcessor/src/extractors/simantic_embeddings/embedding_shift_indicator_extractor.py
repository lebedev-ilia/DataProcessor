from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np

from src.core.base_extractor import BaseExtractor
from src.core.metrics import system_snapshot, process_memory_bytes
from src.core.path_utils import default_artifacts_dir


class EmbeddingShiftIndicatorExtractor(BaseExtractor):
    VERSION = "1.0.0"

    def __init__(self, n_window_chunks: int = 2, cosine_threshold: float = 0.85) -> None:
        self.n_window_chunks = int(max(1, n_window_chunks))
        self.cosine_threshold = float(cosine_threshold)

    @staticmethod
    def _cosine(a: np.ndarray, b: np.ndarray) -> float:
        denom = float(np.linalg.norm(a) * np.linalg.norm(b)) + 1e-8
        return float(np.dot(a, b) / denom)

    def extract(self, doc: Any) -> Dict[str, Any]:
        import time

        t0 = time.perf_counter()
        sys_before = system_snapshot()
        mem_before = process_memory_bytes()

        # берем пути к чанковым эмбеддингам из результатов TranscriptChunkEmbedder
        # ожидается, что до этого экстрактор уже сохранил артефакты
        # путь читается из features.results_by_extractor.TranscriptChunkEmbedder.transcript_chunks_by_source
        # здесь мы не имеем прямого доступа к features; поэтому читаем из артефактов по шаблону
        # предпочтём whisper, затем youtube_auto
        from glob import glob
        import os
        from pathlib import Path

        artifacts_dir = default_artifacts_dir()

        def _latest(pattern: str) -> Optional[Path]:
            files = glob(str(artifacts_dir / pattern))
            if not files:
                return None
            files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            return Path(files[0])

        path = _latest("transcript_whisper_embedding_*.npy") or _latest("transcript_youtube_auto_embedding_*.npy")

        if path is None:
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
                "result": {"embedding_shift_indicator": {"error": "no_transcripts_found"}},
                "error": None,
            }

        emb = np.load(path)
        emb = np.asarray(emb, dtype=np.float32)
        if emb.ndim == 1:
            emb = emb.reshape(1, -1)
        n_chunks = emb.shape[0]
        win = min(self.n_window_chunks, max(1, n_chunks // 2))

        if n_chunks < 2:
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
                "result": {"embedding_shift_indicator": {"error": "not_enough_chunks", "n_chunks": int(n_chunks)}},
                "error": None,
            }

        start_emb = emb[:win].mean(axis=0)
        end_emb = emb[-win:].mean(axis=0)
        cosine_shift = self._cosine(start_emb, end_emb)
        shift_flag = bool(cosine_shift < self.cosine_threshold)

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
            "result": {
                "embedding_shift_indicator": {
                    "cosine_begin_end": float(cosine_shift),
                    "shift_flag": shift_flag,
                    "n_chunks": int(n_chunks),
                    "n_window_chunks": int(win),
                    "path": str(path.resolve()),
                }
            },
            "error": None,
        }


