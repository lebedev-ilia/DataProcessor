from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from src.core.base_extractor import BaseExtractor
from src.core.metrics import system_snapshot, process_memory_bytes
from src.schemas.models import VideoDocument
from src.core.text_utils import normalize_whitespace


class TranscriptAggregatorExtractor(BaseExtractor):
    VERSION = "1.0.0"
    DEFAULT_EMBED_DIM = 384

    def __init__(
        self,
        artifacts_dir: Optional[str] = None,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = "cpu",
        decay_rate: float = 0.01,
    ) -> None:
        from src.core.path_utils import default_artifacts_dir  # local import to avoid cycles

        self.artifacts_dir = Path(artifacts_dir).expanduser().resolve() if artifacts_dir else default_artifacts_dir()
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.device = str(device or "cpu")
        self.decay_rate = decay_rate
        # no heavy model; pure tensor ops

    @staticmethod
    def _normalize(vec: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(vec, p=2, dim=-1)

    @staticmethod
    def _position_decay(n_chunks: int, decay_rate: float = 0.01) -> torch.Tensor:
        positions = torch.arange(n_chunks, dtype=torch.float32)
        weights = torch.exp(-decay_rate * positions)
        return weights

    def _aggregate_mean(
        self,
        chunk_embeddings: List[np.ndarray],
        asr_confidence: Optional[List[float]] = None,
        decay_rate: float = 0.01,
    ) -> Dict[str, Any]:
        if not chunk_embeddings:
            return {"embedding": np.zeros((self.DEFAULT_EMBED_DIM,), dtype=np.float32), "count": 0, "std": 0.0}

        emb = torch.tensor(np.stack(chunk_embeddings), dtype=torch.float32, device="cpu")
        n_chunks = emb.shape[0]
        weights = self._position_decay(n_chunks, decay_rate)

        if asr_confidence is not None and len(asr_confidence) == n_chunks:
            weights = weights * torch.tensor(asr_confidence, dtype=torch.float32)
        weights = weights / (weights.sum() + 1e-8)

        weighted = (emb * weights.unsqueeze(1)).sum(dim=0)
        normed = self._normalize(weighted)

        return {"embedding": normed.cpu().numpy(), "count": n_chunks, "std": float(emb.std().item())}

    def _aggregate_maxpool(self, chunk_embeddings: List[np.ndarray]) -> Dict[str, Any]:
        if not chunk_embeddings:
            return {"embedding": np.zeros((self.DEFAULT_EMBED_DIM,), dtype=np.float32), "count": 0, "std": 0.0}

        emb = torch.tensor(np.stack(chunk_embeddings), dtype=torch.float32)
        maxpooled = emb.max(dim=0).values
        normed = self._normalize(maxpooled)

        return {"embedding": normed.cpu().numpy(), "count": emb.shape[0], "std": float(emb.std().item())}

    @staticmethod
    def _hash_text(text: str, model_name: str) -> str:
        import hashlib
        payload = (model_name + "||" + text.strip()).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _load_source_embeddings(self, text: str, source_key: str) -> Optional[np.ndarray]:
        h = self._hash_text(text, self.model_name)
        emb_path = self.artifacts_dir / f"transcript_{source_key}_embedding_{h}.npy"
        if not emb_path.exists():
            return None
        try:
            arr = np.load(emb_path)
            arr = np.asarray(arr, dtype=np.float32)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            return arr
        except Exception:
            return None

    def extract(self, doc: VideoDocument) -> Dict[str, Any]:
        import time

        t0 = time.perf_counter()
        sys_before = system_snapshot()
        mem_before = process_memory_bytes()

        transcripts = getattr(doc, "transcripts", {}) or {}
        whisper_text = normalize_whitespace(transcripts.get("whisper", "")) if isinstance(transcripts, dict) else ""
        youtube_text = normalize_whitespace(transcripts.get("youtube_auto", "")) if isinstance(transcripts, dict) else ""

        # confidences: optional list for whisper
        whisper_conf = None
        if isinstance(transcripts, dict):
            whisper_conf = transcripts.get("whisper_confidence")
            if not isinstance(whisper_conf, list):
                whisper_conf = None

        load_s = 0.0
        agg_s = 0.0

        # Load embeddings for each source if available
        t_load = time.perf_counter()
        emb_whisper = self._load_source_embeddings(whisper_text, "whisper") if whisper_text else None
        emb_youtube = self._load_source_embeddings(youtube_text, "youtube_auto") if youtube_text else None
        load_s = time.perf_counter() - t_load

        results: Dict[str, Any] = {}
        t_agg = time.perf_counter()
        # Aggregate per source
        if emb_whisper is not None:
            chunks_list = [emb_whisper[i] for i in range(emb_whisper.shape[0])]
            mean_res = self._aggregate_mean(chunks_list, asr_confidence=whisper_conf, decay_rate=self.decay_rate)
            max_res = self._aggregate_maxpool(chunks_list)
            h_w = self._hash_text(whisper_text, self.model_name)
            mean_path_w = self.artifacts_dir / f"transcript_whisper_agg_mean_{h_w}.npy"
            max_path_w = self.artifacts_dir / f"transcript_whisper_agg_max_{h_w}.npy"
            tmp_mean_w = mean_path_w.with_suffix(".tmp.npy")
            tmp_max_w = max_path_w.with_suffix(".tmp.npy")
            np.save(tmp_mean_w, mean_res["embedding"].astype(np.float32))
            np.save(tmp_max_w, max_res["embedding"].astype(np.float32))
            tmp_mean_w.replace(mean_path_w)
            tmp_max_w.replace(max_path_w)
            results["whisper"] = {
                "aggregate_mean": {"path": str(mean_path_w.resolve()), "count": int(mean_res["count"]), "std": float(mean_res["std"])},
                "aggregate_maxpool": {"path": str(max_path_w.resolve()), "count": int(max_res["count"]), "std": float(max_res["std"])},
            }

        if emb_youtube is not None:
            chunks_list = [emb_youtube[i] for i in range(emb_youtube.shape[0])]
            mean_res = self._aggregate_mean(chunks_list, asr_confidence=None, decay_rate=self.decay_rate)
            max_res = self._aggregate_maxpool(chunks_list)
            h_y = self._hash_text(youtube_text, self.model_name)
            mean_path_y = self.artifacts_dir / f"transcript_youtube_auto_agg_mean_{h_y}.npy"
            max_path_y = self.artifacts_dir / f"transcript_youtube_auto_agg_max_{h_y}.npy"
            tmp_mean_y = mean_path_y.with_suffix(".tmp.npy")
            tmp_max_y = max_path_y.with_suffix(".tmp.npy")
            np.save(tmp_mean_y, mean_res["embedding"].astype(np.float32))
            np.save(tmp_max_y, max_res["embedding"].astype(np.float32))
            tmp_mean_y.replace(mean_path_y)
            tmp_max_y.replace(max_path_y)
            results["youtube_auto"] = {
                "aggregate_mean": {"path": str(mean_path_y.resolve()), "count": int(mean_res["count"]), "std": float(mean_res["std"])},
                "aggregate_maxpool": {"path": str(max_path_y.resolve()), "count": int(max_res["count"]), "std": float(max_res["std"])},
            }

        # Combined over all sources (if both present)
        if emb_whisper is not None or emb_youtube is not None:
            combined_list: List[np.ndarray] = []
            if emb_whisper is not None:
                combined_list.extend([emb_whisper[i] for i in range(emb_whisper.shape[0])])
            if emb_youtube is not None:
                combined_list.extend([emb_youtube[i] for i in range(emb_youtube.shape[0])])
            if combined_list:
                mean_res = self._aggregate_mean(combined_list, asr_confidence=None, decay_rate=self.decay_rate)
                max_res = self._aggregate_maxpool(combined_list)
                h_c = self._hash_text((whisper_text + "\n" + youtube_text), self.model_name)
                mean_path_c = self.artifacts_dir / f"transcript_combined_agg_mean_{h_c}.npy"
                max_path_c = self.artifacts_dir / f"transcript_combined_agg_max_{h_c}.npy"
                tmp_mean_c = mean_path_c.with_suffix(".tmp.npy")
                tmp_max_c = max_path_c.with_suffix(".tmp.npy")
                np.save(tmp_mean_c, mean_res["embedding"].astype(np.float32))
                np.save(tmp_max_c, max_res["embedding"].astype(np.float32))
                tmp_mean_c.replace(mean_path_c)
                tmp_max_c.replace(max_path_c)
                results["combined"] = {
                    "aggregate_mean": {"path": str(mean_path_c.resolve()), "count": int(mean_res["count"]), "std": float(mean_res["std"])},
                    "aggregate_maxpool": {"path": str(max_path_c.resolve()), "count": int(max_res["count"]), "std": float(max_res["std"])},
                }
        agg_s = time.perf_counter() - t_agg

        sys_after = system_snapshot()
        mem_after = process_memory_bytes()
        total_s = time.perf_counter() - t0

        return {
            "device": self.device,
            "version": self.VERSION,
            "system": {
                "pre_init": sys_before,
                "post_init": sys_before,  # aggregator has no heavy init
                "post_process": sys_after,
                "peaks": {
                    "ram_peak_mb": int(max(mem_before, mem_after) / 1024 / 1024),
                    "gpu_peak_mb": 0,
                },
            },
            "timings_s": {"load": round(load_s, 3), "aggregate": round(agg_s, 3), "total": round(total_s, 3)},
            "result": {"transcript_aggregates": results},
            "error": None,
        }


