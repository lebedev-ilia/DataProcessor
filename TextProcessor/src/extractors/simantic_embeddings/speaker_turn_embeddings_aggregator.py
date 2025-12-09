from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from src.core.base_extractor import BaseExtractor
from src.core.metrics import system_snapshot, process_memory_bytes
from src.core.model_registry import get_model
from src.core.text_utils import normalize_whitespace


def _hash_payload(payload: str) -> str:
    import hashlib as _h
    return _h.sha256(payload.encode("utf-8")).hexdigest()


def _l2n(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n > 0:
        return v / n
    return v


class SpeakerTurnEmbeddingsAggregatorExtractor(BaseExtractor):
    VERSION = "1.0.0"

    def __init__(
        self,
        model_name: str = "intfloat/multilingual-e5-large",
        artifacts_dir: str = "/home/ilya/Рабочий стол/DataProcessor/TextProcessor/.artifacts",
        device: Optional[str] = None,
        fp16: bool = True,
        batch_size: int = 64,
    ) -> None:
        self.model_name = model_name
        self.artifacts_dir = Path(artifacts_dir)
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.fp16 = fp16 and ("cuda" in self.device)
        self.batch_size = batch_size
        self._model = get_model(self.model_name, self.device, self.fp16)

    def _encode_texts(self, texts: List[str]) -> np.ndarray:
        if not texts:
            print("No texts to encode")
            return np.zeros((0, 0), dtype=np.float32)
        out: List[np.ndarray] = []
        for i in range(0, len(texts), self.batch_size):
            batch = texts[i : i + self.batch_size]
            batch = [normalize_whitespace(t) for t in batch]
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

    def extract(self, doc: Any) -> Dict[str, Any]:
        import time

        t0 = time.perf_counter()
        sys_before = system_snapshot()
        mem_before = process_memory_bytes()

        speakers: Dict[str, Dict[str, Any]] = getattr(doc, "speakers", {}) or {}
        if not isinstance(speakers, dict) or not speakers:
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
                "result": {"speaker_embeddings": {}},
                "error": None,
            }

        # group texts per speaker
        name_to_texts: Dict[str, List[str]] = {}
        for _ts, turn in speakers.items():
            try:
                name = str(turn.get("name", "")).strip() or "unknown"
                text = str(turn.get("description", "")).strip()
                if not text:
                    print(f"No text found for speaker {_ts}")
                    continue
                name_to_texts.setdefault(name, []).append(text)
            except Exception as e:
                print(f"Error processing speaker {_ts}: {e}")
                continue

        results: Dict[str, Any] = {}
        for speaker, texts in name_to_texts.items():
            embs = self._encode_texts(texts)
            if embs.size == 0:
                print(f"No embeddings found for speaker {speaker}")
                continue
            mean_emb = _l2n(np.mean(embs, axis=0))
            max_emb = _l2n(np.max(embs, axis=0))

            # save artifacts
            h = _hash_payload(self.model_name + "||" + speaker + "||" + "\n".join(texts))
            mean_path = self.artifacts_dir / f"speaker_{speaker}_mean_{h}.npy"
            max_path = self.artifacts_dir / f"speaker_{speaker}_max_{h}.npy"
            tmp_m = mean_path.with_suffix(".tmp.npy")
            tmp_x = max_path.with_suffix(".tmp.npy")
            np.save(tmp_m, mean_emb.astype(np.float32))
            np.save(tmp_x, max_emb.astype(np.float32))
            tmp_m.replace(mean_path)
            tmp_x.replace(max_path)
            # write meta with model info (one per speaker embedding)
            import json as _json
            _meta = {"model": self.model_name, "speaker": speaker}
            mean_meta = mean_path.with_suffix(".meta.json")
            max_meta = max_path.with_suffix(".meta.json")
            mean_meta.write_text(_json.dumps(_meta, ensure_ascii=False, indent=2))
            max_meta.write_text(_json.dumps(_meta, ensure_ascii=False, indent=2))

            results[speaker] = {
                "mean": {"path": str(mean_path.resolve()), "count_turns": len(texts)},
                "max": {"path": str(max_path.resolve()), "count_turns": len(texts)},
            }

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
            "result": {"speaker_embeddings": results},
            "error": None,
        }


