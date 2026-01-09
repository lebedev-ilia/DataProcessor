"""
Emotion diarization extractor (Triton-backed) + Segmenter time windows.

Policy:
- NO runtime downloads (ModelManager enforced).
- Uses Segmenter `audio/segments.json` family: `emotion`.
- `<5s` audio -> ERROR.
- Truly silent audio -> EMPTY (payload.status="empty", empty_reason="audio_silent").

Outputs (payload):
- emotion_probs: np.ndarray [N, C] float32 (per-window probabilities)
- emotion_id: np.ndarray [N] int32 (argmax per window)
- emotion_confidence: np.ndarray [N] float32 (max prob per window)
- emotion_mean_probs: np.ndarray [C] float32
- emotion_entropy: float
- dominant_emotion_id: int
- dominant_emotion_prob: float
- segment_start_sec/end_sec/center_sec: lists[float] aligned with N
"""

from __future__ import annotations

import os
import time
import logging
from typing import Any, Dict, List, Optional

import numpy as np

from src.core.base_extractor import BaseExtractor, ExtractorResult
from src.core.audio_utils import AudioUtils
from dp_triton import TritonHttpClient, TritonError  # type: ignore

logger = logging.getLogger(__name__)


class EmotionDiarizationExtractor(BaseExtractor):
    name = "emotion_diarization_extractor"
    version = "2.0.0"
    description = "Emotion diarization via Triton (probs + aggregates)"
    category = "speech"
    dependencies = ["numpy", "dp_triton", "dp_models"]
    estimated_duration = 6.0

    gpu_required = False
    gpu_preferred = True
    gpu_memory_required = 0.0  # Triton-backed

    def __init__(
        self,
        device: str = "auto",
        model_size: str = "small",
        sample_rate: int = 16000,
        batch_size: int = 16,
    ):
        super().__init__(device=device)
        self.model_size = str(model_size or "small").strip().lower()
        if self.model_size not in ("small", "large"):
            raise ValueError(f"emotion_diarization | unsupported model_size={self.model_size}. Expected: small|large")
        self.sample_rate = int(sample_rate)
        self.batch_size = max(1, int(batch_size))

        self.audio_utils = AudioUtils(device=device, sample_rate=self.sample_rate)

        # ModelManager: resolve Triton runtime params (no-network).
        try:
            from dp_models import get_global_model_manager  # type: ignore

            self._mm = get_global_model_manager()
        except Exception as e:
            raise RuntimeError(f"emotion_diarization | ModelManager is required but failed to init: {e}") from e

        spec_name = f"emotion_diarization_{self.model_size}_triton"
        try:
            self.model_spec = self._mm.get_spec(model_name=spec_name)
            _dev, _prec, rt, _eng, wd, _arts = self._mm.resolve(self.model_spec)
            if str(rt) != "triton":
                raise RuntimeError(f"emotion_diarization | expected runtime=triton in spec {spec_name}, got {rt}")
            self.model_name = str(self.model_spec.model_name)
            self.weights_digest = str(wd)
            rp = self.model_spec.runtime_params or {}
            self.triton_http_url = self._expand_env(str(rp.get("triton_http_url") or os.environ.get("TRITON_HTTP_URL") or ""))
            self.triton_model_name = str(rp.get("triton_model_name") or "")
            self.triton_model_version = rp.get("triton_model_version")
            self.triton_input_name = str(rp.get("triton_input_name") or "AUDIO__0")
            self.triton_input_datatype = str(rp.get("triton_input_datatype") or "FP32")
            self.triton_output_name = str(rp.get("triton_output_probs_name") or "PROBS__0")
            self.triton_output_datatype = str(rp.get("triton_output_probs_datatype") or "FP32")
            self.emotion_labels = rp.get("emotion_labels") if isinstance(rp.get("emotion_labels"), list) else []
            if not self.triton_http_url or not self.triton_model_name:
                raise RuntimeError("emotion_diarization | Triton runtime_params missing triton_http_url/triton_model_name")
        except Exception as e:
            raise RuntimeError(f"emotion_diarization | failed to resolve model spec via ModelManager: {e}") from e

        self._client = TritonHttpClient(base_url=self.triton_http_url, timeout_sec=10.0)

    def _expand_env(self, s: str) -> str:
        if "${" not in str(s):
            return str(s)
        import re

        def repl(m):
            return os.environ.get(m.group(1), "")

        return re.sub(r"\$\{([^}]+)\}", repl, str(s))

    @staticmethod
    def _rms_and_peak(x: np.ndarray) -> tuple[float, float]:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return 0.0, 0.0
        rms = float(np.sqrt(float(np.mean(x * x)) + 1e-12))
        peak = float(np.max(np.abs(x)) + 1e-12)
        return rms, peak

    def run_segments(self, input_uri: str, tmp_path: str, segments: List[Dict[str, Any]]) -> ExtractorResult:
        start_time = time.time()
        try:
            if not self._validate_input(input_uri):
                return self._create_result(False, error="Некорректный входной файл", processing_time=time.time() - start_time)
            if not isinstance(segments, list) or not segments:
                raise ValueError("segments is empty (no-fallback)")

            dur_sec = float(max((float(s.get("end_sec", 0.0)) for s in segments), default=0.0))
            if dur_sec < 5.0:
                raise RuntimeError(f"emotion_diarization | audio too short (<5s): duration_sec={dur_sec:.3f}")

            waves: list[np.ndarray] = []
            starts: list[float] = []
            ends: list[float] = []
            centers: list[float] = []
            lens: list[int] = []
            for seg in segments:
                ss = int(seg.get("start_sample"))
                es = int(seg.get("end_sample"))
                st = float(seg.get("start_sec"))
                en = float(seg.get("end_sec"))
                c = float(seg.get("center_sec"))
                wav_t, sr = self.audio_utils.load_audio_segment(input_uri, start_sample=ss, end_sample=es, target_sr=self.sample_rate)
                wav = self.audio_utils.to_numpy(wav_t)
                wav = wav[0] if wav.ndim == 2 else wav.reshape(-1)
                wav = np.asarray(wav, dtype=np.float32).reshape(-1)
                if int(sr) != int(self.sample_rate):
                    raise RuntimeError(f"emotion_diarization | segment SR mismatch: got {sr} expected {self.sample_rate}")
                waves.append(wav)
                lens.append(int(wav.shape[0]))
                starts.append(st)
                ends.append(en)
                centers.append(c)

            max_len = int(max(lens) if lens else 0)
            if max_len <= 0:
                raise RuntimeError("emotion_diarization | no audio samples in segments")

            concat = np.concatenate([w for w in waves if w.size], axis=0) if waves else np.zeros((0,), dtype=np.float32)
            rms, peak = self._rms_and_peak(concat)
            # Conservative silence detection: require both low peak and low rms.
            if peak < 1e-3 and rms < 1e-4:
                payload: Dict[str, Any] = {
                    "status": "empty",
                    "empty_reason": "audio_silent",
                    "segments_count": int(len(segments)),
                    "sample_rate": int(self.sample_rate),
                    "rms": float(rms),
                    "peak": float(peak),
                    "model_name": self.model_name,
                    "emotion_labels": self.emotion_labels,
                    "device_used": "cuda",
                }
                return self._create_result(True, payload=payload, processing_time=time.time() - start_time)

            if not self._client.ready():
                raise TritonError(f"Triton is not ready at {self.triton_http_url}", error_code="triton_unavailable")

            padded = np.zeros((len(waves), max_len), dtype=np.float32)
            for i, w in enumerate(waves):
                padded[i, : int(w.shape[0])] = w

            probs_chunks: list[np.ndarray] = []
            for start in range(0, padded.shape[0], self.batch_size):
                batch = padded[start : start + self.batch_size]
                res = self._client.infer(
                    model_name=self.triton_model_name,
                    model_version=(str(self.triton_model_version) if self.triton_model_version else None),
                    input_name=self.triton_input_name,
                    input_tensor=batch,
                    output_name=self.triton_output_name,
                    datatype=self.triton_input_datatype,
                ).output
                p = np.asarray(res, dtype=np.float32)
                if p.ndim != 2 or p.shape[0] != batch.shape[0]:
                    raise RuntimeError(f"emotion_diarization | unexpected probs shape from Triton: {p.shape}")
                probs_chunks.append(p)

            probs = np.concatenate(probs_chunks, axis=0) if probs_chunks else np.zeros((0, 0), dtype=np.float32)
            if probs.shape[0] != padded.shape[0]:
                raise RuntimeError(f"emotion_diarization | probs batch mismatch: probs={probs.shape} windows={padded.shape[0]}")

            # normalize (defensive)
            s = np.sum(probs, axis=1, keepdims=True) + 1e-9
            probs = probs / s

            emotion_id = np.argmax(probs, axis=1).astype(np.int32)
            emotion_conf = np.max(probs, axis=1).astype(np.float32)

            mean_probs = np.mean(probs, axis=0).astype(np.float32) if probs.size else np.zeros((0,), dtype=np.float32)
            ent = float(-np.sum(mean_probs * np.log(mean_probs + 1e-9))) if mean_probs.size else 0.0
            dominant_id = int(np.argmax(mean_probs)) if mean_probs.size else -1
            dominant_prob = float(np.max(mean_probs)) if mean_probs.size else 0.0

            payload = {
                "emotion_probs": probs,
                "emotion_id": emotion_id,
                "emotion_confidence": emotion_conf,
                "emotion_labels": self.emotion_labels,
                "emotion_mean_probs": mean_probs,
                "emotion_entropy": float(ent),
                "dominant_emotion_id": int(dominant_id),
                "dominant_emotion_prob": float(dominant_prob),
                "segment_start_sec": starts,
                "segment_end_sec": ends,
                "segment_center_sec": centers,
                "segments_count": int(len(segments)),
                "sample_rate": int(self.sample_rate),
                "device_used": "cuda",
                "rms": float(rms),
                "peak": float(peak),
                "model_name": self.model_name,
            }
            return self._create_result(True, payload=payload, processing_time=time.time() - start_time)

        except TritonError as e:
            return self._create_result(False, error=str(e), processing_time=time.time() - start_time)
        except Exception as e:
            return self._create_result(False, error=str(e), processing_time=time.time() - start_time)

    def run(self, input_uri: str, tmp_path: str) -> ExtractorResult:
        return self._create_result(
            success=False,
            error="emotion_diarization_extractor | run() is not supported in production. Use run_segments() with Segmenter families.emotion windows.",
            processing_time=0.0,
        )

    def _validate_input(self, input_uri: str) -> bool:
        if not super()._validate_input(input_uri):
            return False
        audio_extensions = {".wav", ".mp3", ".flac", ".m4a", ".mp4", ".avi", ".mov"}
        if not any(input_uri.lower().endswith(ext) for ext in audio_extensions):
            self.logger.error(f"Файл не является поддерживаемым аудио/видео форматом: {input_uri}")
            return False
        return True

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_size": self.model_size,
            "sample_rate": self.sample_rate,
            "batch_size": self.batch_size,
            "device": self.device,
            "model_name": getattr(self, "model_name", None),
            "weights_digest": getattr(self, "weights_digest", None),
        }


