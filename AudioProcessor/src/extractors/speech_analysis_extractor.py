"""
SpeechAnalysisExtractor (non-baseline, production-safe).

Goal:
- Provide a compact "speech overview" by combining:
  - ASR token IDs (Triton-backed) on Segmenter families.asr windows
  - Speaker diarization (Triton-backed) on Segmenter families.diarization windows
  - Optional pitch (signal processing) on full audio

Important:
- No raw transcript text is stored.
- No "alignment" between ASR tokens and diarization speakers is attempted (Whisper token timing is not available).
- No runtime downloads (ModelManager enforced by sub-extractors).

Empty/Error policy:
- If audio is < 5 sec -> error.
- If audio is truly silent -> empty (payload.status="empty", empty_reason="audio_silent").
"""

from __future__ import annotations

import time
import logging
from typing import Any, Dict, List, Optional

import numpy as np

from src.core.base_extractor import BaseExtractor, ExtractorResult
from src.core.audio_utils import AudioUtils
from src.extractors.asr_extractor import ASRExtractor
from src.extractors.speaker_diarization_extractor import SpeakerDiarizationExtractor
from src.extractors.pitch_extractor import PitchExtractor

logger = logging.getLogger(__name__)


class SpeechAnalysisExtractor(BaseExtractor):
    name = "speech_analysis_extractor"
    version = "2.0.0"
    description = "Speech analysis bundle: ASR token stats + diarization + optional pitch (no raw text)"
    category = "speech"
    dependencies = ["dp_models", "dp_triton", "numpy", "librosa"]
    estimated_duration = 10.0

    gpu_required = False
    gpu_preferred = True
    gpu_memory_required = 0.0  # Triton-backed ASR/diarization; pitch is CPU

    def __init__(
        self,
        device: str = "auto",
        *,
        asr_model_size: str = "small",
        diarization_model_size: str = "small",
        sample_rate: int = 16000,
        pitch_enabled: bool = False,
        pitch_backend: str = "classic",
    ):
        super().__init__(device=device)
        self.sample_rate = int(sample_rate)
        self.pitch_enabled = bool(pitch_enabled)

        self.audio_utils = AudioUtils(device=device, sample_rate=self.sample_rate)

        # Sub-extractors (already audited):
        self.asr_extractor = ASRExtractor(device=device, model_size=str(asr_model_size), sample_rate=self.sample_rate)
        self.diarization_extractor = SpeakerDiarizationExtractor(
            device=device,
            model_size=str(diarization_model_size),
            sample_rate=self.sample_rate,
        )
        self.pitch_extractor = PitchExtractor(
            device=device,
            sample_rate=self.sample_rate,
            backend=str(pitch_backend or "classic"),
        )

    @staticmethod
    def _rms_and_peak(x: np.ndarray) -> tuple[float, float]:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return 0.0, 0.0
        rms = float(np.sqrt(float(np.mean(x * x)) + 1e-12))
        peak = float(np.max(np.abs(x)) + 1e-12)
        return rms, peak

    def run_bundle(self, input_uri: str, tmp_path: str, *, asr_segments: List[Dict[str, Any]], diar_segments: List[Dict[str, Any]]) -> ExtractorResult:
        start_time = time.time()
        try:
            if not self._validate_input(input_uri):
                return self._create_result(False, error="Некорректный входной файл", processing_time=time.time() - start_time)
            if not isinstance(asr_segments, list) or not asr_segments:
                raise ValueError("speech_analysis | asr_segments is empty (no-fallback)")
            if not isinstance(diar_segments, list) or not diar_segments:
                raise ValueError("speech_analysis | diar_segments is empty (no-fallback)")

            dur_sec = float(
                max(
                    max((float(s.get("end_sec", 0.0)) for s in asr_segments), default=0.0),
                    max((float(s.get("end_sec", 0.0)) for s in diar_segments), default=0.0),
                )
            )
            if dur_sec < 5.0:
                raise RuntimeError(f"speech_analysis | audio too short (<5s): duration_sec={dur_sec:.3f}")

            # Silence detection on a small probe window (first diar window): load segment and check rms/peak.
            # We do not want to accidentally mask broken extraction; load_audio_segment will fail on broken files.
            try:
                probe = diar_segments[0]
                wav_t, _sr = self.audio_utils.load_audio_segment(
                    input_uri,
                    start_sample=int(probe.get("start_sample")),
                    end_sample=int(probe.get("end_sample")),
                    target_sr=self.sample_rate,
                )
                wav = self.audio_utils.to_numpy(wav_t)
                wav = wav[0] if wav.ndim == 2 else wav.reshape(-1)
                rms, peak = self._rms_and_peak(wav)
            except Exception as e:
                raise RuntimeError(f"speech_analysis | failed to probe audio for silence detection: {e}") from e

            if peak < 1e-3 and rms < 1e-4:
                payload: Dict[str, Any] = {
                    "status": "empty",
                    "empty_reason": "audio_silent",
                    "duration_sec": float(dur_sec),
                    "sample_rate": int(self.sample_rate),
                    "device_used": "cuda",
                }
                return self._create_result(True, payload=payload, processing_time=time.time() - start_time)

            # Run sub-extractors (segments-driven)
            asr_res = self.asr_extractor.run_segments(input_uri, tmp_path, asr_segments)
            if not asr_res.success:
                raise RuntimeError(f"speech_analysis | asr failed: {asr_res.error}")
            diar_res = self.diarization_extractor.run_segments(input_uri, tmp_path, diar_segments)
            if not diar_res.success:
                raise RuntimeError(f"speech_analysis | diarization failed: {diar_res.error}")

            pitch_payload = None
            if self.pitch_enabled:
                p = self.pitch_extractor.run(input_uri, tmp_path)
                if p.success and isinstance(p.payload, dict):
                    pitch_payload = p.payload

            asr_payload = asr_res.payload or {}
            diar_payload = diar_res.payload or {}

            # ASR token stats (token IDs are stored in asr_extractor NPZ; here we keep small summaries)
            token_ids_by_segment = asr_payload.get("token_ids_by_segment") or []
            if isinstance(token_ids_by_segment, list):
                token_counts = np.asarray([len(x or []) for x in token_ids_by_segment], dtype=np.float32)
            else:
                token_counts = np.zeros((0,), dtype=np.float32)

            lang_ids = np.asarray(asr_payload.get("lang_id_by_segment") or [], dtype=np.int32).reshape(-1)

            token_total = float(np.sum(token_counts)) if token_counts.size else 0.0
            token_mean = float(np.mean(token_counts)) if token_counts.size else 0.0
            token_std = float(np.std(token_counts)) if token_counts.size else 0.0
            token_density = float(token_total / max(1e-6, dur_sec))

            # Diarization stats
            speaker_segments = diar_payload.get("speaker_segments") or []
            if not isinstance(speaker_segments, list):
                speaker_segments = []
            speaker_ids = np.asarray(diar_payload.get("speaker_ids") or [], dtype=np.int32).reshape(-1)
            speaker_count = int(diar_payload.get("speaker_count") or (len(set(int(s.get("speaker_id", 0)) for s in speaker_segments)) if speaker_segments else 0))

            # Dominant speaker share by total duration over diar windows
            dur_by_spk: Dict[int, float] = {}
            for s in speaker_segments:
                try:
                    sid = int(s.get("speaker_id", 0))
                    d = float(s.get("duration", float(s.get("end", 0.0)) - float(s.get("start", 0.0))) or 0.0)
                    dur_by_spk[sid] = dur_by_spk.get(sid, 0.0) + max(0.0, d)
                except Exception:
                    continue
            total_speech_dur = float(sum(dur_by_spk.values())) if dur_by_spk else 0.0
            dominant_share = float(max(dur_by_spk.values()) / max(1e-6, total_speech_dur)) if dur_by_spk else 0.0

            # Pitch summaries (if present)
            pitch_f0_mean = float(pitch_payload.get("f0_mean", 0.0) or 0.0) if isinstance(pitch_payload, dict) else 0.0
            pitch_f0_std = float(pitch_payload.get("f0_std", 0.0) or 0.0) if isinstance(pitch_payload, dict) else 0.0

            payload_out: Dict[str, Any] = {
                "duration_sec": float(dur_sec),
                "sample_rate": int(self.sample_rate),
                "device_used": "cuda",
                # ASR summaries
                "asr_segments_count": int(asr_payload.get("segments_count") or len(token_counts)),
                "asr_token_total": float(token_total),
                "asr_token_mean": float(token_mean),
                "asr_token_std": float(token_std),
                "asr_token_density_per_sec": float(token_density),
                "asr_lang_id_by_segment": lang_ids.tolist(),
                # Diarization summaries
                "diar_segments_count": int(diar_payload.get("segments_count") or len(speaker_segments)),
                "speaker_count": int(speaker_count),
                "dominant_speaker_share": float(dominant_share),
                "speaker_ids": speaker_ids.tolist(),
                # Optional pitch
                "pitch_enabled": bool(self.pitch_enabled and pitch_payload is not None),
                "pitch_f0_mean": float(pitch_f0_mean),
                "pitch_f0_std": float(pitch_f0_std),
            }

            return self._create_result(True, payload=payload_out, processing_time=time.time() - start_time)
        except Exception as e:
            return self._create_result(False, error=str(e), processing_time=time.time() - start_time)

    def run(self, input_uri: str, tmp_path: str) -> ExtractorResult:
        return self._create_result(
            success=False,
            error="speech_analysis_extractor | requires Segmenter window families; use run_bundle(asr_segments, diar_segments).",
            processing_time=0.0,
        )


