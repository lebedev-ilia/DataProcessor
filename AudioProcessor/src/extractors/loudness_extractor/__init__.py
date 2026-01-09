"""
Экстрактор громкости: RMS, peak, dBFS и опционально LUFS (если доступен pyloudnorm).
"""

# NOTE: This package replaces the historical `loudness_extractor.py` module.
# Backward compatible import path:
#   from src.extractors.loudness_extractor import LoudnessExtractor

import time
import logging
from typing import Dict, Any, Optional

import numpy as np

from src.core.base_extractor import BaseExtractor, ExtractorResult
from src.core.audio_utils import AudioUtils

logger = logging.getLogger(__name__)


class LoudnessExtractor(BaseExtractor):
    """Извлекает метрики громкости.

    - RMS и peak по всему треку
    - dBFS (20*log10(rms + eps))
    - LUFS при наличии `pyloudnorm` (иначе аккуратно пропускается)
    - frame-wise RMS статистики (mean/std/median/p10/p90) для short-term dynamics
    """

    name = "loudness"
    version = "1.1.0"
    description = "Метрики громкости (RMS, peak, dBFS, опционально LUFS)"
    category = "loudness"
    dependencies = ["numpy", "pyloudnorm"]
    estimated_duration = 0.5

    gpu_required = False
    gpu_preferred = False
    gpu_memory_required = 0.0

    def __init__(
        self,
        device: str = "auto",
        sample_rate: int = 22050,
        frame_length: int = 2048,
        hop_length: int = 512,
        mix_to_mono: bool = True,
    ):
        super().__init__(device=device)
        self.sample_rate = sample_rate
        self.frame_length = int(frame_length)
        self.hop_length = int(hop_length)
        self.audio_utils = AudioUtils(device=device, sample_rate=sample_rate)
        self.mix_to_mono = bool(mix_to_mono)

    def _compute_from_np(self, x: np.ndarray, sr: int) -> Dict[str, Any]:
        x = np.asarray(x, dtype=np.float32)
        if x.size == 0:
            raise ValueError("Пустой аудиосигнал")

        eps = 1e-12
        rms = float(np.sqrt(float(np.mean(x * x)) + eps))
        peak = float(np.max(np.abs(x)) + eps)
        dbfs = float(20.0 * np.log10(rms + eps))

        if x.size >= self.frame_length:
            sq = x * x
            window = np.ones(self.frame_length, dtype=np.float32)
            window_sums = np.convolve(sq, window, mode="valid")
            rms_frames = np.sqrt(window_sums / float(self.frame_length) + eps)
            if self.hop_length > 0:
                rms_frames = rms_frames[:: self.hop_length]
        else:
            rms_frames = np.array([rms], dtype=np.float32)

        f_mean = float(np.mean(rms_frames))
        f_std = float(np.std(rms_frames))
        f_median = float(np.median(rms_frames))
        f_p10 = float(np.percentile(rms_frames, 10))
        f_p90 = float(np.percentile(rms_frames, 90))
        frames_count = int(rms_frames.shape[0])

        lufs_value: Optional[float] = None
        try:
            import pyloudnorm as pyln  # type: ignore

            try:
                meter = pyln.Meter(sr)
                lufs_value = float(meter.integrated_loudness(x))
            except Exception as e:
                logger.warning(f"Loudness: pyloudnorm present but failed to compute LUFS: {e}")
                lufs_value = None
        except Exception:
            lufs_value = None

        return {
            "rms": rms,
            "peak": peak,
            "dbfs": dbfs,
            "lufs": lufs_value,
            "sample_rate": int(sr),
            "duration": float(x.shape[-1] / sr),
            "frame_length": int(self.frame_length),
            "hop_length": int(self.hop_length),
            "frames_count": frames_count,
            "frame_rms_mean": f_mean,
            "frame_rms_std": f_std,
            "frame_rms_median": f_median,
            "frame_rms_p10": f_p10,
            "frame_rms_p90": f_p90,
            "frame_rms_stats_vector": [f_mean, f_std, f_median, f_p10, f_p90],
        }

    def run_segments(self, input_uri: str, tmp_path: str, segments: list[dict]) -> ExtractorResult:
        """
        Compute loudness over Segmenter-provided audio windows and aggregate.
        Produces:
          - per-segment rms/dbfs/peak (+ optional lufs) sequences
          - aggregated stats over segment RMS (mean/std/median/p10/p90)
        """
        start_time = time.time()
        try:
            if not self._validate_input(input_uri):
                return self._create_result(
                    success=False,
                    error="Некорректный входной файл",
                    processing_time=time.time() - start_time,
                )
            if not isinstance(segments, list) or not segments:
                raise ValueError("segments is empty (no-fallback)")

            seg_rms: list[float] = []
            seg_peak: list[float] = []
            seg_dbfs: list[float] = []
            seg_lufs: list[float] = []
            centers: list[float] = []
            lufs_present = True

            for seg in segments:
                ss = int(seg.get("start_sample"))
                es = int(seg.get("end_sample"))
                c = float(seg.get("center_sec"))
                waveform_t, sr = self.audio_utils.load_audio_segment(
                    input_uri, start_sample=ss, end_sample=es, target_sr=self.sample_rate, mix_to_mono=self.mix_to_mono
                )
                x = self.audio_utils.to_numpy(waveform_t)
                if x.ndim == 2:
                    x = np.mean(x, axis=0) if self.mix_to_mono else x[0]
                m = self._compute_from_np(x, int(sr))
                seg_rms.append(float(m["rms"]))
                seg_peak.append(float(m["peak"]))
                seg_dbfs.append(float(m["dbfs"]))
                if m["lufs"] is None:
                    lufs_present = False
                    seg_lufs.append(float("nan"))
                else:
                    seg_lufs.append(float(m["lufs"]))
                centers.append(float(c))

            seg_rms_arr = np.asarray(seg_rms, dtype=np.float32)
            # Aggregate segment RMS stats (robust across long videos)
            agg = {
                "segment_rms_mean": float(np.mean(seg_rms_arr)) if seg_rms_arr.size else float("nan"),
                "segment_rms_std": float(np.std(seg_rms_arr)) if seg_rms_arr.size else float("nan"),
                "segment_rms_median": float(np.median(seg_rms_arr)) if seg_rms_arr.size else float("nan"),
                "segment_rms_p10": float(np.percentile(seg_rms_arr, 10)) if seg_rms_arr.size else float("nan"),
                "segment_rms_p90": float(np.percentile(seg_rms_arr, 90)) if seg_rms_arr.size else float("nan"),
            }

            # Keep backward-compatible globals by also computing full-track metrics.
            wav_full_t, sr_full = self.audio_utils.load_audio(input_uri, self.sample_rate)
            x_full = self.audio_utils.to_numpy(wav_full_t)
            if x_full.ndim == 2:
                x_full = np.mean(x_full, axis=0) if self.mix_to_mono else x_full[0]
            full = self._compute_from_np(x_full, int(sr_full))

            payload: Dict[str, Any] = {
                **full,
                **agg,
                "segments_count": int(len(segments)),
                "segment_centers_sec": np.asarray(centers, dtype=np.float32),
                "segment_rms": seg_rms_arr,
                "segment_peak": np.asarray(seg_peak, dtype=np.float32),
                "segment_dbfs": np.asarray(seg_dbfs, dtype=np.float32),
                "segment_lufs": np.asarray(seg_lufs, dtype=np.float32),
                "lufs_present": bool(lufs_present),
                "device_used": self.device,
            }
            return self._create_result(True, payload=payload, processing_time=time.time() - start_time)
        except Exception as e:
            processing_time = time.time() - start_time
            self._log_extraction_error(input_uri, str(e), processing_time)
            return self._create_result(False, error=str(e), processing_time=processing_time)

    def run(self, input_uri: str, tmp_path: str) -> ExtractorResult:
        start_time = time.time()
        try:
            if not self._validate_input(input_uri):
                return self._create_result(
                    success=False,
                    error="Некорректный входной файл",
                    processing_time=time.time() - start_time,
                )

            self._log_extraction_start(input_uri)

            waveform_t, sr = self.audio_utils.load_audio(input_uri, self.sample_rate)
            x = self.audio_utils.to_numpy(waveform_t)
            if x.ndim == 2:
                x = np.mean(x, axis=0) if self.mix_to_mono else x[0]

            metrics = self._compute_from_np(x, int(sr))

            processing_time = time.time() - start_time

            payload: Dict[str, Any] = {
                **metrics,
                "device_used": self.device,
            }

            self._log_extraction_success(input_uri, processing_time)
            return self._create_result(True, payload=payload, processing_time=processing_time)

        except Exception as e:
            processing_time = time.time() - start_time
            self._log_extraction_error(input_uri, str(e), processing_time)
            return self._create_result(False, error=str(e), processing_time=processing_time)


