"""
Экстрактор громкости: RMS, peak, dBFS и опционально LUFS (если доступен pyloudnorm).
"""
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

            # Mix to mono (опционально) — безопасный подход
            if x.ndim == 2:
                if self.mix_to_mono:
                    x = np.mean(x, axis=0)
                else:
                    x = x[0]

            x = x.astype(np.float32)
            if x.size == 0:
                raise ValueError("Пустой аудиосигнал")

            # Small epsilon to avoid log(0)
            eps = 1e-12

            # Global metrics
            rms = float(np.sqrt(float(np.mean(x * x)) + eps))
            peak = float(np.max(np.abs(x)) + eps)
            dbfs = float(20.0 * np.log10(rms + eps))

            # Frame-wise RMS (fast via convolution)
            if x.size >= self.frame_length:
                # sum of squares over frames via convolution (valid mode)
                sq = x * x
                window = np.ones(self.frame_length, dtype=np.float32)
                # valid length: N - frame_length + 1
                window_sums = np.convolve(sq, window, mode="valid")
                # RMS per sample-window
                rms_frames = np.sqrt(window_sums / float(self.frame_length) + eps)
                # downsample to hop positions
                if self.hop_length > 0:
                    rms_frames = rms_frames[:: self.hop_length]
            else:
                # short signal -> single frame RMS
                rms_frames = np.array([rms], dtype=np.float32)

            # Frame-wise statistics
            f_mean = float(np.mean(rms_frames))
            f_std = float(np.std(rms_frames))
            f_median = float(np.median(rms_frames))
            f_p10 = float(np.percentile(rms_frames, 10))
            f_p90 = float(np.percentile(rms_frames, 90))
            frames_count = int(rms_frames.shape[0])

            # Try to compute LUFS (pyloudnorm). If not available — gracefully skip.
            lufs_value: Optional[float] = None
            try:
                import pyloudnorm as pyln  # type: ignore
                try:
                    meter = pyln.Meter(sr)  # BS.1770 meter
                    # pyloudnorm expects float samples in [-1..1]
                    lufs_value = float(meter.integrated_loudness(x))
                except Exception as e:
                    # pyloudnorm present but failed: log and continue
                    logger.warning(f"Loudness: pyloudnorm present but failed to compute LUFS: {e}")
                    lufs_value = None
            except Exception:
                # pyloudnorm not installed — acceptable; we don't fail extraction
                logger.debug("Loudness: pyloudnorm not installed, skipping LUFS computation")
                lufs_value = None

            processing_time = time.time() - start_time

            payload: Dict[str, Any] = {
                "rms": rms,
                "peak": peak,
                "dbfs": dbfs,
                "lufs": lufs_value,
                "sample_rate": sr,
                "duration": float(x.shape[-1] / sr),
                "frame_length": self.frame_length,
                "hop_length": self.hop_length,
                "frames_count": frames_count,
                # frame RMS aggregate stats and vector for easy ML concatenation
                "frame_rms_mean": f_mean,
                "frame_rms_std": f_std,
                "frame_rms_median": f_median,
                "frame_rms_p10": f_p10,
                "frame_rms_p90": f_p90,
                "frame_rms_stats_vector": [f_mean, f_std, f_median, f_p10, f_p90],
                "device_used": self.device,
            }

            self._log_extraction_success(input_uri, processing_time)
            return self._create_result(True, payload=payload, processing_time=processing_time)

        except Exception as e:
            processing_time = time.time() - start_time
            self._log_extraction_error(input_uri, str(e), processing_time)
            return self._create_result(False, error=str(e), processing_time=processing_time)