"""
Спектральный экстрактор: centroid, bandwidth, flatness, rolloff, ZCR, contrast.
"""
import time
import logging
from typing import Dict, Any

import numpy as np

from src.core.base_extractor import BaseExtractor, ExtractorResult
from src.core.audio_utils import AudioUtils
from src.utils.prof import timeit

logger = logging.getLogger(__name__)


class SpectralExtractor(BaseExtractor):
    """Извлекает базовые спектральные признаки и их статистики."""

    name = "spectral"
    version = "1.1.0"
    description = "Спектральные признаки: centroid, bandwidth, flatness, rolloff, ZCR, contrast"
    category = "spectral"
    dependencies = ["librosa", "numpy"]
    estimated_duration = 1.2

    gpu_required = False
    gpu_preferred = False
    gpu_memory_required = 0.0

    def __init__(self, device: str = "auto", sample_rate: int = 22050, hop_length: int = 512, n_fft: int = 2048, average_channels: bool = True, keep_contrast_bands: bool = True):
        super().__init__(device=device)
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.audio_utils = AudioUtils(device=device, sample_rate=sample_rate)
        self.average_channels = bool(average_channels)
        self.keep_contrast_bands = bool(keep_contrast_bands)

    def run(self, input_uri: str, tmp_path: str) -> ExtractorResult:
        start_time = time.time()
        try:
            if not self._validate_input(input_uri):
                return self._create_result(False, error="Некорректный входной файл", processing_time=time.time() - start_time)

            self._log_extraction_start(input_uri)

            with timeit("spectral: load_audio"):
                y_t, sr = self.audio_utils.load_audio(input_uri, self.sample_rate)
            y = self.audio_utils.to_numpy(y_t)
            if y.ndim == 2:
                # усредняем каналы для более репрезентативной спектральной оценки
                y = np.mean(y, axis=0) if self.average_channels else y[0]

            import librosa

            with timeit("spectral: centroid"):
                centroid = librosa.feature.spectral_centroid(y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length)[0]
            with timeit("spectral: bandwidth"):
                bandwidth = librosa.feature.spectral_bandwidth(y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length)[0]
            with timeit("spectral: flatness"):
                flatness = librosa.feature.spectral_flatness(y=y, n_fft=self.n_fft, hop_length=self.hop_length)[0]
            with timeit("spectral: rolloff"):
                rolloff = librosa.feature.spectral_rolloff(y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length)[0]
            with timeit("spectral: zcr"):
                zcr = librosa.feature.zero_crossing_rate(y=y, hop_length=self.hop_length)[0]
            try:
                with timeit("spectral: contrast"):
                    contrast_full = librosa.feature.spectral_contrast(y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length)
                contrast = contrast_full.mean(axis=0)
            except Exception:
                contrast_full = None
                contrast = None

            # Дополнительные признаки: спектральный наклон и flatness в дБ
            # Для наклона используем линейную регрессию амплитуды (в дБ) по частоте на каждом кадре
            try:
                import librosa
                with timeit("spectral: stft"):
                    S = np.abs(librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)) + 1e-12
                S_db = 20.0 * np.log10(S)
                freqs = librosa.fft_frequencies(sr=sr, n_fft=self.n_fft)
                # Нормализуем частоты для устойчивости
                x_f = (freqs - freqs.mean()) / (freqs.std() + 1e-12)
                # Оценка наклона по МНК: slope = cov(x, y) / var(x) для каждого кадра
                x_centered = x_f[:, None]
                y_centered = S_db - S_db.mean(axis=0, keepdims=True)
                num = np.sum(x_centered * y_centered, axis=0)
                den = np.sum(x_centered * x_centered, axis=0) + 1e-12
                spectral_slope = (num / den).astype(np.float32)
                spectral_flatness_db = (10.0 * np.log10(flatness + 1e-12)).astype(np.float32)
            except Exception:
                spectral_slope = None
                spectral_flatness_db = None

            def stats(x: np.ndarray) -> Dict[str, float]:
                return {
                    "mean": float(np.mean(x)),
                    "std": float(np.std(x)),
                    "min": float(np.min(x)),
                    "max": float(np.max(x)),
                }

            payload: Dict[str, Any] = {
                "spectral_centroid_stats": stats(centroid.astype(np.float32)),
                "spectral_bandwidth_stats": stats(bandwidth.astype(np.float32)),
                "spectral_flatness_stats": stats(flatness.astype(np.float32)),
                "spectral_rolloff_stats": stats(rolloff.astype(np.float32)),
                "zcr_stats": stats(zcr.astype(np.float32)),
                "spectral_contrast_stats": (stats(contrast.astype(np.float32)) if contrast is not None else None),
                # Полосы контраста по желанию
                **({"spectral_contrast_bands": contrast_full.astype(np.float32).tolist()} if (contrast_full is not None and self.keep_contrast_bands) else {}),
                # Доп. метрики
                **({"spectral_slope_stats": stats(spectral_slope)} if spectral_slope is not None else {}),
                **({"spectral_flatness_db_stats": stats(spectral_flatness_db)} if spectral_flatness_db is not None else {}),
                "sample_rate": sr,
                "hop_length": self.hop_length,
                "n_fft": self.n_fft,
                "average_channels": self.average_channels,
                "keep_contrast_bands": self.keep_contrast_bands,
                "duration": float(y.shape[-1] / sr),
                "device_used": self.device,
            }

            dt = time.time() - start_time
            self._log_extraction_success(input_uri, dt)
            return self._create_result(True, payload=payload, processing_time=dt)

        except Exception as e:
            dt = time.time() - start_time
            self._log_extraction_error(input_uri, str(e), dt)
            return self._create_result(False, error=str(e), processing_time=dt)


