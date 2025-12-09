"""
Экстрактор спектральной энтропии (измеряет распределённость энергии по спектру).
"""
import time
import logging
from typing import Dict, Any

import numpy as np

from src.core.base_extractor import BaseExtractor, ExtractorResult
from src.core.audio_utils import AudioUtils

logger = logging.getLogger(__name__)


class SpectralEntropyExtractor(BaseExtractor):
    name = "spectral_entropy"
    version = "1.1.0"
    description = "Спектральная энтропия и её статистики"
    category = "spectral"
    dependencies = ["librosa", "numpy"]
    estimated_duration = 0.9

    gpu_required = False
    gpu_preferred = False
    gpu_memory_required = 0.0

    def __init__(
        self,
        device: str = "auto",
        sample_rate: int = 22050,
        n_fft: int = 2048,
        hop_length: int = 512,
        average_channels: bool = True,
        smoothing_window: int = 0,
        use_mel: bool = False,
        n_mels: int = 128,
    ):
        super().__init__(device=device)
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.audio_utils = AudioUtils(device=device, sample_rate=sample_rate)
        self.average_channels = bool(average_channels)
        self.smoothing_window = int(smoothing_window)
        self.use_mel = bool(use_mel)
        self.n_mels = int(n_mels)

    def run(self, input_uri: str, tmp_path: str) -> ExtractorResult:
        start_time = time.time()
        try:
            if not self._validate_input(input_uri):
                return self._create_result(False, error="Некорректный входной файл", processing_time=time.time() - start_time)

            self._log_extraction_start(input_uri)

            y_t, sr = self.audio_utils.load_audio(input_uri, self.sample_rate)
            y = self.audio_utils.to_numpy(y_t)
            if y.ndim == 2:
                # [C, T] -> моно: усредняем каналы для репрезентативности
                y = np.mean(y, axis=0) if self.average_channels else y[0]

            import librosa

            # Спектр мощности: STFT или Mel
            if not self.use_mel:
                S = np.abs(librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)) ** 2  # power
            else:
                S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=self.n_fft, hop_length=self.hop_length, n_mels=self.n_mels, power=2.0)
            # Нормируем по частотной оси для каждой временной колонки, затем энтропия Шеннона
            P = S / (np.sum(S, axis=0, keepdims=True) + 1e-12)
            ent = -np.sum(P * np.log2(P + 1e-12), axis=0)

            # Доп. метрики: spectral flatness и spread (на основе P)
            # Flatness: геометрическое/арифметическое среднее по частоте
            # Добавляем eps для численной устойчивости
            eps = 1e-12
            logP = np.log(P + eps)
            flatness = np.exp(np.mean(logP, axis=0)) / (np.mean(P, axis=0) + eps)
            # Spread: дисперсия индекса частоты (нормированный индекс 0..1) под P
            freq_idx = np.linspace(0.0, 1.0, P.shape[0], dtype=np.float32).reshape(-1, 1)
            mu = np.sum(freq_idx * P, axis=0)
            spread = np.sqrt(np.sum(((freq_idx - mu) ** 2) * P, axis=0))

            # Опциональное скользящее сглаживание для визуализации/стабильности
            if self.smoothing_window and self.smoothing_window > 1:
                w = self.smoothing_window
                kernel = np.ones(w, dtype=np.float32) / float(w)
                ent = np.convolve(ent, kernel, mode="same")
                flatness = np.convolve(flatness, kernel, mode="same")
                spread = np.convolve(spread, kernel, mode="same")

            ent = ent.astype(np.float32)
            flatness = flatness.astype(np.float32)
            spread = spread.astype(np.float32)
            payload: Dict[str, Any] = {
                "spectral_entropy_series": ent.tolist(),
                "spectral_entropy_mean": float(np.mean(ent)),
                "spectral_entropy_std": float(np.std(ent)),
                "spectral_flatness_series": flatness.tolist(),
                "spectral_flatness_mean": float(np.mean(flatness)),
                "spectral_flatness_std": float(np.std(flatness)),
                "spectral_spread_series": spread.tolist(),
                "spectral_spread_mean": float(np.mean(spread)),
                "spectral_spread_std": float(np.std(spread)),
                "sample_rate": sr,
                "n_fft": self.n_fft,
                "hop_length": self.hop_length,
                "use_mel": self.use_mel,
                "n_mels": self.n_mels,
                "average_channels": self.average_channels,
                "smoothing_window": self.smoothing_window,
                "duration": float(len(y) / sr),
                "device_used": self.device,
            }

            dt = time.time() - start_time
            self._log_extraction_success(input_uri, dt)
            return self._create_result(True, payload=payload, processing_time=dt)

        except Exception as e:
            dt = time.time() - start_time
            self._log_extraction_error(input_uri, str(e), dt)
            return self._create_result(False, error=str(e), processing_time=dt)


