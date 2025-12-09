"""
Экстрактор онсетов (пики атак) с использованием librosa.
"""
import time
import logging
from typing import Dict, Any, Optional

import numpy as np
import librosa

from src.core.base_extractor import BaseExtractor, ExtractorResult
from src.core.audio_utils import AudioUtils

logger = logging.getLogger(__name__)


class OnsetExtractor(BaseExtractor):
    """Выделение моментов онсетов и метрик по ним."""

    name = "onset"
    version = "1.1.0"
    description = "Определение онсетов (атака звука)"
    category = "rhythm"
    dependencies = ["librosa", "numpy"]
    estimated_duration = 0.8

    gpu_required = False
    gpu_preferred = False
    gpu_memory_required = 0.0

    def __init__(
        self,
        device: str = "auto",
        sample_rate: int = 22050,
        hop_length: int = 512,
        pre_max: int = 3,
        post_max: int = 3,
        pre_avg: int = 3,
        post_avg: int = 5,
        delta: float = 0.2,
        wait: int = 10,
    ):
        super().__init__(device=device)
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.audio_utils = AudioUtils(device=device, sample_rate=sample_rate)
        self.librosa_params = dict(
            pre_max=pre_max,
            post_max=post_max,
            pre_avg=pre_avg,
            post_avg=post_avg,
            delta=delta,
            wait=wait,
        )

    def run(self, input_uri: str, tmp_path: str) -> ExtractorResult:
        start_time = time.time()
        try:
            if not self._validate_input(input_uri):
                return self._create_result(
                    False, error="Некорректный входной файл", processing_time=time.time() - start_time
                )

            self._log_extraction_start(input_uri)

            # Загружаем аудио
            y_t, sr = self.audio_utils.load_audio(input_uri, self.sample_rate)
            y = self.audio_utils.to_numpy(y_t)
            if y.ndim == 2:
                # Выбираем канал с максимальной RMS энергии
                rms = np.mean(y**2, axis=1)
                y = y[np.argmax(rms)]

            onset_times = self._extract_onsets(y, sr)

            intervals = np.diff(onset_times) if onset_times.size > 1 else np.array([])
            avg_interval = float(np.mean(intervals)) if intervals.size else None
            density = float(onset_times.size / (y.shape[-1] / sr + 1e-9))

            interval_stats = {
                "interval_std": float(np.std(intervals)) if intervals.size else None,
                "interval_min": float(np.min(intervals)) if intervals.size else None,
                "interval_max": float(np.max(intervals)) if intervals.size else None,
                "interval_median": float(np.median(intervals)) if intervals.size else None,
            }

            payload: Dict[str, Any] = {
                "onset_times": onset_times.astype(np.float32),
                "onset_count": int(onset_times.size),
                "avg_interval_sec": avg_interval,
                "onset_density_per_sec": density,
                "sample_rate": sr,
                "hop_length": self.hop_length,
                "duration": float(y.shape[-1] / sr),
                "device_used": self.device,
                **interval_stats,
                "insufficient_onsets": onset_times.size <= 1,
            }

            dt = time.time() - start_time
            self._log_extraction_success(input_uri, dt)
            return self._create_result(True, payload=payload, processing_time=dt)

        except Exception as e:
            dt = time.time() - start_time
            self._log_extraction_error(input_uri, str(e), dt)
            return self._create_result(False, error=str(e), processing_time=dt)

    def _extract_onsets(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Попытка извлечения онсетов через Essentia, fallback на Librosa."""
        try:
            import essentia.standard as es
            audio = y.astype(np.float32)
            od = es.OnsetRate()
            _, onset_times = od(audio)
            onset_times = np.array(onset_times, dtype=np.float32)
            if onset_times.ndim == 0:
                onset_times = onset_times.reshape(1)
            if onset_times.size > 0:
                # logger.info(f"Onset: Essentia, найдено {onset_times.size} онсетов")
                return onset_times
            logger.warning("Onset: Essentia вернула 0 онсетов, fallback на Librosa")
        except Exception as e:
            pass
            # logger.info(f"Onset: Essentia недоступна, fallback на Librosa (причина: {e})")

        # Librosa fallback
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=self.hop_length)
        onset_frames = librosa.onset.onset_detect(
            onset_envelope=onset_env, sr=sr, hop_length=self.hop_length, units="time", **self.librosa_params
        )
        onset_times = np.array(onset_frames, dtype=np.float32)
        if onset_times.ndim == 0:
            onset_times = onset_times.reshape(1)
        # logger.info(f"Onset: Librosa, найдено {onset_times.size} онсетов")
        return onset_times
