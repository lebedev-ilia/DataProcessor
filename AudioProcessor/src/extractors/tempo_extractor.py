"""
Экстрактор оценки темпа (BPM) на базе librosa.
"""
import time
import logging
from typing import Dict, Any, Optional

import numpy as np

from src.core.base_extractor import BaseExtractor, ExtractorResult
from src.core.audio_utils import AudioUtils

logger = logging.getLogger(__name__)


class TempoExtractor(BaseExtractor):
    """Экстрактор темпа (BPM) с использованием onset-энергии и beat tracking.

    Лёгкий по зависимостям (librosa) и подходит для CPU/GPU пайплайна.
    """

    name = "tempo"
    version = "1.1.0"
    description = "Оценка темпа (BPM) на основе onset-энергии"
    category = "rhythm"
    dependencies = ["librosa", "numpy"]
    estimated_duration = 1.0

    gpu_required = False
    gpu_preferred = False
    gpu_memory_required = 0.0

    def __init__(
        self,
        device: str = "auto",
        sample_rate: int = 22050,
        hop_length: int = 512,
        aggregate: str = "median",
        average_channels: bool = True,
        windowed_bpm: bool = False,
        window_sec: float = 15.0,
        step_sec: float = 5.0,
    ):
        super().__init__(device=device)
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.aggregate = aggregate
        self.audio_utils = AudioUtils(device=device, sample_rate=sample_rate)
        self.average_channels = bool(average_channels)
        self.windowed_bpm = bool(windowed_bpm)
        self.window_sec = float(window_sec)
        self.step_sec = float(step_sec)

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

            # Загружаем аудио
            waveform_t, sr = self.audio_utils.load_audio(input_uri, self.sample_rate)

            # Переводим в numpy для librosa
            # Ожидаем форму (channels, samples) или (1, samples)
            waveform_np = self.audio_utils.to_numpy(waveform_t)
            if waveform_np.ndim == 2:
                waveform_np = np.mean(waveform_np, axis=0) if self.average_channels else waveform_np[0]

            # Вычисляем onset-энергию и темп
            import librosa  # локальный импорт для лёгкого fallback при отсутствии

            onset_env = librosa.onset.onset_strength(
                y=waveform_np, sr=sr, hop_length=self.hop_length
            )

            # Получаем массив оценок темпа (при aggregate=None) и агрегированное значение
            # Новое API с 0.10: librosa.feature.rhythm.tempo
            try:
                from librosa.feature.rhythm import tempo as rhythm_tempo
                tempo_all = rhythm_tempo(
                    onset_envelope=onset_env,
                    sr=sr,
                    hop_length=self.hop_length,
                    aggregate=None,
                )
            except (ImportError, AttributeError):
                # Fallback на старое API (до 1.0)
                # self.logger.info(
                #     "Tempo: librosa.feature.rhythm.tempo недоступно, fallback на librosa.beat.tempo"
                # )
                tempo_all = librosa.beat.tempo(
                    onset_envelope=onset_env,
                    sr=sr,
                    hop_length=self.hop_length,
                    aggregate=None,
                )

            if tempo_all is None or len(tempo_all) == 0:
                raise RuntimeError("Не удалось оценить темп")

            tempo_median = float(np.median(tempo_all))
            tempo_mean = float(np.mean(tempo_all))
            tempo_std = float(np.std(tempo_all))

            # Простая оценка "уверенности" как инверсия относительного разброса
            confidence = float(1.0 / (1.0 + (tempo_std / (tempo_mean + 1e-6))))

            processing_time = time.time() - start_time

            # Опциональная оценка BPM по окнам для лупов/диджейских треков
            window_series = None
            try:
                if self.windowed_bpm:
                    win = max(1, int(self.window_sec * sr))
                    step = max(1, int(self.step_sec * sr))
                    if len(waveform_np) >= win:
                        bpms = []
                        times = []
                        for start in range(0, len(waveform_np) - win + 1, step):
                            segment = waveform_np[start:start+win]
                            env_seg = librosa.onset.onset_strength(y=segment, sr=sr, hop_length=self.hop_length)
                            try:
                                from librosa.feature.rhythm import tempo as rhythm_tempo
                                t_all = rhythm_tempo(onset_envelope=env_seg, sr=sr, hop_length=self.hop_length, aggregate=None)
                            except Exception:
                                t_all = librosa.beat.tempo(onset_envelope=env_seg, sr=sr, hop_length=self.hop_length, aggregate=None)
                            if t_all is not None and len(t_all) > 0:
                                bpms.append(float(np.median(t_all)))
                                times.append(float(start / sr))
                        if bpms:
                            window_series = {
                                "times_sec": times,
                                "bpm": bpms,
                                "bpm_mean": float(np.mean(bpms)),
                                "bpm_median": float(np.median(bpms)),
                                "bpm_std": float(np.std(bpms)),
                            }
            except Exception:
                window_series = None

            # Предупреждения: экстремальный темп или низкая уверенность
            warnings: list[str] = []
            if tempo_median < 40 or tempo_median > 220:
                warnings.append("tempo_out_of_range")
            if confidence < 0.3:
                warnings.append("low_confidence")
            if np.max(np.abs(waveform_np)) < 0.02:
                warnings.append("signal_too_quiet")

            payload: Dict[str, Any] = {
                "tempo_estimates": tempo_all.astype(np.float32),
                "tempo_bpm": tempo_median if self.aggregate == "median" else tempo_mean,
                "tempo_bpm_mean": tempo_mean,
                "tempo_bpm_median": tempo_median,
                "tempo_bpm_std": tempo_std,
                "confidence": confidence,
                "windowed_bpm": window_series,
                "warnings": warnings,
                "sample_rate": sr,
                "hop_length": self.hop_length,
                "duration": float(waveform_np.shape[-1] / sr),
                "device_used": self.device,
            }

            self._log_extraction_success(input_uri, processing_time)
            return self._create_result(True, payload=payload, processing_time=processing_time)

        except Exception as e:
            processing_time = time.time() - start_time
            self._log_extraction_error(input_uri, str(e), processing_time)
            return self._create_result(False, error=str(e), processing_time=processing_time)


