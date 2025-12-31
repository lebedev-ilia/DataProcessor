"""
Ритмический экстрактор: оценки биений и регулярности (librosa).
"""
import time
import logging
from typing import Dict, Any

import numpy as np

from src.core.base_extractor import BaseExtractor, ExtractorResult
from src.core.audio_utils import AudioUtils

logger = logging.getLogger(__name__)


class RhythmicExtractor(BaseExtractor):
    """Базовые ритмические метрики: такты, регулярность, средний период, плотность ударов."""

    name = "rhythmic"
    version = "1.1.0"
    description = "Ритмические метрики: темп, биты, регулярность"
    category = "rhythm"
    dependencies = ["librosa", "numpy"]
    estimated_duration = 1.2

    gpu_required = False
    gpu_preferred = False
    gpu_memory_required = 0.0

    def __init__(self, device: str = "auto", sample_rate: int = 22050, hop_length: int = 512, average_channels: bool = True):
        super().__init__(device=device)
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.audio_utils = AudioUtils(device=device, sample_rate=sample_rate)
        self.average_channels = bool(average_channels)

    def run(self, input_uri: str, tmp_path: str) -> ExtractorResult:
        start_time = time.time()
        try:
            if not self._validate_input(input_uri):
                return self._create_result(False, error="Некорректный входной файл", processing_time=time.time() - start_time)

            self._log_extraction_start(input_uri)

            y_t, sr = self.audio_utils.load_audio(input_uri, self.sample_rate)
            y = self.audio_utils.to_numpy(y_t)
            if y.ndim == 2:
                # [C, T] -> моно: усредняем каналы для более устойчивой оценки ритма
                y = np.mean(y, axis=0) if self.average_channels else y[0]

            # Essentia в приоритете; fallback на librosa
            beat_times = None
            tempo = 0.0
            try:
                import essentia
                import essentia.standard as es
                audio = y.astype(np.float32)
                # Onset detection + beat tracking в Essentia
                od = es.OnsetRate()
                onset_rate, onset_times = od(audio)
                bt = es.BeatTrackerMultiFeature()
                beats, ticks = bt(audio)
                beat_times = np.array(beats, dtype=np.float32)
                # Темп оценим как медиану межударных интервалов
                if beat_times.size > 1:
                    intervals = np.diff(beat_times)
                    tempo = float(60.0 / (np.median(intervals) + 1e-6))
            except Exception as e:
                # self.logger.info(
                #     f"Rhythmic: Essentia недоступна, fallback на librosa (причина: {e})"
                # )
                import librosa
                onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=self.hop_length)
                tempo, beat_frames = librosa.beat.beat_track(onset_envelope=onset_env, sr=sr, hop_length=self.hop_length)
                beat_times = librosa.frames_to_time(beat_frames, sr=sr, hop_length=self.hop_length)

            # Интервалы между ударами
            intervals = np.diff(beat_times) if beat_times.size > 1 else np.array([])
            avg_period = float(np.mean(intervals)) if intervals.size else 0.0
            std_period = float(np.std(intervals)) if intervals.size else 0.0
            # Коэффициент вариации (std/mean) как относительная нерегулярность, регулярность = 1/(1+cv)
            cv = float(std_period / (avg_period + 1e-9)) if intervals.size else 0.0
            regularity = float(1.0 / (1.0 + cv))
            beat_density = float(beat_times.size / (y.shape[-1] / sr + 1e-9))

            # Доп. метрики темпа и интервалов
            median_period = float(np.median(intervals)) if intervals.size else 0.0
            min_period = float(np.min(intervals)) if intervals.size else 0.0
            max_period = float(np.max(intervals)) if intervals.size else 0.0
            median_bpm = float(60.0 / (median_period + 1e-9)) if intervals.size else float(tempo)

            payload: Dict[str, Any] = {
                "rhythm_tempo_bpm": float(tempo),
                "rhythm_beats_count": int(beat_times.size),
                "rhythm_avg_period_sec": avg_period,
                "rhythm_period_std_sec": std_period,
                "rhythm_regularity": regularity,
                "rhythm_beat_density": beat_density,
                "rhythm_median_period_sec": median_period,
                "rhythm_min_period_sec": min_period,
                "rhythm_max_period_sec": max_period,
                "rhythm_median_bpm": median_bpm,
                "sample_rate": sr,
                "hop_length": self.hop_length,
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


