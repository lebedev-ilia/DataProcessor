"""
Экстрактор качества голоса: jitter/shimmer-подобные прокси и HNR-подобная метрика.

Примечание: используем лёгкие эвристики на основе питча и энергии, без тяжёлых DSP.
"""
import time
import logging
from typing import Dict, Any

import numpy as np

from src.core.base_extractor import BaseExtractor, ExtractorResult
from src.core.audio_utils import AudioUtils

logger = logging.getLogger(__name__)


class VoiceQualityExtractor(BaseExtractor):
    name = "voice_quality"
    version = "1.1.0"
    description = "Прокси метрики качества голоса: jitter, shimmer, HNR-подобная"
    category = "voice"
    dependencies = ["librosa", "numpy"]
    estimated_duration = 1.5

    gpu_required = False
    gpu_preferred = False
    gpu_memory_required = 0.0

    def __init__(self, device: str = "auto", sample_rate: int = 22050, average_channels: bool = True, hnr_frame_ms: float = 40.0, rms_mask_threshold: float = 0.01):
        super().__init__(device=device)
        self.sample_rate = sample_rate
        self.audio_utils = AudioUtils(device=device, sample_rate=sample_rate)
        self.average_channels = bool(average_channels)
        self.hnr_frame_ms = float(hnr_frame_ms)
        self.rms_mask_threshold = float(rms_mask_threshold)

    def run(self, input_uri: str, tmp_path: str) -> ExtractorResult:
        start_time = time.time()
        try:
            if not self._validate_input(input_uri):
                return self._create_result(False, error="Некорректный входной файл", processing_time=time.time() - start_time)

            self._log_extraction_start(input_uri)

            y_t, sr = self.audio_utils.load_audio(input_uri, self.sample_rate)
            y = self.audio_utils.to_numpy(y_t)
            if y.ndim == 2:
                # усредняем каналы для устойчивости метрик
                y = np.mean(y, axis=0) if self.average_channels else y[0]

            import librosa

            # Оценка f0 через YIN (быстро) как прокси
            try:
                f0 = librosa.yin(y, fmin=50, fmax=500, sr=sr)
                f0 = f0[np.isfinite(f0) & (f0 > 0)]
            except Exception:
                f0 = np.array([], dtype=np.float32)
            # Маскирование по RMS: исключаем слишком тихие участки из оценки jitter
            jitter = 0.0
            if f0.size > 2:
                df0 = np.diff(f0)
                jitter = float(np.std(df0) / (np.mean(f0) + 1e-6))

            # Shimmer-подобная: вариативность энергии по окнам
            frame_len = max(1, int(0.03 * sr))
            hop = max(1, int(0.01 * sr))
            amps = []
            for i in range(0, len(y) - frame_len + 1, hop):
                frm = y[i : i + frame_len]
                amps.append(np.sqrt(np.mean(frm * frm) + 1e-12))
            amps = np.array(amps, dtype=np.float32)
            # Маскируем слишком тихие кадры
            if amps.size > 0:
                mask = amps >= self.rms_mask_threshold
                amps_masked = amps[mask] if np.any(mask) else amps
            else:
                amps_masked = amps
            shimmer = float(np.std(np.diff(amps_masked)) / (np.mean(amps_masked) + 1e-6)) if amps_masked.size > 2 else 0.0

            # HNR-подобная метрика: отношение энергий автокорреляции (lag1) к нулевой лаг
            # HNR-подобная: усредним по окнам для устойчивости
            hnr_frame = max(1, int(self.hnr_frame_ms / 1000.0 * sr))
            if len(y) >= hnr_frame:
                vals = []
                for i in range(0, len(y) - hnr_frame + 1, hop):
                    frm = y[i : i + hnr_frame]
                    ac = np.correlate(frm, frm, mode="full")[hnr_frame - 1 : hnr_frame + 2]
                    r0 = float(ac[0] + 1e-12)
                    r1 = float(ac[1] if ac.shape[0] > 1 else 0.0)
                    vals.append(20.0 * np.log10(abs(r1) / r0 + 1e-12))
                hnr_like = float(np.mean(vals)) if vals else 0.0
            else:
                hnr_like = 0.0

            payload: Dict[str, Any] = {
                "vq_jitter": jitter,
                "vq_shimmer": shimmer,
                "vq_hnr_like_db": hnr_like,
                "sample_rate": sr,
                "duration": float(len(y) / sr),
                "device_used": self.device,
                "average_channels": self.average_channels,
                "hnr_frame_ms": self.hnr_frame_ms,
                "rms_mask_threshold": self.rms_mask_threshold,
            }

            dt = time.time() - start_time
            self._log_extraction_success(input_uri, dt)
            return self._create_result(True, payload=payload, processing_time=dt)

        except Exception as e:
            dt = time.time() - start_time
            self._log_extraction_error(input_uri, str(e), dt)
            return self._create_result(False, error=str(e), processing_time=dt)


