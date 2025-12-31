"""
Экстрактор базовых метрик качества аудио (лёгкий, без тяжёлых зависимостей).

Метрики:
- dc_offset (среднее смещение)
- clipping_ratio (доля отсечённых сэмплов)
- crest_factor (отношение peak/RMS в dB)
- dynamic_range_db (разница между 95 и 5 перцентилями уровня в дБ)
- snr_db (грубая оценка через медиану/перцентиль шума)
"""
import time
import logging
from typing import Dict, Any

import numpy as np

from src.core.base_extractor import BaseExtractor, ExtractorResult
from src.core.audio_utils import AudioUtils

logger = logging.getLogger(__name__)


class QualityExtractor(BaseExtractor):
    name = "quality"
    version = "1.1.0"
    description = "Базовые метрики качества аудио (DC offset, clipping, crest, DR, SNR)"
    category = "quality"
    dependencies = ["numpy"]
    estimated_duration = 0.5

    gpu_required = False
    gpu_preferred = False
    gpu_memory_required = 0.0

    def __init__(
        self,
        device: str = "auto",
        sample_rate: int = 22050,
        average_channels: bool = True,
        frame_len_ms: float = 50.0,
        hop_ms: float = 25.0,
        clip_threshold: float = 0.999,
    ):
        super().__init__(device=device)
        self.sample_rate = sample_rate
        self.audio_utils = AudioUtils(device=device, sample_rate=sample_rate)
        self.average_channels = bool(average_channels)
        self.frame_len_ms = float(frame_len_ms)
        self.hop_ms = float(hop_ms)
        self.clip_threshold = float(clip_threshold)

    def run(self, input_uri: str, tmp_path: str) -> ExtractorResult:
        start_time = time.time()
        try:
            if not self._validate_input(input_uri):
                return self._create_result(False, error="Некорректный входной файл", processing_time=time.time() - start_time)

            self._log_extraction_start(input_uri)

            wav_t, sr = self.audio_utils.load_audio(input_uri, self.sample_rate)
            x = self.audio_utils.to_numpy(wav_t)
            if x.ndim == 2:
                # [C, T] -> моно по настройке
                if self.average_channels:
                    x = np.mean(x, axis=0)
                else:
                    x = x[0]

            x = x.astype(np.float32)
            eps = 1e-12

            # DC offset
            dc_offset = float(np.mean(x))

            # Clipping ratio (настраиваемый порог, по умолчанию 0.999)
            thr = self.clip_threshold
            clipping_ratio = float(np.mean(np.abs(x) >= thr))

            # RMS/peak
            rms = float(np.sqrt(np.mean(x**2) + eps))
            peak = float(np.max(np.abs(x) + eps))
            crest_factor_db = float(20.0 * np.log10((peak + eps) / (rms + eps)))

            # Динамический диапазон (через перцентили уровней кадров)
            frame_len = max(1, int((self.frame_len_ms / 1000.0) * sr))
            hop = max(1, int((self.hop_ms / 1000.0) * sr))

            # Векторизованное формирование кадров через stride_tricks
            levels: list = []
            if len(x) >= frame_len:
                n_frames = 1 + (len(x) - frame_len) // hop
                if n_frames > 0:
                    try:
                        import numpy as _np
                        shape = (n_frames, frame_len)
                        strides = (x.strides[-1] * hop, x.strides[-1])
                        frames = _np.lib.stride_tricks.as_strided(x, shape=shape, strides=strides)
                        # RMS по кадрам
                        rms_frames = _np.sqrt(_np.mean(frames * frames, axis=1) + eps)
                        # Уровни в дБ
                        levels = (20.0 * _np.log10(rms_frames + eps)).tolist()
                    except Exception:
                        # Fallback к циклу при отсутствии поддержки
                        levels = []
                        for i in range(0, len(x) - frame_len + 1, hop):
                            frm = x[i : i + frame_len]
                            lvl = 20.0 * np.log10(np.sqrt(np.mean(frm * frm) + eps))
                            levels.append(lvl)
            if levels:
                p5 = float(np.percentile(levels, 5))
                p95 = float(np.percentile(levels, 95))
                dynamic_range_db = float(p95 - p5)
            else:
                dynamic_range_db = 0.0

            # Грубая SNR: сигнал = 95 перцентиль уровня, шум = 5 перцентиль
            if levels:
                noise_db = float(np.percentile(levels, 5))
                signal_db = float(np.percentile(levels, 95))
                snr_db = float(max(0.0, signal_db - noise_db))
            else:
                snr_db = 0.0

            payload: Dict[str, Any] = {
                "dc_offset": dc_offset,
                "clipping_ratio": clipping_ratio,
                "crest_factor_db": crest_factor_db,
                "dynamic_range_db": dynamic_range_db,
                "snr_db": snr_db,
                "sample_rate": sr,
                "duration": float(len(x) / sr),
                "device_used": self.device,
            }

            dt = time.time() - start_time
            self._log_extraction_success(input_uri, dt)
            return self._create_result(True, payload=payload, processing_time=dt)

        except Exception as e:
            dt = time.time() - start_time
            self._log_extraction_error(input_uri, str(e), dt)
            return self._create_result(False, error=str(e), processing_time=dt)


