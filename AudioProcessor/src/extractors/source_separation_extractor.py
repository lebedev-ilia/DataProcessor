"""
Экстрактор source separation на базе Open-Unmix (UMX, PyTorch).

Возвращает доли энергии по источникам (vocals, drums, bass, other) и суммарные метрики.
"""
import time
import logging
from typing import Dict, Any

import numpy as np

from src.core.base_extractor import BaseExtractor, ExtractorResult
from src.core.audio_utils import AudioUtils
from src.utils.prof import timeit

logger = logging.getLogger(__name__)


class SourceSeparationExtractor(BaseExtractor):
    name = "source_separation"
    version = "1.1.0"
    description = "Разделение на источники (vocals, drums, bass, other) с помощью Open-Unmix"
    category = "source_separation"
    dependencies = ["openunmix", "torch", "numpy"]
    estimated_duration = 10.0

    gpu_required = False
    gpu_preferred = True
    gpu_memory_required = 2.0

    def __init__(
        self,
        device: str = "auto",
        sample_rate: int = 44100,
        average_channels: bool = True,
        allow_gpu_separation: bool = False,
        return_per_source_rms: bool = True,
    ):
        super().__init__(device=device)
        # Open-Unmix обучен на 44.1kHz, стерео
        self.sample_rate = sample_rate
        self.audio_utils = AudioUtils(device=device, sample_rate=sample_rate)
        self.average_channels = bool(average_channels)
        self.allow_gpu_separation = bool(allow_gpu_separation)
        self.return_per_source_rms = bool(return_per_source_rms)

        # Ленивая инициализация модели
        self._oum_predict = None

    def _load_openunmix(self):
        """Безопасная загрузка Open-Unmix predict API (один раз)."""
        if self._oum_predict is not None:
            return self._oum_predict
        try:
            import openunmix  # type: ignore
            try:
                from openunmix import predict as oum_predict  # type: ignore
            except Exception:
                oum_predict = None
        except Exception as e:
            # self.logger.info(
            #     f"SourceSeparation: Open-Unmix недоступен (причина: {e})"
            # )
            oum_predict = None
        self._oum_predict = oum_predict
        return self._oum_predict

    def run(self, input_uri: str, tmp_path: str) -> ExtractorResult:
        start_time = time.time()
        try:
            if not self._validate_input(input_uri):
                return self._create_result(False, error="Некорректный входной файл", processing_time=time.time() - start_time)

            self._log_extraction_start(input_uri)

            # Загружаем и ресемплим в 44.1kHz
            with timeit("sep: load_audio"):
                wav_t, sr = self.audio_utils.load_audio(input_uri, self.sample_rate)
            x = self.audio_utils.to_numpy(wav_t)
            # Приведём к стерео (C, T) с C=2, при необходимости усредняя каналы
            if x.ndim == 1:
                if self.average_channels:
                    # моно → стерео копированием, энергии остаются корректными
                    x = np.stack([x, x], axis=0)
                else:
                    x = np.stack([x, x], axis=0)
            elif x.shape[0] == 1:
                if self.average_channels:
                    # уже моно [1, T] → стерео копированием
                    x = np.concatenate([x, x], axis=0)
                else:
                    x = np.concatenate([x, x], axis=0)
            else:
                # Многоканал → стерео
                if self.average_channels:
                    mono = np.mean(x, axis=0)
                    x = np.stack([mono, mono], axis=0)
                else:
                    # Берём первые два канала
                    x = x[:2, :]

            import torch
            with timeit("sep: load_openunmix"):
                oum_predict = self._load_openunmix()

            estimates = None
            # По умолчанию CPU безопаснее (STFT/window device mismatch). Можно включить GPU через флаг.
            sep_device = "cuda" if (self.allow_gpu_separation and self.device == "cuda") else "cpu"
            if oum_predict is not None and hasattr(oum_predict, "separate"):
                # Тензор (batch=1, channels=2, samples)
                audio_t = torch.from_numpy(x).float().unsqueeze(0)
                if sep_device == "cuda" and torch.cuda.is_available():
                    audio_t = audio_t.cuda()
                with torch.no_grad():
                    with timeit("sep: openunmix.separate"):
                        estimates = oum_predict.separate(audio_t, rate=self.sample_rate)
                # self.logger.info(
                #     f"SourceSeparation: run with sep_device={sep_device}, input_shape={list(audio_t.shape)}"
                # )
            else:
                pass
                # self.logger.info(
                #     "SourceSeparation: API openunmix.predict.separate недоступен, используем нулевые доли"
                # )

            # Перевод в numpy, расчёт энергий
            def energy_of(arr: np.ndarray) -> float:
                return float(np.sum(arr.astype(np.float32) ** 2))

            with timeit("sep: total energy"):
                total_energy = energy_of(x)
            shares: Dict[str, float] = {}
            per_source_rms: Dict[str, float] = {}
            for k in ["vocals", "drums", "bass", "other"]:
                if isinstance(estimates, dict) and k in estimates:
                    est = estimates[k].squeeze(0).detach().cpu().numpy()
                    with timeit(f"sep: energy {k}"):
                        shares[k] = float(energy_of(est) / (total_energy + 1e-12))
                    if self.return_per_source_rms:
                        with timeit(f"sep: rms {k}"):
                            rms = float(np.sqrt(np.mean(est.astype(np.float32) ** 2) + 1e-12))
                        per_source_rms[k] = rms
                else:
                    shares[k] = 0.0
                    if self.return_per_source_rms:
                        per_source_rms[k] = 0.0

            duration = float(x.shape[-1] / self.sample_rate)

            payload: Dict[str, Any] = {
                "energy_total": total_energy,
                "share_vocals": shares.get("vocals", 0.0),
                "share_drums": shares.get("drums", 0.0),
                "share_bass": shares.get("bass", 0.0),
                "share_other": shares.get("other", 0.0),
                # По желанию — RMS по источникам
                **({
                    "rms_vocals": per_source_rms.get("vocals", 0.0),
                    "rms_drums": per_source_rms.get("drums", 0.0),
                    "rms_bass": per_source_rms.get("bass", 0.0),
                    "rms_other": per_source_rms.get("other", 0.0),
                } if self.return_per_source_rms else {}),
                "sample_rate": self.sample_rate,
                "duration": duration,
                # Фактическое устройство выполнения для Open-Unmix
                "device_used": sep_device,
            }

            dt = time.time() - start_time
            self._log_extraction_success(input_uri, dt)
            return self._create_result(True, payload=payload, processing_time=dt)

        except Exception as e:
            dt = time.time() - start_time
            self._log_extraction_error(input_uri, str(e), dt)
            return self._create_result(False, error=str(e), processing_time=dt)


