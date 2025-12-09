"""
Экстрактор HPSS: разложение на гармоническую и перкуссионную компоненты (librosa.decompose.hpss).
"""
import os
import time
import logging
import uuid
from typing import Dict, Any, Optional

import numpy as np

from src.core.base_extractor import BaseExtractor, ExtractorResult
from src.core.audio_utils import AudioUtils

logger = logging.getLogger(__name__)


class HPSSExtractor(BaseExtractor):
    name = "hpss"
    version = "1.1.0"
    description = "Harmonic-Percussive Source Separation признаки и доли энергии"
    category = "source_separation"
    dependencies = ["librosa", "numpy"]
    estimated_duration = 1.3

    gpu_required = False
    gpu_preferred = False
    gpu_memory_required = 0.0

    def __init__(
        self,
        device: str = "auto",
        sample_rate: int = 22050,
        n_fft: int = 2048,
        hop_length: int = 512,
        return_waveforms: bool = False,
        hpss_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__(device=device)
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.audio_utils = AudioUtils(device=device, sample_rate=sample_rate)
        self.return_waveforms = bool(return_waveforms)
        self.hpss_kwargs = hpss_kwargs or {}

    def run(self, input_uri: str, tmp_path: str, shared_features: Optional[Dict[str, Any]] = None) -> ExtractorResult:
        """
        shared_features (опционально):
            - 'stft_complex': комплексный STFT (np.ndarray)
            - 'stft_magnitude' / 'stft_power': модуль или мощность спектра
        """
        start_time = time.time()
        try:
            if not self._validate_input(input_uri):
                return self._create_result(False, error="Некорректный входной файл", processing_time=time.time() - start_time)

            self._log_extraction_start(input_uri)

            y_t, sr = self.audio_utils.load_audio(input_uri, self.sample_rate)
            y = self.audio_utils.to_numpy(y_t)

            if y.ndim == 2:
                y = y[0]

            y = y.astype(np.float32)
            if y.size == 0:
                raise ValueError("Пустой аудиосигнал")

            # Попробуем использовать shared STFT
            S_mag = None
            S_complex = None
            if shared_features:
                S_complex = shared_features.get("stft_complex")
                if S_complex is not None:
                    S_mag = np.abs(S_complex)
                else:
                    S_mag = shared_features.get("stft_magnitude") or shared_features.get("stft_power")

            import librosa

            if S_mag is None:
                S_complex = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)
                S_mag = np.abs(S_complex)

            # Выполняем HPSS (передаём kwargs при наличии)
            try:
                H, P = librosa.decompose.hpss(S_mag, **self.hpss_kwargs)
            except Exception as e:
                logger.debug(f"HPSS: hpss kwargs failed ({e}), retrying with defaults")
                H, P = librosa.decompose.hpss(S_mag)

            energy_total = float(np.sum(S_mag ** 2) + 1e-12)
            energy_h = float(np.sum(H ** 2))
            energy_p = float(np.sum(P ** 2))
            share_h = float(energy_h / energy_total)
            share_p = float(energy_p / energy_total)

            payload: Dict[str, Any] = {
                "hpss_harmonic_share": share_h,
                "hpss_percussive_share": share_p,
                "hpss_energy_total": energy_total,
                "hpss_energy_harmonic": energy_h,
                "hpss_energy_percussive": energy_p,
                "sample_rate": sr,
                "n_fft": self.n_fft,
                "hop_length": self.hop_length,
                "duration": float(len(y) / sr),
                "device_used": self.device,
                "hpss_frames": int(S_mag.shape[1]) if S_mag is not None else 0,
            }

            # По запросу восстанавливаем временные сигналы и сохраняем в tmp
            if self.return_waveforms:
                if S_complex is None:
                    S_complex = librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)
                phase = np.angle(S_complex)
                H_complex = H * np.exp(1j * phase)
                P_complex = P * np.exp(1j * phase)

                h_wav = librosa.istft(H_complex, hop_length=self.hop_length, length=len(y))
                p_wav = librosa.istft(P_complex, hop_length=self.hop_length, length=len(y))

                os.makedirs(tmp_path, exist_ok=True)
                h_fname = f"{self.name}_harmonic_{uuid.uuid4().hex}.npy"
                p_fname = f"{self.name}_percussive_{uuid.uuid4().hex}.npy"
                h_path = os.path.join(tmp_path, h_fname)
                p_path = os.path.join(tmp_path, p_fname)
                np.save(h_path, h_wav.astype(np.float32))
                np.save(p_path, p_wav.astype(np.float32))

                payload["hpss_harmonic_npy"] = h_path
                payload["hpss_percussive_npy"] = p_path
                payload["hpss_waveform_length"] = int(len(y))

                self.logger.info(
                    f"HPSS: return_waveforms=True, saved harmonic/percussive to npy, frames={payload['hpss_frames']}"
                )

            dt = time.time() - start_time
            self._log_extraction_success(input_uri, dt)
            return self._create_result(True, payload=payload, processing_time=dt)

        except Exception as e:
            dt = time.time() - start_time
            self._log_extraction_error(input_uri, str(e), dt)
            return self._create_result(False, error=str(e), processing_time=dt)
