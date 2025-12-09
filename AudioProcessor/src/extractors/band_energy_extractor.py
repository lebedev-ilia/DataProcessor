"""
Экстрактор энергий по частотным полосам (низ/середина/высокие) и их долей.
"""
import time
import logging
from typing import Dict, Any, List, Tuple

import numpy as np

from src.core.base_extractor import BaseExtractor, ExtractorResult
from src.core.audio_utils import AudioUtils

logger = logging.getLogger(__name__)


class BandEnergyExtractor(BaseExtractor):
    name = "band_energy"
    version = "1.1.0"
    description = "Энергии по полосам (low/mid/high) и доли энергии"
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
        bands: List[Tuple[float, float]] | None = None,
        n_fft: int = 2048,
        hop_length: int = 512,
        use_mel_bands: bool = True,
        n_mels: int = 3,
        return_time_series: bool = True,
        prefer_essentia: bool = True,
    ):
        super().__init__(device=device)
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        # По умолчанию: low [0-200), mid [200-2000), high [2000-nyq)
        self.bands = bands or [(0.0, 200.0), (200.0, 2000.0), (2000.0, sample_rate / 2.0)]
        self.use_mel_bands = use_mel_bands
        self.n_mels = max(3, int(n_mels))
        self.return_time_series = return_time_series
        self.prefer_essentia = prefer_essentia
        self.audio_utils = AudioUtils(device=device, sample_rate=sample_rate)

    def run(self, input_uri: str, tmp_path: str) -> ExtractorResult:
        start_time = time.time()
        try:
            if not self._validate_input(input_uri):
                return self._create_result(False, error="Некорректный входной файл", processing_time=time.time() - start_time)

            self._log_extraction_start(input_uri)

            y_t, sr = self.audio_utils.load_audio(input_uri, self.sample_rate)
            y = self.audio_utils.to_numpy(y_t)
            if y.ndim == 2:
                y = y[0]

            # По умолчанию используем векторизованный путь через librosa/NumPy.
            # При включенном prefer_essentia и наличии модуля — используем Essentia.
            import importlib.util
            essentia_available = importlib.util.find_spec("essentia") is not None
            if self.prefer_essentia and essentia_available:
                try:
                    import essentia  # type: ignore
                    import essentia.standard as es  # type: ignore
                    audio = y.astype(np.float32)
                    frame_cutter = es.FrameCutter(frameSize=self.n_fft, hopSize=self.hop_length, startFromZero=True)
                    window = es.Windowing(type='hann')
                    spectrum = es.Spectrum()

                    num_bins = int(self.n_fft // 2 + 1)
                    freqs = np.linspace(0.0, sr / 2.0, num=num_bins, dtype=np.float32)
                    # Адаптивные полосы при необходимости
                    bands_to_use = self.bands
                    if self.use_mel_bands:
                        import librosa
                        mel_edges = librosa.mel_frequencies(n_mels=self.n_mels, fmin=0.0, fmax=sr / 2.0)
                        bands_to_use = list(zip(mel_edges[:-1], mel_edges[1:]))
                    band_masks = [(freqs >= lo) & (freqs < hi) for (lo, hi) in bands_to_use]

                    total_energy = 0.0
                    accum_energies = [0.0 for _ in band_masks]
                    per_frame: List[List[float]] = []

                    while True:
                        frame = frame_cutter(audio)
                        if frame.size == 0:
                            break
                        win = window(frame)
                        spec = spectrum(win)  # magnitude
                        pwr = np.asarray(spec, dtype=np.float32) ** 2
                        total_energy += float(np.sum(pwr))
                        frame_energies = []
                        for i, mask in enumerate(band_masks):
                            e = float(np.sum(pwr[mask]))
                            accum_energies[i] += e
                            frame_energies.append(e)
                        if self.return_time_series:
                            per_frame.append(frame_energies)

                    if total_energy == 0.0:
                        total_energy = float(np.sum(audio.astype(np.float32) ** 2) + 1e-12)

                    energies = accum_energies
                    band_means = np.asarray(energies, dtype=np.float32) / max(1, len(per_frame)) if self.return_time_series else np.asarray(energies, dtype=np.float32)
                    total_mean = float(np.sum(band_means) + 1e-12)
                    shares = (band_means / total_mean).astype(np.float32).tolist()

                    payload: Dict[str, Any] = {
                        "band_edges": [(float(lo), float(hi)) for lo, hi in bands_to_use],
                        "band_energies": [float(e) for e in energies],
                        "band_energy_shares": [float(s) for s in shares],
                        "total_energy": float(total_energy),
                        "sample_rate": sr,
                        "n_fft": self.n_fft,
                        "hop_length": self.hop_length,
                        "duration": float(len(y) / sr),
                        "device_used": self.device,
                    }
                    if self.return_time_series and per_frame:
                        payload["band_energy_ts"] = per_frame  # shape ~ [frames, num_bands]

                    dt = time.time() - start_time
                    self._log_extraction_success(input_uri, dt)
                    return self._create_result(True, payload=payload, processing_time=dt)
                except Exception as e:
                    pass
                    # self.logger.info(
                    #    f"BandEnergy: fallback на librosa (причина: {e})"
                    #)

            import librosa

            # STFT -> power spectrogram (freq_bins, frames), float32
            S = np.abs(librosa.stft(y, n_fft=self.n_fft, hop_length=self.hop_length)).astype(np.float32) ** 2
            freqs = librosa.fft_frequencies(sr=sr, n_fft=self.n_fft).astype(np.float32)

            # Полосы: фиксированные или мел-шкала
            bands_to_use = self.bands
            if self.use_mel_bands:
                mel_edges = librosa.mel_frequencies(n_mels=self.n_mels, fmin=0.0, fmax=sr / 2.0)
                bands_to_use = list(zip(mel_edges[:-1], mel_edges[1:]))
            # self.logger.info(
            #    f"BandEnergy: используем librosa (use_mel_bands={self.use_mel_bands}, n_mels={self.n_mels}, frames={S.shape[1]})"
            #)

            # Векторизованный биннинг: матрица масок (freq_bins, num_bands)
            masks = []
            for lo, hi in bands_to_use:
                masks.append(((freqs >= float(lo)) & (freqs < float(hi))).astype(np.float32))
            mask_matrix = np.stack(masks, axis=1)  # (freq_bins, num_bands)

            # Пер-кадровые энергии по полосам: (num_bands, frames)
            band_energy_ts = mask_matrix.T @ S  # матричное умножение

            # Агрегаты по кадрам
            band_means = band_energy_ts.mean(axis=1)  # (num_bands,)
            band_stds = band_energy_ts.std(axis=1)
            band_medians = np.median(band_energy_ts, axis=1)

            total_mean = float(np.sum(band_means) + 1e-12)
            shares = (band_means / total_mean).astype(np.float32)

            # Скалярные суммы энергий (по всем кадрам)
            energies = band_energy_ts.sum(axis=1).astype(np.float32)
            total_energy = float(np.sum(energies) + 1e-12)

            payload: Dict[str, Any] = {
                "band_edges": [(float(lo), float(hi)) for lo, hi in bands_to_use],
                "band_energies": [float(e) for e in energies.tolist()],
                "band_energy_mean": [float(m) for m in band_means.tolist()],
                "band_energy_std": [float(s) for s in band_stds.tolist()],
                "band_energy_median": [float(med) for med in band_medians.tolist()],
                "band_energy_shares": [float(s) for s in shares.tolist()],
                "total_energy": total_energy,
                "sample_rate": sr,
                "n_fft": self.n_fft,
                "hop_length": self.hop_length,
                "duration": float(len(y) / sr),
                "device_used": self.device,
            }
            if self.return_time_series:
                # Возвращаем per-frame кривые энергий по полосам (список списков)
                payload["band_energy_ts"] = band_energy_ts.astype(np.float32).tolist()

            dt = time.time() - start_time
            self._log_extraction_success(input_uri, dt)
            return self._create_result(True, payload=payload, processing_time=dt)

        except Exception as e:
            dt = time.time() - start_time
            self._log_extraction_error(input_uri, str(e), dt)
            return self._create_result(False, error=str(e), processing_time=dt)
