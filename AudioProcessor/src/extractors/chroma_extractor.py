"""
Экстрактор хрома-фич (pitch class profile) на базе librosa.
"""
import time
import logging
import os
import importlib
import uuid

import numpy as np

from src.core.base_extractor import BaseExtractor, ExtractorResult
from src.core.audio_utils import AudioUtils

logger = logging.getLogger(__name__)


class ChromaExtractor(BaseExtractor):
    """Извлечение хрома-признаков с настройкой строя, нормализацией и агрегатами."""

    name = "chroma"
    version = "1.1.0"
    description = "Хрома (12-полосный профиль классов высот) с тюнингом и агрегатами"
    category = "spectral"
    dependencies = ["librosa", "numpy"]
    estimated_duration = 1.2

    gpu_required = False
    gpu_preferred = False
    gpu_memory_required = 0.0

    def __init__(
        self,
        device: str = "auto",
        sample_rate: int = 22050,
        hop_length: int = 512,
        n_fft: int = 4096,
        mix_to_mono: bool = True,
        chroma_type: str = "cqt",
        normalize: str | None = "l1",
        return_time_series: bool = False,
        max_frames: int = 300,
    ):
        super().__init__(device=device)
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.n_fft = n_fft
        self.audio_utils = AudioUtils(device=device, sample_rate=sample_rate)
        self.mix_to_mono = mix_to_mono
        assert chroma_type in ("cqt", "stft")
        self.chroma_type = chroma_type
        assert normalize in (None, "l1", "l2")
        self.normalize = normalize
        self.return_time_series = return_time_series
        self.max_frames = int(max_frames)

    def run(self, input_uri: str, tmp_path: str) -> ExtractorResult:
        start_time = time.time()
        try:
            if not self._validate_input(input_uri):
                return self._create_result(False, error="Некорректный входной файл", processing_time=time.time() - start_time)

            self._log_extraction_start(input_uri)

            y_t, sr = self.audio_utils.load_audio(input_uri, self.sample_rate)
            y = self.audio_utils.to_numpy(y_t)

            # Сведение в моно (опционально)
            if y.ndim == 2:
                if self.mix_to_mono:
                    y = np.mean(y, axis=0)
                else:
                    y = y[0]

            y = y.astype(np.float32)
            if y.size == 0:
                raise ValueError("Пустой аудиосигнал")

            # Essentia доступность (оставляем как опцию, но по умолчанию используем librosa)
            use_essentia = False
            if importlib.util.find_spec("essentia") is not None:
                try:
                    import essentia.standard as es  # noqa: F401
                    use_essentia = False
                except Exception:
                    use_essentia = False

            # Основной путь: librosa
            import librosa

            # Оценка строя
            try:
                tuning = float(librosa.estimate_tuning(y=y, sr=sr))
            except Exception:
                tuning = 0.0
            # self.logger.info(
            #    f"Chroma: librosa path, type={self.chroma_type}, tuning={tuning:.3f}, normalize={self.normalize}, mix_to_mono={self.mix_to_mono}"
            #)

            if self.chroma_type == "cqt":
                try:
                    chroma = librosa.feature.chroma_cqt(
                        y=y, sr=sr, hop_length=self.hop_length, n_chroma=12, tuning=tuning
                    )
                except Exception:
                    chroma = librosa.feature.chroma_stft(
                        y=y, sr=sr, hop_length=self.hop_length, n_fft=self.n_fft
                    )
            else:
                chroma = librosa.feature.chroma_stft(
                    y=y, sr=sr, hop_length=self.hop_length, n_fft=self.n_fft
                )

            chroma = chroma.astype(np.float32)
            if chroma.ndim != 2 or chroma.shape[0] != 12:
                chroma = np.zeros((12, max(1, chroma.shape[-1] if chroma.ndim > 1 else 1)), dtype=np.float32)

            # Нормализация по кадру
            if self.normalize == "l1":
                frame_sums = chroma.sum(axis=0, keepdims=True) + 1e-12
                chroma = chroma / frame_sums
            elif self.normalize == "l2":
                norms = np.linalg.norm(chroma, ord=2, axis=0, keepdims=True) + 1e-12
                chroma = chroma / norms

            # Агрегаты
            mean = chroma.mean(axis=1)
            std = chroma.std(axis=1)
            min_vals = chroma.min(axis=1)
            max_vals = chroma.max(axis=1)
            median = np.median(chroma, axis=1)
            p25 = np.percentile(chroma, 25, axis=1)
            p75 = np.percentile(chroma, 75, axis=1)

            stat_features = np.concatenate([mean, std, min_vals, max_vals, median, p25, p75]).astype(float)

            payload: Dict[str, Any] = {
                "chroma_mean": mean.tolist(),
                "chroma_std": std.tolist(),
                "chroma_min": min_vals.tolist(),
                "chroma_max": max_vals.tolist(),
                "chroma_median": median.tolist(),
                "chroma_p25": p25.tolist(),
                "chroma_p75": p75.tolist(),
                "chroma_stats_vector": stat_features.tolist(),
                "sample_rate": sr,
                "hop_length": self.hop_length,
                "n_fft": self.n_fft,
                "duration": float(y.shape[-1] / sr),
                "device_used": self.device,
                "chroma_frames": int(chroma.shape[1]),
            }

            # Опциональный time-series с даунсемплингом или сохранением
            if self.return_time_series:
                max_frames = int(self.max_frames)
                if chroma.shape[1] > max_frames:
                    idx = np.linspace(0, chroma.shape[1] - 1, num=max_frames, dtype=int)
                    chroma_ds = chroma[:, idx]
                else:
                    chroma_ds = chroma

                if chroma_ds.size <= 12 * 500:
                    payload["chroma"] = chroma_ds.tolist()
                else:
                    os.makedirs(tmp_path, exist_ok=True)
                    fname = f"{self.name}_{uuid.uuid4().hex}.npy"
                    fpath = os.path.join(tmp_path, fname)
                    np.save(fpath, chroma_ds)
                    payload["chroma_npy"] = fpath
                # self.logger.info(
                #     f"Chroma: return_time_series=True, frames_out={chroma_ds.shape[1]}, saved={'npy' if 'chroma_npy' in payload else 'inline'}"
                # )

            dt = time.time() - start_time
            self._log_extraction_success(input_uri, dt)
            return self._create_result(True, payload=payload, processing_time=dt)

        except Exception as e:
            dt = time.time() - start_time
            self._log_extraction_error(input_uri, str(e), dt)
            return self._create_result(False, error=str(e), processing_time=dt)


