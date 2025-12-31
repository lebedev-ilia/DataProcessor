"""
Экстрактор тональности (ключ + мажор/минор) на основе хрома и корреляции с шаблонами Krumhansl.
"""
import time
import logging
import importlib
from typing import Dict, Any, Tuple, Optional, List

import numpy as np

from src.core.base_extractor import BaseExtractor, ExtractorResult
from src.core.audio_utils import AudioUtils

logger = logging.getLogger(__name__)


class KeyExtractor(BaseExtractor):
    name = "key"
    version = "1.1.0"
    description = "Определение ключа (тональности) через шаблоны Krumhansl на хроме"
    category = "music_theory"
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
        chroma_type: str = "cqt",  # "cqt" | "stft"
        use_beat_sync: bool = False,
        top_k: int = 3,
    ):
        super().__init__(device=device)
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        assert chroma_type in ("cqt", "stft")
        self.chroma_type = chroma_type
        self.use_beat_sync = bool(use_beat_sync)
        self.top_k = int(top_k)
        self.audio_utils = AudioUtils(device=device, sample_rate=sample_rate)

    def run(self, input_uri: str, tmp_path: str, shared_features: Optional[Dict[str, Any]] = None) -> ExtractorResult:
        start_time = time.time()
        try:
            if not self._validate_input(input_uri):
                return self._create_result(False, error="Некорректный входной файл", processing_time=time.time() - start_time)

            self._log_extraction_start(input_uri)

            y_t, sr = self.audio_utils.load_audio(input_uri, self.sample_rate)
            y = self.audio_utils.to_numpy(y_t)
            if y.ndim == 2:
                # mix to mono (параметризовать можно при надобности)
                y = np.mean(y, axis=0)

            y = y.astype(np.float32)
            if y.size == 0:
                raise ValueError("Пустой аудиосигнал")

            # Попробуем Essentia KeyExtractor, если он доступен и корректно импортируется
            use_essentia = importlib.util.find_spec("essentia") is not None
            if use_essentia:
                try:
                    import essentia.standard as es  # type: ignore
                    ek = es.KeyExtractor()
                    k, scale, strength = ek(y)
                    key_name = str(k)
                    key_mode = str(scale).lower()
                    confidence = float(strength)
                    payload = {
                        "key_name": key_name,
                        "key_mode": key_mode,
                        "key_confidence": confidence,
                        "sample_rate": sr,
                        "hop_length": self.hop_length,
                        "duration": float(len(y) / sr),
                        "device_used": self.device,
                    }
                    dt = time.time() - start_time
                    self._log_extraction_success(input_uri, dt)
                    return self._create_result(True, payload=payload, processing_time=dt)
                except Exception as e:
                    logger.info(f"KeyExtractor: Essentia недоступна/вызов завершился ошибкой, fallback на librosa (reason: {e})")
                    use_essentia = False  # fallback

            # --- Librosa-based fallback (Krumhansl) ---
            import librosa

            # Try to reuse provided chroma if present in shared_features
            chroma = None
            if shared_features:
                chroma = shared_features.get("chroma")  # expected shape (12, frames)
                # allow also stft/magnitude -> compute chroma later
                stft_complex = shared_features.get("stft_complex")

            # If no shared chroma — compute it here
            tuning = 0.0
            try:
                tuning = librosa.estimate_tuning(y=y, sr=sr)
            except Exception:
                tuning = 0.0

            if chroma is None:
                if self.chroma_type == "cqt":
                    try:
                        chroma = librosa.feature.chroma_cqt(y=y, sr=sr, hop_length=self.hop_length, tuning=tuning)
                    except Exception:
                        chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=self.hop_length)
                else:
                    chroma = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=self.hop_length)

            # If beat synchronous requested — aggregate chroma per beat
            if self.use_beat_sync:
                try:
                    tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=self.hop_length)
                    if len(beats) > 0:
                        chroma_sync = librosa.util.sync(chroma, beats, aggregate=np.mean)
                        profile = np.mean(chroma_sync, axis=1)
                    else:
                        profile = np.mean(chroma, axis=1)
                except Exception:
                    profile = np.mean(chroma, axis=1)
            else:
                profile = np.mean(chroma, axis=1)

            # Normalize PCP/profile
            v = profile.astype(np.float32)
            s = float(v.sum() + 1e-12)
            v = v / s

            # estimate key distribution and pick best
            scores = self._score_key_profiles(v)  # returns 24-vector (C major, C minor, C# major, ...)
            # Build mapping keys
            keys = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']
            # scores format: [maj_C, min_C, maj_C#, min_C#, ..., maj_B, min_B]
            pairs = []
            for i in range(12):
                maj = scores[2 * i]
                min_ = scores[2 * i + 1]
                pairs.append((keys[i], float(maj), float(min_)))

            # choose best among all majors/minors
            flat = np.array(scores, dtype=np.float32)
            best_idx = int(np.argmax(flat))
            best_root = keys[(best_idx // 2) % 12]
            best_mode = "major" if (best_idx % 2) == 0 else "minor"
            best_score = float(flat[best_idx])

            # prepare top_k
            order = np.argsort(flat)[::-1]
            top_k = []
            for idx in order[: self.top_k]:
                k_name = keys[(idx // 2) % 12]
                k_mode = "major" if (idx % 2) == 0 else "minor"
                top_k.append({"key": k_name, "mode": k_mode, "score": float(flat[idx])})

            payload: Dict[str, Any] = {
                "key_name": best_root,
                "key_mode": best_mode,
                "key_confidence": best_score,
                "key_scores": [float(x) for x in flat.tolist()],
                "key_top_k": top_k,
                "sample_rate": sr,
                "hop_length": self.hop_length,
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

    def _score_key_profiles(self, pcp: np.ndarray) -> List[float]:
        """
        Return 24 scores: [maj_C, min_C, maj_C#, min_C#, ..., maj_B, min_B]
        Uses Krumhansl-Schmuckler profiles and Pearson-like correlation (zero-mean cosine).
        """
        major = np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09, 2.52, 5.19, 2.39, 3.66, 2.29, 2.88], dtype=np.float32)
        minor = np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53, 2.54, 4.75, 3.98, 2.69, 3.34, 3.17], dtype=np.float32)
        # normalize profiles to sum=1 (optional, but keeps scale consistent)
        major = major / major.sum()
        minor = minor / minor.sum()

        v = pcp.astype(np.float32)
        # center (zero-mean) — needed for Pearson-like correlation
        v0 = v - v.mean()

        scores = []
        for shift in range(12):
            maj_rot = np.roll(major, shift)
            min_rot = np.roll(minor, shift)
            # zero-mean rotate
            maj0 = maj_rot - maj_rot.mean()
            min0 = min_rot - min_rot.mean()
            # compute cosine between zero-mean vectors (Pearson correlation proportional)
            denom_maj = (np.linalg.norm(v0) * np.linalg.norm(maj0) + 1e-12)
            denom_min = (np.linalg.norm(v0) * np.linalg.norm(min0) + 1e-12)
            score_maj = float(np.dot(v0, maj0) / denom_maj)
            score_min = float(np.dot(v0, min0) / denom_min)
            scores.append(score_maj)
            scores.append(score_min)

        # Optionally rescale to [0,1] for easier interpretation
        arr = np.array(scores, dtype=np.float32)
        # shift to positive range
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-12)
        return arr.tolist()