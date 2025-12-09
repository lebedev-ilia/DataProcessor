"""
PitchExtractor: извлечение основной частоты (f0) с использованием PYIN/YIN (и CREPE при наличии).
Интеграция с общим интерфейсом BaseExtractor и AudioUtils.
"""
import time
import logging
from typing import Dict, Any, Optional

import numpy as np
import librosa

from src.core.base_extractor import BaseExtractor, ExtractorResult
from src.core.audio_utils import AudioUtils


logger = logging.getLogger(__name__)


class PitchExtractor(BaseExtractor):
    name: str = "pitch"
    version: str = "1.1.0"
    description: str = "Оценка основной частоты (f0) с помощью PYIN/YIN/CREPE (опционально)"
    category: str = "spectral"
    dependencies = ["librosa", "numpy"]

    def __init__(
        self,
        device: Optional[str] = None,
        sample_rate: int = 22050,
        fmin: float = 50.0,
        fmax: float = 2000.0,
        hop_length: int = 512,
        frame_length: int = 2048,
        backend: str = "classic",  # classic | torchcrepe
        channel_mode: str = "first",  # first | mean | max
        torchcrepe_batch_size: int = 1,
    ) -> None:
        super().__init__(device)
        self.sample_rate = sample_rate
        self.fmin = fmin
        self.fmax = fmax
        self.hop_length = hop_length
        self.frame_length = frame_length
        self.audio_utils = AudioUtils(device=device, sample_rate=sample_rate)
        self.backend = backend
        self.channel_mode = channel_mode
        self.torchcrepe_batch_size = int(torchcrepe_batch_size)

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

            # Загружаем аудио через общую утилиту (с ресемплингом до sample_rate)
            waveform_t, sr = self.audio_utils.load_audio(input_uri, target_sr=self.sample_rate)
            waveform_t = self.audio_utils.normalize_audio(waveform_t)

            # Преобразуем к моно с учетом многоканального входа
            # Форма waveform_t ожидается [C, T]
            if waveform_t.dim() == 2 and waveform_t.shape[0] > 1:
                if self.channel_mode == "mean":
                    waveform_t = waveform_t.mean(dim=0, keepdim=True)
                elif self.channel_mode == "max":
                    waveform_t, _ = waveform_t.max(dim=0, keepdim=True)
                else:
                    waveform_t = waveform_t[:1, :]

            # В librosa ожидается ndarray (моно)
            audio_np = waveform_t.squeeze(0).cpu().numpy()

            # Извлекаем признаки
            features = self._extract_pitch_features(audio_np, sr)
            # Если есть серия из torchcrepe — сохраняем в .npy и не включаем сам массив в JSON
            try:
                from pathlib import Path
                import numpy as np
                series = features.get("f0_series_torchcrepe")
                if isinstance(series, list) and len(series) > 0:
                    out_dir = Path(tmp_path)
                    out_dir.mkdir(parents=True, exist_ok=True)
                    stem = Path(input_uri).stem
                    npy_path = out_dir / f"{stem}_f0_torchcrepe.npy"
                    np.save(npy_path, np.asarray(series, dtype=np.float32))
                    features["f0_series_torchcrepe_npy"] = str(npy_path)
                    features["f0_count_torchcrepe"] = int(len(series))
                    # Убираем саму серию из JSON
                    features.pop("f0_series_torchcrepe", None)
            except Exception:
                pass

            processing_time = time.time() - start_time
            self._log_extraction_success(input_uri, processing_time)
            return self._create_result(success=True, payload=features, processing_time=processing_time)

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Ошибка извлечения pitch: {e}"
            self._log_extraction_error(input_uri, error_msg, processing_time)
            return self._create_result(success=False, error=error_msg, processing_time=processing_time)

    def _extract_pitch_features(self, audio: np.ndarray, sr: int) -> Dict[str, Any]:
        features: Dict[str, Any] = {
            "device_used": self.device,
            "sample_rate": sr,
        }

        # Если выбран backend torchcrepe — используем его (при наличии) и возвращаем результаты
        if self.backend == "torchcrepe":
            try:
                f0_tc = self._extract_torchcrepe(audio, sr)
                if f0_tc is not None and f0_tc.size > 0:
                    feats = self._calc_stats(f0_tc, prefix="torchcrepe")
                    features.update(feats)
                    # Итог на базе torchcrepe
                    features.update({
                        "f0_mean": feats.get("f0_mean_torchcrepe", 0.0),
                        "f0_std": feats.get("f0_std_torchcrepe", 0.0),
                        "f0_min": feats.get("f0_min_torchcrepe", 0.0),
                        "f0_max": feats.get("f0_max_torchcrepe", 0.0),
                        "f0_median": feats.get("f0_median_torchcrepe", 0.0),
                        "f0_method": "torchcrepe",
                    })
                    return features
                else:
                    pass
                    # self.logger.info("Pitch: torchcrepe produced empty output; falling back to classic")
            except Exception as e:
                self.logger.warning(f"Pitch: torchcrepe failed ({e}); falling back to classic")

        # PYIN (наиболее устойчивый из классики, требует моно)
        # self.logger.info(f"Pitch: Starting PYIN extraction with fmin={self.fmin}, fmax={self.fmax}, audio_shape={audio.shape}, sr={sr}")
        try:
            f0_pyin, voiced_flag, voiced_probs = librosa.pyin(
                audio,
                fmin=self.fmin,
                fmax=self.fmax,
                sr=sr,
                hop_length=self.hop_length,
                frame_length=self.frame_length,
            )
            # self.logger.info(f"Pitch: PYIN raw output - f0_shape={f0_pyin.shape if f0_pyin is not None else None}, voiced_flag_shape={voiced_flag.shape if voiced_flag is not None else None}")
            
            f0_pyin_clean = f0_pyin[~np.isnan(f0_pyin)] if f0_pyin is not None else np.array([])
            voiced_flag_clean = voiced_flag[~np.isnan(voiced_flag)] if voiced_flag is not None else np.array([])
            
            # self.logger.info(f"Pitch: PYIN after NaN removal - f0_clean_size={f0_pyin_clean.size}, voiced_clean_size={voiced_flag_clean.size}")
            
            if f0_pyin_clean.size > 0:
                features.update(self._calc_stats(f0_pyin_clean, prefix="pyin"))
                features["voiced_fraction_pyin"] = float(np.mean(voiced_flag_clean)) if voiced_flag_clean.size > 0 else 0.0
                features["voiced_probability_mean_pyin"] = float(np.nanmean(voiced_probs)) if voiced_probs is not None else 0.0
                # self.logger.info(f"Pitch: PYIN success - f0_mean={features.get('f0_mean_pyin', 0):.2f}, voiced_fraction={features.get('voiced_fraction_pyin', 0):.3f}")
            else:
                self.logger.warning("Pitch: PYIN produced empty output after NaN removal - no pitch detected")
                features.update(self._zero_stats(prefix="pyin"))
                features["voiced_fraction_pyin"] = 0.0
                features["voiced_probability_mean_pyin"] = 0.0
        except Exception as e:
            self.logger.error(f"Pitch: PYIN failed with exception: {e}")
            features.update(self._zero_stats(prefix="pyin"))
            features["voiced_fraction_pyin"] = 0.0
            features["voiced_probability_mean_pyin"] = 0.0

        # YIN
        # self.logger.info(f"Pitch: Starting YIN extraction with fmin={self.fmin}, fmax={self.fmax}")
        try:
            f0_yin = librosa.yin(
                audio,
                fmin=self.fmin,
                fmax=self.fmax,
                sr=sr,
                hop_length=self.hop_length,
                frame_length=self.frame_length,
            )
            # self.logger.info(f"Pitch: YIN raw output - f0_shape={f0_yin.shape if f0_yin is not None else None}")
            
            f0_yin_clean = f0_yin[~np.isnan(f0_yin)] if f0_yin is not None else np.array([])
            # self.logger.info(f"Pitch: YIN after NaN removal - f0_clean_size={f0_yin_clean.size}")
            
            if f0_yin_clean.size > 0:
                features.update(self._calc_stats(f0_yin_clean, prefix="yin"))
                # self.logger.info(f"Pitch: YIN success - f0_mean={features.get('f0_mean_yin', 0):.2f}")
            else:
                self.logger.warning("Pitch: YIN produced empty output after NaN removal - no pitch detected")
                features.update(self._zero_stats(prefix="yin"))
        except Exception as e:
            self.logger.error(f"Pitch: YIN failed with exception: {e}")
            features.update(self._zero_stats(prefix="yin"))

        # CREPE удален - не используется

        # Итоговые агрегаты
        # Улучшенный выбор лучшего метода: учитываем voiced_fraction и количество валидных фреймов
        score_pyin = (
            0.6 * float(features.get("f0_mean_pyin", 0.0)) +
            0.3 * float(features.get("voiced_fraction_pyin", 0.0)) * 100.0 +
            0.1 * float(features.get("f0_count_pyin", 0))
        )
        score_yin = (
            0.6 * float(features.get("f0_mean_yin", 0.0)) +
            0.3 * float(features.get("voiced_fraction_yin", 0.0)) * 100.0 +
            0.1 * float(features.get("f0_count_yin", 0))
        )
        best_method = "pyin" if score_pyin >= score_yin else "yin"
        f0_map = {
            "pyin": features.get("f0_series_pyin"),
            "yin": features.get("f0_series_yin"),
        }
        best_series = f0_map.get(best_method)
        if isinstance(best_series, list) and len(best_series) > 0:
            best_arr = np.asarray(best_series, dtype=np.float32)
            features["f0_mean"] = float(np.mean(best_arr))
            features["f0_std"] = float(np.std(best_arr))
            features["f0_min"] = float(np.min(best_arr))
            features["f0_max"] = float(np.max(best_arr))
            features["f0_median"] = float(np.median(best_arr))
            features["f0_method"] = best_method
            # self.logger.info(f"Pitch: selected best method '{best_method}' with {len(best_arr)} samples")
            # Доп. метрики стабильности
            if best_arr.size > 1:
                diff = np.diff(best_arr)
                features["pitch_variation"] = float(np.std(diff))
                features["pitch_stability"] = float(1.0 / (1.0 + np.std(diff)))
                features["pitch_range"] = float(features["f0_max"] - features["f0_min"])
                # Delta-признаки основной частоты
                features["f0_delta_mean"] = float(np.mean(diff))
                features["f0_delta_std"] = float(np.std(diff))
                features["f0_delta_abs_mean"] = float(np.mean(np.abs(diff)))
        else:
            self.logger.warning("Pitch: all methods returned empty/invalid output; using zeros")
            features.update({
                "f0_mean": 0.0,
                "f0_std": 0.0,
                "f0_min": 0.0,
                "f0_max": 0.0,
                "f0_median": 0.0,
                "f0_method": "none",
                "pitch_variation": 0.0,
                "pitch_stability": 0.0,
                "pitch_range": 0.0,
                "f0_delta_mean": 0.0,
                "f0_delta_std": 0.0,
                "f0_delta_abs_mean": 0.0,
            })
        
        # Ensure we always have some basic features even if all methods failed
        if not features.get("f0_mean"):
            features["f0_mean"] = 0.0
        if not features.get("f0_std"):
            features["f0_std"] = 0.0
        if not features.get("f0_min"):
            features["f0_min"] = 0.0
        if not features.get("f0_max"):
            features["f0_max"] = 0.0
        if not features.get("f0_median"):
            features["f0_median"] = 0.0
        if not features.get("f0_method"):
            features["f0_method"] = "none"

        return features


    def _extract_torchcrepe(self, audio: np.ndarray, sr: int) -> Optional[np.ndarray]:
        """Извлечение f0 через torchcrepe (PyTorch, GPU/CPU)."""
        try:
            import torch
            import torchcrepe
        except Exception as e:
            # self.logger.info(f"Pitch: torchcrepe not installed; skipping (reason: {e})")
            return None

        try:
            # Приводим к 16 kHz — torchcrepe обучен на 16 kHz
            if sr != 16000:
                audio_16k = librosa.resample(audio, orig_sr=sr, target_sr=16000).astype(np.float32, copy=False)
                sr_tc = 16000
            else:
                audio_16k = audio.astype(np.float32, copy=False)
                sr_tc = sr

            # Вход torchcrepe: torch.Tensor [batch=1, time]
            wav = torch.from_numpy(audio_16k).unsqueeze(0)
            # Перемещаем на устройство экстрактора, если это cuda и доступно
            if self.device == "cuda" and torch.cuda.is_available():
                wav = wav.cuda(non_blocking=True)

            # frame hop/size: torchcrepe использует hop размером по частоте кадров (default 80 samples @16kHz ~ 200Hz)
            # Выставим разумные параметры
            fmin = float(self.fmin)
            fmax = float(self.fmax)
            model = "tiny"  # точнее, можно "tiny" для скорости
            with torch.inference_mode():
                f0, pd = torchcrepe.predict(
                    wav, sr_tc,
                    fmin=fmin, fmax=fmax,
                    model=model,
                    batch_size=self.torchcrepe_batch_size,
                    device=wav.device,
                    return_periodicity=True,
                )
                # Очистка и перенос на CPU
                f0 = f0.squeeze(0).float().cpu().numpy()
                pd = pd.squeeze(0).float().cpu().numpy()

            # Фильтр по периодичности (доверие), убираем нули
            mask = pd > 0.1
            f0 = f0[mask]
            f0 = f0[f0 > 0]
            return f0 if f0.size > 0 else None
        except Exception as e:
            self.logger.warning(f"Pitch: torchcrepe extraction failed: {e}")
            return None

    def _calc_stats(self, f0: np.ndarray, prefix: str) -> Dict[str, Any]:
        f0 = f0.astype(np.float32, copy=False)
        stats: Dict[str, Any] = {
            f"f0_mean_{prefix}": float(np.mean(f0)) if f0.size else 0.0,
            f"f0_std_{prefix}": float(np.std(f0)) if f0.size else 0.0,
            f"f0_min_{prefix}": float(np.min(f0)) if f0.size else 0.0,
            f"f0_max_{prefix}": float(np.max(f0)) if f0.size else 0.0,
            f"f0_median_{prefix}": float(np.median(f0)) if f0.size else 0.0,
            f"f0_count_{prefix}": int(f0.size),
            f"f0_series_{prefix}": f0.tolist(),
        }
        return stats

    def _zero_stats(self, prefix: str) -> Dict[str, Any]:
        return {
            f"f0_mean_{prefix}": 0.0,
            f"f0_std_{prefix}": 0.0,
            f"f0_min_{prefix}": 0.0,
            f"f0_max_{prefix}": 0.0,
            f"f0_median_{prefix}": 0.0,
            f"f0_count_{prefix}": 0,
            f"f0_series_{prefix}": [],
        }