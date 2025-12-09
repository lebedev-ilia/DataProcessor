"""
Mel-спектрограмма экстрактор с поддержкой GPU (улучшенная версия).
"""
import warnings
import time
import logging
import os
import uuid
from typing import Dict, Any, Optional, Tuple

import numpy as np
import torch
import torchaudio

# Подавляем шумные предупреждения численных операций
warnings.filterwarnings("ignore", category=RuntimeWarning, module="numpy")
warnings.filterwarnings("ignore", message="invalid value encountered in sqrt")
warnings.filterwarnings("ignore", message="invalid value encountered in log")

from src.core.base_extractor import BaseExtractor, ExtractorResult
from src.core.audio_utils import AudioUtils
from src.utils.prof import timeit

logger = logging.getLogger(__name__)


class MelExtractor(BaseExtractor):
    """Экстрактор Mel-спектрограммы с поддержкой GPU и безопасной сериализацией."""
    
    name = "mel_extractor"
    version = "1.1.0"
    description = "Извлечение Mel-спектрограммы признаков"
    category = "spectral"
    dependencies = ["torch", "torchaudio", "numpy"]
    estimated_duration = 3.0
    
    gpu_required = False
    gpu_preferred = True
    gpu_memory_required = 1.0  # GB
    
    def __init__(
        self, 
        device: str = "auto",
        sample_rate: int = 22050,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
        power: float = 2.0,
        mix_to_mono: bool = True,
        save_threshold: int = 12 * 500,  # элементы порог, выше => сохраняем .npy
    ):
        super().__init__(device=device)
        # Resolve device to torch.device
        if device == "auto":
            self.torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.torch_device = torch.device(device)
        self.sample_rate = int(sample_rate)
        self.n_fft = int(n_fft)
        self.hop_length = int(hop_length)
        self.n_mels = int(n_mels)
        self.fmin = float(fmin)
        self.fmax = float(fmax) if fmax is not None else float(self.sample_rate // 2)
        self.power = float(power)
        self.mix_to_mono = bool(mix_to_mono)
        # Threshold to decide whether to embed mel matrix into payload or save to disk
        self.save_threshold = int(save_threshold)
        
        self.audio_utils = AudioUtils(device=str(self.torch_device), sample_rate=self.sample_rate)
        # Build torchaudio transforms (moved to torch_device)
        self._setup_transforms()
        # Скрываем инфо-лог инициализации
        # logger.debug(f"MelExtractor initialized on device={self.torch_device}")

    def _setup_transforms(self) -> None:
        """Инициализация torchaudio трансформов на целевом устройстве."""
        try:
            # MelSpectrogram accepts float args; create on CPU then move to device to avoid CUDA init at import time
            self.mel_spectrogram = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                f_min=self.fmin,
                f_max=self.fmax,
                power=self.power,
            ).to(self.torch_device)

            # Амплитуда -> dB
            self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype="power" if self.power != 1.0 else "amplitude").to(self.torch_device)
        except Exception as e:
            logger.exception(f"Ошибка настройки Mel трансформов: {e}")
            raise

    def run(self, input_uri: str, tmp_path: str) -> ExtractorResult:
        start_time = time.time()
        try:
            if not self._validate_input(input_uri):
                return self._create_result(False, error="Некорректный входной файл", processing_time=time.time() - start_time)

            self._log_extraction_start(input_uri)

            # load_audio скорее всего возвращает torch.Tensor (channel, time) — следуем этому контракту
            with timeit("mel: load_audio"):
                waveform, sr = self.audio_utils.load_audio(input_uri, self.sample_rate)
            # some AudioUtils return numpy; convert if needed
            if isinstance(waveform, np.ndarray):
                waveform = torch.from_numpy(waveform)

            # Ensure 2D tensor: (channels, time)
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)  # (1, T)
            elif waveform.ndim == 2:
                # OK: (channels, time)
                pass
            else:
                # collapse extras
                waveform = waveform.reshape(waveform.shape[0], -1)

            # Mix to mono if requested
            if waveform.shape[0] > 1 and self.mix_to_mono:
                waveform = waveform.mean(dim=0, keepdim=True)

            # Normalize if helper exists (keeps signature safe)
            try:
                with timeit("mel: normalize_audio"):
                    waveform = self.audio_utils.normalize_audio(waveform)
            except Exception:
                # If audio_utils.normalize_audio expects numpy, convert back and forth
                try:
                    with timeit("mel: normalize_audio (numpy fallback)"):
                        np_wave = waveform.cpu().numpy()
                        np_wave = self.audio_utils.normalize_audio(np_wave)
                        waveform = torch.from_numpy(np_wave)
                except Exception:
                    # fallback: ensure within [-1,1] by simple clipping
                    with timeit("mel: clamp [-1,1]"):
                        waveform = waveform.clamp(-1.0, 1.0)

            # Move to device and dtype float32
            waveform = waveform.to(dtype=torch.float32, device=self.torch_device)

            # Ensure shape (batch/channel, time) accepted by torchaudio: transforms expect (..., time) or (channel, time)
            # Our mel_spectrogram can accept (channel, time) or (batch, channel, time) — safe to pass as-is
            with torch.inference_mode():
                # Use autocast on CUDA to speed up (and allow fp16 inside)
                if self.torch_device.type == "cuda":
                    # autocast is beneficial for some ops
                    with torch.amp.autocast("cuda"):
                        with timeit("mel: MelSpectrogram"):
                            mel_spec = self.mel_spectrogram(waveform)  # shape: (channel, n_mels, frames)
                        with timeit("mel: AmplitudeToDB"):
                            mel_db = self.amplitude_to_db(mel_spec)
                else:
                    with timeit("mel: MelSpectrogram"):
                        mel_spec = self.mel_spectrogram(waveform)
                    with timeit("mel: AmplitudeToDB"):
                        mel_db = self.amplitude_to_db(mel_spec)

            # Convert to CPU numpy safely
            try:
                with timeit("mel: to_cpu"):
                    mel_db_cpu = mel_db.detach().cpu()
            except Exception:
                with timeit("mel: to_cpu (fallback)"):
                    mel_db_cpu = mel_db.cpu()

            with timeit("mel: tensor->numpy"):
                mel_np = mel_db_cpu.numpy()  # dtype float32 normally

            # Squeeze channel dim if single channel: result shape (n_mels, frames)
            if mel_np.ndim == 3 and mel_np.shape[0] == 1:
                mel_np = mel_np[0]

            # sanitize NaN/inf and clip dB to a safe dynamic range
            with timeit("mel: sanitize+clip"):
                mel_np = np.nan_to_num(mel_np, nan=-120.0, posinf=-120.0, neginf=-120.0)
                mel_np = np.clip(mel_np, -120.0, 0.0).astype(np.float32)

            # compute statistics (per mel bin and per time)
            try:
                with timeit("mel: stats computation"):
                    # per-frequency (mel bin) statistics over time (axis=1 if shape (n_mels, frames))
                    if mel_np.ndim == 2:
                        mel_mean = np.mean(mel_np, axis=1)
                        mel_std = np.std(mel_np, axis=1)
                        mel_min = np.min(mel_np, axis=1)
                        mel_max = np.max(mel_np, axis=1)

                        # per-time statistics
                        freq_mean = np.mean(mel_np, axis=0)
                        freq_std = np.std(mel_np, axis=0)
                    elif mel_np.ndim == 3:
                        # (channels, n_mels, frames) — compute on first channel for stats
                        chan0 = mel_np[0]
                        mel_mean = np.mean(chan0, axis=1)
                        mel_std = np.std(chan0, axis=1)
                        mel_min = np.min(chan0, axis=1)
                        mel_max = np.max(chan0, axis=1)
                        freq_mean = np.mean(chan0, axis=0)
                        freq_std = np.std(chan0, axis=0)
                    else:
                        mel_mean = mel_std = mel_min = mel_max = np.array([], dtype=np.float32)
                        freq_mean = freq_std = np.array([], dtype=np.float32)
            except Exception as e:
                logger.warning(f"Mel stats computation failed: {e}")
                mel_mean = mel_std = mel_min = mel_max = np.array([], dtype=np.float32)
                freq_mean = freq_std = np.array([], dtype=np.float32)

            # spectral centroid & bandwidth computed from mel bins (approx.) on linear scale to avoid overflow
            try:
                with timeit("mel: centroid+bandwidth"):
                    # convert dB to linear power scale in float64 for numerical stability
                    mel_lin = np.power(10.0, (mel_np.astype(np.float64) / 10.0))
                    freqs = np.linspace(self.fmin, self.fmax, self.n_mels, dtype=np.float64)
                    mel_sum = np.sum(mel_lin, axis=0)
                    mel_sum = np.where(mel_sum > 0.0, mel_sum, 1e-12)
                    spectral_centroid = (np.sum(freqs[:, None] * mel_lin, axis=0) / mel_sum)
                    diff_sq = (freqs[:, None] - spectral_centroid) ** 2
                    bandwidth_num = np.sum(diff_sq * mel_lin, axis=0)
                    bandwidth_ratio = bandwidth_num / mel_sum
                    spectral_bandwidth = np.sqrt(np.maximum(bandwidth_ratio, 0.0))
                    # cast back to float32 and sanitize
                    spectral_centroid = np.nan_to_num(spectral_centroid).astype(np.float32)
                    spectral_bandwidth = np.nan_to_num(spectral_bandwidth).astype(np.float32)
            except Exception:
                spectral_centroid = np.array([], dtype=np.float32)
                spectral_bandwidth = np.array([], dtype=np.float32)

            # compact stats vector (concatenate mean/std/min/max per mel) for easy ML concat
            try:
                with timeit("mel: concat stats vector"):
                    mel_stats_vector = np.concatenate([mel_mean, mel_std, mel_min, mel_max]).astype(float).tolist()
            except Exception:
                mel_stats_vector = []

            # Сохраняем все массивы в .npy файлы
            with timeit("mel: ensure tmp dir"):
                os.makedirs(tmp_path, exist_ok=True)
            
            # Сохраняем mel spectrogram
            mel_elements = int(np.prod(mel_np.shape))
            mel_npy_path = None
            if mel_elements > 0:
                fname = f"mel_spectrogram_{uuid.uuid4().hex}.npy"
                mel_npy_path = os.path.join(tmp_path, fname)
                try:
                    with timeit("mel: save mel npy"):
                        np.save(mel_npy_path, mel_np.astype(np.float32))
                    # logger.debug(f"Mel spectrogram saved to {mel_npy_path} (shape={mel_np.shape})")
                except Exception as e:
                    logger.warning(f"Failed to save mel spectrogram npy: {e}")
                    mel_npy_path = None
            
            # Сохраняем статистики массивов
            arrays_to_save = {
                "mel_mean": mel_mean,
                "mel_std": mel_std, 
                "mel_min": mel_min,
                "mel_max": mel_max,
                "freq_mean": freq_mean,
                "freq_std": freq_std,
                "spectral_centroid": spectral_centroid,
                "spectral_bandwidth": spectral_bandwidth,
                "mel_stats_vector": np.array(mel_stats_vector) if mel_stats_vector else np.array([])
            }
            
            saved_arrays = {}
            for name, array in arrays_to_save.items():
                if array.size > 0:
                    fname = f"{name}_{uuid.uuid4().hex}.npy"
                    array_path = os.path.join(tmp_path, fname)
                    try:
                        with timeit(f"mel: save {name} npy"):
                            np.save(array_path, array.astype(np.float32))
                        saved_arrays[name] = array_path
                        logger.debug(f"{name} saved to {array_path} (shape={array.shape})")
                    except Exception as e:
                        logger.warning(f"Failed to save {name} npy: {e}")
                        saved_arrays[name] = None
                else:
                    saved_arrays[name] = None

            duration = float(waveform.shape[-1] / float(sr)) if waveform.shape[-1] > 0 else 0.0
            processing_time = time.time() - start_time

            payload: Dict[str, Any] = {
                "mel_spectrogram_npy": mel_npy_path,
                "mel_shape": tuple(int(x) for x in mel_np.shape),
                "mel_elements": mel_elements,
                "sample_rate": sr,
                "n_fft": self.n_fft,
                "hop_length": self.hop_length,
                "n_mels": self.n_mels,
                "fmin": self.fmin,
                "fmax": self.fmax,
                "power": self.power,
                "duration": duration,
                "device_used": str(self.torch_device),
                "processing_time": processing_time,
                # Пути к .npy файлам с массивами
                "mel_mean_npy": saved_arrays["mel_mean"],
                "mel_std_npy": saved_arrays["mel_std"],
                "mel_min_npy": saved_arrays["mel_min"],
                "mel_max_npy": saved_arrays["mel_max"],
                "freq_mean_npy": saved_arrays["freq_mean"],
                "freq_std_npy": saved_arrays["freq_std"],
                "spectral_centroid_npy": saved_arrays["spectral_centroid"],
                "spectral_bandwidth_npy": saved_arrays["spectral_bandwidth"],
                "mel_stats_vector_npy": saved_arrays["mel_stats_vector"],
                # Размеры массивов
                "mel_mean_shape": list(mel_mean.shape) if mel_mean.size > 0 else [],
                "mel_std_shape": list(mel_std.shape) if mel_std.size > 0 else [],
                "mel_min_shape": list(mel_min.shape) if mel_min.size > 0 else [],
                "mel_max_shape": list(mel_max.shape) if mel_max.size > 0 else [],
                "freq_mean_shape": list(freq_mean.shape) if freq_mean.size > 0 else [],
                "freq_std_shape": list(freq_std.shape) if freq_std.size > 0 else [],
                "spectral_centroid_shape": list(spectral_centroid.shape) if spectral_centroid.size > 0 else [],
                "spectral_bandwidth_shape": list(spectral_bandwidth.shape) if spectral_bandwidth.size > 0 else [],
                "mel_stats_vector_shape": list(np.array(mel_stats_vector).shape) if mel_stats_vector else [],
                # Статистики по массивам
                "mel_mean_stats": {
                    "mean": float(np.mean(mel_mean)) if mel_mean.size > 0 else 0.0,
                    "std": float(np.std(mel_mean)) if mel_mean.size > 0 else 0.0,
                    "min": float(np.min(mel_mean)) if mel_mean.size > 0 else 0.0,
                    "max": float(np.max(mel_mean)) if mel_mean.size > 0 else 0.0,
                    "size": int(mel_mean.size)
                },
                "mel_std_stats": {
                    "mean": float(np.mean(mel_std)) if mel_std.size > 0 else 0.0,
                    "std": float(np.std(mel_std)) if mel_std.size > 0 else 0.0,
                    "min": float(np.min(mel_std)) if mel_std.size > 0 else 0.0,
                    "max": float(np.max(mel_std)) if mel_std.size > 0 else 0.0,
                    "size": int(mel_std.size)
                },
                "freq_mean_stats": {
                    "mean": float(np.mean(freq_mean)) if freq_mean.size > 0 else 0.0,
                    "std": float(np.std(freq_mean)) if freq_mean.size > 0 else 0.0,
                    "min": float(np.min(freq_mean)) if freq_mean.size > 0 else 0.0,
                    "max": float(np.max(freq_mean)) if freq_mean.size > 0 else 0.0,
                    "size": int(freq_mean.size)
                },
                "spectral_centroid_stats": {
                    "mean": float(np.mean(spectral_centroid)) if spectral_centroid.size > 0 else 0.0,
                    "std": float(np.std(spectral_centroid)) if spectral_centroid.size > 0 else 0.0,
                    "min": float(np.min(spectral_centroid)) if spectral_centroid.size > 0 else 0.0,
                    "max": float(np.max(spectral_centroid)) if spectral_centroid.size > 0 else 0.0,
                    "size": int(spectral_centroid.size)
                },
                "mel_stats_vector_stats": {
                    "mean": float(np.mean(mel_stats_vector)) if mel_stats_vector else 0.0,
                    "std": float(np.std(mel_stats_vector)) if mel_stats_vector else 0.0,
                    "min": float(np.min(mel_stats_vector)) if mel_stats_vector else 0.0,
                    "max": float(np.max(mel_stats_vector)) if mel_stats_vector else 0.0,
                    "size": len(mel_stats_vector) if mel_stats_vector else 0
                }
            }

            self._log_extraction_success(input_uri, processing_time)
            return self._create_result(True, payload=payload, processing_time=processing_time)

        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Ошибка извлечения Mel-спектрограммы: {e}"
            logger.exception(error_msg)
            self._log_extraction_error(input_uri, error_msg, processing_time)
            return self._create_result(False, error=error_msg, processing_time=processing_time)

    def _validate_input(self, input_uri: str) -> bool:
        # Reuse parent's validation and add extension check (optional)
        if not super()._validate_input(input_uri):
            return False
        # quick check for plausible audio extension
        audio_extensions = {".wav", ".mp3", ".flac", ".m4a", ".ogg"}
        if not any(input_uri.lower().endswith(ext) for ext in audio_extensions):
            pass
            # logger.debug(f"MelExtractor: file extension not typical audio: {input_uri}")
        return True
