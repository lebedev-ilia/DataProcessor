"""
MFCC экстрактор с поддержкой GPU.
"""
import time
import logging
import numpy as np
import torch
import torchaudio
from typing import Dict, Any, Optional

from src.core.base_extractor import BaseExtractor, ExtractorResult
from src.core.audio_utils import AudioUtils

logger = logging.getLogger(__name__)


class MFCCExtractor(BaseExtractor):
    """Экстрактор MFCC признаков с поддержкой GPU."""
    
    name = "mfcc_extractor"
    version = "1.1.0"
    description = "Извлечение MFCC (Mel-frequency cepstral coefficients) признаков"
    category = "spectral"
    dependencies = ["torch", "torchaudio"]
    estimated_duration = 2.0
    
    # Предпочитает GPU, но может работать на CPU
    gpu_required = False
    gpu_preferred = True
    gpu_memory_required = 0.5  # 500MB
    
    def __init__(
        self, 
        device: str = "auto",
        sample_rate: int = 22050,
        n_mfcc: int = 13,
        n_fft: int = 2048,
        hop_length: int = 512,
        n_mels: int = 128,
        fmin: float = 0.0,
        fmax: Optional[float] = None,
        normalize_mfcc: bool = False,
        min_gpu_duration_sec: float = 3.0
    ):
        """
        Инициализация MFCC экстрактора.
        
        Args:
            device: Устройство для обработки
            sample_rate: Частота дискретизации
            n_mfcc: Количество MFCC коэффициентов
            n_fft: Размер окна FFT
            hop_length: Шаг окна
            n_mels: Количество мел-фильтров
            fmin: Минимальная частота
            fmax: Максимальная частота
        """
        super().__init__(device=device)
        
        self.sample_rate = sample_rate
        self.n_mfcc = n_mfcc
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax or sample_rate // 2
        self.normalize_mfcc = normalize_mfcc
        self.min_gpu_duration_sec = float(min_gpu_duration_sec)
        
        self.audio_utils = AudioUtils(device=device, sample_rate=sample_rate)
        
        # Инициализируем трансформы
        self._setup_transforms()
    
    def _setup_transforms(self):
        """Настройка трансформов для MFCC."""
        try:
            # Создаем CPU-трансформы (всегда)
            self.mel_spectrogram_cpu = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                f_min=self.fmin,
                f_max=self.fmax
            )
            self.mfcc_transform_cpu = torchaudio.transforms.MFCC(
                sample_rate=self.sample_rate,
                n_mfcc=self.n_mfcc,
                melkwargs={
                    'n_fft': self.n_fft,
                    'hop_length': self.hop_length,
                    'n_mels': self.n_mels,
                    'f_min': self.fmin,
                    'f_max': self.fmax
                }
            )

            # И при наличии CUDA — дубли на GPU
            if self.device == "cuda" and torch.cuda.is_available():
                self.mel_spectrogram_gpu = torchaudio.transforms.MelSpectrogram(
                    sample_rate=self.sample_rate,
                    n_fft=self.n_fft,
                    hop_length=self.hop_length,
                    n_mels=self.n_mels,
                    f_min=self.fmin,
                    f_max=self.fmax
                ).to(self.device)
                self.mfcc_transform_gpu = torchaudio.transforms.MFCC(
                    sample_rate=self.sample_rate,
                    n_mfcc=self.n_mfcc,
                    melkwargs={
                        'n_fft': self.n_fft,
                        'hop_length': self.hop_length,
                        'n_mels': self.n_mels,
                        'f_min': self.fmin,
                        'f_max': self.fmax
                    }
                ).to(self.device)
            else:
                self.mel_spectrogram_gpu = None
                self.mfcc_transform_gpu = None
            
            # Убираем шумный инфо-лог при настройке трансформов
            # self.logger.debug(f"MFCC трансформы настроены для {self.device}")
            
        except Exception as e:
            self.logger.error(f"Ошибка настройки MFCC трансформов: {e}")
            raise
    
    def run(self, input_uri: str, tmp_path: str) -> ExtractorResult:
        """
        Извлечение MFCC признаков.
        
        Args:
            input_uri: Путь к аудио файлу
            tmp_path: Временная директория
            
        Returns:
            ExtractorResult с MFCC признаками
        """
        start_time = time.time()
        
        try:
            # Валидация входного файла
            if not self._validate_input(input_uri):
                return self._create_result(
                    success=False,
                    error="Некорректный входной файл",
                    processing_time=time.time() - start_time
                )
            
            self._log_extraction_start(input_uri)
            
            # Загружаем аудио
            waveform, sample_rate = self.audio_utils.load_audio(input_uri, self.sample_rate)
            
            # Нормализуем аудио
            waveform = self.audio_utils.normalize_audio(waveform)
            
            # Перемещаем на нужное устройство (эвристика для малых файлов)
            duration_sec = waveform.shape[1] / float(sample_rate)
            use_gpu = self.device == "cuda" and torch.cuda.is_available() and duration_sec >= self.min_gpu_duration_sec
            if use_gpu:
                waveform = self.audio_utils._move_to_device(waveform)
            
            # Извлекаем MFCC
            mfcc_features = self._extract_mfcc_features(waveform, prefer_gpu=use_gpu)
            
            # Вычисляем статистики
            mfcc_stats = self._compute_mfcc_statistics(mfcc_features)
            
            processing_time = time.time() - start_time
            
            # Создаем результат
            payload = {
                "mfcc_features": self.audio_utils.to_numpy(mfcc_features),
                "mfcc_statistics": mfcc_stats,
                "sample_rate": sample_rate,
                "n_mfcc": self.n_mfcc,
                "n_fft": self.n_fft,
                "hop_length": self.hop_length,
                "n_mels": self.n_mels,
                "fmin": self.fmin,
                "fmax": self.fmax,
                "duration": waveform.shape[1] / sample_rate,
                "device_used": self.device
            }
            
            self._log_extraction_success(input_uri, processing_time)
            
            return self._create_result(
                success=True,
                payload=payload,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Ошибка извлечения MFCC: {str(e)}"
            self._log_extraction_error(input_uri, error_msg, processing_time)
            
            return self._create_result(
                success=False,
                error=error_msg,
                processing_time=processing_time
            )
    
    def _extract_mfcc_features(self, waveform: torch.Tensor, prefer_gpu: bool = False) -> torch.Tensor:
        """Извлечение MFCC признаков."""
        try:
            # Выбираем трансформ (GPU/CPU) и применяем
            if prefer_gpu and self.mfcc_transform_gpu is not None:
                mfcc = self.mfcc_transform_gpu(waveform)
            else:
                # Гарантируем CPU для CPU-трансформа
                if waveform.device.type != "cpu":
                    waveform = waveform.cpu()
                mfcc = self.mfcc_transform_cpu(waveform)

            # Не выполняем дополнительный лог — MFCC уже основаны на лог-мел спектрограмме

            # Доп. опция: нормализация по времени (z-score)
            if self.normalize_mfcc:
                # Ожидается форма [B, n_mfcc, T] или [n_mfcc, T]
                if mfcc.dim() == 3:
                    mean = mfcc.mean(dim=2, keepdim=True)
                    std = mfcc.std(dim=2, keepdim=True).clamp(min=1e-8)
                else:
                    mean = mfcc.mean(dim=1, keepdim=True)
                    std = mfcc.std(dim=1, keepdim=True).clamp(min=1e-8)
                mfcc = (mfcc - mean) / std
            
            return mfcc
            
        except Exception as e:
            self.logger.error(f"Ошибка извлечения MFCC признаков: {e}")
            raise
    
    def _compute_mfcc_statistics(self, mfcc_features: torch.Tensor) -> Dict[str, Any]:
        """Вычисление статистик MFCC."""
        try:
            # Приводим к форме [n_mfcc, T]
            if mfcc_features.dim() == 3:
                mfcc_2d = mfcc_features[0]
            else:
                mfcc_2d = mfcc_features

            # Вычисления в torch (GPU-совместимые)
            mean = mfcc_2d.mean(dim=1)
            std = mfcc_2d.std(dim=1)
            min_vals = mfcc_2d.min(dim=1).values
            max_vals = mfcc_2d.max(dim=1).values

            # Дельты с использованием torchaudio.functional.compute_deltas
            deltas = torchaudio.functional.compute_deltas(mfcc_2d)
            delta_deltas = torchaudio.functional.compute_deltas(deltas)

            delta_mean = deltas.mean(dim=1)
            delta_std = deltas.std(dim=1)
            delta_delta_mean = delta_deltas.mean(dim=1)
            delta_delta_std = delta_deltas.std(dim=1)

            feature_shape = tuple(int(x) for x in mfcc_2d.shape)

            return {
                "mfcc_mean": mean.detach().cpu().numpy().tolist(),
                "mfcc_std": std.detach().cpu().numpy().tolist(),
                "mfcc_min": min_vals.detach().cpu().numpy().tolist(),
                "mfcc_max": max_vals.detach().cpu().numpy().tolist(),
                "delta_mean": delta_mean.detach().cpu().numpy().tolist(),
                "delta_std": delta_std.detach().cpu().numpy().tolist(),
                "delta_delta_mean": delta_delta_mean.detach().cpu().numpy().tolist(),
                "delta_delta_std": delta_delta_std.detach().cpu().numpy().tolist(),
                "feature_shape": feature_shape,
                "delta_shape": tuple(int(x) for x in deltas.shape),
                "delta_delta_shape": tuple(int(x) for x in delta_deltas.shape),
                "total_features": feature_shape[0] * 4
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка вычисления статистик MFCC: {e}")
            return {
                "mfcc_mean": [],
                "mfcc_std": [],
                "mfcc_min": [],
                "mfcc_max": [],
                "delta_mean": [],
                "delta_std": [],
                "delta_delta_mean": [],
                "delta_delta_std": [],
                "feature_shape": (0, 0),
                "total_features": 0
            }
    
    def _validate_input(self, input_uri: str) -> bool:
        """Валидация входного файла."""
        if not super()._validate_input(input_uri):
            return False
        
        # Проверяем, что это аудио файл
        audio_extensions = {'.wav', '.mp3', '.flac', '.m4a'}
        if not any(input_uri.lower().endswith(ext) for ext in audio_extensions):
            self.logger.error(f"Файл не является поддерживаемым аудио форматом: {input_uri}")
            return False
        
        # Проверяем существование и базовую информативность файла (без torchaudio.info)
        try:
            import os as _os
            if not _os.path.exists(input_uri):
                self.logger.error(f"Файл не существует: {input_uri}")
                return False
            if _os.path.getsize(input_uri) <= 0:
                self.logger.error(f"Пустой файл: {input_uri}")
                return False
        except Exception as e:
            self.logger.error(f"Не удалось проверить файл: {e}")
            return False
        
        return True
