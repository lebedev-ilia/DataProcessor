"""
Утилиты для работы с аудио файлами с поддержкой GPU.
"""
import warnings
import os
import logging
import numpy as np
import torch
import torchaudio
import librosa
import soundfile as sf
from typing import Tuple, Optional, Union
from pathlib import Path

# Подавляем warnings от torchaudio
warnings.filterwarnings("ignore", message="torchaudio._backend.set_audio_backend has been deprecated")

logger = logging.getLogger(__name__)


class AudioUtils:
    """Утилиты для работы с аудио."""
    
    def __init__(self, device: str = "cpu", sample_rate: int = 22050):
        """
        Инициализация утилит аудио.
        
        Args:
            device: Устройство для обработки ('cuda', 'cpu')
            sample_rate: Частота дискретизации
        """
        self.device = device
        self.sample_rate = sample_rate
        self.logger = logging.getLogger(f"{__name__}.AudioUtils")
        
        # Настройка torchaudio для GPU (убрано deprecated предупреждение)
        # torchaudio.set_audio_backend deprecated, используем по умолчанию
    
    def load_audio(self, file_path: str, target_sr: Optional[int] = None) -> Tuple[torch.Tensor, int]:
        """
        Загрузка аудио файла с поддержкой GPU.
        
        Args:
            file_path: Путь к аудио файлу
            target_sr: Целевая частота дискретизации
            
        Returns:
            Tuple[torch.Tensor, int]: (аудио данные, частота дискретизации)
        """
        try:
            if self.device == "cuda" and torch.cuda.is_available():
                return self._load_audio_gpu(file_path, target_sr)
            else:
                return self._load_audio_cpu(file_path, target_sr)
        except Exception as e:
            self.logger.error(f"Ошибка загрузки аудио {file_path}: {e}")
            # Fallback на CPU
            if self.device == "cuda":
                self.logger.warning("Переключение на CPU загрузку")
                return self._load_audio_cpu(file_path, target_sr)
            raise

    def load_audio_segment(
        self,
        file_path: str,
        *,
        start_sample: int,
        end_sample: int,
        target_sr: Optional[int] = None,
        mix_to_mono: bool = True,
    ) -> Tuple[torch.Tensor, int]:
        """
        Load a slice of audio by sample indices (fail-fast).

        This is used for Segmenter-aligned audio windows (audio/segments.json),
        to avoid writing temporary WAV files and to keep performance stable.
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Аудио файл не найден: {file_path}")
        if os.path.getsize(file_path) == 0:
            raise ValueError(f"Аудио файл пуст: {file_path}")

        start = int(start_sample)
        end = int(end_sample)
        if start < 0 or end < 0 or end < start:
            raise ValueError(f"Некорректные границы сегмента: start_sample={start} end_sample={end}")

        # Prefer soundfile for reliable frame slicing (PCM wav).
        target_sr = target_sr or self.sample_rate
        try:
            with sf.SoundFile(file_path, mode="r") as f:
                sr = int(f.samplerate)
                total = int(len(f))
                start_c = min(start, total)
                end_c = min(end, total)
                if end_c <= start_c:
                    raise ValueError(f"Пустой сегмент после clip: start={start_c} end={end_c} total={total}")
                f.seek(start_c)
                frames = int(end_c - start_c)
                data = f.read(frames=frames, dtype="float32", always_2d=True)  # shape [frames, ch]

            if data.size == 0:
                raise ValueError("Пустые данные сегмента")
            # Mix to mono if requested
            if data.ndim == 2:
                if mix_to_mono:
                    data = np.mean(data, axis=1)
                else:
                    data = data[:, 0]
            data = np.asarray(data, dtype=np.float32)

            # Resample if needed
            if sr != target_sr:
                data = librosa.resample(data, orig_sr=sr, target_sr=int(target_sr))
                sr = int(target_sr)

            waveform = torch.from_numpy(data).float().unsqueeze(0)
            if self.device == "cuda" and torch.cuda.is_available():
                waveform = waveform.to(self.device, non_blocking=True)
            return waveform, sr
        except Exception as e:
            raise RuntimeError(f"Не удалось загрузить сегмент аудио: {e}") from e
    
    def _load_audio_gpu(self, file_path: str, target_sr: Optional[int] = None) -> Tuple[torch.Tensor, int]:
        """Загрузка аудио на GPU."""
        try:
            # Загружаем на CPU сначала
            waveform, sample_rate = torchaudio.load(file_path)
            
            # Ресемплинг если нужно
            if target_sr and target_sr != sample_rate:
                resampler = torchaudio.transforms.Resample(sample_rate, target_sr)
                waveform = resampler(waveform)
                sample_rate = target_sr
            
            # Перемещаем на GPU
            waveform = waveform.to(self.device)
            
            # Конвертируем в моно если стерео
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            return waveform, sample_rate
            
        except Exception as e:
            self.logger.error(f"GPU загрузка не удалась: {e}")
            raise
    
    def _load_audio_cpu(self, file_path: str, target_sr: Optional[int] = None) -> Tuple[torch.Tensor, int]:
        """Загрузка аудио на CPU."""
        # Предварительные проверки
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Аудио файл не найден: {file_path}")
        if os.path.getsize(file_path) == 0:
            raise ValueError(f"Аудио файл пуст: {file_path}")

        target_sr = target_sr or self.sample_rate

        # 1) Попытка через soundfile (быстро и надёжно для PCM)
        try:
            data, sr = sf.read(file_path, always_2d=False)
            if data is None or (isinstance(data, np.ndarray) and data.size == 0):
                raise ValueError("Пустые данные при чтении soundfile")

            # Приводим к float32 ndarray, моно
            if isinstance(data, np.ndarray) and data.ndim == 2:
                data = np.mean(data, axis=1)
            data = data.astype(np.float32, copy=False)

            # Ресемплинг при необходимости
            if sr != target_sr:
                data = librosa.resample(data, orig_sr=sr, target_sr=target_sr)
                sr = target_sr

            waveform = torch.from_numpy(data).float().unsqueeze(0)
            return waveform, sr
        except Exception as e_sf:
            self.logger.warning(f"soundfile не удалось: {e_sf}")

        # 2) Попытка через torchaudio
        try:
            waveform_t, sr = torchaudio.load(file_path)
            if waveform_t.numel() == 0:
                raise ValueError("Пустые данные при чтении torchaudio")

            # К моно
            if waveform_t.shape[0] > 1:
                waveform_t = torch.mean(waveform_t, dim=0, keepdim=True)

            # Ресемплинг
            if sr != target_sr:
                resampler = torchaudio.transforms.Resample(sr, target_sr)
                waveform_t = resampler(waveform_t)
                sr = target_sr

            return waveform_t.float(), sr
        except Exception as e_ta:
            self.logger.warning(f"torchaudio не удалось: {e_ta}")

        # 3) Попытка через librosa (универсальный fallback)
        try:
            data, sr = librosa.load(
                file_path,
                sr=target_sr,
                mono=True
            )
            if data is None or (isinstance(data, np.ndarray) and data.size == 0):
                raise ValueError("Пустые данные при чтении librosa")

            waveform = torch.from_numpy(data).float().unsqueeze(0)
            return waveform, sr
        except Exception as e_lb:
            self.logger.error("CPU загрузка не удалась через все бэкенды")
            # Сохраняем контекст ошибок для диагностики
            raise RuntimeError(f"Не удалось загрузить аудио. soundfile: {e_sf}; torchaudio: {e_ta}; librosa: {e_lb}")
    
    def extract_audio_from_video(self, video_path: str, output_path: str, target_sr: Optional[int] = None) -> str:
        """
        Извлечение аудио из видео файла.
        
        Args:
            video_path: Путь к видео файлу
            output_path: Путь для сохранения аудио
            
        Returns:
            str: Путь к извлеченному аудио файлу
        """
        try:
            import subprocess
            
            # Создаем директорию если не существует
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            # Определяем желаемую частоту дискретизации
            out_sr = int(target_sr or self.sample_rate)

            # Команда ffmpeg для извлечения аудио
            cmd = [
                'ffmpeg',
                '-hide_banner', '-loglevel', 'error', '-nostdin',
                '-i', video_path,
                '-vn',  # Без видео
                '-acodec', 'pcm_s16le',  # PCM 16-bit
                '-ar', str(out_sr),  # Частота дискретизации
                '-ac', '1',  # Моно
                '-y',  # Перезаписать файл
                output_path
            ]
            
            # Выполняем команду
            result = subprocess.run(
                cmd, 
                capture_output=True, 
                text=True, 
                check=True
            )
            
            if os.path.exists(output_path):
                # Снижаем уровень логирования успешной операции, чтобы избежать шума/дубликатов
                # self.logger.debug(f"Аудио успешно извлечено: {output_path}")
                return output_path
            else:
                raise RuntimeError("Файл аудио не был создан")
                
        except subprocess.CalledProcessError as e:
            error_msg = f"Ошибка ffmpeg: {e.stderr}"
            self.logger.error(error_msg)
            raise RuntimeError(error_msg)
        except Exception as e:
            self.logger.error(f"Ошибка извлечения аудио: {e}")
            raise
    
    def get_audio_info(self, file_path: str) -> dict:
        """
        Получение информации об аудио файле.
        
        Args:
            file_path: Путь к аудио файлу
            
        Returns:
            dict: Информация об аудио
        """
        try:
            # Загружаем аудио
            waveform, sample_rate = self.load_audio(file_path)
            
            duration = waveform.shape[1] / sample_rate
            
            return {
                "sample_rate": sample_rate,
                "duration": duration,
                "channels": waveform.shape[0],
                "samples": waveform.shape[1],
                "file_size": os.path.getsize(file_path) if os.path.exists(file_path) else 0
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка получения информации об аудио: {e}")
            return {}
    
    def normalize_audio(self, waveform: torch.Tensor) -> torch.Tensor:
        """
        Нормализация аудио.
        
        Args:
            waveform: Аудио тензор
            
        Returns:
            torch.Tensor: Нормализованный аудио
        """
        try:
            # Нормализация по максимуму
            max_val = torch.max(torch.abs(waveform))
            if max_val > 0:
                waveform = waveform / max_val
            
            return waveform
            
        except Exception as e:
            self.logger.error(f"Ошибка нормализации аудио: {e}")
            return waveform
    
    def apply_preemphasis(self, waveform: torch.Tensor, coeff: float = 0.97) -> torch.Tensor:
        """
        Применение предыскажения.
        
        Args:
            waveform: Аудио тензор
            coeff: Коэффициент предыскажения
            
        Returns:
            torch.Tensor: Аудио с предыскажением
        """
        try:
            if self.device == "cuda" and waveform.device.type == "cuda":
                # GPU версия
                preemphasized = torch.zeros_like(waveform)
                preemphasized[..., 0] = waveform[..., 0]
                preemphasized[..., 1:] = waveform[..., 1:] - coeff * waveform[..., :-1]
                return preemphasized
            else:
                # CPU версия
                preemphasized = np.zeros_like(waveform.numpy())
                preemphasized[..., 0] = waveform.numpy()[..., 0]
                preemphasized[..., 1:] = waveform.numpy()[..., 1:] - coeff * waveform.numpy()[..., :-1]
                return torch.from_numpy(preemphasized).float()
                
        except Exception as e:
            self.logger.error(f"Ошибка применения предыскажения: {e}")
            return waveform
    
    def to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Конвертация тензора в numpy array."""
        if tensor.requires_grad:
            tensor = tensor.detach()
        return tensor.cpu().numpy()
    
    def to_torch(self, array: np.ndarray, device: Optional[str] = None) -> torch.Tensor:
        """Конвертация numpy array в torch tensor."""
        device = device or self.device
        tensor = torch.from_numpy(array).float()
        if device == "cuda" and torch.cuda.is_available():
            tensor = tensor.cuda()
        return tensor
    
    def _move_to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Перемещение тензора на нужное устройство."""
        if self.device == "cuda" and tensor.device.type != "cuda":
            return tensor.cuda()
        elif self.device == "cpu" and tensor.device.type != "cuda":
            return tensor.cpu()
        return tensor
