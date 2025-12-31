"""
Базовый класс для всех экстракторов с поддержкой GPU.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Union
import time
import logging
import torch
import numpy as np
from pathlib import Path

logger = logging.getLogger(__name__)


class ExtractorResult:
    """Результат работы экстрактора."""
    
    def __init__(
        self,
        name: str,
        version: str,
        success: bool,
        payload: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        processing_time: Optional[float] = None,
        device_used: str = "cpu"
    ):
        self.name = name
        self.version = version
        self.success = success
        self.payload = payload or {}
        self.error = error
        self.processing_time = processing_time
        self.device_used = device_used


class BaseExtractor(ABC):
    """Базовый класс для всех экстракторов с поддержкой GPU."""
    
    # Эти атрибуты должны быть переопределены в подклассах
    name: str = "base_extractor"
    version: str = "0.1.0"
    description: str = "Базовый экстрактор"
    category: str = "core"
    dependencies: list = []
    estimated_duration: float = 1.0
    
    # GPU настройки
    gpu_required: bool = False  # Требует ли экстрактор GPU
    gpu_preferred: bool = False  # Предпочитает ли GPU, но может работать на CPU
    gpu_memory_required: float = 0.0  # Требуемая память GPU в GB
    
    def __init__(self, device: Optional[str] = None, gpu_memory_limit: float = 0.8):
        """
        Инициализация экстрактора.
        
        Args:
            device: Устройство для обработки ('cuda', 'cpu', 'auto')
            gpu_memory_limit: Лимит памяти GPU (0.0-1.0)
        """
        self.logger = logging.getLogger(f"{__name__}.{self.name}")
        self.gpu_memory_limit = gpu_memory_limit
        
        # Определяем устройство
        if device == "auto":
            self.device = self._auto_detect_device()
        elif device is None:
            self.device = "cpu"  # По умолчанию CPU для безопасности
        else:
            self.device = device
            
        # Проверяем доступность GPU
        self.gpu_available = self._check_gpu_availability()
        
        # Устанавливаем финальное устройство
        if self.device == "cuda" and not self.gpu_available:
            self.logger.warning(f"GPU недоступен для {self.name}, переключаемся на CPU")
            self.device = "cpu"
            
        # Скрываем логи инициализации (оставляем только warnings/errors)
        # self.logger.debug(f"Инициализация {self.name} на устройстве: {self.device}")
    
    def _auto_detect_device(self) -> str:
        """Автоматическое определение лучшего устройства."""
        if self.gpu_required:
            if torch.cuda.is_available():
                return "cuda"
            else:
                raise RuntimeError(f"Экстрактор {self.name} требует GPU, но CUDA недоступна")
        
        if self.gpu_preferred and torch.cuda.is_available():
            # Проверяем доступную память GPU
            gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
            if gpu_memory >= self.gpu_memory_required:
                return "cuda"
        
        return "cpu"
    
    def _check_gpu_availability(self) -> bool:
        """Проверка доступности GPU с учетом требований."""
        if not torch.cuda.is_available():
            return False
            
        if self.gpu_required:
            # Проверяем доступную память
            try:
                gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
                available_memory = gpu_memory * (1 - self.gpu_memory_limit)
                return available_memory >= self.gpu_memory_required
            except Exception as e:
                self.logger.warning(f"Ошибка проверки GPU памяти: {e}")
                return False
                
        return True
    
    def _setup_gpu_memory(self):
        """Настройка памяти GPU для предотвращения OOM."""
        if self.device == "cuda":
            try:
                # Очищаем кэш
                torch.cuda.empty_cache()
                
                # Устанавливаем лимит памяти
                if self.gpu_memory_limit < 1.0:
                    total_memory = torch.cuda.get_device_properties(0).total_memory
                    max_memory = int(total_memory * self.gpu_memory_limit)
                    torch.cuda.set_per_process_memory_fraction(self.gpu_memory_limit)
                    
                # self.logger.info(f"GPU память настроена: лимит {self.gpu_memory_limit*100:.1f}%")
            except Exception as e:
                self.logger.warning(f"Не удалось настроить GPU память: {e}")
    
    def _move_to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Перемещение тензора на нужное устройство."""
        if self.device == "cuda" and tensor.device.type != "cuda":
            return tensor.cuda()
        elif self.device == "cpu" and tensor.device.type != "cpu":
            return tensor.cpu()
        return tensor
    
    def _to_numpy(self, tensor: torch.Tensor) -> np.ndarray:
        """Конвертация тензора в numpy array."""
        if tensor.requires_grad:
            tensor = tensor.detach()
        return tensor.cpu().numpy()
    
    @abstractmethod
    def run(self, input_uri: str, tmp_path: str) -> ExtractorResult:
        """
        Запуск экстрактора на заданном входе.
        
        Args:
            input_uri: URI к входному аудио/видео файлу
            tmp_path: Путь к временной директории для обработки
            
        Returns:
            ExtractorResult с извлеченными признаками или информацией об ошибке
        """
        pass
    
    def _create_result(
        self,
        success: bool,
        payload: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None,
        processing_time: Optional[float] = None
    ) -> ExtractorResult:
        """Создание результата экстрактора."""
        return ExtractorResult(
            name=self.name,
            version=self.version,
            success=success,
            payload=payload,
            error=error,
            processing_time=processing_time,
            device_used=self.device
        )
    
    def _time_execution(self, func, *args, **kwargs):
        """Измерение времени выполнения функции."""
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            return result, execution_time
        except Exception as e:
            execution_time = time.time() - start_time
            self.logger.error(f"Ошибка в {func.__name__}: {str(e)}")
            raise
    
    def _validate_input(self, input_uri: str) -> bool:
        """Валидация входного URI."""
        if not input_uri:
            self.logger.error("Входной URI пуст")
            return False
        
        # Поддерживаем S3 URI и локальные файлы
        if not (input_uri.startswith("s3://") or 
                input_uri.startswith("/") or 
                input_uri.endswith(('.wav', '.mp3', '.flac', '.mp4', '.avi', '.mov'))):
            self.logger.error(f"Неподдерживаемый формат URI: {input_uri}")
            return False
        
        return True
    
    def _log_extraction_start(self, input_uri: str):
        """Логирование начала извлечения."""
        # Убираем избыточные логи запуска
        pass
    
    def _log_extraction_success(self, input_uri: str, processing_time: float):
        """Логирование успешного извлечения."""
        # Убираем избыточные логи успешных операций
        pass
    
    def _log_extraction_error(self, input_uri: str, error: str, processing_time: float):
        """Логирование ошибки извлечения."""
        self.logger.error(
            f"❌ {self.name} не удался для {input_uri} "
            f"после {processing_time:.2f}s: {error}"
        )
    
    def get_info(self) -> Dict[str, Any]:
        """Получение информации об экстракторе."""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "category": self.category,
            "device": self.device,
            "gpu_required": self.gpu_required,
            "gpu_preferred": self.gpu_preferred,
            "gpu_available": self.gpu_available,
            "estimated_duration": self.estimated_duration
        }
    
    def __str__(self) -> str:
        return f"{self.name} v{self.version} ({self.device})"
    
    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}(name='{self.name}', device='{self.device}')>"
