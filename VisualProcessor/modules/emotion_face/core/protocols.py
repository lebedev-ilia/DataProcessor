"""
Протоколы (интерфейсы) для типизации и тестируемости.
"""
from typing import Protocol, List, Dict, Any, Optional
import numpy as np


class FaceDetector(Protocol):
    """Протокол для детектора лиц."""
    
    def get(self, frame: np.ndarray) -> List[Any]:
        """
        Детектирует лица на кадре.
        
        Args:
            frame: Кадр в формате BGR (OpenCV).
        
        Returns:
            Список обнаруженных лиц.
        """
        ...


class EmotionModel(Protocol):
    """Протокол для модели анализа эмоций."""
    
    def __call__(self, batch_tensor: Any) -> Dict[str, Any]:
        """
        Предсказывает эмоции для батча кадров.
        
        Args:
            batch_tensor: Тензор с батчем кадров.
        
        Returns:
            Словарь с предсказаниями (valence, arousal, expression).
        """
        ...
    
    def eval(self) -> 'EmotionModel':
        """Переводит модель в режим оценки."""
        ...


class FrameManagerProtocol(Protocol):
    """Протокол для менеджера кадров."""
    
    def get(self, idx: int) -> np.ndarray:
        """
        Получает кадр по индексу.
        
        Args:
            idx: Индекс кадра.
        
        Returns:
            Кадр в формате numpy array.
        """
        ...
    
    def close(self) -> None:
        """Закрывает менеджер и освобождает ресурсы."""
        ...
    
    @property
    def total_frames(self) -> int:
        """Общее количество кадров."""
        ...
    
    @property
    def fps(self) -> float:
        """FPS видео."""
        ...


class LoggerProtocol(Protocol):
    """Протокол для логгера."""
    
    def log(self, message: str, level: str = "INFO") -> None:
        """Логирует сообщение."""
        ...
    
    def start_stage(self, stage_name: str) -> None:
        """Начинает отсчет времени для этапа."""
        ...
    
    def end_stage(self, stage_name: str) -> float:
        """Заканчивает отсчет времени для этапа."""
        ...

