"""
Кастомные исключения для обработки видео.
"""
from typing import Optional


class VideoProcessingError(Exception):
    """Базовое исключение для обработки видео."""
    
    def __init__(self, message: str, details: Optional[dict] = None):
        """
        Инициализация исключения.
        
        Args:
            message: Сообщение об ошибке.
            details: Дополнительные детали ошибки.
        """
        super().__init__(message)
        self.message = message
        self.details = details or {}
    
    def __str__(self) -> str:
        """Строковое представление исключения."""
        if self.details:
            details_str = ", ".join(f"{k}={v}" for k, v in self.details.items())
            return f"{self.message} ({details_str})"
        return self.message


class ConfigurationError(VideoProcessingError):
    """Ошибка конфигурации."""
    pass


class ConfigurationValidationError(ConfigurationError):
    """Ошибка валидации конфигурации."""
    pass


class FrameSelectionError(VideoProcessingError):
    """Ошибка выбора кадров."""
    pass


class EmotionAnalysisError(VideoProcessingError):
    """Ошибка анализа эмоций."""
    pass


class ValidationError(VideoProcessingError):
    """Ошибка валидации качества."""
    pass


class MemoryError(VideoProcessingError):
    """Ошибка нехватки памяти."""
    pass

class ModelError(VideoProcessingError):
    """Ошибка работы с моделью."""
    pass

