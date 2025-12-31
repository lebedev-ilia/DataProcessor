"""
Модуль для управления стратегией повторных попыток обработки.
"""
from dataclasses import dataclass
from typing import Dict, Any, Optional, Callable
from core.processing_config import ProcessingParams


@dataclass
class QualityMetrics:
    """Метрики качества обработки."""
    is_valid: bool
    is_acceptable: bool
    diversity_score: float = 0.0
    transition_count: int = 0
    frames_count: int = 0
    keyframes_count: int = 0
    is_monotonic: bool = False
    neutral_percentage: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует метрики в словарь."""
        return {
            'is_valid': self.is_valid,
            'is_acceptable': self.is_acceptable,
            'diversity_score': self.diversity_score,
            'transition_count': self.transition_count,
            'frames_count': self.frames_count,
            'keyframes_count': self.keyframes_count,
            'is_monotonic': self.is_monotonic,
            'neutral_percentage': self.neutral_percentage
        }


class RetryStrategy:
    """
    Стратегия повторных попыток обработки видео.
    Управляет логикой повторных попыток с адаптацией параметров.
    """
    
    def __init__(self, max_retries: int = 2):
        """
        Инициализация стратегии.
        
        Args:
            max_retries: Максимальное количество повторных попыток.
        """
        self.max_retries = max_retries
        self.attempts = 0
        self.quality_history: list[QualityMetrics] = []
    
    def should_retry(self, quality_metrics: QualityMetrics) -> bool:
        """
        Определяет, нужно ли делать повторную попытку.
        
        Args:
            quality_metrics: Метрики качества текущей попытки.
        
        Returns:
            True, если нужно повторить попытку.
        """
        self.quality_history.append(quality_metrics)
        
        # Если качество приемлемое, не нужно повторять
        if quality_metrics.is_acceptable:
            return False
        
        # Если достигнут лимит попыток, не нужно повторять
        if self.attempts >= self.max_retries:
            return False
        
        return True
    
    def next_attempt(self) -> bool:
        """
        Переходит к следующей попытке.
        
        Returns:
            True, если можно продолжить, False если достигнут лимит.
        """
        if self.attempts >= self.max_retries:
            return False
        
        self.attempts += 1
        return True
    
    def adjust_parameters(
        self,
        current_params: ProcessingParams,
        quality_metrics: QualityMetrics,
        video_type: str,
        segments_count: int,
        faces_found: int,
        log_func: Optional[Callable[[str], None]] = None
    ) -> ProcessingParams:
        """
        Адаптирует параметры для следующей попытки на основе метрик качества.
        
        Args:
            current_params: Текущие параметры обработки.
            quality_metrics: Метрики качества предыдущей попытки.
            video_type: Тип видео (STATIC_FACE, CONTINUOUS_FACE, DYNAMIC_FACES).
            segments_count: Количество сегментов.
            faces_found: Количество найденных лиц.
            log_func: Функция для логирования (опционально).
        
        Returns:
            Новые параметры обработки.
        """
        new_params = current_params.copy()
        
        if log_func:
            log_func(f"[RetryStrategy] Попытка {self.attempts + 1}/{self.max_retries + 1}")
        
        # Для монотонных видео сразу снижаем требования
        if quality_metrics.neutral_percentage > 0.8:
            if log_func:
                log_func("[RetryStrategy] Видео явно монотонное, снижаю требования")
            new_params.min_diversity = 0.05
            new_params.quality_threshold = 0.15
            return new_params
        
        # Адаптация в зависимости от типа видео
        if video_type == "STATIC_FACE" or (segments_count == 1 and faces_found > 100):
            if log_func:
                log_func("[RetryStrategy] Стратегия для STATIC_FACE: увеличиваю выборку")
            new_params.samples_per_segment = 100
            new_params.segment_max_gap = 0.2
            new_params.keyframe_threshold = 0.15
            new_params.min_diversity = 0.1
        
        elif quality_metrics.diversity_score < 0.2:
            if log_func:
                log_func("[RetryStrategy] Стратегия для LOW_DIVERSITY: максимальная детализация")
            new_params.keyframe_threshold = 0.1
            new_params.samples_per_segment = 80
            new_params.segment_max_gap = 0.3
        
        elif quality_metrics.transition_count < 2:
            if log_func:
                log_func("[RetryStrategy] Стратегия для FEW_TRANSITIONS: снижаю требования")
            new_params.quality_threshold = 0.25
            new_params.min_diversity = 0.1
            new_params.samples_per_segment = 60
        
        # Общие корректировки
        if self.attempts == 1:
            # Первая повторная попытка: снижаем пороги
            new_params.face_detection_threshold *= 0.8
            new_params.scan_stride_multiplier *= 0.7
            new_params.keyframe_threshold *= 0.8
        
        return new_params
    
    def get_safe_params(self) -> ProcessingParams:
        """
        Возвращает безопасные параметры для обработки после ошибки.
        
        Returns:
            Безопасные параметры обработки.
        """
        return ProcessingParams(
            face_detection_threshold=0.3,
            scan_stride_multiplier=0.5,
            keyframe_threshold=0.2,
            quality_threshold=0.3,
            min_diversity=0.1,
            segment_max_gap=1.0,
            samples_per_segment=15
        )
    
    def reset(self):
        """Сбрасывает счетчик попыток."""
        self.attempts = 0
        self.quality_history.clear()

