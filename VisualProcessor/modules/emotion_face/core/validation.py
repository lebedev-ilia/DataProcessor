"""
Модуль для валидации качества обработки видео.
"""
from dataclasses import dataclass
from typing import Dict, Any, List, Optional
from core.retry_strategy import QualityMetrics


@dataclass
class ValidationCriteria:
    """Критерии валидации."""
    min_frames_ratio: float = 0.8  # Минимум 80% от target_length
    min_keyframes: int = 3
    min_transitions: int = 2
    min_diversity: float = 0.2
    
    # Для монотонных видео
    monotonic_min_frames_ratio: float = 0.8
    monotonic_min_keyframes: int = 1
    monotonic_min_diversity: float = 0.05


class ValidationLogic:
    """
    Единая логика валидации качества обработки.
    Устраняет несогласованность между разными проверками.
    """
    
    def __init__(self, criteria: Optional[ValidationCriteria] = None):
        """
        Инициализация валидатора.
        
        Args:
            criteria: Критерии валидации. Если None, используются значения по умолчанию.
        """
        self.criteria = criteria or ValidationCriteria()
    
    def validate_quality(
        self,
        smoothed_emotions: List[Dict[str, Any]],
        quality_metrics: Dict[str, Any],
        target_length: int,
        keyframes_count: int,
        is_monotonic: bool = False,
        neutral_percentage: float = 0.0,
        logger = None
    ) -> QualityMetrics:
        """
        Валидирует качество обработки с единой логикой.
        
        Args:
            smoothed_emotions: Список сглаженных эмоций.
            quality_metrics: Метрики качества из validate_sequence_quality.
            target_length: Целевая длина последовательности.
            keyframes_count: Количество ключевых кадров.
            is_monotonic: Является ли видео монотонным.
            neutral_percentage: Процент нейтральных эмоций.
        
        Returns:
            QualityMetrics с результатами валидации.
        """
        # Базовые метрики
        is_valid = quality_metrics.get("is_valid", False)
        quality_details = quality_metrics.get("metrics", {})
        diversity_score = quality_details.get("diversity_score", 0.0)
        transition_count = quality_details.get("significant_transitions", 0)
        frames_count = len(smoothed_emotions)
        
        # Определяем, является ли видео монотонным
        is_monotonic_video = (
            is_monotonic or 
            neutral_percentage > 0.7 or
            quality_metrics.get("is_monotonic", False)
        )
        
        # Выбираем критерии в зависимости от типа видео
        if is_monotonic_video:
            min_frames = int(target_length * self.criteria.monotonic_min_frames_ratio)
            min_keyframes = self.criteria.monotonic_min_keyframes
            min_diversity = self.criteria.monotonic_min_diversity
            min_transitions = 0  # Для монотонных видео не требуем переходов
        else:
            min_frames = int(target_length * self.criteria.min_frames_ratio)
            min_keyframes = self.criteria.min_keyframes
            min_diversity = self.criteria.min_diversity
            min_transitions = self.criteria.min_transitions
        
        # Проверяем критерии
        has_enough_frames = frames_count >= min_frames
        has_enough_keyframes = keyframes_count >= min_keyframes
        has_enough_diversity = diversity_score >= min_diversity
        has_enough_transitions = transition_count >= min_transitions
        
        # Итоговая оценка
        is_acceptable = (
            is_valid and
            has_enough_frames and
            has_enough_keyframes and
            (has_enough_diversity or is_monotonic_video)
        )

        logger.info(
            f"[VALIDATION] is_valid: {is_valid} | has_enough_frames: {has_enough_frames} | has_enough_keyframes: {has_enough_keyframes} | has_enough_diversity: {has_enough_diversity} | is_monotonic_video: {is_monotonic_video} | has_enough_transitions: {has_enough_transitions} ({transition_count}>={min_transitions})"
            )
        
        return QualityMetrics(
            is_valid=is_valid,
            is_acceptable=is_acceptable,
            diversity_score=diversity_score,
            transition_count=transition_count,
            frames_count=frames_count,
            keyframes_count=keyframes_count,
            is_monotonic=is_monotonic_video,
            neutral_percentage=neutral_percentage
        )
    
    def get_validation_summary(self, metrics: QualityMetrics) -> Dict[str, Any]:
        """
        Возвращает краткое описание результатов валидации.
        
        Args:
            metrics: Метрики качества.
        
        Returns:
            Словарь с описанием результатов.
        """
        return {
            "is_acceptable": metrics.is_acceptable,
            "is_valid": metrics.is_valid,
            "is_monotonic": metrics.is_monotonic,
            "frames_count": metrics.frames_count,
            "keyframes_count": metrics.keyframes_count,
            "diversity_score": metrics.diversity_score,
            "transition_count": metrics.transition_count,
            "neutral_percentage": metrics.neutral_percentage
        }

