"""
Модуль для структурированного логирования с метриками.
"""
import sys
import time
from typing import Dict, Any
from core.processing_config import ProcessingMetrics


class StructuredLogger:
    """
    Структурированный логгер с поддержкой метрик.
    """
    
    def __init__(self, enable_metrics: bool = True):
        """
        Инициализация логгера.
        
        Args:
            enable_metrics: Включить сбор метрик.
        """
        self.enable_metrics = enable_metrics
        self.metrics = ProcessingMetrics()
        self.stage_timings: Dict[str, float] = {}
        self.stage_start_times: Dict[str, float] = {}
    
    def log(self, message: str, level: str = "INFO"):
        """
        Логирует сообщение.
        
        Args:
            message: Текст сообщения.
            level: Уровень логирования (INFO, WARNING, ERROR).
        """
        prefix = f"[{level}]"
        print(f"{prefix} {message}", file=sys.stderr)
    
    def start_stage(self, stage_name: str):
        """
        Начинает отсчет времени для этапа обработки.
        
        Args:
            stage_name: Название этапа.
        """
        self.stage_start_times[stage_name] = time.time()
        self.log(f"Начало этапа: {stage_name}")
    
    def end_stage(self, stage_name: str) -> float:
        """
        Заканчивает отсчет времени для этапа обработки.
        
        Args:
            stage_name: Название этапа.
        
        Returns:
            Время выполнения этапа в секундах.
        """
        if stage_name in self.stage_start_times:
            elapsed = time.time() - self.stage_start_times[stage_name]
            self.stage_timings[stage_name] = elapsed
            self.metrics.timing[stage_name] = elapsed
            self.log(f"Завершение этапа: {stage_name} (время: {elapsed:.2f}с)")
            del self.stage_start_times[stage_name]
            return elapsed
        return 0.0
    
    def log_metrics(self, metrics: Dict[str, Any], label: str = "Метрики"):
        """
        Логирует структурированные метрики.
        
        Args:
            metrics: Словарь с метриками.
            label: Название группы метрик.
        """
        self.log(f"=== {label} ===")
        for key, value in metrics.items():
            if isinstance(value, float):
                self.log(f"  {key}: {value:.3f}")
            elif isinstance(value, dict):
                self.log(f"  {key}:")
                for sub_key, sub_value in value.items():
                    if isinstance(sub_value, float):
                        self.log(f"    {sub_key}: {sub_value:.3f}")
                    else:
                        self.log(f"    {sub_key}: {sub_value}")
            else:
                self.log(f"  {key}: {value}")
        self.log("=" * (len(label) + 8))
    
    def log_quality_metrics(self, quality_metrics: Dict[str, Any]):
        """
        Логирует метрики качества обработки.
        
        Args:
            quality_metrics: Метрики качества.
        """
        self.log("=== Метрики качества ===")
        self.log(f"  Валидность: {quality_metrics.get('is_valid', False)}")
        self.log(f"  Приемлемость: {quality_metrics.get('is_acceptable', False)}")
        self.log(f"  Монотонность: {quality_metrics.get('is_monotonic', False)}")
        
        metrics = quality_metrics.get('metrics', {})
        if metrics:
            self.log("  Детали:")
            self.log(f"    Разнообразие: {metrics.get('diversity_score', 0):.3f}")
            self.log(f"    Переходы: {metrics.get('significant_transitions', 0)}")
            self.log(f"    Ключевые кадры: {quality_metrics.get('keyframes_count', 0)}")
            self.log(f"    Количество кадров: {quality_metrics.get('frames_count', 0)}")
        
        self.log("=" * 25)
    
    def get_summary(self) -> Dict[str, Any]:
        """
        Возвращает сводку всех метрик.
        
        Returns:
            Словарь с метриками.
        """
        return {
            "timing": self.stage_timings.copy(),
            "metrics": self.metrics.to_dict()
        }
    
    def reset(self):
        """Сбрасывает все метрики."""
        self.stage_timings.clear()
        self.stage_start_times.clear()
        self.metrics = ProcessingMetrics()

