"""
Экспорт метрик в различные форматы.
"""
import json
import time
from typing import Dict, Any, Optional
from pathlib import Path
from core.processing_config import ProcessingMetrics


class MetricsExporter:
    """Экспортер метрик в различные форматы."""
    
    @staticmethod
    def export_json(metrics: ProcessingMetrics, output_path: Optional[str] = None) -> str:
        """
        Экспортирует метрики в JSON формат.
        
        Args:
            metrics: Метрики для экспорта.
            output_path: Путь для сохранения. Если None, возвращает JSON строку.
        
        Returns:
            JSON строка или путь к сохраненному файлу.
        """
        data = {
            "timestamp": time.time(),
            "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime()),
            "metrics": metrics.to_dict()
        }
        
        json_str = json.dumps(data, indent=2, ensure_ascii=False)
        
        if output_path:
            path = Path(output_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, 'w', encoding='utf-8') as f:
                f.write(json_str)
            return str(path)
        
        return json_str
    
    @staticmethod
    def export_prometheus_format(metrics: ProcessingMetrics, metric_prefix: str = "video_processing") -> str:
        """
        Экспортирует метрики в формате Prometheus.
        
        Args:
            metrics: Метрики для экспорта.
            metric_prefix: Префикс для метрик.
        
        Returns:
            Строка в формате Prometheus.
        """
        lines = []
        
        # Timing metrics
        for stage, duration in metrics.timing.items():
            metric_name = f"{metric_prefix}_stage_duration_seconds"
            lines.append(f'{metric_name}{{stage="{stage}"}} {duration}')
        
        # Memory metrics
        for label, memory_mb in metrics.memory_usage.items():
            metric_name = f"{metric_prefix}_memory_usage_mb"
            lines.append(f'{metric_name}{{label="{label}"}} {memory_mb}')
        
        # Quality scores
        for metric_name, value in metrics.quality_scores.items():
            full_metric_name = f"{metric_prefix}_quality_{metric_name}"
            lines.append(f"{full_metric_name} {value}")
        
        return "\n".join(lines)
    
    @staticmethod
    def export_summary(metrics: ProcessingMetrics) -> Dict[str, Any]:
        """
        Экспортирует краткую сводку метрик.
        
        Args:
            metrics: Метрики для экспорта.
        
        Returns:
            Словарь с краткой сводкой.
        """
        total_time = sum(metrics.timing.values())
        
        return {
            "total_processing_time": total_time,
            "stages_count": len(metrics.timing),
            "average_stage_time": total_time / len(metrics.timing) if metrics.timing else 0,
            "peak_memory_mb": max(metrics.memory_usage.values()) if metrics.memory_usage else 0,
            "quality_scores": metrics.quality_scores,
            "processing_stats": metrics.processing_stats
        }
    
    @staticmethod
    def export_for_logging(metrics: ProcessingMetrics) -> Dict[str, Any]:
        """
        Экспортирует метрики в формате для структурированного логирования.
        
        Args:
            metrics: Метрики для экспорта.
        
        Returns:
            Словарь для логирования.
        """
        return {
            "metrics": {
                "timing": metrics.timing,
                "memory": metrics.memory_usage,
                "quality": metrics.quality_scores
            },
            "stats": metrics.processing_stats,
            "timestamp": time.time()
        }


class StructuredMetricsLogger:
    """
    Логгер для структурированных метрик.
    """
    
    def __init__(self, enable_json_logging: bool = True, json_log_path: Optional[str] = None):
        """
        Инициализация логгера метрик.
        
        Args:
            enable_json_logging: Включить логирование в JSON.
            json_log_path: Путь для сохранения JSON логов.
        """
        self.enable_json_logging = enable_json_logging
        self.json_log_path = json_log_path
        self.metrics_history: list = []
    
    def log_metrics(self, metrics: ProcessingMetrics, label: str = "processing") -> None:
        """
        Логирует метрики в структурированном формате.
        
        Args:
            metrics: Метрики для логирования.
            label: Метка для логирования.
        """
        log_data = MetricsExporter.export_for_logging(metrics)
        log_data["label"] = label
        
        if self.enable_json_logging:
            json_str = json.dumps(log_data, indent=2, ensure_ascii=False)
            print(f"[METRICS] {json_str}")
            
            if self.json_log_path:
                self._append_to_json_log(log_data)
        
        self.metrics_history.append(log_data)
    
    def _append_to_json_log(self, log_data: Dict[str, Any]) -> None:
        """Добавляет запись в JSON лог файл."""
        if self.json_log_path:
            path = Path(self.json_log_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Читаем существующие логи или создаем новый список
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    try:
                        logs = json.load(f)
                    except json.JSONDecodeError:
                        logs = []
            else:
                logs = []
            
            logs.append(log_data)
            
            # Сохраняем обновленные логи
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(logs, f, indent=2, ensure_ascii=False)
    
    def get_history(self) -> list:
        """Возвращает историю метрик."""
        return self.metrics_history.copy()
    
    def clear_history(self) -> None:
        """Очищает историю метрик."""
        self.metrics_history.clear()

