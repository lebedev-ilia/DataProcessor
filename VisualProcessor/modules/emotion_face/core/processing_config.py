"""
Модуль для работы с конфигурацией обработки видео.
"""
import yaml
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Any, Optional
from core.exceptions import ConfigurationValidationError


@dataclass
class ProcessingParams:
    """Параметры обработки видео."""
    face_detection_threshold: float = 0.5
    scan_stride_multiplier: float = 1.0
    keyframe_threshold: float = 0.3
    quality_threshold: float = 0.4
    min_diversity: float = 0.2
    segment_max_gap: float = 0.5
    samples_per_segment: int = 10
    
    def __post_init__(self):
        """Валидация параметров после инициализации."""
        # Проверка диапазонов
        if not 0.0 <= self.face_detection_threshold <= 1.0:
            raise ConfigurationValidationError(
                f"face_detection_threshold must be in [0, 1], got {self.face_detection_threshold}"
            )
        
        if not 0.0 < self.scan_stride_multiplier <= 10.0:
            raise ConfigurationValidationError(
                f"scan_stride_multiplier must be in (0, 10], got {self.scan_stride_multiplier}"
            )
        
        if not 0.0 <= self.keyframe_threshold <= 1.0:
            raise ConfigurationValidationError(
                f"keyframe_threshold must be in [0, 1], got {self.keyframe_threshold}"
            )
        
        if not 0.0 <= self.quality_threshold <= 1.0:
            raise ConfigurationValidationError(
                f"quality_threshold must be in [0, 1], got {self.quality_threshold}"
            )
        
        if not 0.0 <= self.min_diversity <= 1.0:
            raise ConfigurationValidationError(
                f"min_diversity must be in [0, 1], got {self.min_diversity}"
            )
        
        if not 0.0 < self.segment_max_gap <= 10.0:
            raise ConfigurationValidationError(
                f"segment_max_gap must be in (0, 10], got {self.segment_max_gap}"
            )
        
        if not 1 <= self.samples_per_segment <= 1000:
            raise ConfigurationValidationError(
                f"samples_per_segment must be in [1, 1000], got {self.samples_per_segment}"
            )
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует параметры в словарь."""
        return {
            'face_detection_threshold': self.face_detection_threshold,
            'scan_stride_multiplier': self.scan_stride_multiplier,
            'keyframe_threshold': self.keyframe_threshold,
            'quality_threshold': self.quality_threshold,
            'min_diversity': self.min_diversity,
            'segment_max_gap': self.segment_max_gap,
            'samples_per_segment': self.samples_per_segment
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ProcessingParams':
        """Создает параметры из словаря."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
    
    def copy(self) -> 'ProcessingParams':
        """Создает копию параметров."""
        return ProcessingParams(
            face_detection_threshold=self.face_detection_threshold,
            scan_stride_multiplier=self.scan_stride_multiplier,
            keyframe_threshold=self.keyframe_threshold,
            quality_threshold=self.quality_threshold,
            min_diversity=self.min_diversity,
            segment_max_gap=self.segment_max_gap,
            samples_per_segment=self.samples_per_segment
        )


@dataclass
class ProcessingMetrics:
    """Метрики обработки видео."""
    timing: Dict[str, float] = field(default_factory=dict)
    memory_usage: Dict[str, float] = field(default_factory=dict)
    quality_scores: Dict[str, float] = field(default_factory=dict)
    processing_stats: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Преобразует метрики в словарь."""
        return {
            'timing': self.timing,
            'memory_usage': self.memory_usage,
            'quality_scores': self.quality_scores,
            'processing_stats': self.processing_stats
        }


class ConfigLoader:
    """Загрузчик конфигурации из YAML файла."""
    
    @staticmethod
    def load(config_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Загружает конфигурацию из YAML файла.
        
        Args:
            config_path: Путь к файлу конфигурации. Если None, используется config.yaml в корне проекта.
        
        Returns:
            Словарь с конфигурацией.
        
        Raises:
            ConfigurationError: Если файл не найден или некорректен.
        """
        from core.exceptions import ConfigurationError
        
        if config_path is None:
            config_path = Path(__file__).parent.parent / "config.yaml"
        else:
            config_path = Path(config_path)
        
        if not config_path.exists():
            raise ConfigurationError(
                f"Config file not found: {config_path}",
                details={"config_path": str(config_path)}
            )
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
        except yaml.YAMLError as e:
            raise ConfigurationError(
                f"Failed to parse YAML config file: {e}",
                details={"config_path": str(config_path), "error": str(e)}
            ) from e
        
        if config is None:
            raise ConfigurationError(
                "Config file is empty or invalid",
                details={"config_path": str(config_path)}
            )
        
        # Базовая валидация структуры
        ConfigLoader._validate_config(config)
        
        return config
    
    @staticmethod
    def _validate_config(config: Dict[str, Any]) -> None:
        """
        Валидирует базовую структуру конфигурации.
        
        Args:
            config: Словарь с конфигурацией.
        
        Raises:
            ConfigurationValidationError: Если структура некорректна.
        """
        from core.exceptions import ConfigurationValidationError
        
        required_sections = ["processing", "validation"]
        for section in required_sections:
            if section not in config:
                raise ConfigurationValidationError(
                    f"Missing required config section: {section}",
                    details={"section": section}
                )
        
        # Валидация processing секции
        processing = config.get("processing", {})
        if "target_length" in processing:
            target_length = processing["target_length"]
            if not isinstance(target_length, int) or target_length <= 0:
                raise ConfigurationValidationError(
                    f"processing.target_length must be positive integer, got {target_length}",
                    details={"section": "processing", "key": "target_length", "value": target_length}
                )
        
        if "max_retries" in processing:
            max_retries = processing["max_retries"]
            if not isinstance(max_retries, int) or max_retries < 0:
                raise ConfigurationValidationError(
                    f"processing.max_retries must be non-negative integer, got {max_retries}",
                    details={"section": "processing", "key": "max_retries", "value": max_retries}
                )
    
    @staticmethod
    def get_default_params() -> ProcessingParams:
        """Возвращает параметры по умолчанию."""
        return ProcessingParams()

