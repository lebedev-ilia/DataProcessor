"""
Модуль core для обработки видео и анализа эмоций.
"""
from .video_processor import VideoEmotionProcessor
from .processing_config import ProcessingParams, ProcessingMetrics
from .memory_manager import memory_context, cleanup_memory
from .retry_strategy import RetryStrategy, QualityMetrics
from .validation import ValidationLogic, ValidationCriteria
from .logger import StructuredLogger
from .exceptions import (
    VideoProcessingError, ConfigurationError, ConfigurationValidationError,
    FrameSelectionError, EmotionAnalysisError, ValidationError,
    MemoryError, ModelError
)
from .protocols import (
    FaceDetector, EmotionModel, FrameManagerProtocol, LoggerProtocol
)
from .validators import (
    validate_processing_params,
    validate_target_length
)
from .edge_cases import (
    handle_empty_video, handle_no_faces,
    handle_very_short_video, handle_very_long_video, check_video_duration
)
from .cache_with_ttl import TTLCache, FaceScanCacheWithTTL
from .metrics_exporter import MetricsExporter, StructuredMetricsLogger

__all__ = [
    "VideoEmotionProcessor",
    "ProcessingParams",
    "ProcessingMetrics",
    "memory_context",
    "cleanup_memory",
    "RetryStrategy",
    "QualityMetrics",
    "ValidationLogic",
    "ValidationCriteria",
    "StructuredLogger",
    "VideoProcessingError",
    "ConfigurationError",
    "ConfigurationValidationError",
    "FrameSelectionError",
    "EmotionAnalysisError",
    "ValidationError",
    "MemoryError",
    "ModelError",
    "FaceDetector",
    "EmotionModel",
    "FrameManagerProtocol",
    "LoggerProtocol",
    "validate_processing_params",
    "validate_target_length",
    "handle_empty_video",
    "handle_no_faces",
    "handle_very_short_video",
    "handle_very_long_video",
    "check_video_duration",
    "TTLCache",
    "FaceScanCacheWithTTL",
    "MetricsExporter",
    "StructuredMetricsLogger"
]

