"""
Модуль core для обработки видео и анализа эмоций.
"""
from .video_processor import VideoEmotionProcessor
from .processing_config import ConfigLoader, ProcessingParams, ProcessingMetrics
from .memory_manager import memory_context, cleanup_memory, calculate_optimal_batch_size
from .retry_strategy import RetryStrategy, QualityMetrics
from .validation import ValidationLogic, ValidationCriteria
from .logger import StructuredLogger
from .exceptions import (
    VideoProcessingError, ConfigurationError, ConfigurationValidationError,
    FrameSelectionError, EmotionAnalysisError, ValidationError,
    MemoryError, VideoFileError, ModelError
)
from .protocols import (
    FaceDetector, EmotionModel, FrameManagerProtocol, LoggerProtocol
)
from .validators import (
    validate_video_file, validate_processing_params,
    validate_target_length, validate_chunk_size
)
from .edge_cases import (
    validate_edge_cases, handle_empty_video, handle_no_faces,
    handle_very_short_video, handle_very_long_video, check_video_duration
)
from .cache_with_ttl import TTLCache, FaceScanCacheWithTTL
from .metrics_exporter import MetricsExporter, StructuredMetricsLogger

__all__ = [
    "VideoEmotionProcessor",
    "ConfigLoader",
    "ProcessingParams",
    "ProcessingMetrics",
    "memory_context",
    "cleanup_memory",
    "calculate_optimal_batch_size",
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
    "VideoFileError",
    "ModelError",
    "FaceDetector",
    "EmotionModel",
    "FrameManagerProtocol",
    "LoggerProtocol",
    "validate_video_file",
    "validate_processing_params",
    "validate_target_length",
    "validate_chunk_size",
    "validate_edge_cases",
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

