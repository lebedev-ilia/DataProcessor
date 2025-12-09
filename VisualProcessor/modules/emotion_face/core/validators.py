"""
Валидаторы для входных данных.
"""
from pathlib import Path
from typing import Optional
import cv2
from core.exceptions import VideoFileError, ConfigurationValidationError


def validate_video_file(video_path: str) -> None:
    """
    Валидирует видео файл.
    
    Args:
        video_path: Путь к видео файлу.
    
    Raises:
        VideoFileError: Если файл некорректен.
    """
    path = Path(video_path)
    
    # Проверка существования
    if not path.exists():
        raise VideoFileError(
            f"Video file not found: {video_path}",
            details={"video_path": str(video_path)}
        )
    
    # Проверка расширения
    valid_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm'}
    if path.suffix.lower() not in valid_extensions:
        raise VideoFileError(
            f"Unsupported video format: {path.suffix}. Supported: {valid_extensions}",
            details={"video_path": str(video_path), "extension": path.suffix}
        )
    
    # Проверка возможности открытия
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        raise VideoFileError(
            f"Cannot open video file: {video_path}",
            details={"video_path": str(video_path)}
        )
    
    # Проверка наличия кадров
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_count <= 0:
        cap.release()
        raise VideoFileError(
            f"Video file has no frames: {video_path}",
            details={"video_path": str(video_path), "frame_count": frame_count}
        )
    
    # Проверка FPS
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        raise VideoFileError(
            f"Invalid FPS in video file: {video_path}",
            details={"video_path": str(video_path), "fps": fps}
        )
    
    cap.release()


def validate_processing_params(params: 'ProcessingParams') -> None:
    """
    Валидирует параметры обработки.
    
    Args:
        params: Параметры обработки.
    
    Raises:
        ConfigurationValidationError: Если параметры некорректны.
    """
    from core.processing_config import ProcessingParams
    
    # Валидация уже выполняется в __post_init__, но можно добавить дополнительные проверки
    if not isinstance(params, ProcessingParams):
        raise ConfigurationValidationError(
            f"params must be ProcessingParams instance, got {type(params)}",
            details={"params_type": str(type(params))}
        )


def validate_target_length(target_length: Optional[int], default: int = 256) -> int:
    """
    Валидирует и нормализует target_length.
    
    Args:
        target_length: Целевая длина последовательности.
        default: Значение по умолчанию.
    
    Returns:
        Валидированное значение target_length.
    
    Raises:
        ConfigurationValidationError: Если значение некорректно.
    """
    if target_length is None:
        return default
    
    if not isinstance(target_length, int):
        raise ConfigurationValidationError(
            f"target_length must be integer, got {type(target_length)}",
            details={"target_length": target_length, "type": str(type(target_length))}
        )
    
    if target_length <= 0:
        raise ConfigurationValidationError(
            f"target_length must be positive, got {target_length}",
            details={"target_length": target_length}
        )
    
    if target_length > 10000:
        raise ConfigurationValidationError(
            f"target_length too large: {target_length} (max 10000)",
            details={"target_length": target_length}
        )
    
    return target_length


def validate_chunk_size(chunk_size: Optional[int], default: int = 32) -> int:
    """
    Валидирует и нормализует chunk_size.
    
    Args:
        chunk_size: Размер чанка.
        default: Значение по умолчанию.
    
    Returns:
        Валидированное значение chunk_size.
    
    Raises:
        ConfigurationValidationError: Если значение некорректно.
    """
    if chunk_size is None:
        return default
    
    if not isinstance(chunk_size, int):
        raise ConfigurationValidationError(
            f"chunk_size must be integer, got {type(chunk_size)}",
            details={"chunk_size": chunk_size, "type": str(type(chunk_size))}
        )
    
    if chunk_size <= 0:
        raise ConfigurationValidationError(
            f"chunk_size must be positive, got {chunk_size}",
            details={"chunk_size": chunk_size}
        )
    
    if chunk_size > 1024:
        raise ConfigurationValidationError(
            f"chunk_size too large: {chunk_size} (max 1024)",
            details={"chunk_size": chunk_size}
        )
    
    return chunk_size

