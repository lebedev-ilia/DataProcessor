"""
Pydantic модели для AudioProcessor.
"""
from pydantic import BaseModel, Field, validator
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
from enum import Enum


class ProcessingStatus(str, Enum):
    """Статус обработки."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class DeviceType(str, Enum):
    """Тип устройства."""
    CPU = "cpu"
    CUDA = "cuda"
    AUTO = "auto"


class ProcessRequest(BaseModel):
    """Запрос на обработку видео."""
    video_path: str = Field(..., description="Путь к видео файлу")
    output_dir: str = Field(..., description="Директория для сохранения результатов")
    extractor_names: Optional[List[str]] = Field(None, description="Список экстракторов для запуска")
    extract_audio: bool = Field(True, description="Извлекать ли аудио из видео")
    device: DeviceType = Field(DeviceType.AUTO, description="Устройство для обработки")
    sample_rate: int = Field(22050, description="Частота дискретизации")
    
    class Config:
        json_schema_extra = {
            "example": {
                "video_path": "/path/to/video.mp4",
                "output_dir": "/path/to/output",
                "extractor_names": ["video_audio", "mfcc", "mel"],
                "extract_audio": True,
                "device": "auto",
                "sample_rate": 22050
            }
        }


class ProcessResponse(BaseModel):
    """Ответ на запрос обработки."""
    success: bool = Field(..., description="Успешность обработки")
    video_path: str = Field(..., description="Путь к обработанному видео")
    output_dir: str = Field(..., description="Директория с результатами")
    extracted_audio_path: Optional[str] = Field(None, description="Путь к извлеченному аудио")
    processing_time: float = Field(..., description="Время обработки в секундах")
    extractor_results: Dict[str, Dict[str, Any]] = Field(..., description="Результаты экстракторов")
    errors: List[str] = Field(default_factory=list, description="Список ошибок")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "video_path": "/path/to/video.mp4",
                "output_dir": "/path/to/output",
                "extracted_audio_path": "/path/to/output/video_extracted_audio.wav",
                "processing_time": 15.5,
                "extractor_results": {
                    "video_audio": {"success": True, "processing_time": 2.1},
                    "mfcc": {"success": True, "processing_time": 3.2},
                    "mel": {"success": True, "processing_time": 4.1}
                },
                "errors": []
            }
        }


class ExtractorInfo(BaseModel):
    """Информация об экстракторе."""
    name: str = Field(..., description="Название экстрактора")
    version: str = Field(..., description="Версия экстрактора")
    description: str = Field(..., description="Описание экстрактора")
    category: str = Field(..., description="Категория экстрактора")
    device: str = Field(..., description="Устройство экстрактора")
    gpu_required: bool = Field(..., description="Требует ли GPU")
    gpu_preferred: bool = Field(..., description="Предпочитает ли GPU")
    gpu_available: bool = Field(..., description="Доступен ли GPU")
    estimated_duration: float = Field(..., description="Ожидаемое время выполнения")


class ProcessorInfo(BaseModel):
    """Информация о процессоре."""
    device: str = Field(..., description="Устройство процессора")
    max_workers: int = Field(..., description="Максимальное количество воркеров")
    gpu_memory_limit: float = Field(..., description="Лимит памяти GPU")
    sample_rate: int = Field(..., description="Частота дискретизации")
    available_extractors: List[str] = Field(..., description="Доступные экстракторы")
    total_extractors: int = Field(..., description="Общее количество экстракторов")


class HealthResponse(BaseModel):
    """Ответ проверки здоровья."""
    status: str = Field(..., description="Статус сервиса")
    timestamp: str = Field(..., description="Время проверки")
    version: str = Field(..., description="Версия сервиса")
    device: str = Field(..., description="Устройство")
    gpu_available: bool = Field(..., description="Доступность GPU")
    extractors_count: int = Field(..., description="Количество экстракторов")
    uptime: Optional[float] = Field(None, description="Время работы в секундах")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "timestamp": "2023-10-25T10:30:00Z",
                "version": "1.0.0",
                "device": "cuda",
                "gpu_available": True,
                "extractors_count": 3,
                "uptime": 3600.0
            }
        }


class ErrorResponse(BaseModel):
    """Ответ с ошибкой."""
    error: str = Field(..., description="Сообщение об ошибке")
    detail: Optional[str] = Field(None, description="Детали ошибки")
    timestamp: str = Field(..., description="Время ошибки")
    request_id: Optional[str] = Field(None, description="ID запроса")
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "Файл не найден",
                "detail": "Видео файл по указанному пути не существует",
                "timestamp": "2023-10-25T10:30:00Z",
                "request_id": "req-123"
            }
        }


class BatchProcessRequest(BaseModel):
    """Запрос на пакетную обработку."""
    video_paths: List[str] = Field(..., description="Список путей к видео файлам")
    output_base_dir: str = Field(..., description="Базовая директория для результатов")
    extractor_names: Optional[List[str]] = Field(None, description="Список экстракторов")
    extract_audio: bool = Field(True, description="Извлекать ли аудио")
    device: DeviceType = Field(DeviceType.AUTO, description="Устройство")
    sample_rate: int = Field(22050, description="Частота дискретизации")
    max_workers: int = Field(4, description="Максимальное количество воркеров")
    
    @validator('video_paths')
    def validate_video_paths(cls, v):
        if not v:
            raise ValueError('Список путей к видео не может быть пустым')
        if len(v) > 100:
            raise ValueError('Слишком много видео для пакетной обработки (максимум 100)')
        return v


class BatchProcessResponse(BaseModel):
    """Ответ на пакетную обработку."""
    success: bool = Field(..., description="Общий успех обработки")
    total_videos: int = Field(..., description="Общее количество видео")
    successful: int = Field(..., description="Успешно обработано")
    failed: int = Field(..., description="Не удалось обработать")
    processing_time: float = Field(..., description="Общее время обработки")
    results: List[ProcessResponse] = Field(..., description="Результаты обработки")
    errors: List[str] = Field(default_factory=list, description="Общие ошибки")
    
    class Config:
        json_schema_extra = {
            "example": {
                "success": True,
                "total_videos": 3,
                "successful": 2,
                "failed": 1,
                "processing_time": 45.2,
                "results": [],
                "errors": ["Ошибка обработки video3.mp4"]
            }
        }
