"""
Настройки конфигурации для AudioProcessor.
"""
import os
from typing import List, Optional
from pydantic import BaseSettings, Field


class Settings(BaseSettings):
    """Настройки приложения."""
    
    # === API ===
    host: str = Field("0.0.0.0", env="HOST")
    port: int = Field(8000, env="PORT")
    reload: bool = Field(False, env="RELOAD")
    debug: bool = Field(False, env="DEBUG")
    
    # === Обработка ===
    device: str = Field("auto", env="DEVICE")  # auto, cpu, cuda
    max_workers: int = Field(4, env="MAX_WORKERS")
    gpu_memory_limit: float = Field(0.8, env="GPU_MEMORY_LIMIT")
    sample_rate: int = Field(22050, env="SAMPLE_RATE")
    
    # === Логирование ===
    log_level: str = Field("INFO", env="LOG_LEVEL")
    log_format: str = Field("json", env="LOG_FORMAT")
    
    # === Безопасность ===
    cors_origins: List[str] = Field(
        default_factory=lambda: ["http://localhost:3000", "http://localhost:8000"],
        env="CORS_ORIGINS"
    )
    
    # === Файловая система ===
    max_file_size: int = Field(100 * 1024 * 1024, env="MAX_FILE_SIZE")  # 100MB
    temp_dir: str = Field("/tmp/audioprocessor", env="TEMP_DIR")
    
    # === GPU настройки ===
    cuda_visible_devices: str = Field("0", env="CUDA_VISIBLE_DEVICES")
    gpu_batch_size: int = Field(8, env="GPU_BATCH_SIZE")
    
    class Config:
        env_file = ".env"
        case_sensitive = False


# Глобальный экземпляр настроек
settings = Settings()


def get_settings() -> Settings:
    """Получение настроек."""
    return settings
