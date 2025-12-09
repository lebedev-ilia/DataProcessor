"""
Основное FastAPI приложение для AudioProcessor.
"""
import os
import time
import logging
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager

from ..core.main_processor import MainProcessor
from ..schemas.models import (
    ProcessRequest, ProcessResponse, ProcessorInfo, HealthResponse, 
    ErrorResponse, BatchProcessRequest, BatchProcessResponse
)
from .endpoints import router

from ..utils.silence import silence_all
silence_all()
logger = logging.getLogger(__name__)

# Глобальная переменная для процессора
processor: MainProcessor = None
startup_time = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Управление жизненным циклом приложения."""
    global processor, startup_time
    
    # Startup
    startup_time = time.time()
    # Silenced
    
    try:
        # Инициализируем процессор
        device = os.getenv("DEVICE", "auto")
        max_workers = int(os.getenv("MAX_WORKERS", "4"))
        gpu_memory_limit = float(os.getenv("GPU_MEMORY_LIMIT", "0.8"))
        sample_rate = int(os.getenv("SAMPLE_RATE", "22050"))
        
        processor = MainProcessor(
            device=device,
            max_workers=max_workers,
            gpu_memory_limit=gpu_memory_limit,
            sample_rate=sample_rate
        )
        
        # Silenced
        
    except Exception as e:
        # Silenced
        raise
    
    yield
    
    # Shutdown
    # Silenced


# Создание FastAPI приложения
app = FastAPI(
    title="AudioProcessor",
    description="Система для извлечения аудио из видео и применения экстракторов фичей с поддержкой GPU",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
)

# Добавление CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # В продакшене указать конкретные домены
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Подключение роутера
app.include_router(router)


@app.get("/", response_model=dict)
async def root():
    """Корневой эндпоинт."""
    return {
        "service": "AudioProcessor",
        "version": "1.0.0",
        "status": "running",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Проверка здоровья сервиса."""
    try:
        if processor is None:
            raise HTTPException(status_code=503, detail="Процессор не инициализирован")
        
        # Получаем информацию о процессоре
        processor_info = processor.get_processor_info()
        
        # Вычисляем время работы
        uptime = time.time() - startup_time if startup_time else 0
        
        return HealthResponse(
            status="healthy",
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            version="1.0.0",
            device=processor_info["device"],
            gpu_available=processor_info["device"] == "cuda",
            extractors_count=processor_info["total_extractors"],
            uptime=uptime
        )
        
    except Exception as e:
        # Silenced
        raise HTTPException(status_code=503, detail="Сервис недоступен")


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Обработчик HTTP исключений."""
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            error=exc.detail,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        ).dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request: Request, exc: Exception):
    """Обработчик общих исключений."""
    # Silenced
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            error="Внутренняя ошибка сервера",
            detail=str(exc),
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        ).dict()
    )


# Middleware для логирования запросов
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Логирование HTTP запросов."""
    start_time = time.time()
    
    # Обрабатываем запрос
    response = await call_next(request)
    
    # Вычисляем время обработки
    process_time = time.time() - start_time
    
    # Silenced
    
    return response


if __name__ == "__main__":
    import uvicorn
    
    # Параметры запуска
    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))
    reload = os.getenv("RELOAD", "false").lower() == "true"
    
    # Silenced
    
    uvicorn.run(
        "main:app",
        host=host,
        port=port,
        reload=reload,
        log_level="critical"
    )
