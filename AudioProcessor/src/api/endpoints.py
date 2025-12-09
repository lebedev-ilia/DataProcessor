"""
API эндпоинты для AudioProcessor.
"""
import os
import logging
from typing import List
from fastapi import APIRouter, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse

from ..schemas.models import (
    ProcessRequest, ProcessResponse, ProcessorInfo, 
    BatchProcessRequest, BatchProcessResponse, ExtractorInfo
)

logger = logging.getLogger(__name__)

# Создание роутера
router = APIRouter(prefix="/api/v1", tags=["audio-processor"])


@router.post("/process", response_model=ProcessResponse)
async def process_video(request: ProcessRequest):
    """
    Обработка одного видео файла.
    
    Args:
        request: Запрос на обработку видео
        
    Returns:
        Результат обработки видео
    """
    try:
        # Проверяем существование видео файла
        if not os.path.exists(request.video_path):
            raise HTTPException(
                status_code=404, 
                detail=f"Видео файл не найден: {request.video_path}"
            )
        
        # Создаем выходную директорию
        os.makedirs(request.output_dir, exist_ok=True)
        
        # Импортируем процессор из глобального контекста
        from .main import processor
        
        if processor is None:
            raise HTTPException(
                status_code=503, 
                detail="Процессор не инициализирован"
            )
        
        # Обрабатываем видео
        result = processor.process_video(
            video_path=request.video_path,
            output_dir=request.output_dir,
            extractor_names=request.extractor_names,
            extract_audio=request.extract_audio
        )
        
        # Конвертируем результат в Pydantic модель
        response = ProcessResponse(
            success=result["success"],
            video_path=result["video_path"],
            output_dir=result["output_dir"],
            extracted_audio_path=result["extracted_audio_path"],
            processing_time=result["processing_time"],
            extractor_results=result["extractor_results"],
            errors=result["errors"]
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка обработки видео: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка обработки: {str(e)}")


@router.post("/batch", response_model=BatchProcessResponse)
async def process_batch(request: BatchProcessRequest):
    """
    Пакетная обработка видео файлов.
    
    Args:
        request: Запрос на пакетную обработку
        
    Returns:
        Результат пакетной обработки
    """
    try:
        # Импортируем процессор из глобального контекста
        from .main import processor
        
        if processor is None:
            raise HTTPException(
                status_code=503, 
                detail="Процессор не инициализирован"
            )
        
        # Проверяем существование всех видео файлов
        missing_files = []
        for video_path in request.video_paths:
            if not os.path.exists(video_path):
                missing_files.append(video_path)
        
        if missing_files:
            raise HTTPException(
                status_code=404,
                detail=f"Видео файлы не найдены: {missing_files}"
            )
        
        # Создаем базовую директорию
        os.makedirs(request.output_base_dir, exist_ok=True)
        
        # Обрабатываем каждое видео
        results = []
        successful = 0
        failed = 0
        errors = []
        
        for i, video_path in enumerate(request.video_paths):
            try:
                # Создаем отдельную директорию для каждого видео
                video_name = os.path.splitext(os.path.basename(video_path))[0]
                video_output_dir = os.path.join(request.output_base_dir, video_name)
                
                # Обрабатываем видео
                result = processor.process_video(
                    video_path=video_path,
                    output_dir=video_output_dir,
                    extractor_names=request.extractor_names,
                    extract_audio=request.extract_audio
                )
                
                # Конвертируем в Pydantic модель
                process_response = ProcessResponse(
                    success=result["success"],
                    video_path=result["video_path"],
                    output_dir=result["output_dir"],
                    extracted_audio_path=result["extracted_audio_path"],
                    processing_time=result["processing_time"],
                    extractor_results=result["extractor_results"],
                    errors=result["errors"]
                )
                
                results.append(process_response)
                
                if result["success"]:
                    successful += 1
                else:
                    failed += 1
                    errors.extend(result["errors"])
                
            except Exception as e:
                error_msg = f"Ошибка обработки {video_path}: {str(e)}"
                logger.error(error_msg)
                errors.append(error_msg)
                failed += 1
                
                # Создаем результат с ошибкой
                error_result = ProcessResponse(
                    success=False,
                    video_path=video_path,
                    output_dir="",
                    extracted_audio_path=None,
                    processing_time=0.0,
                    extractor_results={},
                    errors=[error_msg]
                )
                results.append(error_result)
        
        # Создаем общий результат
        response = BatchProcessResponse(
            success=failed == 0,
            total_videos=len(request.video_paths),
            successful=successful,
            failed=failed,
            processing_time=sum(r.processing_time for r in results),
            results=results,
            errors=errors
        )
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка пакетной обработки: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка пакетной обработки: {str(e)}")


@router.get("/extractors", response_model=List[ExtractorInfo])
async def get_extractors():
    """
    Получение списка доступных экстракторов.
    
    Returns:
        Список доступных экстракторов
    """
    try:
        from .main import processor
        
        if processor is None:
            raise HTTPException(
                status_code=503, 
                detail="Процессор не инициализирован"
            )
        
        # Получаем информацию об экстракторах
        extractors_info = processor.get_available_extractors()
        
        # Конвертируем в Pydantic модели
        extractors = []
        for name, info in extractors_info.items():
            extractor = ExtractorInfo(
                name=info["name"],
                version=info["version"],
                description=info["description"],
                category=info["category"],
                device=info["device"],
                gpu_required=info["gpu_required"],
                gpu_preferred=info["gpu_preferred"],
                gpu_available=info["gpu_available"],
                estimated_duration=info["estimated_duration"]
            )
            extractors.append(extractor)
        
        return extractors
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка получения экстракторов: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка получения экстракторов: {str(e)}")


@router.get("/processor", response_model=ProcessorInfo)
async def get_processor_info():
    """
    Получение информации о процессоре.
    
    Returns:
        Информация о процессоре
    """
    try:
        from .main import processor
        
        if processor is None:
            raise HTTPException(
                status_code=503, 
                detail="Процессор не инициализирован"
            )
        
        # Получаем информацию о процессоре
        info = processor.get_processor_info()
        
        # Конвертируем в Pydantic модель
        processor_info = ProcessorInfo(
            device=info["device"],
            max_workers=info["max_workers"],
            gpu_memory_limit=info["gpu_memory_limit"],
            sample_rate=info["sample_rate"],
            available_extractors=info["available_extractors"],
            total_extractors=info["total_extractors"]
        )
        
        return processor_info
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка получения информации о процессоре: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка получения информации: {str(e)}")


@router.get("/download/{file_path:path}")
async def download_file(file_path: str):
    """
    Скачивание файла результата.
    
    Args:
        file_path: Путь к файлу относительно output_dir
        
    Returns:
        Файл для скачивания
    """
    try:
        # Проверяем существование файла
        if not os.path.exists(file_path):
            raise HTTPException(status_code=404, detail="Файл не найден")
        
        # Проверяем, что файл находится в разрешенной директории
        # (базовая проверка безопасности)
        if ".." in file_path or file_path.startswith("/"):
            raise HTTPException(status_code=403, detail="Доступ запрещен")
        
        # Возвращаем файл
        return FileResponse(
            path=file_path,
            filename=os.path.basename(file_path),
            media_type='application/octet-stream'
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Ошибка скачивания файла: {e}")
        raise HTTPException(status_code=500, detail=f"Ошибка скачивания: {str(e)}")
