"""
Модуль для управления памятью при обработке видео.
"""
import gc
import torch
from contextlib import contextmanager
from functools import wraps
from typing import Callable, Any
import sys


@contextmanager
def memory_context():
    """
    Контекстный менеджер для управления памятью.
    Автоматически очищает память при выходе из контекста.
    """
    try:
        yield
    finally:
        cleanup_memory()


def cleanup_memory():
    """Полная очистка памяти."""
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Принудительный сбор мусора
    for i in range(3):
        gc.collect()


def memory_cleanup(func: Callable) -> Callable:
    """
    Декоратор для автоматической очистки памяти после выполнения функции.
    
    Usage:
        @memory_cleanup
        def my_function():
            # код функции
            pass
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            cleanup_memory()
    
    return wrapper


def calculate_optimal_batch_size(
    frame_shape: tuple,
    available_memory_mb: float,
    model_memory_estimate_mb: float = 500
) -> int:
    """
    Динамический расчет размера батча на основе доступной памяти.
    
    Args:
        frame_shape: (H, W, C) - размер одного кадра
        available_memory_mb: доступная память в MB
        model_memory_estimate_mb: оценка памяти, занимаемой моделью в MB
    
    Returns:
        Оптимальный размер батча (от 1 до 64)
    """
    if frame_shape is None or len(frame_shape) < 3:
        return 16  # значение по умолчанию
    
    H, W, C = frame_shape[:3]
    # float32 = 4 байта на пиксель
    frame_memory_mb = (H * W * C * 4) / (1024 ** 2)
    
    # Вычисляем максимальное количество кадров
    available_for_frames = available_memory_mb - model_memory_estimate_mb
    if available_for_frames <= 0:
        return 1
    
    max_frames = int((available_for_frames * 0.7) / frame_memory_mb)  # 70% от доступной памяти
    
    # Ограничиваем размер батча
    return min(64, max(1, max_frames))

