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

