"""
Кэш с поддержкой времени жизни (TTL).
"""
import time
from collections import OrderedDict
from typing import Dict, Any, Optional, Tuple, TypeVar, Generic

T = TypeVar('T')


class TTLCache(Generic[T]):
    """
    Кэш с временем жизни (TTL) и LRU стратегией.
    
    Элементы автоматически удаляются после истечения TTL.
    """
    
    def __init__(self, max_size: int = 10, default_ttl: float = 3600.0):
        """
        Инициализация кэша с TTL.
        
        Args:
            max_size: Максимальное количество элементов в кэше.
            default_ttl: Время жизни элементов в секундах (по умолчанию 1 час).
        
        Raises:
            ValueError: Если max_size <= 0 или default_ttl <= 0.
        """
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}")
        if default_ttl <= 0:
            raise ValueError(f"default_ttl must be positive, got {default_ttl}")
        
        self.cache: OrderedDict[Any, Tuple[T, float, float]] = OrderedDict()
        self.max_size = max_size
        self.default_ttl = default_ttl
    
    def get(self, key: Any, ttl: Optional[float] = None) -> Optional[T]:
        """
        Получает значение из кэша.
        
        Args:
            key: Ключ для поиска.
            ttl: Время жизни для обновления (опционально).
        
        Returns:
            Значение или None, если не найдено или истек TTL.
        """
        if key not in self.cache:
            return None
        
        value, created_at, item_ttl = self.cache[key]
        current_time = time.time()
        
        # Проверка истечения TTL
        if current_time - created_at > item_ttl:
            # Элемент истек, удаляем его
            del self.cache[key]
            return None
        
        # Обновляем время доступа (LRU)
        self.cache.move_to_end(key)
        
        # Обновляем TTL, если указан новый
        if ttl is not None:
            self.cache[key] = (value, current_time, ttl)
        
        return value
    
    def put(self, key: Any, value: T, ttl: Optional[float] = None) -> None:
        """
        Сохраняет значение в кэш.
        
        Args:
            key: Ключ для сохранения.
            value: Значение для сохранения.
            ttl: Время жизни в секундах. Если None, используется default_ttl.
        """
        current_time = time.time()
        item_ttl = ttl if ttl is not None else self.default_ttl
        
        # Если ключ уже существует, обновляем и перемещаем в конец
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            # Если кэш переполнен, удаляем самый старый элемент
            if len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)
        
        self.cache[key] = (value, current_time, item_ttl)
    
    def clear(self) -> None:
        """Очищает весь кэш."""
        self.cache.clear()
    
    def cleanup_expired(self) -> int:
        """
        Удаляет все истекшие элементы из кэша.
        
        Returns:
            Количество удаленных элементов.
        """
        current_time = time.time()
        expired_keys = [
            key for key, (_, created_at, ttl) in self.cache.items()
            if current_time - created_at > ttl
        ]
        
        for key in expired_keys:
            del self.cache[key]
        
        return len(expired_keys)
    
    def __len__(self) -> int:
        """Возвращает количество элементов в кэше (только не истекших)."""
        self.cleanup_expired()
        return len(self.cache)
    
    def __contains__(self, key: Any) -> bool:
        """Проверяет наличие ключа в кэше (с учетом TTL)."""
        return self.get(key) is not None
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Возвращает статистику кэша.
        
        Returns:
            Словарь со статистикой.
        """
        self.cleanup_expired()
        current_time = time.time()
        
        total_items = len(self.cache)
        expired_count = 0
        min_ttl_remaining = float('inf')
        max_ttl_remaining = 0.0
        
        for _, (_, created_at, ttl) in self.cache.items():
            remaining = ttl - (current_time - created_at)
            if remaining < 0:
                expired_count += 1
            else:
                min_ttl_remaining = min(min_ttl_remaining, remaining)
                max_ttl_remaining = max(max_ttl_remaining, remaining)
        
        return {
            "total_items": total_items,
            "expired_items": expired_count,
            "max_size": self.max_size,
            "default_ttl": self.default_ttl,
            "min_ttl_remaining": min_ttl_remaining if min_ttl_remaining != float('inf') else 0,
            "max_ttl_remaining": max_ttl_remaining
        }


class FaceScanCacheWithTTL(TTLCache[Tuple[list, int]]):
    """
    Кэш для результатов сканирования лиц с поддержкой TTL.
    
    Наследуется от TTLCache и специализирован для хранения
    результатов сканирования лиц (timeline, scanned_count).
    """
    
    def __init__(self, max_size: int = 10, ttl_seconds: float = 1800.0):
        """
        Инициализация кэша сканирования лиц.
        
        Args:
            max_size: Максимальное количество кэшированных результатов.
            ttl_seconds: Время жизни результатов в секундах (по умолчанию 30 минут).
        """
        super().__init__(max_size=max_size, default_ttl=ttl_seconds)
    
    def get_scan_result(self, scan_stride: int, detect_thr: float) -> Optional[Tuple[list, int]]:
        """
        Получает результат сканирования из кэша.
        
        Args:
            scan_stride: Шаг сканирования.
            detect_thr: Порог детекции.
        
        Returns:
            Кортеж (timeline, scanned_count) или None.
        """
        key = (scan_stride, detect_thr)
        return self.get(key)
    
    def put_scan_result(
        self,
        scan_stride: int,
        detect_thr: float,
        timeline: list,
        scanned_count: int,
        ttl: Optional[float] = None
    ) -> None:
        """
        Сохраняет результат сканирования в кэш.
        
        Args:
            scan_stride: Шаг сканирования.
            detect_thr: Порог детекции.
            timeline: Список индексов кадров с лицами.
            scanned_count: Количество просканированных кадров.
            ttl: Время жизни в секундах (опционально).
        """
        key = (scan_stride, detect_thr)
        self.put(key, (timeline, scanned_count), ttl=ttl)

