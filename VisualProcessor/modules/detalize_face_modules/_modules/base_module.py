"""
Базовый интерфейс для всех модулей обработки фич лица.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional
import logging


class FaceModule(ABC):
    """
    Базовый класс для всех модулей обработки фич лица.
    
    Каждый модуль:
    1. Объявляет свои зависимости через required_inputs()
    2. Обрабатывает данные через process()
    3. Возвращает результаты в виде словаря
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        :param config: конфигурация модуля (параметры модели, пороги, etc.)
        """
        self.config = config or {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self._initialized = False

    @abstractmethod
    def required_inputs(self) -> List[str]:
        """
        Возвращает список ключей, которые должны быть в shared_data для работы модуля.
        
        Примеры:
        - ["coords", "bbox"] - нужны landmarks и bounding box
        - ["coords", "bbox", "roi"] - нужны landmarks, bbox и ROI
        - ["coords", "pose"] - зависит от результатов других модулей
        
        :return: список строк-ключей
        """
        pass

    @abstractmethod
    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Обрабатывает данные и возвращает результаты.
        
        :param data: словарь с данными (coords, bbox, roi, pose, etc.)
        :return: словарь с результатами, ключи должны соответствовать названию модуля
                 например: {"geometry": {...}, "pose": {...}}
        """
        pass

    def initialize(self) -> None:
        """
        Инициализация модуля (загрузка моделей, etc.).
        Вызывается автоматически перед первым использованием.
        """
        if not self._initialized:
            self._do_initialize()
            self._initialized = True

    def _do_initialize(self) -> None:
        """
        Внутренний метод инициализации. Переопределяется в подклассах.
        """
        pass

    @property
    def module_name(self) -> str:
        """
        Имя модуля (используется как ключ для результатов).
        По умолчанию - имя класса в нижнем регистре без суффикса "Module".
        """
        name = self.__class__.__name__
        if name.endswith("Module"):
            name = name[:-6]
        return name.lower()

    def can_process(self, data: Dict[str, Any]) -> bool:
        """
        Проверяет, может ли модуль обработать данные (все зависимости доступны).
        
        :param data: словарь с данными
        :return: True если все required_inputs присутствуют в data
        """
        required = self.required_inputs()
        return all(key in data for key in required)

