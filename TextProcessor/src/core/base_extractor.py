from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict


class BaseExtractor(ABC):
    """
    Базовый интерфейс для всех экстракторов признаков.
    Экстрактор принимает входной документ и возвращает словарь признаков.
    """

    @property
    def name(self) -> str:
        return self.__class__.__name__

    @abstractmethod
    def extract(self, doc: Any) -> Dict[str, Any]:
        """
        Выполнить извлечение признаков из входного документа.

        :param doc: входной объект документа (см. schemas.models.VideoDocument)
        :return: словарь признаков {feature_name: value}
        """
        raise NotImplementedError
