"""
Модульная система для извлечения детальных фич лица.

Архитектура:
- DetalizeFaceExtractor: главный оркестратор (рефакторинг)
- FaceModule: базовый интерфейс для всех модулей
- modules/: конкретные реализации модулей (Geometry, Pose, Quality, etc.)
- utils/: вспомогательные утилиты (landmarks, bbox, validation)
"""

from .modules.base_module import FaceModule

__all__ = ["FaceModule"]

