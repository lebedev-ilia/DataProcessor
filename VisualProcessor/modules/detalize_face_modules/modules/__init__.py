"""
Модули для извлечения различных типов фич лица.
"""

from .base_module import FaceModule
from .geometry_module import GeometryModule
from .pose_module import PoseModule
from .quality_module import QualityModule
from .lighting_module import LightingModule
from .skin_module import SkinModule
from .accessories_module import AccessoriesModule
from .eyes_module import EyesModule
from .motion_module import MotionModule
from .structure_module import StructureModule
from .professional_module import ProfessionalModule
from .lip_reading_module import LipReadingModule
from .face_3d_module import Face3DModule

# Registry для автоматической загрузки модулей
MODULE_REGISTRY = {
    "geometry": GeometryModule,
    "pose": PoseModule,
    "quality": QualityModule,
    "lighting": LightingModule,
    "skin": SkinModule,
    "accessories": AccessoriesModule,
    "eyes": EyesModule,
    "motion": MotionModule,
    "structure": StructureModule,
    "professional": ProfessionalModule,
    "lip_reading": LipReadingModule,
    "face_3d": Face3DModule,
}

__all__ = [
    "FaceModule",
    "GeometryModule",
    "PoseModule",
    "QualityModule",
    "LightingModule",
    "SkinModule",
    "AccessoriesModule",
    "EyesModule",
    "MotionModule",
    "StructureModule",
    "ProfessionalModule",
    "LipReadingModule",
    "Face3DModule",
    "MODULE_REGISTRY",
]

