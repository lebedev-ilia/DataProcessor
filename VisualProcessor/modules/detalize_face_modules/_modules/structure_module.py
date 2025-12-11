"""
Модуль для извлечения структурных фич лица.
"""

from typing import Dict, List, Any, Optional
import numpy as np

from _modules.base_module import FaceModule


class StructureModule(FaceModule):
    """
    Модуль для извлечения структурных фич лица.
    """

    def required_inputs(self) -> List[str]:
        """Требуются coords и pose."""
        return ["coords", "pose"]

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Обрабатывает данные и возвращает структурные фичи."""
        coords = data["coords"]
        pose = data["pose"]

        normalized = coords[:, :3].copy()
        normalized[:, 0] -= np.mean(normalized[:, 0])
        normalized[:, 1] -= np.mean(normalized[:, 1])
        normalized[:, :2] /= np.max(np.ptp(normalized[:, :2], axis=0) + 1e-6)

        face_mesh_vector = normalized[:100, :].flatten().tolist()
        identity_shape_vector = normalized[:50, :3].flatten().tolist()
        expression_vector = (normalized[50:100, :3] - normalized[:50, :3]).flatten().tolist()
        jaw_pose_vector = [pose["yaw"], pose["pitch"], pose["roll"]]
        eye_pose_vector = [pose["pitch"], pose["roll"]]
        mouth_shape_params = [
            float(np.max(normalized[:, 0]) - np.min(normalized[:, 0])),
            float(np.max(normalized[:, 1]) - np.min(normalized[:, 1])),
        ]
        symmetry = 1.0 - float(
            np.mean(np.abs(normalized[:, 0] + normalized[::-1, 0])) / (np.max(np.abs(normalized[:, 0])) + 1e-6)
        )
        uniqueness = float(np.std(normalized[:, :2]))

        # Более детальные параметры для совместимости с 3DMM
        # Identity shape vector (базовая форма лица)
        identity_params_count = len(identity_shape_vector)
        
        # Expression vector (параметры выражения)
        expression_params_count = len(expression_vector)
        
        # Дополнительные структурные метрики
        face_width = float(np.max(normalized[:, 0]) - np.min(normalized[:, 0]))
        face_height = float(np.max(normalized[:, 1]) - np.min(normalized[:, 1]))
        face_depth = float(np.max(normalized[:, 2]) - np.min(normalized[:, 2])) if normalized.shape[1] > 2 else 0.0
        face_aspect_ratio = face_width / max(face_height, 1e-6)
        
        return {
            "structure": {
                "face_mesh_vector": face_mesh_vector,
                "identity_shape_vector": identity_shape_vector,
                "expression_vector": expression_vector,
                "jaw_pose_vector": jaw_pose_vector,
                "eye_pose_vector": eye_pose_vector,
                "mouth_shape_params": mouth_shape_params,
                "face_symmetry_score": float(symmetry),
                "face_uniqueness_score": float(uniqueness),
                "identity_params_count": identity_params_count,
                "expression_params_count": expression_params_count,
                "face_dimensions": {
                    "width": face_width,
                    "height": face_height,
                    "depth": face_depth,
                    "aspect_ratio": face_aspect_ratio,
                },
            }
        }

