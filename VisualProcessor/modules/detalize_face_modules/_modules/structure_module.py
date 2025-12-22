"""
Модуль для извлечения структурных фич лица.
"""

from typing import Dict, List, Any, Optional
import numpy as np
import hashlib

from _modules.base_module import FaceModule
from _utils.compression_utils import simple_landmark_projection


class StructureModule(FaceModule):
    """
    Модуль для извлечения структурных фич лица.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.face_mesh_proj_dim = self.config.get("face_mesh_proj_dim", 32)  # Сжатие face_mesh_vector
        self.identity_proj_dim = self.config.get("identity_proj_dim", 16)  # Сжатие identity_shape_vector
        self.expression_proj_dim = self.config.get("expression_proj_dim", 8)  # Сжатие expression_vector
        self.use_privacy_preserving = self.config.get("use_privacy_preserving", True)  # Privacy-preserving для identity

    def required_inputs(self) -> List[str]:
        """Требуются coords и pose."""
        return ["coords", "pose"]
    
    def _privacy_preserving_hash(self, vector: np.ndarray, n_bits: int = 64) -> str:
        """
        Применяет privacy-preserving hashing к identity-вектору.
        Использует SHA-256 для создания необратимого хеша.
        
        :param vector: вектор для хеширования
        :param n_bits: количество бит в хеше (по умолчанию 64)
        :return: хеш-строка
        """
        # Нормализуем вектор перед хешированием
        vector_normalized = (vector - np.mean(vector)) / (np.std(vector) + 1e-6)
        vector_bytes = vector_normalized.tobytes()
        
        # Создаем хеш
        hash_obj = hashlib.sha256(vector_bytes)
        hash_hex = hash_obj.hexdigest()
        
        # Возвращаем первые n_bits/4 символов (каждый символ = 4 бита)
        return hash_hex[:n_bits // 4]

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Обрабатывает данные и возвращает структурные фичи."""
        coords = data["coords"]
        pose = data["pose"]

        normalized = coords[:, :3].copy()
        normalized[:, 0] -= np.mean(normalized[:, 0])
        normalized[:, 1] -= np.mean(normalized[:, 1])
        normalized[:, :2] /= np.max(np.ptp(normalized[:, :2], axis=0) + 1e-6)

        # Сжимаем векторы
        face_mesh_vector_raw = normalized[:100, :].flatten()
        face_mesh_vector_compressed = simple_landmark_projection(
            face_mesh_vector_raw,
            target_dim=self.face_mesh_proj_dim
        )
        face_mesh_vector = face_mesh_vector_compressed.tolist()

        # Identity shape vector (privacy-preserving)
        identity_shape_vector_raw = normalized[:50, :3].flatten()
        identity_shape_vector_compressed = simple_landmark_projection(
            identity_shape_vector_raw,
            target_dim=self.identity_proj_dim
        )
        
        if self.use_privacy_preserving:
            # Применяем privacy-preserving hashing для identity
            identity_hash = self._privacy_preserving_hash(identity_shape_vector_compressed)
            # Возвращаем только хеш (необратимый) вместо raw вектора
            identity_shape_vector = {"hash": identity_hash, "dim": self.identity_proj_dim}
        else:
            identity_shape_vector = identity_shape_vector_compressed.tolist()

        # Expression vector (можно хранить без хеширования, т.к. не идентифицирует личность)
        expression_vector_raw = (normalized[50:100, :3] - normalized[:50, :3]).flatten()
        expression_vector_compressed = simple_landmark_projection(
            expression_vector_raw,
            target_dim=self.expression_proj_dim
        )
        expression_vector = expression_vector_compressed.tolist()
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
        if isinstance(identity_shape_vector, dict):
            identity_params_count = identity_shape_vector.get("dim", self.identity_proj_dim)
        else:
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

