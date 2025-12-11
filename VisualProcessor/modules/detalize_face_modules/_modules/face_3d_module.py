"""
Модуль для 3D face reconstruction (упрощенная версия).
Извлекает 3DMM-подобные параметры из landmarks без использования полных DECA моделей.
"""

from typing import Dict, List, Any, Optional
import numpy as np

from _modules.base_module import FaceModule


class Face3DModule(FaceModule):
    """
    Модуль для извлечения 3D face reconstruction features.
    
    Это упрощенная версия, которая:
    - Использует 3D landmarks от MediaPipe для создания базовой 3D mesh
    - Применяет PCA для параметризации формы лица (identity, expression)
    - Извлекает параметры, аналогичные 3DMM (100-300 параметров)
    - Не требует отдельной установки DECA/EMOCA моделей
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._pca_components = None
        self._mean_shape = None
        self.use_pca = self.config.get("use_pca", True)
        self.n_identity_params = self.config.get("n_identity_params", 100)
        self.n_expression_params = self.config.get("n_expression_params", 50)

    def required_inputs(self) -> List[str]:
        """Требуются coords и pose."""
        return ["coords", "pose"]

    def _do_initialize(self) -> None:
        """Инициализация PCA компонент (опционально, для будущего улучшения)."""
        # В упрощенной версии PCA не инициализируется заранее
        # В будущем можно загрузить предобученные компоненты
        pass

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Обрабатывает данные и возвращает 3D face reconstruction features."""
        coords = data["coords"]  # (N, 3) - 3D landmarks
        pose = data.get("pose", {})

        if coords is None or coords.size == 0:
            return {"face_3d": {}}

        n_landmarks = len(coords)

        # Normalize coordinates: центрируем и масштабируем
        mean_coords = np.mean(coords[:, :3], axis=0)
        centered_coords = coords[:, :3] - mean_coords
        scale = np.std(centered_coords) + 1e-6
        normalized_coords = centered_coords / scale

        # Face mesh vector (вектор 3D точек)
        max_landmarks_for_mesh = min(n_landmarks, 200)
        face_mesh_vector_3d = normalized_coords[:max_landmarks_for_mesh, :3].flatten()
        if len(face_mesh_vector_3d) > 300:
            step = len(face_mesh_vector_3d) // 300
            face_mesh_vector_3d = face_mesh_vector_3d[::step][:300]
        face_mesh_vector = face_mesh_vector_3d.tolist()

        # Identity shape vector
        identity_landmarks = min(100, n_landmarks)
        identity_coords = normalized_coords[:identity_landmarks, :3]

        # Базовые статистики
        identity_shape_vector = [
            float(np.mean(identity_coords[:, 0])),
            float(np.std(identity_coords[:, 0])),
            float(np.mean(identity_coords[:, 1])),
            float(np.std(identity_coords[:, 1])),
            float(np.mean(identity_coords[:, 2])),
            float(np.std(identity_coords[:, 2])),
        ]

        # PCA-like разложение
        if identity_coords.shape[0] > 3:
            cov_matrix = np.cov(identity_coords.T)
            eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
            n_pcs = min(self.n_identity_params - 6, len(eigenvalues))
            if n_pcs > 0:
                sorted_idx = np.argsort(eigenvalues)[::-1]
                top_eigenvectors = eigenvectors[:, sorted_idx[:n_pcs]]
                projected = identity_coords @ top_eigenvectors  # исправлено
                identity_shape_vector.extend([float(x) for x in projected.flatten()[:n_pcs * 3]])

        while len(identity_shape_vector) < self.n_identity_params:
            identity_shape_vector.append(0.0)
        identity_shape_vector = identity_shape_vector[:self.n_identity_params]

        # Expression vector
        expression_landmarks = min(150, n_landmarks)
        expression_coords = normalized_coords[:expression_landmarks, :3]
        neutral_coords = np.mean(expression_coords, axis=0)
        expression_deviations = expression_coords - neutral_coords

        expression_vector = [
            float(np.mean(expression_deviations[:, 0])),
            float(np.std(expression_deviations[:, 0])),
            float(np.mean(expression_deviations[:, 1])),
            float(np.std(expression_deviations[:, 1])),
            float(np.mean(expression_deviations[:, 2])),
            float(np.std(expression_deviations[:, 2])),
        ]

        if expression_deviations.shape[0] > 3:
            expr_cov = np.cov(expression_deviations.T)
            expr_eigenvalues, expr_eigenvectors = np.linalg.eigh(expr_cov)
            n_expr_pcs = min(self.n_expression_params - 6, len(expr_eigenvalues))
            if n_expr_pcs > 0:
                sorted_idx = np.argsort(expr_eigenvalues)[::-1]
                top_eigenvectors = expr_eigenvectors[:, sorted_idx[:n_expr_pcs]]
                projected = expression_deviations @ top_eigenvectors  # исправлено
                expression_vector.extend([float(x) for x in projected.flatten()[:n_expr_pcs * 3]])

        while len(expression_vector) < self.n_expression_params:
            expression_vector.append(0.0)
        expression_vector = expression_vector[:self.n_expression_params]

        # Jaw pose vector
        yaw = pose.get("yaw", 0.0)
        pitch = pose.get("pitch", 0.0)
        roll = pose.get("roll", 0.0)
        jaw_pose_vector = [float(yaw), float(pitch), float(roll)]

        # Eye pose vector
        eye_pose_vector = [float(pitch), float(roll)]

        # Mouth shape params
        mouth_landmarks_indices = [13, 14, 61, 291]
        mouth_coords = [coords[idx, :3] for idx in mouth_landmarks_indices if idx < len(coords)]
        if len(mouth_coords) >= 2:
            mouth_coords = np.array(mouth_coords)
            mouth_width = float(np.linalg.norm(mouth_coords[2] - mouth_coords[3]) if len(mouth_coords) >= 4 else 0.0)
            mouth_height = float(np.linalg.norm(mouth_coords[0] - mouth_coords[1]))
            mouth_depth = float(np.mean(mouth_coords[:, 2]))
            mouth_shape_params = [
                mouth_width,
                mouth_height,
                mouth_depth,
                float(mouth_width / max(mouth_height, 1e-6)),
            ]
        else:
            mouth_shape_params = [0.0, 0.0, 0.0, 0.0]

        # Face symmetry score
        face_center_x = np.mean(normalized_coords[:, 0])
        left_half = normalized_coords[normalized_coords[:, 0] < face_center_x]
        right_half = normalized_coords[normalized_coords[:, 0] >= face_center_x]
        if len(left_half) > 0 and len(right_half) > 0:
            right_half_mirrored = right_half.copy()
            right_half_mirrored[:, 0] = face_center_x - (right_half_mirrored[:, 0] - face_center_x)
            min_len = min(len(left_half), len(right_half_mirrored))
            if min_len > 0:
                left_sampled = left_half[:min_len]
                right_sampled = right_half_mirrored[:min_len]
                symmetry_error = np.mean(np.abs(left_sampled - right_sampled))
                symmetry_score = float(1.0 / (1.0 + symmetry_error))
            else:
                symmetry_score = 0.5
        else:
            symmetry_score = 0.5

        # Face uniqueness score
        uniqueness_score = float(np.std(normalized_coords.flatten()))

        return {
            "face_3d": {
                "face_mesh_vector": face_mesh_vector,
                "identity_shape_vector": identity_shape_vector,
                "expression_vector": expression_vector,
                "jaw_pose_vector": jaw_pose_vector,
                "eye_pose_vector": eye_pose_vector,
                "mouth_shape_params": mouth_shape_params,
                "face_symmetry_score": symmetry_score,
                "face_uniqueness_score": uniqueness_score,
                "mesh_num_vertices": len(face_mesh_vector) // 3,
                "identity_params_count": len(identity_shape_vector),
                "expression_params_count": len(expression_vector),
            }
        }

