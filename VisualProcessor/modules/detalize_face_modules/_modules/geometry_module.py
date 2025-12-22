"""
Модуль для извлечения геометрических фич лица.
"""

from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
import numpy as np

from _modules.base_module import FaceModule
from _utils.landmarks_utils import LANDMARKS
from _utils.compression_utils import simple_landmark_projection


def _safe_distance(coords, i, j):
    """Безопасное вычисление расстояния между landmarks."""
    if i >= len(coords) or j >= len(coords):
        return 0.0
    return float(np.linalg.norm(coords[i][:2] - coords[j][:2]))


def _roll_angle(coords: np.ndarray) -> float:
    """Оценка поворота головы вокруг Z-оси (roll) в градусах."""
    left = coords[LANDMARKS["left_eye_outer"], :2]
    right = coords[LANDMARKS["right_eye_outer"], :2]
    return float(np.degrees(np.arctan2(right[1] - left[1], right[0] - left[0])))


class GeometryModule(FaceModule):
    """
    Модуль для извлечения геометрических фич лица.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.history_size = self.config.get("history_size", 30)  # 30 кадров ≈ 1-1.5 сек при 25-30 fps
        self._aspect_history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.history_size)
        )
        self.face_shape_proj_dim = self.config.get("face_shape_proj_dim", 16)  # Сжатие face_shape_vector

    def required_inputs(self) -> List[str]:
        """Требуются coords, bbox и frame_shape."""
        return ["coords", "bbox", "frame_shape", "face_idx"]

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Обрабатывает данные и возвращает геометрические фичи."""
        coords = data["coords"]
        bbox = data["bbox"]
        frame_shape = data["frame_shape"]
        face_idx = data.get("face_idx", 0)
        detection_confidence = data.get("detection_confidence", 1.0)

        frame_h, frame_w = frame_shape[:2]

        # ----- BBox geometry -----
        width = float(bbox[2] - bbox[0])
        height = float(bbox[3] - bbox[1])
        area = width * height
        ratio = width / max(height, 1e-6)
        bbox_diagonal = float(np.sqrt(width**2 + height**2))

        cx = float(bbox[0] + width / 2)
        cy = float(bbox[1] + height / 2)

        frame_cx = frame_w / 2
        frame_cy = frame_h / 2

        # Нормализованные координаты центра (0-1)
        face_center_x_norm = float(cx / max(frame_w, 1e-6))
        face_center_y_norm = float(cy / max(frame_h, 1e-6))
        
        dist_center = float(
            np.linalg.norm([cx - frame_cx, cy - frame_cy]) / max(frame_w, frame_h)
        )

        # ----- Stability -----
        self._aspect_history[face_idx].append(ratio)
        history = self._aspect_history[face_idx]

        stability = (
            float(1.0 - (np.std(history) / (np.mean(history) + 1e-5)))
            if len(history) > 3
            else 0.0
        )

        # ----- Morphometrics (нормализованные по bbox diagonal) -----
        left_cheek = 234
        right_cheek = 454
        left_jaw = 172
        right_jaw = 397
        left_zygom = 127
        right_zygom = 356
        left_brow = 70
        right_brow = 301

        forehead_point = (coords[left_brow] + coords[right_brow]) / 2
        nose_tip = coords[1]

        jaw_width = _safe_distance(coords, left_jaw, right_jaw)
        cheekbone_width = _safe_distance(coords, left_zygom, right_zygom)
        cheek_width = _safe_distance(coords, left_cheek, right_cheek)
        forehead_height = float(np.linalg.norm(forehead_point[:2] - nose_tip[:2]))

        # Нормализуем по bbox diagonal
        jaw_width_norm = float(jaw_width / max(bbox_diagonal, 1e-6))
        cheekbone_width_norm = float(cheekbone_width / max(bbox_diagonal, 1e-6))
        cheek_width_norm = float(cheek_width / max(bbox_diagonal, 1e-6))
        forehead_height_norm = float(forehead_height / max(bbox_diagonal, 1e-6))

        # ----- Face shape vector (сжатый) -----
        FACE_OVAL_LANDMARKS = [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            397, 365, 379, 378, 400, 377, 152, 176, 149, 150, 136,
            172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
        ]

        face_shape_vector_raw = coords[FACE_OVAL_LANDMARKS, :2].flatten()
        # Сжимаем до 16 dims
        face_shape_vector_compressed = simple_landmark_projection(
            face_shape_vector_raw, 
            target_dim=self.face_shape_proj_dim
        )
        face_shape_vector = face_shape_vector_compressed.tolist()

        # ----- Landmark stability (fraction of missing/low-conf points) -----
        # Используем detection_confidence как proxy для confidence landmarks
        landmark_stability = float(detection_confidence)  # Можно улучшить, анализируя отдельные landmarks

        # ----- Output -----
        return {
            "geometry": {
                "face_bbox_area": area,
                "face_relative_size": area / (frame_w * frame_h + 1e-6),
                "face_box_ratio": ratio,
                "face_bbox_position": {"cx": cx, "cy": cy},
                "face_center_x_norm": face_center_x_norm,
                "face_center_y_norm": face_center_y_norm,
                "face_dist_from_center": dist_center,
                "face_rotation_in_frame": float(_roll_angle(coords)),
                "aspect_ratio_stability": stability,
                "jaw_width": float(jaw_width),
                "jaw_width_norm": jaw_width_norm,
                "cheekbone_width": float(cheekbone_width),
                "cheekbone_width_norm": cheekbone_width_norm,
                "cheek_width": float(cheek_width),
                "cheek_width_norm": cheek_width_norm,
                "forehead_height": float(forehead_height),
                "forehead_height_norm": forehead_height_norm,
                "face_shape_vector": face_shape_vector,
                "landmark_stability": landmark_stability,
            }
        }

