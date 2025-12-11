"""
Вспомогательные функции для работы с лицом.
"""

import numpy as np
import cv2
from typing import List
from .landmarks_utils import LANDMARKS


def safe_distance(coords, i, j):
    """Безопасное вычисление расстояния между landmarks."""
    if i >= len(coords) or j >= len(coords):
        return 0.0
    return float(np.linalg.norm(coords[i][:2] - coords[j][:2]))


def eye_opening(coords: np.ndarray) -> float:
    """Среднее вертикальное расстояние между верхним и нижним веком для обоих глаз."""
    left = safe_distance(coords, LANDMARKS["left_eye_upper"], LANDMARKS["left_eye_lower"])
    right = safe_distance(coords, LANDMARKS["right_eye_upper"], LANDMARKS["right_eye_lower"])
    return float((left + right) / 2.0)


def eye_box(coords: np.ndarray) -> List[float]:
    """Вычисляет bounding box, покрывающий оба глаза."""
    left_eye_points = coords[
        [LANDMARKS["left_eye_inner"], LANDMARKS["left_eye_outer"],
         LANDMARKS["left_eye_upper"], LANDMARKS["left_eye_lower"]], :2
    ]
    right_eye_points = coords[
        [LANDMARKS["right_eye_inner"], LANDMARKS["right_eye_outer"],
         LANDMARKS["right_eye_upper"], LANDMARKS["right_eye_lower"]], :2
    ]
    all_points = np.vstack([left_eye_points, right_eye_points])
    return [
        float(np.min(all_points[:, 0])),
        float(np.min(all_points[:, 1])),
        float(np.max(all_points[:, 0])),
        float(np.max(all_points[:, 1])),
    ]


def lower_face_box(coords: np.ndarray) -> List[float]:
    """Вычисляет bounding box для нижней части лица (рот + подбородок)."""
    jaw_indices = [LANDMARKS["mouth_left"], LANDMARKS["mouth_right"], LANDMARKS["chin"]]
    pts = coords[jaw_indices, :2]
    return [
        float(np.min(pts[:, 0])),
        float(np.min(pts[:, 1])),
        float(np.max(pts[:, 0])),
        float(np.max(pts[:, 1])),
    ]


def slice_roi(roi: np.ndarray, box: List[float]) -> np.ndarray:
    """Обрезает прямоугольную область из ROI, безопасно обрабатывая выход за границы."""
    if roi.size <= 1:
        return np.zeros((1, 1, 3), dtype=roi.dtype)
    x_min = int(np.clip(box[0], 0, roi.shape[1] - 1))
    y_min = int(np.clip(box[1], 0, roi.shape[0] - 1))
    x_max = int(np.clip(box[2], x_min + 1, roi.shape[1]))
    y_max = int(np.clip(box[3], y_min + 1, roi.shape[0]))
    return roi[y_min:y_max, x_min:x_max]

