"""
Модуль для извлечения фич кожи лица.
"""

from typing import Dict, List, Any, Optional
import cv2
import numpy as np

from _modules.base_module import FaceModule
from _utils.landmarks_utils import LANDMARKS
from _utils.face_helpers import eye_opening


class SkinModule(FaceModule):
    """
    Модуль для извлечения фич кожи лица.
    """

    def required_inputs(self) -> List[str]:
        """Требуются roi и coords_roi."""
        return ["roi", "coords_roi"]

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Обрабатывает данные и возвращает фичи кожи."""
        roi = data["roi"]
        coords_roi = data["coords_roi"]

        if roi.size <= 1:
            return {"skin": {}}

        h, w = roi.shape[:2]

        # Convert color spaces
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)

        sat = hsv[:, :, 1] / 255.0
        avg_sat = float(np.mean(sat))

        # Extract lips ROI
        lip_idx = [LANDMARKS["upper_lip"], LANDMARKS["lower_lip"]]
        lip_coords = coords_roi[lip_idx, :2]

        lip_x1 = int(max(0, np.min(lip_coords[:, 0])))
        lip_y1 = int(max(0, np.min(lip_coords[:, 1])))
        lip_x2 = int(min(w - 1, np.max(lip_coords[:, 0])))
        lip_y2 = int(min(h - 1, np.max(lip_coords[:, 1])))

        lip_roi = roi[lip_y1:lip_y2, lip_x1:lip_x2] if lip_x2 > lip_x1 and lip_y2 > lip_y1 else None

        if lip_roi is not None and lip_roi.size > 0:
            lip_hsv = cv2.cvtColor(lip_roi, cv2.COLOR_BGR2HSV)
            lip_sat = float(np.mean(lip_hsv[:, :, 1] / 255.0))
        else:
            lip_sat = 0.0

        # Lipstick detection
        lipstick_intensity = float(np.clip((lip_sat - avg_sat) * 1.5, 0.0, 1.0))
        makeup_prob = lipstick_intensity
        eye_shadow_prob = float(lipstick_intensity * 0.4)

        # Skin smoothness (техническая метрика, не связанная с привлекательностью)
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        laplacian_var = cv2.Laplacian(blur, cv2.CV_64F).var()

        skin_smoothness = float(1 / (1 + laplacian_var * 0.1))
        # skin_defect_score удален - рискованная метрика, может быть дискриминационной

        # Beard/mustache detection
        chin_idx = LANDMARKS["chin"]
        chin_y = int(coords_roi[chin_idx, 1])
        lower_roi = roi[chin_y: h] if 0 < chin_y < h else roi

        if lower_roi.size > 0:
            lower_lab = cv2.cvtColor(lower_roi, cv2.COLOR_BGR2LAB)
            L_lower = lower_lab[:, :, 0]
            darkness = 1.0 - float(np.mean(L_lower) / 255.0)
            texture = float(cv2.Laplacian(cv2.cvtColor(lower_roi, cv2.COLOR_BGR2GRAY),
                                        cv2.CV_64F).var() / 50.0)
            beard_prob = float(np.clip(0.6 * darkness + 0.4 * texture, 0.0, 1.0))
        else:
            beard_prob = 0.0

        mustache_prob = float(beard_prob * 0.7)
        face_hair_density = float((beard_prob + mustache_prob) / 2)

        # Eyebrow shape vector
        eyebrow_pts = coords_roi[[LANDMARKS["left_brow"], LANDMARKS["right_brow"]], :2]
        eyebrow_shape_vector = eyebrow_pts.flatten().tolist()

        # Eyelid openness
        eyelid_openness = eye_opening(coords_roi)

        return {
            "skin": {
                # Безопасные фичи (non-sensitive)
                "skin_smoothness": skin_smoothness,  # Техническая метрика
                "face_hair_density": face_hair_density,  # Бинарная/плотность растительности
                "beard_prob": beard_prob,  # Вероятность наличия бороды
                "mustache_prob": mustache_prob,  # Вероятность наличия усов
                # Аккуратно с макияжем (может быть чувствительным, но полезным для классификации формата)
                "makeup_presence_prob": makeup_prob,  # С осторожностью и аудитом
                "lipstick_intensity": lipstick_intensity,  # С осторожностью
                "eye_shadow_prob": eye_shadow_prob,  # С осторожностью
                # Структурные фичи (не связанные с привлекательностью)
                "eyebrow_shape_vector": eyebrow_shape_vector,
                "eyelid_openness": eyelid_openness,
                # Удалено: skin_defect_score - рискованная метрика
            }
        }

