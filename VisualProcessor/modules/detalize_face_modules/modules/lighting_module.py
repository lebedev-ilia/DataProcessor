"""
Модуль для извлечения фич освещения лица.
"""

from typing import Dict, List, Any, Optional
import cv2
import numpy as np

from modules.base_module import FaceModule
from utils.landmarks_utils import LANDMARKS


class LightingModule(FaceModule):
    """
    Модуль для извлечения фич освещения лица.
    """

    def required_inputs(self) -> List[str]:
        """Требуются roi и опционально coords_roi."""
        return ["roi"]

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Обрабатывает данные и возвращает фичи освещения."""
        roi = data["roi"]
        coords_roi = data.get("coords_roi")

        if roi.size <= 3:
            return {"lighting": {}}

        h, w = roi.shape[:2]

        if h < 10 or w < 10:
            return {"lighting": {}}

        # --- Convert to HSV ---
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        value = hsv[:, :, 2].astype(np.float32)

        # --- Global brightness ---
        avg_light = float(np.mean(value) / 255.0)

        # --- Uniformity (left/right) ---
        mid_x = w // 2
        left_mean = float(np.mean(value[:, :mid_x])) if mid_x > 1 else 0
        right_mean = float(np.mean(value[:, mid_x:])) if w - mid_x > 1 else 0
        uniformity = float(1.0 - abs(left_mean - right_mean) / 255.0)
        uniformity = float(np.clip(uniformity, 0.0, 1.0))

        # --- Contrast (robust, percentile-based) ---
        p5 = np.percentile(value, 5)
        p95 = np.percentile(value, 95)
        contrast = float(np.clip((p95 - p5) / 255.0, 0.0, 1.0))

        # --- White balance estimation ---
        mean_rgb = np.mean(roi.reshape(-1, 3), axis=0).astype(np.float32)
        b, g, r = mean_rgb

        white_balance_shift = {
            "r_minus_g": float((r - g) / 255.0),
            "r_minus_b": float((r - b) / 255.0),
            "g_minus_b": float((g - b) / 255.0),
        }

        # --- LAB for skin color vector ---
        lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
        L_channel = lab[:, :, 0].astype(np.float32)
        A_channel = lab[:, :, 1].astype(np.float32)
        B_channel = lab[:, :, 2].astype(np.float32)

        skin_vector = [
            float(np.mean(A_channel)),
            float(np.mean(B_channel)),
        ]

        # --- Highlight / shadow metrics ---
        highlight = float(np.max(value) / 255.0)
        shadow = float(np.min(value) / 255.0)
        glare_score = float(np.clip(p95 / 255.0, 0.0, 1.0))
        shading_score = float(np.clip(1.0 - p5 / 255.0, 0.0, 1.0))

        # --- Skin tone classification (Fitzpatrick approx) ---
        mean_L = float(np.mean(L_channel))
        if mean_L > 80:
            skin_tone_index = 1
        elif mean_L > 70:
            skin_tone_index = 2
        elif mean_L > 60:
            skin_tone_index = 3
        elif mean_L > 50:
            skin_tone_index = 4
        elif mean_L > 40:
            skin_tone_index = 5
        else:
            skin_tone_index = 6

        # --- Zone lighting analysis ---
        zone_lighting = {}
        if coords_roi is not None and coords_roi.shape[0] > 0:
            try:
                yy = np.clip(coords_roi[:, 1].astype(int), 0, h - 1)
                forehead_y = yy[LANDMARKS["forehead"]]
                nose_y = yy[LANDMARKS["nose_tip"]]
                chin_y = yy[LANDMARKS["chin"]]

                top = max(0, int(forehead_y * 0.6))
                mid = int(nose_y)
                bot = int(chin_y)

                if top < forehead_y:
                    zone = value[:forehead_y]
                    if zone.size > 0:
                        zone_lighting["forehead_brightness"] = float(np.mean(zone) / 255.0)

                if forehead_y < mid:
                    zone = value[forehead_y:mid]
                    if zone.size > 0:
                        zone_lighting["cheek_brightness"] = float(np.mean(zone) / 255.0)

                if mid < bot:
                    zone = value[mid:bot]
                    if zone.size > 0:
                        zone_lighting["chin_brightness"] = float(np.mean(zone) / 255.0)
            except Exception:
                pass

        # --- Combined lighting proxy score ---
        lighting_proxy = float(np.clip(
            0.4 * avg_light +
            0.3 * uniformity +
            0.2 * contrast +
            0.1 * (1.0 - shadow),
            0.0,
            1.0
        ))

        result = {
            "average_lighting_on_face": avg_light,
            "light_uniformity_score": uniformity,
            "face_contrast": contrast,
            "white_balance_shift": white_balance_shift,
            "skin_color_vector": skin_vector,
            "highlight_intensity": highlight,
            "shadow_depth": shadow,
            "glare_score": glare_score,
            "shading_score": shading_score,
            "skin_tone_index": skin_tone_index,
            "lighting_proxy_score": lighting_proxy,
        }

        if zone_lighting:
            result["zone_lighting"] = zone_lighting

        return {"lighting": result}

