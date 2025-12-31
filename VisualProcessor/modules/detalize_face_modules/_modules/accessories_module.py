"""
Модуль для извлечения фич аксессуаров лица.
"""

from typing import Dict, List, Any, Optional
import cv2
import numpy as np

from _modules.base_module import FaceModule
from _utils.landmarks_utils import LANDMARKS
from _utils.face_helpers import eye_box, lower_face_box


class AccessoriesModule(FaceModule):
    """
    Модуль для извлечения фич аксессуаров лица (очки, маска, шапка, etc.).
    """

    def required_inputs(self) -> List[str]:
        """Требуются roi и coords_roi."""
        return ["roi", "coords_roi"]

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Обрабатывает данные и возвращает фичи аксессуаров."""
        roi = data["roi"]
        coords_roi = data["coords_roi"]

        if roi is None or roi.size < 16:
            return {"accessories": {}}

        h, w = roi.shape[:2]
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # Eyes region / glasses
        glasses_prob = 0.0
        sunglasses_prob = 0.0
        try:
            ex1, ey1, ex2, ey2 = eye_box(coords_roi)
            ex1, ey1 = int(max(0, ex1)), int(max(0, ey1))
            ex2, ey2 = int(min(w - 1, ex2)), int(min(h - 1, ey2))
            eye_roi = gray[ey1:ey2, ex1:ex2] if ex2 > ex1 and ey2 > ey1 else np.array([], dtype=np.uint8)

            if eye_roi.size:
                edges = cv2.Canny(eye_roi, 50, 150)
                edge_density = float(np.sum(edges > 0) / (eye_roi.size + 1e-6))
                glasses_prob = float(np.clip(edge_density * 3.0, 0.0, 1.0))

                mean_eye = float(np.mean(eye_roi))
                sunglasses_prob = float(np.clip((1.0 - (mean_eye / 255.0)) * 1.2, 0.0, 1.0))
                if sunglasses_prob > 0.4 and edge_density > 0.02:
                    sunglasses_prob = float(np.clip(sunglasses_prob + edge_density * 0.5, 0.0, 1.0))
                    glasses_prob = max(glasses_prob, 0.15)
        except Exception:
            pass

        # Mask detection
        mask_prob = 0.0
        try:
            lx1, ly1, lx2, ly2 = lower_face_box(coords_roi)
            lx1, ly1 = int(max(0, lx1)), int(max(0, ly1))
            lx2, ly2 = int(min(w - 1, lx2)), int(min(h - 1, ly2))
            lower_roi = roi[ly1:ly2, lx1:lx2] if lx2 > lx1 and ly2 > ly1 else np.array([], dtype=np.uint8)

            if lower_roi.size:
                lower_hsv = cv2.cvtColor(lower_roi, cv2.COLOR_BGR2HSV)
                v = lower_hsv[:, :, 2].astype(np.float32) / 255.0
                v_std = float(np.std(v))
                v_mean = float(np.mean(v))
                color_homogeneity = 1.0 - v_std
                coverage = float(lower_roi.size / (roi.size + 1e-6))
                mask_prob = float(np.clip(color_homogeneity * 1.2 * coverage, 0.0, 1.0))
        except Exception:
            pass

        # Hat / helmet detection
        top_strip = roi[0:max(1, h // 6), :, :]
        top_gray = cv2.cvtColor(top_strip, cv2.COLOR_BGR2GRAY) if top_strip.size else np.array([], dtype=np.uint8)
        hat_prob = 0.0
        helmet_prob = 0.0
        if top_gray.size:
            top_mean = float(np.mean(top_gray))
            top_std = float(np.std(top_gray))
            hat_prob = float(np.clip((1.0 - top_std / (np.mean(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)) + 1e-6)) * (1.0 - top_mean / 255.0), 0.0, 1.0))
            helmet_prob = float(np.clip(hat_prob * 0.4 if top_mean < 80 and top_std < 30 else 0.0, 0.0, 1.0))

        # Earrings detection
        earrings_prob = 0.0
        try:
            left_cheek = coords_roi[LANDMARKS["left_cheek"], :2].astype(int)
            right_cheek = coords_roi[LANDMARKS["right_cheek"], :2].astype(int)
            ear_y = int((left_cheek[1] + right_cheek[1]) / 2)
            ear_region_size = min(30, max(10, w // 8))

            def detect_ear_spot(cx):
                x1 = max(0, cx - ear_region_size)
                x2 = min(w, cx + ear_region_size)
                y1 = max(0, ear_y - ear_region_size // 2)
                y2 = min(h, ear_y + ear_region_size // 2)
                region = roi[y1:y2, x1:x2]
                if region.size == 0:
                    return 0.0
                gray_r = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
                bright_frac = float(np.sum(gray_r > np.mean(gray_r) * 1.25) / (gray_r.size + 1e-6))
                return bright_frac

            left_spot = detect_ear_spot(int(left_cheek[0]))
            right_spot = detect_ear_spot(int(right_cheek[0]))
            earrings_prob = float(np.clip((left_spot + right_spot) / 2.0, 0.0, 1.0))
        except Exception:
            pass
        earrings_presence = bool(earrings_prob > 0.03)

        # Jewelry / necklace detection
        jewelry_probability = 0.0
        try:
            chin_y = int(coords_roi[LANDMARKS["chin"], 1])
            neck_y = min(h - 1, chin_y + int(h * 0.06))
            if 0 <= neck_y < h:
                neck_h = min(30, h - neck_y)
                neck_roi = roi[neck_y:neck_y + neck_h, :]
                if neck_roi.size:
                    neck_gray = cv2.cvtColor(neck_roi, cv2.COLOR_BGR2GRAY)
                    jewelry_probability = float(np.clip(np.std(neck_gray) / 80.0, 0.0, 1.0))
        except Exception:
            pass

        # Hair accessories
        hair_accessories = 0.0
        try:
            forehead_y = int(coords_roi[LANDMARKS["forehead"], 1])
            hair_top_y = max(0, forehead_y - 20)
            hair_roi = roi[hair_top_y:forehead_y, :]
            if hair_roi.size:
                gray_hr = cv2.cvtColor(hair_roi, cv2.COLOR_BGR2GRAY)
                hr_std = float(np.std(gray_hr))
                whole_std = float(np.std(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)))
                if whole_std > 1e-6:
                    hair_accessories = float(np.clip(max(0.0, (hr_std - whole_std * 1.2) / 60.0), 0.0, 1.0))
        except Exception:
            pass

        return {
            "accessories": {
                "glasses_prob": float(glasses_prob),
                "sunglasses_prob": float(sunglasses_prob),
                "mask_prob": float(mask_prob),
                "hat_prob": float(hat_prob),
                "helmet_prob": float(helmet_prob),
                "earrings_presence": bool(earrings_presence),
                "earrings_prob": float(earrings_prob),
                "jewelry_probability": float(jewelry_probability),
                "hair_accessories": float(hair_accessories),
            }
        }

