"""
Модуль для извлечения фич качества изображения лица.
"""

from typing import Dict, List, Any, Optional, Tuple
import cv2
import numpy as np

from _modules.base_module import FaceModule


class QualityModule(FaceModule):
    """
    Модуль для извлечения фич качества изображения лица.
    """

    def required_inputs(self) -> List[str]:
        """Требуются roi, bbox и frame_shape."""
        return ["roi", "bbox", "frame_shape"]

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Обрабатывает данные и возвращает фичи качества."""
        roi = data["roi"]
        bbox = data["bbox"]
        frame_shape = data["frame_shape"]

        if roi.size <= 3:
            return {"quality": {}}

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # --- Blur (Variance of Laplacian) ---
        blur_raw = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score = float(np.clip(blur_raw / 300.0, 0.0, 1.0))

        # --- Sharpness (Sobel) ---
        sobel = cv2.Sobel(gray, cv2.CV_32F, 1, 1, ksize=3)
        sharpness = float(np.mean(np.abs(sobel)))
        sharpness_score = float(np.clip(sharpness / 200.0, 0.0, 1.0))

        # --- Texture contrast ---
        texture_quality = float(np.std(gray) / 255.0)

        # --- Focus metric (local variance) ---
        mean, std = cv2.meanStdDev(gray)
        focus_metric = float(np.clip(std[0][0] / 255.0, 0.0, 1.0))

        # --- Noise level ---
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = float(np.std(gray.astype(np.float32) - blurred.astype(np.float32)))
        noise_level = float(np.clip(noise / 40.0, 0.0, 1.0))

        # --- Motion blur heuristic ---
        motion_blur_score = float(np.clip(1.0 - blur_raw / 500.0, 0.0, 1.0))

        # --- Artifact score (inverse texture) ---
        artifact_score = float(np.clip(1.0 - texture_quality, 0.0, 1.0))

        # --- Face visibility ratio ---
        frame_area = frame_shape[0] * frame_shape[1]
        face_area = max((bbox[2] - bbox[0]), 1) * max((bbox[3] - bbox[1]), 1)
        ratio = face_area / max(frame_area, 1)

        face_visibility_ratio = float(np.clip(ratio, 0.0, 1.0))
        occlusion_score = float(np.clip(1.0 - face_visibility_ratio, 0.0, 1.0))

        # --- Combined "quality proxy" score ---
        quality_proxy = float(np.clip(
            0.4 * blur_score +
            0.3 * sharpness_score +
            0.2 * focus_metric +
            0.1 * (1.0 - noise_level),
            0.0,
            1.0
        ))

        return {
            "quality": {
                "face_blur_score": blur_score,
                "sharpness_score": sharpness_score,
                "texture_quality": texture_quality,
                "focus_metric": focus_metric,
                "noise_level": noise_level,
                "motion_blur_score": motion_blur_score,
                "artifact_score": artifact_score,
                "resolution_of_face": float(face_area),
                "face_visibility_ratio": face_visibility_ratio,
                "occlusion_score": occlusion_score,
                "quality_proxy_score": quality_proxy,
            }
        }

