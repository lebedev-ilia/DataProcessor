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
        coords = data.get("coords")
        detection_confidence = data.get("detection_confidence", 1.0)

        if roi.size <= 3:
            return {"quality": {}}

        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)

        # --- Объединенные метрики качества ---
        # face_sharpness: комбинация blur и sharpness
        blur_raw = cv2.Laplacian(gray, cv2.CV_64F).var()
        blur_score = float(np.clip(blur_raw / 300.0, 0.0, 1.0))
        
        sobel = cv2.Sobel(gray, cv2.CV_32F, 1, 1, ksize=3)
        sharpness_raw = float(np.mean(np.abs(sobel)))
        sharpness_score = float(np.clip(sharpness_raw / 200.0, 0.0, 1.0))
        
        # Объединенная метрика резкости
        face_sharpness = float(np.clip(
            0.6 * blur_score + 0.4 * sharpness_score,
            0.0, 1.0
        ))

        # --- face_noise_level ---
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        noise = float(np.std(gray.astype(np.float32) - blurred.astype(np.float32)))
        face_noise_level = float(np.clip(noise / 40.0, 0.0, 1.0))

        # --- face_exposure_score ---
        mean, std = cv2.meanStdDev(gray)
        exposure_mean = float(mean[0][0] / 255.0)
        exposure_std = float(std[0][0] / 255.0)
        # Хорошая экспозиция: средняя яркость около 0.5, достаточный контраст
        face_exposure_score = float(np.clip(
            1.0 - abs(exposure_mean - 0.5) * 2.0,  # Близость к 0.5
            0.0, 1.0
        ))

        # --- Улучшенный occlusion_score ---
        # Оцениваем по fraction of landmarks with low confidence + visible mouth/eyes fraction
        occlusion_proxy = 0.0
        if coords is not None:
            # Используем detection_confidence как proxy для confidence landmarks
            low_conf_fraction = 1.0 - detection_confidence
            
            # Проверяем видимость ключевых частей лица (рот, глаза)
            # Упрощенная версия: если landmarks доступны, считаем что части видны
            visible_parts_fraction = detection_confidence  # Можно улучшить, анализируя конкретные landmarks
            
            occlusion_proxy = float(np.clip(
                low_conf_fraction * 0.6 + (1.0 - visible_parts_fraction) * 0.4,
                0.0, 1.0
            ))
        else:
            # Fallback: используем face_visibility_ratio
            frame_area = frame_shape[0] * frame_shape[1]
            face_area = max((bbox[2] - bbox[0]), 1) * max((bbox[3] - bbox[1]), 1)
            ratio = face_area / max(frame_area, 1)
            occlusion_proxy = float(np.clip(1.0 - ratio, 0.0, 1.0))

        # --- Combined "quality proxy" score (стандартизованная шкала [0..1]) ---
        quality_proxy_score = float(np.clip(
            0.4 * face_sharpness +
            0.3 * (1.0 - face_noise_level) +
            0.2 * face_exposure_score +
            0.1 * (1.0 - occlusion_proxy),
            0.0,
            1.0
        ))

        # quality_confidence
        quality_confidence = float(detection_confidence)

        return {
            "quality": {
                # Объединенные метрики
                "face_sharpness": face_sharpness,
                "face_noise_level": face_noise_level,
                "face_exposure_score": face_exposure_score,
                "occlusion_proxy": occlusion_proxy,
                "quality_proxy_score": quality_proxy_score,
                "quality_confidence": quality_confidence,
                # Обратная совместимость (deprecated, но оставляем)
                "face_blur_score": blur_score,
                "sharpness_score": sharpness_score,
                "noise_level": face_noise_level,
                "face_visibility_ratio": float(np.clip((bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) / max(frame_shape[0] * frame_shape[1], 1), 0.0, 1.0)),
            }
        }

