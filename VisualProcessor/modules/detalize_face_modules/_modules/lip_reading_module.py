"""
Модуль для извлечения продвинутых lip reading features.
Извлекает детальные характеристики движения губ для предсказания speech activity.
"""

from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
import numpy as np

from _modules.base_module import FaceModule
from _utils.landmarks_utils import LANDMARKS
from _utils.face_helpers import safe_distance


class LipReadingModule(FaceModule):
    """
    Модуль для извлечения продвинутых lip reading features.
    
    Извлекает:
    - Mouth shape parameters (ширина, высота, площадь)
    - Lip contour features
    - Phoneme-like features (формы губ, характерные для разных звуков)
    - Speech activity probability
    - Temporal patterns (скорость движения губ, цикличность)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.fps = self.config.get("fps", 30.0)
        self._lip_history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=int(self.fps * 2))  # История на 2 секунды
        )

    def required_inputs(self) -> List[str]:
        """Требуются coords, motion и face_idx."""
        return ["coords", "motion", "face_idx"]

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Обрабатывает данные и возвращает продвинутые lip reading features."""
        coords = data["coords"]
        motion = data.get("motion", {})
        face_idx = data["face_idx"]

        # Extract basic mouth landmarks
        upper_lip_idx = LANDMARKS.get("upper_lip", 13)
        lower_lip_idx = LANDMARKS.get("lower_lip", 14)
        mouth_left_idx = LANDMARKS.get("mouth_left", 61)
        mouth_right_idx = LANDMARKS.get("mouth_right", 291)

        # Basic mouth measurements
        mouth_width = safe_distance(coords, mouth_left_idx, mouth_right_idx)
        mouth_height = safe_distance(coords, upper_lip_idx, lower_lip_idx)
        
        # Mouth area approximation (ellipse)
        mouth_area = np.pi * (mouth_width / 2.0) * (mouth_height / 2.0)
        
        # Mouth aspect ratio
        mouth_aspect_ratio = mouth_height / max(mouth_width, 1e-6)
        
        # Lip corner positions
        if (mouth_left_idx < len(coords) and 
            mouth_right_idx < len(coords) and
            upper_lip_idx < len(coords) and
            lower_lip_idx < len(coords)):
            
            mouth_center = (
                (coords[mouth_left_idx][:2] + coords[mouth_right_idx][:2]) / 2.0
            )
            upper_lip_center = coords[upper_lip_idx][:2]
            lower_lip_center = coords[lower_lip_idx][:2]
            
            # Vertical lip separation
            lip_separation = np.linalg.norm(upper_lip_center - lower_lip_center)
            
            # Horizontal asymmetry
            left_half_width = np.linalg.norm(mouth_center - coords[mouth_left_idx][:2])
            right_half_width = np.linalg.norm(coords[mouth_right_idx][:2] - mouth_center)
            lip_asymmetry = abs(left_half_width - right_half_width) / max(mouth_width, 1e-6)
        else:
            lip_separation = 0.0
            lip_asymmetry = 0.0
            mouth_center = np.array([0.0, 0.0])

        # Lip contour features
        # Используем доступные индексы для контура губ
        lip_contour_points = []
        for key in ["mouth_left", "upper_lip", "mouth_right", "lower_lip"]:
            idx = LANDMARKS.get(key)
            if idx is not None and idx < len(coords):
                lip_contour_points.append(coords[idx][:2])
        
        if len(lip_contour_points) >= 4:
            lip_contour_points = np.array(lip_contour_points)
            
            # Lip contour curvature (сколько губы изогнуты)
            # Вычисляем как отношение периметра к площади
            perimeter = (
                np.linalg.norm(lip_contour_points[1] - lip_contour_points[0]) +
                np.linalg.norm(lip_contour_points[2] - lip_contour_points[1]) +
                np.linalg.norm(lip_contour_points[3] - lip_contour_points[2]) +
                np.linalg.norm(lip_contour_points[0] - lip_contour_points[3])
            )
            contour_compactness = (perimeter ** 2) / max(mouth_area, 1e-6)
            
            # Upper vs lower lip prominence
            upper_lip_height = np.linalg.norm(lip_contour_points[1] - mouth_center)
            lower_lip_height = np.linalg.norm(lip_contour_points[3] - mouth_center)
            lip_prominence_ratio = upper_lip_height / max(lower_lip_height, 1e-6)
        else:
            contour_compactness = 0.0
            lip_prominence_ratio = 1.0

        # Phoneme-like features (приблизительная классификация форм)
        # Эти признаки помогают различать разные звуки
        phoneme_features = {
            "round_shape": float(np.clip(mouth_aspect_ratio, 0.0, 1.0)),  # Округлая форма (О, У)
            "wide_shape": float(np.clip(mouth_width / max(mouth_height, 1e-6), 0.0, 2.0) / 2.0),  # Широкая форма (И, Э)
            "narrow_shape": float(np.clip(1.0 - mouth_aspect_ratio, 0.0, 1.0)),  # Узкая форма
            "open_shape": float(np.clip(lip_separation / max(mouth_width, 1e-6), 0.0, 1.0)),  # Открытый рот (А, Э)
        }

        # Temporal features (история движения губ)
        current_frame_features = {
            "mouth_width": float(mouth_width),
            "mouth_height": float(mouth_height),
            "mouth_area": float(mouth_area),
            "lip_separation": float(lip_separation),
            "mouth_aspect_ratio": float(mouth_aspect_ratio),
        }
        
        if face_idx not in self._lip_history:
            self._lip_history[face_idx] = deque(maxlen=int(self.fps * 2))
        
        history = self._lip_history[face_idx]
        history.append(current_frame_features)

        # Compute temporal statistics
        if len(history) >= 2:
            widths = [f["mouth_width"] for f in history]
            heights = [f["mouth_height"] for f in history]
            areas = [f["mouth_area"] for f in history]
            separations = [f["lip_separation"] for f in history]
            
            # Motion statistics
            width_var = float(np.var(widths))
            height_var = float(np.var(heights))
            area_var = float(np.var(areas))
            separation_var = float(np.var(separations))
            
            # Rate of change
            width_velocity = float(np.mean(np.abs(np.diff(widths)))) if len(widths) > 1 else 0.0
            height_velocity = float(np.mean(np.abs(np.diff(heights)))) if len(heights) > 1 else 0.0
            area_velocity = float(np.mean(np.abs(np.diff(areas)))) if len(areas) > 1 else 0.0
            
            # Speech activity indicators
            # Быстрые изменения формы рта обычно указывают на речь
            mouth_motion_intensity = float(
                (width_velocity + height_velocity + area_velocity) / 3.0
            )
            
            # Цикличность (для различения речи от просто движений)
            # Речь имеет более циклический паттерн
            if len(areas) >= 4:
                # Автокорреляция для обнаружения циклов
                autocorr = np.correlate(areas, areas, mode='full')
                mid = len(autocorr) // 2
                if mid > 0:
                    # Ищем пики в автокорреляции (кроме нулевого смещения)
                    autocorr_peaks = autocorr[mid+1:mid+min(len(areas)//2, 30)]
                    if len(autocorr_peaks) > 0:
                        cycle_strength = float(np.max(autocorr_peaks) / max(autocorr[mid], 1e-6))
                    else:
                        cycle_strength = 0.0
                else:
                    cycle_strength = 0.0
            else:
                cycle_strength = 0.0
            
            # Speech probability based on motion patterns
            # Используем комбинацию интенсивности движения и цикличности
            motion_from_motion_module = motion.get("mouth_motion_score", 0.0)
            talking_from_motion = motion.get("talking_motion_score", 0.0)
            
            speech_activity_prob = float(np.clip(
                (mouth_motion_intensity * 0.4 + 
                 cycle_strength * 0.3 + 
                 talking_from_motion * 0.3) * 2.0,
                0.0, 1.0
            ))
        else:
            width_var = height_var = area_var = separation_var = 0.0
            width_velocity = height_velocity = area_velocity = 0.0
            mouth_motion_intensity = 0.0
            cycle_strength = 0.0
            speech_activity_prob = motion.get("speech_activity_prob", 0.0)

        return {
            "lip_reading": {
                # Basic geometric features
                "mouth_width": float(mouth_width),
                "mouth_height": float(mouth_height),
                "mouth_area": float(mouth_area),
                "mouth_aspect_ratio": float(mouth_aspect_ratio),
                "lip_separation": float(lip_separation),
                "lip_asymmetry": float(lip_asymmetry),
                
                # Contour features
                "lip_contour_compactness": float(contour_compactness),
                "lip_prominence_ratio": float(lip_prominence_ratio),
                
                # Phoneme-like features
                "phoneme_features": phoneme_features,
                
                # Temporal features
                "mouth_motion_intensity": float(mouth_motion_intensity),
                "width_velocity": float(width_velocity),
                "height_velocity": float(height_velocity),
                "area_velocity": float(area_velocity),
                "cycle_strength": float(cycle_strength),
                
                # Speech activity
                "speech_activity_prob": float(speech_activity_prob),
                
                # Variability metrics
                "width_variance": float(width_var),
                "height_variance": float(height_var),
                "area_variance": float(area_var),
                "separation_variance": float(separation_var),
            }
        }

