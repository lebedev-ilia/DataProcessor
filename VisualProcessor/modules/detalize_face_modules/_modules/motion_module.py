"""
Модуль для извлечения фич движения лица.
"""

from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
import numpy as np

from _modules.base_module import FaceModule
from _utils.landmarks_utils import LANDMARKS
from _utils.face_helpers import safe_distance, eye_opening


class MotionModule(FaceModule):
    """
    Модуль для извлечения фич движения лица.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.fps = self.config.get("fps", 30.0)
        self._temporal_state_history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=int(self.fps // 2))
        )

    def required_inputs(self) -> List[str]:
        """Требуются coords, geometry и face_idx."""
        return ["coords", "geometry", "face_idx"]

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Обрабатывает данные и возвращает фичи движения."""
        coords = data["coords"]
        geometry = data["geometry"]
        face_idx = data["face_idx"]

        # Current frame features
        center = geometry["face_bbox_position"]["cx"], geometry["face_bbox_position"]["cy"]
        mouth_gap = safe_distance(coords, LANDMARKS["upper_lip"], LANDMARKS["lower_lip"])
        jaw_distance = safe_distance(coords, LANDMARKS["chin"], LANDMARKS["nose_tip"])
        eyebrow_height = coords[LANDMARKS["forehead"], 1] - coords[LANDMARKS["left_brow"], 1]
        eye_opening_val = eye_opening(coords)

        # Initialize temporal history
        if face_idx not in self._temporal_state_history:
            self._temporal_state_history[face_idx] = deque(maxlen=int(self.fps // 2))

        history = self._temporal_state_history[face_idx]
        history.append({
            "center": center,
            "mouth_gap": mouth_gap,
            "jaw_distance": jaw_distance,
            "eyebrow_height": eyebrow_height,
            "eye_opening": eye_opening_val,
        })

        # If not enough history, return zeros
        if len(history) < 2:
            return {
                "motion": {
                    "face_speed": 0.0,
                    "face_acceleration": 0.0,
                    "micro_expression_rate": 0.0,
                    "jaw_movement_intensity": 0.0,
                    "eyebrows_motion_score": 0.0,
                    "mouth_motion_score": 0.0,
                    "head_motion_energy": 0.0,
                    "talking_motion_score": 0.0,
                }
            }

        # Compute differences over history
        centers = np.array([s["center"] for s in history])
        mouth_gaps = np.array([s["mouth_gap"] for s in history])
        jaw_distances = np.array([s["jaw_distance"] for s in history])
        eyebrow_heights = np.array([s["eyebrow_height"] for s in history])
        eye_openings = np.array([s["eye_opening"] for s in history])

        displacements = np.linalg.norm(np.diff(centers, axis=0), axis=1)
        mouth_deltas = np.abs(np.diff(mouth_gaps))
        jaw_deltas = np.abs(np.diff(jaw_distances))
        eyebrow_deltas = np.abs(np.diff(eyebrow_heights))
        eye_deltas = np.abs(np.diff(eye_openings))

        # Dynamic thresholding
        speed_threshold = 0.5
        motion_threshold = 0.3

        filtered_displacements = displacements[displacements > speed_threshold]
        filtered_mouth = mouth_deltas[mouth_deltas > motion_threshold]
        filtered_jaw = jaw_deltas[jaw_deltas > motion_threshold]
        filtered_eyebrow = eyebrow_deltas[eyebrow_deltas > motion_threshold]
        filtered_eye = eye_deltas[eye_deltas > motion_threshold]

        # Compute metrics
        face_speed = float(np.mean(filtered_displacements)) if filtered_displacements.size else 0.0
        face_acceleration = float(np.mean(np.diff(filtered_displacements))) if filtered_displacements.size > 1 else 0.0
        mouth_motion = float(np.mean(filtered_mouth)) if filtered_mouth.size else 0.0
        jaw_motion = float(np.mean(filtered_jaw)) if filtered_jaw.size else 0.0
        eyebrow_motion = float(np.mean(filtered_eyebrow)) if filtered_eyebrow.size else 0.0
        head_motion = float(np.mean(filtered_eye)) if filtered_eye.size else 0.0

        micro_expression_rate = mouth_motion + eyebrow_motion
        talking_motion_score = mouth_motion
        head_motion_energy = head_motion

        return {
            "motion": {
                "face_speed": face_speed,
                "face_acceleration": face_acceleration,
                "micro_expression_rate": micro_expression_rate,
                "jaw_movement_intensity": jaw_motion,
                "eyebrows_motion_score": eyebrow_motion,
                "mouth_motion_score": mouth_motion,
                "head_motion_energy": head_motion_energy,
                "talking_motion_score": talking_motion_score,
            }
        }

