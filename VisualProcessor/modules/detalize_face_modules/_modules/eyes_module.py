"""
Модуль для извлечения фич глаз.
"""

from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
import numpy as np

from _modules.base_module import FaceModule
from _utils.landmarks_utils import LANDMARKS
from _utils.face_helpers import safe_distance


def _estimate_blink_rate(history: deque) -> float:
    """Оценка частоты моргания на основе истории открытия глаз."""
    if len(history) < 2:
        return 0.0
    diffs = np.diff(list(history))
    threshold = -0.5 * np.std(list(history)) if np.std(list(history)) > 0 else -0.01
    blinks = np.sum(diffs < threshold)
    return float(blinks / max(len(history), 1))


class EyesModule(FaceModule):
    """
    Модуль для извлечения фич глаз.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self._blink_history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=30)
        )

    def required_inputs(self) -> List[str]:
        """Требуются coords, pose и face_idx."""
        return ["coords", "pose", "face_idx"]

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Обрабатывает данные и возвращает фичи глаз."""
        coords = data["coords"]
        pose = data["pose"]
        face_idx = data["face_idx"]

        # Eye opening
        left_open = safe_distance(coords, LANDMARKS["left_eye_upper"], LANDMARKS["left_eye_lower"])
        right_open = safe_distance(coords, LANDMARKS["right_eye_upper"], LANDMARKS["right_eye_lower"])
        avg_open = (left_open + right_open) / 2.0

        # Save to history
        self._blink_history[face_idx].append(avg_open)
        if len(self._blink_history[face_idx]) > 30:
            self._blink_history[face_idx].popleft()

        # Blink rate
        blink_rate = _estimate_blink_rate(self._blink_history[face_idx])
        blink_intensity = float(np.std(list(self._blink_history[face_idx]))) if len(self._blink_history[face_idx]) > 1 else 0.0

        # Gaze estimation from head pose
        yaw = float(np.radians(pose.get("yaw", 0.0)))
        pitch = float(np.radians(pose.get("pitch", 0.0)))

        gaze_x = np.sin(yaw) * np.cos(pitch)
        gaze_y = np.sin(pitch)
        gaze_z = np.cos(yaw) * np.cos(pitch)

        gaze_vector = [float(gaze_x), float(gaze_y), float(gaze_z)]

        # Probability of looking at camera
        gaze_at_camera_prob = float(np.clip(1 - abs(pose.get("yaw", 0)) / 30.0, 0.0, 1.0))

        # Eye redness
        eye_redness_prob = float(np.clip((left_open + right_open) / 40.0, 0.0, 1.0))

        # Iris position
        left_width = safe_distance(coords, LANDMARKS["left_eye_inner"], LANDMARKS["left_eye_outer"])
        right_width = safe_distance(coords, LANDMARKS["right_eye_inner"], LANDMARKS["right_eye_outer"])

        left_iris = left_open / max(left_width, 1e-6)
        right_iris = right_open / max(right_width, 1e-6)

        iris_position = {
            "left": float(np.clip(left_iris, 0.0, 1.0)),
            "right": float(np.clip(right_iris, 0.0, 1.0)),
        }

        return {
            "eyes": {
                "eye_opening_ratio": {
                    "left": float(left_open),
                    "right": float(right_open),
                    "average": float(avg_open),
                },
                "blink_rate": blink_rate,
                "blink_intensity": blink_intensity,
                "gaze_vector": gaze_vector,
                "gaze_at_camera_prob": gaze_at_camera_prob,
                "attention_score": float((gaze_at_camera_prob + avg_open) / 2.0),
                "eye_redness_prob": eye_redness_prob,
                "iris_position": iris_position,
            }
        }

