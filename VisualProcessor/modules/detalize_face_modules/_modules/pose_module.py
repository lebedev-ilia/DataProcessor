"""
Модуль для извлечения фич позы головы.
"""

from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
import numpy as np

from _modules.base_module import FaceModule
from _utils.landmarks_utils import LANDMARKS


def _average_coords(coords: np.ndarray, indices: List[int]) -> np.ndarray:
    """Вычисляет средние координаты для заданных индексов landmarks."""
    if len(indices) == 0 or coords.size == 0:
        return np.zeros((coords.shape[1],), dtype=np.float32)
    return np.mean(coords[indices, :], axis=0)


class PoseModule(FaceModule):
    """
    Модуль для извлечения фич позы головы (yaw, pitch, roll).
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.history_size = self.config.get("history_size", 30)  # 30 кадров ≈ 1-1.5 сек
        self.head_turn_threshold = self.config.get("head_turn_threshold", 7.0)  # Порог для head_turn_frequency
        self._pose_history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.history_size)
        )

    def required_inputs(self) -> List[str]:
        """Требуются coords, frame_shape и face_idx."""
        return ["coords", "frame_shape", "face_idx"]

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Обрабатывает данные и возвращает фичи позы."""
        coords = data["coords"]
        frame_shape = data["frame_shape"]
        face_idx = data["face_idx"]

        # --- Landmarks ---
        left_eye = _average_coords(coords, [LANDMARKS["left_eye_inner"], LANDMARKS["left_eye_outer"]])
        right_eye = _average_coords(coords, [LANDMARKS["right_eye_inner"], LANDMARKS["right_eye_outer"]])

        nose = coords[LANDMARKS["nose_tip"], :3]
        chin = coords[LANDMARKS["chin"], :3]
        forehead = coords[10, :3] if 10 < len(coords) else nose.copy()

        # Distance between eyes
        eye_distance = np.linalg.norm(right_eye[:2] - left_eye[:2]) + 1e-6

        # Skull width
        left_ear = coords[234, :3] if 234 < len(coords) else left_eye.copy()
        right_ear = coords[454, :3] if 454 < len(coords) else right_eye.copy()
        skull_width = np.linalg.norm(right_ear[:2] - left_ear[:2]) + 1e-6

        # 1. ROLL
        roll = np.degrees(np.arctan2(
            right_eye[1] - left_eye[1],
            right_eye[0] - left_eye[0]
        ))

        # 2. YAW
        mid_eye = (left_eye + right_eye) / 2
        nose_horizontal_offset = nose[0] - mid_eye[0]
        normalized_horizontal = nose_horizontal_offset / skull_width
        normalized_horizontal = np.clip(normalized_horizontal, -1.2, 1.2)

        yaw_2d = np.degrees(np.arctan2(
            nose_horizontal_offset,
            skull_width * 0.55
        ))

        mid_face_z = (left_eye[2] + right_eye[2] + chin[2]) / 3
        depth_offset = nose[2] - mid_face_z

        yaw_3d = np.degrees(np.arctan2(
            nose_horizontal_offset,
            abs(depth_offset) + skull_width * 0.35
        ))

        yaw = yaw_2d * 0.7 + yaw_3d * 0.3

        # 3. PITCH
        eye_y = mid_eye[1]
        chin_y = chin[1]
        face_vertical_span = abs(chin_y - eye_y) + 1e-6

        expected_nose_y = eye_y + face_vertical_span * 0.35
        nose_vertical_offset = nose[1] - expected_nose_y

        pitch_2d = np.degrees(np.arctan2(
            nose_vertical_offset,
            face_vertical_span * 0.8
        ))

        forehead_chin_vertical = forehead[1] - chin[1]
        pitch_3d = np.degrees(np.arctan2(
            forehead_chin_vertical,
            abs(depth_offset) + face_vertical_span * 0.3
        ))

        pitch = pitch_2d * 0.65 + pitch_3d * 0.35

        # 4. History
        if face_idx not in self._pose_history:
            self._pose_history[face_idx] = deque(maxlen=30)

        current_pose = {"yaw": float(yaw), "pitch": float(pitch), "roll": float(roll)}
        self._pose_history[face_idx].append(current_pose)

        # Pose variability
        hist = list(self._pose_history[face_idx])
        if len(hist) > 1:
            variability = float(np.mean([
                np.std([p["yaw"] for p in hist]),
                np.std([p["pitch"] for p in hist]),
                np.std([p["roll"] for p in hist])
            ]))
        else:
            variability = 0.0

        # Head turn frequency (используем порог из config)
        if len(hist) >= 3:
            yaw_diffs = [
                abs(hist[i]["yaw"] - hist[i - 1]["yaw"])
                for i in range(1, len(hist))
            ]
            turns = [1.0 if diff > self.head_turn_threshold else 0.0 for diff in yaw_diffs]
            turn_frequency = float(np.mean(turns))
        else:
            turn_frequency = 0.0

        # Attention-to-camera
        attention = float(np.exp(-((yaw / 25.0) ** 2)))

        # Нормализуем углы для компактного представления
        yaw_norm = float(np.clip(yaw / 90.0, -1.0, 1.0))
        pitch_norm = float(np.clip(pitch / 90.0, -1.0, 1.0))
        roll_norm = float(np.clip(roll / 90.0, -1.0, 1.0))

        # Нормализуем looking_direction_vector (unit vector)
        looking_dir_raw = np.array([
            right_eye[0] - left_eye[0],
            right_eye[1] - left_eye[1],
            right_eye[2] - left_eye[2],
        ])
        looking_dir_norm = looking_dir_raw / (np.linalg.norm(looking_dir_raw) + 1e-6)
        looking_direction_vector = [
            float(looking_dir_norm[0]),
            float(looking_dir_norm[1]),
            float(looking_dir_norm[2]),
        ]

        # Confidences (используем detection_confidence как proxy)
        detection_confidence = data.get("detection_confidence", 1.0)
        pose_conf = float(detection_confidence)  # Можно улучшить, анализируя стабильность позы
        landmark_conf = float(detection_confidence)
        tracking_conf = float(detection_confidence)

        return {
            "pose": {
                "yaw": float(yaw),
                "pitch": float(pitch),
                "roll": float(roll),
                "yaw_norm": yaw_norm,
                "pitch_norm": pitch_norm,
                "roll_norm": roll_norm,
                "head_pose_variability": variability,
                "pose_stability_score": float(np.clip(1 - abs(yaw) / 45.0, 0.0, 1.0)),
                "head_turn_frequency": turn_frequency,
                "attention_to_camera_ratio": attention,
                "looking_direction_vector": looking_direction_vector,
                "pose_conf": pose_conf,
                "landmark_conf": landmark_conf,
                "tracking_conf": tracking_conf,
            }
        }

