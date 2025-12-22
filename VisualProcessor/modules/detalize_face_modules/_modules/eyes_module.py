"""
Модуль для извлечения фич глаз.
"""

from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, deque
import numpy as np

from _modules.base_module import FaceModule
from _utils.landmarks_utils import LANDMARKS
from _utils.face_helpers import safe_distance


def _estimate_blink_rate(history: deque, blink_flags: deque, fps: float = 30.0) -> Tuple[float, float]:
    """
    Улучшенная оценка частоты моргания с использованием hysteresis.
    
    :return: (blink_rate, last_blink_timestamp)
    """
    if len(history) < 2:
        return 0.0, 0.0
    
    history_list = list(history)
    blink_flags_list = list(blink_flags)
    
    # Подсчитываем количество морганий
    blink_count = sum(blink_flags_list)
    
    # Частота моргания (blinks per minute)
    if len(history_list) > 0:
        time_window_seconds = len(history_list) / fps
        blink_rate_per_minute = float((blink_count / max(time_window_seconds, 1e-6)) * 60.0)
    else:
        blink_rate_per_minute = 0.0
    
    # Последний timestamp моргания
    last_blink_idx = -1
    for i in range(len(blink_flags_list) - 1, -1, -1):
        if blink_flags_list[i]:
            last_blink_idx = i
            break
    
    last_blink_timestamp = float(last_blink_idx / fps) if last_blink_idx >= 0 else 0.0
    
    return blink_rate_per_minute, last_blink_timestamp


class EyesModule(FaceModule):
    """
    Модуль для извлечения фич глаз.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.fps = self.config.get("fps", 30.0)
        self.blink_window_size = self.config.get("blink_window_size", int(self.fps * 1.5))  # 1-2 секунды
        self._blink_history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.blink_window_size)
        )
        self._blink_flags: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=self.blink_window_size)
        )
        self._last_blink_timestamp: Dict[int, float] = defaultdict(float)

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
        
        # Нормализуем открытие глаз (0-1)
        eye_opening_left_norm = float(np.clip(left_open / 20.0, 0.0, 1.0))
        eye_opening_right_norm = float(np.clip(right_open / 20.0, 0.0, 1.0))
        eye_opening_avg_norm = (eye_opening_left_norm + eye_opening_right_norm) / 2.0

        # Save to history
        self._blink_history[face_idx].append(avg_open)
        
        # Blink detection с hysteresis
        # Пороги для обнаружения моргания
        if len(self._blink_history[face_idx]) >= 2:
            history_list = list(self._blink_history[face_idx])
            mean_open = np.mean(history_list)
            std_open = np.std(history_list) if len(history_list) > 1 else 0.0
            
            # Моргание: резкое падение открытия глаз ниже порога
            blink_threshold = mean_open - 0.5 * std_open
            is_blinking = avg_open < blink_threshold
            
            # Hysteresis: предотвращаем множественные детекции одного моргания
            if len(self._blink_flags[face_idx]) > 0:
                last_was_blinking = self._blink_flags[face_idx][-1]
                # Если предыдущий кадр уже был морганием, требуем более низкий порог для продолжения
                if last_was_blinking:
                    blink_threshold = mean_open - 0.7 * std_open
                    is_blinking = avg_open < blink_threshold
            
            self._blink_flags[face_idx].append(is_blinking)
        else:
            self._blink_flags[face_idx].append(False)

        # Blink rate (blinks per minute)
        fps = self.fps
        blink_rate, last_blink_timestamp = _estimate_blink_rate(
            self._blink_history[face_idx],
            self._blink_flags[face_idx],
            fps
        )
        self._last_blink_timestamp[face_idx] = last_blink_timestamp
        
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

        # eye_redness_prob удален - не нужен для модели (чувствительный/irrelevant)

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
                "eye_opening_left": eye_opening_left_norm,
                "eye_opening_right": eye_opening_right_norm,
                "blink_rate": blink_rate,  # blinks per minute
                "blink_intensity": blink_intensity,
                "blink_flag": bool(self._blink_flags[face_idx][-1]) if len(self._blink_flags[face_idx]) > 0 else False,
                "last_blink_timestamp": last_blink_timestamp,
                "gaze_vector": gaze_vector,
                "gaze_at_camera_prob": gaze_at_camera_prob,
                "attention_score": float((gaze_at_camera_prob + eye_opening_avg_norm) / 2.0),
                "iris_position": iris_position,
            }
        }

