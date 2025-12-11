"""
Утилиты для работы с landmarks лица.
"""

import math
from typing import Optional, Dict, Any
import numpy as np

# Landmark indices из MediaPipe Face Mesh
LANDMARKS = {
    "nose_tip": 1,
    "chin": 152,
    "left_eye_inner": 133,
    "left_eye_outer": 33,
    "right_eye_inner": 362,
    "right_eye_outer": 263,
    "left_eye_upper": 159,
    "left_eye_lower": 145,
    "right_eye_upper": 386,
    "right_eye_lower": 374,
    "upper_lip": 13,
    "lower_lip": 14,
    "mouth_left": 61,
    "mouth_right": 291,
    "left_brow": 70,
    "right_brow": 300,
    "left_cheek": 234,
    "right_cheek": 454,
    "forehead": 10,
}

# MediaPipe Face Mesh lip contour indices (for 468 landmarks with refine_landmarks=True)
# Upper lip: 12-16, 17-21 (outer and inner)
# Lower lip: 14, 15, 16, 17, 18, 19, 20, 21 (outer and inner)
# Full lip contour from FACEMESH_LIPS
LIP_LANDMARKS = {
    # Outer lip contour (approximate indices for MediaPipe 468)
    "upper_lip_outer": [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318],
    "lower_lip_outer": [78, 95, 88, 178, 87, 14, 317, 402, 318, 324],
    "mouth_corner_left": 61,
    "mouth_corner_right": 291,
    # Inner lip contour
    "upper_lip_inner": [78, 191, 80, 81, 82, 13, 312, 311, 310, 415],
    "lower_lip_inner": [78, 95, 88, 178, 87, 14, 317, 402, 318, 324],
}


def landmarks_to_ndarray(landmark_list, width: int, height: int, scale_z: bool = True) -> np.ndarray:
    """
    Converts MediaPipe landmarks to ndarray of shape (N, 3).

    Args:
        landmark_list: MediaPipe landmark proto.
        width, height: frame dimensions.
        scale_z: whether to scale z-coordinate by image diagonal.

    Returns:
        ndarray: (num_landmarks, 3)
    """
    if not landmark_list or not landmark_list.landmark:
        return np.zeros((0, 3), dtype=np.float32)

    diag = math.sqrt(width**2 + height**2)

    coords = np.array([
        [
            lm.x * width,
            lm.y * height,
            lm.z * (diag if scale_z else 1.0)
        ]
        for lm in landmark_list.landmark
    ], dtype=np.float32)

    return coords


def validate_face_landmarks(
    bbox: np.ndarray,
    coords: np.ndarray,
    width: int,
    height: int,
    min_face_size: int = 30,
    max_face_size_ratio: float = 0.8,
    min_aspect_ratio: float = 0.6,
    max_aspect_ratio: float = 1.4,
    validate_landmarks: bool = True,
) -> bool:
    """
    Validate face detection to filter out false positives.

    Checks:
    - reasonable face size (min/max)
    - aspect ratio
    - key landmark presence
    - landmarks inside bbox (with margin)
    - eye symmetry
    """
    # --- Safety checks ---
    if bbox is None or bbox.size != 4:
        return False
    if coords is None or coords.size == 0:
        return False

    x_min, y_min, x_max, y_max = bbox.astype(float)
    face_width = max(0.0, x_max - x_min)
    face_height = max(0.0, y_max - y_min)

    # --- Basic size constraints ---
    if face_width < min_face_size or face_height < min_face_size:
        return False

    max_dim = float(max(width, height))
    if face_width > max_dim * max_face_size_ratio:
        return False
    if face_height > max_dim * max_face_size_ratio:
        return False

    # --- Aspect ratio check ---
    if face_height == 0:
        return False

    aspect_ratio = face_width / face_height
    if not (min_aspect_ratio <= aspect_ratio <= max_aspect_ratio):
        return False

    # --- Landmark-based validation ---
    if validate_landmarks:
        # Key landmarks required for a valid face
        required = [
            "left_eye_inner",
            "right_eye_inner",
            "nose_tip",
            "mouth_left",
            "mouth_right",
        ]

        margin = min(face_width, face_height) * 0.15

        for name in required:
            idx = LANDMARKS.get(name)
            if idx is None or idx >= len(coords):
                return False

            x, y = coords[idx][:2]

            # Expanded box check
            if (
                x < x_min - margin
                or x > x_max + margin
                or y < y_min - margin
                or y > y_max + margin
            ):
                return False

        # --- Eye symmetry check ---
        try:
            left_eye_y = coords[LANDMARKS["left_eye_inner"]][1]
            right_eye_y = coords[LANDMARKS["right_eye_inner"]][1]
        except Exception:
            return False

        eye_diff = abs(left_eye_y - right_eye_y)

        # Allow slightly larger tolerance (15% → 18%)
        if eye_diff > face_height * 0.18:
            return False

    return True

