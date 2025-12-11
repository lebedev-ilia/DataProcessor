"""
Утилиты для работы с bounding box лица.
"""

import numpy as np


def compute_bbox(coords: np.ndarray, width: int, height: int) -> np.ndarray:
    """
    Computes bounding box (x_min, y_min, x_max, y_max) from landmark coords.

    Args:
        coords: ndarray of shape (N, 3)
        width, height: frame dimensions.

    Returns:
        ndarray: (4,) float32
    """
    if coords.size == 0:
        return np.array([0, 0, 0, 0], dtype=np.float32)

    x_min = np.clip(coords[:, 0].min(), 0, width - 1)
    y_min = np.clip(coords[:, 1].min(), 0, height - 1)
    x_max = np.clip(coords[:, 0].max(), 0, width - 1)
    y_max = np.clip(coords[:, 1].max(), 0, height - 1)

    # Prevent zero-area bbox
    if x_max <= x_min:
        x_max = min(x_min + 1, width - 1)

    if y_max <= y_min:
        y_max = min(y_min + 1, height - 1)

    return np.array([x_min, y_min, x_max, y_max], dtype=np.float32)


def extract_roi(frame: np.ndarray, bbox: np.ndarray) -> np.ndarray:
    """
    Safely extracts ROI from the frame using the bounding box.
    Handles:
    - negative or reversed bbox
    - coordinates outside frame
    - degenerate ROIs
    """
    if frame is None or frame.size == 0:
        return np.zeros((1, 1, 3), dtype=np.uint8)

    if bbox is None or bbox.size != 4:
        return np.zeros((1, 1, 3), dtype=frame.dtype)

    # Convert to ints safely
    x_min, y_min, x_max, y_max = map(int, bbox)

    # Fix reversed bbox: (x_max < x_min) or (y_max < y_min)
    if x_max < x_min:
        x_min, x_max = x_max, x_min
    if y_max < y_min:
        y_min, y_max = y_max, y_min

    # Clip to frame
    h, w = frame.shape[:2]
    x_min = max(0, min(w - 1, x_min))
    x_max = max(0, min(w,     x_max))
    y_min = max(0, min(h - 1, y_min))
    y_max = max(0, min(h,     y_max))

    # Check area (ROI must be at least 2x2)
    if (x_max - x_min) < 2 or (y_max - y_min) < 2:
        return np.zeros((1, 1, 3), dtype=frame.dtype)

    # Extract region
    return frame[y_min:y_max, x_min:x_max].copy()

