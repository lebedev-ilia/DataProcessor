"""
cut_detection: lightweight visual features (histogram / SSIM / image helpers).
"""

from __future__ import annotations

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def frame_histogram_diff(frameA: np.ndarray, frameB: np.ndarray, bins: int = 32) -> float:
    """Compute histogram difference (normalized) between two RGB frames."""
    hsvA = cv2.cvtColor(frameA, cv2.COLOR_RGB2HSV)
    hsvB = cv2.cvtColor(frameB, cv2.COLOR_RGB2HSV)
    hA = cv2.calcHist([hsvA], [2], None, [bins], [0, 256]).flatten()
    hB = cv2.calcHist([hsvB], [2], None, [bins], [0, 256]).flatten()
    hA = hA / (hA.sum() + 1e-9)
    hB = hB / (hB.sum() + 1e-9)
    return float(np.linalg.norm(hA - hB, ord=1))


def frame_ssim(frameA: np.ndarray, frameB: np.ndarray) -> float:
    """Convert to gray and compute SSIM (0..1). Return 1-SSIM as drop measure."""
    grayA = cv2.cvtColor(frameA, cv2.COLOR_RGB2GRAY)
    grayB = cv2.cvtColor(frameB, cv2.COLOR_RGB2GRAY)
    try:
        dr = float(grayB.max() - grayB.min())
        dr = dr if dr > 1e-9 else 1.0
        s = ssim(grayA, grayB, data_range=dr)
        return float(1.0 - s)
    except Exception:
        return 0.0


def ImageFromRGB(frame_rgb):
    try:
        from PIL import Image

        return Image.fromarray(frame_rgb)
    except Exception:
        return frame_rgb


def ImageFromCV(frame: np.ndarray):
    # FrameManager.get() returns RGB, so we must NOT assume BGR here.
    return ImageFromRGB(frame)


