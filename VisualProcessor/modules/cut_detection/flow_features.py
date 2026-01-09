"""
cut_detection: optical-flow related utilities.

Separated from the main module to keep files small and reusable.
"""

from __future__ import annotations

from typing import Optional, Tuple

import cv2
import numpy as np


def optical_flow_magnitude(prev_gray: np.ndarray, gray: np.ndarray) -> Tuple[float, Optional[np.ndarray], Optional[np.ndarray]]:
    """Farneback optical flow average magnitude."""
    try:
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        return float(np.mean(mag)), mag, ang
    except Exception:
        return 0.0, None, None


def resize_gray_max_side(gray: np.ndarray, max_side: int) -> np.ndarray:
    """
    Deterministic performance knob:
    - if max_side <= 0: no resize
    - else if max(H,W) > max_side: downscale with aspect ratio preserved
    """
    try:
        ms = int(max_side)
    except Exception:
        ms = 0
    if ms <= 0:
        return gray
    try:
        h, w = int(gray.shape[0]), int(gray.shape[1])
    except Exception:
        return gray
    m = max(h, w)
    if m <= ms:
        return gray
    s = float(ms) / float(max(1, m))
    nh = max(3, int(round(float(h) * s)))
    nw = max(3, int(round(float(w) * s)))
    if nh == h and nw == w:
        return gray
    try:
        return cv2.resize(gray, (nw, nh), interpolation=cv2.INTER_AREA)
    except Exception:
        return gray


def optical_flow_direction_consistency(flow_angles: Optional[np.ndarray], window_size: int = 5) -> float:
    """Compute direction consistency using circular statistics."""
    if flow_angles is None:
        return 0.0
    cos_angles = np.cos(flow_angles)
    sin_angles = np.sin(flow_angles)
    mean_cos = np.mean(cos_angles)
    mean_sin = np.mean(sin_angles)
    consistency = np.sqrt(mean_cos**2 + mean_sin**2)
    return float(consistency)


def estimate_global_motion_homography(prev_gray: np.ndarray, gray: np.ndarray):
    """
    Estimate global camera motion using RANSAC homography.
    Returns homography matrix and inlier ratio.
    """
    try:
        orb = cv2.ORB_create(nfeatures=500)
        kp1, des1 = orb.detectAndCompute(prev_gray, None)
        kp2, des2 = orb.detectAndCompute(gray, None)

        if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
            return None, 0.0

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)

        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)

        if len(good_matches) < 10:
            return None, 0.0

        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        inlier_ratio = float(np.sum(mask)) / len(good_matches) if mask is not None else 0.0
        return H, inlier_ratio
    except Exception:
        return None, 0.0


