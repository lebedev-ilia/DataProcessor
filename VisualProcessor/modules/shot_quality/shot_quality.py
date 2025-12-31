"""
shot_quality.py

Production shot/frame quality module.

Key requirements (project-wide contract):
- Frame sampling is owned by Segmenter/DataProcessor and provided via metadata.json.
- This module MUST NOT fallback if dependencies / indices are missing.
- All heavy representations (per-frame CLIP embeddings) are NOT stored for long videos.
  We keep compact per-frame features and per-shot aggregates.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import cv2

import torch

from modules.base_module import BaseModule
from utils.frame_manager import FrameManager


MODULE_NAME = "shot_quality"
VERSION = "2.0"


def _require_npz_key(d: Dict[str, Any], key: str, provider: str) -> Any:
    if key not in d:
        raise RuntimeError(f"{MODULE_NAME} | missing key '{key}' in provider '{provider}' result")
    return d[key]


def _as_int32(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.int32)


def _as_float32(x: Any) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def _ensure_same_indices(expected: np.ndarray, actual: np.ndarray, name: str) -> None:
    if expected.shape != actual.shape or not np.array_equal(expected, actual):
        raise RuntimeError(
            f"{MODULE_NAME} | frame_indices mismatch with {name}. "
            f"Expected shape={expected.shape}, got shape={actual.shape}. "
            "Contract: all core providers must be computed on the same frame_indices as shot_quality."
        )


def _softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / (np.sum(e, axis=axis, keepdims=True) + 1e-9)


# -----------------------------
# Image-quality primitives
# -----------------------------

def sharpness_tenengrad(gray: np.ndarray) -> float:
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    g = gx * gx + gy * gy
    return float(np.mean(g))


def _sharpness_laplacian_var(gray: np.ndarray) -> float:
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _sharpness_smd2(gray: np.ndarray) -> float:
    diff1 = np.abs(gray[:, 1:].astype(np.float32) - gray[:, :-1].astype(np.float32))
    diff2 = np.abs(gray[1:, :].astype(np.float32) - gray[:-1, :].astype(np.float32))
    return float(np.mean(diff1) + np.mean(diff2))


def _edge_clarity_index(frame_bgr: np.ndarray) -> float:
    edges = cv2.Canny(frame_bgr, 100, 200)
    return float(np.mean(edges) / 255.0)


def _focus_gradient_score(gray: np.ndarray) -> float:
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(gx**2 + gy**2)
    return float(np.mean(gradient_magnitude))


def sharpness_secondary(gray: np.ndarray, frame_bgr: np.ndarray) -> float:
    # compact aggregation with mild normalization
    lap = _sharpness_laplacian_var(gray)
    smd = _sharpness_smd2(gray)
    edge = _edge_clarity_index(frame_bgr)
    grad = _focus_gradient_score(gray)

    lap_n = np.tanh(lap / 500.0)
    smd_n = np.tanh(smd / 200.0)
    grad_n = np.tanh(grad / 50.0)
    edge_n = float(np.clip(edge, 0.0, 1.0))

    score = 0.35 * lap_n + 0.25 * smd_n + 0.25 * grad_n + 0.15 * edge_n
    return float(np.clip(score, 0.0, 1.0))


def motion_blur_probability(gray: np.ndarray) -> float:
    fft = np.fft.fft2(gray.astype(np.float32))
    mag = np.log(np.abs(fft) + 1.0)
    blur = 1.0 - (float(np.mean(mag)) / (float(np.max(mag)) + 1e-9))
    return float(np.clip(blur, 0.0, 1.0))


def spatial_frequency_mean(gray: np.ndarray) -> float:
    h, w = gray.shape[:2]
    if h == 0 or w == 0:
        return 0.0
    fft = np.fft.fft2(gray.astype(np.float32))
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    y, x = np.ogrid[:h, :w]
    cy, cx = h // 2, w // 2
    distances = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    diag = float(np.sqrt(h**2 + w**2))
    norm_dist = distances / (diag + 1e-8)
    return float(np.sum(magnitude * norm_dist) / (np.sum(magnitude) + 1e-10))


def noise_level_luma(gray: np.ndarray) -> float:
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    diff = np.mean(np.abs(gray.astype(np.float32) - blur.astype(np.float32))) / 255.0
    return float(np.clip(diff, 0.0, 1.0))


def noise_level_chroma(frame_bgr: np.ndarray) -> float:
    yuv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2YUV)
    u = yuv[:, :, 1].astype(np.float32)
    v = yuv[:, :, 2].astype(np.float32)
    u_blur = cv2.GaussianBlur(u, (3, 3), 0)
    v_blur = cv2.GaussianBlur(v, (3, 3), 0)
    diff = (np.mean(np.abs(u - u_blur)) + np.mean(np.abs(v - v_blur))) / 2.0 / 255.0
    return float(np.clip(diff, 0.0, 1.0))


def iso_estimated_value(noise_luma: float) -> float:
    # map [0..0.1] -> [100..6400], clamp
    x = float(np.clip(noise_luma / 0.1, 0.0, 1.0))
    return float(100.0 + x * (6400.0 - 100.0))


def grain_strength(gray: np.ndarray) -> float:
    hp = cv2.Laplacian(gray, cv2.CV_64F)
    s = float(np.std(hp) / 255.0)
    return float(np.clip(s, 0.0, 1.0))


def noise_spatial_entropy(gray: np.ndarray, block: int = 8) -> float:
    from scipy.stats import entropy  # local import (heavy)

    h, w = gray.shape[:2]
    if h < block or w < block:
        return 0.0
    ent = []
    for y in range(0, h - block + 1, block):
        for x in range(0, w - block + 1, block):
            patch = gray[y : y + block, x : x + block]
            hist = cv2.calcHist([patch], [0], None, [16], [0, 256]).flatten()
            hist = hist / (hist.sum() + 1e-9)
            ent.append(float(entropy(hist)))
    return float(np.mean(ent)) if ent else 0.0


def exposure_metrics(gray: np.ndarray) -> Dict[str, float]:
    p5 = float(np.percentile(gray, 5))
    p95 = float(np.percentile(gray, 95))
    under = float(np.mean(gray < p5))
    over = float(np.mean(gray > p95))
    mid = float(np.mean((gray >= p5) & (gray <= p95)))
    # skew proxy: use normalized distance between p5 and p95 to mean
    mean = float(np.mean(gray))
    skew = float((mean - p5) / (p95 - p5 + 1e-9))
    return {
        "underexposure_ratio": under,
        "overexposure_ratio": over,
        "midtones_balance": mid,
        "exposure_histogram_skewness": skew,
        "highlight_recovery_potential": float(1.0 - over),
        "shadow_recovery_potential": float(1.0 - under),
    }


def contrast_metrics(gray: np.ndarray) -> Dict[str, float]:
    global_contrast = float(np.std(gray))
    local_contrast = float(np.mean(np.abs(cv2.Laplacian(gray, cv2.CV_64F))))
    dyn = float((float(np.max(gray)) - float(np.min(gray))) / 255.0)
    clarity = float(np.clip(local_contrast / 10.0, 0.0, 1.0))
    micro = float(np.std(cv2.Laplacian(gray, cv2.CV_64F)) / 255.0)
    return {
        "contrast_global": global_contrast,
        "contrast_local": local_contrast,
        "contrast_dynamic_range": dyn,
        "contrast_clarity_score": clarity,
        "microcontrast": float(np.clip(micro, 0.0, 1.0)),
    }


def color_metrics(frame_bgr: np.ndarray) -> Dict[str, Any]:
    b, g, r = cv2.split(frame_bgr)
    wb_r = float(np.mean(r))
    wb_g = float(np.mean(g))
    wb_b = float(np.mean(b))
    means = np.array([wb_r, wb_g, wb_b], dtype=np.float32)
    cast_idx = int(np.argmax(means))
    cast_type = ["red", "green", "blue"][cast_idx]
    # Simple color fidelity: entropy of hue histogram
    hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    hist = cv2.calcHist([h], [0], None, [32], [0, 180]).flatten()
    hist = hist / (hist.sum() + 1e-9)
    from scipy.stats import entropy  # local import
    cfi = float(np.clip(entropy(hist) / np.log(32), 0.0, 1.0))
    # Uniformity: inverse of std over HSV channels
    s_std = float(np.std(hsv[:, :, 1]) / 255.0)
    v_std = float(np.std(hsv[:, :, 2]) / 255.0)
    uniform = float(np.clip(1.0 - (s_std + v_std) / 2.0, 0.0, 1.0))
    # Skin-tone accuracy (proxy): fraction of pixels with R>G>B in HSV skin mask
    skin_mask = ((hsv[:, :, 0] >= 0) & (hsv[:, :, 0] <= 25) & (hsv[:, :, 1] > 40) & (hsv[:, :, 2] > 50))
    if np.any(skin_mask):
        rr = r[skin_mask].astype(np.float32)
        gg = g[skin_mask].astype(np.float32)
        bb = b[skin_mask].astype(np.float32)
        ok = float(np.mean((rr > gg) & (gg > bb)))
    else:
        ok = 0.0
    # Color noise level: std of LAB ab residual
    lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    ab = lab[:, :, 1:3]
    ab_blur = cv2.GaussianBlur(ab, (3, 3), 0)
    color_noise = float(np.clip(np.std(ab - ab_blur) / 255.0, 0.0, 1.0))
    return {
        "wb_r": wb_r,
        "wb_g": wb_g,
        "wb_b": wb_b,
        "color_cast_type": cast_type,
        "skin_tone_accuracy_score": ok,
        "color_fidelity_index": cfi,
        "color_noise_level": color_noise,
        "color_uniformity_score": uniform,
    }


def compression_metrics(gray: np.ndarray) -> Dict[str, float]:
    # blockiness: differences across 8px boundaries
    g = gray.astype(np.float32)
    block = float(np.mean(np.abs(g[:, 8:] - g[:, :-8])))
    # banding: low high-frequency energy proxy
    blur = cv2.GaussianBlur(g, (9, 9), 0)
    band = float(np.clip(1.0 - (np.mean(np.abs(g - blur)) / 50.0), 0.0, 1.0))
    # ringing: std of Laplacian-of-Gaussian response
    log = cv2.Laplacian(cv2.GaussianBlur(g, (3, 3), 0), cv2.CV_64F)
    ringing = float(np.clip(np.std(log) / 20.0, 0.0, 1.0))
    bitrate = float(np.clip(1.0 - (block / 50.0 + band) / 2.0, 0.0, 1.0))
    # codec entropy: entropy of block stds
    from scipy.stats import entropy  # local import
    h, w = gray.shape[:2]
    bs = 8
    stds = []
    for y in range(0, h - bs + 1, bs):
        for x in range(0, w - bs + 1, bs):
            stds.append(float(np.std(g[y : y + bs, x : x + bs])))
    if stds:
        hist, _ = np.histogram(stds, bins=20)
        p = hist.astype(np.float32)
        p = p / (p.sum() + 1e-9)
        codec_ent = float(np.clip(entropy(p) / np.log(20), 0.0, 1.0))
    else:
        codec_ent = 0.0
    return {
        "compression_blockiness_score": block,
        "banding_intensity": band,
        "ringing_artifacts_level": ringing,
        "bitrate_estimation_score": bitrate,
        "codec_artifact_entropy": codec_ent,
    }


def lens_metrics(frame_bgr: np.ndarray, gray: np.ndarray) -> Dict[str, Any]:
    h, w = gray.shape[:2]
    if h == 0 or w == 0:
        return {
            "vignetting_level": 0.0,
            "chromatic_aberration_level": 0.0,
            "distortion_type": "none",
            "lens_sharpness_drop_off": 0.0,
            "lens_obstruction_probability": 0.0,
            "lens_dirt_probability": 0.0,
            "veiling_glare_score": 0.0,
        }
    # vignetting: center vs corners brightness
    cy, cx = h // 2, w // 2
    center = gray[max(0, cy - h // 10) : min(h, cy + h // 10), max(0, cx - w // 10) : min(w, cx + w // 10)]
    corners = np.concatenate(
        [
            gray[: h // 10, : w // 10].flatten(),
            gray[: h // 10, -w // 10 :].flatten(),
            gray[-h // 10 :, : w // 10].flatten(),
            gray[-h // 10 :, -w // 10 :].flatten(),
        ]
    )
    vign = float(np.mean(center) - np.mean(corners))
    # chromatic aberration: edge mismatch between R and B
    b, g, r = cv2.split(frame_bgr)
    e_r = cv2.Canny(r, 100, 200).astype(np.float32)
    e_b = cv2.Canny(b, 100, 200).astype(np.float32)
    ca = float(np.mean(np.abs(e_r - e_b)) / 255.0)
    # distortion type: not robust without line detection; keep "none" deterministic
    distortion_type = "none"
    # sharpness drop-off: laplacian center vs corners
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    lap_center = float(np.var(lap[max(0, cy - h // 10) : min(h, cy + h // 10), max(0, cx - w // 10) : min(w, cx + w // 10)]))
    lap_corners = float(np.var(corners.astype(np.float32)))
    drop = float(np.clip(1.0 - (lap_corners / (lap_center + 1e-9)), 0.0, 1.0))
    # obstruction: high local residual ratio
    blur = cv2.GaussianBlur(gray, (9, 9), 0)
    resid = np.abs(gray.astype(np.float32) - blur.astype(np.float32)) / 255.0
    obstruction = float(np.clip(np.mean(resid > 0.5), 0.0, 1.0))
    # dirt: small dark blobs ratio
    thr = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 31, 5)
    dirt = float(np.clip(np.mean(thr > 0), 0.0, 1.0))
    # veiling glare: bright fraction * inverse contrast
    bright = float(np.mean(gray > 220))
    contrast = float(np.std(gray) / 255.0)
    glare = float(np.clip(bright * (1.0 - contrast), 0.0, 1.0))
    return {
        "vignetting_level": vign,
        "chromatic_aberration_level": ca,
        "distortion_type": distortion_type,
        "lens_sharpness_drop_off": drop,
        "lens_obstruction_probability": obstruction,
        "lens_dirt_probability": dirt,
        "veiling_glare_score": glare,
    }


def fog_haziness_score(gray: np.ndarray) -> float:
    lap = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(1.0 / (lap + 1.0))


def temporal_flicker(prev_gray: Optional[np.ndarray], gray: np.ndarray) -> float:
    if prev_gray is None:
        return 0.0
    return float(np.mean(np.abs(prev_gray.astype(np.float32) - gray.astype(np.float32))) / 255.0)


def rolling_shutter_artifacts_score(prev_frame_bgr: Optional[np.ndarray], curr_frame_bgr: np.ndarray) -> float:
    if prev_frame_bgr is None:
        return 0.0
    gray_prev = cv2.cvtColor(prev_frame_bgr, cv2.COLOR_BGR2GRAY)
    gray_curr = cv2.cvtColor(curr_frame_bgr, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(gray_prev, gray_curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    vflow = flow[:, :, 1]
    h, w = vflow.shape[:2]
    strips = 5
    sw = max(1, w // strips)
    strip_means = [float(np.mean(np.abs(vflow[:, i * sw : (i + 1) * sw]))) for i in range(strips)]
    var = float(np.std(strip_means)) if len(strip_means) > 1 else 0.0
    return float(np.clip(var / 5.0, 0.0, 1.0))


def depth_metrics(depth_map: np.ndarray) -> Dict[str, float]:
    if depth_map is None or not np.isfinite(depth_map).any():
        raise RuntimeError(f"{MODULE_NAME} | core_depth_midas produced invalid depth map (NaN/empty)")
    dm = depth_map.astype(np.float32)
    mean = float(np.nanmean(dm))
    std = float(np.nanstd(dm))
    # gradient magnitude mean
    gx = cv2.Sobel(dm, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(dm, cv2.CV_32F, 0, 1, ksize=3)
    grad = float(np.nanmean(np.sqrt(gx * gx + gy * gy)))
    return {"depth_mean": mean, "depth_std": std, "depth_grad_mean": grad}


def _bbox_from_landmarks(face_landmarks_xy: np.ndarray, w: int, h: int) -> Optional[Tuple[int, int, int, int]]:
    # landmarks: (468,3) normalized x,y
    if face_landmarks_xy.size == 0:
        return None
    xs = face_landmarks_xy[:, 0]
    ys = face_landmarks_xy[:, 1]
    if not np.isfinite(xs).any() or not np.isfinite(ys).any():
        return None
    x1 = int(np.clip(np.min(xs) * w, 0, w - 1))
    x2 = int(np.clip(np.max(xs) * w, 0, w - 1))
    y1 = int(np.clip(np.min(ys) * h, 0, h - 1))
    y2 = int(np.clip(np.max(ys) * h, 0, h - 1))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2, y2


@dataclass
class _ShotBoundaries:
    shot_start_frames: np.ndarray  # (S,)
    shot_end_frames: np.ndarray    # (S,)
    shot_ids_for_frames: np.ndarray  # (N,)


class ShotQualityModule(BaseModule):
    def __init__(self, rs_path: Optional[str] = None, device: str = "cuda", **kwargs: Any):
        super().__init__(rs_path=rs_path, logger_name=MODULE_NAME, **kwargs)
        self.device = device

    @property
    def module_name(self) -> str:
        return MODULE_NAME

    def required_dependencies(self) -> List[str]:
        return [
            "core_clip",
            "core_depth_midas",
            "core_object_detections",
            "core_face_landmarks",
            "cut_detection",
        ]

    def _do_initialize(self) -> None:
        if self.device == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"{MODULE_NAME} | device=cuda requested but torch.cuda.is_available() is False")

    def process(self, frame_manager: FrameManager, frame_indices: List[int], config: Dict[str, Any]) -> Dict[str, Any]:
        self.initialize()

        if not frame_indices:
            raise ValueError(f"{MODULE_NAME} | frame_indices is empty")

        frame_indices_np = np.asarray([int(i) for i in frame_indices], dtype=np.int32)

        deps = self.load_all_dependencies()

        core_clip = deps.get("core_clip")
        core_depth = deps.get("core_depth_midas")
        core_det = deps.get("core_object_detections")
        core_lm = deps.get("core_face_landmarks")
        cut_det = deps.get("cut_detection")

        if core_clip is None or core_depth is None or core_det is None or core_lm is None or cut_det is None:
            raise RuntimeError(f"{MODULE_NAME} | missing required dependency results (None)")

        # --- Validate + align indices across core providers ---
        clip_idx = _as_int32(_require_npz_key(core_clip, "frame_indices", "core_clip"))
        clip_emb = _as_float32(_require_npz_key(core_clip, "frame_embeddings", "core_clip"))
        _ensure_same_indices(frame_indices_np, clip_idx, "core_clip")

        depth_idx = _as_int32(_require_npz_key(core_depth, "frame_indices", "core_depth_midas"))
        depth_maps = _as_float32(_require_npz_key(core_depth, "depth_maps", "core_depth_midas"))
        _ensure_same_indices(frame_indices_np, depth_idx, "core_depth_midas")

        det_idx = _as_int32(_require_npz_key(core_det, "frame_indices", "core_object_detections"))
        boxes = _as_float32(_require_npz_key(core_det, "boxes", "core_object_detections"))
        valid_mask = np.asarray(_require_npz_key(core_det, "valid_mask", "core_object_detections"), dtype=bool)
        class_ids = _as_int32(_require_npz_key(core_det, "class_ids", "core_object_detections"))
        _ensure_same_indices(frame_indices_np, det_idx, "core_object_detections")

        lm_idx = _as_int32(_require_npz_key(core_lm, "frame_indices", "core_face_landmarks"))
        face = _as_float32(_require_npz_key(core_lm, "face_landmarks", "core_face_landmarks"))
        face_present = np.asarray(_require_npz_key(core_lm, "face_present", "core_face_landmarks"), dtype=bool)
        has_any_face = bool(np.asarray(_require_npz_key(core_lm, "has_any_face", "core_face_landmarks")).item())
        empty_reason_faces = _require_npz_key(core_lm, "empty_reason", "core_face_landmarks")
        _ensure_same_indices(frame_indices_np, lm_idx, "core_face_landmarks")

        # --- CLIP-based shot-quality probabilities (from core_clip outputs) ---
        prompts = _require_npz_key(core_clip, "shot_quality_prompts", "core_clip")
        txt_emb = _as_float32(_require_npz_key(core_clip, "shot_quality_text_embeddings", "core_clip"))
        if txt_emb.ndim != 2 or clip_emb.ndim != 2 or txt_emb.shape[1] != clip_emb.shape[1]:
            raise RuntimeError(
                f"{MODULE_NAME} | core_clip text/image embedding dim mismatch: "
                f"text={txt_emb.shape}, image={clip_emb.shape}"
            )

        # Adaptive matmul batching on GPU
        img_t = torch.from_numpy(clip_emb)
        txt_t = torch.from_numpy(txt_emb)
        if self.device == "cuda":
            img_t = img_t.to("cuda")
            txt_t = txt_t.to("cuda")
        img_t = img_t.to(torch.float16 if self.device == "cuda" else torch.float32)
        txt_t = txt_t.to(torch.float16 if self.device == "cuda" else torch.float32)

        n = int(frame_indices_np.shape[0])
        p = int(txt_emb.shape[0])
        quality_probs = np.zeros((n, p), dtype=np.float16)

        # choose chunk size based on free GPU memory (very conservative)
        chunk = 2048
        if self.device == "cuda":
            try:
                free_b, _total_b = torch.cuda.mem_get_info()
                # heuristic: allow ~256MB for activations
                if free_b < 2_000_000_000:
                    chunk = 512
                elif free_b < 6_000_000_000:
                    chunk = 1024
            except Exception:
                pass

        with torch.no_grad():
            for start in range(0, n, chunk):
                end = min(n, start + chunk)
                logits = img_t[start:end] @ txt_t.T
                probs = torch.softmax(logits, dim=-1)
                quality_probs[start:end] = probs.detach().cpu().to(torch.float16).numpy()

        # --- Per-frame feature extraction (pixels + depth + detections + face ROI) ---
        feature_names: List[str] = []
        rows: List[np.ndarray] = []

        prev_gray = None
        prev_frame_bgr = None

        # Pre-allocate per-feature arrays in dict, then stack
        feats: Dict[str, np.ndarray] = {}
        def alloc(name: str, dtype=np.float32) -> None:
            feats[name] = np.zeros((n,), dtype=dtype)

        # Scalars
        alloc("sharpness_tenengrad")
        alloc("sharpness_secondary")
        alloc("motion_blur_probability")
        alloc("spatial_frequency_mean")
        alloc("noise_level_luma")
        alloc("noise_level_chroma")
        alloc("noise_chroma_ratio")
        alloc("iso_estimated_value")
        alloc("grain_strength")
        alloc("noise_spatial_entropy")
        # exposure (6)
        for k in [
            "underexposure_ratio","overexposure_ratio","midtones_balance",
            "exposure_histogram_skewness","highlight_recovery_potential","shadow_recovery_potential"
        ]:
            alloc(k)
        # contrast (5)
        for k in [
            "contrast_global","contrast_local","contrast_dynamic_range","contrast_clarity_score","microcontrast"
        ]:
            alloc(k)
        # color (store cast as int code, mapping in meta)
        alloc("wb_r"); alloc("wb_g"); alloc("wb_b")
        alloc("skin_tone_accuracy_score"); alloc("color_fidelity_index"); alloc("color_noise_level"); alloc("color_uniformity_score")
        feats["color_cast_type_id"] = np.zeros((n,), dtype=np.int32)
        # compression (5)
        for k in ["compression_blockiness_score","banding_intensity","ringing_artifacts_level","bitrate_estimation_score","codec_artifact_entropy"]:
            alloc(k)
        # lens (7)
        alloc("vignetting_level")
        alloc("chromatic_aberration_level")
        feats["distortion_type_id"] = np.zeros((n,), dtype=np.int32)  # only "none" for now
        alloc("lens_sharpness_drop_off")
        alloc("lens_obstruction_probability")
        alloc("lens_dirt_probability")
        alloc("veiling_glare_score")
        # fog
        alloc("fog_haziness_score")
        # temporal
        alloc("temporal_flicker_score")
        alloc("rolling_shutter_artifacts_score")
        # depth (3)
        alloc("depth_mean"); alloc("depth_std"); alloc("depth_grad_mean")
        # detections (2)
        feats["objects_count"] = np.zeros((n,), dtype=np.int32)
        alloc("objects_area_mean")
        # face ROI (2)
        alloc("face_sharpness_tenengrad")
        alloc("face_noise_level_luma")

        cast_map = {"red": 0, "green": 1, "blue": 2}
        distortion_map = {"none": 0}

        for i, frame_idx in enumerate(frame_indices_np.tolist()):
            frame_rgb = frame_manager.get(int(frame_idx))
            if frame_rgb.ndim != 3 or frame_rgb.shape[2] != 3:
                raise RuntimeError(f"{MODULE_NAME} | invalid frame shape at idx={frame_idx}: {frame_rgb.shape}")
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
            gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

            feats["sharpness_tenengrad"][i] = sharpness_tenengrad(gray)
            feats["sharpness_secondary"][i] = sharpness_secondary(gray, frame_bgr)
            feats["motion_blur_probability"][i] = motion_blur_probability(gray)
            feats["spatial_frequency_mean"][i] = spatial_frequency_mean(gray)

            nl = noise_level_luma(gray)
            nc = noise_level_chroma(frame_bgr)
            feats["noise_level_luma"][i] = nl
            feats["noise_level_chroma"][i] = nc
            feats["noise_chroma_ratio"][i] = float(nc / (nl + 1e-8))
            feats["iso_estimated_value"][i] = iso_estimated_value(nl)
            feats["grain_strength"][i] = grain_strength(gray)
            feats["noise_spatial_entropy"][i] = noise_spatial_entropy(gray)

            exp = exposure_metrics(gray)
            for k, v in exp.items():
                feats[k][i] = float(v)
            con = contrast_metrics(gray)
            for k, v in con.items():
                feats[k][i] = float(v)
            col = color_metrics(frame_bgr)
            feats["wb_r"][i] = float(col["wb_r"])
            feats["wb_g"][i] = float(col["wb_g"])
            feats["wb_b"][i] = float(col["wb_b"])
            feats["color_cast_type_id"][i] = int(cast_map[col["color_cast_type"]])
            feats["skin_tone_accuracy_score"][i] = float(col["skin_tone_accuracy_score"])
            feats["color_fidelity_index"][i] = float(col["color_fidelity_index"])
            feats["color_noise_level"][i] = float(col["color_noise_level"])
            feats["color_uniformity_score"][i] = float(col["color_uniformity_score"])

            comp = compression_metrics(gray)
            for k, v in comp.items():
                feats[k][i] = float(v)
            lens = lens_metrics(frame_bgr, gray)
            feats["vignetting_level"][i] = float(lens["vignetting_level"])
            feats["chromatic_aberration_level"][i] = float(lens["chromatic_aberration_level"])
            feats["distortion_type_id"][i] = int(distortion_map[lens["distortion_type"]])
            feats["lens_sharpness_drop_off"][i] = float(lens["lens_sharpness_drop_off"])
            feats["lens_obstruction_probability"][i] = float(lens["lens_obstruction_probability"])
            feats["lens_dirt_probability"][i] = float(lens["lens_dirt_probability"])
            feats["veiling_glare_score"][i] = float(lens["veiling_glare_score"])

            feats["fog_haziness_score"][i] = fog_haziness_score(gray)
            feats["temporal_flicker_score"][i] = temporal_flicker(prev_gray, gray)
            feats["rolling_shutter_artifacts_score"][i] = rolling_shutter_artifacts_score(prev_frame_bgr, frame_bgr)
            prev_gray = gray
            prev_frame_bgr = frame_bgr

            dm = depth_metrics(depth_maps[i])
            feats["depth_mean"][i] = dm["depth_mean"]
            feats["depth_std"][i] = dm["depth_std"]
            feats["depth_grad_mean"][i] = dm["depth_grad_mean"]

            # object detections summary (area in normalized units)
            vm = valid_mask[i]
            bxs = boxes[i][vm]
            feats["objects_count"][i] = int(bxs.shape[0])
            if bxs.size:
                # assume xyxy in pixels; normalize by frame area
                x1y1 = bxs[:, :2]
                x2y2 = bxs[:, 2:4]
                wh = np.clip(x2y2 - x1y1, 0.0, None)
                areas = wh[:, 0] * wh[:, 1] / float(frame_bgr.shape[0] * frame_bgr.shape[1] + 1e-9)
                feats["objects_area_mean"][i] = float(np.mean(areas))
            else:
                feats["objects_area_mean"][i] = 0.0

            # face ROI metrics (use first face only). Valid empty output is allowed:
            # if no faces detected -> keep NaN for face_* features.
            h, w = frame_bgr.shape[:2]
            if face_present.ndim >= 2 and face_present[i, 0]:
                face_lm = face[i, 0]  # (468,3)
                bb = _bbox_from_landmarks(face_lm, w=w, h=h)
                if bb is not None:
                    x1, y1, x2, y2 = bb
                    roi = gray[y1:y2, x1:x2]
                    feats["face_sharpness_tenengrad"][i] = sharpness_tenengrad(roi) if roi.size else np.nan
                    feats["face_noise_level_luma"][i] = noise_level_luma(roi) if roi.size else np.nan
                else:
                    feats["face_sharpness_tenengrad"][i] = np.nan
                    feats["face_noise_level_luma"][i] = np.nan
            else:
                feats["face_sharpness_tenengrad"][i] = np.nan
                feats["face_noise_level_luma"][i] = np.nan

        # Build frame feature matrix in stable order
        ordered_keys: List[str] = [
            # sharpness
            "sharpness_tenengrad","sharpness_secondary","motion_blur_probability","spatial_frequency_mean",
            # noise
            "noise_level_luma","noise_level_chroma","noise_chroma_ratio","iso_estimated_value","grain_strength","noise_spatial_entropy",
            # exposure
            "underexposure_ratio","overexposure_ratio","midtones_balance","exposure_histogram_skewness","highlight_recovery_potential","shadow_recovery_potential",
            # contrast
            "contrast_global","contrast_local","contrast_dynamic_range","contrast_clarity_score","microcontrast",
            # color
            "wb_r","wb_g","wb_b","color_cast_type_id","skin_tone_accuracy_score","color_fidelity_index","color_noise_level","color_uniformity_score",
            # compression
            "compression_blockiness_score","banding_intensity","ringing_artifacts_level","bitrate_estimation_score","codec_artifact_entropy",
            # lens
            "vignetting_level","chromatic_aberration_level","distortion_type_id","lens_sharpness_drop_off","lens_obstruction_probability","lens_dirt_probability","veiling_glare_score",
            # fog
            "fog_haziness_score",
            # temporal
            "temporal_flicker_score","rolling_shutter_artifacts_score",
            # depth
            "depth_mean","depth_std","depth_grad_mean",
            # objects
            "objects_count","objects_area_mean",
            # face
            "face_sharpness_tenengrad","face_noise_level_luma",
        ]

        feature_names = ordered_keys
        frame_features = np.stack(
            [
                feats[k].astype(np.float32) if feats[k].dtype != np.int32 else feats[k].astype(np.float32)
                for k in ordered_keys
            ],
            axis=1,
        )

        # --- Shot segmentation from cut_detection results ---
        # cut_detection returns indices as positions in its own frame_indices list.
        detections = cut_det.get("detections")
        if not isinstance(detections, dict):
            raise RuntimeError(f"{MODULE_NAME} | cut_detection results missing 'detections' dict")

        hard_pos = detections.get("hard_cut_indices")
        if not isinstance(hard_pos, list):
            raise RuntimeError(f"{MODULE_NAME} | cut_detection.detections.hard_cut_indices missing/invalid")

        # Reconstruct cut_detection frame_indices from metadata (Segmenter contract)
        meta_full = getattr(frame_manager, "meta", None)
        if not isinstance(meta_full, dict):
            raise RuntimeError(f"{MODULE_NAME} | FrameManager.meta is missing; cannot align cut_detection")
        cd_block = meta_full.get("cut_detection")
        if not isinstance(cd_block, dict) or "frame_indices" not in cd_block:
            raise RuntimeError(f"{MODULE_NAME} | metadata missing cut_detection.frame_indices (required for alignment)")
        cd_indices = np.asarray([int(x) for x in cd_block.get("frame_indices")], dtype=np.int32)
        if cd_indices.size == 0:
            raise RuntimeError(f"{MODULE_NAME} | cut_detection.frame_indices empty")

        # Convert cut positions -> global frame indices (cut at cd_indices[pos])
        cut_frames = []
        for pos in hard_pos:
            p_int = int(pos)
            if p_int < 0 or p_int >= int(cd_indices.shape[0]):
                raise RuntimeError(f"{MODULE_NAME} | cut_detection hard_cut_indices out of bounds: {p_int}")
            cut_frames.append(int(cd_indices[p_int]))
        cut_frames = sorted(set(cut_frames))

        # Shot boundaries in global frame indices
        start_frames = [int(frame_indices_np[0])]
        for cf in cut_frames:
            if cf > start_frames[-1]:
                start_frames.append(cf)
        start_frames = sorted(set(start_frames))
        # end frames are next start - 1 (in global frame index space, approximate)
        end_frames = start_frames[1:] + [int(frame_indices_np[-1])]
        shot_start_frames = np.asarray(start_frames, dtype=np.int32)
        shot_end_frames = np.asarray(end_frames, dtype=np.int32)

        # Assign each frame to shot via bisect over start_frames
        import bisect
        shot_ids = np.zeros((n,), dtype=np.int32)
        for i, fi in enumerate(frame_indices_np.tolist()):
            sid = bisect.bisect_right(start_frames, int(fi)) - 1
            shot_ids[i] = int(max(0, sid))

        s = int(shot_start_frames.shape[0])

        # per-shot aggregates over frame_features
        shot_mean = np.zeros((s, frame_features.shape[1]), dtype=np.float32)
        shot_std = np.zeros((s, frame_features.shape[1]), dtype=np.float32)
        shot_min = np.zeros((s, frame_features.shape[1]), dtype=np.float32)
        shot_max = np.zeros((s, frame_features.shape[1]), dtype=np.float32)
        shot_counts = np.zeros((s,), dtype=np.int32)
        for sid in range(s):
            mask = shot_ids == sid
            if not np.any(mask):
                raise RuntimeError(f"{MODULE_NAME} | empty shot segment sid={sid} after alignment")
            seg = frame_features[mask]
            shot_counts[sid] = int(seg.shape[0])
            shot_mean[sid] = np.nanmean(seg, axis=0)
            shot_std[sid] = np.nanstd(seg, axis=0)
            shot_min[sid] = np.nanmin(seg, axis=0)
            shot_max[sid] = np.nanmax(seg, axis=0)

        meta_out = {
            "producer": MODULE_NAME,
            "version": VERSION,
            "created_at": datetime.utcnow().isoformat(),
            "frame_count": int(n),
            "shot_count": int(s),
            "clip_model_name": str(_require_npz_key(core_clip, "model_name", "core_clip")) if "model_name" in core_clip else None,
            "cast_type_map": cast_map,
            "distortion_type_map": distortion_map,
            "quality_prompts": [str(x) for x in np.asarray(prompts, dtype=object).tolist()],
            "faces_available": has_any_face,
            "faces_empty_reason": None if empty_reason_faces is None else str(np.asarray(empty_reason_faces, dtype=object).item()),
            "note_empty_faces": "If no faces detected, face_* features are NaN and face_present is False. This is a valid output (provider ran successfully).",
        }

        return {
            # index
            "frame_indices": frame_indices_np,
            # frame-level arrays
            "feature_names": np.asarray(feature_names, dtype=object),
            "frame_features": frame_features,
            "quality_probs": quality_probs,
            # shot segmentation + aggregates
            "shot_ids": shot_ids,
            "shot_start_frame": shot_start_frames,
            "shot_end_frame": shot_end_frames,
            "shot_frame_count": shot_counts,
            "shot_features_mean": shot_mean,
            "shot_features_std": shot_std,
            "shot_features_min": shot_min,
            "shot_features_max": shot_max,
            # meta
            "meta": np.asarray(meta_out, dtype=object),
        }


