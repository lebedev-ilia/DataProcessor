"""
cut_detection.py
Cut / Transition detection and shot statistics pipeline (module 5.1).

Features produced (examples):
- hard_cuts_count, hard_cuts_per_minute, hard_cut_strength_mean
- fade_in_count, fade_out_count, dissolve_count, avg_fade_duration
- whip_pan_transitions_count, zoom_transition_count, motion_cut_intensity_score
- transition_wipe_count, transition_slide_count, transition_glitch_count...
- cuts_per_minute, median_cut_interval, cut_interval_std, cut_interval_cv, cut_interval_entropy
- avg_shot_length, median_shot_length, short_shots_ratio, long_shots_ratio, very_long_shots_count
- jump_cuts_count, jump_cut_ratio_per_minute, jump_cut_intensity
- scene_count, avg_scene_length, scene_to_shot_ratio
- audio_cut_alignment_score, audio_spike_cut_ratio
- edit_style_* probabilities (zero-shot via CLIP text prompts)
...


"""

import os, sys
_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if _path not in sys.path:
    sys.path.append(_path)
    
import cv2
import math
import time
import hashlib
from collections import deque
from typing import Dict, List, Any, Optional
import numpy as np

import scipy.stats
from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter1d

try:
    import torch  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore

try:
    from torchvision import models  # type: ignore
    import torchvision.transforms as T  # type: ignore
except Exception:  # pragma: no cover
    models = None  # type: ignore
    T = None  # type: ignore

try:
    import librosa  # type: ignore
except Exception:  # pragma: no cover
    librosa = None  # type: ignore

try:
    import clip  # type: ignore
except Exception:  # pragma: no cover
    clip = None  # type: ignore

from skimage.metrics import structural_similarity as ssim
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler

# Split large helper blocks into smaller files (keep public API stable).
from modules.cut_detection.visual_features import ImageFromCV, ImageFromRGB, frame_histogram_diff, frame_ssim  # noqa: E402
from modules.cut_detection.flow_features import (  # noqa: E402
    estimate_global_motion_homography,
    optical_flow_direction_consistency,
    optical_flow_magnitude,
    resize_gray_max_side as _resize_gray_max_side,
)

from modules.base_module import BaseModule
from utils.frame_manager import FrameManager
from utils.logger import get_logger

NAME = "CutDetectionPipeline"
VERSION = "2.0"
SCHEMA_VERSION = "cut_detection_npz_v1"

logger = get_logger(NAME)


def float_or_zero(x):
    return float(x) if np.isfinite(x) else 0.0

def seconds_from_fps(n_frames, fps):
    return n_frames / float(fps) if fps > 0 else 0.0

def _require_union_times_s(frame_manager: FrameManager, frame_indices: List[int]) -> np.ndarray:
    """
    Segmenter contract: union_timestamps_sec is source-of-truth for time axis.
    No-fallback: if missing/invalid -> error (production).
    """
    meta = getattr(frame_manager, "meta", None)
    if not isinstance(meta, dict):
        raise RuntimeError("cut_detection | FrameManager.meta missing (requires union_timestamps_sec)")
    ts = meta.get("union_timestamps_sec")
    if not isinstance(ts, list) or not ts:
        raise RuntimeError("cut_detection | union_timestamps_sec missing/empty in frames metadata (no-fallback)")
    uts = np.asarray(ts, dtype=np.float32)
    fi = np.asarray([int(i) for i in frame_indices], dtype=np.int32)
    if fi.size == 0:
        raise RuntimeError("cut_detection | frame_indices is empty (no-fallback)")
    if int(np.max(fi)) >= int(uts.shape[0]):
        raise RuntimeError("cut_detection | union_timestamps_sec does not cover frame_indices (no-fallback)")
    times_s = uts[fi]
    # enforce monotonic non-decreasing
    if times_s.size >= 2 and np.any(np.diff(times_s) < -1e-3):
        raise RuntimeError("cut_detection | union_timestamps_sec is not monotonic (no-fallback)")
    return times_s.astype(np.float32)


def _video_length_seconds(times_s: np.ndarray) -> float:
    if times_s.size == 0:
        return 0.0
    if times_s.size == 1:
        return 0.0
    return float(max(times_s[-1] - times_s[0], 0.0))


def _resolve_audio_path(frames_dir: str, config: Optional[Dict[str, Any]]) -> Optional[str]:
    """
    Auto-resolve Segmenter audio path (audio/audio.wav) relative to frames_dir (video/).
    User may override with config['audio_path'].
    """
    if isinstance(config, dict):
        p = config.get("audio_path")
        if isinstance(p, str) and p:
            return p
    # frames_dir usually ends with ".../<video_id>/video"
    base = os.path.dirname(os.path.abspath(frames_dir))
    cand = os.path.join(base, "audio", "audio.wav")
    return cand if os.path.exists(cand) else None


def _resolve_clip_download_root(config: Optional[Dict[str, Any]]) -> Optional[str]:
    """
    DEPRECATED (baseline GPU-only):
    `cut_detection` MUST NOT load CLIP weights locally (no-network, single source-of-truth).
    CLIP is served via Triton and resolved via `dp_models` + `core_clip` artifacts.
    """
    return None


def _cut_timing_statistics_from_times(cut_times_s: List[float], video_length_s: float) -> Dict[str, Any]:
    """
    Stats for cut timing (time-based, not fps-based).
    """
    if not cut_times_s:
        return {
            "cuts_per_minute": 0.0,
            "median_cut_interval": float("nan"),
            "min_cut_interval": float("nan"),
            "max_cut_interval": float("nan"),
            "cut_interval_std": float("nan"),
            "cut_interval_cv": float("nan"),
            "cut_interval_entropy": float("nan"),
            "cut_rhythm_uniformity_score": float("nan"),
        }
    if video_length_s <= 0:
        return {
            "cuts_per_minute": float("nan"),
            "median_cut_interval": float("nan"),
            "min_cut_interval": float("nan"),
            "max_cut_interval": float("nan"),
            "cut_interval_std": float("nan"),
            "cut_interval_cv": float("nan"),
            "cut_interval_entropy": float("nan"),
            "cut_rhythm_uniformity_score": float("nan"),
        }
    t = np.asarray(sorted(set(float(x) for x in cut_times_s)), dtype=np.float32)
    intervals = np.diff(t)
    if intervals.size == 0:
        intervals = np.asarray([float(video_length_s)], dtype=np.float32)
    cpm = float(len(t) / max(video_length_s, 1e-6) * 60.0)
    median = float(np.median(intervals))
    mn = float(np.min(intervals))
    mx = float(np.max(intervals))
    std = float(np.std(intervals))
    mean_int = float(np.mean(intervals))
    cv = float(std / (mean_int + 1e-9))
    n_bins = int(min(20, max(2, intervals.size)))
    hist, _ = np.histogram(intervals, bins=n_bins)
    hist = hist.astype(np.float64) + 1e-9
    ent = float(scipy.stats.entropy(hist))
    max_entropy = float(np.log(n_bins)) if n_bins > 1 else 1.0
    ent_normalized = float(ent / (max_entropy + 1e-9))
    cv_clipped = float(np.clip(cv, 0.0, 1.0))
    uniformity = float(1.0 - cv_clipped)
    return {
        "cuts_per_minute": cpm,
        "median_cut_interval": median,
        "min_cut_interval": mn,
        "max_cut_interval": mx,
        "cut_interval_std": std,
        "cut_interval_cv": cv,
        "cut_interval_entropy": ent_normalized,
        "cut_rhythm_uniformity_score": uniformity,
    }


def get_embedding_model(device='cpu', model_name='resnet18'):
    """Initialize and return a pre-trained embedding model."""
    if torch is None or models is None or T is None:
        raise RuntimeError("cut_detection | torch/torchvision is required for deep features (use_deep_features=true)")

    if model_name == 'resnet18':
        model = models.resnet18(pretrained=True)
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
    else:
        raise ValueError(f"cut_detection | unsupported embedding model: {model_name}")

    model = torch.nn.Sequential(*list(model.children())[:-1])  # remove final FC
    model.eval().to(device)
    transform = T.Compose(
        [
            T.Resize((224, 224)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return model, transform

def feature_embedding_diff(frameA, frameB, embed_model=None, transform=None, device='cpu'):
    """Compute cosine distance between deep embeddings (ResNet/ViT)."""
    if embed_model is None or transform is None or torch is None:
        raise RuntimeError("cut_detection | deep feature model is not initialized (no fallback allowed)")
    # embed
    imgA = transform(ImageFromRGB(frameA)).unsqueeze(0).to(device)
    imgB = transform(ImageFromRGB(frameB)).unsqueeze(0).to(device)
    with torch.no_grad():
        eA = embed_model(imgA)
        eB = embed_model(imgB)
        # flatten spatial dimensions
        eA = eA.view(eA.size(0), -1)
        eB = eB.view(eB.size(0), -1)
        # normalize and cosine dist
        eA = eA / (eA.norm(dim=1, keepdim=True)+1e-9)
        eB = eB / (eB.norm(dim=1, keepdim=True)+1e-9)
        sim = (eA * eB).sum().item()
        return float(1.0 - sim)

def morphological_clean_cuts(cut_flags, min_neighbors=1):
    """
    Morphological cleaning: remove isolated cut detections.
    cut_flags: binary array (1 = cut detected, 0 = no cut)
    min_neighbors: minimum number of neighboring cuts within ±N frames
    """
    cleaned = cut_flags.copy()
    n = len(cut_flags)
    window = 3  # Check ±3 frames
    
    for i in range(n):
        if cut_flags[i] == 1:
            # Count neighbors
            start = max(0, i - window)
            end = min(n, i + window + 1)
            neighbors = np.sum(cut_flags[start:end]) - 1  # Exclude self
            if neighbors < min_neighbors:
                cleaned[i] = 0
    
    return cleaned

# -----------------------------
# High-level cut detectors
# -----------------------------

def detect_hard_cuts(
    frame_manager, 
    frame_indices, 
    hist_thresh=None, 
    ssim_thresh=None, 
    flow_thresh=None,
    ssim_max_side: int = 0,
    flow_max_side: int = 0,
    external_flow_mags: Optional[np.ndarray] = None,
    use_deep_features=True, 
    use_adaptive_thresholds=True, 
    temporal_smoothing=True, 
    embed_model=None, 
    transform=None, 
    device='cpu',
    return_model_facing: bool = False,
    # Optional performance mode: cascade-gate expensive computations (SSIM/flow/deep) using cheap histogram signal.
    # Default is OFF to preserve quality; enable explicitly for "fast" profile experiments.
    cascade_enabled: bool = False,
    cascade_keep_top_p: float = 0.25,
    cascade_hist_margin: float = 0.0,
    ):
    """
    Improved hard cut detection with adaptive thresholds, deep features, and temporal smoothing.
    frames: list of BGR frames
    returns list of cut_indices (frame index where cut occurs) and strengths
    Strategy: combine histogram diff, SSIM drop, optical flow jump, and deep embeddings.
    """
    n = len(frame_indices)
    if n < 2:
        return ([], [], {}) if bool(return_model_facing) else ([], [])
    
    ext = None
    if external_flow_mags is not None:
        try:
            ext = np.asarray(external_flow_mags, dtype=np.float32).reshape(-1)
        except Exception:
            ext = None
        if ext is not None and int(ext.size) != int(max(0, n - 1)):
            raise RuntimeError(
                f"detect_hard_cuts | external_flow_mags length mismatch: got {int(ext.size)} expected {int(max(0, n-1))}"
            )

    def _ssim_drop_gray(grayA: np.ndarray, grayB: np.ndarray) -> float:
        try:
            dr = float(grayB.max() - grayB.min())
            dr = dr if dr > 1e-9 else 1.0
            s = ssim(grayA, grayB, data_range=dr)
            return float(1.0 - s)
        except Exception:
            return 0.0

    exp_len = int(max(0, n - 1))

    def _adaptive_thresh(vals: np.ndarray, k: float, *, default_if_empty: float) -> float:
        try:
            v = np.asarray(vals, dtype=np.float32).reshape(-1)
        except Exception:
            v = np.zeros((0,), dtype=np.float32)
        if int(v.size) <= 0:
            return float(default_if_empty)
        return float(np.median(v) + float(k) * float(np.std(v)))

    # Compute frame differences (optionally in a 2-pass cascade mode)
    if bool(cascade_enabled):
        # Pass 1: histogram diffs only (cheap)
        hdiffs_l: list[float] = []
        prev_frame = frame_manager.get(frame_indices[0])
        for i in range(1, n):
            fB = frame_manager.get(frame_indices[i])
            hdiffs_l.append(float(frame_histogram_diff(prev_frame, fB)))
            prev_frame = fB
        hdiffs = np.asarray(hdiffs_l, dtype=np.float32).reshape(-1)

        # Decide hist threshold early (needed for gating)
        if use_adaptive_thresholds:
            hist_thresh_eff = _adaptive_thresh(hdiffs, 2.0, default_if_empty=float("inf")) if hist_thresh is None else float(hist_thresh)
        else:
            hist_thresh_eff = float(hist_thresh or 0.5)
        hist_thresh = hist_thresh_eff

        # Candidate mask: keep all "near-threshold" and ensure at least top-P by histogram.
        m = float(cascade_hist_margin or 0.0)
        kp = float(cascade_keep_top_p)
        kp = 1.0 if kp > 1.0 else (0.0 if kp < 0.0 else kp)
        cand = np.zeros((exp_len,), dtype=bool)
        if exp_len > 0:
            cand |= (hdiffs >= (float(hist_thresh_eff) - m))
            if kp > 0.0 and kp < 1.0:
                try:
                    q = float(np.quantile(hdiffs, 1.0 - kp))
                    cand |= (hdiffs >= q)
                except Exception:
                    pass
            elif kp >= 1.0:
                cand[:] = True

        # Pass 2: compute expensive signals only for candidates
        # Use NaN for "not computed" to make encoder-side handling explicit.
        ssim_diffs = np.full((exp_len,), np.nan, dtype=np.float32)
        flow_mags = np.full((exp_len,), np.nan, dtype=np.float32)
        deep_diffs = np.full((exp_len,), np.nan, dtype=np.float32)
        did_ssim = np.zeros((exp_len,), dtype=bool)
        did_flow = np.zeros((exp_len,), dtype=bool)
        did_deep = np.zeros((exp_len,), dtype=bool)

        prev_frame = frame_manager.get(frame_indices[0])
        prev_gray_full = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
        for i in range(1, n):
            fA = prev_frame
            fB = frame_manager.get(frame_indices[i])
            gray_full = cv2.cvtColor(fB, cv2.COLOR_RGB2GRAY)

            j = int(i - 1)
            if bool(cand[j]):
                # SSIM (candidate only)
                prev_gray_ssim = _resize_gray_max_side(prev_gray_full, int(ssim_max_side))
                gray_ssim = _resize_gray_max_side(gray_full, int(ssim_max_side))
                ssim_diffs[j] = float(_ssim_drop_gray(prev_gray_ssim, gray_ssim))
                did_ssim[j] = True

                # Flow magnitude
                if ext is not None:
                    flow_mags[j] = float(ext[j])
                    did_flow[j] = True
                else:
                    prev_gray_flow = _resize_gray_max_side(prev_gray_full, int(flow_max_side))
                    gray_flow = _resize_gray_max_side(gray_full, int(flow_max_side))
                    flow_mag, _, _ = optical_flow_magnitude(prev_gray_flow, gray_flow)
                    flow_mags[j] = float(flow_mag)
                    did_flow[j] = True

                # Deep features (candidate only)
                if use_deep_features and embed_model is not None:
                    deep_diffs[j] = float(feature_embedding_diff(fA, fB, embed_model, transform, device))
                    did_deep[j] = True

            # Keep state for next pair
            prev_frame = fB
            prev_gray_full = gray_full

        # If flow is external, consider it "computed" everywhere (thresholds should use all values)
        if ext is not None and exp_len > 0:
            flow_mags = np.asarray(ext, dtype=np.float32).reshape(-1)
            did_flow[:] = True

        # Adaptive thresholds based on computed subset only (to avoid zero-padding bias)
        if use_adaptive_thresholds:
            ssim_vals = ssim_diffs[did_ssim]
            flow_vals = flow_mags[did_flow]
            deep_vals = deep_diffs[did_deep] if use_deep_features else np.zeros((0,), dtype=np.float32)
            ssim_thresh = _adaptive_thresh(ssim_vals, 1.5, default_if_empty=float("inf")) if ssim_thresh is None else float(ssim_thresh)
            flow_thresh = _adaptive_thresh(flow_vals, 2.0, default_if_empty=float("inf")) if flow_thresh is None else float(flow_thresh)
            deep_thresh = _adaptive_thresh(deep_vals, 1.5, default_if_empty=float("inf")) if (use_deep_features and deep_thresh is None) else (float(deep_thresh) if use_deep_features else 0.0)
        else:
            ssim_thresh = float(ssim_thresh or 0.25)
            flow_thresh = float(flow_thresh or 4.0)
            deep_thresh = float(0.3) if use_deep_features else 0.0

    else:
        # Original single-pass: compute all signals for all pairs
        hdiffs_l: list[float] = []
        ssim_l: list[float] = []
        flow_l: list[float] = []
        deep_l: list[float] = []

        prev_frame = frame_manager.get(frame_indices[0])
        prev_gray_full = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
        prev_gray_ssim = _resize_gray_max_side(prev_gray_full, int(ssim_max_side))
        prev_gray_flow = _resize_gray_max_side(prev_gray_full, int(flow_max_side))

        for i in range(1, n):
            fA = prev_frame
            fB = frame_manager.get(frame_indices[i])
            gray_full = cv2.cvtColor(fB, cv2.COLOR_RGB2GRAY)
            gray_ssim = _resize_gray_max_side(gray_full, int(ssim_max_side))
            gray_flow = _resize_gray_max_side(gray_full, int(flow_max_side))

            hdiffs_l.append(float(frame_histogram_diff(prev_frame, fB)))
            ssim_l.append(float(_ssim_drop_gray(prev_gray_ssim, gray_ssim)))
            if ext is not None:
                flow_l.append(float(ext[int(i - 1)]))
            else:
                flow_mag, _, _ = optical_flow_magnitude(prev_gray_flow, gray_flow)
                flow_l.append(float(flow_mag))

            if use_deep_features and embed_model is not None:
                deep_l.append(float(feature_embedding_diff(fA, fB, embed_model, transform, device)))
            else:
                deep_l.append(0.0)

            prev_frame = fB
            prev_gray_full = gray_full
            prev_gray_ssim = gray_ssim
            prev_gray_flow = gray_flow

        hdiffs = np.asarray(hdiffs_l, dtype=np.float32).reshape(-1)
        ssim_diffs = np.asarray(ssim_l, dtype=np.float32).reshape(-1)
        flow_mags = np.asarray(flow_l, dtype=np.float32).reshape(-1)
        deep_diffs = np.asarray(deep_l, dtype=np.float32).reshape(-1)
        # In full mode everything is computed (except deep if disabled).
        did_ssim = np.ones_like(ssim_diffs, dtype=bool)
        did_flow = np.ones_like(flow_mags, dtype=bool)
        did_deep = np.ones_like(deep_diffs, dtype=bool) if use_deep_features else np.zeros_like(deep_diffs, dtype=bool)
    
    # Adaptive thresholds based on local statistics (single-pass mode only; cascade computed above)
    if not bool(cascade_enabled):
        if use_adaptive_thresholds:
            hist_thresh = _adaptive_thresh(hdiffs, 2.0, default_if_empty=float("inf")) if hist_thresh is None else float(hist_thresh)
            ssim_thresh = _adaptive_thresh(ssim_diffs, 1.5, default_if_empty=float("inf")) if ssim_thresh is None else float(ssim_thresh)
            flow_thresh = _adaptive_thresh(flow_mags, 2.0, default_if_empty=float("inf")) if flow_thresh is None else float(flow_thresh)
            deep_thresh = _adaptive_thresh(deep_diffs, 1.5, default_if_empty=float("inf")) if use_deep_features else 0.0
        else:
            hist_thresh = float(hist_thresh or 0.5)
            ssim_thresh = float(ssim_thresh or 0.25)
            flow_thresh = float(flow_thresh or 4.0)
            deep_thresh = float(0.3) if use_deep_features else 0.0
    
    # Compute scores (raw, before postprocessing)
    trig_hist = (hdiffs > hist_thresh)
    trig_ssim = (ssim_diffs > ssim_thresh)
    trig_flow = (flow_mags > flow_thresh)
    trig_deep = (deep_diffs > deep_thresh) if use_deep_features else np.zeros_like(trig_hist, dtype=bool)

    scores = np.zeros(len(hdiffs), dtype=np.float32)
    scores += trig_hist.astype(np.float32)
    scores += trig_ssim.astype(np.float32)
    scores += trig_flow.astype(np.float32)
    if use_deep_features:
        scores += trig_deep.astype(np.float32)
    
    # Temporal smoothing to reduce false positives.
    #
    # IMPORTANT: `scores` is a small integer-like signal (0..3 for CPU-no-deep).
    # Gaussian smoothing + strict local-max filtering can suppress isolated true hard-cuts
    # (common pattern: [0,0,3,0,0] -> smoothed peak < 2). For hard-cuts we want to keep
    # isolated spikes and rely on min-distance + (optional) morphological cleaning instead.
    # NOTE: median/gaussian smoothing on a sparse integer score tends to erase isolated true cuts.
    # For hard cuts we keep the thresholded signal and rely on min-distance + morphological cleaning.
    cut_candidates = [(i + 1, float(s)) for i, s in enumerate(scores) if float(s) >= 2.0]
    
    # Morphological cleaning: remove isolated detections
    cut_flag_array = np.zeros(len(frame_indices) - 1, dtype=int)
    for idx, _ in cut_candidates:
        if idx - 1 < len(cut_flag_array):
            cut_flag_array[idx - 1] = 1
    cut_flag_array = morphological_clean_cuts(cut_flag_array, min_neighbors=0)
    
    # Rebuild candidates from cleaned flags
    cleaned_candidates = [(i+1, scores[i]) for i in range(len(cut_flag_array)) if cut_flag_array[i] == 1]
    
    # Remove cuts that are too close (within 5 frames)
    cut_idxs = []
    strengths = []
    for idx, strength in cleaned_candidates:
        if not cut_idxs or idx - cut_idxs[-1] > 5:
            cut_idxs.append(idx)
            strengths.append(float(strength))
    
    if bool(return_model_facing):
        flow_source = "core_optical_flow" if ext is not None else "internal_farneback"
        flow_mag_units = "core_optical_flow_norm_per_sec_mean" if ext is not None else "farneback_mean_mag_px"
        debug = {
            "hist_diff_l1": np.asarray(hdiffs, dtype=np.float32),
            "ssim_drop": np.asarray(ssim_diffs, dtype=np.float32),
            "flow_mag": np.asarray(flow_mags, dtype=np.float32),
            "hard_score": np.asarray(scores, dtype=np.float32),
            "deep_cosine_dist": (np.asarray(deep_diffs, dtype=np.float32) if use_deep_features else None),
            "valid_mask": {
                "ssim": np.asarray(did_ssim, dtype=bool) if "did_ssim" in locals() else np.ones_like(scores, dtype=bool),
                "flow": np.asarray(did_flow, dtype=bool) if "did_flow" in locals() else np.ones_like(scores, dtype=bool),
                "deep": np.asarray(did_deep, dtype=bool) if "did_deep" in locals() else np.zeros_like(scores, dtype=bool),
            },
            "thresholds": {
                "hist": float(hist_thresh),
                "ssim": float(ssim_thresh),
                "flow": float(flow_thresh),
                "deep": float(deep_thresh) if use_deep_features else 0.0,
            },
            "triggers": {
                "hist": np.asarray(trig_hist, dtype=bool),
                "ssim": np.asarray(trig_ssim, dtype=bool),
                "flow": np.asarray(trig_flow, dtype=bool),
                "deep": np.asarray(trig_deep, dtype=bool),
            },
            "cascade": {
                "enabled": bool(cascade_enabled),
                "keep_top_p": float(cascade_keep_top_p),
                "hist_margin": float(cascade_hist_margin),
            },
            "flow_source": str(flow_source),
            "flow_mag_units": str(flow_mag_units),
        }
        return cut_idxs, strengths, debug

    return cut_idxs, strengths

def detect_soft_cuts(
    frame_manager,
    frame_indices,
    fps,
    fade_threshold=0.02,
    min_duration_frames=4,
    use_flow_consistency=True,
    flow_max_side: int = 0,
    external_flow_mags: Optional[np.ndarray] = None,
    return_model_facing: bool = False,
):
    """
    Improved soft cut detection with gradient-based analysis and optical flow consistency.
    Detect fade-in/out and dissolves by monitoring brightness/histogram changes over a window.
    Returns events: list of dicts {'type':'fade_in'/'fade_out'/'dissolve', 'start', 'end', 'duration_s'}
    """
    n = len(frame_indices)
    if n < 3:
        return ([], {}) if bool(return_model_facing) else []
    
    # Multi-channel gradient analysis (HSV + Lab)
    hsv_values: List[float] = []
    lab_values: List[float] = []
    hist_diffs_pair: List[float] = []   # (N-1,)
    flow_mags_pair: List[float] = []    # (N-1,)
    flow_valid_mask: List[bool] = []    # (N-1,)
    
    prev_gray_full = cv2.cvtColor(frame_manager.get(frame_indices[0]), cv2.COLOR_RGB2GRAY)
    prev_gray = _resize_gray_max_side(prev_gray_full, int(flow_max_side))
    prev_hsv_hist = None

    ext = None
    if external_flow_mags is not None:
        try:
            ext = np.asarray(external_flow_mags, dtype=np.float32).reshape(-1)
        except Exception:
            ext = None
        if ext is not None and int(ext.size) != int(max(0, n - 1)):
            raise RuntimeError(
                f"detect_soft_cuts | external_flow_mags length mismatch: got {int(ext.size)} expected {int(max(0, n-1))}"
            )
    
    for i, idx in enumerate(frame_indices):
        f = frame_manager.get(idx)

        # HSV analysis
        hsv = cv2.cvtColor(f, cv2.COLOR_RGB2HSV)
        v = hsv[:,:,2].mean() / 255.0
        hsv_values.append(v)
        
        # Lab color space for better perceptual uniformity
        lab = cv2.cvtColor(f, cv2.COLOR_RGB2LAB)
        l = lab[:,:,0].mean() / 255.0
        lab_values.append(l)
        
        # Histogram gradient (all channels)
        hsv_hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], [0, 180, 0, 256, 0, 256]).flatten()
        hsv_hist = hsv_hist / (hsv_hist.sum() + 1e-9)
        if prev_hsv_hist is not None:
            hist_diff = float(np.linalg.norm(hsv_hist - prev_hsv_hist, ord=1))
            hist_diffs_pair.append(hist_diff)
        prev_hsv_hist = hsv_hist
        prev_hsv_hist = hsv_hist
        
        # Optical flow magnitude (per-pair) for consistency check
        if i > 0:
            j = int(i - 1)
            if ext is not None:
                flow_mags_pair.append(float(ext[j]))
                flow_valid_mask.append(True)
            else:
                gray_full = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
                gray = _resize_gray_max_side(gray_full, int(flow_max_side))
                flow_mag, _, _ = optical_flow_magnitude(prev_gray, gray)
                flow_mags_pair.append(float(flow_mag))
                flow_valid_mask.append(True)
                prev_gray = gray
    
    hsv_values_np = np.asarray(hsv_values, dtype=np.float32)
    lab_values_np = np.asarray(lab_values, dtype=np.float32)
    hist_diffs_np = np.asarray(hist_diffs_pair, dtype=np.float32)  # (N-1,)
    flow_mags_np = np.asarray(flow_mags_pair, dtype=np.float32)    # (N-1,)
    
    events = []
    
    # Fade detection: gradient-based with cumulative distribution
    hsv_deriv = np.diff(hsv_values_np)
    lab_deriv = np.diff(lab_values_np)
    hist_deriv = np.diff(hist_diffs_np) if hist_diffs_np.size > 1 else np.zeros((0,), dtype=np.float32)
    
    # Smooth derivatives
    hsv_smooth = medfilt(hsv_deriv, kernel_size=5)
    lab_smooth = medfilt(lab_deriv, kernel_size=5)
    
    # Find fade regions
    i = 0
    while i < len(hsv_smooth):
        if abs(hsv_smooth[i]) < 0.01 and abs(lab_smooth[i]) < 0.01:
            j = i
            cum_hsv = 0.0
            cum_lab = 0.0
            while j < len(hsv_smooth) and abs(hsv_smooth[j]) < 0.02 and abs(lab_smooth[j]) < 0.02:
                cum_hsv += hsv_smooth[j]
                cum_lab += lab_smooth[j]
                j += 1
            duration = j - i
            if duration >= min_duration_frames:
                hsv_change = abs(hsv_values_np[j] - hsv_values_np[i]) if j < len(hsv_values_np) else 0
                lab_change = abs(lab_values_np[j] - lab_values_np[i]) if j < len(lab_values_np) else 0
                if hsv_change > fade_threshold or lab_change > fade_threshold:
                    typ = 'fade_in' if (hsv_values_np[j] > hsv_values_np[i] if j < len(hsv_values_np) else False) else 'fade_out'
                    # NOTE: duration_s will be re-computed by caller using union_timestamps_sec (source-of-truth).
                    start = i
                    end = min(j, len(frame_indices)-1)
                    events.append({'type': typ, 'start': start, 'end': end, 'duration_s': float("nan")})
            i = j
        else:
            i += 1
    
    # Dissolve detection: moderate histogram drift with low flow consistency
    # Improved: check for linear mixing correlation and exclude exposure changes
    if use_flow_consistency:
        window_size = min(10, n // 5)
        for i in range(window_size, n - window_size):
            # Check for gradual histogram change
            # hist_diffs_np/flow_mags_np are per-pair => align with center pair index (i-1)
            p = max(0, min(int(i - 1), int(n - 2)))
            ws = int(window_size // 2)
            hist_window = hist_diffs_np[max(0, p - ws): min(int(n - 1), p + ws)]
            flow_window = flow_mags_np[max(0, p - ws): min(int(n - 1), p + ws)]
            
            # Gradual histogram change (low variance in changes)
            hist_var = np.var(hist_window)
            hist_mean = np.mean(hist_window)
            flow_mean = np.mean(flow_window)
            
            # Check for exposure changes (global brightness shift across entire frame)
            # Exposure changes affect entire frame uniformly, dissolves affect content distribution
            hsv_window = hsv_values_np[i-window_size//2:i+window_size//2]
            lab_window = lab_values_np[i-window_size//2:i+window_size//2]
            hsv_gradient = np.abs(np.diff(hsv_window))
            lab_gradient = np.abs(np.diff(lab_window))
            # Low gradient variance = uniform exposure change (not dissolve)
            is_exposure_change = (np.var(hsv_gradient) < 0.0001 and np.var(lab_gradient) < 0.0001 and 
                                 np.mean(np.abs(hsv_gradient)) > 0.01)  # Uniform but significant change
            
            # Dissolve: gradual histogram change + low motion + not exposure change
            # Also check for linear correlation in histogram changes (dissolve = linear mixing)
            hist_window_full = hist_diffs_np[max(0, p - window_size): min(int(n - 1), p + window_size)]
            if len(hist_window_full) > 3:
                # Compute correlation of histogram changes (should be smooth/linear for dissolve)
                hist_correlation = np.corrcoef(hist_window_full[:-1], hist_window_full[1:])[0, 1] if len(hist_window_full) > 1 else 0
                is_smooth_mixing = hist_correlation > 0.5  # Positive correlation = smooth transition
            else:
                is_smooth_mixing = True
            
            if (hist_var < 0.001 and hist_mean > 0.01 and flow_mean < 2.0 and 
                not is_exposure_change and is_smooth_mixing):
                # Check if not already detected as fade
                is_fade = any(e['start'] <= i <= e['end'] for e in events)
                if not is_fade:
                    start = i - window_size//2
                    end = i + window_size//2
                    events.append({'type': 'dissolve', 'start': start, 'end': end, 'duration_s': float("nan")})
    
    if bool(return_model_facing):
        dbg = {
            "soft_hsv_v": hsv_values_np.astype(np.float32, copy=False),
            "soft_lab_l": lab_values_np.astype(np.float32, copy=False),
            "soft_hist_diff_l1": hist_diffs_np.astype(np.float32, copy=False),   # (N-1,)
            "soft_flow_mag": flow_mags_np.astype(np.float32, copy=False),       # (N-1,)
            "soft_flow_valid_mask": np.asarray(flow_valid_mask, dtype=bool).reshape(-1),
            "soft_flow_source": ("core_optical_flow" if ext is not None else "internal_farneback"),
        }
        return events, dbg

    return events


def detect_motion_based_cuts(
    frame_manager,
    frame_indices, 
    flow_spike_factor=None, 
    use_direction_analysis=True, 
    adaptive_threshold=True, 
    detect_speed_ramps=True,
    use_camera_motion_compensation=True,
    flow_max_side: int = 0,
    external_flow_mags: Optional[np.ndarray] = None,
    motion_cascade_enabled: bool = True,
    return_model_facing: bool = False,
):
    """
    Improved motion-based cut detection with direction analysis and adaptive thresholds.
    Detect whip pans / zoom transitions / speed ramp cuts by measuring spikes in optical-flow magnitude variance.
    Returns list of indices, intensities, and types ('whip_pan', 'zoom', or 'speed_ramp').
    """
    n = len(frame_indices)
    if n < 2:
        return ([], [], [], {}) if bool(return_model_facing) else ([], [], [])
    
    ext = None
    if external_flow_mags is not None:
        try:
            ext = np.asarray(external_flow_mags, dtype=np.float32).reshape(-1)
        except Exception:
            ext = None
        if ext is not None and int(ext.size) != int(max(0, n - 1)):
            raise RuntimeError(
                f"detect_motion_based_cuts | external_flow_mags length mismatch: got {int(ext.size)} expected {int(max(0, n-1))}"
            )

    exp_len = int(max(0, n - 1))
    mags_array = np.zeros((exp_len,), dtype=np.float32)
    if ext is not None:
        mags_array = np.asarray(ext, dtype=np.float32).reshape(-1)

    # Debug curves: fill NaN where not computed.
    motion_flow_mag = mags_array.astype(np.float32, copy=False) if exp_len > 0 else np.zeros((0,), dtype=np.float32)
    motion_dir_consistency = np.full((exp_len,), np.nan, dtype=np.float32)
    motion_mag_variance = np.full((exp_len,), np.nan, dtype=np.float32)
    motion_camera_motion_flag = np.zeros((exp_len,), dtype=bool)
    motion_dir_valid_mask = np.zeros((exp_len,), dtype=bool)
    motion_var_valid_mask = np.zeros((exp_len,), dtype=bool)
    motion_cam_valid_mask = np.zeros((exp_len,), dtype=bool)

    # For non-external mode or if we disable cascade, fall back to original full compute.
    if ext is None or not bool(motion_cascade_enabled):
        mags = []
        direction_consistencies = []
        mag_variances = []  # For speed ramp detection
        camera_motion_flags = []

        prev_gray = cv2.cvtColor(frame_manager.get(frame_indices[0]), cv2.COLOR_RGB2GRAY)
        # Resize frames for faster flow computation if large (or if explicitly requested).
        sample_frame = frame_manager.get(frame_indices[0])
        h, w = sample_frame.shape[:2]
        fm_side = int(flow_max_side) if flow_max_side is not None else 0
        use_low_res = bool(fm_side > 0) or (h * w > 640 * 480)
        target_size = (fm_side, fm_side) if fm_side > 0 else ((256, 256) if use_low_res else None)

        for i in range(1, n):
            gray = cv2.cvtColor(frame_manager.get(frame_indices[i]), cv2.COLOR_RGB2GRAY)

            is_camera_motion = False
            if use_camera_motion_compensation:
                H, inlier_ratio = estimate_global_motion_homography(prev_gray, gray)
                is_camera_motion = inlier_ratio > 0.7 if H is not None else False

            if use_low_res and target_size is not None:
                prev_gray_small = cv2.resize(prev_gray, target_size)
                gray_small = cv2.resize(gray, target_size)
                mag, mag_map, angles = optical_flow_magnitude(prev_gray_small, gray_small)
                mag = mag * (h * w) / float(max(1, int(target_size[0]) * int(target_size[1])))
            else:
                mag, mag_map, angles = optical_flow_magnitude(prev_gray, gray)

            mags.append(float(mag))
            camera_motion_flags.append(bool(is_camera_motion))

            if angles is not None and use_direction_analysis:
                direction_consistencies.append(float(optical_flow_direction_consistency(angles)))
            else:
                direction_consistencies.append(0.0)

            if detect_speed_ramps and mag_map is not None:
                mag_variances.append(float(np.var(mag_map)))
            else:
                mag_variances.append(0.0)

            prev_gray = gray

        mags_array = np.asarray(mags, dtype=np.float32).reshape(-1)
        camera_motion_flags_np = np.asarray(camera_motion_flags, dtype=bool).reshape(-1)
        direction_consistencies_np = np.asarray(direction_consistencies, dtype=np.float32).reshape(-1)
        mag_variances_np = np.asarray(mag_variances, dtype=np.float32).reshape(-1)

        motion_flow_mag = mags_array
        motion_camera_motion_flag = camera_motion_flags_np
        motion_cam_valid_mask[:] = True
        if use_direction_analysis:
            motion_dir_consistency = direction_consistencies_np.astype(np.float32, copy=False)
            motion_dir_valid_mask[:] = True
        motion_mag_variance = mag_variances_np.astype(np.float32, copy=False)
        motion_var_valid_mask[:] = bool(detect_speed_ramps)

    # Thresholds based on mags_array (either external or computed)
    if adaptive_threshold:
        if flow_spike_factor is None:
            threshold = float(np.percentile(mags_array, 95)) if mags_array.size else float("inf")
        else:
            median = float(np.median(mags_array)) if mags_array.size else 0.0
            std = float(np.std(mags_array)) + 1e-9
            threshold = median + float(flow_spike_factor) * std
    else:
        median = float(np.median(mags_array)) if mags_array.size else 0.0
        std = float(np.std(mags_array)) + 1e-9
        threshold = median + float(flow_spike_factor or 3.0) * std

    spike_mask = mags_array > float(threshold)

    # External+cascade path: compute camera motion + direction + variance ONLY for candidate spikes.
    if ext is not None and bool(motion_cascade_enabled) and spike_mask.size:
        prev_gray = cv2.cvtColor(frame_manager.get(frame_indices[0]), cv2.COLOR_RGB2GRAY)
        sample_frame = frame_manager.get(frame_indices[0])
        h, w = sample_frame.shape[:2]
        fm_side = int(flow_max_side) if flow_max_side is not None else 0
        use_low_res = bool(fm_side > 0) or (h * w > 640 * 480)
        target_size = (fm_side, fm_side) if fm_side > 0 else ((256, 256) if use_low_res else None)

        spike_js = np.where(spike_mask)[0]  # 0-based pair indices
        for j in spike_js.tolist():
            i = int(j + 1)  # frame position index in [1..n-1]
            if i <= 0 or i >= n:
                continue
            gray_prev = cv2.cvtColor(frame_manager.get(frame_indices[i - 1]), cv2.COLOR_RGB2GRAY)
            gray = cv2.cvtColor(frame_manager.get(frame_indices[i]), cv2.COLOR_RGB2GRAY)

            is_camera_motion = False
            if use_camera_motion_compensation:
                H, inlier_ratio = estimate_global_motion_homography(gray_prev, gray)
                is_camera_motion = inlier_ratio > 0.7 if H is not None else False
                motion_camera_motion_flag[j] = bool(is_camera_motion)
                motion_cam_valid_mask[j] = True

            if use_camera_motion_compensation and bool(is_camera_motion):
                continue  # filtered out later (same logic as original)

            # Compute angles + mag_map for classification
            if use_low_res and target_size is not None:
                prev_small = cv2.resize(gray_prev, target_size)
                curr_small = cv2.resize(gray, target_size)
                _, mag_map, angles = optical_flow_magnitude(prev_small, curr_small)
            else:
                _, mag_map, angles = optical_flow_magnitude(gray_prev, gray)

            if angles is not None and use_direction_analysis:
                motion_dir_consistency[j] = float(optical_flow_direction_consistency(angles))
                motion_dir_valid_mask[j] = True

            if detect_speed_ramps and mag_map is not None:
                motion_mag_variance[j] = float(np.var(mag_map))
                motion_var_valid_mask[j] = True

        # Apply camera-motion filter
        if use_camera_motion_compensation:
            spike_mask = spike_mask & ~motion_camera_motion_flag

    spike_idxs = np.where(spike_mask)[0] + 1
    intensities = mags_array[spike_idxs - 1].tolist() if spike_idxs.size else []

    # Speed ramp threshold computed on available variances (spike-only in cascade mode)
    vv = motion_mag_variance[np.isfinite(motion_mag_variance)]
    speed_ramp_threshold = float(np.percentile(vv, 90)) if (detect_speed_ramps and vv.size > 0) else 0.0

    types: List[str] = []
    if use_direction_analysis:
        for idx in spike_idxs.tolist():
            j = int(idx - 1)
            consistency = float(motion_dir_consistency[j]) if (j < exp_len and np.isfinite(motion_dir_consistency[j])) else 0.0
            mag_var = float(motion_mag_variance[j]) if (j < exp_len and np.isfinite(motion_mag_variance[j])) else 0.0
            is_cam_motion = bool(motion_camera_motion_flag[j]) if j < exp_len else False
            if detect_speed_ramps and mag_var > speed_ramp_threshold:
                types.append("speed_ramp")
            elif consistency > 0.6 or is_cam_motion:
                types.append("whip_pan")
            else:
                types.append("zoom")
    else:
        types = ["motion_cut"] * int(len(spike_idxs))

    if bool(return_model_facing):
        dbg = {
            "motion_flow_mag": motion_flow_mag.astype(np.float32, copy=False),
            "motion_dir_consistency": motion_dir_consistency.astype(np.float32, copy=False),
            "motion_mag_variance": motion_mag_variance.astype(np.float32, copy=False),
            "motion_camera_motion_flag": motion_camera_motion_flag.astype(bool, copy=False),
            "motion_dir_valid_mask": motion_dir_valid_mask.astype(bool, copy=False),
            "motion_var_valid_mask": motion_var_valid_mask.astype(bool, copy=False),
            "motion_cam_valid_mask": motion_cam_valid_mask.astype(bool, copy=False),
            "motion_flow_source": ("core_optical_flow" if ext is not None else "internal_farneback"),
            "motion_threshold": float(threshold),
        }
        return spike_idxs.tolist(), intensities, types, dbg

    return spike_idxs.tolist(), intensities, types

# Stylized transitions classifier (zero-shot with CLIP via Triton).
class StylizedTransitionZeroShot:
    """
    Baseline policy:
    - NO local CLIP weights in this module (no-network).
    - text embeddings are produced by `core_clip` and loaded from its NPZ artifact.
    - image embeddings are produced via Triton CLIP image encoder (resolved via ModelManager).
    """

    _CLIP_MEAN = np.asarray([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
    _CLIP_STD = np.asarray([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)

    def __init__(
        self,
        *,
        client,
        triton_model_name: str,
        triton_model_version: Optional[str],
        triton_input_name: str,
        triton_output_name: str,
        triton_input_datatype: str,
        prompts: List[str],
        text_embeddings: np.ndarray,
        use_temporal_aggregation: bool = True,
        use_multimodal: bool = True,
        image_size: int = 224,
    ):
        self.client = client
        self.triton_model_name = str(triton_model_name)
        self.triton_model_version = str(triton_model_version) if triton_model_version else None
        self.triton_input_name = str(triton_input_name)
        self.triton_output_name = str(triton_output_name)
        self.triton_input_datatype = str(triton_input_datatype or "FP32")
        self.use_temporal_aggregation = bool(use_temporal_aggregation)
        self.use_multimodal = bool(use_multimodal)
        self.image_size = int(image_size)

        self.labels = [str(x) for x in (prompts or [])]
        if not self.labels:
            raise RuntimeError("cut_detection | core_clip prompts are missing/empty (no-fallback)")

        te = np.asarray(text_embeddings, dtype=np.float32)
        if te.ndim != 2 or te.shape[0] != len(self.labels):
            raise RuntimeError(
                f"cut_detection | invalid core_clip text_embeddings shape: {te.shape} (expected [P,D], P={len(self.labels)})"
            )
        # Ensure normalized embeddings (core_clip already normalizes, but we enforce anyway).
        norms = np.linalg.norm(te, axis=-1, keepdims=True) + 1e-9
        self.text_embeddings = te / norms

        # Feature cache for efficiency (cache_key -> np.ndarray[D])
        self.feature_cache: Dict[Any, np.ndarray] = {}
    
    def get_edit_style_labels(self):
        """Return labels for edit style classification from FEATURES.MD"""
        return [
            "fast-cut montage",
            "slow-paced editorial",
            "social media style",
            "documentary style",
            "cinematic editing",
            "meme-style editing",
            "high-action-edit"
        ]

    def _create_multimodal_input(self, frames_window):
        """Create multi-modal input: concatenate frame differences and optical flow visualization."""
        if len(frames_window) < 2:
            return frames_window[len(frames_window)//2] if frames_window else None
        
        # Create difference frame (frames are RGB)
        mid_idx = len(frames_window) // 2
        if mid_idx > 0:
            diff_frame = cv2.absdiff(frames_window[mid_idx-1], frames_window[mid_idx])
        else:
            diff_frame = frames_window[mid_idx]
        
        # Create optical flow visualization
        if len(frames_window) >= 2:
            prev_gray = cv2.cvtColor(frames_window[mid_idx-1], cv2.COLOR_RGB2GRAY)
            curr_gray = cv2.cvtColor(frames_window[mid_idx], cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            # Visualize flow as HSV
            hsv = np.zeros((flow.shape[0], flow.shape[1], 3), dtype=np.uint8)
            hsv[...,0] = ang * 180 / np.pi / 2
            hsv[...,1] = 255
            hsv[...,2] = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
            flow_rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        else:
            flow_rgb = diff_frame
        
        # Concatenate: original | diff | flow (side by side, resized to fit)
        h, w = frames_window[mid_idx].shape[:2]
        target_h, target_w = h, w * 3
        combined = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        
        combined[:, :w] = cv2.resize(frames_window[mid_idx], (w, h))
        combined[:, w:2*w] = cv2.resize(diff_frame, (w, h))
        combined[:, 2*w:] = cv2.resize(flow_rgb, (w, h))
        
        return combined

    def _preprocess_one(self, img_rgb_uint8: np.ndarray) -> np.ndarray:
        """
        Preprocess RGB uint8 image to CLIP float32 tensor: (1,3,S,S)
        """
        try:
            from PIL import Image  # type: ignore
        except Exception as e:
            raise RuntimeError(f"cut_detection | PIL is required for CLIP preprocess: {e}") from e

        s = int(self.image_size)
        pil = Image.fromarray(img_rgb_uint8)
        pil = pil.resize((s, s), resample=Image.BICUBIC)
        arr = np.asarray(pil, dtype=np.float32) / 255.0
        arr = (arr - self._CLIP_MEAN) / (self._CLIP_STD + 1e-12)
        arr = np.transpose(arr, (2, 0, 1)).astype(np.float32)
        return arr[None, ...]

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        x = x - float(np.max(x))
        e = np.exp(x)
        return e / (float(np.sum(e)) + 1e-9)

    def predict_transition(self, frames_window, cache_key=None):
        """
        Improved transition prediction with temporal aggregation and multi-modal input.
        frames_window: small sequence of consecutive frames (RGB)
        Return probs per label (zero-shot)
        """
        # Use cache if available
        if cache_key is not None and cache_key in self.feature_cache:
            img_feat = self.feature_cache[cache_key]
        else:
            # Create input: multi-modal or single frame
            if self.use_multimodal and len(frames_window) >= 2:
                input_img = self._create_multimodal_input(frames_window)
                if input_img is None:
                    input_img = frames_window[len(frames_window)//2]
            else:
                input_img = frames_window[len(frames_window)//2]

            if not isinstance(input_img, np.ndarray):
                raise RuntimeError("cut_detection | expected RGB ndarray input for CLIP preprocess (no-fallback)")
            inp = self._preprocess_one(input_img)  # (1,3,S,S) float32

            res = self.client.infer(
                model_name=self.triton_model_name,
                model_version=self.triton_model_version,
                input_name=self.triton_input_name,
                input_tensor=inp,
                output_name=self.triton_output_name,
                datatype=self.triton_input_datatype,
            )
            out = np.asarray(res.output, dtype=np.float32).reshape(1, -1)
            out = out / (np.linalg.norm(out, axis=-1, keepdims=True) + 1e-9)
            img_feat = out.reshape(-1)
            if cache_key is not None:
                self.feature_cache[cache_key] = img_feat

        # Compute similarity with cached core_clip text embeddings
        logits = np.matmul(img_feat.reshape(1, -1), self.text_embeddings.T).reshape(-1)
        probs = self._softmax(logits)
        return {self.labels[i]: float(probs[i]) for i in range(len(self.labels))}
    
    def predict_transition_temporal(self, frames_window, window_size=5):
        """Temporal aggregation: average probabilities over a window."""
        
        probs_list = []
        for i in range(max(0, len(frames_window)//2 - window_size//2), 
                      min(len(frames_window), len(frames_window)//2 + window_size//2 + 1)):
            sub_window = frames_window[max(0, i-window_size//2):min(len(frames_window), i+window_size//2+1)]
            if sub_window:
                probs = self.predict_transition(sub_window)
                probs_list.append(probs)
        
        if not probs_list:
            return {lbl: 0.0 for lbl in self.labels}
        
        # Average probabilities
        avg_probs = {lbl: 0.0 for lbl in self.labels}
        for probs in probs_list:
            for lbl, val in probs.items():
                avg_probs[lbl] += val
        for lbl in self.labels:
            avg_probs[lbl] /= len(probs_list)
        
        return avg_probs

# -----------------------------
# Jump-cut detection (pose/face based)
# -----------------------------
def detect_jump_cuts(
    frame_manager,
    frame_indices,
    use_background_embedding: bool = True,
    embed_model=None,
    transform=None,
    device: str = "cuda",
):
    """
    Jump cut detection c опорой только на видеосигнал и deep‑эмбеддинги.

    ВАЖНО:
    - В исходной версии использовались Mediapipe face/pose модели (face_detector, pose_detector).
      Теперь они полностью убраны из модуля — никакой прямой зависимости
      от внутренних моделей нет.
    - Логика основывается на сравнении background‑эмбеддингов и (опционально) face‑эмбеддингов,
      полученных из переданной embed_model (ResNet и т.п.), которая инициализируется снаружи
      (через BaseModule / core‑провайдеры), без прямого запуска моделей внутри cut_detection.

    Returns:
        jump_idxs: List[int] — индексы кадров с jump‑cut
        jump_scores: List[float] — интенсивность jump‑cut (0..1+)
    """

    prev_landmarks = None
    prev_pose_landmarks = None
    prev_frame = None
    prev_background_embedding = None
    prev_face_embedding = None  # For face ID similarity
    jump_idxs = []
    jump_scores = []
    
    for i, idx in enumerate(frame_indices):

        f = frame_manager.get(idx)

        img_rgb = f

        # В новой версии face_bbox и face_landmarks могут быть заданы только извне
        # через готовые данные (core_face_landmarks). Здесь мы не вызываем Mediapipe.
        face_landmarks = None
        face_bbox = None

        # Face ID embedding (для устойчивой проверки похожести лица между кадрами)
        face_embedding = None
        if embed_model is not None and transform is not None and face_bbox is not None:
            try:
                # Extract face region (expand bbox slightly)
                h, w = f.shape[:2]
                x1, y1, x2, y2 = face_bbox
                x1, y1 = max(0, int((x1 - 0.1) * w)), max(0, int((y1 - 0.1) * h))
                x2, y2 = min(w, int((x2 + 0.1) * w)), min(h, int((y2 + 0.1) * h))
                face_roi = f[y1:y2, x1:x2]
                if face_roi.size > 0:
                    img_tensor = transform(ImageFromCV(face_roi)).unsqueeze(0).to(device)
                    with torch.no_grad():
                        face_emb = embed_model(img_tensor)
                        face_emb = face_emb.view(face_emb.size(0), -1)
                        face_emb = face_emb / (face_emb.norm(dim=1, keepdim=True)+1e-9)
                        face_embedding = face_emb.cpu().numpy()[0]
            except Exception:
                pass
        
        # Background embedding (используем deep‑фичи, при наличии face_bbox маскируем лицо)
        background_embedding = None
        if use_background_embedding and embed_model is not None and transform is not None:
            try:
                # If face detected, mask it out for background comparison
                bg_frame = f.copy()
                if face_bbox is not None:
                    h, w = bg_frame.shape[:2]
                    x1, y1, x2, y2 = face_bbox
                    # Expand mask to exclude face region more completely
                    mask_expand = 0.15
                    x1_mask = max(0, int((x1 - mask_expand) * w))
                    y1_mask = max(0, int((y1 - mask_expand) * h))
                    x2_mask = min(w, int((x2 + mask_expand) * w))
                    y2_mask = min(h, int((y2 + mask_expand) * h))
                    # Blur face region to reduce its contribution
                    bg_frame[y1_mask:y2_mask, x1_mask:x2_mask] = cv2.GaussianBlur(
                        bg_frame[y1_mask:y2_mask, x1_mask:x2_mask], (15, 15), 5)
                
                img_tensor = transform(ImageFromCV(bg_frame)).unsqueeze(0).to(device)
                with torch.no_grad():
                    bg_emb = embed_model(img_tensor)
                    bg_emb = bg_emb.view(bg_emb.size(0), -1)
                    bg_emb = bg_emb / (bg_emb.norm(dim=1, keepdim=True)+1e-9)
                    background_embedding = bg_emb.cpu().numpy()[0]
            except Exception:
                pass
        
        # Check for jump cut
        if prev_landmarks is not None or prev_pose_landmarks is not None:
            score = 0.0
            max_score = 0.0
            confidence = 1.0
            
            # Face similarity check (improved with face ID embedding)
            if face_landmarks is not None and prev_landmarks is not None:
                # Use face embedding if available (better ID matching)
                if face_embedding is not None and prev_face_embedding is not None:
                    face_sim = np.dot(face_embedding, prev_face_embedding)
                    face_change = 1.0 - face_sim
                else:
                    # Fallback to landmark-based similarity
                    a = face_landmarks - face_landmarks.mean()
                    b = prev_landmarks - prev_landmarks.mean()
                    denom = (np.linalg.norm(a)+1e-9)*(np.linalg.norm(b)+1e-9)
                    cos_face = np.dot(a, b) / denom
                    face_change = 1.0 - cos_face
                score += face_change
                max_score += 1.0
                # Lower confidence if face similarity is very high (likely same person)
                if face_change < 0.2:
                    confidence *= 0.7
            
            # Ранее здесь учитывалась поза (pose_landmarks через Mediapipe).
            # Примечание: зависимости от внутренних pose‑моделей удалены.
            # Для jump‑cut хватает изменений лица и/или фона.
            
            # Background similarity check (improved with foreground masking)
            background_similar = True
            bg_sim_value = 0.0
            if use_background_embedding and background_embedding is not None and prev_background_embedding is not None:
                # Cosine similarity between background embeddings (with masked foreground)
                bg_sim = np.dot(background_embedding, prev_background_embedding)
                bg_sim_value = bg_sim
                background_similar = bg_sim > 0.85  # High similarity = same background
            else:
                # Fallback to SSIM
                s = frame_ssim(prev_frame, f)
                background_similar = s < 0.2  # Low SSIM drop = similar background
                bg_sim_value = 1.0 - s
            
            # Jump cut: large pose/face change + similar background
            # Use confidence-weighted threshold
            threshold = 0.3 / confidence  # Lower threshold if confidence is higher
            if max_score > 0:
                normalized_score = score / max_score
                if normalized_score > threshold and background_similar:  # Significant pose change + similar background
                    jump_idxs.append(i)
                    jump_scores.append(float(normalized_score * bg_sim_value))  # Weight by background similarity
        
        # Update previous state
        prev_landmarks = face_landmarks
        prev_frame = f
        if background_embedding is not None:
            prev_background_embedding = background_embedding
        if face_embedding is not None:
            prev_face_embedding = face_embedding
    
    return jump_idxs, jump_scores

# -----------------------------
# Scene Boundary Detection
# -----------------------------
def scene_boundaries_from_shots(shot_cut_indices, shots_duration_frames, fps,
                                min_scene_shots=2, use_semantic_clustering=True,
                                frame_embeddings=None, audio_events=None,
                                embed_model=None, transform=None, device='cpu'):
    """
    Improved scene boundary detection with semantic clustering and audio+visual fusion.
    Group consecutive shots into scenes using embeddings, audio events, and adaptive thresholds.
    shot_cut_indices: list of frame indices where cuts happen
    shots_duration_frames: list of durations in frames for each shot
    Returns scene boundaries as list of shot index ranges [(s0,e0), ...]
    """
    shot_count = len(shots_duration_frames)
    if shot_count == 0:
        return []
    
    durations_seconds = [seconds_from_fps(d, fps) for d in shots_duration_frames]
    
    # Semantic clustering approach
    if use_semantic_clustering and frame_embeddings is not None:
        shot_embeddings = []

        for i, (start_idx, duration) in enumerate(
            zip([0] + shot_cut_indices, shots_duration_frames)
        ):
            mid_frame_idx = start_idx + duration // 2

            if mid_frame_idx < len(frame_embeddings):
                shot_embeddings.append(frame_embeddings[mid_frame_idx])
            else:
                if frame_embeddings is not None and frame_embeddings.shape[0] > 0:
                    shot_embeddings.append(frame_embeddings[-1])
                else:
                    shot_embeddings.append(np.zeros(512, dtype=np.float32))

        shot_embeddings = np.asarray(shot_embeddings, dtype=np.float32)
        
        # Normalize embeddings
        if len(shot_embeddings) > 0:
            scaler = StandardScaler()
            shot_embeddings_scaled = scaler.fit_transform(shot_embeddings)
            
            # DBSCAN clustering for scene boundaries
            if len(shot_embeddings_scaled) > 1:
                clustering = DBSCAN(eps=0.5, min_samples=min_scene_shots).fit(shot_embeddings_scaled)
                labels = clustering.labels_
                
                # Group shots by cluster
                scenes = []
                current_scene = []
                current_label = labels[0]
                
                for i, label in enumerate(labels):
                    if label == current_label and label != -1:  # -1 is noise in DBSCAN
                        current_scene.append(i)
                    else:
                        if current_scene:
                            scenes.append((min(current_scene), max(current_scene)))
                        if label != -1:
                            current_scene = [i]
                            current_label = label
                        else:
                            current_scene = []
                            current_label = labels[i+1] if i+1 < len(labels) else -1
                
                if current_scene:
                    scenes.append((min(current_scene), max(current_scene)))
                
                if scenes:
                    return scenes
    
    # Audio + visual fusion approach
    if audio_events is not None:
        # Use audio events (onsets, silences) to determine scene boundaries
        scenes = []
        shot_idx = 0
        current_scene_start = 0
        last_audio_event_time = 0.0
        
        for i, duration in enumerate(durations_seconds):
            shot_start_time = sum(durations_seconds[:i])
            shot_end_time = shot_start_time + duration
            
            # Check for significant audio events near shot boundaries
            nearby_events = [e for e in audio_events if abs(e - shot_start_time) < 2.0]
            
            # Long pause or significant audio change suggests scene boundary
            time_since_last_event = shot_start_time - last_audio_event_time
            if (time_since_last_event > 5.0 or len(nearby_events) > 0) and i > current_scene_start:
                # End current scene
                scenes.append((current_scene_start, i-1))
                current_scene_start = i
                if nearby_events:
                    last_audio_event_time = nearby_events[0]
        
        if current_scene_start < shot_count:
            scenes.append((current_scene_start, shot_count-1))
        
        if scenes:
            return scenes
    
    # Fallback: adaptive time-based grouping
    scenes = []
    shot_idx = 0
    
    while shot_idx < shot_count:
        start = shot_idx
        total_time = 0.0
        shot_count_in_scene = 0
        
        # Dynamic threshold based on content type (action vs dialogue)
        # Longer scenes for dialogue, shorter for action
        base_threshold = 8.0
        if shot_idx < len(durations_seconds):
            avg_shot_length = durations_seconds[shot_idx]
            # Short shots suggest action -> longer scene threshold
            # Long shots suggest dialogue -> shorter scene threshold
            threshold = base_threshold * (1.0 + 0.5 * (1.0 - avg_shot_length / 3.0))
        else:
            threshold = base_threshold
        
        while shot_idx < shot_count and (total_time < threshold or shot_count_in_scene < min_scene_shots):
            total_time += durations_seconds[shot_idx]
            shot_idx += 1
            shot_count_in_scene += 1
        
        end = shot_idx - 1
        scenes.append((start, end))
    
    return scenes

# -----------------------------
# Audio-assisted detection
# -----------------------------
def audio_onset_strength(audio_path, sr=22050, hop_length=512, use_multiband=True):
    """
    Improved audio onset detection with multi-band analysis and dynamic thresholding.
    Compute onset/envelope strength for audio file. Returns onset envelope, times, and RMS.
    """
    y, sr = librosa.load(audio_path, sr=sr)
    
    # Multi-band analysis
    if use_multiband:
        # Separate into low and high frequency bands
        y_low = librosa.effects.preemphasis(y, coef=0.97)
        y_high = y - y_low
        
        onset_env_low = librosa.onset.onset_strength(y=y_low, sr=sr, hop_length=hop_length, aggregate=np.median)
        onset_env_high = librosa.onset.onset_strength(y=y_high, sr=sr, hop_length=hop_length, aggregate=np.median)
        
        # Combine bands (weighted)
        onset_env = 0.6 * onset_env_low + 0.4 * onset_env_high
    else:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    
    times = librosa.times_like(onset_env, sr=sr, hop_length=hop_length)
    
    # RMS for dynamic thresholding
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    
    # Loudness (perceptual)
    loudness = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=hop_length)[0]
    
    return onset_env, times, rms, loudness

def cluster_onsets(onset_env, onset_times, window=0.1):
    """Cluster onset peaks into groups for stable matching."""
    # Find peaks
    peaks = []
    for i in range(1, len(onset_env)-1):
        if onset_env[i] > onset_env[i-1] and onset_env[i] > onset_env[i+1]:
            if onset_env[i] > np.percentile(onset_env, 75):  # Top 25% peaks
                peaks.append((onset_times[i], onset_env[i]))
    
    if not peaks:
        return []
    
    # Cluster peaks that are close in time
    clusters = []
    current_cluster = [peaks[0]]
    
    for peak_time, peak_strength in peaks[1:]:
        if peak_time - current_cluster[-1][0] < window:
            current_cluster.append((peak_time, peak_strength))
        else:
            # Finalize cluster (use strongest peak)
            if current_cluster:
                strongest = max(current_cluster, key=lambda x: x[1])
                clusters.append(strongest[0])
            current_cluster = [(peak_time, peak_strength)]
    
    if current_cluster:
        strongest = max(current_cluster, key=lambda x: x[1])
        clusters.append(strongest[0])
    
    return clusters

def audio_cut_alignment_score(cut_times_seconds, onset_env, onset_times, window=0.5, 
                               use_dynamic_threshold=True, rms=None, use_clustering=True):
    """
    Improved cut alignment with dynamic thresholding and onset clustering.
    For each cut time, check if there's onset within +/- window sec. Return fraction aligned.
    """
    if len(cut_times_seconds) == 0:
        return 0.0
    
    # Dynamic thresholding based on RMS/loudness
    if use_dynamic_threshold and rms is not None:
        # Normalize onset_env by RMS
        rms_normalized = (rms - rms.min()) / (rms.max() - rms.min() + 1e-9)
        threshold = np.mean(onset_env) + np.std(onset_env) * (1.0 + rms_normalized)
        significant_onsets = onset_env > threshold
    else:
        threshold = np.mean(onset_env) + np.std(onset_env)
        significant_onsets = onset_env > threshold
    
    # Use clustering for more stable matching
    if use_clustering:
        onset_clusters = cluster_onsets(onset_env, onset_times, window=window)
        aligned = 0
        for ct in cut_times_seconds:
            # Check if cut aligns with any cluster
            if any(np.abs(cluster_time - ct) <= window for cluster_time in onset_clusters):
                aligned += 1
    else:
        aligned = 0
        for ct in cut_times_seconds:
            # Find significant onsets in window
            mask = (np.abs(onset_times - ct) <= window) & significant_onsets
            if np.any(mask):
                aligned += 1
    
    return float(aligned / len(cut_times_seconds))

def detect_scene_whoosh_transitions(
    audio_path,
    scene_boundaries_times,
    sr=22050,
    hop_length=512,
    n_fft=2048,
    window_sec=0.5,
):
    """
    Detect whoosh-like audio transitions near scene boundaries.

    Whoosh characteristics:
    - Rising high-frequency content
    - High spectral flux (rapid spectral change)
    - Short transient duration (0.1–0.5s)

    Returns:
        List[float]: probability (0–1) of whoosh for each scene boundary
    """

    if audio_path is None or not os.path.exists(audio_path):
        return None

    try:
        # === Load audio ===
        y, sr = librosa.load(audio_path, sr=sr, mono=True)

        # === STFT ===
        stft = librosa.stft(y, n_fft=n_fft, hop_length=hop_length)
        magnitude = np.abs(stft)
        n_frames = magnitude.shape[1]

        # === Time axis ===
        times = librosa.frames_to_time(
            np.arange(n_frames),
            sr=sr,
            hop_length=hop_length,
            n_fft=n_fft,
        )

        spectral_rolloff = librosa.feature.spectral_rolloff(
            S=magnitude, sr=sr, roll_percent=0.85
        )[0]

        # === Spectral flux (same length as others) ===
        spectral_flux = np.zeros(n_frames)
        spectral_flux[1:] = np.sum(
            np.diff(magnitude, axis=1) ** 2, axis=0
        )

        # === High-frequency energy ===
        freqs = librosa.fft_frequencies(sr=sr, n_fft=n_fft)
        hf_mask = freqs > 5000  # whoosh ≈ high-frequency sweep
        high_freq_energy = magnitude[hf_mask].sum(axis=0)

        # === Normalize features (robust) ===
        def norm(x):
            return (x - np.median(x)) / (np.std(x) + 1e-9)

        spectral_rolloff_n = norm(spectral_rolloff)
        spectral_flux_n = norm(spectral_flux)
        high_freq_energy_n = norm(high_freq_energy)

        # === Global thresholds ===
        flux_thr = np.percentile(spectral_flux_n, 85)
        hf_thr = np.percentile(high_freq_energy_n, 85)

        # === Evaluate each scene boundary ===
        whoosh_probs = []

        for scene_time in scene_boundaries_times:
            mask = (
                (times >= scene_time - window_sec)
                & (times <= scene_time + window_sec)
            )

            if not np.any(mask):
                whoosh_probs.append(0.0)
                continue

            roll = spectral_rolloff_n[mask]
            flux = spectral_flux_n[mask]
            hf = high_freq_energy_n[mask]

            # === Feature scores ===

            # 1. Rising rolloff (HF sweep)
            roll_diff = np.diff(roll)
            roll_score = np.mean(roll_diff[roll_diff > 0]) if np.any(roll_diff > 0) else 0.0

            # 2. High spectral flux
            flux_score = np.mean(flux > flux_thr)

            # 3. HF energy spike
            hf_score = np.mean(hf > hf_thr)

            # === Combine (soft probability) ===
            prob = (
                0.35 * np.tanh(roll_score) +
                0.35 * flux_score +
                0.30 * hf_score
            )

            whoosh_probs.append(float(np.clip(prob, 0.0, 1.0)))

        return whoosh_probs

    except Exception as e:
        print(f"[WhooshDetectionError] {e}")
        return None


def analyze_scene_transition_types(scene_boundaries_shot_idx, shot_boundaries_frames, 
                                   hard_cuts, soft_events, motion_cuts, stylized_counts,
                                   fps):
    """
    Analyze transition types between scenes.
    Returns dict with transition type counts and distribution.
    """
    if not scene_boundaries_shot_idx or len(scene_boundaries_shot_idx) < 2:
        return {
            'hard_cut_transitions': 0,
            'fade_transitions': 0,
            'dissolve_transitions': 0,
            'motion_transitions': 0,
            'stylized_transitions': 0,
            'transition_type_distribution': {},
            'total_scene_transitions': 0
        }
    
    # Convert shot indices to frame indices
    scene_transition_types = []
    hard_cuts_set = set(hard_cuts) if hard_cuts else set()
    motion_cuts_set = set(motion_cuts) if motion_cuts else set()
    
    for i in range(len(scene_boundaries_shot_idx) - 1):
        scene_end_shot = scene_boundaries_shot_idx[i][1]
        scene_start_shot = scene_boundaries_shot_idx[i+1][0]
        
        # Find transition between scenes (the shot boundary between scenes)
        transition_shot_idx = scene_end_shot + 1
        if transition_shot_idx < len(shot_boundaries_frames) - 1:
            transition_frame = shot_boundaries_frames[transition_shot_idx]
            
            # Check what type of transition it is
            transition_type = 'hard_cut'  # default
            
            # Check for hard cut (most common)
            if transition_frame in hard_cuts_set:
                transition_type = 'hard_cut'
            # Check for soft transitions (fade/dissolve)
            elif soft_events:
                for soft_event in soft_events:
                    if soft_event.get('start', -1) <= transition_frame <= soft_event.get('end', -1):
                        transition_type = soft_event.get('type', 'hard_cut')
                        break
                # If found soft event, keep it; otherwise check motion
                if transition_type != 'hard_cut':
                    pass  # already set
                elif transition_frame in motion_cuts_set:
                    transition_type = 'motion_transition'
            # Check for motion transitions
            elif transition_frame in motion_cuts_set:
                transition_type = 'motion_transition'
            # Check for stylized transitions (if present)
            elif stylized_counts and sum(stylized_counts.values()) > 0:
                transition_type = 'stylized_transition'
            
            scene_transition_types.append(transition_type)
    
    # Count transition types
    transition_counts = {}
    for trans_type in scene_transition_types:
        transition_counts[trans_type] = transition_counts.get(trans_type, 0) + 1
    
    return {
        'hard_cut_transitions': transition_counts.get('hard_cut', 0),
        'fade_transitions': transition_counts.get('fade_in', 0) + transition_counts.get('fade_out', 0),
        'dissolve_transitions': transition_counts.get('dissolve', 0),
        'motion_transitions': transition_counts.get('motion_transition', 0),
        'stylized_transitions': transition_counts.get('stylized_transition', 0),
        'transition_type_distribution': transition_counts,
        'total_scene_transitions': len(scene_transition_types)
    }


def cut_timing_statistics(cut_frame_indices, fps, video_length_s):
    """
    From cut indices (frame numbers), compute statistics.
    Improved formulas: normalized entropy, CV-based uniformity.
    """
    if not cut_frame_indices:
        return {
            'cuts_per_minute': 0.0,
            'median_cut_interval': None,
            'min_cut_interval': None,
            'max_cut_interval': None,
            'cut_interval_std': None,
            'cut_interval_cv': None,  # coefficient of variation
            'cut_interval_entropy': None,
            'cut_rhythm_uniformity_score': None
        }
    times = np.array(cut_frame_indices, dtype=np.float32) / float(fps)
    intervals = np.diff(times)
    if intervals.size == 0:
        intervals = np.array([video_length_s])
    cpm = len(cut_frame_indices) / video_length_s * 60.0  # Only per_minute
    median = float(np.median(intervals))
    mn = float(np.min(intervals))
    mx = float(np.max(intervals))
    std = float(np.std(intervals))
    mean_int = float(np.mean(intervals))
    cv = std / (mean_int + 1e-9)
    n_bins = min(20, len(intervals))
    hist, _ = np.histogram(intervals, bins=n_bins)
    hist = hist + 1e-9
    ent = float(scipy.stats.entropy(hist))
    max_entropy = np.log(n_bins) if n_bins > 1 else 1.0
    ent_normalized = ent / (max_entropy + 1e-9)
    cv_clipped = np.clip(cv, 0.0, 1.0)
    uniformity = float(1.0 - cv_clipped)
    return {
        'cuts_per_minute': float(cpm),
        'median_cut_interval': median,
        'min_cut_interval': mn,
        'max_cut_interval': mx,
        'cut_interval_std': std,
        'cut_interval_cv': float(cv),
        'cut_interval_entropy': float(ent_normalized),
        'cut_rhythm_uniformity_score': uniformity
    }

def shot_length_stats(shot_frame_lengths, fps):
    """
    Compute shot length statistics including percentiles and histogram.
    """
    durations_s = np.array([seconds_from_fps(l, fps) for l in shot_frame_lengths])
    if durations_s.size == 0:
        return {}
    avg = float(durations_s.mean())
    med = float(np.median(durations_s))
    short_ratio = float((durations_s < 1.0).sum() / durations_s.size)
    long_ratio = float((durations_s > 4.0).sum() / durations_s.size)
    very_long = int((durations_s > 10.0).sum())
    extremely_short = int((durations_s < 0.25).sum())
    percentiles = np.percentile(durations_s, [10, 25, 75, 90])
    hist, bin_edges = np.histogram(durations_s, bins=8)
    hist_normalized = hist / (hist.sum() + 1e-9)  # Normalize to probabilities
    
    return {
        'avg_shot_length': avg,
        'median_shot_length': med,
        'shot_length_p10': float(percentiles[0]),
        'shot_length_p25': float(percentiles[1]),
        'shot_length_p75': float(percentiles[2]),
        'shot_length_p90': float(percentiles[3]),
        'short_shots_ratio': short_ratio,
        'long_shots_ratio': long_ratio,
        'very_long_shots_count': very_long,
        'extremely_short_shots_count': extremely_short,
        'shot_length_histogram': hist_normalized.tolist()  # 8-bin normalized histogram
    }


def _shot_length_stats_from_durations(shot_durations_s: List[float]) -> Dict[str, Any]:
    """
    Shot-length statistics computed directly from time-axis durations (seconds).
    """
    durations_s = np.asarray(
        [float(x) for x in shot_durations_s if x is not None and np.isfinite(float(x)) and float(x) >= 0.0],
        dtype=np.float32,
    )
    if durations_s.size == 0:
        return {
            "avg_shot_length": float("nan"),
            "median_shot_length": float("nan"),
            "shot_length_p10": float("nan"),
            "shot_length_p25": float("nan"),
            "shot_length_p75": float("nan"),
            "shot_length_p90": float("nan"),
            "short_shots_ratio": float("nan"),
            "long_shots_ratio": float("nan"),
            "very_long_shots_count": 0,
            "extremely_short_shots_count": 0,
            "shot_length_histogram": [],
        }
    avg = float(durations_s.mean())
    med = float(np.median(durations_s))
    short_ratio = float(np.mean(durations_s < 1.0))
    long_ratio = float(np.mean(durations_s > 4.0))
    very_long = int(np.sum(durations_s > 10.0))
    extremely_short = int(np.sum(durations_s < 0.25))
    percentiles = np.percentile(durations_s, [10, 25, 75, 90])
    hist, _ = np.histogram(durations_s, bins=8)
    hist_normalized = (hist.astype(np.float32) / (float(np.sum(hist)) + 1e-9)).tolist()
    return {
        "avg_shot_length": avg,
        "median_shot_length": med,
        "shot_length_p10": float(percentiles[0]),
        "shot_length_p25": float(percentiles[1]),
        "shot_length_p75": float(percentiles[2]),
        "shot_length_p90": float(percentiles[3]),
        "short_shots_ratio": short_ratio,
        "long_shots_ratio": long_ratio,
        "very_long_shots_count": very_long,
        "extremely_short_shots_count": extremely_short,
        "shot_length_histogram": hist_normalized,
    }

def classify_edit_style(cut_timing_stats, shot_stats, motion_cuts_count, jump_cuts_count,
                        stylized_counts, hard_cuts_count, duration_s):
    """
    Classify editing style based on cut statistics and patterns.
    Returns probabilities for each style from FEATURES.MD.
    """
    # Extract key metrics
    cpm = cut_timing_stats.get('cuts_per_minute', 0.0)
    median_interval = cut_timing_stats.get('median_cut_interval', 0.0)
    cut_std = cut_timing_stats.get('cut_interval_std', 0.0)
    uniformity = cut_timing_stats.get('cut_rhythm_uniformity_score', 0.0)
    
    avg_shot_length = shot_stats.get('avg_shot_length', 0.0)
    short_shots_ratio = shot_stats.get('short_shots_ratio', 0.0)
    extremely_short_count = shot_stats.get('extremely_short_shots_count', 0)
    
    # Normalize metrics
    jump_cut_ratio = jump_cuts_count / (duration_s / 60.0 + 1e-9)
    total_cuts = hard_cuts_count
    motion_transition_ratio = motion_cuts_count / (total_cuts + 1e-9)
    
    # Initialize probabilities
    styles = {
        'fast': 0.0,
        'cinematic': 0.0,
        'meme': 0.0,
        'social': 0.0,
        'slow': 0.0,
        'high_action': 0.0
    }
    
    if cpm > 20 and avg_shot_length < 2.0 and cut_std > 0.5:
        styles['fast'] = min(1.0, (cpm / 60.0) * 0.5 + (1.0 - avg_shot_length / 3.0) * 0.3 + cut_std * 0.2)
    
    if cpm < 8 and avg_shot_length > 5.0 and uniformity > 0.7:
        styles['slow'] = min(1.0, (1.0 - cpm / 15.0) * 0.4 + (avg_shot_length / 10.0) * 0.3 + uniformity * 0.3)
    
    if jump_cut_ratio > 3.0 and cpm > 15 and short_shots_ratio > 0.3:
        styles['social'] = min(1.0, (jump_cut_ratio / 10.0) * 0.4 + (cpm / 40.0) * 0.3 + short_shots_ratio * 0.3)
    
    if cpm < 6 and avg_shot_length > 8.0 and cut_std < 0.3:
        styles['slow'] = max(styles['slow'], min(1.0, (1.0 - cpm / 12.0) * 0.5 + (avg_shot_length / 15.0) * 0.3 + (1.0 - cut_std) * 0.2))
        if styles['slow'] < 0.3:
            styles['slow'] = min(1.0, (1.0 - cpm / 12.0) * 0.5 + (avg_shot_length / 15.0) * 0.3)
    
    stylized_count_total = sum(stylized_counts.values()) if stylized_counts else 0
    if 5 < cpm < 15 and avg_shot_length > 3.0 and stylized_count_total > total_cuts * 0.2:
        styles['cinematic'] = min(1.0, (avg_shot_length / 6.0) * 0.4 + (stylized_count_total / max(total_cuts, 1)) * 0.4 + (1.0 - abs(cpm - 10) / 10.0) * 0.2)
    
    if extremely_short_count > 5 and cpm > 25 and jump_cut_ratio > 2.0 and uniformity < 0.5:
        styles['meme'] = min(1.0, (extremely_short_count / 20.0) * 0.3 + (cpm / 50.0) * 0.3 + (jump_cut_ratio / 8.0) * 0.2 + (1.0 - uniformity) * 0.2)
    
    if motion_transition_ratio > 0.3 and cpm > 18 and 1.0 < avg_shot_length < 4.0:
        styles['high_action'] = min(1.0, motion_transition_ratio * 0.4 + (cpm / 40.0) * 0.3 + (1.0 - abs(avg_shot_length - 2.5) / 2.5) * 0.3)
    
    total = sum(styles.values()) + 1e-9
    for key in styles:
        styles[key] = styles[key] / total
    
    return styles


class CutDetectionPipeline(BaseModule):
    VERSION = VERSION
    SCHEMA_VERSION = SCHEMA_VERSION
    def __init__(
        self,
        rs_path: Optional[str] = None,
        fps: float = 30,
        device: str = 'auto',
        clip_zero_shot: bool = True,
        use_deep_features: bool = True,
        use_adaptive_thresholds: bool = True,
        use_semantic_clustering: bool = True,
        fade_threshold: float = 0.02,
        min_duration_frames: int = 4,
        use_flow_consistency: bool = True,
        # Performance knobs (quality-preserving):
        # - If Segmenter already downsized frames, these may not change anything.
        # - If frames are high-res (e.g., 720p+), these cap internal SSIM/flow compute resolution.
        ssim_max_side: int = 512,
        flow_max_side: int = 320,
        prefer_core_optical_flow: bool = False,
        require_core_optical_flow: bool = False,
        write_model_facing_npz: bool = True,
        require_model_facing_npz: bool = False,
        hard_cuts_cascade: bool = False,
        hard_cuts_cascade_keep_top_p: float = 0.25,
        hard_cuts_cascade_hist_margin: float = 0.0,
        clip_image_model_spec: Optional[str] = None,
        **kwargs: Any
    ):
        """
        Args:
            rs_path: Путь к хранилищу результатов
            fps: Частота кадров видео
            device: Устройство для обработки ('auto', 'cpu', 'cuda')
            clip_zero_shot: Использовать CLIP для классификации переходов
            use_deep_features: Использовать глубокие признаки для детекции
            use_adaptive_thresholds: Использовать адаптивные пороги
            use_semantic_clustering: Использовать семантическую кластеризацию
            fade_threshold: Порог для детекции fade переходов
            min_duration_frames: Минимальная длительность в кадрах
            use_flow_consistency: Использовать консистентность оптического потока
            **kwargs: Дополнительные параметры для BaseModule
        """
        # Определяем устройство
        if device == 'auto':
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        super().__init__(rs_path=rs_path, logger_name="cut_detection", **kwargs)
        
        self.fps = fps
        self.device = device
        self.use_deep_features = use_deep_features
        self.use_adaptive_thresholds = use_adaptive_thresholds
        self.use_semantic_clustering = use_semantic_clustering

        self.fade_threshold = fade_threshold
        self.min_duration_frames = min_duration_frames
        self.use_flow_consistency = use_flow_consistency
        self.ssim_max_side = int(ssim_max_side)
        self.flow_max_side = int(flow_max_side)
        self.prefer_core_optical_flow = bool(prefer_core_optical_flow)
        self.require_core_optical_flow = bool(require_core_optical_flow)
        self.write_model_facing_npz = bool(write_model_facing_npz)
        self.require_model_facing_npz = bool(require_model_facing_npz)
        self.hard_cuts_cascade = bool(hard_cuts_cascade)
        self.hard_cuts_cascade_keep_top_p = float(hard_cuts_cascade_keep_top_p)
        self.hard_cuts_cascade_hist_margin = float(hard_cuts_cascade_hist_margin)
        
        # Модели будут инициализированы в _do_initialize()
        self.embed_model = None
        self.transform = None
        self.clip_detector = None
        self._clip_zero_shot = clip_zero_shot
        self._clip_image_model_spec = str(clip_image_model_spec) if isinstance(clip_image_model_spec, str) and clip_image_model_spec else None
        self._clip_models_used_entry = None

    def get_models_used(self, config: Dict[str, Any], metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        cut_detection is mostly heuristic. If CLIP is enabled, we record it in models_used[].
        Baseline policy: CLIP MUST be resolved via dp_models (Triton spec) and core_clip outputs.
        """
        if not bool(self._clip_zero_shot):
            return []
        if self._clip_models_used_entry is not None:
            return [self._clip_models_used_entry]
        if not self._clip_image_model_spec:
            raise RuntimeError("cut_detection | use_clip=true requires --clip-image-model-spec (dp_models Triton spec)")
        from dp_models import get_global_model_manager  # type: ignore
        mm = get_global_model_manager()
        rm = mm.get(model_name=str(self._clip_image_model_spec))
        self._clip_models_used_entry = rm.models_used_entry
        return [rm.models_used_entry]
    
    def _do_initialize(self) -> None:
        """Инициализация моделей."""
        # Initialize embedding model for deep features
        if self.use_deep_features:
            # Production decision: forbid local torchvision pretrained weights (no-network policy).
            raise RuntimeError(
                "cut_detection | use_deep_features=true is not supported in baseline (no-network policy). "
                "If needed later, move this model into Triton and wire it as a core provider."
            )
        
        # Initialize CLIP detector
        if self._clip_zero_shot:
            # 1) load prompts + text embeddings from core_clip (single source-of-truth)
            core = self.load_core_provider("core_clip", file_name="embeddings.npz")
            if not isinstance(core, dict):
                raise RuntimeError("cut_detection | core_clip artifact is missing (required when use_clip=true)")
            prompts = core.get("cut_detection_transition_prompts")
            text_emb = core.get("cut_detection_transition_text_embeddings")
            if prompts is None or text_emb is None:
                raise RuntimeError(
                    "cut_detection | core_clip artifact missing cut_detection_transition_* fields (update core_clip schema)"
                )
            prompts_list = [str(x) for x in np.asarray(prompts).tolist()]
            text_emb_np = np.asarray(text_emb, dtype=np.float32)

            # 2) resolve Triton CLIP image encoder via ModelManager spec
            if not self._clip_image_model_spec:
                raise RuntimeError("cut_detection | use_clip=true requires --clip-image-model-spec (dp_models Triton spec)")
            from dp_models import get_global_model_manager  # type: ignore
            mm = get_global_model_manager()
            rm = mm.get(model_name=str(self._clip_image_model_spec))
            rp = rm.spec.runtime_params or {}
            handle = rm.handle or {}
            if not isinstance(rp, dict) or not rp:
                raise RuntimeError("cut_detection | CLIP image Triton spec has empty runtime_params")
            if not isinstance(handle, dict) or "client" not in handle:
                raise RuntimeError("cut_detection | CLIP image Triton spec has empty client handle")
            client = handle["client"]
            self._clip_models_used_entry = rm.models_used_entry

            self.clip_detector = StylizedTransitionZeroShot(
                client=client,
                triton_model_name=str(rp.get("triton_model_name")),
                triton_model_version=str(rp.get("triton_model_version") or "") or None,
                triton_input_name=str(rp.get("triton_input_name")),
                triton_output_name=str(rp.get("triton_output_name")),
                triton_input_datatype=str(rp.get("triton_input_datatype") or "FP32"),
                prompts=prompts_list,
                text_embeddings=text_emb_np,
                use_temporal_aggregation=True,
                use_multimodal=True,
                image_size=224,
            )
            self.logger.info("CLIP detector initialized (Triton + core_clip prompts)")

    def process(
        self,
        frame_manager: FrameManager,
        frame_indices: List[int],
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Основной метод обработки видео (интерфейс BaseModule).
        
        Args:
            frame_manager: Менеджер кадров
            frame_indices: Список индексов кадров для обработки
            config: Конфигурация модуля (может содержать audio_path для аудио-анализа)
        
        Returns:
            Словарь с результатами детекции переходов и статистикой
        """
        self.initialize()  # Гарантируем инициализацию моделей
        
        # Обновляем fps по факту (analysis timeline из frames_dir/metadata.json)
        try:
            self.fps = float(getattr(frame_manager, "fps", self.fps) or self.fps)
        except Exception:
            pass

        # audio path: auto-resolve Segmenter audio.wav if not provided
        frames_dir = str(getattr(frame_manager, "frames_dir", "")) if getattr(frame_manager, "frames_dir", None) is not None else ""
        audio_path = _resolve_audio_path(frames_dir, config)

        return self._process_video_frames(frame_manager, frame_indices, audio_path=audio_path)
    
    def _process_video_frames(self, frame_manager, frame_indices, audio_path=None):
        """
        Improved pipeline with all enhanced features.
        frames_bgr: list of BGR frames (np.uint8)
        audio_path: optional path to audio file for audio-assisted detection
        Returns dict of features and detections
        """
        n = len(frame_indices)
        times_s = _require_union_times_s(frame_manager, frame_indices)
        video_length_s = _video_length_seconds(times_s)
        if video_length_s <= 0.0:
            raise RuntimeError("cut_detection | invalid video length from union_timestamps_sec (no-fallback)")
        # Sampling quality check (decision): enforce reasonable max gap.
        # With min=400 frames on 20min video average gap ~3s; allow up to 6s hard cap.
        if times_s.size >= 2:
            max_gap = float(np.max(np.diff(times_s)))
            if max_gap > 6.0:
                raise RuntimeError(
                    f"cut_detection | sampling too sparse: max_gap_sec={max_gap:.2f} (>6.0). "
                    "Increase cut_detection sampling budget in Segmenter."
                )
        
        tik = time.time()

        # Pre-compute frame embeddings for semantic clustering
        frame_embeddings = None
        if self.use_semantic_clustering and self.embed_model is not None:
            frame_embeddings = []
            for idx in frame_indices[::10]:  # Sample every 10th frame for efficiency
                frame = frame_manager.get(idx)
                img_tensor = self.transform(ImageFromCV(frame)).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    emb = self.embed_model(img_tensor)
                    emb = emb.view(emb.size(0), -1)
                    emb = emb / (emb.norm(dim=1, keepdim=True)+1e-9)
                    frame_embeddings.append(emb.cpu().numpy()[0])
            frame_embeddings = np.array(frame_embeddings)

        
        tok = round(time.time() - tik, 2)
        logger.info(f"Frame embeddings success | Time: {tok}")
        tik = time.time()

        # Optional fast-path: reuse core_optical_flow motion curve to avoid duplicate optical flow computation.
        external_flow_mags = None
        if bool(self.prefer_core_optical_flow) or bool(self.require_core_optical_flow):
            try:
                core_flow = self.load_core_provider("core_optical_flow", file_name="flow.npz")
            except Exception as e:
                core_flow = None
                if bool(self.require_core_optical_flow):
                    raise RuntimeError(f"cut_detection | require_core_optical_flow=true but core_optical_flow load failed: {e}") from e
            if isinstance(core_flow, dict):
                try:
                    idx = np.asarray(core_flow.get("frame_indices"), dtype=np.int32).reshape(-1)
                    mcurve = np.asarray(core_flow.get("motion_norm_per_sec_mean"), dtype=np.float32).reshape(-1)
                    # Expect mcurve aligned with idx; first value is 0 for the first frame.
                    if idx.size >= 2 and mcurve.size == idx.size:
                        fi = np.asarray(frame_indices, dtype=np.int32).reshape(-1)
                        if fi.size == idx.size and np.all(fi == idx):
                            external_flow_mags = mcurve[1:]
                            logger.info("cut_detection | using core_optical_flow motion curve (aligned frame_indices)")
                        else:
                            msg = f"core_optical_flow frame_indices mismatch: core_n={int(idx.size)} cut_n={int(fi.size)}"
                            if bool(self.require_core_optical_flow):
                                raise RuntimeError(f"cut_detection | require_core_optical_flow=true but {msg}")
                            logger.warning("cut_detection | %s (falling back to local flow)", msg)
                except Exception as e:
                    if bool(self.require_core_optical_flow):
                        raise
                    logger.warning("cut_detection | failed to use core_optical_flow (fall back): %s", e)

        hard_idxs, hard_strengths, hard_dbg = detect_hard_cuts(
            frame_manager=frame_manager,
            frame_indices=frame_indices,
            use_deep_features=self.use_deep_features,
            use_adaptive_thresholds=self.use_adaptive_thresholds,
            temporal_smoothing=True,
            ssim_max_side=int(self.ssim_max_side),
            flow_max_side=int(self.flow_max_side),
            external_flow_mags=external_flow_mags,
            embed_model=self.embed_model,
            transform=self.transform,
            device=self.device,
            return_model_facing=True,
            cascade_enabled=bool(self.hard_cuts_cascade),
            cascade_keep_top_p=float(self.hard_cuts_cascade_keep_top_p),
            cascade_hist_margin=float(self.hard_cuts_cascade_hist_margin),
        )

        tok = round(time.time() - tik, 2)
        logger.info(f"Hard cuts success | Time: {tok}")
        tik = time.time()

        soft_events, soft_dbg = detect_soft_cuts(
            frame_manager=frame_manager,
            frame_indices=frame_indices,
            fps=self.fps,  # legacy arg; durations are recomputed below using union_timestamps_sec
            fade_threshold=self.fade_threshold,
            min_duration_frames=self.min_duration_frames,
            use_flow_consistency=self.use_flow_consistency,
            flow_max_side=int(self.flow_max_side),
            external_flow_mags=external_flow_mags,
            return_model_facing=True,
        )
        # Recompute duration_s for soft events using time-axis
        for e in soft_events:
            if not isinstance(e, dict):
                continue
            sp = e.get("start")
            ep = e.get("end")
            try:
                sp_i = int(sp)
                ep_i = int(ep)
                sp_i = max(0, min(sp_i, n - 1))
                ep_i = max(0, min(ep_i, n - 1))
                e["duration_s"] = float(max(times_s[ep_i] - times_s[sp_i], 0.0))
            except Exception:
                e["duration_s"] = float("nan")

        tok = round(time.time() - tik, 2)
        logger.info(f"Soft cuts success | Time: {tok}")
        tik = time.time()

        motion_idxs, motion_int, motion_types, motion_dbg = detect_motion_based_cuts(
            frame_manager=frame_manager,
            frame_indices=frame_indices,
            use_direction_analysis=True,
            adaptive_threshold=True,
            detect_speed_ramps=True,
            use_camera_motion_compensation=True,
            flow_max_side=int(self.flow_max_side),
            external_flow_mags=external_flow_mags,
            motion_cascade_enabled=True,
            return_model_facing=True,
        )

        tok = round(time.time() - tik, 2)
        logger.info(f"Motion-based cuts success | Time: {tok}")
        tik = time.time()

        stylized_counts = {}
        stylized_probs_per_cut = []
        if self.clip_detector is not None:
            candidate_windows = []
            candidate_scores = []

            # Reuse already computed hard-cut curves (cheapest + consistent with core_optical_flow and cascade).
            # We consider per-pair curves with index j=0..N-2; map to sampled-frame position idx=j+1.
            h = np.asarray((hard_dbg or {}).get("hist_diff_l1") or np.zeros((max(0, n - 1),), dtype=np.float32), dtype=np.float32).reshape(-1)
            sdrop = np.asarray((hard_dbg or {}).get("ssim_drop") or np.zeros((max(0, n - 1),), dtype=np.float32), dtype=np.float32).reshape(-1)
            fmag = np.asarray((hard_dbg or {}).get("flow_mag") or np.zeros((max(0, n - 1),), dtype=np.float32), dtype=np.float32).reshape(-1)
            # Candidate score heuristic (same as before; NaN treated as 0).
            sdrop = np.nan_to_num(sdrop, nan=0.0, posinf=0.0, neginf=0.0)
            fmag = np.nan_to_num(fmag, nan=0.0, posinf=0.0, neginf=0.0)
            scores = h + (sdrop * 2.0) + np.minimum(fmag / 10.0, 1.0)

            # Use the original threshold for now; can be made adaptive later.
            for j in range(int(min(scores.size, max(0, n - 1)))):
                if float(scores[j]) > 0.3:
                    candidate_windows.append(int(j + 1))
                    candidate_scores.append(float(scores[j]))
            
            logger.info(f"CLIP candidate-first: {len(candidate_windows)}/{n-1} windows selected")
            
            window = 5
            for candidate_idx in candidate_windows:
                if candidate_idx >= len(frame_indices):
                    continue
                
                actual_frame_idx = frame_indices[candidate_idx] if candidate_idx < len(frame_indices) else None
                if actual_frame_idx is None:
                    continue
                
                start_idx = max(0, candidate_idx - window//2)
                end_idx = min(len(frame_indices), candidate_idx + window//2 + 1)
                
                window_frames = [frame_manager.get(frame_indices[i]) for i in range(start_idx, end_idx)]
                
                if not window_frames:
                    continue
                
                if self.clip_detector.use_temporal_aggregation:
                    probs = self.clip_detector.predict_transition_temporal(window_frames, window_size=5)
                else:
                    probs = self.clip_detector.predict_transition(window_frames)
                
                label = max(probs.keys(), key=lambda k: probs[k])
                stylized_counts[label] = stylized_counts.get(label, 0) + 1
                stylized_probs_per_cut.append(probs)
            
            # Initialize missing labels to 0
            labels = self.clip_detector.labels if self.clip_detector else []
            for lbl in labels:
                if lbl not in stylized_counts:
                    stylized_counts[lbl] = 0
        else:
            labels = self.clip_detector.labels if self.clip_detector else []
            stylized_counts = {lbl: 0 for lbl in labels}

        tok = round(time.time() - tik, 2)
        logger.info(f"Stylized transitions via CLIP success | Time: {tok}")
        tik = time.time()

        # 5. Jump cuts detection with improvements
        # Jump cuts:
        # Baseline policy (decision): compute jump-cuts only at hard cuts, using core_face_landmarks + core_object_detections if available.
        jump_idxs, jump_scores = self._detect_jump_cuts_from_cores(
            frame_manager=frame_manager,
            frame_indices=frame_indices,
            hard_cut_positions=hard_idxs,
        )

        tok = round(time.time() - tik, 2)
        logger.info(f"Jump cuts success | Time: {tok}")
        tik = time.time()

        # 6. Shots segmentation
        # NOTE: `hard_idxs` are positions in the sampled sequence (0..n-1).
        shot_boundaries_pos = [0] + hard_idxs + [n]
        shot_lengths = [shot_boundaries_pos[i + 1] - shot_boundaries_pos[i] for i in range(len(shot_boundaries_pos) - 1)]

        # Also provide union-frame indices for boundaries (best-effort end marker).
        # last boundary (pos==n) is mapped to the last sampled frame index.
        if n > 0:
            shot_boundaries_frame_indices = [
                int(frame_indices[p]) if p < n else int(frame_indices[-1]) for p in shot_boundaries_pos
            ]
        else:
            shot_boundaries_frame_indices = []

        # Downstream helpers expect shot boundaries in "frame index" domain (timestamps via /fps).
        # Use sampled frame_indices mapping (Segmenter contract: frame_indices are source-frame indices).
        # Fallback to positions if something is off (best-effort, but should not happen with valid sampling).
        shot_boundaries = shot_boundaries_frame_indices if shot_boundaries_frame_indices else shot_boundaries_pos

        tok = round(time.time() - tik, 2)
        logger.info(f"Shots segmentation success | Time: {tok}")
        tik = time.time()

        # 7. Audio processing для сцено‑зависимых метрик (опционально)
        # Обрабатываем аудио только если оно участвует во взаимосвязи с видео (alignment, whoosh и т.п.).
        # Чисто аудио‑фичи не считаем.
        audio_events = None
        onset_env, onset_times, rms, loudness = None, None, None, None
        if audio_path is not None and os.path.exists(audio_path):
            onset_env, onset_times, rms, loudness = audio_onset_strength(audio_path, use_multiband=True)
            # Extract significant audio events (будут использоваться в сценах и alignment)
            threshold = np.mean(onset_env) + np.std(onset_env)
            audio_events = onset_times[onset_env > threshold].tolist()
        
        tok = round(time.time() - tik, 2)
        logger.info(f"Audio processing success | Time: {tok}")
        tik = time.time()
        
        # 8. Scenes grouping
        # Сначала пробуем использовать внешний модуль scene_classification (npz/json),
        # чтобы не дублировать логику и использовать внешние артефакты как источник истины.
        scenes = None
        try:
            scene_data = self.load_dependency_results("scene_classification")
        except Exception:
            scene_data = None
        
        if isinstance(scene_data, dict):
            # Вариант 1: scenes лежит на верхнем уровне
            if "scenes" in scene_data and isinstance(scene_data["scenes"], dict):
                scenes_raw = scene_data["scenes"]
            # Вариант 2: scenes внутри aggregated
            elif (
                "aggregated" in scene_data
                and isinstance(scene_data["aggregated"], dict)
                and "scenes" in scene_data["aggregated"]
                and isinstance(scene_data["aggregated"]["scenes"], dict)
            ):
                scenes_raw = scene_data["aggregated"]["scenes"]
            else:
                scenes_raw = None
            
            if scenes_raw:
                # Преобразуем структуру scene_classification в список (start, end)
                tmp_scenes = []
                for _, s in scenes_raw.items():
                    try:
                        start_f = int(s.get("start_frame"))
                        end_f = int(s.get("end_frame"))
                        if end_f >= start_f:
                            tmp_scenes.append((start_f, end_f))
                    except Exception:
                        continue
                if tmp_scenes:
                    scenes = tmp_scenes
        
        # Если внешние сцены недоступны — используем внутреннюю эвристику
        if scenes is None:
            scenes = scene_boundaries_from_shots(
                hard_idxs,
                shot_lengths,
                self.fps,
                use_semantic_clustering=self.use_semantic_clustering,
                frame_embeddings=frame_embeddings,
                audio_events=audio_events,
                embed_model=self.embed_model,
                transform=self.transform,
                device=self.device,
            )
        scene_count = len(scenes)
        scene_avg_len = float(np.mean([end-start+1 for (start,end) in scenes])) if scenes else 0.0

        tok = round(time.time() - tik, 2)
        logger.info(f"Scenes grouping success | Time: {tok}")
        tik = time.time()

        # 9. Audio assisted cut alignment
        audio_align_score = None
        audio_spike_ratio = None
        if audio_path is not None and os.path.exists(audio_path) and onset_env is not None:
            cut_times = [ci / float(self.fps) for ci in hard_idxs]
            audio_align_score = audio_cut_alignment_score(
                cut_times, onset_env, onset_times, window=0.5,
                use_dynamic_threshold=True,
                rms=rms,
                use_clustering=True
            )
            # spike ratio: fraction of cuts that align with strong onset peaks
            audio_spike_ratio = float(np.sum(onset_env > (np.mean(onset_env)+np.std(onset_env))) / (len(onset_env)+1e-9))

        # 9. Aggregation stats
        hard_cut_times_s = [float(times_s[int(p)]) for p in hard_idxs if 0 <= int(p) < int(times_s.size)]
        cut_timing_stats_dict = _cut_timing_statistics_from_times(hard_cut_times_s, video_length_s)

        # Shot duration stats in seconds using time-axis
        shot_durations_s: List[float] = []
        for a, b in zip(shot_boundaries_pos[:-1], shot_boundaries_pos[1:]):
            a_i = int(max(0, min(int(a), n - 1)))
            b_i = int(max(0, min(int(b), n)))
            if b_i >= n:
                dur = float(times_s[-1] - times_s[a_i])
            else:
                dur = float(times_s[b_i] - times_s[a_i])
            shot_durations_s.append(float(max(dur, 0.0)))
        shot_stats = _shot_length_stats_from_durations(shot_durations_s)

        tok = round(time.time() - tik, 2)
        logger.info(f"Audio assisted success | Time: {tok}")
        tik = time.time()

        # 10. Compose features
        features = {}
        # hard cuts
        features['hard_cuts_count'] = len(hard_idxs)
        if hard_strengths:
            features['hard_cut_strength_mean'] = float(np.mean(hard_strengths))
            # Percentiles for strength distribution
            strengths_array = np.array(hard_strengths)
            percentiles = np.percentile(strengths_array, [25, 50, 75])
            features['hard_cut_strength_p25'] = float(percentiles[0])
            features['hard_cut_strength_p50'] = float(percentiles[1])
            features['hard_cut_strength_p75'] = float(percentiles[2])
        else:
            features['hard_cut_strength_mean'] = 0.0
            features['hard_cut_strength_p25'] = 0.0
            features['hard_cut_strength_p50'] = 0.0
            features['hard_cut_strength_p75'] = 0.0
        features['hard_cuts_per_minute'] = float(cut_timing_stats_dict['cuts_per_minute'])

        # soft cuts
        features['fade_in_count'] = sum(1 for e in soft_events if e['type']=='fade_in')
        features['fade_out_count'] = sum(1 for e in soft_events if e['type']=='fade_out')
        features['dissolve_count'] = sum(1 for e in soft_events if e['type']=='dissolve')
        features['avg_fade_duration'] = float(np.mean([e['duration_s'] for e in soft_events])) if soft_events else 0.0

        # motion-based
        features['motion_cuts_count'] = len(motion_idxs)
        features['motion_cut_intensity_score'] = float(np.mean(motion_int)) if motion_int else 0.0
        features['flow_spike_ratio'] = float(len(motion_idxs) / (len(hard_idxs)+1e-9))
        # Motion type counts (including speed ramp)
        if motion_types:
            features['whip_pan_transitions_count'] = sum(1 for t in motion_types if t == 'whip_pan')
            features['zoom_transition_count'] = sum(1 for t in motion_types if t == 'zoom')
            features['speed_ramp_cuts_count'] = sum(1 for t in motion_types if t == 'speed_ramp')
        else:
            features['whip_pan_transitions_count'] = 0
            features['zoom_transition_count'] = 0
            features['speed_ramp_cuts_count'] = 0

        # stylized transitions counts
        for k,v in stylized_counts.items():
            key = f"transition_{k.replace(' ','_').lower()}_count"
            features[key] = int(v)

        # jump cuts
        features['jump_cuts_count'] = len(jump_idxs)
        features['jump_cut_intensity'] = float(np.mean(jump_scores)) if jump_scores else 0.0
        features['jump_cut_ratio_per_minute'] = float(len(jump_idxs) / (video_length_s/60.0+1e-9))

        # timing & rhythm
        features.update(cut_timing_stats_dict)
        features.update(shot_stats)

        # scene
        features['scene_count'] = scene_count
        features['avg_scene_length_shots'] = scene_avg_len
        features['scene_to_shot_ratio'] = float(scene_count / (len(shot_lengths)+1e-9))

        tok = round(time.time() - tik, 2)
        logger.info(f"Compose success | Time: {tok}")
        tik = time.time()

        # Scene transition types analysis
        if scenes:
            scene_transition_analysis = analyze_scene_transition_types(
                scenes, shot_boundaries, hard_idxs, soft_events, motion_idxs,
                stylized_counts, self.fps
            )
            features.update({
                'scene_hard_cut_transitions': scene_transition_analysis.get('hard_cut_transitions', 0),
                'scene_fade_transitions': scene_transition_analysis.get('fade_transitions', 0),
                'scene_dissolve_transitions': scene_transition_analysis.get('dissolve_transitions', 0),
                'scene_motion_transitions': scene_transition_analysis.get('motion_transitions', 0),
                'scene_stylized_transitions': scene_transition_analysis.get('stylized_transitions', 0)
            })
        else:
            features.update({
                'scene_hard_cut_transitions': 0,
                'scene_fade_transitions': 0,
                'scene_dissolve_transitions': 0,
                'scene_motion_transitions': 0,
                'scene_stylized_transitions': 0
            })

        tok = round(time.time() - tik, 2)
        logger.info(f"Scene transition success | Time: {tok}")
        tik = time.time()

        # audio‑video взаимосвязь (оставляем только метрики, привязанные к видео)
        features['audio_cut_alignment_score'] = float_or_zero(audio_align_score) if audio_align_score is not None else 0.0
        features['audio_spike_cut_ratio'] = float_or_zero(audio_spike_ratio) if audio_spike_ratio is not None else 0.0
        
        # Scene whoosh transition probability (анализируем только в точках сценовых переходов)
        scene_whoosh_prob = None
        features['scene_whoosh_transition_prob'] = 0.0
        if audio_path is not None and os.path.exists(audio_path) and scenes:
            scene_boundaries_times = []
            for scene_start, scene_end in scenes:
                # Get time of transition (start of next scene)
                if scene_start < len(shot_boundaries) - 1:
                    transition_frame = shot_boundaries[scene_start] if scene_start > 0 else 0
                    transition_time = transition_frame / float(self.fps)
                    scene_boundaries_times.append(transition_time)
            
            if scene_boundaries_times:
                whoosh_probs = detect_scene_whoosh_transitions(
                    audio_path, scene_boundaries_times
                )
                if whoosh_probs:
                    scene_whoosh_prob = float(np.mean(whoosh_probs))
                    features['scene_whoosh_transition_prob'] = scene_whoosh_prob

        tok = round(time.time() - tik, 2)
        logger.info(f"Scene whoosh transition success | Time: {tok}")
        tik = time.time()

        # stylistic edit classification (zero-shot): we can compute per-video by averaging stylized_probs_per_cut
        if self.clip_detector is not None and stylized_probs_per_cut:
            # average prob per label
            labels = self.clip_detector.labels
            avg_probs = {lbl: 0.0 for lbl in labels}
            count = len(stylized_probs_per_cut)
            for p in stylized_probs_per_cut:
                for lbl,val in p.items():
                    avg_probs[lbl] += val
            for lbl in labels:
                avg_probs[lbl] /= count
                features[f"edit_style_{lbl.replace(' ','_').lower()}_prob"] = float(avg_probs[lbl])
        else:
            # fill zeros for labels (if CLIP детектор недоступен)
            if self.clip_detector is not None:
                labels = self.clip_detector.labels
                for lbl in labels:
                    features[f"edit_style_{lbl.replace(' ','_').lower()}_prob"] = 0.0

        tok = round(time.time() - tik, 2)
        logger.info(f"stylistic edit classification success | Time: {tok}")

        # Edit style classification based on statistics (from FEATURES.MD)
        edit_styles = classify_edit_style(
            cut_timing_stats_dict, shot_stats, len(motion_idxs), len(jump_idxs),
            stylized_counts, len(hard_idxs), video_length_s
        )
        features['edit_style_fast_prob'] = float(edit_styles.get('fast', 0.0))
        features['edit_style_slow_prob'] = float(edit_styles.get('slow', 0.0))
        features['edit_style_cinematic_prob'] = float(edit_styles.get('cinematic', 0.0))
        features['edit_style_meme_prob'] = float(edit_styles.get('meme', 0.0))
        features['edit_style_social_prob'] = float(edit_styles.get('social', 0.0))
        features['edit_style_high_action_prob'] = float(edit_styles.get('high_action', 0.0))

        # Map cut positions -> union frame indices for downstream consumers.
        hard_cut_frame_indices = [int(frame_indices[i]) for i in hard_idxs] if hard_idxs else []
        motion_cut_frame_indices = [int(frame_indices[i]) for i in motion_idxs] if motion_idxs else []
        jump_cut_frame_indices = [int(frame_indices[i]) for i in jump_idxs] if jump_idxs else []

        # Provide raw detections for downstream use
        detections = {
            # positions in sampled sequence
            'hard_cut_pos': hard_idxs,
            'motion_cut_pos': motion_idxs,
            'jump_cut_pos': jump_idxs,
            # union-domain frame indices
            'hard_cut_frame_indices': hard_cut_frame_indices,
            'motion_cut_frame_indices': motion_cut_frame_indices,
            'jump_cut_frame_indices': jump_cut_frame_indices,
            # legacy keys (compat): kept but they are POSITIONS (not frame indices)
            'hard_cut_indices': hard_idxs,
            'hard_cut_strengths': hard_strengths,
            'soft_events': soft_events,
            'motion_cut_indices': motion_idxs,
            'motion_cut_intensities': motion_int,
            'motion_cut_types': motion_types,
            'stylized_counts': stylized_counts,
            'jump_cut_indices': jump_idxs,
            'jump_cut_scores': jump_scores,
            # both representations: positions in sampled sequence, and union frame indices
            'shot_boundaries_pos': shot_boundaries_pos,
            'shot_boundaries_frame_indices': shot_boundaries_frame_indices,
            'scene_boundaries_shot_idx': scenes
        }

        # -----------------------------
        # Model-facing NPZ (v1): dense curves + unified events stream
        # -----------------------------
        model_facing_path = None
        if bool(self.write_model_facing_npz):
            try:
                fi_np = np.asarray(frame_indices, dtype=np.int32).reshape(-1)
                ts_np = np.asarray(times_s, dtype=np.float32).reshape(-1)
                pair_times_s = (0.5 * (ts_np[1:] + ts_np[:-1])).astype(np.float32)
                pair_dt_s = (ts_np[1:] - ts_np[:-1]).astype(np.float32)

                dbg = hard_dbg or {}
                exp_len = int(max(0, n - 1))

                def _arr_1d(key: str, *, dtype, default_value: float = 0.0) -> np.ndarray:
                    v = dbg.get(key, None)
                    if v is None:
                        return np.full((exp_len,), float(default_value), dtype=dtype)
                    a = np.asarray(v, dtype=dtype).reshape(-1)
                    if int(a.size) != exp_len:
                        out = np.full((exp_len,), float(default_value), dtype=dtype)
                        m = min(exp_len, int(a.size))
                        if m > 0:
                            out[:m] = a[:m]
                        return out
                    return a

                hdiff = _arr_1d("hist_diff_l1", dtype=np.float32, default_value=0.0)
                # In cascade mode the signal may be intentionally not computed -> prefer NaN + valid_mask.
                ssim_drop = _arr_1d("ssim_drop", dtype=np.float32, default_value=float("nan"))
                flow_mag = _arr_1d("flow_mag", dtype=np.float32, default_value=float("nan"))
                deep_cosine_dist = _arr_1d("deep_cosine_dist", dtype=np.float32, default_value=float("nan"))
                hard_score = _arr_1d("hard_score", dtype=np.float32, default_value=0.0)

                # unified events arrays
                ev_t: list[float] = []
                ev_type: list[int] = []
                ev_strength: list[float] = []
                ev_pair: list[int] = []
                ev_contrib: list[list[bool]] = []
                ev_start: list[float] = []
                ev_end: list[float] = []

                def _nearest_pair_index(t: float) -> int:
                    if pair_times_s.size == 0:
                        return 0
                    j = int(np.argmin(np.abs(pair_times_s - float(t))))
                    return max(0, min(j, int(pair_times_s.size) - 1))

                trig = dbg.get("triggers") or {}
                trig_hist = np.asarray(trig.get("hist"), dtype=bool).reshape(-1) if trig.get("hist") is not None else np.zeros((exp_len,), dtype=bool)
                trig_ssim = np.asarray(trig.get("ssim"), dtype=bool).reshape(-1) if trig.get("ssim") is not None else np.zeros((exp_len,), dtype=bool)
                trig_flow = np.asarray(trig.get("flow"), dtype=bool).reshape(-1) if trig.get("flow") is not None else np.zeros((exp_len,), dtype=bool)
                trig_deep = np.asarray(trig.get("deep"), dtype=bool).reshape(-1) if trig.get("deep") is not None else np.zeros((exp_len,), dtype=bool)
                vmask = dbg.get("valid_mask") or {}
                ssim_valid_mask = np.asarray(vmask.get("ssim"), dtype=bool).reshape(-1) if vmask.get("ssim") is not None else np.ones((exp_len,), dtype=bool)
                flow_valid_mask = np.asarray(vmask.get("flow"), dtype=bool).reshape(-1) if vmask.get("flow") is not None else np.ones((exp_len,), dtype=bool)
                deep_valid_mask = np.asarray(vmask.get("deep"), dtype=bool).reshape(-1) if vmask.get("deep") is not None else np.zeros((exp_len,), dtype=bool)

                thr = dbg.get("thresholds") or {}
                threshold_hist = np.full((exp_len,), float(thr.get("hist", float("nan"))), dtype=np.float32)
                threshold_ssim = np.full((exp_len,), float(thr.get("ssim", float("nan"))), dtype=np.float32)
                threshold_flow = np.full((exp_len,), float(thr.get("flow", float("nan"))), dtype=np.float32)
                threshold_deep = np.full((exp_len,), float(thr.get("deep", float("nan"))), dtype=np.float32)
                # If a signal was not computed at all (all mask=false), the threshold is semantically undefined.
                if exp_len > 0 and not bool(np.any(ssim_valid_mask)):
                    threshold_ssim[:] = np.nan
                if exp_len > 0 and not bool(np.any(flow_valid_mask)):
                    threshold_flow[:] = np.nan
                if exp_len > 0 and not bool(np.any(deep_valid_mask)):
                    threshold_deep[:] = np.nan

                # Hard cuts (type_id=1)
                for pos, strength in zip(hard_idxs or [], hard_strengths or []):
                    try:
                        p = int(pos) - 1
                    except Exception:
                        continue
                    if p < 0 or p >= int(pair_times_s.size):
                        continue
                    ev_t.append(float(pair_times_s[p]))
                    ev_type.append(1)
                    ev_strength.append(float(strength))
                    ev_pair.append(int(p))
                    ev_contrib.append([bool(trig_hist[p]), bool(trig_ssim[p]), bool(trig_flow[p]), bool(trig_deep[p])])
                    ev_start.append(float(pair_times_s[p]))
                    ev_end.append(float(pair_times_s[p]))

                # Soft events (fade/dissolve)
                for e in soft_events or []:
                    if not isinstance(e, dict):
                        continue
                    try:
                        typ = str(e.get("type") or "")
                        sp = int(e.get("start"))
                        ep = int(e.get("end"))
                    except Exception:
                        continue
                    sp = max(0, min(sp, n - 1))
                    ep = max(0, min(ep, n - 1))
                    t_mid = 0.5 * float(ts_np[sp] + ts_np[ep]) if ts_np.size else 0.0
                    p = _nearest_pair_index(t_mid)
                    if typ == "fade_in":
                        tid = 2
                    elif typ == "fade_out":
                        tid = 3
                    elif typ == "dissolve":
                        tid = 4
                    else:
                        continue
                    ev_t.append(float(t_mid))
                    ev_type.append(int(tid))
                    ev_strength.append(float(e.get("duration_s")) if e.get("duration_s") is not None else float("nan"))
                    ev_pair.append(int(p))
                    ev_start.append(float(ts_np[sp]) if ts_np.size else float(t_mid))
                    ev_end.append(float(ts_np[ep]) if ts_np.size else float(t_mid))

                # Motion events
                for pos, strength, mtyp in zip(motion_idxs or [], motion_int or [], motion_types or []):
                    try:
                        p = int(pos) - 1
                    except Exception:
                        continue
                    if p < 0 or p >= int(pair_times_s.size):
                        continue
                    mt = str(mtyp or "")
                    if mt == "whip_pan":
                        tid = 6
                    elif mt == "zoom":
                        tid = 7
                    elif mt == "speed_ramp":
                        tid = 8
                    else:
                        tid = 5
                    ev_t.append(float(pair_times_s[p]))
                    ev_type.append(int(tid))
                    ev_strength.append(float(strength))
                    ev_pair.append(int(p))
                    ev_start.append(float(pair_times_s[p]))
                    ev_end.append(float(pair_times_s[p]))

                # Jump cuts (type_id=9)
                for pos, strength in zip(jump_idxs or [], jump_scores or []):
                    try:
                        p = int(pos) - 1
                    except Exception:
                        continue
                    if p < 0 or p >= int(pair_times_s.size):
                        continue
                    ev_t.append(float(pair_times_s[p]))
                    ev_type.append(9)
                    ev_strength.append(float(strength))
                    ev_pair.append(int(p))
                    ev_start.append(float(pair_times_s[p]))
                    ev_end.append(float(pair_times_s[p]))

                E = int(len(ev_t))
                event_times_s = np.asarray(ev_t, dtype=np.float32).reshape(-1)
                event_type_id = np.asarray(ev_type, dtype=np.int16).reshape(-1)
                event_strength = np.asarray(ev_strength, dtype=np.float32).reshape(-1)
                event_pair_index = np.asarray(ev_pair, dtype=np.int32).reshape(-1)
                event_start_time_s = np.asarray(ev_start, dtype=np.float32).reshape(-1) if ev_start else np.zeros((E,), dtype=np.float32)
                event_end_time_s = np.asarray(ev_end, dtype=np.float32).reshape(-1) if ev_end else np.zeros((E,), dtype=np.float32)
                if ev_contrib:
                    contrib = np.asarray(ev_contrib, dtype=bool)
                    if contrib.ndim != 2 or contrib.shape[0] != E:
                        contrib = np.zeros((E, 4), dtype=bool)
                else:
                    contrib = np.zeros((E, 4), dtype=bool)

                model_facing = {
                    "frame_indices": fi_np,
                    "union_timestamps_sec": ts_np,
                    "times_s": ts_np,
                    "pair_times_s": pair_times_s,
                    "pair_dt_s": pair_dt_s,
                    "hist_diff_l1": hdiff,
                    "ssim_drop": ssim_drop,
                    "flow_mag": flow_mag,
                    "hard_score": hard_score,
                    "deep_cosine_dist": deep_cosine_dist,
                "ssim_valid_mask": ssim_valid_mask,
                "flow_valid_mask": flow_valid_mask,
                "deep_valid_mask": deep_valid_mask,
                "threshold_hist": threshold_hist,
                "threshold_ssim": threshold_ssim,
                "threshold_flow": threshold_flow,
                "threshold_deep": threshold_deep,
                    "event_times_s": event_times_s,
                    "event_start_time_s": event_start_time_s,
                    "event_end_time_s": event_end_time_s,
                    "event_type_id": event_type_id,
                    "event_strength": event_strength,
                    "event_pair_index": event_pair_index,
                    "event_contrib_mask": contrib,
                }

                # Optional: include soft/motion raw curves (model-facing inputs for encoder).
                if isinstance(locals().get("soft_dbg"), dict):
                    for k in (
                        "soft_hsv_v",
                        "soft_lab_l",
                        "soft_hist_diff_l1",
                        "soft_flow_mag",
                        "soft_flow_valid_mask",
                    ):
                        if k in soft_dbg:
                            model_facing[k] = soft_dbg[k]
                if isinstance(locals().get("motion_dbg"), dict):
                    for k in (
                        "motion_flow_mag",
                        "motion_dir_consistency",
                        "motion_mag_variance",
                        "motion_camera_motion_flag",
                        "motion_dir_valid_mask",
                        "motion_var_valid_mask",
                        "motion_cam_valid_mask",
                    ):
                        if k in motion_dbg:
                            model_facing[k] = motion_dbg[k]

                # Save with BaseModule.save_results(), then rename to stable prefix.
                frames_dir = str(getattr(frame_manager, "frames_dir", "")) if getattr(frame_manager, "frames_dir", None) is not None else ""
                meta = self.load_metadata(frames_dir) if frames_dir else {}
                save_meta = {
                    "total_frames": meta.get("total_frames"),
                    "processed_frames": int(len(frame_indices)),
                    "frames_dir": frames_dir,
                    "platform_id": meta.get("platform_id"),
                    "video_id": meta.get("video_id"),
                    "run_id": meta.get("run_id"),
                    "sampling_policy_version": meta.get("sampling_policy_version"),
                    "config_hash": meta.get("config_hash"),
                    "dataprocessor_version": meta.get("dataprocessor_version"),
                    "analysis_fps": meta.get("analysis_fps"),
                    "analysis_width": meta.get("analysis_width"),
                    "analysis_height": meta.get("analysis_height"),
                    "schema_version": "cut_detection_model_facing_npz_v1",
                    "cut_detection_config": {
                        "ssim_max_side": int(self.ssim_max_side),
                        "flow_max_side": int(self.flow_max_side),
                        "prefer_core_optical_flow": bool(self.prefer_core_optical_flow),
                        "require_core_optical_flow": bool(self.require_core_optical_flow),
                        "use_deep_features": bool(self.use_deep_features),
                        "use_adaptive_thresholds": bool(self.use_adaptive_thresholds),
                        "temporal_smoothing": True,
                        "hard_cuts_cascade": bool(self.hard_cuts_cascade),
                        "hard_cuts_cascade_keep_top_p": float(self.hard_cuts_cascade_keep_top_p),
                        "hard_cuts_cascade_hist_margin": float(self.hard_cuts_cascade_hist_margin),
                    },
                    "flow_source": str(dbg.get("flow_source") or "unknown"),
                    "flow_mag_units": str(dbg.get("flow_mag_units") or "unknown"),
                    "thresholds": dbg.get("thresholds") or {},
                    "event_type_map": {
                        1: "hard_cut",
                        2: "fade_in",
                        3: "fade_out",
                        4: "dissolve",
                        5: "motion_cut",
                        6: "whip_pan",
                        7: "zoom",
                        8: "speed_ramp",
                        9: "jump_cut",
                    },
                    "event_contrib_sources": ["hist", "ssim", "flow", "deep"],
                }
                try:
                    save_meta["models_used"] = self.get_models_used(config={"clip_image_model_spec": self._clip_image_model_spec}, metadata=meta or {})
                except Exception:
                    save_meta["models_used"] = []

                tmp_path = self.save_results(results=model_facing, metadata=save_meta, use_compressed=False)
                base = os.path.basename(tmp_path)
                prefix = f"{self.module_name}_features_"
                if base.startswith(prefix):
                    new_base = f"{self.module_name}_model_facing_" + base[len(prefix):]
                else:
                    new_base = f"{self.module_name}_model_facing_{base}"
                new_path = os.path.join(os.path.dirname(tmp_path), new_base)
                os.replace(tmp_path, new_path)
                model_facing_path = new_path
                logger.info("cut_detection | model-facing NPZ saved: %s", model_facing_path)
            except Exception as e:
                if bool(self.require_model_facing_npz):
                    raise RuntimeError(f"cut_detection | require_model_facing_npz=true but write failed: {e}") from e
                logger.warning("cut_detection | failed to write model-facing NPZ (non-fatal): %s", e)

        # Provide frame_indices for consumers (union-domain indices).
        return {
            "frame_indices": np.asarray(frame_indices, dtype=np.int32),
            "times_s": times_s,
            "features": features,
            "detections": detections,
            "model_facing_npz_path": model_facing_path,
        }

    def _detect_jump_cuts_from_cores(
        self,
        *,
        frame_manager: FrameManager,
        frame_indices: List[int],
        hard_cut_positions: List[int],
        face_change_thresh: float = 0.35,
        bg_ssim_thresh: float = 0.80,
        person_score_thresh: float = 0.35,
    ) -> tuple[List[int], List[float]]:
        """
        Jump cut = hard cut with strong face geometry change AND similar background.
        Uses:
        - core_face_landmarks: face_landmarks + face_present
        - core_object_detections: person presence (optional gate)
        If cores do not cover required frames, candidate is skipped (valid empty at sub-feature level).
        Returns:
            jump_idxs: positions in sampled sequence (same domain as hard_cut_positions)
            jump_scores: float scores
        """
        if self.rs_path is None:
            raise RuntimeError("cut_detection | rs_path is required to load core_* dependencies (no-fallback)")

        # Load core_face_landmarks
        try:
            face_npz = self.load_core_provider("core_face_landmarks")
        except Exception as e:
            raise RuntimeError(f"cut_detection | failed to load core_face_landmarks artifact: {e}") from e
        if not isinstance(face_npz, dict):
            raise RuntimeError("cut_detection | core_face_landmarks artifact missing/invalid (no-fallback)")
        fi_face = face_npz.get("frame_indices")
        face_present = face_npz.get("face_present")
        face_landmarks = face_npz.get("face_landmarks")
        if fi_face is None or face_present is None or face_landmarks is None:
            raise RuntimeError("cut_detection | core_face_landmarks missing required keys (no-fallback)")
        fi_face = np.asarray(fi_face, dtype=np.int32)
        face_present = np.asarray(face_present)
        face_landmarks = np.asarray(face_landmarks)
        face_map = {int(x): i for i, x in enumerate(fi_face.tolist())}

        # Load core_object_detections (required in baseline per audit decision)
        person_map: Dict[int, bool] = {}
        try:
            det_npz = self.load_core_provider("core_object_detections")
        except Exception as e:
            raise RuntimeError(f"cut_detection | failed to load core_object_detections artifact: {e}") from e
        if not isinstance(det_npz, dict) or det_npz.get("frame_indices") is None:
            raise RuntimeError("cut_detection | core_object_detections artifact missing/invalid (no-fallback)")
        if isinstance(det_npz, dict) and det_npz.get("frame_indices") is not None:
            d_fi = np.asarray(det_npz.get("frame_indices"), dtype=np.int32)
            boxes = np.asarray(det_npz.get("boxes")) if det_npz.get("boxes") is not None else None
            scores = np.asarray(det_npz.get("scores")) if det_npz.get("scores") is not None else None
            class_ids = np.asarray(det_npz.get("class_ids")) if det_npz.get("class_ids") is not None else None
            valid_mask = np.asarray(det_npz.get("valid_mask")) if det_npz.get("valid_mask") is not None else None
            class_names = det_npz.get("class_names")
            person_id = 0
            try:
                if isinstance(class_names, np.ndarray):
                    names = [str(x) for x in class_names.tolist()]
                    for s in names:
                        if ":person" in s:
                            person_id = int(s.split(":", 1)[0])
                            break
            except Exception:
                person_id = 0
            if boxes is not None and scores is not None and class_ids is not None and valid_mask is not None:
                for i, ufi in enumerate(d_fi.tolist()):
                    m = valid_mask[i].astype(bool) if valid_mask.ndim >= 2 else None
                    if m is None:
                        person_map[int(ufi)] = False
                        continue
                    ok = False
                    for j in range(class_ids.shape[1]):
                        if not bool(m[j]):
                            continue
                        if int(class_ids[i, j]) != int(person_id):
                            continue
                        if float(scores[i, j]) >= float(person_score_thresh):
                            ok = True
                            break
                    person_map[int(ufi)] = bool(ok)
            else:
                raise RuntimeError("cut_detection | core_object_detections missing required arrays (no-fallback)")

        def _landmark_xy(u_idx: int) -> Optional[np.ndarray]:
            k = face_map.get(int(u_idx))
            if k is None:
                return None
            # face_landmarks: (N, max_faces, max_landmarks, 3). Use first face.
            if face_landmarks.ndim != 4:
                return None
            if face_present.ndim < 2:
                return None
            if not bool(face_present[k, 0]):
                return None
            pts = face_landmarks[k, 0, :, :2].astype(np.float32)  # normalized x,y
            if pts.size == 0 or not np.isfinite(pts).any():
                return None
            return pts

        def _face_change(a: np.ndarray, b: np.ndarray) -> float:
            # a,b: (L,2) normalized. Center+scale then cosine distance.
            a0 = a - np.nanmean(a, axis=0, keepdims=True)
            b0 = b - np.nanmean(b, axis=0, keepdims=True)
            na = float(np.linalg.norm(a0) + 1e-9)
            nb = float(np.linalg.norm(b0) + 1e-9)
            a1 = (a0 / na).reshape(-1)
            b1 = (b0 / nb).reshape(-1)
            sim = float(np.dot(a1, b1))
            return float(max(0.0, 1.0 - sim))

        # Evaluate only at hard cuts
        jump_pos: List[int] = []
        jump_score: List[float] = []
        for p in hard_cut_positions:
            p_i = int(p)
            if p_i <= 0 or p_i >= len(frame_indices):
                continue
            u0 = int(frame_indices[p_i - 1])
            u1 = int(frame_indices[p_i])

            # person gate (needs both frames)
            if not (person_map.get(u0, False) and person_map.get(u1, False)):
                continue

            lm0 = _landmark_xy(u0)
            lm1 = _landmark_xy(u1)
            if lm0 is None or lm1 is None:
                continue
            fc = _face_change(lm0, lm1)
            if fc < float(face_change_thresh):
                continue

            # background similarity: blur face bbox (from landmarks) and compute SSIM
            f0 = frame_manager.get(u0)
            f1 = frame_manager.get(u1)
            h, w = f0.shape[:2]
            # bbox in pixels from normalized landmarks
            x0 = float(np.nanmin(lm0[:, 0]))
            y0 = float(np.nanmin(lm0[:, 1]))
            x1 = float(np.nanmax(lm0[:, 0]))
            y1 = float(np.nanmax(lm0[:, 1]))
            # expand
            ex = 0.15
            xa = int(max(0, (x0 - ex) * w))
            xb = int(min(w, (x1 + ex) * w))
            ya = int(max(0, (y0 - ex) * h))
            yb = int(min(h, (y1 + ex) * h))
            f0m = f0.copy()
            f1m = f1.copy()
            if xb > xa and yb > ya:
                f0m[ya:yb, xa:xb] = cv2.GaussianBlur(f0m[ya:yb, xa:xb], (15, 15), 5)
                f1m[ya:yb, xa:xb] = cv2.GaussianBlur(f1m[ya:yb, xa:xb], (15, 15), 5)
            # SSIM (reuse existing helper; it returns 1-SSIM drop)
            drop = frame_ssim(f0m, f1m)
            ssim_val = float(1.0 - drop)
            if ssim_val < float(bg_ssim_thresh):
                continue

            jump_pos.append(p_i)
            # score: face change weighted by bg similarity
            jump_score.append(float(fc * ssim_val))

        return jump_pos, jump_score