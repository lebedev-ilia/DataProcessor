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

from modules.base_module import BaseModule
from utils.frame_manager import FrameManager
from utils.logger import get_logger

NAME = "CutDetectionPipeline"
VERSION = "1.0"

logger = get_logger(NAME)


def float_or_zero(x):
    return float(x) if np.isfinite(x) else 0.0

def seconds_from_fps(n_frames, fps):
    return n_frames / float(fps) if fps > 0 else 0.0


def frame_histogram_diff(frameA, frameB, bins=32):
    """Compute histogram difference (normalized) between two RGB frames."""
    hsvA = cv2.cvtColor(frameA, cv2.COLOR_RGB2HSV)
    hsvB = cv2.cvtColor(frameB, cv2.COLOR_RGB2HSV)
    hA = cv2.calcHist([hsvA], [2], None, [bins], [0,256]).flatten()
    hB = cv2.calcHist([hsvB], [2], None, [bins], [0,256]).flatten()
    hA = hA / (hA.sum() + 1e-9)
    hB = hB / (hB.sum() + 1e-9)
    return float(np.linalg.norm(hA - hB, ord=1))  # L1

def frame_ssim(frameA, frameB):
    """Convert to gray and compute SSIM (0..1). Return 1-SSIM as drop measure."""
    grayA = cv2.cvtColor(frameA, cv2.COLOR_RGB2GRAY)
    grayB = cv2.cvtColor(frameB, cv2.COLOR_RGB2GRAY)
    try:
        s = ssim(grayA, grayB, data_range=grayB.max() - grayB.min())
        return float(1.0 - s)  # bigger = larger drop
    except Exception:
        return 0.0

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

# helper convert RGB ndarray -> PIL Image
def ImageFromRGB(frame_rgb):
    try:
        from PIL import Image
        return Image.fromarray(frame_rgb)
    except ImportError:
        return frame_rgb

def optical_flow_magnitude(prev_gray, gray):
    """Farneback optical flow average magnitude."""
    try:
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray,
                                            None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        return float(np.mean(mag)), mag, ang
    except Exception:
        return 0.0, None, None

def optical_flow_direction_consistency(flow_angles, window_size=5):
    """Compute direction consistency using circular statistics."""
    if flow_angles is None:
        return 0.0
    # Convert angles to unit vectors
    cos_angles = np.cos(flow_angles)
    sin_angles = np.sin(flow_angles)
    mean_cos = np.mean(cos_angles)
    mean_sin = np.mean(sin_angles)
    # Resultant length (0-1, higher = more consistent direction)
    consistency = np.sqrt(mean_cos**2 + mean_sin**2)
    return float(consistency)

def estimate_global_motion_homography(prev_gray, gray):
    """
    Estimate global camera motion using RANSAC homography.
    Returns homography matrix and inlier ratio.
    """
    try:
        # Detect features using ORB (lightweight)
        orb = cv2.ORB_create(nfeatures=500)
        kp1, des1 = orb.detectAndCompute(prev_gray, None)
        kp2, des2 = orb.detectAndCompute(gray, None)
        
        if des1 is None or des2 is None or len(des1) < 10 or len(des2) < 10:
            return None, 0.0
        
        # Match features
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)
        
        # Apply ratio test
        good_matches = []
        for match_pair in matches:
            if len(match_pair) == 2:
                m, n = match_pair
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        
        if len(good_matches) < 10:
            return None, 0.0
        
        # Extract matched points
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        
        # Estimate homography with RANSAC
        H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        inlier_ratio = float(np.sum(mask)) / len(good_matches) if mask is not None else 0.0
        
        return H, inlier_ratio
    except Exception:
        return None, 0.0

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
    use_deep_features=True, 
    use_adaptive_thresholds=True, 
    temporal_smoothing=True, 
    embed_model=None, 
    transform=None, 
    device='cpu'
    ):
    """
    Improved hard cut detection with adaptive thresholds, deep features, and temporal smoothing.
    frames: list of BGR frames
    returns list of cut_indices (frame index where cut occurs) and strengths
    Strategy: combine histogram diff, SSIM drop, optical flow jump, and deep embeddings.
    """
    n = len(frame_indices)
    if n < 2:
        return [], []
    
    # Compute all frame differences first
    hdiffs = []
    ssim_diffs = []
    flow_mags = []
    deep_diffs = []
    
    prev_gray = cv2.cvtColor(frame_manager.get(frame_indices[0]), cv2.COLOR_RGB2GRAY)
    for i in range(1, n):
        fA = frame_manager.get(frame_indices[i-1])
        fB = frame_manager.get(frame_indices[i])
        hdiff = frame_histogram_diff(fA, fB)
        s = frame_ssim(fA, fB)
        gray = cv2.cvtColor(fB, cv2.COLOR_RGB2GRAY)
        flow_mag, _, _ = optical_flow_magnitude(prev_gray, gray)
        prev_gray = gray
        
        hdiffs.append(hdiff)
        ssim_diffs.append(s)
        flow_mags.append(flow_mag)
        
        # Deep feature difference
        if use_deep_features and embed_model is not None:
            deep_diff = feature_embedding_diff(fA, fB, embed_model, transform, device)
            deep_diffs.append(deep_diff)
        else:
            deep_diffs.append(0.0)
    
    hdiffs = np.array(hdiffs)
    ssim_diffs = np.array(ssim_diffs)
    flow_mags = np.array(flow_mags)
    deep_diffs = np.array(deep_diffs)
    
    # Adaptive thresholds based on local statistics
    if use_adaptive_thresholds:
        window_size = min(30, n // 10)  # local window
        hist_thresh = np.median(hdiffs) + 2.0 * np.std(hdiffs) if hist_thresh is None else hist_thresh
        ssim_thresh = np.median(ssim_diffs) + 1.5 * np.std(ssim_diffs) if ssim_thresh is None else ssim_thresh
        flow_thresh = np.median(flow_mags) + 2.0 * np.std(flow_mags) if flow_thresh is None else flow_thresh
        deep_thresh = np.median(deep_diffs) + 1.5 * np.std(deep_diffs) if use_deep_features else 0.0
    else:
        hist_thresh = hist_thresh or 0.5
        ssim_thresh = ssim_thresh or 0.25
        flow_thresh = flow_thresh or 4.0
        deep_thresh = 0.3 if use_deep_features else 0.0
    
    # Compute scores
    scores = np.zeros(len(hdiffs))
    scores += (hdiffs > hist_thresh).astype(float)
    scores += (ssim_diffs > ssim_thresh).astype(float)
    scores += (flow_mags > flow_thresh).astype(float)
    if use_deep_features:
        scores += (deep_diffs > deep_thresh).astype(float)
    
    # Temporal smoothing to reduce false positives
    if temporal_smoothing and len(scores) > 3:
        # Apply median filter for robustness (removes spikes)
        scores_median = medfilt(scores.astype(float), kernel_size=3)
        # Apply Gaussian smoothing
        scores_smooth = gaussian_filter1d(scores_median, sigma=1.0)
        # Require local maximum
        cut_candidates = []
        for i in range(1, len(scores_smooth)-1):
            if scores_smooth[i] > scores_smooth[i-1] and scores_smooth[i] > scores_smooth[i+1]:
                if scores_smooth[i] >= 2.0:  # require at least two signals
                    cut_candidates.append((i+1, scores_smooth[i]))  # +1 because indices start from frame 1
    else:
        cut_candidates = [(i+1, s) for i, s in enumerate(scores) if s >= 2.0]
    
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
    
    return cut_idxs, strengths

def detect_soft_cuts(frame_manager, frame_indices, fps, fade_threshold=0.02, min_duration_frames=4, use_flow_consistency=True):
    """
    Improved soft cut detection with gradient-based analysis and optical flow consistency.
    Detect fade-in/out and dissolves by monitoring brightness/histogram changes over a window.
    Returns events: list of dicts {'type':'fade_in'/'fade_out'/'dissolve', 'start', 'end', 'duration_s'}
    """
    n = len(frame_indices)
    if n < 3:
        return []
    
    # Multi-channel gradient analysis (HSV + Lab)
    hsv_values = []
    lab_values = []
    hist_diffs = []
    flow_mags = []
    
    prev_gray = cv2.cvtColor(frame_manager.get(frame_indices[0]), cv2.COLOR_RGB2GRAY)
    prev_hsv_hist = None
    
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
            hist_diffs.append(hist_diff)
        else:
            hist_diffs.append(0.0)
        prev_hsv_hist = hsv_hist
        
        # Optical flow for consistency check
        if i > 0:
            gray = cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)
            flow_mag, _, _ = optical_flow_magnitude(prev_gray, gray)
            flow_mags.append(flow_mag)
            prev_gray = gray
        else:
            flow_mags.append(0.0)
    
    hsv_values = np.array(hsv_values)
    lab_values = np.array(lab_values)
    hist_diffs = np.array(hist_diffs)
    flow_mags = np.array(flow_mags)
    
    events = []
    
    # Fade detection: gradient-based with cumulative distribution
    hsv_deriv = np.diff(hsv_values)
    lab_deriv = np.diff(lab_values)
    hist_deriv = np.diff(hist_diffs)
    
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
                hsv_change = abs(hsv_values[j] - hsv_values[i]) if j < len(hsv_values) else 0
                lab_change = abs(lab_values[j] - lab_values[i]) if j < len(lab_values) else 0
                if hsv_change > fade_threshold or lab_change > fade_threshold:
                    typ = 'fade_in' if (hsv_values[j] > hsv_values[i] if j < len(hsv_values) else False) else 'fade_out'
                    events.append({'type': typ, 'start': i, 'end': min(j, len(frame_indices)-1), 
                                 'duration_s': seconds_from_fps(duration, fps)})
            i = j
        else:
            i += 1
    
    # Dissolve detection: moderate histogram drift with low flow consistency
    # Improved: check for linear mixing correlation and exclude exposure changes
    if use_flow_consistency:
        window_size = min(10, n // 5)
        for i in range(window_size, n - window_size):
            # Check for gradual histogram change
            hist_window = hist_diffs[i-window_size//2:i+window_size//2]
            flow_window = flow_mags[i-window_size//2:i+window_size//2]
            
            # Gradual histogram change (low variance in changes)
            hist_var = np.var(hist_window)
            hist_mean = np.mean(hist_window)
            flow_mean = np.mean(flow_window)
            
            # Check for exposure changes (global brightness shift across entire frame)
            # Exposure changes affect entire frame uniformly, dissolves affect content distribution
            hsv_window = hsv_values[i-window_size//2:i+window_size//2]
            lab_window = lab_values[i-window_size//2:i+window_size//2]
            hsv_gradient = np.abs(np.diff(hsv_window))
            lab_gradient = np.abs(np.diff(lab_window))
            # Low gradient variance = uniform exposure change (not dissolve)
            is_exposure_change = (np.var(hsv_gradient) < 0.0001 and np.var(lab_gradient) < 0.0001 and 
                                 np.mean(np.abs(hsv_gradient)) > 0.01)  # Uniform but significant change
            
            # Dissolve: gradual histogram change + low motion + not exposure change
            # Also check for linear correlation in histogram changes (dissolve = linear mixing)
            hist_window_full = hist_diffs[max(0, i-window_size):min(n, i+window_size)]
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
                    events.append({'type': 'dissolve', 'start': i - window_size//2, 
                                 'end': i + window_size//2, 
                                 'duration_s': seconds_from_fps(window_size, fps)})
    
    return events

def detect_motion_based_cuts(
    frame_manager,
    frame_indices, 
    flow_spike_factor=None, 
    use_direction_analysis=True, 
    adaptive_threshold=True, 
    detect_speed_ramps=True,
    use_camera_motion_compensation=True
):
    """
    Improved motion-based cut detection with direction analysis and adaptive thresholds.
    Detect whip pans / zoom transitions / speed ramp cuts by measuring spikes in optical-flow magnitude variance.
    Returns list of indices, intensities, and types ('whip_pan', 'zoom', or 'speed_ramp').
    """
    n = len(frame_indices)
    if n < 2:
        return [], [], []
    
    mags = []
    angles_list = []
    direction_consistencies = []
    mag_variances = []  # For speed ramp detection
    
    prev_gray = cv2.cvtColor(frame_manager.get(frame_indices[0]), cv2.COLOR_RGB2GRAY)
    prev_mag_map = None
    
    # Resize frames for faster flow computation if large
    sample_frame = frame_manager.get(frame_indices[0])
    h, w = sample_frame.shape[:2]
    use_low_res = (h * w > 640 * 480)  # Use low-res for large frames
    target_size = (256, 256) if use_low_res else None
    
    for i in range(1, n):
        gray = cv2.cvtColor(frame_manager.get(frame_indices[i]), cv2.COLOR_RGB2GRAY)
        
        # Camera motion compensation: estimate global motion
        is_camera_motion = False
        if use_camera_motion_compensation:
            H, inlier_ratio = estimate_global_motion_homography(prev_gray, gray)
            # If high inlier ratio, motion is mostly global (camera motion)
            # Subtract global motion contribution for object motion detection
            is_camera_motion = inlier_ratio > 0.7 if H is not None else False
        
        # Compute flow (on low-res if needed)
        if use_low_res:
            prev_gray_small = cv2.resize(prev_gray, target_size)
            gray_small = cv2.resize(gray, target_size)
            mag, mag_map, angles = optical_flow_magnitude(prev_gray_small, gray_small)
            # Scale magnitude back to original frame size
            mag = mag * (h * w) / (target_size[0] * target_size[1])
        else:
            mag, mag_map, angles = optical_flow_magnitude(prev_gray, gray)
        
        # Store camera motion flag (will use later for classification)
        mags.append((mag, is_camera_motion))
        angles_list.append(angles)
        
        # Direction consistency
        if angles is not None and use_direction_analysis:
            consistency = optical_flow_direction_consistency(angles)
            direction_consistencies.append(consistency)
        else:
            direction_consistencies.append(0.0)
        
        # Speed ramp detection: analyze variance in flow magnitude across frame
        # Speed ramps show high variance in magnitude (fast motion in center, slow at edges)
        if detect_speed_ramps and mag_map is not None:
            mag_variance = float(np.var(mag_map))
            mag_variances.append(mag_variance)
        else:
            mag_variances.append(0.0)
        
        prev_gray = gray
        prev_mag_map = mag_map
    
    # Extract magnitudes and camera motion flags
    mags_array = np.array([m[0] for m in mags])
    camera_motion_flags = [m[1] for m in mags]
    direction_consistencies = np.array(direction_consistencies)
    mag_variances = np.array(mag_variances)
    
    # Adaptive threshold using percentiles or z-score
    if adaptive_threshold:
        if flow_spike_factor is None:
            # Use 95th percentile as threshold
            threshold = np.percentile(mags_array, 95)
        else:
            median = np.median(mags_array)
            std = np.std(mags_array) + 1e-9
            threshold = median + flow_spike_factor * std
    else:
        median = np.median(mags_array)
        std = np.std(mags_array) + 1e-9
        threshold = median + (flow_spike_factor or 3.0) * std
    
    # Speed ramp threshold: high magnitude + high variance
    speed_ramp_threshold = np.percentile(mag_variances, 90) if len(mag_variances) > 0 else 0.0
    
    # Find spikes (exclude pure camera motion if compensation enabled)
    spike_mask = mags_array > threshold
    if use_camera_motion_compensation:
        # Filter out spikes that are pure camera motion (already handled by hard cuts)
        spike_mask = spike_mask & ~np.array(camera_motion_flags)
    
    spike_idxs = np.where(spike_mask)[0] + 1  # +1 because indices start from frame 1
    intensities = mags_array[spike_idxs-1].tolist()
    
    # Classify as whip pan vs zoom vs speed ramp using direction consistency and variance
    types = []
    if use_direction_analysis:
        for idx in spike_idxs:
            if idx-1 < len(direction_consistencies):
                consistency = direction_consistencies[idx-1]
                mag_var = mag_variances[idx-1] if idx-1 < len(mag_variances) else 0.0
                is_cam_motion = camera_motion_flags[idx-1] if idx-1 < len(camera_motion_flags) else False
                
                # Speed ramp: high magnitude + high variance (fast motion gradient)
                if detect_speed_ramps and mag_var > speed_ramp_threshold:
                    types.append('speed_ramp')
                # High consistency + high magnitude + camera motion = whip pan
                elif consistency > 0.6 or is_cam_motion:
                    types.append('whip_pan')
                # Low consistency + high magnitude = zoom (object motion)
                else:
                    types.append('zoom')
            else:
                types.append('unknown')
    else:
        types = ['motion_cut'] * len(spike_idxs)
    
    return spike_idxs.tolist(), intensities, types

# Stylized transitions classifier (zero-shot with CLIP if available)
class StylizedTransitionZeroShot:
    def __init__(self, device='cpu', use_temporal_aggregation=True, use_multimodal=True):
        self.device = device
        self.use_temporal_aggregation = use_temporal_aggregation
        self.use_multimodal = use_multimodal
        self.labels = [
            "hard cut",
            "fade",
            "dissolve",
            "whip pan",
            "zoom transition",
            "wipe transition",
            "slide transition",
            "glitch transition",
            "flash transition",
            "luma wipe transition"
        ]
        if clip is None or torch is None:
            raise RuntimeError("cut_detection | CLIP (python package `clip`) + torch are required when use_clip=true")
        self.model, self.preprocess = clip.load("ViT-B/32", device=device)
        self.text_tokens = clip.tokenize(self.labels).to(device)
        # Feature cache for efficiency
        self.feature_cache = {}
    
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

    def predict_transition(self, frames_window, cache_key=None):
        """
        Improved transition prediction with temporal aggregation and multi-modal input.
        frames_window: small sequence of consecutive frames (RGB)
        Return probs per label (zero-shot)
        """
        # Use cache if available
        if cache_key and cache_key in self.feature_cache:
            img_feat = self.feature_cache[cache_key]
        else:
            # Create input: multi-modal or single frame
            if self.use_multimodal and len(frames_window) >= 2:
                input_img = self._create_multimodal_input(frames_window)
                if input_img is None:
                    input_img = frames_window[len(frames_window)//2]
            else:
                mid = frames_window[len(frames_window)//2]
                input_img = mid
            
            pil = ImageFromRGB(input_img) if isinstance(input_img, np.ndarray) else input_img
            image_input = self.preprocess(pil).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                img_feat = self.model.encode_image(image_input)
                img_feat = img_feat / (img_feat.norm(dim=-1, keepdim=True)+1e-9)
            
            if cache_key:
                self.feature_cache[cache_key] = img_feat
        
        # Compute similarity with text features
        with torch.no_grad():
            txt_feat = self.model.encode_text(self.text_tokens)
            txt_feat = txt_feat / (txt_feat.norm(dim=-1, keepdim=True)+1e-9)
            logits = (img_feat @ txt_feat.T).softmax(dim=-1).cpu().numpy()[0]
        
        return {self.labels[i]: float(logits[i]) for i in range(len(self.labels))}
    
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
        
        # Модели будут инициализированы в _do_initialize()
        self.embed_model = None
        self.transform = None
        self.clip_detector = None
        self._clip_zero_shot = clip_zero_shot
    
    def _do_initialize(self) -> None:
        """Инициализация моделей."""
        # Initialize embedding model for deep features
        if self.use_deep_features:
            self.embed_model, self.transform = get_embedding_model(device=self.device, model_name='resnet18')
            self.logger.info("Deep features model initialized")
        
        # Initialize CLIP detector
        if self._clip_zero_shot:
            self.clip_detector = StylizedTransitionZeroShot(
                device=self.device,
                use_temporal_aggregation=True,
                use_multimodal=True
            )
            self.logger.info("CLIP detector initialized")

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

        # Извлекаем audio_path из config если есть
        audio_path = None
        if config and isinstance(config, dict):
            audio_path = config.get('audio_path')
        
        return self._process_video_frames(frame_manager, frame_indices, audio_path=audio_path)
    
    def _process_video_frames(self, frame_manager, frame_indices, audio_path=None):
        """
        Improved pipeline with all enhanced features.
        frames_bgr: list of BGR frames (np.uint8)
        audio_path: optional path to audio file for audio-assisted detection
        Returns dict of features and detections
        """
        n = len(frame_indices)
        duration_s = seconds_from_fps(n, self.fps)
        
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

        hard_idxs, hard_strengths = detect_hard_cuts(
            frame_manager=frame_manager,
            frame_indices=frame_indices,
            use_deep_features=self.use_deep_features,
            use_adaptive_thresholds=self.use_adaptive_thresholds,
            temporal_smoothing=True,
            embed_model=self.embed_model,
            transform=self.transform,
            device=self.device
        )

        tok = round(time.time() - tik, 2)
        logger.info(f"Hard cuts success | Time: {tok}")
        tik = time.time()

        soft_events = detect_soft_cuts(
            frame_manager=frame_manager,
            frame_indices=frame_indices,
            fps=self.fps,
            fade_threshold=self.fade_threshold,
            min_duration_frames=self.min_duration_frames,
            use_flow_consistency=self.use_flow_consistency,
        )

        tok = round(time.time() - tik, 2)
        logger.info(f"Soft cuts success | Time: {tok}")
        tik = time.time()

        motion_idxs, motion_int, motion_types = detect_motion_based_cuts(
            frame_manager=frame_manager,
            frame_indices=frame_indices,
            use_direction_analysis=True,
            adaptive_threshold=True,
            detect_speed_ramps=True,
            use_camera_motion_compensation=True
        )

        tok = round(time.time() - tik, 2)
        logger.info(f"Motion-based cuts success | Time: {tok}")
        tik = time.time()

        stylized_counts = {}
        stylized_probs_per_cut = []
        if self.clip_detector is not None:
            candidate_windows = []
            candidate_scores = []
            
            prev_gray = cv2.cvtColor(frame_manager.get(frame_indices[0]), cv2.COLOR_RGB2GRAY)
            for idx in range(1, min(n-1, len(frame_indices)-1)):
                fA = frame_manager.get(frame_indices[idx-1])
                fB = frame_manager.get(frame_indices[idx])
                
                hdiff = frame_histogram_diff(fA, fB)
                ssim_drop = frame_ssim(fA, fB)
                gray = cv2.cvtColor(fB, cv2.COLOR_RGB2GRAY)
                flow_mag, _, _ = optical_flow_magnitude(prev_gray, gray)
                prev_gray = gray
                
                candidate_score = hdiff + ssim_drop * 2.0 + min(flow_mag / 10.0, 1.0)
                
                if candidate_score > 0.3:  # Adaptive threshold could be percentile-based
                    candidate_windows.append(idx)
                    candidate_scores.append(candidate_score)
            
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
        jump_idxs, jump_scores = detect_jump_cuts(
            frame_manager=frame_manager,
            frame_indices=frame_indices,
            use_background_embedding=True,
            embed_model=self.embed_model,
            transform=self.transform,
            device=self.device,
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
        cut_timing_stats_dict = cut_timing_statistics(hard_idxs, self.fps, duration_s)
        shot_stats = shot_length_stats(shot_lengths, self.fps)

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
        features['jump_cut_ratio_per_minute'] = float(len(jump_idxs) / (duration_s/60.0+1e-9))

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
            stylized_counts, len(hard_idxs), duration_s
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

        # Provide frame_indices for consumers (union-domain indices).
        return {'frame_indices': np.asarray(frame_indices, dtype=np.int32), 'features': features, 'detections': detections}