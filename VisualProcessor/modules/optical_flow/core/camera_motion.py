"""
core/camera_motion.py
Functions to extract camera motion features from RAFT optical flow .pt outputs.

Assumptions about flow .pt:
- Each .pt file contains a NumPy array or Torch tensor with shape either:
    (H, W, 2)  or (2, H, W) or (1, 2, H, W) etc.
- Values are per-pixel displacement in pixels (dx, dy) between two frames.

Outputs:
- per-frame features (dict)
- aggregated per-video features (dict)
"""

from typing import Optional, Tuple, List, Dict, Sequence
import numpy as np
import os
import torch
import math
from glob import glob


# --- helpers for loading ---
def load_flow_tensor(path: str) -> np.ndarray:
    """
    Load .pt or .pth produced by RAFT pipeline.
    Return flow as H x W x 2 numpy array (dx, dy).
    """
    obj = torch.load(path, map_location="cpu")
    # support raw tensor or dict with 'flow' key
    if isinstance(obj, dict):
        # common saving styles: {'flow': tensor} or similar
        possible = ['flow', 'flows', 'disp', 'flow_tensor']
        for k in possible:
            if k in obj:
                t = obj[k]
                break
        else:
            # try first tensor-like value
            for v in obj.values():
                if isinstance(v, (torch.Tensor, np.ndarray)):
                    t = v
                    break
            else:
                raise ValueError(f"Couldn't find tensor in {path}")
    else:
        t = obj

    if isinstance(t, np.ndarray):
        arr = t
    else:
        arr = t.cpu().numpy()

    # normalize shapes:
    if arr.ndim == 4:
        # e.g. 1 x 2 x H x W
        arr = arr.squeeze(0)
    if arr.ndim == 3 and arr.shape[0] == 2:
        # 2 x H x W -> H x W x 2
        arr = np.transpose(arr, (1, 2, 0))
    if arr.ndim == 3 and arr.shape[2] == 2:
        return arr.astype(np.float32)
    raise ValueError(f"Unsupported flow array shape {arr.shape} for {path}")


# --- affine estimation ---
def estimate_affine_from_flow(flow: np.ndarray,
                              mask: Optional[np.ndarray] = None,
                              sample_n: int = 2000,
                              ransac_thresh: float = 3.0) -> Optional[np.ndarray]:
    """
    Estimate 2x3 affine (partial) from flow background vectors.
    - flow: H x W x 2 (dx,dy)
    - mask: boolean mask of background pixels (True=use)
    Returns: 2x3 affine matrix or None
    """
    import cv2

    h, w, _ = flow.shape
    ys, xs = np.mgrid[0:h, 0:w]
    pts = np.stack([xs.ravel().astype(np.float32), ys.ravel().astype(np.float32)], axis=-1)
    disp = flow.reshape(-1, 2).astype(np.float32)
    if mask is not None:
        m = mask.ravel().astype(bool)
        if m.sum() < 10:
            return None
        pts = pts[m]
        disp = disp[m]
    if pts.shape[0] > sample_n:
        idx = np.random.choice(pts.shape[0], sample_n, replace=False)
        pts = pts[idx]
        disp = disp[idx]
    src = pts
    dst = pts + disp
    M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh)
    return M  # None or 2x3


def decompose_affine(M: np.ndarray) -> Dict[str, float]:
    """
    From 2x3 affine, return {scale, rotation, tx, ty}
    rotation in radians (approx)
    """
    if M is None:
        return dict(scale=np.nan, rotation=np.nan, tx=np.nan, ty=np.nan)
    a, b, tx = M[0]
    c, d, ty = M[1]
    scale = math.sqrt(a * a + b * b)
    # rotation approximation: angle of the first column
    rotation = math.atan2(c, a)

    return dict(scale=scale, rotation=rotation, tx=float(tx), ty=float(ty))


# --- frame-level metrics ---
def flow_magnitude_angle(flow: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    dx = flow[..., 0]
    dy = flow[..., 1]
    mag = np.sqrt(dx * dx + dy * dy)
    ang = np.arctan2(dy, dx)  # radians (-pi, pi)
    return mag, ang


def background_mask_by_magnitude(flow: np.ndarray, mag_thresh: float = 0.5) -> np.ndarray:
    """
    Simple background mask: pixels with magnitude <= mag_thresh considered background.
    mag_thresh is in pixels (depends on your flow scale).
    """
    mag, _ = flow_magnitude_angle(flow)
    return mag <= mag_thresh


def compute_shakiness(flow: np.ndarray, background_mask: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    shakiness: variance / mean / max of background flow magnitude
    """
    mag, _ = flow_magnitude_angle(flow)
    if background_mask is not None:
        arr = mag[background_mask]
    else:
        arr = mag.ravel()
    if arr.size == 0:
        return dict(shake_var=0.0, shake_mean=0.0, shake_max=0.0)
    return dict(shake_var=float(np.var(arr)), shake_mean=float(np.mean(arr)), shake_max=float(np.max(arr)))


def detect_zoom_from_affines(prev_affine: Optional[np.ndarray], cur_affine: Optional[np.ndarray], eps: float = 1e-6) -> float:
    """
    Return scale_delta = cur_scale - prev_scale
    """
    if prev_affine is None or cur_affine is None:
        return 0.0
    prev = decompose_affine(prev_affine)
    cur = decompose_affine(cur_affine)
    if math.isnan(prev['scale']) or math.isnan(cur['scale']):
        return 0.0
    return float(cur['scale'] - prev['scale'])

def _safe_float(x):
    """Convert to safe float, replacing NaN/inf with 0.0."""
    try:
        xf = float(x)
        if math.isnan(xf) or math.isinf(xf):
            return 0.0
        return xf
    except Exception:
        return 0.0

def _safe_dict(d: Dict[str, float]) -> Dict[str, float]:
    """Convert all values in dict to safe floats."""
    return {k: _safe_float(v) for k, v in d.items()}

def compute_frame_motion_features(flow: np.ndarray,
                                  flow_prev: Optional[np.ndarray] = None,
                                  mag_bg_thresh: float = 0.5) -> Dict[str, float]:
    """
    Compute a dictionary of camera-related features for a single frame.
    All outputs are guaranteed to be numeric and NaN-free.
    """

    # ---------- BASICS ----------
    mag, ang = flow_magnitude_angle(flow)
    flat = mag.ravel()

    motion_mean = _safe_float(np.mean(flat)) if flat.size else 0.0
    motion_std = _safe_float(np.std(flat)) if flat.size else 0.0
    motion_max = _safe_float(np.max(flat)) if flat.size else 0.0
    motion_energy = _safe_float(np.sum(flat ** 2))

    # ---------- ENTROPY ----------
    try:
        hist, _ = np.histogram(ang.ravel(), bins=36, range=(-math.pi, math.pi))
        p = hist / (hist.sum() + 1e-9)
        motion_entropy = _safe_float(-np.sum([float(x) * math.log(float(x) + 1e-12) for x in p if x > 0]))
    except Exception:
        motion_entropy = 0.0

    # ---------- BACKGROUND ----------
    try:
        bg_mask = background_mask_by_magnitude(flow, mag_bg_thresh)
        background_ratio = _safe_float(np.mean(bg_mask))
    except Exception:
        bg_mask = None
        background_ratio = 0.0

    # ---------- SHAKINESS ----------
    try:
        shakiness = compute_shakiness(flow, background_mask=bg_mask)
        shakiness = _safe_dict(shakiness)
    except Exception:
        shakiness = dict(shake_var=0.0, shake_mean=0.0, shake_max=0.0)

    # ---------- AFFINE ----------
    try:
        M = estimate_affine_from_flow(flow, mask=bg_mask)
        affine = _safe_dict(decompose_affine(M))
    except Exception:
        affine = dict(scale=1.0, rotation=0.0, tx=0.0, ty=0.0)

    # ---------- ROTATION SPEED ----------
    rotation_speed = 0.0
    if flow_prev is not None:
        try:
            Mprev = estimate_affine_from_flow(
                flow_prev,
                mask=background_mask_by_magnitude(flow_prev, mag_bg_thresh)
            )
            prev_affine = _safe_dict(decompose_affine(Mprev))
            rotation_speed = _safe_float(affine["rotation"] - prev_affine["rotation"])
        except Exception:
            rotation_speed = 0.0

    # ---------- OUTPUT ----------
    return dict(
        motion_mean=motion_mean,
        motion_std=motion_std,
        motion_max=motion_max,
        motion_energy=motion_energy,
        motion_entropy=motion_entropy,

        shake_var=shakiness['shake_var'],
        shake_mean=shakiness['shake_mean'],
        shake_max=shakiness['shake_max'],

        affine_scale=affine['scale'],
        affine_rotation=affine['rotation'],
        affine_tx=affine['tx'],
        affine_ty=affine['ty'],

        background_ratio=background_ratio,
        rotation_speed=rotation_speed
    )



# --- aggregation over video ---
def aggregate_video_camera_features(flow_paths: Sequence[str], config: Optional[dict] = None) -> Dict[str, float]:
    """
    Given ordered list of flow file paths (frame t -> t+skip), compute aggregated camera motion features.
    Returns single dict of aggregated features (means, stds, counts).
    """

    if config is None:
        config = {}
    mag_bg_thresh = config.get("mag_bg_thresh", 0.5)
    zoom_eps = config.get("zoom_eps", 1e-3)
    sharp_angle_thresh = config.get("sharp_angle_thresh_deg", 15.0)  # reserved for future use

    per_frame: List[Dict[str, float]] = []
    affines: List[np.ndarray] = []
    flows: List[np.ndarray] = []

    # load sequentially
    for p in sorted(flow_paths):
        try:
            f = load_flow_tensor(p)
        except Exception as e:
            print(f"[camera_motion] failed to load {p}: {e}")
            continue
        flows.append(f)

    n = len(flows)
    if n == 0:
        return {}

    # compute per-frame features and collect affines
    prev_flow = None
    prev_affine = None
    zoom_ins = 0
    zoom_outs = 0
    zoom_deltas = []
    rotation_speeds = []
    shake_vars = []
    # for camera movement type heuristics
    pan_cnt = tilt_cnt = roll_cnt = dolly_cnt = truck_cnt = pedestal_cnt = static_cnt = 0

    for f in flows:
        feats = compute_frame_motion_features(f, flow_prev=prev_flow, mag_bg_thresh=mag_bg_thresh)
        per_frame.append(feats)
        shake_vars.append(feats['shake_var'])
        rotation_speeds.append(feats.get('rotation_speed', 0.0))
        # affine
        bg_mask = background_mask_by_magnitude(f, mag_bg_thresh)
        M = estimate_affine_from_flow(f, mask=bg_mask)
        affines.append(M)
        # detect zoom:
        if prev_affine is not None:
            dz = detect_zoom_from_affines(prev_affine, M)
            zoom_deltas.append(dz)
            if dz > zoom_eps:
                zoom_ins += 1
            elif dz < -zoom_eps:
                zoom_outs += 1
        prev_affine = M
        # movement type heuristic: use affine rotation & tx/ty magnitude
        dec = decompose_affine(M)
        rot = abs(dec['rotation']) if not math.isnan(dec['rotation']) else 0.0
        tnorm = math.hypot(dec['tx'], dec['ty'])
        if rot > math.radians(0.05):
            pan_cnt += 1
        elif tnorm > 0.5:
            if abs(dec['tx']) > abs(dec['ty']):
                truck_cnt += 1
            else:
                pedestal_cnt += 1
        else:
            static_cnt += 1
        prev_flow = f

    # aggregate
    motion_means = [x['motion_mean'] for x in per_frame]
    motion_stds = [x['motion_std'] for x in per_frame]
    rotation_speeds_arr = np.array(rotation_speeds)
    zoom_deltas_arr = np.array(zoom_deltas) if zoom_deltas else np.array([0.0])

    def safe_stats(arr):
        arr = np.array(arr)
        return dict(mean=float(np.nanmean(arr)), std=float(np.nanstd(arr)), max=float(np.nanmax(arr)), min=float(np.nanmin(arr)))

    res: Dict[str, float] = {}
    res.update({f"motion_mean_{k}": v for k, v in safe_stats(motion_means).items()})
    res.update({f"motion_std_{k}": v for k, v in safe_stats(motion_stds).items()})
    res["motion_energy_sum"] = float(sum([x["motion_energy"] for x in per_frame]))
    res["motion_entropy_mean"] = float(np.nanmean([x["motion_entropy"] for x in per_frame]))
    # shake
    res["shake_mean"] = float(np.nanmean(shake_vars))
    res["shake_std"] = float(np.nanstd(shake_vars))
    res["shake_max"] = float(np.nanmax(shake_vars))
    res["zoom_in_count"] = int(zoom_ins)
    res["zoom_out_count"] = int(zoom_outs)
    res["zoom_speed_mean"] = float(np.mean(np.abs(zoom_deltas_arr))) if zoom_deltas_arr.size else 0.0
    res["rotation_speed_mean"] = float(np.nanmean(rotation_speeds_arr))
    res["rotation_speed_std"] = float(np.nanstd(rotation_speeds_arr))
    # movement type ratios
    total_moves = pan_cnt + tilt_cnt + roll_cnt + dolly_cnt + truck_cnt + pedestal_cnt + static_cnt
    total_moves = total_moves or 1
    res["pan_ratio"] = float(pan_cnt / total_moves)
    res["truck_ratio"] = float(truck_cnt / total_moves)
    res["pedestal_ratio"] = float(pedestal_cnt / total_moves)
    res["static_ratio"] = float(static_cnt / total_moves)
    # chaos index: entropy of direction histogram summed over frames
    all_dirs = np.hstack([np.histogram(np.arctan2(f[..., 1], f[..., 0]).ravel(), bins=36, range=(-math.pi, math.pi))[0] for f in flows])
    p = all_dirs / (all_dirs.sum() + 1e-9)
    chaos = -np.sum([x * math.log(x + 1e-12) for x in p if x > 0])
    res["chaos_index"] = float(chaos)
    # simple style heuristics: map shake & zoom to handheld/drone/cinematic (heuristic stub)
    res["style_handheld"] = float(min(1.0, res["shake_mean"] * 2.0))
    res["style_tripod"] = float(max(0.0, 1.0 - res["shake_mean"] * 2.0))
    res["style_cinematic"] = float(max(0.0, 1.0 - res["shake_mean"]))
    res["style_drone"] = float(min(1.0, res["chaos_index"] / 10.0))
    res["style_action_cam"] = float(min(1.0, (res["shake_mean"] + res["motion_energy_sum"] / 1e4)))
    # counts
    res["n_frames"] = int(len(per_frame))
    return res


# small CLI helper (optional)
if __name__ == "__main__":
    import argparse
    import json

    p = argparse.ArgumentParser()
    p.add_argument("flow_dir")
    p.add_argument("--out", default="camera_features.json")
    args = p.parse_args()
    files = sorted(glob(os.path.join(args.flow_dir, "flow_*.pt")))
    features = aggregate_video_camera_features(files)
    with open(args.out, "w", encoding="utf8") as f:
        json.dump(features, f, indent=2, ensure_ascii=False)
    print("Wrote:", args.out)

