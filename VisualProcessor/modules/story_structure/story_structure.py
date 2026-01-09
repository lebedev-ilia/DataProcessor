"""
story_structure (Tier‑0 baseline)

Baseline mode:
- Computes story/energy/coherence proxies from sampled frames.
- Hard dependencies: core_clip, core_optical_flow, core_face_landmarks.
- Time axis is strictly `union_timestamps_sec` (no fallback).
- Per-second normalization for change signals (robust to sampling density).

Legacy / non-baseline experiments are moved to `legacy_story_structure.py` and MUST use ModelManager.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

from modules.base_module import BaseModule
from utils.frame_manager import FrameManager

MODULE_NAME = "story_structure"
VERSION = "2.0"
SCHEMA_VERSION = "story_structure_npz_v2"


def _unbox_object_scalar(x: Any) -> Any:
    if isinstance(x, np.ndarray) and x.dtype == object and x.shape == ():
        try:
            return x.item()
        except Exception:
            return x
    return x


def _require_union_times_s(frame_manager: FrameManager, frame_indices: np.ndarray) -> np.ndarray:
    """
    Segmenter contract: union_timestamps_sec is source-of-truth for time axis.
    No-fallback: if missing/invalid -> error.
    """
    meta = getattr(frame_manager, "meta", None)
    if not isinstance(meta, dict):
        raise RuntimeError("story_structure | FrameManager.meta missing (requires union_timestamps_sec)")
    ts = meta.get("union_timestamps_sec")
    if not isinstance(ts, list) or not ts:
        raise RuntimeError("story_structure | union_timestamps_sec missing/empty in frames metadata (no-fallback)")
    uts = np.asarray(ts, dtype=np.float32)

    if frame_indices.size == 0:
        raise RuntimeError("story_structure | frame_indices is empty (no-fallback)")
    if int(np.max(frame_indices)) >= int(uts.shape[0]):
        raise RuntimeError("story_structure | union_timestamps_sec does not cover frame_indices (no-fallback)")
    times_s = uts[frame_indices.astype(np.int32)]
    if times_s.size >= 2 and np.any(np.diff(times_s) < -1e-3):
        raise RuntimeError("story_structure | union_timestamps_sec is not monotonic for frame_indices (no-fallback)")
    return times_s.astype(np.float32)


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-9
    return x / norms


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if x.size == 0:
        return x
    mu = float(np.mean(x))
    sd = float(np.std(x)) + 1e-6
    return ((x - mu) / sd).astype(np.float32)


def _load_npz_meta_models_used(npz: np.lib.npyio.NpzFile) -> List[Dict[str, Any]]:
    meta = _unbox_object_scalar(npz.get("meta"))
    if isinstance(meta, dict):
        mu = meta.get("models_used")
        if isinstance(mu, list):
            return [x for x in mu if isinstance(x, dict)]
    return []


def _align_by_frame_indices(core_idx: np.ndarray, want_idx: np.ndarray, *, who: str) -> np.ndarray:
    mapping = {int(fi): i for i, fi in enumerate(core_idx.tolist())}
    pos = [mapping.get(int(fi), -1) for fi in want_idx.tolist()]
    if any(p < 0 for p in pos):
        raise RuntimeError(
            f"{MODULE_NAME} | {who}.frame_indices does not cover requested frame_indices. "
            "Segmenter must produce consistent sampling for dependent components."
        )
    return np.asarray(pos, dtype=np.int64)


def _load_core_clip_embeddings_aligned(rs_path: str, fi: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    p = os.path.join(rs_path, "core_clip", "embeddings.npz")
    if not os.path.isfile(p):
        raise FileNotFoundError(f"{MODULE_NAME} | missing core_clip embeddings: {p}")
    npz = np.load(p, allow_pickle=True)
    core_idx = npz.get("frame_indices")
    core_emb = npz.get("frame_embeddings")
    if core_idx is None or core_emb is None:
        raise RuntimeError(f"{MODULE_NAME} | core_clip embeddings.npz missing keys frame_indices/frame_embeddings")
    core_idx = np.asarray(core_idx, dtype=np.int32)
    core_emb = np.asarray(core_emb, dtype=np.float32)
    pos = _align_by_frame_indices(core_idx, fi, who="core_clip")
    return core_emb[pos], _load_npz_meta_models_used(npz)


def _load_core_optical_flow_aligned(rs_path: str, fi: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    p = os.path.join(rs_path, "core_optical_flow", "flow.npz")
    if not os.path.isfile(p):
        raise FileNotFoundError(f"{MODULE_NAME} | missing core_optical_flow flow: {p}")
    npz = np.load(p, allow_pickle=True)
    core_idx = npz.get("frame_indices")
    curve = npz.get("motion_norm_per_sec_mean")
    if core_idx is None or curve is None:
        raise RuntimeError(f"{MODULE_NAME} | core_optical_flow flow.npz missing keys frame_indices/motion_norm_per_sec_mean")
    core_idx = np.asarray(core_idx, dtype=np.int32)
    curve = np.asarray(curve, dtype=np.float32)
    pos = _align_by_frame_indices(core_idx, fi, who="core_optical_flow")
    return curve[pos], _load_npz_meta_models_used(npz)


def _load_core_face_any_present_aligned(rs_path: str, fi: np.ndarray) -> Tuple[np.ndarray, List[Dict[str, Any]], Dict[str, Any]]:
    """
    Returns:
      - any_face_present (N,) bool
      - models_used
      - provider_meta (best-effort, unboxed)
    """
    p = os.path.join(rs_path, "core_face_landmarks", "landmarks.npz")
    if not os.path.isfile(p):
        raise FileNotFoundError(f"{MODULE_NAME} | missing core_face_landmarks landmarks: {p}")
    npz = np.load(p, allow_pickle=True)
    core_idx = npz.get("frame_indices")
    face_present = npz.get("face_present")
    if core_idx is None or face_present is None:
        raise RuntimeError(f"{MODULE_NAME} | core_face_landmarks landmarks.npz missing keys frame_indices/face_present")
    core_idx = np.asarray(core_idx, dtype=np.int32)
    face_present = np.asarray(face_present, dtype=bool)
    if face_present.ndim == 1:
        any_face = face_present
    else:
        any_face = np.any(face_present, axis=1)
    pos = _align_by_frame_indices(core_idx, fi, who="core_face_landmarks")
    meta = _unbox_object_scalar(npz.get("meta"))
    return np.asarray(any_face[pos], dtype=bool), _load_npz_meta_models_used(npz), meta if isinstance(meta, dict) else {}


def _downsample_to_fixed(x: np.ndarray, m: int) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    if m <= 0:
        return np.asarray([], dtype=np.float32)
    if x.size == 0:
        return np.zeros((m,), dtype=np.float32)
    if x.size == 1:
        return np.full((m,), float(x[0]), dtype=np.float32)
    xp = np.linspace(0.0, 1.0, num=int(x.size), dtype=np.float32)
    xq = np.linspace(0.0, 1.0, num=int(m), dtype=np.float32)
    return np.interp(xq, xp, x.astype(np.float32)).astype(np.float32)


class StoryStructureBaselineModule(BaseModule):
    """
    Tier‑0 baseline story_structure.
    """

    VERSION = VERSION
    SCHEMA_VERSION = SCHEMA_VERSION

    @property
    def module_name(self) -> str:
        return MODULE_NAME

    def __init__(self, rs_path: Optional[str] = None, max_frames: int = 200, **kwargs: Any):
        super().__init__(rs_path=rs_path, logger_name=self.module_name, **kwargs)
        self._max_frames = int(max_frames)
        self._last_models_used: List[Dict[str, Any]] = []

    def required_dependencies(self) -> List[str]:
        return ["core_clip", "core_optical_flow", "core_face_landmarks"]

    def get_models_used(self, config: Dict[str, Any], metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        return list(self._last_models_used or [])

    def process(self, frame_manager: FrameManager, frame_indices: List[int], config: Dict[str, Any]) -> Dict[str, Any]:
        if not frame_indices:
            raise ValueError(f"{MODULE_NAME} | frame_indices is empty")
        if self.rs_path is None:
            raise ValueError(f"{MODULE_NAME} | rs_path is required")

        max_frames = int(config.get("max_frames", self._max_frames))
        fi = np.asarray([int(i) for i in frame_indices], dtype=np.int32)
        if max_frames > 0 and int(fi.size) > int(max_frames):
            raise RuntimeError(
                f"{MODULE_NAME} | too many frames: N={int(fi.size)} > max_frames={int(max_frames)} (no-fallback). "
                "Fix Segmenter sampling for story_structure."
            )

        times_s = _require_union_times_s(frame_manager, fi)
        dt = np.diff(times_s).astype(np.float32)
        dt = np.maximum(dt, 1e-3)
        video_len_s = float(times_s[-1] - times_s[0]) if times_s.size >= 2 else 0.0

        emb, mu_clip = _load_core_clip_embeddings_aligned(self.rs_path, fi)
        emb_n = _normalize_rows(np.asarray(emb, dtype=np.float32))
        motion, mu_flow = _load_core_optical_flow_aligned(self.rs_path, fi)
        any_face, mu_face, face_meta = _load_core_face_any_present_aligned(self.rs_path, fi)

        # store combined models_used for meta/model_signature
        self._last_models_used = []
        self._last_models_used.extend(mu_clip)
        self._last_models_used.extend(mu_flow)
        self._last_models_used.extend(mu_face)

        # embedding change rate (per-second)
        if emb_n.shape[0] >= 2:
            sim_next = np.sum(emb_n[1:] * emb_n[:-1], axis=1).astype(np.float32)
            diff_next = (1.0 - sim_next).astype(np.float32)
            diff_rate = (diff_next / dt).astype(np.float32)
        else:
            sim_next = np.asarray([], dtype=np.float32)
            diff_next = np.asarray([], dtype=np.float32)
            diff_rate = np.asarray([], dtype=np.float32)

        # align to frames (pad 0 at first frame)
        if diff_rate.size:
            emb_rate_curve = np.concatenate([np.asarray([0.0], dtype=np.float32), diff_rate], axis=0)
        else:
            emb_rate_curve = np.zeros((emb_n.shape[0],), dtype=np.float32)

        # motion curve is already per-second mean magnitude (core_optical_flow contract)
        motion_curve = np.asarray(motion, dtype=np.float32)
        if motion_curve.shape != emb_rate_curve.shape:
            raise RuntimeError(f"{MODULE_NAME} | motion curve shape mismatch after alignment")

        # Smooth and z-score component curves, then combine
        sigma = float(config.get("energy_smoothing_sigma", 1.0))
        sigma = max(0.0, sigma)
        emb_s = gaussian_filter1d(emb_rate_curve, sigma=sigma).astype(np.float32) if emb_rate_curve.size else emb_rate_curve
        mot_s = gaussian_filter1d(motion_curve, sigma=sigma).astype(np.float32) if motion_curve.size else motion_curve
        emb_z = _zscore(emb_s)
        mot_z = _zscore(mot_s)

        combined = (0.5 * emb_z + 0.5 * mot_z).astype(np.float32)
        combined_s = gaussian_filter1d(combined, sigma=sigma).astype(np.float32) if combined.size else combined
        story_energy_curve = _zscore(combined_s)

        # Hook window: min(5s, 15% of video). If sampling yields too few points, extend to cover at least 3 frames.
        hook_len_s = float(min(5.0, 0.15 * video_len_s)) if video_len_s > 0 else 0.0
        hook_end_t = float(times_s[0] + hook_len_s) if times_s.size else 0.0
        hook_mask = times_s <= hook_end_t if times_s.size else np.zeros((fi.size,), dtype=bool)
        if int(np.sum(hook_mask)) < 3 and fi.size >= 3:
            hook_mask = np.zeros((fi.size,), dtype=bool)
            hook_mask[:3] = True
            hook_end_t = float(times_s[min(2, times_s.size - 1)])

        hook_dur_s = float(max(hook_end_t - float(times_s[0]), 1e-6)) if times_s.size else 1e-6

        hook_emb = story_energy_curve[hook_mask] if story_energy_curve.size else np.asarray([], dtype=np.float32)
        hook_visual_surprise_score = float(np.mean(hook_emb)) if hook_emb.size else float("nan")
        hook_visual_surprise_std = float(np.std(hook_emb)) if hook_emb.size else float("nan")

        hook_motion = mot_s[hook_mask] if mot_s.size else np.asarray([], dtype=np.float32)
        if hook_motion.size:
            hook_motion_intensity = float(np.mean(hook_motion))
            p75 = float(np.percentile(hook_motion, 75))
            p90 = float(np.percentile(hook_motion, 90))
            cut_frames = hook_motion > p75
            spike_frames = hook_motion > p90
            hook_cut_rate = float(np.sum(cut_frames) / hook_dur_s)
            hook_motion_spikes = int(np.sum(spike_frames))
            hook_rhythm_score = float(
                (np.sum(hook_motion[spike_frames]) / (np.mean(hook_motion) + 1e-6)) if np.any(spike_frames) else 0.0
            )
        else:
            hook_motion_intensity = float("nan")
            hook_cut_rate = float("nan")
            hook_motion_spikes = 0
            hook_rhythm_score = float("nan")

        hook_face_presence = float(np.mean(any_face[hook_mask])) if any_face.size and np.any(hook_mask) else 0.0

        # climax = max energy
        if story_energy_curve.size:
            climax_pos = int(np.argmax(story_energy_curve))
            climax_frame = int(fi[climax_pos])
            climax_time = float(times_s[climax_pos])
            climax_strength = float(combined_s[climax_pos]) if combined_s.size else float("nan")
            climax_strength_z = float(story_energy_curve[climax_pos])
            climax_position_norm = float(climax_pos / max(len(fi) - 1, 1))
        else:
            climax_pos = -1
            climax_frame = -1
            climax_time = float("nan")
            climax_strength = float("nan")
            climax_strength_z = float("nan")
            climax_position_norm = float("nan")

        # peaks
        if story_energy_curve.size >= 4:
            p90 = float(np.percentile(story_energy_curve, 90))
            peaks, _ = find_peaks(story_energy_curve, height=p90)
            number_of_peaks = int(len(peaks))
        else:
            number_of_peaks = 0

        # time from hook to climax (normalized by video length)
        if video_len_s > 0 and np.isfinite(climax_time):
            if climax_time <= hook_end_t:
                time_from_hook_to_climax = 0.0
            else:
                time_from_hook_to_climax = float((climax_time - hook_end_t) / max(video_len_s, 1e-6))
        else:
            time_from_hook_to_climax = float("nan")

        # hook energy ratio (raw combined, not z)
        if combined_s.size and np.any(hook_mask):
            hook_energy = float(np.mean(combined_s[hook_mask]))
            avg_energy = float(np.mean(combined_s))
            hook_to_avg_energy_ratio = float(hook_energy / (avg_energy + 1e-6))
        else:
            hook_to_avg_energy_ratio = float("nan")

        # face-based global proxies (safe even if core_face_landmarks is empty)
        main_character_screen_time = float(np.mean(any_face)) if any_face.size else 0.0
        if any_face.size >= 2:
            switches = int(np.sum(np.diff(any_face.astype(np.int8)) != 0))
            speaker_switch_rate = float(switches / max(any_face.size - 1, 1))
            speaker_switches_per_minute = float(switches / max(video_len_s / 60.0, 1e-6)) if video_len_s > 0 else float("nan")
        else:
            speaker_switch_rate = float("nan")
            speaker_switches_per_minute = float("nan")

        subtitles = config.get("subtitles")
        subtitles_present = bool(subtitles) and isinstance(subtitles, list) and any(bool(s) for s in subtitles)

        features: Dict[str, Any] = {
            "n_frames": int(fi.size),
            "max_frames": int(max_frames),
            "video_length_seconds": float(video_len_s),
            # hook
            "hook_visual_surprise_score": hook_visual_surprise_score,
            "hook_visual_surprise_std": hook_visual_surprise_std,
            "hook_motion_intensity": hook_motion_intensity,
            "hook_cut_rate": hook_cut_rate,
            "hook_motion_spikes": int(hook_motion_spikes),
            "hook_rhythm_score": hook_rhythm_score,
            "hook_face_presence": float(hook_face_presence),
            # climax
            "climax_timestamp": int(climax_frame),  # union-frame index
            "climax_time_sec": float(climax_time),
            "climax_position_normalized": float(climax_position_norm),
            "climax_strength": float(climax_strength),
            "climax_strength_normalized": float(climax_strength_z),
            "number_of_peaks": int(number_of_peaks),
            "time_from_hook_to_climax": float(time_from_hook_to_climax),
            "hook_to_avg_energy_ratio": float(hook_to_avg_energy_ratio),
            # character proxies
            "main_character_screen_time": float(main_character_screen_time),
            "speaker_switch_rate": float(speaker_switch_rate),
            "speaker_switches_per_minute": float(speaker_switches_per_minute),
            # subtitles placeholders (legacy topics use ModelManager in legacy_story_structure.py)
            "has_subtitles": bool(subtitles_present),
            "number_of_topics": float("nan"),
            "topic_diversity": float("nan"),
            "semantic_coherence_score": float("nan"),
            # trace
            "core_face_landmarks_empty_reason": face_meta.get("empty_reason") if isinstance(face_meta, dict) else None,
        }

        return {
            "frame_indices": fi,
            "times_s": times_s.astype(np.float32),
            "embedding_sim_next": sim_next,
            "embedding_diff_next": diff_next,
            "embedding_change_rate_per_sec": emb_rate_curve.astype(np.float32),
            "motion_norm_per_sec_mean": mot_s.astype(np.float32),
            "any_face_present": np.asarray(any_face, dtype=bool),
            "story_energy_curve": story_energy_curve.astype(np.float32),
            "story_energy_curve_downsampled_128": _downsample_to_fixed(story_energy_curve, 128),
            "subtitles_present": np.asarray(bool(subtitles_present)),
            "features": features,
        }


