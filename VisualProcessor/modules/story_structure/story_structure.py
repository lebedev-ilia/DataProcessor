"""
`story_structure`

Содержит:
- legacy pipeline `StoryStructurePipelineOptimized` (оставлен для справки; может требовать доп. зависимости)
- production baseline `StoryStructureBaselineModule(BaseModule)`:
  использует только результаты `core_clip` (NPZ) и строго следует `frame_indices` от Segmenter.
"""

from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter1d, uniform_filter1d
from scipy.signal import find_peaks

from modules.base_module import BaseModule
from utils.frame_manager import FrameManager

def _load_core_optical_flow_curve(rs_path: str):
    """
    Загружает покадровую кривую движения из core‑провайдера optical_flow, если есть.

    Используется как приоритетный источник движения для story_structure.
    """
    if not rs_path:
        return None

    stats_path = os.path.join(rs_path, "optical_flow", "statistical_analysis.json")
    if not os.path.isfile(stats_path):
        return None

    try:
        with open(stats_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None

    stats = (data.get("statistics") or {}).get("frame_statistics") or []
    if not stats:
        return None

    curve = []
    for fs in stats:
        v = (
            fs.get("magnitude_mean_px_sec_norm")
            if "magnitude_mean_px_sec_norm" in fs
            else fs.get("magnitude_mean_px_sec", fs.get("magnitude_mean", 0.0))
        )
        curve.append(float(v))

    if not curve:
        return None

    return np.asarray(curve, dtype=np.float32)


def _load_core_clip_embeddings_aligned(rs_path: str, frame_indices: np.ndarray) -> np.ndarray:
    """
    Загружает CLIP‑эмбеддинги из `core_clip/embeddings.npz` и выравнивает их по frame_indices.

    ВАЖНО: `frame_indices` здесь — индексы в union-domain (frames_dir),
    а в core_clip NPZ `frame_embeddings` обычно идут в порядке `core_frame_indices`.
    """
    if not rs_path:
        raise ValueError("story_structure | rs_path is required")

    core_path = os.path.join(rs_path, "core_clip", "embeddings.npz")
    if not os.path.isfile(core_path):
        raise FileNotFoundError(f"story_structure | missing core_clip embeddings: {core_path}")

    data = np.load(core_path, allow_pickle=True)
    core_idx = data.get("frame_indices")
    core_emb = data.get("frame_embeddings")
    if core_idx is None or core_emb is None:
        raise RuntimeError("story_structure | core_clip embeddings.npz missing keys frame_indices/frame_embeddings")

    core_idx = np.asarray(core_idx, dtype=np.int32)
    core_emb = np.asarray(core_emb, dtype=np.float32)

    mapping = {int(fi): i for i, fi in enumerate(core_idx.tolist())}
    pos = [mapping.get(int(fi), -1) for fi in frame_indices.tolist()]
    if any(p < 0 for p in pos):
        raise RuntimeError(
            "story_structure | core_clip does not cover requested frame_indices. "
            "Segmenter must provide consistent indices for core_clip and story_structure."
        )
    return core_emb[np.asarray(pos, dtype=np.int64)]


def compute_optical_flow(frame_manager, frames, rs_path: str = None):
    """
    Compute dense optical flow magnitude per frame.
    Использует только core_optical_flow (RAFT) - обязательное требование.
    """
    core_curve = _load_core_optical_flow_curve(rs_path)
    if core_curve is None or core_curve.size == 0:
        raise RuntimeError(
            f"story_structure | compute_optical_flow | core_optical_flow не найден. "
            f"Убедитесь, что core провайдер optical_flow запущен перед этим модулем. "
            f"rs_path: {rs_path}"
        )
    
    return core_curve

def embedding_diff(embeddings):
    """Compute frame-to-frame embedding difference (cosine distance)"""
    diffs = []
    for i in range(1, len(embeddings)):
        sim = cosine_similarity([embeddings[i-1]], [embeddings[i]])[0][0]
        diffs.append(1 - sim)
    return np.array(diffs)

def smooth_signal(signal, window=3):
    if signal is None or len(signal) == 0:
        return np.array([], dtype=np.float32)
    window = max(int(window), 1)
    return uniform_filter1d(signal.astype(np.float32), size=window)


def enforce_min_run_length(labels: np.ndarray, min_len: int) -> np.ndarray:
    """
    Simple post-processing for clustering labels:
    merge too-short runs into nearest neighbour segment.
    """
    if labels.size == 0 or min_len <= 1:
        return labels

    labels = labels.copy()
    start = 0
    n = len(labels)
    while start < n:
        end = start + 1
        while end < n and labels[end] == labels[start]:
            end += 1
        run_len = end - start
        if run_len < min_len:
            left_label = labels[start - 1] if start > 0 else None
            right_label = labels[end] if end < n else None
            # prefer merging into the side with a defined label, fallback to right
            target = left_label if left_label is not None else right_label
            if target is not None:
                labels[start:end] = target
        start = end
    return labels

# -----------------------------
# Story Structure Pipeline
# -----------------------------
class StoryStructurePipelineOptimized:
    def __init__(
        self,
        frame_manager,
        frame_indices,
        clip_model: str = "ViT-B/32",
        sentence_model: str = "all-MiniLM-L6-v2",
        min_segment_length_seconds: float = 0.5,
        min_story_segments: int = 2,
        max_story_segments: int = 8,
        rs_path: str = None,
    ):
        self.frame_manager = frame_manager
        self.frame_indices = frame_indices
        self.rs_path = rs_path

        # Load models
        # torch и CLIP удалены - используем только core_clip
        m_name = sentence_model.replace("-", "_")
        model_path = f"{os.path.dirname(__file__)}/models/{m_name}"
        self.sentence_model = SentenceTransformer(model_name_or_path=sentence_model, cache_folder=model_path)

        # Timing / segmentation hyperparams
        self.fps = float(getattr(self.frame_manager, "fps", 30.0) or 30.0)
        self.total_frames = int(getattr(self.frame_manager, "total_frames", len(self.frame_indices)))
        self.min_segment_length_seconds = float(min_segment_length_seconds)
        self.min_story_segments = int(min_story_segments)
        self.max_story_segments = int(max_story_segments)

        # Mediapipe face_mesh удалён - используем только core_face_landmarks (если нужно)

        # Outputs
        self.features = {}

    # -----------------------------
    # 1. CLIP embeddings
    # -----------------------------
    def compute_clip_embeddings(self):
        """
        Использует только core_clip - обязательное требование.
        """
        core_emb = _load_core_clip_embeddings_aligned(self.rs_path, np.asarray(self.frame_indices, dtype=np.int32))
        self.clip_embeddings = core_emb
        return self.clip_embeddings


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-9
    return x / norms


def _zscore(x: np.ndarray) -> np.ndarray:
    if x.size == 0:
        return x.astype(np.float32)
    x = x.astype(np.float32)
    mu = float(np.mean(x))
    sd = float(np.std(x)) + 1e-6
    return (x - mu) / sd


class StoryStructureBaselineModule(BaseModule):
    """
    Baseline story_structure:
    - story_energy_curve: based on CLIP embedding diffs
    - hook/climax summary stats
    - subtitles/topic features: baseline placeholder (NaN) unless subtitles passed
    """

    @property
    def module_name(self) -> str:
        return "story_structure"

    def required_dependencies(self) -> List[str]:
        return ["core_clip"]

    def process(self, frame_manager: FrameManager, frame_indices: List[int], config: Dict[str, Any]) -> Dict[str, Any]:
        if not frame_indices:
            raise ValueError("story_structure | frame_indices is empty")
        if self.rs_path is None:
            raise ValueError("story_structure | rs_path is required")

        fi = np.asarray([int(i) for i in frame_indices], dtype=np.int32)
        emb = _load_core_clip_embeddings_aligned(self.rs_path, fi)
        emb_n = _normalize_rows(np.asarray(emb, dtype=np.float32))

        # diff signal (cosine distance between consecutive embeddings)
        if emb_n.shape[0] >= 2:
            sim_next = np.sum(emb_n[1:] * emb_n[:-1], axis=1).astype(np.float32)
            diff_next = (1.0 - sim_next).astype(np.float32)
        else:
            sim_next = np.asarray([], dtype=np.float32)
            diff_next = np.asarray([], dtype=np.float32)

        # energy curve aligned to frames: pad 0 at first frame
        if diff_next.size:
            energy = np.concatenate([np.asarray([0.0], dtype=np.float32), diff_next], axis=0)
        else:
            energy = np.zeros((emb_n.shape[0],), dtype=np.float32)
        energy_smooth = gaussian_filter1d(energy, sigma=1).astype(np.float32) if energy.size else energy
        energy_z = _zscore(energy_smooth)

        # timestamps (if Segmenter provided union_timestamps_sec)
        meta = getattr(frame_manager, "meta", {}) or {}
        union_ts = meta.get("union_timestamps_sec")
        fps = float(getattr(frame_manager, "fps", 30.0) or 30.0)

        times_s: Optional[np.ndarray] = None
        if isinstance(union_ts, list) and len(union_ts) > 0:
            uts = np.asarray(union_ts, dtype=np.float32)
            if int(np.max(fi)) < uts.shape[0]:
                times_s = uts[fi]

        if times_s is None:
            # fallback for time computation only (not sampling): use fps
            times_s = (fi.astype(np.float32) / max(fps, 1e-6)).astype(np.float32)

        video_len_s = float(times_s[-1] - times_s[0]) if times_s.size >= 2 else float(times_s[-1]) if times_s.size else 0.0
        hook_len_s = min(5.0, 0.15 * max(video_len_s, 0.0)) if video_len_s > 0 else 0.0
        hook_end_t = float(times_s[0] + hook_len_s) if times_s.size else 0.0
        hook_mask = times_s <= hook_end_t if times_s.size else np.zeros((fi.size,), dtype=bool)

        # hook metrics
        hook_energy = energy_z[hook_mask] if energy_z.size else np.asarray([], dtype=np.float32)
        hook_surprise_mean = float(np.mean(hook_energy)) if hook_energy.size else float("nan")
        hook_surprise_std = float(np.std(hook_energy)) if hook_energy.size else float("nan")

        # climax = max energy_z
        if energy_z.size:
            climax_pos = int(np.argmax(energy_z))
            climax_frame = int(fi[climax_pos])
            climax_time = float(times_s[climax_pos])
            climax_strength = float(energy_smooth[climax_pos])
            climax_strength_z = float(energy_z[climax_pos])
            climax_position_norm = float(climax_pos / max(len(fi) - 1, 1))
        else:
            climax_pos = -1
            climax_frame = -1
            climax_time = float("nan")
            climax_strength = float("nan")
            climax_strength_z = float("nan")
            climax_position_norm = float("nan")

        # peak count (robust)
        if energy_z.size >= 4:
            p90 = float(np.percentile(energy_z, 90))
            peaks, _ = find_peaks(energy_z, height=p90)
            number_of_peaks = int(len(peaks))
        else:
            number_of_peaks = 0

        # subtitles placeholders
        subtitles = config.get("subtitles")
        subtitles_present = bool(subtitles) and isinstance(subtitles, list) and any(bool(s) for s in subtitles)

        features: Dict[str, Any] = {
            # minimal baseline
            "n_frames": int(fi.size),
            "hook_visual_surprise_score": hook_surprise_mean,
            "hook_visual_surprise_std": hook_surprise_std,
            "climax_timestamp": climax_frame,  # union-frame index
            "climax_time_sec": climax_time,
            "climax_position_normalized": climax_position_norm,
            "climax_strength": climax_strength,
            "climax_strength_normalized": climax_strength_z,
            "number_of_peaks": number_of_peaks,
            # placeholders for topic features
            "has_subtitles": bool(subtitles_present),
            "number_of_topics": float("nan"),
            "topic_diversity": float("nan"),
            "semantic_coherence_score": float("nan"),
        }

        return {
            "frame_indices": fi,
            "times_s": times_s.astype(np.float32),
            "embedding_sim_next": sim_next,
            "embedding_diff_next": diff_next,
            "story_energy_curve": energy_z.astype(np.float32),
            "subtitles_present": np.asarray(bool(subtitles_present)),
            "features": np.asarray(features, dtype=object),
        }

    # -----------------------------
    # 2. Story Segmentation
    # -----------------------------
    def story_segmentation(self, n_segments=None):
        """
        Segment video based on smoothed CLIP embeddings.

        - Respects minimal segment length in seconds.
        - Returns normalized average segment duration (0..1) in addition to raw value.
        - Uses simple post-processing to remove very short noisy segments.
        """
        if not hasattr(self, "clip_embeddings"):
            raise RuntimeError("compute_clip_embeddings must be called before story_segmentation")

        # Smooth embeddings frame-wise
        smooth_emb = smooth_signal(self.clip_embeddings, window=3)

        # Determine number of segments if not provided
        frames_count = len(self.frame_indices)
        min_seg_len_frames = max(int(self.min_segment_length_seconds * self.fps), 1)
        if n_segments is None:
            max_reasonable_segments = max(frames_count // max(min_seg_len_frames, 1), 1)
            n_segments = int(
                np.clip(max_reasonable_segments, self.min_story_segments, self.max_story_segments)
            )

        # Hierarchical clustering
        clustering = AgglomerativeClustering(
            n_clusters=n_segments,
            metric="cosine",
            linkage="average",
        )
        labels_raw = clustering.fit_predict(smooth_emb)

        # Post-process labels to enforce minimal run length
        labels = enforce_min_run_length(labels_raw, min_len=min_seg_len_frames)
        self.segment_labels = labels

        # Features
        unique_segments = np.unique(labels)
        segment_durations = [int(np.sum(labels == s)) for s in unique_segments]
        avg_duration_frames = float(np.mean(segment_durations)) if segment_durations else 0.0
        total_frames = float(frames_count) if frames_count > 0 else 1.0

        self.features["number_of_story_segments"] = int(len(unique_segments))
        self.features["avg_story_segment_duration"] = avg_duration_frames
        self.features["avg_story_segment_duration_normalized"] = float(
            avg_duration_frames / total_frames
        )

        # More robust abrupt transitions count after post-processing
        self.features["abrupt_story_transition_count"] = int(np.sum(np.diff(labels) != 0))

        # Narrative continuity score: similarity between consecutive segment-level means
        cont_scores = []
        prev_seg_emb = None
        prev_len = None
        for seg_id in unique_segments:
            idx = np.where(labels == seg_id)[0]
            if idx.size == 0:
                continue
            seg_emb = self.clip_embeddings[idx].mean(axis=0).reshape(1, -1)
            seg_len = idx.size
            if prev_seg_emb is not None:
                sim = cosine_similarity(prev_seg_emb, seg_emb)[0][0]
                # weight by segment lengths
                weight = (prev_len + seg_len) / (2.0 * total_frames)
                cont_scores.append(sim * weight)
            prev_seg_emb = seg_emb
            prev_len = seg_len

        if cont_scores:
            self.features["narrative_continuity_score"] = float(np.sum(cont_scores))
            # std of unweighted similarities for variability signal
            self.features["narrative_continuity_std"] = float(np.std(cont_scores))
        else:
            self.features["narrative_continuity_score"] = 0.0
            self.features["narrative_continuity_std"] = 0.0
        return self.features

    # -----------------------------
    # 3. Hook Features
    # -----------------------------
    def hook_features(self, hook_base_seconds: float = 5.0):
        """
        Hook features for the beginning of the video.

        Effective window = min(hook_base_seconds, 0.15 * video_length_seconds).
        """
        total_frames = len(self.frame_indices)
        if total_frames == 0:
            # nothing to compute
            self.features.update(
                {
                    "hook_motion_intensity": 0.0,
                    "hook_cut_rate": 0.0,
                    "hook_motion_spikes": 0.0,
                    "hook_rhythm_score": 0.0,
                    "hook_face_presence": 0.0,
                    "hook_visual_surprise_score": 0.0,
                    "hook_visual_surprise_std": 0.0,
                    "hook_brightness_spike": 0.0,
                    "hook_saturation_spike": 0.0,
                }
            )
            return self.features

        video_length_seconds = float(total_frames) / self.fps
        effective_window_seconds = min(hook_base_seconds, 0.15 * video_length_seconds)
        n_frames = max(int(effective_window_seconds * self.fps), 1)
        n_frames = min(n_frames, total_frames)
        hook_frames = self.frame_indices[:n_frames]

        # Optical flow
        if len(hook_frames) > 1:
            hook_flow = compute_optical_flow(self.frame_manager, hook_frames)
            hook_flow_smooth = smooth_signal(hook_flow, window=3)
            motion_intensity = float(np.mean(hook_flow_smooth))
            # per-window percentiles (robust to scale)
            p75 = np.percentile(hook_flow_smooth, 75)
            p90 = np.percentile(hook_flow_smooth, 90)
            cut_frames = hook_flow_smooth > p75
            spike_frames = hook_flow_smooth > p90
            hook_cut_rate = float(np.sum(cut_frames) / max(effective_window_seconds, 1e-6))
            hook_motion_spikes = int(np.sum(spike_frames))
            # aggregate rhythm score: normalized sum of peaks
            hook_rhythm_score = float(
                (np.sum(hook_flow_smooth[spike_frames]) / (np.mean(hook_flow_smooth) + 1e-6))
                if np.any(spike_frames)
                else 0.0
            )
        else:
            motion_intensity = 0.0
            hook_cut_rate = 0.0
            hook_motion_spikes = 0
            hook_rhythm_score = 0.0

        self.features["hook_motion_intensity"] = motion_intensity
        self.features["hook_cut_rate"] = hook_cut_rate
        self.features["hook_motion_spikes"] = hook_motion_spikes
        self.features["hook_rhythm_score"] = hook_rhythm_score

        # Face presence in hook window only
        face_count = 0
        for idx in hook_frames:
            frame = self.frame_manager.get(idx)
            results = self.mp_face.process(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if results.multi_face_landmarks:
                face_count += 1
        self.features["hook_face_presence"] = float(face_count / n_frames)

        # Visual surprise: embedding jumps (frame-to-frame cosine distance), per-hook
        hook_emb = self.clip_embeddings[:n_frames]
        diff = embedding_diff(hook_emb)
        if diff.size:
            self.features["hook_visual_surprise_score"] = float(np.mean(diff))
            self.features["hook_visual_surprise_std"] = float(np.std(diff))
        else:
            self.features["hook_visual_surprise_score"] = 0.0
            self.features["hook_visual_surprise_std"] = 0.0

        # Brightness / Saturation spike with per-video normalization (z-score style within hook)
        brightness = []
        saturation = []
        for idx in hook_frames:
            frame = self.frame_manager.get(idx)
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
            brightness.append(float(np.mean(hsv[:, :, 2])))
            saturation.append(float(np.mean(hsv[:, :, 1])))
        brightness = np.array(brightness, dtype=np.float32)
        saturation = np.array(saturation, dtype=np.float32)
        if brightness.size:
            b_mean = float(np.mean(brightness))
            b_std = float(np.std(brightness) + 1e-6)
            self.features["hook_brightness_spike"] = float(
                (np.max(brightness) - b_mean) / b_std
            )
        else:
            self.features["hook_brightness_spike"] = 0.0
        if saturation.size:
            s_mean = float(np.mean(saturation))
            s_std = float(np.std(saturation) + 1e-6)
            self.features["hook_saturation_spike"] = float(
                (np.max(saturation) - s_mean) / s_std
            )
        else:
            self.features["hook_saturation_spike"] = 0.0
        return self.features

    # -----------------------------
    # 4. Climax Detection
    # -----------------------------
    def climax_detection(self):
        """
        Detect story climax on combined motion + visual-change energy curve.

        Adds:
        - climax_position_normalized (0..1)
        - climax_strength_normalized (z-score within video)
        - time_from_hook_to_climax (in fraction of video)
        """
        if len(self.frame_indices) < 2:
            self.features.update(
                {
                    "climax_timestamp": 0,
                    "climax_position_normalized": 0.0,
                    "climax_strength": 0.0,
                    "climax_strength_normalized": 0.0,
                    "number_of_peaks": 0,
                    "climax_duration": 0,
                    "time_from_hook_to_climax": 0.0,
                    "story_energy_curve": [],
                    "hook_to_avg_energy_ratio": 0.0,
                }
            )
            return self.features

        # Combine signals: motion + embedding diff
        motion = compute_optical_flow(self.frame_manager, self.frame_indices)
        motion_smooth = smooth_signal(motion, window=5)
        embed_diff = embedding_diff(self.clip_embeddings)
        embed_diff_smooth = smooth_signal(embed_diff, window=5)
        L = min(len(motion_smooth), len(embed_diff_smooth))
        if L == 0:
            combined_signal = np.zeros(len(self.frame_indices), dtype=np.float32)
        else:
            combined_signal = motion_smooth[:L] + embed_diff_smooth[:L]

        # Peak detection with basic prominence / distance safeguards
        if combined_signal.size:
            # basic normalization for peak picking
            signal_mean = float(np.mean(combined_signal))
            signal_std = float(np.std(combined_signal) + 1e-6)
            norm_sig = (combined_signal - signal_mean) / signal_std
            peaks, props = find_peaks(
                norm_sig,
                prominence=0.5,
                distance=max(int(self.fps * 0.5), 1),
            )
            if peaks.size:
                # take highest prominence peak as climax
                prominences = props.get("prominences", np.ones_like(peaks, dtype=np.float32))
                main_peak_idx = int(peaks[int(np.argmax(prominences))])
            else:
                main_peak_idx = int(np.argmax(norm_sig))
        else:
            norm_sig = combined_signal
            main_peak_idx = 0

        total_frames = float(max(len(self.frame_indices), 1))
        self.features["climax_timestamp"] = int(main_peak_idx)
        self.features["climax_position_normalized"] = float(main_peak_idx / total_frames)

        if combined_signal.size:
            raw_strength = float(combined_signal[main_peak_idx])
            mean_strength = float(np.mean(combined_signal))
            std_strength = float(np.std(combined_signal) + 1e-6)
            self.features["climax_strength"] = raw_strength
            self.features["climax_strength_normalized"] = float(
                (raw_strength - mean_strength) / std_strength
            )
            # number of peaks above robust threshold
            self.features["number_of_peaks"] = int(
                np.sum(combined_signal > np.percentile(combined_signal, 90))
            )
            # duration above high-energy percentile (75th) instead of 50%
            self.features["climax_duration"] = int(
                np.sum(combined_signal > np.percentile(combined_signal, 75))
            )
        else:
            self.features["climax_strength"] = 0.0
            self.features["climax_strength_normalized"] = 0.0
            self.features["number_of_peaks"] = 0
            self.features["climax_duration"] = 0

        # story energy curve (for transformers, can be downsampled externally)
        self.features["story_energy_curve"] = combined_signal.astype(np.float32).tolist()

        # hook vs average energy ratio
        total_len = len(combined_signal)
        if total_len > 0:
            hook_len = min(int(self.fps * 5), int(0.15 * total_len))
            hook_len = max(hook_len, 1)
            hook_energy = float(np.mean(combined_signal[:hook_len]))
            avg_energy = float(np.mean(combined_signal))
            self.features["hook_to_avg_energy_ratio"] = float(
                hook_energy / (avg_energy + 1e-6)
            )
        else:
            self.features["hook_to_avg_energy_ratio"] = 0.0

        # time from hook to climax in % of video
        hook_end_frame = min(int(self.fps * 5), int(0.15 * total_frames))
        hook_end_frame = max(hook_end_frame, 1)
        if main_peak_idx <= hook_end_frame:
            self.features["time_from_hook_to_climax"] = 0.0
        else:
            self.features["time_from_hook_to_climax"] = float(
                (main_peak_idx - hook_end_frame) / total_frames
            )
        return self.features

    # -----------------------------
    # 5. Character-level Features
    # -----------------------------
    def character_features(self):
        """
        Character-related aggregate and curve features.

        Note: true identity-level tracking is not available here; we approximate:
        - number_of_unique_identities ~= max concurrent face count
        - main_character_screen_time ~= доля кадров с хоть одним лицом
        """
        face_counts = []
        face_area_fractions = []
        frame_h = getattr(self.frame_manager, "height", None)
        frame_w = getattr(self.frame_manager, "width", None)
        frame_area = None
        if frame_h is not None and frame_w is not None:
            frame_area = float(frame_h * frame_w)

        for idx in self.frame_indices:
            frame = self.frame_manager.get(idx)
            results = self.mp_face.process(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if results.multi_face_landmarks:
                n_faces = len(results.multi_face_landmarks)
                face_counts.append(n_faces)

                # approximate total face area fraction from landmarks
                if frame_area:
                    total_area = 0.0
                    h, w = frame.shape[:2]
                    for lm_list in results.multi_face_landmarks:
                        xs = [lm.x * w for lm in lm_list.landmark]
                        ys = [lm.y * h for lm in lm_list.landmark]
                        min_x, max_x = min(xs), max(xs)
                        min_y, max_y = min(ys), max(ys)
                        total_area += max(0.0, (max_x - min_x)) * max(0.0, (max_y - min_y))
                    face_area_fractions.append(float(total_area / (frame_area + 1e-6)))
                else:
                    face_area_fractions.append(0.0)
            else:
                face_counts.append(0)
                face_area_fractions.append(0.0)

        if not face_counts:
            self.features["number_of_unique_identities"] = 0
            self.features["main_character_screen_time"] = 0.0
            self.features["speaker_switch_rate"] = 0.0
            self.features["face_presence_curve"] = []
            self.features["face_area_fraction_curve"] = []
            return self.features

        face_counts_arr = np.array(face_counts, dtype=np.int32)
        presence = face_counts_arr > 0

        # proxy for unique identities: max concurrent visible faces
        self.features["number_of_unique_identities"] = int(np.max(face_counts_arr))
        self.features["main_character_screen_time"] = float(
            np.sum(presence) / len(face_counts_arr)
        )
        self.features["speaker_switch_rate"] = float(
            np.sum(np.diff(presence.astype(np.int8)) != 0) / max(len(face_counts_arr) - 1, 1)
        )
        self.features["face_presence_curve"] = face_counts_arr.tolist()
        self.features["face_area_fraction_curve"] = [
            float(x) for x in np.asarray(face_area_fractions, dtype=np.float32)
        ]
        return self.features

    # -----------------------------
    # 6. Topic Features
    # -----------------------------
    def topic_features(self, subtitles=None):
        """
        Topic-level features based on sentence embeddings.

        If subtitles are missing or empty, returns NaN-like values and has_subtitles=False.
        """
        if subtitles is None or len(subtitles) == 0:
            self.features["has_subtitles"] = False
            self.features["number_of_topics"] = 0
            self.features["avg_topic_duration"] = float("nan")
            self.features["avg_topic_duration_normalized"] = float("nan")
            self.features["topic_shift_times"] = []
            self.features["topic_diversity"] = float("nan")
            self.features["topic_diversity_normalized"] = float("nan")
            self.features["semantic_coherence_score"] = float("nan")
            self.features["topic_coherence_std"] = float("nan")
            return self.features

        self.features["has_subtitles"] = True
        embeddings = self.sentence_model.encode(subtitles)
        n_subs = len(subtitles)
        n_clusters = min(5, n_subs)
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters, metric="cosine", linkage="average"
        )
        labels = clustering.fit_predict(embeddings)

        unique_labels = np.unique(labels)
        num_topics = int(len(unique_labels))
        self.features["number_of_topics"] = num_topics

        # durations in number of subtitle units
        durations = [int(np.sum(labels == i)) for i in unique_labels]
        avg_dur = float(np.mean(durations)) if durations else float("nan")
        self.features["avg_topic_duration"] = avg_dur
        self.features["avg_topic_duration_normalized"] = float(
            avg_dur / max(n_subs, 1)
        )

        # topic shift indices (we do not have reliable timecodes here)
        self.features["topic_shift_times"] = np.where(np.diff(labels) != 0)[0].tolist()

        # diversity and normalized diversity
        if n_subs > 0:
            self.features["topic_diversity"] = float(num_topics / n_subs)
            self.features["topic_diversity_normalized"] = float(
                num_topics / np.log(n_subs + 1.0)
            )
        else:
            self.features["topic_diversity"] = float("nan")
            self.features["topic_diversity_normalized"] = float("nan")

        # Semantic coherence – weighted mean and std over topics
        coherences = []
        weights = []
        for lab in unique_labels:
            idx = np.where(labels == lab)[0]
            if len(idx) > 1:
                sims = cosine_similarity(embeddings[idx])
                # upper triangle mean (excluding diagonal)
                triu_idxs = np.triu_indices_from(sims, k=1)
                vals = sims[triu_idxs]
                if vals.size:
                    coherences.append(float(np.mean(vals)))
                    weights.append(float(len(idx)))

        if coherences:
            coherences_arr = np.asarray(coherences, dtype=np.float32)
            weights_arr = np.asarray(weights, dtype=np.float32)
            w_mean = float(np.average(coherences_arr, weights=weights_arr))
            self.features["semantic_coherence_score"] = w_mean
            self.features["topic_coherence_std"] = float(np.std(coherences_arr))
        else:
            self.features["semantic_coherence_score"] = float("nan")
            self.features["topic_coherence_std"] = float("nan")
        return self.features

    # -----------------------------
    # 7. Run All
    # -----------------------------
    def extract_all_features(self, subtitles=None):
        self.compute_clip_embeddings()
        self.story_segmentation()
        self.hook_features()
        self.climax_detection()
        self.character_features()
        self.topic_features(subtitles=subtitles)
        # Optional: downsampled story energy curve for transformer inputs
        if "story_energy_curve" in self.features and self.features["story_energy_curve"]:
            curve = np.asarray(self.features["story_energy_curve"], dtype=np.float32)
            target_len = 128
            if curve.size > target_len:
                # simple adaptive pooling by interpolation
                x_old = np.linspace(0.0, 1.0, num=curve.size)
                x_new = np.linspace(0.0, 1.0, num=target_len)
                curve_ds = np.interp(x_new, x_old, curve)
            else:
                curve_ds = curve
            self.features["story_energy_curve_downsampled_128"] = curve_ds.tolist()
        return self.features
