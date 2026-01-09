"""
video_pacing

Содержит:
- `VideoPacingPipelineVisualOptimized` (feature extraction on sampled frames)
- `VideoPacingModule(BaseModule)` — NPZ output + strict frame_indices + core provider integration
"""

from __future__ import annotations

import os

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2lab
from scipy.stats import entropy
from scipy.signal import find_peaks
from typing import List, Dict, Optional, Any, Tuple

import warnings

warnings.filterwarnings("ignore")

from modules.base_module import BaseModule
from utils.frame_manager import FrameManager


def _require_union_times_s(frame_manager: FrameManager, frame_indices: List[int]) -> np.ndarray:
    """
    Segmenter contract: union_timestamps_sec is source-of-truth for time axis.
    No-fallback: if missing/invalid -> error.
    """
    meta = getattr(frame_manager, "meta", None)
    if not isinstance(meta, dict):
        raise RuntimeError("video_pacing | FrameManager.meta missing (requires union_timestamps_sec)")
    ts = meta.get("union_timestamps_sec")
    if not isinstance(ts, list) or not ts:
        raise RuntimeError("video_pacing | union_timestamps_sec missing/empty in frames metadata (no-fallback)")
    uts = np.asarray(ts, dtype=np.float32)
    fi = np.asarray([int(i) for i in frame_indices], dtype=np.int32)
    if fi.size == 0:
        raise RuntimeError("video_pacing | frame_indices is empty (no-fallback)")
    if int(np.max(fi)) >= int(uts.shape[0]):
        raise RuntimeError("video_pacing | union_timestamps_sec does not cover frame_indices (no-fallback)")
    times_s = uts[fi]
    if times_s.size >= 2 and np.any(np.diff(times_s) < -1e-3):
        raise RuntimeError("video_pacing | union_timestamps_sec is not monotonic for frame_indices (no-fallback)")
    return times_s.astype(np.float32)


def _mad(x: np.ndarray) -> float:
    """Median absolute deviation (robust scale)."""
    if x.size == 0:
        return 0.0
    med = float(np.median(x))
    return float(np.median(np.abs(x - med)))


def gini_coefficient(values: np.ndarray) -> float:
    """Gini для неотрицательного массива."""
    if values.size == 0:
        return 0.0
    vals = values.astype(np.float64)
    if np.any(vals < 0):
        vals = vals - vals.min()
    if np.allclose(vals, 0):
        return 0.0
    vals_sorted = np.sort(vals)
    n = vals_sorted.size
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * vals_sorted) / (n * np.sum(vals_sorted))) - (n + 1) / n)


def _load_core_optical_flow_npz(rs_path: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Loads core_optical_flow NPZ:
    <rs_path>/core_optical_flow/flow.npz
    """
    if not rs_path:
        return None
    p = os.path.join(rs_path, "core_optical_flow", "flow.npz")
    if not os.path.isfile(p):
        return None
    try:
        data = np.load(p, allow_pickle=True)
        idx = data.get("frame_indices")
        curve = data.get("motion_norm_per_sec_mean")
        if idx is None or curve is None:
            return None
        return {
            "frame_indices": np.asarray(idx, dtype=np.int32),
            "motion_norm_per_sec_mean": np.asarray(curve, dtype=np.float32),
            "meta": data.get("meta"),
        }
    except Exception:
        return None


def _load_core_clip_npz(rs_path: Optional[str]) -> Optional[Dict[str, Any]]:
    """
    Пытается загрузить CLIP‑эмбеддинги из core_clip провайдера.

    Ожидается файл:
    <rs_path>/core_clip/embeddings.npz
    """
    if not rs_path:
        return None

    core_path = os.path.join(rs_path, "core_clip", "embeddings.npz")
    if not os.path.isfile(core_path):
        return None

    try:
        data = np.load(core_path, allow_pickle=True)
        frame_indices = data.get("frame_indices")
        emb = data.get("frame_embeddings")
        if frame_indices is None or emb is None:
            return None
        return {
            "frame_indices": np.asarray(frame_indices, dtype=np.int32),
            "frame_embeddings": np.asarray(emb, dtype=np.float32),
            "created_at": data.get("created_at"),
            "version": data.get("version"),
            "model_name": data.get("model_name"),
        }
    except Exception:
        return None


class VideoPacingPipelineVisualOptimized:
    def __init__(
        self,
        frame_manager,
        frame_indices,
        clip_model_name: str = "ViT-B/32",
        batch_size: int = 32,
        downscale_factor: float = 0.25,
        min_shot_length_seconds: float = 0.15,
        shot_detect_k: float = 6.0,
        rs_path: Optional[str] = None,
    ):
        """
        batch_size: батч для CLIP
        downscale_factor: для Optical Flow и color/lighting features
        """
        self.batch_size = int(batch_size)
        self.downscale_factor = float(downscale_factor)

        # Загружаем кадры через FrameManager
        self.frame_manager = frame_manager
        self.frame_indices = [int(i) for i in frame_indices]
        self.total_frames = len(frame_indices)
        # Strict time-axis contract (Segmenter)
        self.times_s = _require_union_times_s(self.frame_manager, self.frame_indices)
        self.video_length_seconds = float(max(self.times_s[-1] - self.times_s[0], 0.0)) if self.times_s.size else 0.0
        self.min_shot_length_seconds = float(min_shot_length_seconds)
        self.shot_detect_k = float(shot_detect_k)
        self.rs_path = rs_path

        # CLIP модель удалена - используем только core_clip

        # Определяем шоты и сцены
        self.shot_boundaries = self._detect_shots_with_merging()

    def _get_resize_frame(self, idx):
        return cv2.resize(
            self.frame_manager.get(idx),
            (0, 0),
            fx=self.downscale_factor,
            fy=self.downscale_factor,
        )

    # -------------------------
    # Shot Detection with SSIM
    # -------------------------

    def _safe_ssim(self, img1, img2):
        h, w = img1.shape[:2]
        min_side = min(h, w)

        # No-fallback: if Segmenter produced frames too small for SSIM, treat as invalid input.
        if min_side < 3:
            raise RuntimeError("video_pacing | frames too small for SSIM (min_side < 3). Check Segmenter sampling/resolution.")

        win_size = min(7, min_side if min_side % 2 == 1 else min_side - 1)
        if win_size < 3:
            raise RuntimeError("video_pacing | frames too small for SSIM window (win_size < 3).")

        return ssim(
            img1,
            img2,
            channel_axis=-1,
            win_size=win_size
        )

    def _detect_shots_with_merging(self) -> List[int]:
        """
        Shot boundary detection on sampled frames.

        Design goals (audit):
        - no hard-coded global thresholds tied to FPS / sampling density
        - robust thresholds derived from per-video statistics (MAD)
        - merging too-short shots uses time axis (union_timestamps_sec)
        """
        if not self.frame_indices:
            return [0]

        # IMPORTANT: store boundaries as POSITIONS in `self.frame_indices` list (0..N-1),
        # not as union frame indices. This keeps all downstream computations consistent.
        # 1-pass feature extraction for transitions
        ssim_scores: List[float] = []
        chi_scores: List[float] = []
        edge_scores: List[float] = []
        vdiff_scores: List[float] = []

        prev_idx = self.frame_indices[0]
        prev_frame = self._get_resize_frame(prev_idx)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)
        prev_hsv = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2HSV)
        prev_v = float(np.mean(prev_hsv[:, :, 2]))
        prev_hist = cv2.calcHist([prev_gray], [0], None, [32], [0, 256])
        prev_hist = cv2.normalize(prev_hist, None).flatten()
        prev_edges = cv2.Canny(prev_gray, 50, 150)

        for pos in range(1, len(self.frame_indices)):
            idx = self.frame_indices[pos]
            curr_frame = self._get_resize_frame(idx)
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2GRAY)
            curr_hsv = cv2.cvtColor(curr_frame, cv2.COLOR_RGB2HSV)
            curr_v = float(np.mean(curr_hsv[:, :, 2]))

            ssim_score = float(self._safe_ssim(prev_frame, curr_frame))
            curr_hist = cv2.calcHist([curr_gray], [0], None, [32], [0, 256])
            curr_hist = cv2.normalize(curr_hist, None).flatten()
            chi_sq = float(0.5 * np.sum(((prev_hist - curr_hist) ** 2) / (prev_hist + curr_hist + 1e-6)))

            curr_edges = cv2.Canny(curr_gray, 50, 150)
            edge_diff = float(np.mean(cv2.absdiff(prev_edges, curr_edges) > 0))
            v_diff = float(abs(curr_v - prev_v))

            ssim_scores.append(ssim_score)
            chi_scores.append(chi_sq)
            edge_scores.append(edge_diff)
            vdiff_scores.append(v_diff)

            prev_frame = curr_frame
            prev_gray = curr_gray
            prev_hsv = curr_hsv
            prev_v = curr_v
            prev_hist = curr_hist
            prev_edges = curr_edges

        ssim_arr = np.asarray(ssim_scores, dtype=np.float32)
        chi_arr = np.asarray(chi_scores, dtype=np.float32)
        edge_arr = np.asarray(edge_scores, dtype=np.float32)
        vdiff_arr = np.asarray(vdiff_scores, dtype=np.float32)

        if ssim_arr.size == 0:
            return [0]

        # Robust per-video thresholds (MAD-based)
        k = float(self.shot_detect_k)
        ssim_med, chi_med, edge_med, vdiff_med = map(float, [np.median(ssim_arr), np.median(chi_arr), np.median(edge_arr), np.median(vdiff_arr)])
        ssim_thr = float(ssim_med - k * (_mad(ssim_arr) + 1e-9))
        chi_thr = float(chi_med + k * (_mad(chi_arr) + 1e-9))
        edge_thr = float(edge_med + k * (_mad(edge_arr) + 1e-9))
        vdiff_thr = float(vdiff_med + k * (_mad(vdiff_arr) + 1e-9))

        # Decision rule: a cut is a strong semantic/visual discontinuity vs local baseline
        cut_mask = (ssim_arr < ssim_thr) & ((chi_arr > chi_thr) | (edge_arr > edge_thr) | (vdiff_arr > vdiff_thr))
        cut_positions = (np.nonzero(cut_mask)[0] + 1).astype(np.int32)  # +1: transition i corresponds to boundary at pos i+1

        shot_positions = [0] + [int(x) for x in cut_positions.tolist()]

        # Объединяем слишком короткие шоты
        if len(shot_positions) <= 1:
            return [0]

        # shot boundaries in POSITIONS
        boundaries = sorted(set(int(x) for x in shot_positions))
        merged = [boundaries[0]]
        last_start = boundaries[0]
        for b in boundaries[1:]:
            dur_s = float(self.times_s[b] - self.times_s[last_start])
            if dur_s < self.min_shot_length_seconds:
                # не открываем новый шот, просто сливаем
                continue
            merged.append(b)
            last_start = b

        if not merged:
            merged = [0]
        return merged

    # -------------------------
    # Shot Features
    # -------------------------
    def extract_shot_features(self) -> Dict:
        boundaries_pos = sorted(set(int(x) for x in self.shot_boundaries))
        if not boundaries_pos:
            boundaries_pos = [0]
        # ensure start boundary exists
        if boundaries_pos[0] != 0:
            boundaries_pos = [0] + boundaries_pos
        bt = self.times_s[np.asarray(boundaries_pos, dtype=np.int32)]
        durations_sec = np.diff(np.concatenate([bt, np.asarray([self.times_s[-1]], dtype=np.float32)])).astype(np.float32)
        if durations_sec.size == 0:
            return {}

        # базовые статистики
        mean_dur = float(np.mean(durations_sec))
        med_dur = float(np.median(durations_sec))
        min_dur = float(np.min(durations_sec))
        max_dur = float(np.max(durations_sec))
        std_dur = float(np.std(durations_sec))

        hist_counts, _ = np.histogram(durations_sec, bins=20)
        dur_entropy = float(entropy(hist_counts + 1e-9))

        # gini
        gini = gini_coefficient(durations_sec)

        # нормализация длительности на длину видео
        norm_mean = float(mean_dur / max(self.video_length_seconds, 1e-6))

        # short_shot_fraction (<0.5 s)
        short_threshold = 0.5
        short_shot_fraction = float(
            np.mean(durations_sec < short_threshold) if durations_sec.size > 0 else 0.0
        )

        # quick_cut_burst_count: >=3 cut за 1 секунду (time-axis, union_timestamps_sec)
        cut_times = bt[1:].astype(np.float32)  # exclude t0
        quick_cut_burst_count = 0
        if cut_times.size >= 3:
            i = 0
            while i < len(cut_times):
                j = i + 1
                while j < len(cut_times) and cut_times[j] - cut_times[i] <= 1.0:
                    j += 1
                if j - i >= 3:
                    quick_cut_burst_count += 1
                i += 1

        # shot length histogram bins (very_short, short, medium, long, very_long)
        bins_sec = np.array([0.0, 0.3, 0.7, 1.5, 3.0, np.inf], dtype=np.float32)
        hist_counts_5, _ = np.histogram(durations_sec, bins=bins_sec)
        total_shots = float(hist_counts_5.sum()) if hist_counts_5.sum() > 0 else 1.0
        hist_fracs = (hist_counts_5 / total_shots).tolist()

        # tempo_entropy: энтропия распределения shot durations (по 5 бинам)
        tempo_entropy_val = float(entropy(hist_counts_5 + 1e-9))

        # cuts per 10 seconds (max/median over sliding 10s windows)
        window = 10.0
        if cut_times.size > 0 and self.video_length_seconds > 0:
            cut_rel = (cut_times - float(self.times_s[0])).astype(np.float32)
            t_edges = np.arange(0.0, self.video_length_seconds + window, window, dtype=np.float32)
            cuts_per_window, _ = np.histogram(cut_rel, bins=t_edges)
            cuts_per_10s_series = cuts_per_window / window
            cuts_per_10s_max = float(cuts_per_10s_series.max())
            cuts_per_10s_median = float(np.median(cuts_per_10s_series))
            cuts_per_10s_global = float(cuts_per_window.sum() / max(self.video_length_seconds / window, 1e-6))
        else:
            cuts_per_10s_series = np.array([0.0], dtype=np.float32)
            cuts_per_10s_max = 0.0
            cuts_per_10s_median = 0.0
            cuts_per_10s_global = 0.0

        # cut_density_map по 8 бинам времени
        if cut_times.size > 0 and self.video_length_seconds > 0:
            cut_rel = (cut_times - float(self.times_s[0])).astype(np.float32)
            bins8 = np.linspace(0.0, self.video_length_seconds, 9, dtype=np.float32)
            cuts8, _ = np.histogram(cut_rel, bins=bins8)
            cut_density_map = (cuts8 / max(self.video_length_seconds / 8.0, 1e-6)).tolist()
        else:
            cut_density_map = [0.0] * 8

        return {
            "shot_duration_mean": mean_dur,
            "shot_duration_median": med_dur,
            "shot_duration_min": min_dur,
            "shot_duration_max": max_dur,
            "shot_duration_std": std_dur,
            "shot_duration_entropy": dur_entropy,
            "shot_duration_mean_normalized": norm_mean,
            "shot_length_gini": gini,
            "cuts_per_10s": cuts_per_10s_global,
            "cuts_per_10s_max": cuts_per_10s_max,
            "cuts_per_10s_median": cuts_per_10s_median,
            "cuts_variance": float(np.var(durations_sec)),
            "short_shot_fraction": short_shot_fraction,
            "quick_cut_burst_count": int(quick_cut_burst_count),
            "shot_length_histogram_5bins": hist_fracs,
            "tempo_entropy": tempo_entropy_val,
            "cut_density_map_8bins": cut_density_map,
            "shots_count": int(durations_sec.size),
        }

    def extract_pace_curve(self) -> Dict:
        boundaries_pos = sorted(set(int(x) for x in self.shot_boundaries))
        if not boundaries_pos:
            boundaries_pos = [0]
        if boundaries_pos[0] != 0:
            boundaries_pos = [0] + boundaries_pos
        bt = self.times_s[np.asarray(boundaries_pos, dtype=np.int32)]
        durations_sec = np.diff(np.concatenate([bt, np.asarray([self.times_s[-1]], dtype=np.float32)])).astype(np.float32)
        if durations_sec.size == 0:
            return {}

        x = np.arange(len(durations_sec), dtype=np.float32)
        if len(durations_sec) >= 2:
            # простая регрессия (псевдо-робастность достигается за счёт клиппинга лог-длительностей)
            y = np.log1p(durations_sec)
            slope = float(np.polyfit(x, y, 1)[0])
        else:
            slope = 0.0
        pace_curve_slope = slope
        pace_curve_slope_normalized = float(slope * float(np.mean(durations_sec))) if float(np.mean(durations_sec)) > 0 else 0.0

        # пики (локальные максимумы длительностей = замедления)
        if len(durations_sec) >= 3:
            peaks, props = find_peaks(
                durations_sec,
                prominence=np.std(durations_sec) if np.std(durations_sec) > 0 else 0.0,
                distance=1,
            )
            pace_curve_peaks = int(len(peaks))
            peak_prom = props.get("prominences", np.zeros_like(peaks, dtype=np.float32))
            pace_curve_peaks_mean_prominence = float(peak_prom.mean()) if peak_prom.size else 0.0
            peak_positions = (peaks / max(len(durations_sec) - 1, 1)).astype(np.float32).tolist()
        else:
            pace_curve_peaks = 0
            pace_curve_peaks_mean_prominence = 0.0
            peak_positions = []

        # периодичность: по автокорреляции, вернуть период в секундах и мощность
        durations_centered = durations_sec - np.mean(durations_sec)
        autocorr = np.correlate(durations_centered, durations_centered, mode="full")
        mid = len(autocorr) // 2
        autocorr = autocorr[mid + 1 :]
        if autocorr.size > 0 and np.max(autocorr) > 0:
            autocorr_norm = autocorr / np.max(autocorr)
            best_lag = int(np.argmax(autocorr_norm) + 1)
            dominant_period_sec = float(best_lag * np.mean(durations_sec))
            power_at_period = float(autocorr_norm[best_lag - 1])
        else:
            dominant_period_sec = 0.0
            power_at_period = 0.0

        return {
            "pace_curve_slope": pace_curve_slope,
            "pace_curve_slope_normalized": pace_curve_slope_normalized,
            "pace_curve_peaks": pace_curve_peaks,
            "pace_curve_peaks_mean_prominence": pace_curve_peaks_mean_prominence,
            "pace_curve_peak_positions": peak_positions,
            "pace_curve_dominant_period_sec": dominant_period_sec,
            "pace_curve_power_at_period": power_at_period,
        }

    # -------------------------
    # Motion / Optical Flow
    # -------------------------
    def extract_motion_features(self) -> Dict:
        """
        Motion‑фичи. Используем только core_optical_flow (RAFT).
        """
        core = _load_core_optical_flow_npz(self.rs_path)
        if core is None:
            raise RuntimeError("video_pacing | core_optical_flow not found (required)")
        core_idx = core["frame_indices"]
        core_curve = core["motion_norm_per_sec_mean"]
        if core_curve is None or core_curve.size == 0:
            raise RuntimeError("video_pacing | core_optical_flow curve is empty")

        # Align to this module's frame_indices (union-domain). Must be fully covered.
        want = np.asarray(self.frame_indices, dtype=np.int32)
        mapping = {int(fi): i for i, fi in enumerate(core_idx.tolist())}
        pos = [mapping.get(int(fi), -1) for fi in want.tolist()]
        if any(p < 0 for p in pos):
            raise RuntimeError(
                "video_pacing | core_optical_flow.frame_indices does not cover this module's frame_indices. "
                "Segmenter must produce consistent sampling for dependent components."
            )
        core_curve = core_curve[np.asarray(pos, dtype=np.int64)]

        # лёгкое сглаживание (ignore NaNs from the first element)
        if np.isnan(core_curve[0]):
            core_curve = core_curve.copy()
            core_curve[0] = 0.0
        if core_curve.size >= 3:
            flow_mags_smooth = np.convolve(core_curve, np.ones(3, dtype=np.float32) / 3.0, mode="same")
        else:
            flow_mags_smooth = core_curve

        # пер-кадровые метрики
        mean_motion = float(np.mean(flow_mags_smooth))
        median_motion = float(np.median(flow_mags_smooth))
        var_motion = float(np.var(flow_mags_smooth))
        perc90_motion = float(np.percentile(flow_mags_smooth, 90))

        high_thr = float(np.percentile(flow_mags_smooth, 75))
        share_high_frames = float(np.mean(flow_mags_smooth > high_thr))

        # пер-шотовые motion-агрегаты и корреляция с длиной шота
        boundaries_pos = sorted(set(int(x) for x in self.shot_boundaries))
        if not boundaries_pos:
            boundaries_pos = [0]
        if boundaries_pos[0] != 0:
            boundaries_pos = [0] + boundaries_pos
        bt = self.times_s[np.asarray(boundaries_pos, dtype=np.int32)]
        durations_sec = np.diff(np.concatenate([bt, np.asarray([self.times_s[-1]], dtype=np.float32)])).astype(np.float32)

        shot_motion_means = []
        for i in range(len(boundaries_pos)):
            start = boundaries_pos[i]
            end = boundaries_pos[i + 1] if i + 1 < len(boundaries_pos) else int(self.total_frames)
            # core_optical_flow curve is per-frame; the first element is typically 0 (no previous frame).
            local = flow_mags_smooth[min(start + 1, flow_mags_smooth.size) : min(end, flow_mags_smooth.size)]
            if local.size > 0:
                shot_motion_means.append(float(np.mean(local)))
            else:
                shot_motion_means.append(0.0)

        motion_shot_corr = 0.0
        if len(shot_motion_means) == len(durations_sec) and len(durations_sec) > 1:
            x = np.asarray(durations_sec, dtype=np.float32)
            y = np.asarray(shot_motion_means, dtype=np.float32)
            if np.std(x) > 0 and np.std(y) > 0:
                motion_shot_corr = float(np.corrcoef(x, y)[0, 1])

        share_high_motion_shots = 0.0
        if shot_motion_means:
            thr_shot = float(np.percentile(shot_motion_means, 75))
            share_high_motion_shots = float(
                np.mean(np.asarray(shot_motion_means, dtype=np.float32) > thr_shot)
            )

        return {
            "mean_motion_speed_per_shot": mean_motion,
            "motion_speed_median": median_motion,
            "motion_speed_variance": var_motion,
            "motion_speed_90perc": perc90_motion,
            "share_of_high_motion_frames": share_high_frames,
            "share_of_high_motion_shots": share_high_motion_shots,
            "motion_shot_corr": motion_shot_corr,
        }

    def _get_clip_frame(self, idx):
        frame = self.frame_manager.get(idx)
        frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
        return frame

    # -------------------------
    # CLIP embeddings (batched)
    # -------------------------
    def extract_content_change_rate(self) -> Dict:
        # Используем только core_clip - обязательное требование
        core = _load_core_clip_npz(self.rs_path)
        if core is None:
            raise RuntimeError("video_pacing | core_clip not found (required)")
        frame_indices = core["frame_indices"]
        embeddings = core["frame_embeddings"]

        # Align to this module's frame_indices (union-domain). If not fully covered, treat as missing.
        want = np.asarray(self.frame_indices, dtype=np.int32)
        mapping = {int(fi): i for i, fi in enumerate(frame_indices.tolist())}
        pos = [mapping.get(int(fi), -1) for fi in want.tolist()]
        if any(p < 0 for p in pos):
            raise RuntimeError(
                "video_pacing | core_clip.frame_indices does not cover this module's frame_indices. "
                "Segmenter must produce consistent sampling for dependent components."
            )
        embeddings = embeddings[np.asarray(pos, dtype=np.int64)]

        # cosine distance между соседними эмбеддингами
        # нормируем эмбеддинги
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
        emb_norm = embeddings / norms
        cos_sim = np.sum(emb_norm[1:] * emb_norm[:-1], axis=1)
        cos_dist = 1.0 - cos_sim  # 0..2
        # Normalize by dt to be robust to variable sampling density.
        dt = np.diff(self.times_s).astype(np.float32)
        dt = np.maximum(dt, 1e-3)
        cos_rate = (cos_dist.astype(np.float32) / dt).astype(np.float32)

        # сглаживание
        if cos_rate.size >= 7:
            kernel_size = 5
            kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
            diff_smooth = np.convolve(cos_rate, kernel, mode="same")
        else:
            diff_smooth = cos_rate

        mean_diff = float(np.mean(diff_smooth))
        std_diff = float(np.std(diff_smooth))

        thr75 = float(np.percentile(diff_smooth, 75))
        high_change_ratio = float(np.mean(diff_smooth > thr75))

        thr_jump = mean_diff + 2.0 * std_diff
        scene_jumps = int(np.sum(diff_smooth > thr_jump))

        # semantic_change_burst_count: >=3 high-change transitions within any 5 seconds window
        high_mask = diff_smooth > thr_jump
        burst_count = 0
        if np.any(high_mask):
            trans_times = ((self.times_s[1:] + self.times_s[:-1]) * 0.5).astype(np.float32)
            high_times = trans_times[high_mask]
            window_s = 5.0
            i = 0
            while i < high_times.size:
                j = i + 1
                while j < high_times.size and float(high_times[j] - high_times[i]) <= window_s:
                    j += 1
                if j - i >= 3:
                    burst_count += 1
                i = j

        return {
            "frame_embedding_diff_mean": mean_diff,
            "frame_embedding_diff_std": std_diff,
            "high_change_frames_ratio": high_change_ratio,
            "scene_embedding_jumps": scene_jumps,
            "semantic_change_burst_count": int(burst_count),
        }

    # -------------------------
    # Color & Lighting Pacing
    # -------------------------
    def extract_color_pacing(self) -> Dict:
        hist_diffs = []
        if not self.frame_indices:
            return {}
        prev_frame = self._get_resize_frame(self.frame_indices[0])
        for idx in self.frame_indices[1:]:
            frame = self._get_resize_frame(idx)
            lab1 = rgb2lab(prev_frame)
            lab2 = rgb2lab(frame)
            deltaE = np.sqrt(np.sum((lab1-lab2)**2, axis=2))
            hist_diffs.append(np.mean(deltaE))
            prev_frame = frame
        hist_diffs = np.array(hist_diffs, dtype=np.float32)
        dt = np.diff(self.times_s).astype(np.float32)
        dt = np.maximum(dt, 1e-3)
        hist_rate = (hist_diffs / dt).astype(np.float32)

        # локальный baseline (скользящее среднее) для дельты цвета
        if hist_rate.size >= 7:
            kernel = np.ones(7, dtype=np.float32) / 7.0
            baseline = np.convolve(hist_rate, kernel, mode="same")
        else:
            baseline = np.full_like(hist_rate, float(np.mean(hist_rate)) if hist_rate.size else 0.0)
        hist_diffs_detrended = hist_rate - baseline

        saturation = np.asarray(
            [
            np.mean(cv2.cvtColor(self._get_resize_frame(idx), cv2.COLOR_RGB2HSV)[:, :, 1])
            for idx in self.frame_indices
            ],
            dtype=np.float32,
        )
        brightness = np.asarray(
            [
            np.mean(cv2.cvtColor(self._get_resize_frame(idx), cv2.COLOR_RGB2HSV)[:, :, 2])
            for idx in self.frame_indices
            ],
            dtype=np.float32,
        )

        # color_change_bursts по detrended DeltaE
        if hist_diffs_detrended.size > 0:
            thr_color = float(
                np.mean(hist_diffs_detrended) + 2.0 * np.std(hist_diffs_detrended)
            )
            peaks, _ = find_peaks(hist_diffs_detrended, height=thr_color, distance=1)
            color_change_bursts = int(len(peaks))
        else:
            color_change_bursts = 0

        sat_rate = np.diff(saturation) / dt if saturation.size >= 2 else np.asarray([], dtype=np.float32)
        bri_rate = np.diff(brightness) / dt if brightness.size >= 2 else np.asarray([], dtype=np.float32)

        return {
            "color_change_rate_mean": float(np.mean(hist_rate)) if hist_rate.size else 0.0,
            "color_change_rate_std": float(np.std(hist_rate)) if hist_rate.size else 0.0,
            "saturation_change_rate": float(np.std(sat_rate)) if sat_rate.size else 0.0,
            "brightness_change_rate": float(np.std(bri_rate)) if bri_rate.size else 0.0,
            "color_change_bursts": color_change_bursts,
        }

    def extract_lighting_pacing(self) -> Dict:
        lum = np.asarray(
            [
            np.mean(cv2.cvtColor(self._get_resize_frame(idx), cv2.COLOR_RGB2GRAY))
            for idx in self.frame_indices
            ],
            dtype=np.float32,
        )
        if lum.size < 2:
            return {
                "luminance_spikes_per_minute": 0.0,
            }

        dt = np.diff(self.times_s).astype(np.float32)
        dt = np.maximum(dt, 1e-3)
        lum_rate = np.diff(lum) / dt

        # Robust spike threshold (MAD-based). Avoid FFT because sampling is non-uniform.
        med = float(np.median(lum_rate))
        thr = float(abs(med) + 6.0 * (_mad(lum_rate) + 1e-9))
        spikes = np.abs(lum_rate - med) > thr
        spikes_count = int(np.sum(spikes))

        time_minutes = max(self.video_length_seconds / 60.0, 1e-6)
        lum_spikes_per_minute = float(spikes_count / time_minutes)

        return {
            "luminance_spikes_per_minute": lum_spikes_per_minute,
        }

    # -------------------------
    # Structural Pacing
    # -------------------------
    def extract_structural_pacing(self) -> Dict:
        boundaries_pos = sorted(set(int(x) for x in self.shot_boundaries))
        if not boundaries_pos:
            boundaries_pos = [0]
        if boundaries_pos[0] != 0:
            boundaries_pos = [0] + boundaries_pos
        bt = self.times_s[np.asarray(boundaries_pos, dtype=np.int32)]
        durations_sec = np.diff(np.concatenate([bt, np.asarray([self.times_s[-1]], dtype=np.float32)])).astype(np.float32)
        if durations_sec.size == 0:
            return {}
        n = len(durations_sec)
        quarter = max(n // 4, 1)
        intro = float(np.median(durations_sec[:quarter]))
        main = float(np.median(durations_sec[quarter : 3 * quarter]))
        climax = float(np.median(durations_sec[3 * quarter :]))
        overall_med = float(np.median(durations_sec))

        if overall_med > 0:
            pacing_symmetry = float((climax - intro) / overall_med)
        else:
            pacing_symmetry = 0.0

        return {
            "intro_speed": intro,
            "main_speed": main,
            "climax_speed": climax,
            "pacing_symmetry": pacing_symmetry,
        }

    # -------------------------
    # Full Pipeline
    # -------------------------
    def extract_all_features(self) -> Dict:
        features = {}
        # Основные визуальные метрики
        features.update(self.extract_shot_features())
        features.update(self.extract_pace_curve())
        # Hard deps: core_optical_flow + core_clip (contract)
        features.update(self.extract_motion_features())
        features.update(self.extract_content_change_rate())
        features.update(self.extract_color_pacing())
        features.update(self.extract_lighting_pacing())
        features.update(self.extract_structural_pacing())

        # Note: AV sync / per-person / object pacing can be added later as separate optional blocks.

        return features


class VideoPacingModule(BaseModule):
    """
    BaseModule wrapper for `video_pacing`.

    Контракты:
    - `frame_indices` приходят только из metadata (Segmenter).
    - Кадры в FrameManager — RGB.
    - Core providers:
      - `core_optical_flow` (motion curve)
      - `core_clip` (semantic content-change rate)
    """

    @property
    def module_name(self) -> str:
        return "video_pacing"

    def __init__(self, rs_path: Optional[str] = None, downscale_factor: float = 0.25, **kwargs: Any):
        super().__init__(rs_path=rs_path, logger_name=self.module_name, **kwargs)
        self._downscale_factor = float(downscale_factor)

    def required_dependencies(self) -> List[str]:
        return ["core_optical_flow", "core_clip"]

    def process(self, frame_manager: FrameManager, frame_indices: List[int], config: Dict[str, Any]) -> Dict[str, Any]:
        if not frame_indices:
            raise ValueError("video_pacing | frame_indices is empty")

        downscale = float(config.get("downscale_factor", self._downscale_factor))
        min_shot_len_s = float(config.get("min_shot_length_seconds", 0.15))
        shot_detect_k = float(config.get("shot_detect_k", 6.0))

        pipeline = VideoPacingPipelineVisualOptimized(
            frame_manager=frame_manager,
            frame_indices=frame_indices,
            downscale_factor=downscale,
            min_shot_length_seconds=min_shot_len_s,
            shot_detect_k=shot_detect_k,
            rs_path=self.rs_path,
        )

        raw_features = pipeline.extract_all_features()

        frame_indices_np = np.asarray([int(i) for i in frame_indices], dtype=np.int32)
        # boundaries are stored as POSITIONS inside pipeline; export as UNION frame indices
        shot_boundaries_pos = np.asarray(pipeline.shot_boundaries, dtype=np.int32)
        if shot_boundaries_pos.size:
            shot_boundaries_frame_indices = frame_indices_np[
                np.clip(shot_boundaries_pos, 0, frame_indices_np.size - 1)
            ]
        else:
            shot_boundaries_frame_indices = np.asarray([], dtype=np.int32)

        return {
            "frame_indices": frame_indices_np,
            "shot_boundary_frame_indices": shot_boundaries_frame_indices,
            "features": raw_features,
        }