"""
video_pacing

Содержит:
- `VideoPacingPipelineVisualOptimized` (feature extraction on sampled frames)
- `VideoPacingModule(BaseModule)` — NPZ output + strict frame_indices + core provider integration
"""

from __future__ import annotations

import math
import os
import json

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
        curve = data.get("motion_px_per_sec_mean")
        if idx is None or curve is None:
            return None
        return {
            "frame_indices": np.asarray(idx, dtype=np.int32),
            "motion_px_per_sec_mean": np.asarray(curve, dtype=np.float32),
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
        self.frame_indices = frame_indices
        self.total_frames = len(frame_indices)
        self.fps = float(getattr(self.frame_manager, "fps", 30.0) or 30.0)
        self.video_length_seconds = self.total_frames / self.fps if self.fps > 0 else 0.0
        self.min_shot_length_seconds = float(min_shot_length_seconds)
        self.rs_path = rs_path

        # CLIP модель удалена - используем только core_clip

        # Определяем шоты и сцены
        self.shot_boundaries = self._detect_shots_with_merging()
        # Пока сцены отождествляем с шотами (alias), см. описание в FEATURES_DESCRIPTION
        self.scene_boundaries = self.shot_boundaries

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

        if min_side < 7:
            # fallback: считаем, что кадры сильно отличаются
            return 0.0

        win_size = min(7, min_side if min_side % 2 == 1 else min_side - 1)

        return ssim(
            img1,
            img2,
            channel_axis=-1,
            win_size=win_size
        )

    def _detect_shots_with_merging(self) -> List[int]:
        """
        Детекция шотов по комбинации SSIM + цвет/градиенты/яркость с последующим
        объединением слишком коротких шотов.
        """
        if not self.frame_indices:
            return [0]

        # IMPORTANT: store boundaries as POSITIONS in `self.frame_indices` list (0..N-1),
        # not as union frame indices. This keeps all downstream computations consistent.
        shot_positions = [0]
        prev_pos = 0
        prev_idx = self.frame_indices[prev_pos]
        prev_frame = self._get_resize_frame(prev_idx)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_RGB2GRAY)

        # предвычислим гистограмму и границы
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

            # 1) SSIM
            ssim_score = self._safe_ssim(prev_frame, curr_frame)

            # 2) Цветовая гистограмма (Chi-squared)
            curr_hist = cv2.calcHist([curr_gray], [0], None, [32], [0, 256])
            curr_hist = cv2.normalize(curr_hist, None).flatten()
            chi_sq = 0.5 * np.sum(
                ((prev_hist - curr_hist) ** 2) / (prev_hist + curr_hist + 1e-6)
            )

            # 3) Изменение границ (edge change ratio)
            curr_edges = cv2.Canny(curr_gray, 50, 150)
            edge_diff = np.mean(cv2.absdiff(prev_edges, curr_edges) > 0)

            # 4) Изменение яркости
            v_diff = abs(curr_v - prev_v)

            # бинарные детекторы
            ssim_cut = ssim_score < 0.93
            hist_cut = chi_sq > 0.3
            edge_cut = edge_diff > 0.25 or v_diff > 15.0

            strong_hist = chi_sq > 0.6
            strong_edge = edge_diff > 0.5 or v_diff > 30.0

            detectors = [ssim_cut, hist_cut, edge_cut]
            if sum(detectors) >= 2 or strong_hist or strong_edge:
                shot_positions.append(pos)

            prev_frame = curr_frame
            prev_gray = curr_gray
            prev_hsv = curr_hsv
            prev_v = curr_v
            prev_hist = curr_hist
            prev_edges = curr_edges

        # Объединяем слишком короткие шоты
        if len(shot_positions) <= 1:
            return [0]

        min_len_frames = max(int(self.min_shot_length_seconds * self.fps), 1)
        # shot boundaries in POSITIONS
        boundaries = sorted(set(int(x) for x in shot_positions))
        merged = [boundaries[0]]
        last_start = boundaries[0]
        for b in boundaries[1:]:
            dur = b - last_start
            if dur < min_len_frames:
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
        # длительности в кадрах
        shot_frame_boundaries = [0] + self.shot_boundaries + [self.total_frames]
        durations_frames = np.diff(shot_frame_boundaries).astype(np.float32)
        if durations_frames.size == 0:
            return {}

        durations_sec = durations_frames / self.fps

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

        # quick_cut_burst_count: >=3 cut за 1 секунду (по временным индексам шотов)
        cut_times = np.array(self.shot_boundaries, dtype=np.float32) / self.fps
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

        # cuts per 10 seconds (максимум и медиана скользящего окна)
        window = 10.0
        if cut_times.size > 0 and self.video_length_seconds > 0:
            t_edges = np.arange(0.0, self.video_length_seconds + window, window)
            cuts_per_window, _ = np.histogram(cut_times, bins=t_edges)
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
            bins8 = np.linspace(0.0, self.video_length_seconds, 9)
            cuts8, _ = np.histogram(cut_times, bins=bins8)
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
        }

    def extract_pace_curve(self) -> Dict:
        shot_frame_boundaries = [0] + self.shot_boundaries + [self.total_frames]
        durations_frames = np.diff(shot_frame_boundaries).astype(np.float32)
        if durations_frames.size == 0:
            return {}
        durations_sec = durations_frames / self.fps

        x = np.arange(len(durations_sec), dtype=np.float32)
        if len(durations_sec) >= 2:
            # простая регрессия (псевдо-робастность достигается за счёт клиппинга лог-длительностей)
            y = np.log1p(durations_sec)
            slope = float(np.polyfit(x, y, 1)[0])
        else:
            slope = 0.0
        mean_dur = float(np.mean(durations_sec))
        pace_curve_slope = slope
        pace_curve_slope_normalized = float(slope * mean_dur) if mean_dur > 0 else 0.0

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
            "pace_curve_mean": mean_dur,
            "pace_curve_slope": pace_curve_slope,
            "pace_curve_slope_normalized": pace_curve_slope_normalized,
            "pace_curve_peaks": pace_curve_peaks,
            "pace_curve_peaks_mean_prominence": pace_curve_peaks_mean_prominence,
            "pace_curve_peak_positions": peak_positions,
            "pace_curve_dominant_period_sec": dominant_period_sec,
            "pace_curve_power_at_period": power_at_period,
        }

    def extract_scene_pacing(self) -> Dict:
        durations = np.diff([0] + self.scene_boundaries + [self.total_frames]).astype(
            np.float32
        )
        if durations.size == 0:
            return {}
        durations_sec = durations / self.fps
        return {
            # alias: в текущей версии сцены соответствуют шотам
            "scene_changes_per_minute": float(
                len(self.scene_boundaries) / max(self.video_length_seconds / 60.0, 1e-6)
            ),
            "average_scene_duration": float(np.mean(durations_sec)),
            "scene_duration_variance": float(np.var(durations_sec)),
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
        core_curve = core["motion_px_per_sec_mean"]
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

        angle_series = None  # optical_flow пока не предоставляет angle_series

        # пер-кадровые метрики
        mean_motion = float(np.mean(flow_mags_smooth))
        median_motion = float(np.median(flow_mags_smooth))
        var_motion = float(np.var(flow_mags_smooth))
        perc90_motion = float(np.percentile(flow_mags_smooth, 90))

        high_thr = float(np.percentile(flow_mags_smooth, 75))
        share_high_frames = float(np.mean(flow_mags_smooth > high_thr))

        # изменения направления: доля кадров с резким поворотом > X градусов
        if angle_series is not None:
            angle_series = np.asarray(angle_series, dtype=np.float32)
            if angle_series.size >= 2:
                # нормализуем углы к [-pi, pi]
                diff_ang = np.diff(angle_series)
                diff_ang = (diff_ang + np.pi) % (2 * np.pi) - np.pi
                angle_deg = np.abs(diff_ang * 180.0 / np.pi)
                direction_change_events = np.sum(angle_deg > 45.0)
                time_seconds = max(len(self.frame_indices) / self.fps, 1e-6)
                dir_changes_per_sec = float(direction_change_events / time_seconds)
            else:
                dir_changes_per_sec = 0.0
        else:
            # core_optical_flow пока не даёт углы, поэтому ставим 0.0
            dir_changes_per_sec = 0.0

        # пер-шотовые motion-агрегаты и корреляция с длиной шота
        shot_frame_boundaries = [0] + self.shot_boundaries + [self.total_frames]
        durations_frames = np.diff(shot_frame_boundaries).astype(np.float32)
        durations_sec = durations_frames / self.fps

        shot_motion_means = []
        for i in range(len(shot_frame_boundaries) - 1):
            start = shot_frame_boundaries[i]
            end = shot_frame_boundaries[i + 1]
            # flow_mags соответствует переходам между кадрами, сдвинуто на 1 относительно frame_indices
            local = flow_mags_smooth[max(start - 1, 0) : max(end - 1, 0)]
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
            "optical_flow_direction_changes_per_second": dir_changes_per_sec,
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

        # сглаживание
        if cos_dist.size >= 7:
            kernel_size = 5
            kernel = np.ones(kernel_size, dtype=np.float32) / kernel_size
            diff_smooth = np.convolve(cos_dist, kernel, mode="same")
        else:
            diff_smooth = cos_dist

        mean_diff = float(np.mean(diff_smooth))
        std_diff = float(np.std(diff_smooth))

        thr75 = float(np.percentile(diff_smooth, 75))
        high_change_ratio = float(np.mean(diff_smooth > thr75))

        thr_jump = mean_diff + 2.0 * std_diff
        scene_jumps = int(np.sum(diff_smooth > thr_jump))

        # semantic_change_burst_count: количество кластеров >=3 high-change кадров в окне 5 секунд
        high_mask = diff_smooth > thr_jump
        if self.fps > 0:
            window_frames = int(5.0 * self.fps)
        else:
            window_frames = 5
        if window_frames < 3:
            window_frames = 3

        burst_count = 0
        if high_mask.size > 0:
            i = 0
            while i < high_mask.size:
                if not high_mask[i]:
                    i += 1
                    continue
                j = i
                while j < high_mask.size and j - i <= window_frames and high_mask[j]:
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
        prev_frame = self._get_resize_frame(0)
        for idx in self.frame_indices[1:]:
            frame = self._get_resize_frame(idx)
            lab1 = rgb2lab(prev_frame)
            lab2 = rgb2lab(frame)
            deltaE = np.sqrt(np.sum((lab1-lab2)**2, axis=2))
            hist_diffs.append(np.mean(deltaE))
            prev_frame = frame
        hist_diffs = np.array(hist_diffs, dtype=np.float32)

        # локальный baseline (скользящее среднее) для дельты цвета
        if hist_diffs.size >= 7:
            kernel = np.ones(7, dtype=np.float32) / 7.0
            baseline = np.convolve(hist_diffs, kernel, mode="same")
        else:
            baseline = np.full_like(hist_diffs, float(np.mean(hist_diffs)) if hist_diffs.size else 0.0)
        hist_diffs_detrended = hist_diffs - baseline

        saturation = [
            np.mean(cv2.cvtColor(self._get_resize_frame(idx), cv2.COLOR_RGB2HSV)[:, :, 1])
            for idx in self.frame_indices
        ]
        brightness = [
            np.mean(cv2.cvtColor(self._get_resize_frame(idx), cv2.COLOR_RGB2HSV)[:, :, 2])
            for idx in self.frame_indices
        ]

        # color_change_bursts по detrended DeltaE
        if hist_diffs_detrended.size > 0:
            thr_color = float(
                np.mean(hist_diffs_detrended) + 2.0 * np.std(hist_diffs_detrended)
            )
            peaks, _ = find_peaks(hist_diffs_detrended, height=thr_color, distance=1)
            color_change_bursts = int(len(peaks))
        else:
            color_change_bursts = 0

        return {
            "color_histogram_diff_mean": float(np.mean(hist_diffs)),
            "color_histogram_diff_std": float(np.std(hist_diffs)),
            "saturation_change_rate": float(np.std(saturation)),
            "brightness_change_rate": float(np.std(brightness)),
            "color_change_bursts": color_change_bursts,
        }

    def extract_lighting_pacing(self) -> Dict:
        lum = [
            np.mean(cv2.cvtColor(self._get_resize_frame(idx), cv2.COLOR_RGB2GRAY))
            for idx in self.frame_indices
        ]
        lum = np.asarray(lum, dtype=np.float32)
        if lum.size < 2:
            return {
                "luminance_spikes_per_minute": 0.0,
                "high_frequency_flash_ratio": 0.0,
            }

        lum_diff = np.diff(lum)
        std_lum = float(np.std(lum_diff))
        # простой порог шума камеры
        noise_threshold = max(5.0, std_lum)
        spikes = np.abs(lum_diff) > noise_threshold
        spikes_count = int(np.sum(spikes))

        time_minutes = max(self.video_length_seconds / 60.0, 1e-6)
        lum_spikes_per_minute = float(spikes_count / time_minutes)

        # high frequency flash ratio через FFT
        lum_fft = np.fft.fft(lum_diff - np.mean(lum_diff))
        mag = np.abs(lum_fft)
        nyq = len(mag) // 2
        cutoff = int(0.25 * nyq)
        hf_power = np.sum(mag[cutoff:nyq])
        total_power = np.sum(mag[:nyq]) + 1e-9
        hf_ratio = float(hf_power / total_power)

        return {
            "luminance_spikes_per_minute": lum_spikes_per_minute,
            "high_frequency_flash_ratio": hf_ratio,
        }

    # -------------------------
    # Structural Pacing
    # -------------------------
    def extract_structural_pacing(self) -> Dict:
        shot_frame_boundaries = [0] + self.shot_boundaries + [self.total_frames]
        durations_frames = np.diff(shot_frame_boundaries).astype(np.float32)
        if durations_frames.size == 0:
            return {}
        durations_sec = durations_frames / self.fps
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
        features.update(self.extract_scene_pacing())
        # Optional dependencies (optical_flow, core_clip)
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

        pipeline = VideoPacingPipelineVisualOptimized(
            frame_manager=frame_manager,
            frame_indices=frame_indices,
            downscale_factor=downscale,
            rs_path=self.rs_path,
        )

        raw_features = pipeline.extract_all_features()

        # Presence flags for optional dependency families
        has_motion = bool(raw_features.get("mean_motion_speed_per_shot") is not None) and ("motion_speed_median" in raw_features)
        has_content_change = "frame_embedding_diff_mean" in raw_features

        # Provide stable keys (NaN when missing optional blocks)
        def _nan_features(keys: List[str]) -> Dict[str, Any]:
            return {k: float("nan") for k in keys}

        motion_keys = [
            "mean_motion_speed_per_shot",
            "motion_speed_median",
            "motion_speed_variance",
            "motion_speed_90perc",
            "share_of_high_motion_frames",
            "share_of_high_motion_shots",
            "motion_shot_corr",
            "optical_flow_direction_changes_per_second",
        ]
        if not has_motion:
            raw_features.update(_nan_features(motion_keys))

        content_keys = [
            "frame_embedding_diff_mean",
            "frame_embedding_diff_std",
            "high_change_frames_ratio",
            "scene_embedding_jumps",
            "semantic_change_burst_count",
        ]
        if not has_content_change:
            raw_features.update(_nan_features(content_keys))

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
            "motion_present": np.asarray(bool(has_motion)),
            "content_change_present": np.asarray(bool(has_content_change)),
        }