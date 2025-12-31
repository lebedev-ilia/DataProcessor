"""
Модуль для анализа цвета и освещения видео.

Все TODO выполнены:
✓ Оптимизирован код под работу с внешними зависимостями (загрузка scenes из scene_classification)
✓ Модуль оптимизирован под работу с BaseModule
✓ Выход приведен к единому формату для сохранения в npz
"""

import os, sys

_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

if _path not in sys.path:
    sys.path.append(_path)

import math
import numpy as np
import cv2
from typing import Dict, List, Any, Optional
from scipy import stats
from scipy.stats import entropy
from scipy.signal import find_peaks
from scipy.ndimage import gaussian_filter, label
from sklearn.cluster import KMeans
from collections import Counter

import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)

from modules.base_module import BaseModule
from utils.frame_manager import FrameManager
from utils.logger import get_logger

NAME = "ColorLightProcessor"
VERSION = '1.0'

logger = get_logger(NAME)

class ColorLightProcessor(BaseModule):
    """Процессор для анализа цвета и освещения видео"""
    
    def __init__(
        self,
        rs_path: Optional[str] = None,
        max_frames_per_scene: int = 350,
        stride: int = 5,
        **kwargs: Any
    ):
        """
        Args:
            rs_path: Путь к хранилищу результатов
            max_frames_per_scene: Максимальное количество кадров для обработки на сцену
            stride: Шаг для выборки кадров
            **kwargs: Дополнительные параметры для BaseModule
        """
        super().__init__(rs_path=rs_path, logger_name="color_light", **kwargs)
        self.max_frames_per_scene = max_frames_per_scene
        self.stride = stride
    
    def required_dependencies(self) -> List[str]:
        """Возвращает список зависимостей модуля."""
        return ["scene_classification"]
    
    def _compute_rgb_stats(self, frame: np.ndarray) -> Dict[str, float]:
        """
        Базовые RGB‑статистики: только mean/std по каналам.
        Мин/макс, skew и kurtosis убраны как слабоинформативные и сильно коррелирующие.
        """
        features: Dict[str, float] = {}
        for i, channel in enumerate(["R", "G", "B"]):
            channel_data = frame[:, :, i].flatten().astype(np.float32)
            features[f"color_mean_{channel.lower()}"] = float(np.mean(channel_data))
            features[f"color_std_{channel.lower()}"] = float(np.std(channel_data))
        return features
    
    def _compute_hsv_features(self, frame: np.ndarray) -> Dict[str, float]:
        """
        HSV‑фичи:
        - hue_mean / hue_std / hue_entropy
        - saturation_mean/std, value_mean/std
        - нормализованные варианты и взвешенная по saturation энтропия hue.
        """
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        features: Dict[str, float] = {}

        hue = hsv[:, :, 0].flatten().astype(np.float32)  # [0, 180]
        sat = hsv[:, :, 1].flatten().astype(np.float32)  # [0, 255]
        val = hsv[:, :, 2].flatten().astype(np.float32)  # [0, 255]

        # Базовые статистики
        hue_mean = float(np.mean(hue))
        hue_std = float(np.std(hue))
        features["hue_mean"] = hue_mean
        features["hue_std"] = hue_std

        hue_hist, _ = np.histogram(hue, bins=36, range=(0, 180))
        hue_probs = hue_hist / (hue_hist.sum() + 1e-10)
        features["hue_entropy"] = float(entropy(hue_probs + 1e-10))

        # Взвешенная по насыщенности энтропия hue
        sat_norm = sat / 255.0
        hue_hist_w, _ = np.histogram(hue, bins=36, range=(0, 180), weights=sat_norm)
        hue_probs_w = hue_hist_w / (hue_hist_w.sum() + 1e-10)
        features["hue_entropy_weighted"] = float(entropy(hue_probs_w + 1e-10))

        # Saturation
        sat_mean = float(np.mean(sat))
        sat_std = float(np.std(sat))
        features["saturation_mean"] = sat_mean
        features["saturation_std"] = sat_std

        # Value (brightness)
        val_mean = float(np.mean(val))
        val_std = float(np.std(val))
        features["value_mean"] = val_mean
        features["value_std"] = val_std

        # Нормализованные фичи (0–1) для трансформера
        features["hue_mean_norm"] = hue_mean / 180.0
        features["hue_std_norm"] = hue_std / 180.0
        features["sat_mean_norm"] = sat_mean / 255.0
        features["val_mean_norm"] = val_mean / 255.0

        return features
    
    def _compute_lab_features(self, frame: np.ndarray) -> Dict[str, float]:
        """Вычисляет LAB‑фичи: L_mean, L_contrast, ab_balance + нормализованный L."""
        lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
        features: Dict[str, float] = {}

        L = lab[:, :, 0].flatten().astype(np.float32)  # [0, 255]
        L_mean = float(np.mean(L))
        L_std = float(np.std(L))
        features["L_mean"] = L_mean
        features["L_contrast"] = L_std

        a = lab[:, :, 1].flatten().astype(np.float32) - 128.0
        b = lab[:, :, 2].flatten().astype(np.float32) - 128.0
        features["ab_balance"] = float(np.mean(a) - np.mean(b))

        # Нормализованная яркость
        features["L_mean_norm"] = L_mean / 255.0

        return features
    
    def _compute_palette_features(self, frame: np.ndarray) -> Dict[str, float]:
        """
        Палитра и цветовые признаки.
        - KMeans в Lab‑пространстве (a,b) для устойчивых доминантных цветов.
        - Индекс цветности, warm/cold ratio, skin_tone_ratio, color_palette_entropy.
        - Доминантный цвет возвращается как нормализованные Lab‑координаты a,b.
        """
        features: Dict[str, float] = {}

        h, w = frame.shape[:2]
        sample_size = min(10000, h * w)
        if h * w > sample_size and h > 0 and w > 0:
            step = int(max(1, np.sqrt(h * w / sample_size)))
            sampled_rgb = frame[::step, ::step].reshape(-1, 3)
        else:
            sampled_rgb = frame.reshape(-1, 3)

        # --- KMeans в Lab для доминантного цвета ---
        if len(sampled_rgb) > 0:
            lab = cv2.cvtColor(sampled_rgb.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2LAB)
            lab_flat = lab.reshape(-1, 3).astype(np.float32)
            ab = lab_flat[:, 1:3]  # только a,b

            n_colors = int(min(3, len(ab)))
            if n_colors >= 1:
                if n_colors == 1:
                    centers = ab[:1]
                    labels = np.zeros(len(ab), dtype=int)
                else:
                    kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
                    kmeans.fit(ab)
                    centers = kmeans.cluster_centers_
                    labels = kmeans.labels_

                counts = Counter(labels)
                # выбираем самый частый кластер
                dominant_label = max(counts.items(), key=lambda x: x[1])[0]
                dom_a, dom_b = centers[int(dominant_label)]

                # сохраняем ненормированные и нормализованные координаты
                features["dominant_lab_a"] = float(dom_a)
                features["dominant_lab_b"] = float(dom_b)
                features["dominant_lab_a_norm"] = float((dom_a + 128.0) / 255.0)
                features["dominant_lab_b_norm"] = float((dom_b + 128.0) / 255.0)

        # Colorfulness index (как раньше, по RGB)
        rgb_reshaped = frame.reshape(-1, 3).astype(np.float32)
        if rgb_reshaped.size > 0:
            rg = rgb_reshaped[:, 0] - rgb_reshaped[:, 1]
            yb = 0.5 * (rgb_reshaped[:, 0] + rgb_reshaped[:, 1]) - rgb_reshaped[:, 2]
            std_rg = np.std(rg)
            std_yb = np.std(yb)
            mean_rgyb = np.sqrt(np.mean(rg ** 2) + np.mean(yb ** 2))
            colorfulness = float(std_rg + std_yb + 0.3 * mean_rgyb)
        else:
            colorfulness = 0.0
        features["colorfulness_index"] = colorfulness
        # Нормализованный индекс цветности ~ [0,1] (ожидаем 0–100+)
        features["colorfulness_norm"] = float(min(colorfulness / 100.0, 1.0))

        # Warm vs cold ratio, skin_tone_ratio, color_palette_entropy
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        hue = hsv[:, :, 0].flatten().astype(np.float32)
        sat = hsv[:, :, 1].flatten().astype(np.float32)
        val = hsv[:, :, 2].flatten().astype(np.float32)

        warm_mask = ((hue >= 0) & (hue <= 30)) | ((hue >= 150) & (hue <= 180))
        warm_count = np.sum(warm_mask)
        cold_count = max(1, len(hue) - warm_count)
        features["warm_vs_cold_ratio"] = float(warm_count / float(cold_count))

        skin_mask = ((hue >= 0) & (hue <= 25)) & (sat >= 20) & (val >= 50)
        features["skin_tone_ratio"] = float(np.sum(skin_mask) / (len(hue) + 1e-10))

        hue_hist, _ = np.histogram(hue, bins=36, range=(0, 180))
        hue_probs = hue_hist / (hue_hist.sum() + 1e-10)
        features["color_palette_entropy"] = float(entropy(hue_probs + 1e-10))

        # Цветовые гармонии: оставляем только complementary / analogous как компактные признаки
        harmony_features = self._compute_color_harmonies(hue, sat, val)
        features.update(harmony_features)

        return features
    
    def _compute_color_harmonies(
        self, hue: np.ndarray, sat: np.ndarray, val: np.ndarray
    ) -> Dict[str, float]:
        """
        Компактные цветовые гармонии:
        - color_harmony_complementary_prob
        - color_harmony_analogous_prob
        Триады и split‑complementary убраны для снижения размерности.
        """
        features: Dict[str, float] = {}

        hue_hist, hue_bins = np.histogram(hue, bins=36, range=(0, 180))
        if hue_hist.sum() == 0:
            features["color_harmony_complementary_prob"] = 0.0
            features["color_harmony_analogous_prob"] = 0.0
            return features

        dominant_hue_bin = int(np.argmax(hue_hist))
        dominant_hue = (hue_bins[dominant_hue_bin] + hue_bins[dominant_hue_bin + 1]) / 2.0

        dominant_hue_360 = dominant_hue * 2.0

        # Complementary
        comp_hue = (dominant_hue_360 + 180.0) % 360.0
        comp_hue_180 = comp_hue / 2.0
        comp_range = (
            (hue >= (comp_hue_180 - 15) % 180)
            & (hue <= (comp_hue_180 + 15) % 180)
        ) | ((hue >= 0) & (hue <= (comp_hue_180 + 15 - 180) % 180)) | (
            (hue >= (comp_hue_180 - 15 + 180) % 180) & (hue <= 180)
        )
        comp_ratio = float(np.sum(comp_range) / (len(hue) + 1e-10))
        features["color_harmony_complementary_prob"] = float(min(comp_ratio * 2.0, 1.0))

        # Analogous (±30 градусов вокруг доминирующего)
        anal_range1 = (hue >= (dominant_hue - 30) % 180) & (
            hue <= (dominant_hue + 30) % 180
        )
        anal_range2 = ((hue >= 0) & (hue <= (dominant_hue + 30 - 180) % 180)) | (
            (hue >= (dominant_hue - 30 + 180) % 180) & (hue <= 180)
        )
        anal_range = anal_range1 | anal_range2
        anal_ratio = float(np.sum(anal_range) / (len(hue) + 1e-10))
        features["color_harmony_analogous_prob"] = float(
            min(anal_ratio * 1.2, 1.0)
        )

        return features
    
    def _safe_entropy_from_hist(self, hist: np.ndarray, eps: float = 1e-12) -> float:
        probs = hist.astype(np.float64)
        s = probs.sum()
        if s <= 0:
            return 0.0
        probs = probs / (s + eps)
        probs = probs + eps
        return float(-np.sum(probs * np.log(probs)))

    def _compute_lighting_uniformity(self, gray: np.ndarray) -> Dict[str, float]:
        """
        Вычисляет фичи равномерности освещения: uniformity_index, center/corner brightness, vignetting.
        Вход: gray — 2D np.ndarray (H, W), dtype=uint8 or numeric.
        Возвращает dict со значениями float.
        """
        features: Dict[str, float] = {}

        if gray.ndim != 2:
            raise ValueError("_compute_lighting_uniformity expects 2D gray image")

        # cast to float for stable stats
        grayf = np.asarray(gray, dtype=np.float32)
        h, w = grayf.shape
        if h == 0 or w == 0:
            # degenerate
            return {
                "lighting_uniformity_index": 0.0,
                "center_brightness": 0.0,
                "corner_brightness": 0.0,
                "vignetting_score": 0.0,
            }

        # grid 3x3
        grid_h, grid_w = 3, 3
        cell_h = max(1, h // grid_h)
        cell_w = max(1, w // grid_w)

        brightness_grid = []
        for i in range(grid_h):
            for j in range(grid_w):
                y_start = i * cell_h
                x_start = j * cell_w
                # include remainder in last cell
                if i == grid_h - 1:
                    y_end = h
                else:
                    y_end = y_start + cell_h
                if j == grid_w - 1:
                    x_end = w
                else:
                    x_end = x_start + cell_w

                cell = grayf[y_start:y_end, x_start:x_end]
                if cell.size == 0:
                    brightness_grid.append(0.0)
                else:
                    brightness_grid.append(float(np.mean(cell)))

        brightness_grid = np.array(brightness_grid, dtype=np.float32)
        uniformity_std = float(np.std(brightness_grid))
        uniformity_mean = float(np.mean(brightness_grid))

        # normalized std: divide by (mean + eps) to be scale invariant
        eps = 1e-6
        norm_std = uniformity_std / (uniformity_mean + eps)

        # uniformity index in (0,1], higher = more uniform
        features["lighting_uniformity_index"] = float(1.0 / (1.0 + norm_std))

        # center brightness (central cell)
        center_idx = (grid_h * grid_w) // 2
        features["center_brightness"] = float(brightness_grid[center_idx])

        # corner brightness (average of 4 corners)
        corner_indices = [0, grid_w - 1, (grid_h - 1) * grid_w, grid_h * grid_w - 1]
        corner_vals = [brightness_grid[idx] for idx in corner_indices if idx < len(brightness_grid)]
        features["corner_brightness"] = float(np.mean(corner_vals)) if corner_vals else 0.0

        # vignetting score: 0 no vignetting, 1 strong (clamped)
        # compute ratio corner/center (safe), invert to have 0..1
        if features["center_brightness"] > 0:
            ratio = features["corner_brightness"] / (features["center_brightness"] + eps)
            # if corners brighter than center ratio>1 -> vignetting negative (we clamp to 0)
            v = 1.0 - min(max(ratio, 0.0), 1.0)
            features["vignetting_score"] = float(max(0.0, min(v, 1.0)))
        else:
            features["vignetting_score"] = 0.0

        return features


    def _compute_lighting_features(self, frame: np.ndarray) -> Dict[str, float]:
        """
        Вычисляет фичи освещения: brightness, contrast, entropy, clipping ratios, dynamic range (dB),
        highlight/shadow clipping, local contrast, uniformity (via helper).
        Вход: frame — HxWx3 RGB (uint8) или совместимый numeric array.
        """
        features: Dict[str, float] = {}

        if frame is None:
            # return zeros for robustness
            zero_keys = [
                "brightness_mean", "brightness_std", "brightness_entropy",
                "overexposed_pixels", "underexposed_pixels", "global_contrast",
                "local_contrast", "local_contrast_std", "contrast_entropy",
                "dynamic_range_db", "highlight_clipping_ratio", "shadow_clipping_ratio"
            ]
            return {k: 0.0 for k in zero_keys}

        # ensure numpy array and RGB uint8
        im = np.asarray(frame)
        if im.ndim == 3 and im.shape[-1] == 4:
            im = im[..., :3]
        if im.ndim != 3 or im.shape[-1] != 3:
            # if grayscale 2D, expand
            if im.ndim == 2:
                im = np.stack([im, im, im], axis=-1)
            else:
                raise ValueError(f"_compute_lighting_features: unexpected frame shape {im.shape}")

        # ensure ordering is RGB for cv2.cvtColor conversion to gray; if your frames are BGR, change accordingly
        # Here we assume incoming frames are RGB (consistent with your pipeline earlier).
        rgb = im.astype(np.uint8)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)  # 2D uint8

        # Basic brightness stats
        gray_f = gray.astype(np.float32)
        total_pixels = gray_f.size
        eps = 1e-10

        brightness_mean = float(np.mean(gray_f))
        brightness_std = float(np.std(gray_f))
        global_contrast = brightness_std  # RMS contrast = std
        features["brightness_mean"] = brightness_mean
        features["brightness_std"] = brightness_std
        features["global_contrast"] = global_contrast

        # Brightness entropy (histogram over 256 bins)
        hist, _ = np.histogram(gray, bins=256, range=(0, 256))
        features["brightness_entropy"] = self._safe_entropy_from_hist(hist)

        # Over/under exposed ratios (fractions)
        overexposed = int(np.sum(gray >= 250))
        underexposed = int(np.sum(gray <= 5))
        over_ratio = float(overexposed / (total_pixels + eps))
        under_ratio = float(underexposed / (total_pixels + eps))
        features["overexposed_pixels"] = over_ratio
        features["underexposed_pixels"] = under_ratio

        # Highlight & shadow clipping (same as above but slightly different thresholds)
        highlight_threshold = 250
        shadow_threshold = 5
        highlight_clipped = int(np.sum(gray >= highlight_threshold))
        shadow_clipped = int(np.sum(gray <= shadow_threshold))
        features["highlight_clipping_ratio"] = float(highlight_clipped / (total_pixels + eps))
        features["shadow_clipping_ratio"] = float(shadow_clipped / (total_pixels + eps))

        # Contrast entropy (coarser histogram)
        contrast_hist, _ = np.histogram(gray, bins=64, range=(0, 256))
        features["contrast_entropy"] = self._safe_entropy_from_hist(contrast_hist)

        # Dynamic range: use max/min luminance and convert to decibels (20*log10)
        # Protect against zero; add tiny eps
        max_lum = float(np.max(gray_f))
        min_lum = float(np.min(gray_f))
        if min_lum <= 0:
            min_lum = 1e-3
        # ratio in linear domain
        dr_ratio = max_lum / (min_lum + eps)
        # convert to decibels (20*log10 for amplitude-like measure)
        dynamic_range_db = 20.0 * math.log10(dr_ratio + eps)
        features["dynamic_range_db"] = float(max(0.0, dynamic_range_db))

        # Local contrast: sliding non-overlapping windows (window_size adaptive)
        h, w = gray.shape
        # choose window size proportional to smaller dimension but not too large
        window_size = max(4, min(32, min(h, w) // 8))  # sensible defaults
        local_stds = []
        for i in range(0, h, window_size):
            for j in range(0, w, window_size):
                window = gray_f[i : min(i + window_size, h), j : min(j + window_size, w)]
                if window.size:
                    local_stds.append(float(np.std(window)))
        if local_stds:
            local_contrast = float(np.mean(local_stds))
            local_contrast_std = float(np.std(local_stds))
            features["local_contrast"] = local_contrast
            features["local_contrast_std"] = local_contrast_std
        else:
            local_contrast = global_contrast
            features["local_contrast"] = float(local_contrast)
            features["local_contrast_std"] = 0.0

        # Lighting uniformity (grid-based) and vignetting via helper
        uniformity_features = self._compute_lighting_uniformity(gray)
        features.update(uniformity_features)

        # Нормализованные lighting‑фичи (0–1) для компактного вектора
        features["global_contrast_norm"] = float(min(global_contrast / 255.0, 1.0))
        features["local_contrast_mean_norm"] = float(min(local_contrast / 255.0, 1.0))
        features["overexposed_ratio"] = over_ratio
        features["underexposed_ratio"] = under_ratio
        features["vignetting_score_norm"] = float(
            uniformity_features.get("vignetting_score", 0.0)
        )

        # Final safety: cast to native floats and ensure keys exist
        for k, v in list(features.items()):
            try:
                features[k] = float(v)
            except Exception:
                features[k] = 0.0

        return features

    def _compute_light_direction(self, frame: np.ndarray) -> Dict[str, float]:
        """Оценивает направление света, количество источников, мягкость/жёсткость."""
        features = {}

        # --- Validate & convert ---
        if frame.ndim == 3 and frame.shape[-1] == 4:
            frame = frame[..., :3]
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError(f"_compute_light_direction: unexpected frame shape {frame.shape}")

        rgb = frame.astype(np.uint8)
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)

        h, w = gray.shape
        eps = 1e-8

        # === 1. GRADIENT-BASED LIGHT DIRECTION (ROBUST CIRCULAR MEAN) ===
        grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

        magnitudes = np.sqrt(grad_x**2 + grad_y**2)
        angles = np.arctan2(grad_y, grad_x)  # radians

        # если нет текстуры — направление неопределено (угол не возвращаем, только источники света)
        if np.sum(magnitudes) < eps:
            pass
        else:
            # оставляем расчёт для возможного дебага, но не сохраняем в фичи,
            # чтобы не засорять финальный вектор сильно шумным признаком
            sin_sum = float(np.sum(np.sin(angles) * (magnitudes + eps)))
            cos_sum = float(np.sum(np.cos(angles) * (magnitudes + eps)))
            _ = np.arctan2(sin_sum, cos_sum)

        # === 2. LIGHT SOURCE COUNT (ROBUST 2D PEAK DETECTION) ===
        # Сглаживаем картинку и ищем bright blobs
        blur = gaussian_filter(gray, sigma=7)
        threshold = np.percentile(blur, 96)

        mask = blur > threshold
        labeled, num_labels = label(mask)  # connected components

        # Ограничиваем количество источников света
        source_count = min(int(num_labels), 5)
        features["light_source_count_estimate"] = float(source_count)

        # === 3. SOFT vs HARD LIGHT (LAPLACIAN VARIANCE + NORMALIZATION) ===
        lap_var = float(cv2.Laplacian(gray, cv2.CV_32F).var())

        # нормировка в диапазон [0..1]
        # low variance → soft light; high → hard
        # границы подогнаны под реальные кадры
        soft_min = 50      # ультрамягкий
        hard_max = 600     # ультражёсткий

        if lap_var <= soft_min:
            soft_prob = 1.0
        elif lap_var >= hard_max:
            soft_prob = 0.0
        else:
            # линейное уменьшение мягкости
            soft_prob = 1.0 - (lap_var - soft_min) / (hard_max - soft_min)

        hard_prob = 1.0 - soft_prob

        features["soft_light_probability"] = float(soft_prob)
        features["hard_light_probability"] = float(hard_prob)
        # Дубликат с более коротким именем для компактного вектора
        features["soft_light_prob"] = float(soft_prob)

        return features

        
    def extract_frame_features(self, frame: np.ndarray, frame_idx: int) -> Dict[str, Any]:
        """Извлекает все frame-level фичи для одного кадра"""
        # Убеждаемся, что frame в RGB формате и правильном типе
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        
        # Если кадр в BGR формате (обычно для OpenCV), конвертируем в RGB
        # Проверяем по предположению: если кадры сохранены через OpenCV, они могут быть BGR
        # Для безопасности конвертируем только если нужно (можно убрать, если кадры уже в RGB)
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            # Предполагаем, что кадры уже в RGB (если нет - раскомментировать следующую строку)
            # frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pass
        
        features = {}
        try:
            # RGB статистики (только mean/std)
            features.update(self._compute_rgb_stats(frame))
        except Exception as e:
            print(
                f"ColorLightProcessor | extract_frame_features | Ошибка вычисления RGB статистики: {e}"
            )
        try:
            # HSV фичи + нормализованные hue/sat/value
            features.update(self._compute_hsv_features(frame))
        except Exception as e:
            print(
                f"ColorLightProcessor | extract_frame_features | Ошибка вычисления HSV фичи: {e}"
            )
        try:
            # LAB фичи (L_mean/L_contrast/ab_balance + L_mean_norm)
            features.update(self._compute_lab_features(frame))
        except Exception as e:
            print(
                f"ColorLightProcessor | extract_frame_features | Ошибка вычисления LAB фичи: {e}"
            )
        try:
            # Палитра, colorfulness, skin_tone_ratio, гармонии
            features.update(self._compute_palette_features(frame))
        except Exception as e:
            print(
                f"ColorLightProcessor | extract_frame_features | Ошибка вычисления Палитра и гармонии: {e}"
            )
        try:
            # Освещение и lighting‑фичи
            features.update(self._compute_lighting_features(frame))
        except Exception as e:
            print(
                f"ColorLightProcessor | extract_frame_features | Ошибка вычисления Освещение: {e}"
            )
        try:
            # Источники света и soft/hard light
            features.update(self._compute_light_direction(frame))
        except Exception as e:
            print(
                f"ColorLightProcessor | extract_frame_features | Ошибка вычисления Направление света: {e}"
            )

        return {
            "frame_idx": frame_idx,
            "features": features
        }
    
    def extract_scene_features(self, frame_features: List[Dict], scene_start: int, scene_end: int) -> Dict[str, Any]:
        """Извлекает scene-level фичи из списка frame features"""
        if not frame_features:
            return {}
        
        scene_features = {}
        
        # Базовые метрики сцены
        num_frames = len(frame_features)
        scene_features['num_frames'] = num_frames
        if self.max_frames_per_scene > 0:
            scene_features['num_frames_norm'] = float(
                min(num_frames / float(self.max_frames_per_scene), 1.0)
            )
        
        # Извлекаем значения фич из всех кадров
        feature_arrays = {}
        for frame_feat in frame_features:
            for key, value in frame_feat['features'].items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    if key not in feature_arrays:
                        feature_arrays[key] = []
                    feature_arrays[key].append(value)
        
        # Усреднение RGB/HSV/LAB фич по сцене
        for key, values in feature_arrays.items():
            if len(values) > 0:
                scene_features[f'{key}_mean'] = float(np.mean(values))
                scene_features[f'{key}_std'] = float(np.std(values))
        
        # Motion + Lighting features
        # Для скорости изменения освещенности используем Value из HSV
        brightness_values = [f['features'].get('value_mean', 0) for f in frame_features]
        if len(brightness_values) > 1:
            brightness_diff = np.diff(brightness_values)
            scene_features['brightness_change_speed'] = float(np.mean(np.abs(brightness_diff)))
            scene_features['scene_flicker_intensity'] = float(np.std(brightness_diff))
        
        # Color change speed (по hue)
        hue_values = [f['features'].get('hue_mean', 0) for f in frame_features]
        if len(hue_values) > 1:
            hue_diff = np.diff(hue_values)
            # Учитываем циклический характер hue
            hue_diff = np.minimum(np.abs(hue_diff), 180 - np.abs(hue_diff))
            scene_features['color_change_speed'] = float(np.mean(np.abs(hue_diff)))
            scene_features['color_transition_variance'] = float(np.var(hue_diff))
        
        # Flash events (резкие скачки яркости)
        if len(brightness_values) > 2:
            brightness_diff = np.diff(brightness_values)
            flash_threshold = np.mean(brightness_values) + 2 * np.std(brightness_values)
            flash_events = np.sum(np.abs(brightness_diff) > flash_threshold)
            scene_features['flash_events_count'] = float(flash_events)
            # Нормированное количество вспышек относительно длины сцены
            scene_features['flash_events_count_norm'] = float(
                flash_events / float(max(1, num_frames - 1))
            )
        
        # Temporal Color Patterns
        if len(frame_features) > 1:
            # Color stability (стабильность цвета)
            color_stability = []
            for i in range(len(frame_features) - 1):
                f1 = frame_features[i]['features']
                f2 = frame_features[i + 1]['features']
                # Используем RGB mean для оценки стабильности
                rgb1 = np.array([f1.get('color_mean_r', 0), f1.get('color_mean_g', 0), f1.get('color_mean_b', 0)])
                rgb2 = np.array([f2.get('color_mean_r', 0), f2.get('color_mean_g', 0), f2.get('color_mean_b', 0)])
                diff = np.linalg.norm(rgb1 - rgb2)
                color_stability.append(diff)
            scene_features['color_stability'] = float(1.0 / (np.mean(color_stability) + 1e-10))
            
            # Color temporal entropy
            hue_seq = [f['features'].get('hue_mean', 0) for f in frame_features]
            hue_hist, _ = np.histogram(hue_seq, bins=18)
            hue_probs = hue_hist / (hue_hist.sum() + 1e-10)
            scene_features['color_temporal_entropy'] = float(entropy(hue_probs + 1e-10))
            
            # Color pattern periodicity (простая оценка через автокорреляцию)
            if len(hue_seq) > 3:
                hue_array = np.array(hue_seq)
                autocorr = np.correlate(hue_array - np.mean(hue_array), 
                                       hue_array - np.mean(hue_array), mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                autocorr = autocorr / (autocorr[0] + 1e-10)
                # Ищем второй пик (первый - это lag=0)
                if len(autocorr) > 2:
                    peaks, _ = find_peaks(autocorr[1:], height=0.3)
                    scene_features['color_pattern_periodicity'] = float(len(peaks) / len(autocorr))
                else:
                    scene_features['color_pattern_periodicity'] = 0.0
            else:
                scene_features['color_pattern_periodicity'] = 0.0
            
            # Scene color shift speed
            if len(hue_values) > 1:
                scene_features['scene_color_shift_speed'] = float(np.mean(np.abs(np.diff(hue_values))))
        
        # Контраст и динамический диапазон (усредненные по сцене)
        contrast_values = [f['features'].get('global_contrast', 0) for f in frame_features]
        if contrast_values:
            scene_features['scene_contrast'] = float(np.mean(contrast_values))
        
        brightness_max = [f['features'].get('brightness_mean', 0) for f in frame_features]
        brightness_min = brightness_max
        if brightness_max:
            scene_features['dynamic_range'] = float(np.max(brightness_max) - np.min(brightness_min))
        
        return scene_features
    
    def _compute_color_style_features(self, all_frame_features: List[Dict]) -> Dict[str, float]:
        """Вычисляет стили цветокоррекции"""
        features = {}
        
        # Собираем статистики по всем кадрам
        hue_means = [f['features'].get('hue_mean', 0) for f in all_frame_features]
        sat_means = [f['features'].get('saturation_mean', 0) for f in all_frame_features]
        rgb_means = []
        for f in all_frame_features:
            rgb_means.append([
                f['features'].get('color_mean_r', 0),
                f['features'].get('color_mean_g', 0),
                f['features'].get('color_mean_b', 0)
            ])
        
        if not rgb_means:
            return {f'style_{k}_prob': 0.0 for k in ['teal_orange', 'film', 'desaturated', 'hyper_saturated', 'vintage', 'tiktok']}
        
        rgb_means = np.array(rgb_means)
        avg_rgb = np.mean(rgb_means, axis=0)
        avg_sat = np.mean(sat_means) if sat_means else 128
        
        # Teal & Orange (теплые и холодные тона одновременно)
        # Оранжевый в RGB: высокий R и G, низкий B
        # Teal: низкий R, средний G, высокий B
        orange_score = (avg_rgb[0] + avg_rgb[1] - avg_rgb[2]) / 255.0
        teal_score = (avg_rgb[2] + avg_rgb[1] - avg_rgb[0]) / 255.0
        features['style_teal_orange_prob'] = float(min(orange_score * teal_score, 1.0))
        
        # Film look (низкая насыщенность, мягкие тона)
        low_sat = avg_sat < 100
        soft_tones = np.std(rgb_means) < 30
        features['style_film_prob'] = float(1.0 if (low_sat and soft_tones) else 0.3)
        
        # Desaturated
        features['style_desaturated_prob'] = float(1.0 - min(avg_sat / 128.0, 1.0))
        
        # Hyper saturated
        features['style_hyper_saturated_prob'] = float(min((avg_sat - 128) / 128.0, 1.0) if avg_sat > 128 else 0.0)
        
        # Vintage (сепия-подобные тона, низкая насыщенность)
        sepia_score = (avg_rgb[0] * 0.393 + avg_rgb[1] * 0.769 + avg_rgb[2] * 0.189) / 255.0
        features['style_vintage_prob'] = float(sepia_score * (1.0 - avg_sat / 255.0))
        
        # TikTok style (высокая насыщенность, яркие цвета)
        high_sat = avg_sat > 150
        bright = np.mean(avg_rgb) > 180
        features['style_tiktok_prob'] = float(1.0 if (high_sat and bright) else 0.2)
        
        return features
    
    def _compute_aesthetic_scores(self, all_frame_features: List[Dict]) -> Dict[str, float]:
        """Вычисляет aesthetic & cinematic scores (упрощенные версии)"""
        features = {}
        
        # Упрощенные оценки (полные модели NIMA/LAION требуют отдельной загрузки)
        contrast_values = [f['features'].get('global_contrast', 0) for f in all_frame_features]
        brightness_values = [f['features'].get('brightness_mean', 0) for f in all_frame_features]
        colorfulness_values = [f['features'].get('colorfulness_index', 0) for f in all_frame_features]
        
        if contrast_values:
            features['nima_mean'] = float(np.mean(contrast_values) / 50.0)  # нормализация
            features['nima_std'] = float(np.std(contrast_values) / 50.0)
        
        if colorfulness_values:
            features['laion_mean'] = float(np.mean(colorfulness_values) / 100.0)
            features['laion_std'] = float(np.std(colorfulness_values) / 100.0)
        
        # Cinematic lighting score (оценка на основе контраста и направления света)
        lighting_scores = []
        for f in all_frame_features:
            contrast = f['features'].get('global_contrast', 0)
            local_contrast = f['features'].get('local_contrast', 0)
            soft_light = f['features'].get('soft_light_probability', 0.5)
            score = (contrast / 50.0) * 0.4 + (local_contrast / 30.0) * 0.3 + soft_light * 0.3
            lighting_scores.append(score)
        
        features['cinematic_lighting_score'] = float(np.mean(lighting_scores)) if lighting_scores else 0.5
        
        # Professional look score (комбинация различных факторов)
        professional_scores = []
        for f in all_frame_features:
            overexposed = f['features'].get('overexposed_pixels', 0)
            underexposed = f['features'].get('underexposed_pixels', 0)
            contrast = f['features'].get('global_contrast', 0)
            colorfulness = f['features'].get('colorfulness_index', 0)
            
            score = (1.0 - overexposed - underexposed) * 0.4
            score += min(contrast / 50.0, 1.0) * 0.3
            score += min(colorfulness / 100.0, 1.0) * 0.3
            professional_scores.append(score)
        
        features['professional_look_score'] = float(np.mean(professional_scores)) if professional_scores else 0.5
        
        return features
    
    def _compute_gini_coefficient(self, values: np.ndarray) -> float:
        """Вычисляет коэффициент Джини"""
        if len(values) == 0:
            return 0.0
        sorted_values = np.sort(values)
        n = len(values)
        index = np.arange(1, n + 1)
        return float((2 * np.sum(index * sorted_values)) / (n * np.sum(sorted_values)) - (n + 1) / n)
    
    def extract_video_features(self, all_scene_features: Dict[str, Dict], all_frame_features: Dict[str, Dict]) -> Dict[str, Any]:
        """Агрегирует video-level фичи."""
        features = {}

        # -------------------------------
        # 1. Проверки
        # -------------------------------
        if not all_scene_features or not all_frame_features:
            return features

        # -------------------------------
        # 2. Scene-level агрегаты
        # -------------------------------
        scene_agg = {}

        for scene_feat in all_scene_features.values():
            for key, val in scene_feat.items():
                if isinstance(val, (float, int)) and not isinstance(val, bool):
                    scene_agg.setdefault(key, []).append(val)

        for key, vals in scene_agg.items():
            if len(vals) > 0:
                features[f"{key}_mean"] = float(np.mean(vals))
                features[f"{key}_std"] = float(np.std(vals))
                features[f"{key}_min"] = float(np.min(vals))
                features[f"{key}_max"] = float(np.max(vals))

        # -------------------------------
        # 3. Собираем frame-level фичи
        # -------------------------------

        # Собираем ВСЕ кадры в список
        frame_list = []
        for scene_dict in all_frame_features.values():
            for frame_idx, frame_feat in scene_dict.items():
                frame_list.append(frame_feat)

        # Нечего анализировать
        if len(frame_list) == 0:
            return features

        # Универсальный safe getter
        def getf(frame, key, default=0.0):
            return frame.get(key, default)

        # -------------------------------
        # 4. Color/Hue distribution
        # -------------------------------
        hue_values = [getf(f, "hue_mean", 0) for f in frame_list]

        if len(hue_values) > 0:
            hue_hist, _ = np.histogram(hue_values, bins=36)
            hue_probs = hue_hist / (hue_hist.sum() + 1e-10)
            features["color_distribution_entropy"] = float(entropy(hue_probs + 1e-10))
            features["color_distribution_gini"] = float(self._compute_gini_coefficient(np.array(hue_values)))

        # -------------------------------
        # 5. Color style features
        # -------------------------------
        features.update(self._compute_color_style_features(frame_list))

        # -------------------------------
        # 6. Aesthetic scores
        # -------------------------------
        features.update(self._compute_aesthetic_scores(frame_list))

        # -------------------------------
        # 7. Global brightness dynamics
        # -------------------------------
        brightness_values = [getf(f, "brightness_mean", 0) for f in frame_list]

        if len(brightness_values) > 1:
            diff = np.diff(brightness_values)
            features["global_brightness_change_speed"] = float(np.mean(np.abs(diff)))

        # -------------------------------
        # 8. Global color change speed
        # -------------------------------
        if len(hue_values) > 1:
            hue_diff = np.diff(hue_values)
            # корректный hue wrap-around
            hue_diff = np.minimum(np.abs(hue_diff), 180 - np.abs(hue_diff))
            features["global_color_change_speed"] = float(np.mean(np.abs(hue_diff)))

        # -------------------------------
        # 9. Strobe transitions
        # -------------------------------
        if len(brightness_values) > 2:
            diff = np.abs(np.diff(brightness_values))
            threshold = np.mean(brightness_values) + 1.5 * np.std(brightness_values)
            strobe_count = np.sum(diff > threshold)
            features["strobe_transition_frequency"] = float(strobe_count / len(diff))

        # -------------------------------
        # 10. Color periodicity + color shift
        # -------------------------------
        if len(hue_values) > 3:
            arr = np.array(hue_values)
            autocorr = np.correlate(arr - np.mean(arr), arr - np.mean(arr), mode="full")
            autocorr = autocorr[len(autocorr)//2:]
            autocorr /= (autocorr[0] + 1e-10)

            if len(autocorr) > 2:
                peaks, _ = find_peaks(autocorr[1:], height=0.2)
                features["global_color_periodicity"] = float(len(peaks) / len(autocorr))
            else:
                features["global_color_periodicity"] = 0.0

            features["global_color_shift"] = float(np.mean(np.abs(np.diff(hue_values))))

        return features

    
    def _create_sequence_inputs(self, all_frame_features: Dict[str, Dict[int, Dict]],
                                all_scene_features: Dict[str, Dict[str, Any]],
                                video_features: Dict[str, Any]) -> Dict[str, List]:
        """
        Создает sequence inputs для трансформера
        
        Args:
            all_frame_features: {scene_label: {frame_idx: {"features": {...}}}}
            all_scene_features: {scene_label: {"feat1":..., "feat2":...}}
            video_features: {"feat": value}
        """
        sequences = {}

        # ============================================================
        # 1) FRAME SEQUENCE → [N_frames_total, D_frame_features]
        # ============================================================
        frame_sequence: List[List[float]] = []

        # Компактный набор фич для VisualTransformer (нормализованные 0–1)
        frame_keys = [
            "hue_mean_norm",
            "hue_std_norm",
            "hue_entropy_weighted",
            "sat_mean_norm",
            "val_mean_norm",
            "L_mean_norm",
            "global_contrast_norm",
            "local_contrast_mean_norm",
            "colorfulness_norm",
            "skin_tone_ratio",
            "overexposed_ratio",
            "underexposed_ratio",
            "vignetting_score_norm",
            "soft_light_prob",
            "dominant_lab_a_norm",
            "dominant_lab_b_norm",
        ]

        # Собираем все кадры ПЛОСКО по всем сценам
        for scene_label, frames_dict in all_frame_features.items():
            for frame_idx, frame_feat in sorted(frames_dict.items(), key=lambda x: x[0]):
                feat_dict = frame_feat["features"]
                frame_vector = [
                    float(feat_dict.get(k, 0.0)) for k in frame_keys
                ]
                frame_sequence.append(frame_vector)

        sequences["frames"] = frame_sequence

        # ============================================================
        # 2) SCENE SEQUENCE → [N_scenes, D_scene_features]
        # ============================================================
        scene_sequence = []

        for scene_label, scene_feat in all_scene_features.items():
            numeric_keys = sorted([
                k for k, v in scene_feat.items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            ])
            scene_vector = [float(scene_feat[k]) for k in numeric_keys]
            scene_sequence.append(scene_vector)

        sequences["scenes"] = scene_sequence

        # ============================================================
        # 3) GLOBAL → [D_global_features]
        # ============================================================
        global_sequence = []

        numeric_keys = sorted([
            k for k, v in video_features.items()
            if isinstance(v, (int, float)) and not isinstance(v, bool)
        ])

        global_sequence = [float(video_features[k]) for k in numeric_keys]
        sequences["global"] = global_sequence

        return sequences

    
    def process(
        self,
        frame_manager: FrameManager,
        frame_indices: List[int],
        config: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Главный метод обработки видео (интерфейс BaseModule).
        
        Args:
            frame_manager: FrameManager для доступа к кадрам
            frame_indices: Список индексов кадров для обработки
            config: Конфигурация модуля (не используется, но требуется BaseModule)
        
        Returns:
            Словарь с результатами в формате для сохранения в npz
        """
        self.initialize()  # Гарантируем инициализацию
        
        # Загружаем scenes из scene_classification
        scene_data = self.load_dependency_results("scene_classification")
        
        if scene_data is None:
            raise RuntimeError(
                f"{self.module_name} | process | scene_classification не найдены. "
                f"Убедитесь, что модуль scene_classification запущен перед этим модулем. "
                f"rs_path: {self.rs_path}"
            )
        
        # Извлекаем scenes из данных
        # Формат может быть разным, проверяем несколько вариантов
        scenes = None
        if isinstance(scene_data, dict):
            # Пробуем найти scenes в разных местах
            if "scenes" in scene_data:
                scenes = scene_data["scenes"]
            elif "aggregated" in scene_data and "scenes" in scene_data["aggregated"]:
                scenes = scene_data["aggregated"]["scenes"]
            else:
                # Возможно, весь словарь и есть scenes
                # Проверяем структуру: должны быть ключи с "indices"
                if all(isinstance(v, dict) and "indices" in v for v in scene_data.values()):
                    scenes = scene_data
        
        if scenes is None:
            raise ValueError(
                f"{self.module_name} | process | Не удалось извлечь scenes из данных scene_classification. "
                f"Проверьте формат сохраненных данных."
            )
        
        # Обработка кадров по сценам
        all_frame_features = {}
        all_scene_features = {}
        
        for i, (scene_label, frame_range) in enumerate(scenes.items()):
                start_frame = frame_range["indices"][0]
                end_frame = frame_range["indices"][-1]

                num_frames_in_scene = end_frame - start_frame + 1

                # ==== STRIDE LOGIC ====
                if self.stride and self.stride > 1:
                    # Берём кадры по шагу
                    frame_indices = np.arange(start_frame, end_frame + 1, self.stride)
                    # Гарантируем включение конца сцены
                    if frame_indices[-1] != end_frame:
                        frame_indices = np.append(frame_indices, end_frame)

                else:
                    # ==== OLD RANDOM SAMPLING LOGIC ====
                    num_samples = min(self.max_frames_per_scene, num_frames_in_scene)

                    if num_samples == 1:
                        frame_indices = np.array([start_frame])
                    else:
                        frame_indices = np.linspace(start_frame, end_frame, num_samples, dtype=int)

                # Убираем дубликаты и сортируем
                frame_indices = np.unique(frame_indices)

                all_frame_features[scene_label] = {}
                scene_frame_features = []

                # ==== FEATURE EXTRACTION ====
                for k, frame_idx in enumerate(frame_indices):
                    try:
                        frame = frame_manager.get(frame_idx)
                        frame_feat = self.extract_frame_features(frame, frame_idx)

                        scene_frame_features.append(frame_feat)
                        all_frame_features[scene_label][frame_idx] = frame_feat

                        logger.info(
                            f"Сцена {i+1}/{len(scenes)} | Кадр {k+1}/{len(frame_indices)} обработан"
                        )
                    except Exception as e:
                        logger.error(f"Failed to process frame {frame_idx}: {e}")
                        continue

                # ==== SCENE FEATURES ====
                if scene_frame_features:
                    scene_feat = self.extract_scene_features(
                        scene_frame_features,
                        start_frame,
                        end_frame
                    )
                    all_scene_features[scene_label] = scene_feat

                logger.info(f"Сцена {i+1}/{len(scenes)} | Scene-level фичи извлечены")
        
        # Video-level фичи
        video_features = self.extract_video_features(all_scene_features, all_frame_features)

        logger.info(f"Видео фичи извлечены")
        
        # Формируем последовательности для трансформера на основе компактных фич
        sequence_inputs = self._create_sequence_inputs(
            all_frame_features, all_scene_features, video_features
        )

        # Формируем результат
        result = {
            "frames": all_frame_features,
            "scenes": all_scene_features,
            "video_features": video_features,
            "sequence_inputs": sequence_inputs,
        }

        return result
