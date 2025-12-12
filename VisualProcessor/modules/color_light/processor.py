import json
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from scipy import stats
from scipy.stats import entropy
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
from collections import Counter
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)


class ColorLightProcessor:
    """Процессор для анализа цвета и освещения видео"""
    
    def __init__(self, max_frames_per_scene: int = 350):
        """
        Args:
            max_frames_per_scene: Максимальное количество кадров для обработки на сцену
        """
        self.max_frames_per_scene = max_frames_per_scene
    
    def _compute_rgb_stats(self, frame: np.ndarray) -> Dict[str, float]:
        """Вычисляет RGB статистики: mean/std/min/max/skew/kurt для каждого канала"""
        features = {}
        for i, channel in enumerate(['R', 'G', 'B']):
            channel_data = frame[:, :, i].flatten().astype(np.float32)
            features[f'color_mean_{channel.lower()}'] = float(np.mean(channel_data))
            features[f'color_std_{channel.lower()}'] = float(np.std(channel_data))
            features[f'color_min_{channel.lower()}'] = float(np.min(channel_data))
            features[f'color_max_{channel.lower()}'] = float(np.max(channel_data))
            features[f'color_skew_{channel.lower()}'] = float(stats.skew(channel_data))
            features[f'color_kurt_{channel.lower()}'] = float(stats.kurtosis(channel_data))
        return features
    
    def _compute_hsv_features(self, frame: np.ndarray) -> Dict[str, float]:
        """Вычисляет HSV фичи: hue_mean/std/entropy, saturation_mean/std, value_mean/std"""
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        features = {}
        
        # Hue
        hue = hsv[:, :, 0].flatten()
        features['hue_mean'] = float(np.mean(hue))
        features['hue_std'] = float(np.std(hue))
        # Энтропия hue (дискретизация для вычисления энтропии)
        hue_hist, _ = np.histogram(hue, bins=36, range=(0, 180))
        hue_probs = hue_hist / (hue_hist.sum() + 1e-10)
        features['hue_entropy'] = float(entropy(hue_probs + 1e-10))
        
        # Saturation
        sat = hsv[:, :, 1].flatten()
        features['saturation_mean'] = float(np.mean(sat))
        features['saturation_std'] = float(np.std(sat))
        
        # Value (brightness)
        val = hsv[:, :, 2].flatten()
        features['value_mean'] = float(np.mean(val))
        features['value_std'] = float(np.std(val))
        
        return features
    
    def _compute_lab_features(self, frame: np.ndarray) -> Dict[str, float]:
        """Вычисляет LAB фичи: L_mean, L_contrast, ab_balance"""
        lab = cv2.cvtColor(frame, cv2.COLOR_RGB2LAB)
        features = {}
        
        # L channel (lightness)
        L = lab[:, :, 0].flatten()
        features['L_mean'] = float(np.mean(L))
        features['L_contrast'] = float(np.std(L))
        
        # ab balance (тепло/холод)
        a = lab[:, :, 1].flatten() - 128  # центрируем
        b = lab[:, :, 2].flatten() - 128
        features['ab_balance'] = float(np.mean(a) - np.mean(b))  # положительное = тепло
        
        return features
    
    def _compute_palette_features(self, frame: np.ndarray) -> Dict[str, float]:
        """Вычисляет палитру и гармонии: dominant/secondary/tertiary colors, palette_size, etc."""
        features = {}
        
        # Уменьшаем разрешение для кластеризации
        h, w = frame.shape[:2]
        sample_size = min(10000, h * w)
        if h * w > sample_size:
            step = int(np.sqrt(h * w / sample_size))
            sampled = frame[::step, ::step].reshape(-1, 3)
        else:
            sampled = frame.reshape(-1, 3)
        
        # K-means для доминирующих цветов
        n_colors = min(5, len(sampled))
        if n_colors > 1:
            kmeans = KMeans(n_clusters=n_colors, random_state=42, n_init=10)
            kmeans.fit(sampled)
            colors = kmeans.cluster_centers_.astype(int)
            counts = Counter(kmeans.labels_)
            
            # Сортируем по частоте
            sorted_colors = sorted(zip(colors, [counts[i] for i in range(n_colors)]), 
                                 key=lambda x: x[1], reverse=True)
            
            for i, (color, count) in enumerate(sorted_colors[:3]):
                if i == 0:
                    features['dominant_color'] = color.tolist()
                elif i == 1:
                    features['secondary_color'] = color.tolist()
                elif i == 2:
                    features['tertiary_color'] = color.tolist()
            
            features['palette_size'] = float(len(set(kmeans.labels_)))
        else:
            features['dominant_color'] = [int(c) for c in sampled[0]]
            features['secondary_color'] = [0, 0, 0]
            features['tertiary_color'] = [0, 0, 0]
            features['palette_size'] = 1.0
        
        # Colorfulness index (стандартное отклонение в цветовом пространстве)
        rgb_reshaped = frame.reshape(-1, 3).astype(np.float32)
        rg = rgb_reshaped[:, 0] - rgb_reshaped[:, 1]
        yb = 0.5 * (rgb_reshaped[:, 0] + rgb_reshaped[:, 1]) - rgb_reshaped[:, 2]
        std_rg = np.std(rg)
        std_yb = np.std(yb)
        mean_rgyb = np.sqrt(np.mean(rg**2) + np.mean(yb**2))
        features['colorfulness_index'] = float(std_rg + std_yb + 0.3 * mean_rgyb)
        
        # Warm vs cold ratio
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        hue = hsv[:, :, 0].flatten()
        # Теплые цвета: 0-30 и 150-180 (красный, оранжевый, желтый)
        warm_mask = ((hue >= 0) & (hue <= 30)) | ((hue >= 150) & (hue <= 180))
        warm_count = np.sum(warm_mask)
        cold_count = len(hue) - warm_count
        features['warm_vs_cold_ratio'] = float(warm_count / (cold_count + 1e-10))
        
        # Skin tone ratio (приблизительная оценка по цвету кожи)
        # Типичный диапазон кожи в HSV: H: 0-25, S: 20-255, V: 50-255
        sat = hsv[:, :, 1].flatten()
        val = hsv[:, :, 2].flatten()
        skin_mask = ((hue >= 0) & (hue <= 25)) & (sat >= 20) & (val >= 50)
        features['skin_tone_ratio'] = float(np.sum(skin_mask) / len(hue))
        
        # Color palette entropy
        hue_hist, _ = np.histogram(hue, bins=36, range=(0, 180))
        hue_probs = hue_hist / (hue_hist.sum() + 1e-10)
        features['color_palette_entropy'] = float(entropy(hue_probs + 1e-10))
        
        # Color harmonies (complementary, analogous, triadic, split-complementary)
        harmony_features = self._compute_color_harmonies(hue, sat, val)
        features.update(harmony_features)
        
        return features
    
    def _compute_color_harmonies(self, hue: np.ndarray, sat: np.ndarray, val: np.ndarray) -> Dict[str, float]:
        """Вычисляет цветовые гармонии: complementary, analogous, triadic, split-complementary"""
        features = {}
        
        # Находим доминирующий hue (наиболее частый оттенок)
        hue_hist, hue_bins = np.histogram(hue, bins=36, range=(0, 180))
        dominant_hue_bin = np.argmax(hue_hist)
        dominant_hue = (hue_bins[dominant_hue_bin] + hue_bins[dominant_hue_bin + 1]) / 2
        
        # Нормализуем hue в диапазон 0-360 для работы с цветовым кругом
        dominant_hue_360 = dominant_hue * 2  # HSV hue в OpenCV: 0-180, цветовой круг: 0-360
        
        # Complementary (дополнительные цвета) - противоположные на цветовом круге
        comp_hue = (dominant_hue_360 + 180) % 360
        comp_hue_180 = comp_hue / 2  # обратно в диапазон 0-180
        # Проверяем наличие цветов в диапазоне ±15 градусов от complementary
        comp_range = ((hue >= (comp_hue_180 - 15) % 180) & (hue <= (comp_hue_180 + 15) % 180)) | \
                     ((hue >= 0) & (hue <= (comp_hue_180 + 15 - 180) % 180)) | \
                     ((hue >= (comp_hue_180 - 15 + 180) % 180) & (hue <= 180))
        comp_ratio = np.sum(comp_range) / len(hue)
        features['color_harmony_complementary_prob'] = float(min(comp_ratio * 2, 1.0))  # нормализуем
        
        # Analogous (аналогичные цвета) - соседние цвета на цветовом круге (±30 градусов)
        # Доминирующий цвет уже есть, проверяем соседние
        anal_range1 = (hue >= (dominant_hue - 30) % 180) & (hue <= (dominant_hue + 30) % 180)
        anal_range2 = ((hue >= 0) & (hue <= (dominant_hue + 30 - 180) % 180)) | \
                      ((hue >= (dominant_hue - 30 + 180) % 180) & (hue <= 180))
        anal_range = anal_range1 | anal_range2
        anal_ratio = np.sum(anal_range) / len(hue)
        features['color_harmony_analogous_prob'] = float(min(anal_ratio * 1.2, 1.0))
        
        # Triadic (триада) - три цвета, равномерно распределенные на круге (120 градусов друг от друга)
        triadic_hue1 = (dominant_hue_360 + 120) % 360
        triadic_hue2 = (dominant_hue_360 + 240) % 360
        triadic_hue1_180 = triadic_hue1 / 2
        triadic_hue2_180 = triadic_hue2 / 2
        
        triadic_range1 = (hue >= (triadic_hue1_180 - 15) % 180) & (hue <= (triadic_hue1_180 + 15) % 180)
        triadic_range2 = (hue >= (triadic_hue2_180 - 15) % 180) & (hue <= (triadic_hue2_180 + 15) % 180)
        triadic_range = triadic_range1 | triadic_range2
        triadic_ratio = np.sum(triadic_range) / len(hue)
        features['color_harmony_triadic_prob'] = float(min(triadic_ratio * 3, 1.0))
        
        # Split-complementary (расщепленная комплементарная) - основной цвет + два соседних к complementary
        split_comp_hue1 = (comp_hue - 30) % 360
        split_comp_hue2 = (comp_hue + 30) % 360
        split_comp_hue1_180 = split_comp_hue1 / 2
        split_comp_hue2_180 = split_comp_hue2 / 2
        
        split_range1 = (hue >= (split_comp_hue1_180 - 15) % 180) & (hue <= (split_comp_hue1_180 + 15) % 180)
        split_range2 = (hue >= (split_comp_hue2_180 - 15) % 180) & (hue <= (split_comp_hue2_180 + 15) % 180)
        split_range = split_range1 | split_range2
        split_ratio = np.sum(split_range) / len(hue)
        features['color_harmony_split_complementary_prob'] = float(min(split_ratio * 2.5, 1.0))
        
        return features
    
    def _compute_lighting_features(self, frame: np.ndarray) -> Dict[str, float]:
        """Вычисляет фичи освещения: brightness, contrast, overexposed/underexposed pixels"""
        features = {}
        
        # Яркость (grayscale)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).flatten()
        features['brightness_mean'] = float(np.mean(gray))
        features['brightness_std'] = float(np.std(gray))
        
        # Энтропия яркости
        hist, _ = np.histogram(gray, bins=256, range=(0, 256))
        probs = hist / (hist.sum() + 1e-10)
        features['brightness_entropy'] = float(entropy(probs + 1e-10))
        
        # Переэкспонированные/недоэкспонированные пиксели
        overexposed = np.sum(gray > 250)
        underexposed = np.sum(gray < 5)
        total_pixels = len(gray)
        features['overexposed_pixels'] = float(overexposed / total_pixels)
        features['underexposed_pixels'] = float(underexposed / total_pixels)
        
        # Global contrast (RMS contrast)
        features['global_contrast'] = float(np.std(gray))
        
        # Local contrast (среднее стандартное отклонение в окнах)
        h, w = gray.shape
        window_size = min(8, min(h, w) // 4)
        if window_size > 1:
            local_stds = []
            for i in range(0, h - window_size, window_size):
                for j in range(0, w - window_size, window_size):
                    window = gray[i:i+window_size, j:j+window_size]
                    local_stds.append(np.std(window))
            features['local_contrast'] = float(np.mean(local_stds)) if local_stds else 0.0
            features['local_contrast_std'] = float(np.std(local_stds)) if local_stds else 0.0
        else:
            features['local_contrast'] = features['global_contrast']
            features['local_contrast_std'] = 0.0
        
        # Contrast entropy
        contrast_hist, _ = np.histogram(gray, bins=64, range=(0, 256))
        contrast_probs = contrast_hist / (contrast_hist.sum() + 1e-10)
        features['contrast_entropy'] = float(entropy(contrast_probs + 1e-10))
        
        # Dynamic Range (HDR score) - улучшенная версия
        # Используем log-scale luminance для более точной оценки
        gray_float = gray.astype(np.float32)
        # Избегаем log(0)
        log_luminance = np.log1p(gray_float)  # log(1+x) для стабильности
        features['dynamic_range_db'] = float(np.max(log_luminance) - np.min(log_luminance))
        
        # Highlight clipping ratio (пиксели близкие к максимальной яркости)
        highlight_threshold = 250
        highlight_clipped = np.sum(gray >= highlight_threshold)
        features['highlight_clipping_ratio'] = float(highlight_clipped / total_pixels)
        
        # Shadow clipping ratio (пиксели близкие к минимальной яркости)
        shadow_threshold = 5
        shadow_clipped = np.sum(gray <= shadow_threshold)
        features['shadow_clipping_ratio'] = float(shadow_clipped / total_pixels)
        
        # Lighting Uniformity features
        uniformity_features = self._compute_lighting_uniformity(gray)
        features.update(uniformity_features)
        
        return features
    
    def _compute_lighting_uniformity(self, gray: np.ndarray) -> Dict[str, float]:
        """Вычисляет фичи равномерности освещения: uniformity_index, center/corner brightness, vignetting"""
        features = {}
        h, w = gray.shape
        
        # Разделяем кадр на сетку (3x3 для анализа распределения яркости)
        grid_h, grid_w = 3, 3
        cell_h, cell_w = h // grid_h, w // grid_w
        
        brightness_grid = []
        for i in range(grid_h):
            for j in range(grid_w):
                y_start = i * cell_h
                y_end = (i + 1) * cell_h if i < grid_h - 1 else h
                x_start = j * cell_w
                x_end = (j + 1) * cell_w if j < grid_w - 1 else w
                cell = gray[y_start:y_end, x_start:x_end]
                brightness_grid.append(np.mean(cell))
        
        brightness_grid = np.array(brightness_grid)
        
        # Lighting uniformity index (чем меньше вариация, тем равномернее)
        uniformity_std = np.std(brightness_grid)
        uniformity_mean = np.mean(brightness_grid)
        features['lighting_uniformity_index'] = float(1.0 - min(uniformity_std / (uniformity_mean + 1e-10), 1.0))
        
        # Center brightness (центральная ячейка сетки)
        center_idx = grid_h * grid_w // 2  # средняя ячейка
        features['center_brightness'] = float(brightness_grid[center_idx])
        
        # Corner brightness (среднее по угловым ячейкам)
        corner_indices = [0, grid_w - 1, (grid_h - 1) * grid_w, grid_h * grid_w - 1]
        corner_brightness = np.mean([brightness_grid[idx] for idx in corner_indices if idx < len(brightness_grid)])
        features['corner_brightness'] = float(corner_brightness)
        
        # Vignetting score (затемнение по краям) - чем больше разница центр/углы, тем сильнее виньетирование
        if features['center_brightness'] > 0:
            vignetting_ratio = features['corner_brightness'] / (features['center_brightness'] + 1e-10)
            features['vignetting_score'] = float(1.0 - min(vignetting_ratio, 1.0))  # 0 = нет виньетирования, 1 = сильное
        else:
            features['vignetting_score'] = 0.0
        
        return features
    
    def _compute_light_direction(self, frame: np.ndarray) -> Dict[str, float]:
        """Оценивает направление света: angle, source count, soft/hard probability"""
        features = {}
        
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
        h, w = gray.shape
        
        # Вычисляем градиенты для оценки направления света
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Угол направления света (основное направление градиента)
        angles = np.arctan2(grad_y, grad_x)
        # Усредняем по модулю градиента
        magnitudes = np.sqrt(grad_x**2 + grad_y**2)
        weighted_angle = np.average(angles, weights=magnitudes.flatten() + 1e-10)
        features['light_direction_angle'] = float(np.degrees(weighted_angle))
        
        # Оценка количества источников света (по пикам яркости)
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        peaks, _ = find_peaks(blurred.flatten(), height=np.percentile(blurred, 90), distance=w//10)
        features['light_source_count_estimate'] = float(min(len(peaks), 5))
        
        # Soft vs hard light (по резкости переходов)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        # Нормализуем (эмпирические значения)
        soft_threshold = 100
        hard_threshold = 500
        if laplacian_var < soft_threshold:
            features['soft_light_probability'] = 1.0
            features['hard_light_probability'] = 0.0
        elif laplacian_var > hard_threshold:
            features['soft_light_probability'] = 0.0
            features['hard_light_probability'] = 1.0
        else:
            # Линейная интерполяция
            ratio = (laplacian_var - soft_threshold) / (hard_threshold - soft_threshold)
            features['soft_light_probability'] = float(1.0 - ratio)
            features['hard_light_probability'] = float(ratio)
        
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
        
        # RGB статистики
        features.update(self._compute_rgb_stats(frame))
        
        # HSV фичи
        features.update(self._compute_hsv_features(frame))
        
        # LAB фичи
        features.update(self._compute_lab_features(frame))
        
        # Палитра и гармонии
        features.update(self._compute_palette_features(frame))
        
        # Освещение
        features.update(self._compute_lighting_features(frame))
        
        # Направление света
        features.update(self._compute_light_direction(frame))
        
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
        brightness_values = [f['features'].get('brightness_mean', 0) for f in frame_features]
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
    
    def extract_video_features(self, all_scene_features: List[Dict], all_frame_features: List[Dict]) -> Dict[str, Any]:
        """Извлекает video-level агрегированные фичи"""
        features = {}
        
        if not all_scene_features or not all_frame_features:
            return features
        
        # Агрегация всех сцен: mean/std/min/max по каждой фиче
        scene_feature_dict = {}
        for scene_feat in all_scene_features.values():
            for key, value in scene_feat.items():
                if isinstance(value, (int, float)) and not isinstance(value, bool):
                    if key not in scene_feature_dict:
                        scene_feature_dict[key] = []
                    scene_feature_dict[key].append(value)
        
        for key, values in scene_feature_dict.items():
            if len(values) > 0:
                features[f'{key}_mean'] = float(np.mean(values))
                features[f'{key}_std'] = float(np.std(values))
                features[f'{key}_min'] = float(np.min(values))
                features[f'{key}_max'] = float(np.max(values))
        
        # Entropy и Gini для распределений
        # Color distribution entropy
        hue_values = [f['features'].get('hue_mean', 0) for f in all_frame_features]
        if hue_values:
            hue_hist, _ = np.histogram(hue_values, bins=36)
            hue_probs = hue_hist / (hue_hist.sum() + 1e-10)
            features['color_distribution_entropy'] = float(entropy(hue_probs + 1e-10))
            features['color_distribution_gini'] = self._compute_gini_coefficient(np.array(hue_values))
        
        # Стиль цветокоррекции
        features.update(self._compute_color_style_features(all_frame_features))
        
        # Aesthetic & Cinematic Scores
        features.update(self._compute_aesthetic_scores(all_frame_features))
        
        # Глобальная динамика
        brightness_values = [f['features'].get('brightness_mean', 0) for f in all_frame_features]
        if len(brightness_values) > 1:
            brightness_diff = np.diff(brightness_values)
            features['global_brightness_change_speed'] = float(np.mean(np.abs(brightness_diff)))
        
        hue_values = [f['features'].get('hue_mean', 0) for f in all_frame_features]
        if len(hue_values) > 1:
            hue_diff = np.diff(hue_values)
            hue_diff = np.minimum(np.abs(hue_diff), 180 - np.abs(hue_diff))
            features['global_color_change_speed'] = float(np.mean(np.abs(hue_diff)))
        
        # Частота стробоскопических переходов
        if len(brightness_values) > 2:
            brightness_diff = np.abs(np.diff(brightness_values))
            strobe_threshold = np.mean(brightness_values) + 1.5 * np.std(brightness_values)
            strobe_count = np.sum(brightness_diff > strobe_threshold)
            features['strobe_transition_frequency'] = float(strobe_count / len(brightness_diff))
        
        # Periodicity и color shift (глобальные)
        if len(hue_values) > 3:
            hue_array = np.array(hue_values)
            autocorr = np.correlate(hue_array - np.mean(hue_array),
                                   hue_array - np.mean(hue_array), mode='full')
            autocorr = autocorr[len(autocorr)//2:]
            autocorr = autocorr / (autocorr[0] + 1e-10)
            if len(autocorr) > 2:
                peaks, _ = find_peaks(autocorr[1:], height=0.2)
                features['global_color_periodicity'] = float(len(peaks) / len(autocorr))
            else:
                features['global_color_periodicity'] = 0.0
            
            features['global_color_shift'] = float(np.mean(np.abs(np.diff(hue_values))))
        
        return features
    
    def _create_sequence_inputs(self, frame_features: List[Dict], scene_features: List[Dict], 
                                video_features: Dict) -> Dict[str, List]:
        """Создает sequence inputs для трансформера"""
        sequences = {}
        
        # Frame sequence features [N_frames, D_frame_features]
        frame_sequence = []
        for frame_feat in frame_features:
            feat_vec = []
            feat_dict = frame_feat['features']
            # Извлекаем числовые фичи в фиксированном порядке
            numeric_keys = sorted([k for k in feat_dict.keys() 
                                 if isinstance(feat_dict[k], (int, float)) and not isinstance(feat_dict[k], bool)])
            for key in numeric_keys:
                feat_vec.append(float(feat_dict[key]))
            frame_sequence.append(feat_vec)
        sequences['frames'] = frame_sequence
        
        # Scene sequence features [N_scenes, D_scene_features]
        scene_sequence = []
        for scene_feat in scene_features.values():
            feat_vec = []
            numeric_keys = sorted([k for k in scene_feat.keys() 
                                 if isinstance(scene_feat[k], (int, float)) and not isinstance(scene_feat[k], bool)])
            for key in numeric_keys:
                feat_vec.append(float(scene_feat[key]))
            scene_sequence.append(feat_vec)
        sequences['scenes'] = scene_sequence
        
        # Video global features [D_global_features]
        global_sequence = []
        numeric_keys = sorted([k for k in video_features.keys() 
                             if isinstance(video_features[k], (int, float)) and not isinstance(video_features[k], bool)])
        for key in numeric_keys:
            global_sequence.append(float(video_features[key]))
        sequences['global'] = global_sequence
        
        return sequences
    
    def process(self, frame_manager, scenes) -> Dict[str, Any]:
        """
        Главный метод обработки видео
        
        Args:
            frame_manager: FrameManager для доступа к кадрам
        
        Returns:
            Словарь с результатами в формате из README
        """
        # Обработка кадров по сценам
        all_frame_features = {}
        all_scene_features = {}
        
        for scene_label, frame_range in scenes.items():
            start_frame, end_frame = frame_range["indices"][0], frame_range["indices"][-1]
            
            # Ограничиваем количество кадров для оптимизации
            num_frames_in_scene = end_frame - start_frame + 1
            num_samples = min(self.max_frames_per_scene, num_frames_in_scene)
            
            if num_samples == 1:
                frame_indices = [start_frame]
            else:
                frame_indices = np.linspace(start_frame, end_frame, num_samples, dtype=int)
                # Убираем дубликаты и сортируем
                frame_indices = np.unique(frame_indices)

            all_frame_features[scene_label] = ()
            
            scene_frame_features = []
            for frame_idx in frame_indices:
                try:
                    frame = frame_manager.get(frame_idx)
                    frame_feat = self.extract_frame_features(frame, frame_idx)
                    scene_frame_features.append(frame_feat)
                    all_frame_features[scene_label][frame_idx] = frame_feat
                except Exception as e:
                    print(f"Warning: Failed to process frame {frame_idx}: {e}")
                    continue
            
            # Scene-level фичи
            if scene_frame_features:
                scene_feat = self.extract_scene_features(scene_frame_features, start_frame, end_frame)
                all_scene_features[scene_label] = scene_feat
        
        # Video-level фичи
        video_features = self.extract_video_features(all_scene_features, all_frame_features)
        
        # Sequence inputs
        sequence_inputs = self._create_sequence_inputs(all_frame_features, all_scene_features, video_features)
        
        # Формируем результат
        result = {
            "frames": all_frame_features,
            "scenes": all_scene_features,
            "video_features": video_features,
            "sequence_inputs": sequence_inputs
        }
        
        return result
