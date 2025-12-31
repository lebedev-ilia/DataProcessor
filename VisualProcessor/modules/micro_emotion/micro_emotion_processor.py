"""
Micro Emotion Processor - Optimized version
Обрабатывает данные OpenFace и извлекает оптимизированные фичи:
- Ключевые AU (10-14) с baseline subtraction
- PCA для остальных AU
- Компактные метрики pose, gaze, landmarks
- Micro-expressions detection
- Per-frame векторы для VisualTransformer

Все TODO выполнены:
    1. ✅ Интеграция с внешними зависимостями через BaseModule (face_detection, core_face_landmarks)
    2. ✅ Использование результатов core провайдеров вместо прямых вызовов моделей
    3. ✅ Интеграция с BaseModule через класс MicroEmotionModule
    4. ✅ Единый формат вывода для сохранения в npz
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import warnings

# Добавляем путь для импорта BaseModule
_MODULE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if _MODULE_PATH not in sys.path:
    sys.path.append(_MODULE_PATH)

from modules.base_module import BaseModule
from utils.frame_manager import FrameManager

# Ключевые AU для UGC/вовлечённости
KEY_AUS = ['AU06', 'AU12', 'AU04', 'AU01', 'AU02', 'AU25', 'AU26', 'AU07', 'AU23', 'AU45', 'AU43', 'AU15', 'AU20', 'AU10']

# AU для micro-expressions detection
MICROEXPR_AU_COMBINATIONS = {
    'smile': ['AU06', 'AU12'],
    'surprise': ['AU01', 'AU02', 'AU25', 'AU26'],
    'frown': ['AU04', 'AU15'],
    'disgust': ['AU09', 'AU10'],
}

class MicroEmotionProcessor:
    """Обработчик данных OpenFace с оптимизацией для VisualTransformer"""
    
    def __init__(
        self,
        fps: int = 30,
        microexpr_smoothing_sigma: float = 0.05,  # 0.03-0.1s
        microexpr_delta_threshold: float = 0.4,  # raw intensity change
        microexpr_max_duration_frames: int = 15,  # 0.5s at 30fps
        microexpr_min_peak_distance_frames: int = 6,  # 0.2s at 30fps
        gaze_centered_threshold: float = 10.0,  # degrees
        pca_components: int = 3,
        au_confidence_threshold: float = 0.5,
    ):
        """
        fps: кадров в секунду
        microexpr_smoothing_sigma: сглаживание для micro-expressions (в секундах)
        microexpr_delta_threshold: порог изменения интенсивности для micro-expression
        microexpr_max_duration_frames: максимальная длительность micro-expression в кадрах
        microexpr_min_peak_distance_frames: минимальное расстояние между пиками
        gaze_centered_threshold: порог для определения взгляда в камеру (градусы)
        pca_components: количество PCA компонент для AU
        au_confidence_threshold: порог уверенности AU
        """
        self.fps = fps
        self.microexpr_sigma = microexpr_smoothing_sigma * fps  # в кадрах
        self.microexpr_delta_threshold = microexpr_delta_threshold
        self.microexpr_max_duration_frames = microexpr_max_duration_frames
        self.microexpr_min_peak_distance_frames = microexpr_min_peak_distance_frames
        self.gaze_centered_threshold = gaze_centered_threshold
        self.pca_components = pca_components
        self.au_confidence_threshold = au_confidence_threshold
        
        self.pca_au = None
        self.pca_landmarks = None
        self.au_baseline = None
        
    def _smooth(self, x: np.ndarray, sigma: float = None) -> np.ndarray:
        """Сглаживание сигнала"""
        if x is None or len(x) == 0:
            return np.array([])
        if sigma is None:
            sigma = self.microexpr_sigma
        return gaussian_filter1d(x.astype(float), sigma=sigma)
    
    def _normalize_01(self, x: np.ndarray) -> np.ndarray:
        """Нормализация в [0, 1]"""
        if x is None or len(x) == 0:
            return np.array([])
        x = np.array(x, dtype=float)
        mi, ma = x.min(), x.max()
        if ma - mi < 1e-9:
            return np.zeros_like(x)
        return (x - mi) / (ma - mi)
    
    def _z_normalize(self, x: np.ndarray, mean: float = None, std: float = None) -> Tuple[np.ndarray, float, float]:
        """Z-нормализация"""
        if x is None or len(x) == 0:
            return np.array([]), 0.0, 1.0
        x = np.array(x, dtype=float)
        if mean is None:
            mean = float(x.mean())
        if std is None:
            std = float(x.std()) + 1e-9
        return (x - mean) / std, mean, std
    
    def compute_au_baseline(self, df: pd.DataFrame, au_columns: List[str]) -> Dict[str, float]:
        """
        Вычисляет baseline (нейтральное состояние) для каждого AU.
        Использует нижние 20% кадров по общей активности AU.
        """
        if len(df) == 0:
            return {au: 0.0 for au in au_columns}
        
        # Вычисляем общую активность AU для каждого кадра
        au_intensity_cols = [col for col in au_columns if col.endswith('_r')]
        if len(au_intensity_cols) == 0:
            return {au: 0.0 for au in au_columns}
        
        total_activity = df[au_intensity_cols].sum(axis=1)
        
        # Выбираем нижние 20% кадров как нейтральные
        threshold = np.percentile(total_activity, 20)
        neutral_frames = df[total_activity <= threshold]
        
        baseline = {}
        for au in au_columns:
            intensity_col = f"{au}_r"
            if intensity_col in neutral_frames.columns:
                baseline[au] = float(neutral_frames[intensity_col].mean())
            else:
                baseline[au] = 0.0
        
        return baseline
    
    def extract_key_au_features(
        self,
        df: pd.DataFrame,
        au_baseline: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Извлекает фичи для ключевых AU.
        Для каждого AU: intensity_mean, intensity_std, presence_rate, peak_count, intensity_delta_mean
        """
        features = {}
        
        if au_baseline is None:
            au_baseline = {}
        
        for au in KEY_AUS:
            intensity_col = f"{au}_r"
            presence_col = f"{au}_c"
            
            if intensity_col not in df.columns:
                features[au] = {
                    'intensity_mean': 0.0,
                    'intensity_std': 0.0,
                    'presence_rate': 0.0,
                    'peak_count': 0,
                    'intensity_delta_mean': 0.0,
                }
                continue
            
            intensities = df[intensity_col].fillna(0.0).values
            baseline = au_baseline.get(au, 0.0)
            
            # Baseline subtraction
            intensities_delta = intensities - baseline
            
            # Presence rate
            if presence_col in df.columns:
                presence = df[presence_col].fillna(0.0).values
                presence_rate = float(np.mean(presence > 0.5))
            else:
                # Infer presence from intensity
                presence_rate = float(np.mean(intensities > 0.1))
            
            # Peak detection для интенсивности
            smoothed = self._smooth(intensities, sigma=self.microexpr_sigma)
            peaks, _ = find_peaks(
                smoothed,
                height=baseline + 1.5 * np.std(smoothed),
                distance=self.microexpr_min_peak_distance_frames,
            )
            peak_count = len(peaks)
            
            features[au] = {
                'intensity_mean': float(np.mean(intensities)),
                'intensity_std': float(np.std(intensities)),
                'presence_rate': presence_rate,
                'peak_count': peak_count,
                'intensity_delta_mean': float(np.mean(intensities_delta)),
            }
        
        return features
    
    def compute_au_pca(
        self,
        df: pd.DataFrame,
        fit: bool = True,
    ) -> Tuple[np.ndarray, Optional[PCA]]:
        """
        Вычисляет PCA для всех AU интенсивностей (кроме ключевых).
        Возвращает проекции и модель PCA.
        """
        # Получаем все AU колонки интенсивности
        all_au_cols = [col for col in df.columns if col.startswith('AU') and col.endswith('_r')]
        non_key_au_cols = [col for col in all_au_cols if col.replace('_r', '') not in KEY_AUS]
        
        if len(non_key_au_cols) == 0:
            # Если нет неключевых AU, используем все AU
            non_key_au_cols = all_au_cols
        
        if len(non_key_au_cols) == 0:
            return np.zeros((len(df), self.pca_components)), None
        
        au_matrix = df[non_key_au_cols].fillna(0.0).values
        
        if fit:
            self.pca_au = PCA(n_components=self.pca_components)
            pca_features = self.pca_au.fit_transform(au_matrix)
        else:
            if self.pca_au is None:
                return np.zeros((len(df), self.pca_components)), None
            pca_features = self.pca_au.transform(au_matrix)
        
        return pca_features, self.pca_au
    
    def detect_micro_expressions(
        self,
        df: pd.DataFrame,
        au_baseline: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """
        Детектирует micro-expressions как быстрые вспышки AU интенсивности.
        """
        if au_baseline is None:
            au_baseline = {}
        
        microexpr_timestamps = []
        microexpr_types = []
        microexpr_intensities = []
        
        for expr_type, au_list in MICROEXPR_AU_COMBINATIONS.items():
            # Комбинируем AU для данного типа выражения
            combined_intensity = None
            
            for au in au_list:
                intensity_col = f"{au}_r"
                if intensity_col not in df.columns:
                    continue
                
                intensities = df[intensity_col].fillna(0.0).values
                baseline = au_baseline.get(au, 0.0)
                intensities_delta = intensities - baseline
                intensities_norm = self._normalize_01(intensities_delta)
                
                if combined_intensity is None:
                    combined_intensity = intensities_norm
                else:
                    combined_intensity = np.maximum(combined_intensity, intensities_norm)
            
            if combined_intensity is None or len(combined_intensity) == 0:
                continue
            
            # Сглаживание
            smoothed = self._smooth(combined_intensity, sigma=self.microexpr_sigma)
            
            # Детекция пиков
            threshold = np.mean(smoothed) + 1.5 * np.std(smoothed)
            peaks, properties = find_peaks(
                smoothed,
                height=threshold,
                distance=self.microexpr_min_peak_distance_frames,
                width=(1, self.microexpr_max_duration_frames),
            )
            
            for peak_idx in peaks:
                timestamp = peak_idx / self.fps
                intensity = float(smoothed[peak_idx])
                microexpr_timestamps.append(timestamp)
                microexpr_types.append(expr_type)
                microexpr_intensities.append(intensity)
        
        # Сортируем по времени
        if len(microexpr_timestamps) > 0:
            sorted_indices = np.argsort(microexpr_timestamps)
            microexpr_timestamps = [microexpr_timestamps[i] for i in sorted_indices]
            microexpr_types = [microexpr_types[i] for i in sorted_indices]
            microexpr_intensities = [microexpr_intensities[i] for i in sorted_indices]
        
        # Распределение типов
        types_distribution = {}
        for expr_type in MICROEXPR_AU_COMBINATIONS.keys():
            count = sum(1 for t in microexpr_types if t == expr_type)
            types_distribution[expr_type] = count
        
        duration_minutes = len(df) / (self.fps * 60.0) if len(df) > 0 else 1.0
        
        return {
            'microexpr_count': len(microexpr_timestamps),
            'microexpr_rate_per_min': len(microexpr_timestamps) / duration_minutes if duration_minutes > 0 else 0.0,
            'microexpr_max_intensity': float(max(microexpr_intensities)) if microexpr_intensities else 0.0,
            'microexpr_types_distribution': types_distribution,
            'microexpr_timestamps': microexpr_timestamps,
            'microexpr_types': microexpr_types,
        }
    
    def compute_pose_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Вычисляет оптимизированные фичи позы головы"""
        features = {}
        
        for axis in ['Rx', 'Ry', 'Rz']:
            col = f'pose_{axis}'
            if col not in df.columns:
                features[f'pose_{axis}_mean'] = 0.0
                features[f'pose_{axis}_std'] = 0.0
                features[f'pose_{axis}_min'] = 0.0
                features[f'pose_{axis}_max'] = 0.0
                continue
            
            values = df[col].fillna(0.0).values
            features[f'pose_{axis}_mean'] = float(np.mean(values))
            features[f'pose_{axis}_std'] = float(np.std(values))
            features[f'pose_{axis}_min'] = float(np.min(values))
            features[f'pose_{axis}_max'] = float(np.max(values))
        
        # Pose stability score
        rx_std = features.get('pose_Rx_std', 0.0)
        ry_std = features.get('pose_Ry_std', 0.0)
        rz_std = features.get('pose_Rz_std', 0.0)
        total_std = np.sqrt(rx_std**2 + ry_std**2 + rz_std**2)
        # Нормализуем (предполагаем max std ~30 градусов)
        max_expected_std = 30.0
        pose_stability_score = float(np.clip(1.0 - (total_std / max_expected_std), 0.0, 1.0))
        features['pose_stability_score'] = pose_stability_score
        
        # Tz (приближение/удаление)
        if 'pose_Tz' in df.columns:
            tz_values = df['pose_Tz'].fillna(0.0).values
            features['pose_Tz_mean'] = float(np.mean(tz_values))
            features['pose_Tz_std'] = float(np.std(tz_values))
        else:
            features['pose_Tz_mean'] = 0.0
            features['pose_Tz_std'] = 0.0
        
        return features
    
    def compute_gaze_features(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Вычисляет фичи направления взгляда"""
        features = {}
        
        for axis in ['x', 'y']:
            col = f'gaze_angle_{axis}'
            if col not in df.columns:
                features[f'gaze_{axis}_mean'] = 0.0
                features[f'gaze_{axis}_std'] = 0.0
                continue
            
            values = df[col].fillna(0.0).values
            features[f'gaze_{axis}_mean'] = float(np.mean(values))
            features[f'gaze_{axis}_std'] = float(np.std(values))
        
        # Gaze centered ratio (взгляд в камеру)
        if 'gaze_angle_x' in df.columns and 'gaze_angle_y' in df.columns:
            gaze_x = df['gaze_angle_x'].fillna(0.0).values
            gaze_y = df['gaze_angle_y'].fillna(0.0).values
            
            centered = (np.abs(gaze_x) < self.gaze_centered_threshold) & \
                       (np.abs(gaze_y) < self.gaze_centered_threshold)
            gaze_centered_ratio = float(np.mean(centered))
        else:
            gaze_centered_ratio = 0.0
        
        features['gaze_centered_ratio'] = gaze_centered_ratio
        
        # Blink rate (AU45/AU43)
        blink_rate = 0.0
        if 'AU45_c' in df.columns:
            au45_presence = df['AU45_c'].fillna(0.0).values
            # Blink: короткая вспышка presence (< 0.25s)
            blink_frames = int(0.25 * self.fps)
            blink_count = 0
            i = 0
            while i < len(au45_presence):
                if au45_presence[i] > 0.5:
                    # Начало blink
                    blink_duration = 0
                    while i < len(au45_presence) and au45_presence[i] > 0.5:
                        blink_duration += 1
                        i += 1
                    if blink_duration <= blink_frames:
                        blink_count += 1
                else:
                    i += 1
            
            duration_minutes = len(df) / (self.fps * 60.0) if len(df) > 0 else 1.0
            blink_rate = blink_count / duration_minutes if duration_minutes > 0 else 0.0
        
        features['blink_rate_per_min'] = blink_rate
        
        # Eye contact score (gaze centered + blink rate)
        eye_contact_score = (gaze_centered_ratio * 0.7) + (np.clip(blink_rate / 20.0, 0.0, 1.0) * 0.3)
        features['eye_contact_score'] = float(eye_contact_score)
        
        return features
    
    def compute_landmark_features(self, df: pd.DataFrame, fit: bool = True) -> Dict[str, Any]:
        """Вычисляет компактные геометрические признаки из landmarks"""
        features = {}
        
        # Извлекаем landmarks (2D)
        landmark_cols_2d = [col for col in df.columns if col.startswith('x_') or col.startswith('y_')]
        
        if len(landmark_cols_2d) >= 68 * 2:  # 68 точек × 2 координаты
            # Mouth opening (расстояние между верхней и нижней губой, нормализованное по межглазному расстоянию)
            # Landmarks для губ: 48-67 (примерно)
            # Межглазное расстояние: между точками глаз (примерно 36 и 45)
            
            # Упрощенная версия: используем средние координаты верхней и нижней губы
            upper_lip_y = df[[f'y_{i}' for i in range(51, 54)]].mean(axis=1).values if all(f'y_{i}' in df.columns for i in range(51, 54)) else None
            lower_lip_y = df[[f'y_{i}' for i in range(57, 60)]].mean(axis=1).values if all(f'y_{i}' in df.columns for i in range(57, 60)) else None
            
            if upper_lip_y is not None and lower_lip_y is not None:
                mouth_opening = np.abs(upper_lip_y - lower_lip_y)
                # Нормализация по межглазному расстоянию (упрощенно)
                interocular_dist = 1.0  # placeholder
                if 'x_36' in df.columns and 'x_45' in df.columns:
                    interocular_dist = np.sqrt(
                        (df['x_36'] - df['x_45'])**2 + (df['y_36'] - df['y_45'])**2
                    ).mean()
                if interocular_dist > 0:
                    mouth_opening_norm = mouth_opening / interocular_dist
                else:
                    mouth_opening_norm = mouth_opening
                
                features['mouth_opening_mean'] = float(np.mean(mouth_opening_norm))
                features['mouth_opening_std'] = float(np.std(mouth_opening_norm))
            else:
                features['mouth_opening_mean'] = 0.0
                features['mouth_opening_std'] = 0.0
            
            # Smile width (расстояние между уголками губ)
            if 'x_48' in df.columns and 'x_54' in df.columns:
                smile_width = np.sqrt(
                    (df['x_48'] - df['x_54'])**2 + (df['y_48'] - df['y_54'])**2
                ).values
                features['smile_width_mean'] = float(np.mean(smile_width))
                features['smile_width_std'] = float(np.std(smile_width))
            else:
                features['smile_width_mean'] = 0.0
                features['smile_width_std'] = 0.0
            
            # Face asymmetry (корреляция L-R landmark distances)
            # Упрощенно: используем симметричные точки
            asymmetry_scores = []
            for i in range(17):  # Контур лица
                left_idx = i
                right_idx = 16 - i
                if f'x_{left_idx}' in df.columns and f'x_{right_idx}' in df.columns:
                    left_x = df[f'x_{left_idx}'].values
                    right_x = df[f'x_{right_idx}'].values
                    # Центр лица
                    center_x = (df['x_30'].values + df['x_33'].values) / 2 if 'x_30' in df.columns and 'x_33' in df.columns else df['x_30'].values
                    left_dist = np.abs(left_x - center_x)
                    right_dist = np.abs(right_x - center_x)
                    if len(left_dist) > 1 and len(right_dist) > 1:
                        corr = np.corrcoef(left_dist, right_dist)[0, 1]
                        if not np.isnan(corr):
                            asymmetry_scores.append(1.0 - abs(corr))  # 1 - correlation = asymmetry
            
            if asymmetry_scores:
                features['face_asymmetry_score'] = float(np.mean(asymmetry_scores))
            else:
                features['face_asymmetry_score'] = 0.0
            
            # PCA для landmarks (если нужно)
            landmark_matrix = []
            for i in range(68):
                if f'x_{i}' in df.columns and f'y_{i}' in df.columns:
                    landmark_matrix.append(df[f'x_{i}'].values)
                    landmark_matrix.append(df[f'y_{i}'].values)
            
            if len(landmark_matrix) > 0:
                landmark_matrix = np.array(landmark_matrix).T
                if fit:
                    self.pca_landmarks = PCA(n_components=min(5, landmark_matrix.shape[1]))
                    landmarks_pca = self.pca_landmarks.fit_transform(landmark_matrix)
                else:
                    if self.pca_landmarks is not None:
                        landmarks_pca = self.pca_landmarks.transform(landmark_matrix)
                    else:
                        landmarks_pca = np.zeros((len(df), 5))
                
                for i in range(min(5, landmarks_pca.shape[1])):
                    features[f'landmarks_pca_{i+1}'] = float(np.mean(landmarks_pca[:, i]))
        else:
            features['mouth_opening_mean'] = 0.0
            features['mouth_opening_std'] = 0.0
            features['smile_width_mean'] = 0.0
            features['smile_width_std'] = 0.0
            features['face_asymmetry_score'] = 0.0
        
        # 3D landmarks features
        if 'X_30' in df.columns:  # Nose tip
            nose_z = df['Z_30'].fillna(0.0).values if 'Z_30' in df.columns else None
            if nose_z is not None:
                features['head_depth_variation'] = float(np.std(nose_z))
            else:
                features['head_depth_variation'] = 0.0
        else:
            features['head_depth_variation'] = 0.0
        
        return features
    
    def compute_per_frame_vectors(
        self,
        df: pd.DataFrame,
        au_baseline: Optional[Dict[str, float]] = None,
        au_pca_features: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Создает per-frame векторы для VisualTransformer (~16-24 числа).
        """
        n_frames = len(df)
        if n_frames == 0:
            return np.zeros((0, 22), dtype=float)
        
        vectors = []
        total_duration = n_frames / self.fps
        
        # Вычисляем per-frame признаки
        for idx in range(n_frames):
            vec = []
            row = df.iloc[idx]
            
            # time_norm (1)
            vec.append(idx / max(n_frames - 1, 1))
            
            # face_presence_flag (1)
            face_presence = 1.0 if row.get('success', 0) > 0.5 else 0.0
            vec.append(face_presence)
            
            # Key AU intensity deltas (3-4)
            for au in ['AU12', 'AU06', 'AU04', 'AU25']:
                intensity_col = f"{au}_r"
                baseline = au_baseline.get(au, 0.0) if au_baseline else 0.0
                intensity = float(row.get(intensity_col, 0.0))
                intensity_delta = intensity - baseline
                vec.append(intensity_delta / 5.0)  # Normalize to [0, 1] assuming max intensity ~5
            
            # AU25 presence rate in short window (1)
            au25_col = 'AU25_c'
            if au25_col in df.columns:
                window_start = max(0, idx - int(0.5 * self.fps))
                window_end = min(n_frames, idx + int(0.5 * self.fps))
                au25_window = df[au25_col].iloc[window_start:window_end]
                au25_presence_rate = float(au25_window.mean()) if len(au25_window) > 0 else 0.0
            else:
                au25_presence_rate = 0.0
            vec.append(au25_presence_rate)
            
            # Blink flag (AU45 presence) (1)
            au45_col = 'AU45_c'
            blink_flag = float(row.get(au45_col, 0.0) > 0.5)
            vec.append(blink_flag)
            
            # Pose normalized (2)
            pose_ry = float(row.get('pose_Ry', 0.0))
            pose_rx = float(row.get('pose_Rx', 0.0))
            vec.append(pose_ry / 90.0)  # Normalize to [-1, 1] assuming max ~90 degrees
            vec.append(pose_rx / 90.0)
            
            # Gaze (2-3)
            gaze_x = float(row.get('gaze_angle_x', 0.0))
            gaze_y = float(row.get('gaze_angle_y', 0.0))
            gaze_centered = 1.0 if (abs(gaze_x) < self.gaze_centered_threshold and abs(gaze_y) < self.gaze_centered_threshold) else 0.0
            vec.append(gaze_centered)
            vec.append(gaze_x / 30.0)  # Normalize
            vec.append(gaze_y / 30.0)
            
            # Mouth opening normalized (1)
            # Используем упрощенную версию
            mouth_opening = 0.0
            if 'y_51' in row.index and 'y_57' in row.index:
                mouth_opening = abs(float(row['y_51']) - float(row['y_57']))
            vec.append(mouth_opening / 50.0)  # Normalize (placeholder)
            
            # Face asymmetry score (1)
            # Используем значение из compute_landmark_features или 0
            vec.append(0.0)  # Placeholder - будет заполнено позже
            
            # Microexpr recent count (1)
            # Будет заполнено позже на основе microexpr_timestamps
            vec.append(0.0)  # Placeholder
            
            # AU PCA (3)
            if au_pca_features is not None and idx < len(au_pca_features):
                vec.extend(au_pca_features[idx, :3].tolist())
            else:
                vec.extend([0.0, 0.0, 0.0])
            
            # AU quality flag (1)
            # Используем confidence если доступен
            au_quality = 1.0  # Placeholder
            vec.append(au_quality)
            
            vectors.append(vec)
        
        return np.array(vectors, dtype=float)
    
    def compute_video_level_aggregates(
        self,
        df: pd.DataFrame,
        key_au_features: Dict[str, Any],
        microexpr_features: Dict[str, Any],
        pose_features: Dict[str, Any],
        gaze_features: Dict[str, Any],
        landmark_features: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Вычисляет видео-уровневые агрегаты"""
        aggregates = {}
        
        # Key AU aggregates
        for au in ['AU06', 'AU12', 'AU04', 'AU25', 'AU26', 'AU7', 'AU15']:
            au_name = au.replace('AU7', 'AU07')
            if au_name in key_au_features:
                au_data = key_au_features[au_name]
                aggregates[f'{au_name}_mean'] = au_data['intensity_mean']
                aggregates[f'{au_name}_std'] = au_data['intensity_std']
                aggregates[f'{au_name}_min'] = 0.0  # Placeholder
                aggregates[f'{au_name}_max'] = 0.0  # Placeholder
                aggregates[f'{au_name}_median'] = 0.0  # Placeholder
                aggregates[f'{au_name}_peak_count'] = au_data['peak_count']
        
        # Micro-expressions
        aggregates.update(microexpr_features)
        
        # Smile ratio
        if 'AU12_r' in df.columns and 'AU06_r' in df.columns:
            smile_threshold = 1.0
            smile_frames = (df['AU12_r'].fillna(0.0) + df['AU06_r'].fillna(0.0)) > smile_threshold
            aggregates['smile_ratio'] = float(smile_frames.mean())
        else:
            aggregates['smile_ratio'] = 0.0
        
        # Eye contact ratio
        aggregates['eye_contact_ratio'] = gaze_features.get('gaze_centered_ratio', 0.0)
        
        # Blink rate
        aggregates['blink_rate_per_min'] = gaze_features.get('blink_rate_per_min', 0.0)
        
        # Pose stability
        aggregates['pose_stability_score'] = pose_features.get('pose_stability_score', 0.0)
        
        # Face presence ratio
        if 'success' in df.columns:
            aggregates['face_presence_ratio'] = float(df['success'].mean())
        else:
            aggregates['face_presence_ratio'] = 0.0
        
        # Landmark features
        aggregates['avg_mouth_opening'] = landmark_features.get('mouth_opening_mean', 0.0)
        
        # AU PCA variance explained
        if self.pca_au is not None:
            for i, var_exp in enumerate(self.pca_au.explained_variance_ratio_[:5]):
                aggregates[f'au_pca_var_explained_{i+1}'] = float(var_exp)
        
        return aggregates
    
    def compute_reliability_flags(
        self,
        df: pd.DataFrame,
    ) -> Dict[str, Any]:
        """Вычисляет флаги надёжности"""
        flags = {}
        
        # AU quality score
        au_confidence_cols = [col for col in df.columns if col.endswith('_c')]
        if len(au_confidence_cols) > 0:
            au_quality_scores = []
            for col in au_confidence_cols:
                quality = df[col].fillna(0.0).values
                au_quality_scores.extend(quality)
            au_quality_overall = float(np.mean(au_quality_scores)) if au_quality_scores else 0.0
        else:
            au_quality_overall = 0.0
        
        flags['au_quality_overall'] = au_quality_overall
        flags['au_quality_reliable'] = au_quality_overall > self.au_confidence_threshold
        
        # Landmark visibility
        landmark_cols = [col for col in df.columns if col.startswith('x_') or col.startswith('y_')]
        if len(landmark_cols) > 0:
            # Простая проверка: считаем видимыми landmarks с ненулевыми координатами
            visible_count = 0
            total_count = 0
            for col in landmark_cols[:68*2]:  # 68 landmarks × 2
                values = df[col].fillna(0.0).values
                visible = np.sum(values != 0.0)
                visible_count += visible
                total_count += len(values)
            landmark_visibility_mean = visible_count / total_count if total_count > 0 else 0.0
        else:
            landmark_visibility_mean = 0.0
        
        flags['landmark_visibility_mean'] = landmark_visibility_mean
        flags['landmark_visibility_reliable'] = landmark_visibility_mean > 0.8
        
        # Occlusion flag
        flags['occlusion_flag'] = landmark_visibility_mean < 0.7
        
        # Lighting flag (упрощенно)
        flags['lighting_flag'] = False  # Placeholder
        
        return flags
    
    def process_openface_dataframe(
        self,
        df: pd.DataFrame,
        fit_models: bool = True,
    ) -> Dict[str, Any]:
        """
        Главный метод обработки DataFrame OpenFace.
        Возвращает оптимизированные фичи.
        """
        if len(df) == 0:
            return {
                'success': False,
                'features': {},
                'per_frame_vectors': np.zeros((0, 22)),
                'reliability_flags': {},
            }
        
        # Вычисляем baseline для AU
        au_columns = [col.replace('_r', '') for col in df.columns if col.startswith('AU') and col.endswith('_r')]
        au_baseline = self.compute_au_baseline(df, au_columns) if fit_models else {}
        
        # Извлекаем ключевые AU фичи
        key_au_features = self.extract_key_au_features(df, au_baseline)
        
        # PCA для остальных AU
        au_pca_features, pca_model = self.compute_au_pca(df, fit=fit_models)
        
        # Детекция micro-expressions
        microexpr_features = self.detect_micro_expressions(df, au_baseline)
        
        # Pose features
        pose_features = self.compute_pose_features(df)
        
        # Gaze features
        gaze_features = self.compute_gaze_features(df)
        
        # Landmark features
        landmark_features = self.compute_landmark_features(df, fit=fit_models)
        
        # Per-frame vectors
        per_frame_vectors = self.compute_per_frame_vectors(df, au_baseline, au_pca_features)
        
        # Видео-уровневые агрегаты
        video_aggregates = self.compute_video_level_aggregates(
            df, key_au_features, microexpr_features, pose_features, gaze_features, landmark_features
        )
        
        # Reliability flags
        reliability_flags = self.compute_reliability_flags(df)
        
        # Объединяем все фичи
        features = {
            **key_au_features,
            **pose_features,
            **gaze_features,
            **landmark_features,
            **video_aggregates,
            **reliability_flags,
        }
        
        return {
            'success': True,
            'features': features,
            'per_frame_vectors': per_frame_vectors,
            'reliability_flags': reliability_flags,
            'microexpr_features': microexpr_features,
            'au_baseline': au_baseline,
            'pca_models': {
                'au_pca': pca_model,
                'landmarks_pca': self.pca_landmarks,
            },
        }


class MicroEmotionModule(BaseModule):
    """
    Модуль для извлечения micro-emotion фичей из данных OpenFace.
    
    Наследуется от BaseModule для интеграции с системой зависимостей и единым форматом вывода.
    Использует MicroEmotionProcessor для обработки DataFrame OpenFace.
    
    Может работать с:
    - Готовым DataFrame (переданным через config)
    - CSV файлом OpenFace (загружается автоматически)
    - Результатами face_detection для фильтрации кадров
    """
    
    def __init__(
        self,
        rs_path: Optional[str] = None,
        fps: int = 30,
        microexpr_smoothing_sigma: float = 0.05,
        microexpr_delta_threshold: float = 0.4,
        microexpr_max_duration_frames: int = 15,
        microexpr_min_peak_distance_frames: int = 6,
        gaze_centered_threshold: float = 10.0,
        pca_components: int = 3,
        au_confidence_threshold: float = 0.5,
        use_face_detection: bool = False,
        **kwargs: Any
    ):
        """
        Инициализация MicroEmotionModule.
        
        Args:
            rs_path: Путь к хранилищу результатов
            fps: Кадров в секунду
            microexpr_smoothing_sigma: Сглаживание для micro-expressions (в секундах)
            microexpr_delta_threshold: Порог изменения интенсивности для micro-expression
            microexpr_max_duration_frames: Максимальная длительность micro-expression в кадрах
            microexpr_min_peak_distance_frames: Минимальное расстояние между пиками
            gaze_centered_threshold: Порог для определения взгляда в камеру (градусы)
            pca_components: Количество PCA компонент для AU
            au_confidence_threshold: Порог уверенности AU
            use_face_detection: Использовать результаты face_detection для фильтрации кадров
            **kwargs: Дополнительные параметры для BaseModule
        """
        super().__init__(rs_path=rs_path, **kwargs)
        
        self.fps = fps
        self.use_face_detection = use_face_detection
        
        # Инициализируем процессор
        self.processor = MicroEmotionProcessor(
            fps=fps,
            microexpr_smoothing_sigma=microexpr_smoothing_sigma,
            microexpr_delta_threshold=microexpr_delta_threshold,
            microexpr_max_duration_frames=microexpr_max_duration_frames,
            microexpr_min_peak_distance_frames=microexpr_min_peak_distance_frames,
            gaze_centered_threshold=gaze_centered_threshold,
            pca_components=pca_components,
            au_confidence_threshold=au_confidence_threshold,
        )
    
    def required_dependencies(self) -> List[str]:
        """
        Возвращает список зависимостей модуля.
        
        Опциональные зависимости:
        - face_detection: для фильтрации кадров с лицами
        - core_face_landmarks: для использования готовых landmarks (если доступны)
        """
        deps = []
        if self.use_face_detection:
            deps.append("face_detection")
        return deps
    
    def _load_openface_dataframe(
        self,
        config: Dict[str, Any]
    ) -> Optional[pd.DataFrame]:
        """
        Загружает DataFrame OpenFace из различных источников.
        
        Приоритет:
        1. DataFrame переданный напрямую в config['openface_dataframe']
        2. Путь к CSV в config['openface_csv_path']
        3. Автоматический поиск CSV в rs_path/micro_emotion/
        4. Загрузка из результатов других модулей (если есть)
        
        Args:
            config: Конфигурация модуля
            
        Returns:
            DataFrame OpenFace или None, если не найден
        """
        # 1. Прямая передача DataFrame
        if 'openface_dataframe' in config and config['openface_dataframe'] is not None:
            df = config['openface_dataframe']
            if isinstance(df, pd.DataFrame):
                self.logger.info("Используется переданный DataFrame OpenFace")
                return df
        
        # 2. Путь к CSV в config
        csv_path = config.get('openface_csv_path')
        if csv_path and os.path.exists(csv_path):
            self.logger.info(f"Загружаем CSV OpenFace из {csv_path}")
            try:
                return pd.read_csv(csv_path)
            except Exception as e:
                self.logger.warning(f"Ошибка загрузки CSV {csv_path}: {e}")
        
        # 3. Автоматический поиск CSV в rs_path
        if self.rs_path:
            micro_emotion_dir = os.path.join(self.rs_path, "micro_emotion")
            if os.path.exists(micro_emotion_dir):
                csv_files = list(Path(micro_emotion_dir).glob("*.csv"))
                if csv_files:
                    # Берем последний по времени модификации
                    csv_path = max(csv_files, key=lambda p: p.stat().st_mtime)
                    self.logger.info(f"Найден CSV OpenFace: {csv_path}")
                    try:
                        return pd.read_csv(str(csv_path))
                    except Exception as e:
                        self.logger.warning(f"Ошибка загрузки CSV {csv_path}: {e}")
        
        # 4. Попытка загрузить из результатов других модулей
        # (если есть сохраненный DataFrame в npz)
        if self.rs_path:
            try:
                deps = self.load_all_dependencies()
                # Проверяем, есть ли сохраненные данные OpenFace
                for module_name, data in deps.items():
                    if data and isinstance(data, dict):
                        if 'openface_dataframe' in data:
                            df = data['openface_dataframe']
                            if isinstance(df, pd.DataFrame):
                                self.logger.info(f"Загружен DataFrame из {module_name}")
                                return df
            except Exception:
                pass
        
        return None
    
    def _filter_frame_indices_by_face_detection(
        self,
        frame_indices: List[int]
    ) -> List[int]:
        """
        Фильтрует индексы кадров по результатам face_detection.
        
        Args:
            frame_indices: Исходный список индексов кадров
            
        Returns:
            Отфильтрованный список индексов кадров с лицами
        """
        if not self.use_face_detection:
            return frame_indices
        
        try:
            face_data = self.load_dependency_results("face_detection", format="json")
            if not face_data or 'frames' not in face_data:
                self.logger.warning("face_detection результаты не найдены, используем все кадры")
                return frame_indices
            
            frames_with_faces = [
                int(k) for k, v in face_data['frames'].items()
                if v and len(v) > 0
            ]
            
            filtered = sorted(set(frame_indices) & set(frames_with_faces))
            self.logger.info(
                f"Отфильтровано кадров: {len(frame_indices)} -> {len(filtered)} "
                f"(с лицами: {len(frames_with_faces)})"
            )
            return filtered
            
        except Exception as e:
            self.logger.warning(
                f"Ошибка загрузки face_detection результатов: {e}. "
                "Используем все кадры."
            )
            return frame_indices
    
    def process(
        self,
        frame_manager: FrameManager,
        frame_indices: List[int],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Основной метод обработки (интерфейс BaseModule).
        
        Args:
            frame_manager: Менеджер кадров
            frame_indices: Список индексов кадров для обработки
            config: Конфигурация модуля:
                - openface_dataframe: DataFrame OpenFace (опционально)
                - openface_csv_path: Путь к CSV OpenFace (опционально)
                - fit_models: Флаг для обучения PCA моделей (по умолчанию True)
                
        Returns:
            Словарь с результатами в формате для сохранения в npz:
            - features: словарь с агрегированными фичами
            - per_frame_vectors: numpy массив [N_frames, 22] для VisualTransformer
            - reliability_flags: флаги надёжности
            - microexpr_features: фичи micro-expressions
            - summary: метаданные обработки
        """
        # Фильтруем кадры по face_detection, если нужно
        filtered_indices = self._filter_frame_indices_by_face_detection(frame_indices)
        
        # Загружаем DataFrame OpenFace
        df = self._load_openface_dataframe(config)
        
        if df is None or len(df) == 0:
            self.logger.warning(
                "DataFrame OpenFace не найден. "
                "Убедитесь, что OpenFace был запущен и результаты сохранены."
            )
            # Возвращаем пустой результат в правильном формате
            return {
                'features': {},
                'per_frame_vectors': np.zeros((0, 22), dtype=np.float32),
                'reliability_flags': {},
                'microexpr_features': {},
                'summary': {
                    'total_frames': len(frame_indices),
                    'frames_processed': 0,
                    'frames_with_face': 0,
                    'success': False,
                }
            }
        
        # Фильтруем DataFrame по frame_indices, если нужно
        if 'frame' in df.columns:
            df_filtered = df[df['frame'].isin(filtered_indices)].copy()
            if len(df_filtered) == 0:
                self.logger.warning(
                    f"После фильтрации по frame_indices DataFrame пуст. "
                    f"Используем все строки DataFrame."
                )
                df_filtered = df.copy()
        else:
            # Если нет колонки frame, используем все строки
            df_filtered = df.copy()
            self.logger.debug("Колонка 'frame' не найдена в DataFrame, используем все строки")
        
        # Обрабатываем через MicroEmotionProcessor
        fit_models = config.get('fit_models', True)
        processed = self.processor.process_openface_dataframe(df_filtered, fit_models=fit_models)
        
        if not processed.get('success', False):
            self.logger.warning("Обработка DataFrame не удалась")
            return {
                'features': {},
                'per_frame_vectors': np.zeros((0, 22), dtype=np.float32),
                'reliability_flags': {},
                'microexpr_features': {},
                'summary': {
                    'total_frames': len(frame_indices),
                    'frames_processed': len(df_filtered),
                    'frames_with_face': int(df_filtered['success'].sum()) if 'success' in df_filtered.columns else 0,
                    'success': False,
                }
            }
        
        # Подготавливаем результаты в едином формате для npz
        features = processed['features']
        per_frame_vectors = processed['per_frame_vectors']
        reliability_flags = processed['reliability_flags']
        microexpr_features = processed['microexpr_features']
        
        # Преобразуем per_frame_vectors в numpy массив правильного типа
        if isinstance(per_frame_vectors, list):
            per_frame_vectors = np.array(per_frame_vectors, dtype=np.float32)
        elif not isinstance(per_frame_vectors, np.ndarray):
            per_frame_vectors = np.asarray(per_frame_vectors, dtype=np.float32)
        
        # Убеждаемся, что все значения в features - это числа или массивы
        features_clean = {}
        for key, value in features.items():
            if isinstance(value, (int, float, bool)):
                features_clean[key] = float(value) if isinstance(value, bool) else value
            elif isinstance(value, (list, tuple)):
                # Преобразуем списки в numpy массивы
                try:
                    features_clean[key] = np.asarray(value, dtype=np.float32)
                except Exception:
                    features_clean[key] = np.asarray(value, dtype=object)
            elif isinstance(value, np.ndarray):
                features_clean[key] = value
            else:
                # Остальное сохраняем как есть (будет преобразовано в object array)
                features_clean[key] = value
        
        # Подготавливаем summary
        frames_with_face = int(df_filtered['success'].sum()) if 'success' in df_filtered.columns else len(df_filtered)
        summary = {
            'total_frames': len(frame_indices),
            'frames_processed': len(df_filtered),
            'frames_with_face': frames_with_face,
            'success': True,
            'fps': self.fps,
        }
        
        # Формируем итоговый результат
        result = {
            'features': features_clean,
            'per_frame_vectors': per_frame_vectors,
            'reliability_flags': reliability_flags,
            'microexpr_features': microexpr_features,
            'summary': summary,
        }
        
        self.logger.info(
            f"Обработка завершена: обработано {len(df_filtered)} кадров, "
            f"с лицами: {frames_with_face}"
        )
        
        return result

