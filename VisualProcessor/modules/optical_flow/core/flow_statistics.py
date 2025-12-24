"""
flow_statistics.py - Production модуль для статистического анализа оптического потока
Версия: 1.0.0
"""

import numpy as np
import torch
from scipy import stats
import pandas as pd
import json
import os
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
import warnings
from scipy.signal import savgol_filter, find_peaks, detrend
import logging
from glob import glob

from core.camera_motion import (
    aggregate_video_camera_features,
    compute_frame_motion_features,
    load_flow_tensor,
)
from core.advanced_features import (
    MotionEnergyImage,
    ForegroundBackgroundMotion,
    MotionClusters,
    SmoothnessJerkiness,
)

name = "FlowStatisticsAnalyzer"

from utils.logger import get_logger
logger = get_logger(name)

# Подавление предупреждений
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

class FlowFrameStatistics:
    """Класс для расчета статистик одного кадра оптического потока."""
    
    @staticmethod
    def calculate(flow_tensor: Union[torch.Tensor, str], 
                  config = None,
                  fps: float = 25.0,
                  frame_step: int = 1,
                  quality_features: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Вычисляет статистики для одного кадра потока.
        
        Args:
            flow_tensor: Тензор [2, H, W] или путь к .pt файлу
            config: Конфигурация анализа
            fps: кадров в секунду исходного видео
            frame_step: шаг между кадрами, использованный при расчёте потока
            quality_features: дополнительные предрасчитанные фичи качества (fb_error, confidence и т.п.)
            
        Returns:
            Словарь со статистиками
        """
        
        # Загрузка тензора
        if isinstance(flow_tensor, str):
            try:
                flow_tensor = torch.load(flow_tensor, map_location='cpu')
            except Exception as e:
                logger.error(f"Ошибка загрузки файла {flow_tensor}: {e}")
                raise
        
        # Валидация входных данных
        if not isinstance(flow_tensor, torch.Tensor):
            raise TypeError(f"Ожидается torch.Tensor, получен {type(flow_tensor)}")
        
        if flow_tensor.dim() != 3 or flow_tensor.shape[0] != 2:
            raise ValueError(f"Неверная форма тензора: {flow_tensor.shape}. Ожидается [2, H, W]")
        
        # Извлечение компонентов
        dx = flow_tensor[0].numpy().astype(np.float32)
        dy = flow_tensor[1].numpy().astype(np.float32)
        
        # Основные вычисления
        magnitude = np.sqrt(dx**2 + dy**2)
        direction = np.arctan2(dy, dx)
        frame_dt = max(frame_step, 1) / max(fps, 1e-6)
        px_per_sec_scale = 1.0 / max(frame_dt, 1e-6)
        magnitude_px_sec = magnitude * px_per_sec_scale
        
        # Расчет всех статистик
        stats_dict = {
            **FlowFrameStatistics._calculate_magnitude_stats(magnitude, magnitude_px_sec, config),
            **FlowFrameStatistics._calculate_direction_stats(direction, config),
            **FlowFrameStatistics._calculate_component_stats(dx, dy),
            **FlowFrameStatistics._calculate_motion_stats(magnitude_px_sec, config),
            **FlowFrameStatistics._calculate_histogram_stats(magnitude),
            **FlowFrameStatistics._calculate_spatial_stats(dx, dy, magnitude)
        }

        if quality_features:
            stats_dict.update(quality_features)
        
        # Добавляем метаинформацию
        stats_dict.update({
            'frame_shape': f"{dx.shape}",
            'pixel_count': int(magnitude.size),
            'frame_dt_seconds': float(frame_dt)
        })
        
        return stats_dict
    
    @staticmethod
    def _calculate_magnitude_stats(magnitude: np.ndarray, magnitude_px_sec: np.ndarray, config) -> Dict[str, float]:
        """Статистики величины движения."""
        flat_mag = magnitude.flatten()
        percentiles = np.percentile(flat_mag, [25, 50, 75, 95])
        noise_floor = getattr(config, 'noise_floor', 0.05)
        
        return {
            'magnitude_mean': float(np.mean(flat_mag)),
            'magnitude_std': float(np.std(flat_mag)),
            'magnitude_max': float(np.max(flat_mag)),
            'magnitude_median': float(percentiles[1]),
            'magnitude_iqr': float(percentiles[2] - percentiles[0]),
            'magnitude_p95': float(percentiles[3]),
            'fraction_zero_motion': float(np.mean(flat_mag <= 1e-6)),
            'fraction_below_noise_floor': float(np.mean(flat_mag <= noise_floor)),
            'magnitude_mean_px_sec': float(np.mean(magnitude_px_sec)),
            'magnitude_std_px_sec': float(np.std(magnitude_px_sec)),
            'magnitude_p95_px_sec': float(np.percentile(magnitude_px_sec.flatten(), 95))
        }
    
    @staticmethod
    def _calculate_direction_stats(direction: np.ndarray, config) -> Dict[str, float]:
        """Статистики направления движения."""
        flat_dir = direction.flatten()
        sin_mean = float(np.mean(np.sin(flat_dir)))
        cos_mean = float(np.mean(np.cos(flat_dir)))
        R = np.sqrt(sin_mean**2 + cos_mean**2)
        
        return {
            'dir_sin_mean': sin_mean,
            'dir_cos_mean': cos_mean,
            'dir_resultant_length': float(R),
            'dir_dispersion': float(1.0 - R),
            'direction_std_circular': float(FlowFrameStatistics._circular_std(flat_dir)),
            'direction_entropy': float(FlowFrameStatistics._directional_entropy(flat_dir, config.direction_bins))
        }
    
    @staticmethod
    def _circular_mean(angles: np.ndarray) -> float:
        """Среднее значение для циклических данных."""
        sin_sum = np.mean(np.sin(angles))
        cos_sum = np.mean(np.cos(angles))
        return np.arctan2(sin_sum, cos_sum)
    
    @staticmethod
    def _circular_std(angles: np.ndarray) -> float:
        """Стандартное отклонение для циклических данных."""
        R = np.sqrt(np.mean(np.sin(angles))**2 + np.mean(np.cos(angles))**2)
        return np.sqrt(-2 * np.log(R + 1e-10))
    
    @staticmethod
    def _directional_entropy(angles: np.ndarray, bins: int) -> float:
        """Энтропия распределения направлений."""
        try:
            hist, _ = np.histogram(angles, bins=bins, range=(-np.pi, np.pi))
            hist = hist + 1.0  # Laplace сглаживание
            hist = hist / (hist.sum() + 1e-10)
            return float(-np.sum(hist * np.log(hist + 1e-10)))  # натуральный лог
        except:
            return 0.0
    
    @staticmethod
    def _calculate_component_stats(dx: np.ndarray, dy: np.ndarray) -> Dict[str, float]:
        """Статистики компонент X и Y."""
        return {
            'dx_mean': float(np.mean(dx)),
            'dy_mean': float(np.mean(dy)),
            'dx_std': float(np.std(dx)),
            'dy_std': float(np.std(dy)),
            'dx_abs_mean': float(np.mean(np.abs(dx))),
            'dy_abs_mean': float(np.mean(np.abs(dy)))
        }
    
    @staticmethod
    def _calculate_motion_stats(magnitude_px_sec: np.ndarray, 
                                config) -> Dict[str, float]:
        """Статистики движущихся пикселей."""
        stats = {}
        thresholds = getattr(config, 'motion_thresholds', [1.0])
        for threshold in thresholds:
            moving_pixels = np.sum(magnitude_px_sec > threshold) / magnitude_px_sec.size
            stats[f'moving_pixels_{threshold}'] = float(moving_pixels)

        # Адаптивный порог: median + k * MAD
        flat = magnitude_px_sec.flatten()
        median = np.median(flat)
        mad = np.median(np.abs(flat - median)) + 1e-6
        k = getattr(config, 'moving_threshold_k', 3.0)
        adaptive_thresh = median + k * mad
        stats['moving_pixels_rel'] = float(np.mean(flat > adaptive_thresh))

        return stats
    
    @staticmethod
    def _calculate_histogram_stats(magnitude: np.ndarray) -> Dict[str, float]:
        """Гистограммные статистики."""
        flat_mag = magnitude.flatten()
        if len(flat_mag) < 4:
            return {'magnitude_skew': 0.0, 'magnitude_kurtosis': 0.0}
        
        try:
            return {
                'magnitude_skew': float(stats.skew(flat_mag)),
                'magnitude_kurtosis': float(stats.kurtosis(flat_mag))
            }
        except:
            return {'magnitude_skew': 0.0, 'magnitude_kurtosis': 0.0}
    
    @staticmethod
    def _calculate_spatial_stats(dx: np.ndarray, dy: np.ndarray, 
                                 magnitude: np.ndarray) -> Dict[str, float]:
        """Пространственные статистики."""
        try:
            # Градиент величины
            grad_y, grad_x = np.gradient(magnitude)
            spatial_gradient = np.mean(np.sqrt(grad_x**2 + grad_y**2))
            
            # Консистентность потока
            div = np.gradient(dx, axis=1) + np.gradient(dy, axis=0)
            flow_consistency = 1.0 / (1.0 + np.mean(np.abs(div)))
            
            return {
                'spatial_gradient': float(spatial_gradient),
                'flow_consistency': float(flow_consistency),
                'flow_divergence_mean': float(np.mean(div))
            }
        except:
            return {
                'spatial_gradient': 0.0,
                'flow_consistency': 0.0,
                'flow_divergence_mean': 0.0
            }
    
class SpatialAnalyzer:
    """Анализатор пространственных агрегатов."""
    
    @staticmethod
    def analyze(flow_tensor: torch.Tensor, config = None) -> pd.DataFrame:
        """
        Разбивает кадр на регионы и вычисляет статистики для каждого.
        
        Args:
            flow_tensor: Тензор оптического потока [2, H, W]
            config: Конфигурация анализа
            
        Returns:
            DataFrame с региональными статистиками
        """    
        # Извлечение компонентов
        dx = flow_tensor[0].numpy().astype(np.float32)
        dy = flow_tensor[1].numpy().astype(np.float32)
        magnitude = np.sqrt(dx**2 + dy**2)
        
        H, W = magnitude.shape
        grid_sizes = getattr(config, 'grid_sizes', None) or [getattr(config, 'grid_size', (4, 4))]
        regional_stats = []

        for grid_idx, grid in enumerate(grid_sizes):
            rows, cols = grid
            region_h, region_w = H // max(rows, 1), W // max(cols, 1)
            
            for i in range(rows):
                for j in range(cols):
                    # Определяем границы региона
                    y_start = i * region_h
                    y_end = (i + 1) * region_h if i < rows - 1 else H
                    x_start = j * region_w
                    x_end = (j + 1) * region_w if j < cols - 1 else W
                    
                    # Извлекаем регион
                    region_mag = magnitude[y_start:y_end, x_start:x_end]
                    region_dx = dx[y_start:y_end, x_start:x_end]
                    region_dy = dy[y_start:y_end, x_start:x_end]
                    
                    # Вычисляем статистики региона
                    region_stats = SpatialAnalyzer._calculate_region_stats(
                        region_mag, region_dx, region_dy, i, j,
                        (y_start, y_end, x_start, x_end), magnitude
                    )
                    region_stats['grid_id'] = f"{rows}x{cols}"
                    region_stats['grid_level'] = grid_idx
                    regional_stats.append(region_stats)
        
        return pd.DataFrame(regional_stats)
    
    @staticmethod
    def _calculate_region_stats(region_mag: np.ndarray, region_dx: np.ndarray,
                               region_dy: np.ndarray, i: int, j: int,
                               coords: Tuple[int, int, int, int], 
                               global_magnitude: np.ndarray) -> Dict[str, Any]:
        """Вычисляет статистики для одного региона."""
        if region_mag.size == 0:
            return {}
        
        region_mean = np.mean(region_mag)
        global_mean = np.mean(global_magnitude)
        
        return {
            'region_id': f"R{i}_{j}",
            'grid_position': f"{i},{j}",
            'pixel_coords': f"{coords[0]}:{coords[1]},{coords[2]}:{coords[3]}",
            'region_size': int(region_mag.size),
            
            # Базовые статистики
            'region_magnitude_mean': float(region_mean),
            'region_magnitude_std': float(np.std(region_mag)),
            'region_dx_mean': float(np.mean(region_dx)),
            'region_dy_mean': float(np.mean(region_dy)),
            
            # Относительные метрики
            'relative_activity': float(region_mean / (global_mean + 1e-10)),
            'motion_dominance': float(np.sum(region_mag > global_mean) / region_mag.size),
            
            # Пространственные паттерны
            'region_gradient': float(SpatialAnalyzer._calculate_region_gradient(region_mag)),
            'flow_divergence': float(SpatialAnalyzer._calculate_region_divergence(region_dx, region_dy)),
            'direction_histogram': SpatialAnalyzer._direction_histogram(region_dx, region_dy)
        }
    
    @staticmethod
    def _calculate_region_gradient(region_mag: np.ndarray) -> float:
        """Вычисляет градиент внутри региона."""
        if region_mag.size < 4:
            return 0.0
        try:
            grad_y, grad_x = np.gradient(region_mag)
            return np.mean(np.sqrt(grad_x**2 + grad_y**2))
        except:
            return 0.0
    
    @staticmethod
    def _calculate_region_divergence(dx: np.ndarray, dy: np.ndarray) -> float:
        """Вычисляет дивергенцию потока в регионе."""
        if dx.size < 4:
            return 0.0
        try:
            div = np.gradient(dx, axis=1) + np.gradient(dy, axis=0)
            return np.mean(div)
        except:
            return 0.0
    
    @staticmethod
    def _direction_histogram(dx: np.ndarray, dy: np.ndarray, bins: int = 8) -> list:
        """Гистограмма направлений региона в числовом виде."""
        if dx.size == 0:
            return []
        direction = np.arctan2(dy, dx).flatten()
        hist, _ = np.histogram(direction, bins=bins, range=(-np.pi, np.pi))
        return hist.astype(int).tolist()

class TemporalAnalyzer:
    """Анализатор временных трендов."""
    
    @staticmethod
    def analyze(frame_stats_list: List[Dict[str, Any]], 
                fps: float, skip: int,
                config = None) -> Dict[str, Any]:
        """
        Анализирует временные тренды по последовательности статистик кадров.
        
        Args:
            frame_stats_list: Список статистик кадров
            fps: Кадров в секунду исходного видео
            skip: Шаг пропуска кадров
            config: Конфигурация анализа
            
        Returns:
            Словарь с результатами временного анализа
        """   
        if len(frame_stats_list) < config.min_frames_for_temporal:
            logger.warning(f"Недостаточно кадров для временного анализа: {len(frame_stats_list)}")
            return {'error': 'insufficient_data'}
        
        try:
            df = pd.DataFrame(frame_stats_list)
            moving_keys = [c for c in df.columns if c.startswith('moving_pixels_')]
            moving_key = moving_keys[0] if moving_keys else None
            
            # Ключевые временные ряды
            time_series = {
                'magnitude': df['magnitude_mean'].values,
                'moving_pixels': df[moving_key].values if moving_key else df['magnitude_mean'].values * 0,
                'direction_std': df['direction_std_circular'].values if 'direction_std_circular' in df else df['dir_dispersion'].values,
                'spatial_gradient': df['spatial_gradient'].values
            }
            
            # Временные метки в секундах
            time_seconds = np.arange(len(df)) * skip / fps
            
            return {
                'trends': TemporalAnalyzer._calculate_trends(time_series, time_seconds, config),
                'periodicity': TemporalAnalyzer._detect_periodicity(time_series, fps, skip),
                'transitions': TemporalAnalyzer._detect_transitions(time_series),
                'segments': TemporalAnalyzer._temporal_segmentation(time_series['magnitude']),
                'summary': TemporalAnalyzer._calculate_summary(time_series, time_seconds)
            }
        except Exception as e:
            logger.error(f"Ошибка временного анализа: {e}")
            return {'error': str(e)}
    
    @staticmethod
    def _calculate_trends(time_series: Dict[str, np.ndarray], 
                         time_seconds: np.ndarray,
                         config) -> Dict[str, Any]:
        """Вычисляет тренды для каждой метрики."""
        trends = {}
        
        for metric_name, values in time_series.items():
            # Линейный тренд
            try:
                coeffs = np.polyfit(time_seconds, values, 1)
                slope = coeffs[0]
            except:
                slope = 0.0
            
            # Сглаживание
            if len(values) > config.savgol_window:
                window = min(config.savgol_window, len(values))
                if window % 2 == 0:
                    window -= 1  # Окно должно быть нечетным
                try:
                    smoothed = savgol_filter(values, window, 2)
                except:
                    smoothed = values
            else:
                smoothed = values
            
            # Классификация тренда
            if abs(slope) < 0.001:
                trend_type = 'stable'
            elif slope > 0:
                trend_type = 'increasing'
            else:
                trend_type = 'decreasing'
            
            trends[metric_name] = {
                'slope': float(slope),
                'trend_type': trend_type,
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'range': float(np.max(values) - np.min(values)),
                'has_trend': abs(slope) > 0.001
            }
        
        return trends
    
    @staticmethod
    def _detect_periodicity(time_series: Dict[str, np.ndarray], 
                           fps: float, skip: int) -> Dict[str, Any]:
        """Обнаруживает периодические паттерны."""
        results = {}
        
        for metric_name, values in time_series.items():
            if len(values) < 20:
                results[metric_name] = {'has_periodicity': False, 'reason': 'insufficient_data'}
                continue
            
            try:
                # FFT анализ с детрендингом и окном Хэнна
                n = len(values)
                values_detrended = detrend(values)
                window = np.hanning(n)
                windowed = values_detrended * window
                yf = np.fft.fft(windowed)
                xf = np.fft.fftfreq(n, d=(skip/fps))
                
                power = np.abs(yf[:n//2])**2
                freqs = xf[:n//2]
                
                # Ищем значимые частоты
                mask = (freqs > 0.1) & (freqs < 5.0)
                if np.any(mask) and len(power[mask]) > 0:
                    dominant_idx = np.argmax(power[mask])
                    dominant_freq = freqs[mask][dominant_idx]
                    
                    if dominant_freq > 0:
                        dominant_period = 1.0 / dominant_freq
                        background = np.median(power[mask]) + 1e-10
                        significance = power[mask][dominant_idx] / background
                        
                        if significance > 2.0:
                            results[metric_name] = {
                                'has_periodicity': True,
                                'dominant_frequency_hz': float(dominant_freq),
                                'dominant_period_seconds': float(dominant_period),
                                'significance': float(significance)
                            }
                            continue
            
                results[metric_name] = {'has_periodicity': False}
            except:
                results[metric_name] = {'has_periodicity': False, 'reason': 'analysis_error'}
        
        return results
    
    @staticmethod
    def _detect_transitions(time_series: Dict[str, np.ndarray], 
                           window_size: int = 5) -> Dict[str, Any]:
        """Обнаруживает резкие изменения."""
        transitions = {}
        
        for metric_name, values in time_series.items():
            if len(values) < window_size * 2:
                transitions[metric_name] = {'transition_count': 0, 'reason': 'insufficient_data'}
                continue
            
            try:
                # Скользящие статистики
                series = pd.Series(values)
                rolling_mean = series.rolling(window=window_size, center=True, min_periods=1).mean()
                rolling_std = series.rolling(window=window_size, center=True, min_periods=1).std()
                
                # Z-score аномалии
                z_scores = np.abs((values - rolling_mean) / (rolling_std.replace(0, 1e-10)))
                anomaly_mask = z_scores > 2.0
                
                if np.any(anomaly_mask):
                    anomaly_indices = np.where(anomaly_mask)[0]
                    
                    # Группировка близких аномалий
                    transition_points = []
                    current_group = []
                    
                    for idx in anomaly_indices:
                        if not current_group or idx - current_group[-1] <= window_size:
                            current_group.append(idx)
                        else:
                            if current_group:
                                center = int(np.mean(current_group))
                                transition_points.append({
                                    'frame_index': int(center),
                                    'z_score': float(z_scores[center]),
                                    'value_change': float(values[center] - np.mean(values))
                                })
                            current_group = [idx]
                    
                    if current_group:
                        center = int(np.mean(current_group))
                        transition_points.append({
                            'frame_index': int(center),
                            'z_score': float(z_scores[center]),
                            'value_change': float(values[center] - np.mean(values))
                        })
                    
                    transitions[metric_name] = {
                        'transition_count': len(transition_points),
                        'transition_points': transition_points,
                        'max_z_score': float(np.max(z_scores))
                    }
                else:
                    transitions[metric_name] = {'transition_count': 0}
            except:
                transitions[metric_name] = {'transition_count': 0, 'reason': 'analysis_error'}
        
        return transitions
    
    @staticmethod
    def _temporal_segmentation(magnitude_series: np.ndarray, 
                              n_segments: int = 5) -> Dict[str, Any]:
        """Сегментирует на однородные участки."""
        if len(magnitude_series) < n_segments * 2:
            return {'segments': [], 'error': 'insufficient_data'}
        
        try:
            # Используем K-means для сегментации
            from sklearn.cluster import KMeans
            
            # Подготовка данных для кластеризации
            X = magnitude_series.reshape(-1, 1)
            kmeans = KMeans(n_clusters=n_segments, random_state=42, n_init=10)
            labels = kmeans.fit_predict(X)
            
            # Находим границы сегментов
            boundaries = []
            for i in range(1, len(labels)):
                if labels[i] != labels[i-1]:
                    boundaries.append(i)
            
            # Формируем сегменты
            segments = []
            start_idx = 0
            
            for boundary in boundaries:
                segment_data = magnitude_series[start_idx:boundary]
                segments.append({
                    'start_frame': int(start_idx),
                    'end_frame': int(boundary),
                    'length_frames': len(segment_data),
                    'mean_magnitude': float(np.mean(segment_data)),
                    'std_magnitude': float(np.std(segment_data)),
                    'cluster_label': int(labels[start_idx])
                })
                start_idx = boundary
            
            # Последний сегмент
            if start_idx < len(magnitude_series):
                segment_data = magnitude_series[start_idx:]
                segments.append({
                    'start_frame': int(start_idx),
                    'end_frame': len(magnitude_series),
                    'length_frames': len(segment_data),
                    'mean_magnitude': float(np.mean(segment_data)),
                    'std_magnitude': float(np.std(segment_data)),
                    'cluster_label': int(labels[start_idx])
                })
            
            return {
                'segments': segments,
                'boundary_frames': boundaries,
                'method': 'kmeans'
            }
        except Exception as e:
            logger.warning(f"Ошибка сегментации: {e}, используем равномерное разбиение")
            return TemporalAnalyzer._uniform_segmentation(magnitude_series, n_segments)
    
    @staticmethod
    def _uniform_segmentation(magnitude_series: np.ndarray, 
                             n_segments: int = 5) -> Dict[str, Any]:
        """Равномерная сегментация (запасной вариант)."""
        segment_length = len(magnitude_series) // n_segments
        segments = []
        
        for i in range(n_segments):
            start_idx = i * segment_length
            end_idx = (i + 1) * segment_length if i < n_segments - 1 else len(magnitude_series)
            
            segment_data = magnitude_series[start_idx:end_idx]
            if len(segment_data) > 0:
                segments.append({
                    'start_frame': int(start_idx),
                    'end_frame': int(end_idx),
                    'length_frames': len(segment_data),
                    'mean_magnitude': float(np.mean(segment_data)),
                    'std_magnitude': float(np.std(segment_data))
                })
        
        return {'segments': segments, 'method': 'uniform'}
    
    @staticmethod
    def _calculate_summary(time_series: Dict[str, np.ndarray], 
                          time_seconds: np.ndarray) -> Dict[str, Any]:
        """Вычисляет сводные метрики."""
        magnitude = time_series['magnitude']
        
        # Обнаружение пиков
        try:
            peaks, properties = find_peaks(magnitude, 
                                         height=np.mean(magnitude) * 1.5,
                                         distance=5)
            peak_count = len(peaks)
        except:
            peak_count = 0
        
        # Стабильность
        stability_metrics = []
        for values in time_series.values():
            cv = np.std(values) / (np.mean(values) + 1e-10)
            stability_metrics.append(1.0 / (1.0 + cv))
        
        return {
            'total_duration_seconds': float(time_seconds[-1]) if len(time_seconds) > 0 else 0.0,
            'avg_magnitude': float(np.mean(magnitude)),
            'magnitude_variability': float(np.std(magnitude) / (np.mean(magnitude) + 1e-10)),
            'activity_peaks_count': peak_count,
            'stability_score': float(np.mean(stability_metrics)) if stability_metrics else 0.0
        }

class FlowStatisticsAnalyzer:
    """Основной класс для статистического анализа оптического потока."""
    
    def __init__(self, config = None):
        self.config = config
        self.frame_analyzer = FlowFrameStatistics()
        self.spatial_analyzer = SpatialAnalyzer()
        self.temporal_analyzer = TemporalAnalyzer()
    
    def analyze_video(self, flow_dir: Union[str, Path], video_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Полный статистический анализ для обработанного видео.
        
        Args:
            flow_dir: Директория с flow файлами (.pt)
            video_metadata: Метаданные видео
            
        Returns:
            Словарь с результатами анализа
        """
        logger.info(f"Начинаем анализ для {flow_dir}")
        
        # Валидация входных данных
        flow_dir = Path(flow_dir)
        if not flow_dir.exists():
            raise FileNotFoundError(f"Директория не найдена: {flow_dir}")
        
        # 1. Сбор flow файлов
        flow_files = sorted([f for f in flow_dir.glob("*.pt")])
        if not flow_files:
            logger.warning(f"Нет flow файлов в {flow_dir}")
            return {'error': 'no_flow_files'}
        
        logger.info(f"Найдено {len(flow_files)} flow файлов")
        fps = video_metadata.get('video_properties', {}).get('fps', 25.0)
        frame_step = video_metadata.get('processing_parameters', {}).get('frame_skip', 1)
        
        # 2. Базовые статистики по кадрам
        frame_stats_list = self._analyze_frames(flow_files, fps=fps, frame_step=frame_step)
        
        # 2.1 Анализ движения камеры (опционально)
        camera_motion_results = None
        if getattr(self.config, 'enable_camera_motion', False):
            camera_motion_results = self._analyze_camera_motion(flow_files)
            # Добавляем покадровые фичи камеры в CSV
            if camera_motion_results and camera_motion_results.get('per_frame'):
                per_frame = camera_motion_results['per_frame']
                limit = min(len(frame_stats_list), len(per_frame))
                for idx in range(limit):
                    cam_prefixed = {f"camera_{k}": v for k, v in per_frame[idx].items()}
                    frame_stats_list[idx].update(cam_prefixed)
        
        # 3. Пространственный анализ (выборочно)
        spatial_results = self._analyze_spatial(flow_files)
        
        # 4. Временной анализ
        temporal_results = self._analyze_temporal(frame_stats_list, video_metadata)
        
        # 5. Продвинутые фичи (опционально)
        advanced_results = {}
        if getattr(self.config, 'enable_advanced_features', False):
            advanced_results = self._analyze_advanced_features(flow_files, video_metadata)
        
        # 6. Формирование результатов
        results = self._compile_results(
            frame_stats_list, spatial_results, temporal_results, 
            video_metadata, flow_dir, camera_motion_results, advanced_results
        )
        
        # 7. Сохранение
        self._save_results(results, flow_dir.parent)
        
        logger.info(f"Анализ завершен для {flow_dir}")
        return results
    
    def _analyze_frames(self, flow_files: List[Path], fps: float = 25.0, frame_step: int = 1) -> List[Dict[str, Any]]:
        """Анализ статистик каждого кадра."""
        frame_stats_list = []
        
        logger.info("Анализ статистик по кадрам...")
        for flow_file in flow_files:
            try:
                frame_idx_val = self._extract_frame_index(flow_file.name)
                quality_path = flow_file.parent / "quality" / f"quality_{frame_idx_val:06d}.json"
                quality_features = None
                if quality_path.exists():
                    try:
                        with open(quality_path, 'r', encoding='utf-8') as f:
                            quality_features = json.load(f)
                    except Exception as e:
                        logger.warning(f"Ошибка чтения quality {quality_path}: {e}")

                stats = self.frame_analyzer.calculate(
                    str(flow_file),
                    self.config,
                    fps=fps,
                    frame_step=frame_step,
                    quality_features=quality_features
                )
                stats['flow_filename'] = flow_file.name
                stats['frame_index'] = frame_idx_val
                frame_stats_list.append(stats)
            except Exception as e:
                logger.warning(f"Ошибка анализа {flow_file}: {e}")
                continue
        
        logger.info(f"Успешно проанализировано {len(frame_stats_list)} кадров")

        # Нормализация по видео (z-score) ключевых покадровых фич
        if frame_stats_list:
            df = pd.DataFrame(frame_stats_list)
            norm_keys = [
                'magnitude_mean_px_sec', 'magnitude_std_px_sec', 'magnitude_p95_px_sec',
                'dx_mean', 'dy_mean', 'flow_confidence_mean', 'fb_error_mean'
            ]
            for key in norm_keys:
                if key in df.columns:
                    mean_v = df[key].mean()
                    std_v = df[key].std() + 1e-6
                    for stats in frame_stats_list:
                        stats[f"{key}_norm"] = float((stats.get(key, 0.0) - mean_v) / std_v)

            motion_thresholds = getattr(self.config, 'motion_thresholds', [1.0])
            motion_key = f"moving_pixels_{motion_thresholds[0]}"

            sequence_keys = [
                'magnitude_mean_px_sec_norm',
                'magnitude_std_px_sec_norm',
                'magnitude_p95_px_sec_norm',
                'dir_sin_mean',
                'dir_cos_mean',
                'dir_dispersion',
                'dx_mean_norm',
                'dy_mean_norm',
                motion_key,
                'moving_pixels_rel',
                'flow_confidence_mean_norm',
                'occlusion_fraction',
                'fb_error_mean_norm',
                'flow_consistency'
            ]

            for stats in frame_stats_list:
                stats['sequence_features'] = {
                    k: stats[k] for k in sequence_keys if k in stats
                }

        return frame_stats_list
    
    def _analyze_spatial(self, flow_files: List[Path]) -> Dict[str, Any]:
        """Пространственный анализ (выборочный)."""
        spatial_results = {}
        sample_rate = self.config.spatial_sample_rate
        
        logger.info("Пространственный анализ...")
        for i in range(0, len(flow_files), sample_rate):
            if i >= len(flow_files):
                break
                
            flow_file = flow_files[i]
            try:
                flow_tensor = torch.load(flow_file, map_location='cpu')
                spatial_df = self.spatial_analyzer.analyze(flow_tensor, self.config)
                
                # Анализ регионов интереса
                roi_analysis = self._analyze_regions_of_interest(spatial_df)
                
                spatial_results[f"frame_{i:06d}"] = {
                    'regional_stats': spatial_df.to_dict('records'),
                    'roi_analysis': roi_analysis,
                    'flow_filename': flow_file.name
                }
            except Exception as e:
                logger.warning(f"Ошибка пространственного анализа {flow_file}: {e}")
                continue
        
        return spatial_results
    
    def _analyze_regions_of_interest(self, spatial_df: pd.DataFrame) -> Dict[str, Any]:
        """Анализ регионов интереса."""
        if len(spatial_df) == 0:
            return {}
        
        # Выбираем сетку среднего разрешения (предпочтительно 4x4), иначе первую
        candidate = spatial_df[spatial_df['grid_id'] == '4x4']
        if candidate.empty:
            candidate = spatial_df

        # Сортируем по активности
        top_regions = candidate.nlargest(self.config.top_regions_count, 'region_magnitude_mean')
        
        return {
            'top_regions': top_regions['region_id'].tolist(),
            'activity_concentration': float(
                top_regions['region_magnitude_mean'].sum() / 
                spatial_df['region_magnitude_mean'].sum()
            ),
            'spatial_distribution': self._analyze_spatial_distribution(top_regions),
            'direction_hist_summed': SpatialAnalyzer._sum_direction_hists(top_regions)
        }
    
    def _analyze_spatial_distribution(self, top_regions: pd.DataFrame) -> str:
        """Анализ пространственного распределения."""
        if len(top_regions) < 2:
            return 'single_region'
        
        # Извлекаем координаты сетки
        positions = []
        for pos in top_regions['grid_position']:
            if isinstance(pos, str) and ',' in pos:
                try:
                    row, col = map(int, pos.split(','))
                    positions.append([row, col])
                except:
                    continue
        
        if len(positions) < 2:
            return 'single_region'
        
        # Вычисляем дисперсию
        positions_array = np.array(positions)
        variance = np.var(positions_array, axis=0).sum()
        
        if variance < 1.0:
            return 'concentrated'
        elif variance < 4.0:
            return 'scattered'
        else:
            return 'distributed'

    @staticmethod
    def _sum_direction_hists(top_regions: pd.DataFrame) -> list:
        """Суммирует числовые гистограммы направлений."""
        hist_list = []
        for hist in top_regions.get('direction_histogram', []):
            if isinstance(hist, (list, np.ndarray)) and len(hist) > 0:
                hist_list.append(np.array(hist, dtype=np.int64))
        if not hist_list:
            return []
        return np.sum(hist_list, axis=0).astype(int).tolist()
    
    def _analyze_temporal(self, frame_stats_list: List[Dict[str, Any]], 
                         video_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Временной анализ."""
        fps = video_metadata.get('video_properties', {}).get('fps', 25.0)
        skip = video_metadata.get('processing_parameters', {}).get('frame_skip', 5)
        
        logger.info("Временной анализ...")
        return self.temporal_analyzer.analyze(frame_stats_list, fps, skip, self.config)
    
    def _analyze_camera_motion(self, flow_files: List[Path]) -> Dict[str, Any]:
        """Анализ движения камеры и агрегирование."""
        if not flow_files:
            return {'error': 'no_flow_files'}
        
        try:
            mag_bg_thresh = getattr(self.config, 'camera_motion_config', {}).get('mag_bg_thresh', 0.5)
            flows = []
            for flow_file in flow_files:
                try:
                    flows.append(load_flow_tensor(str(flow_file)))
                except Exception as e:
                    logger.warning(f"Ошибка загрузки для camera_motion {flow_file}: {e}")
                    continue
            
            per_frame = []
            
            prev = None
            for flow in flows:
                feats = compute_frame_motion_features(flow, flow_prev=prev, mag_bg_thresh=mag_bg_thresh)
                per_frame.append(feats)
                prev = flow
            
            summary = aggregate_video_camera_features(
                [str(f) for f in flow_files],
                config=getattr(self.config, 'camera_motion_config', {})
            )
            
            return {
                'summary': summary,
                'per_frame': per_frame
            }
        except Exception as e:
            logger.error(f"Ошибка анализа движения камеры: {e}")
            return {'error': str(e)}
    
    def _analyze_advanced_features(self, flow_files: List[Path],
                                   video_metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Анализ продвинутых фичей: MEI, FG/BG, Clusters, Smoothness."""
        logger.info("Анализ продвинутых фичей...")
        results = {}
        
        try:
            # Загружаем все потоки
            flows = []
            magnitudes = []
            for flow_file in flow_files:
                try:
                    flow = torch.load(flow_file, map_location='cpu')
                    if isinstance(flow, torch.Tensor):
                        flow = flow.numpy()
                    if flow.shape[0] == 2:
                        flow = np.transpose(flow, (1, 2, 0))
                    flows.append(flow)
                    mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
                    magnitudes.append(mag)
                except Exception as e:
                    logger.warning(f"Ошибка загрузки {flow_file}: {e}")
                    continue
            
            if not flows:
                return {'error': 'no_flows_loaded'}
            
            # 1. Motion Energy Image
            if getattr(self.config, 'enable_mei', True):
                try:
                    fps = video_metadata.get('video_properties', {}).get('fps', 25.0)
                    skip = video_metadata.get('processing_parameters', {}).get('frame_skip', 1)
                    mei, mei_features = MotionEnergyImage.compute_mei(
                        magnitudes,
                        fps=fps,
                        frame_skip=skip
                    )
                    results['motion_energy_image'] = {
                        'features': mei_features,
                        'mei_shape': list(mei.shape)
                    }
                except Exception as e:
                    logger.warning(f"Ошибка MEI: {e}")
                    results['motion_energy_image'] = {'error': str(e)}
            
            # 2. Foreground vs Background Motion
            if getattr(self.config, 'enable_fg_bg', True):
                try:
                    fg_bg_results = []
                    for flow in flows[:min(50, len(flows))]:  # Ограничиваем для скорости
                        fg_bg = ForegroundBackgroundMotion.separate_motion(
                            flow,
                            method=getattr(self.config, 'fg_bg_method', 'magnitude_threshold'),
                            threshold=getattr(self.config, 'fg_bg_threshold', 0.5)
                        )
                        fg_bg_results.append(fg_bg['features'])
                    
                    # Агрегируем статистики
                    if fg_bg_results:
                        avg_fg_bg = {
                            'foreground_motion_energy_mean': float(np.mean([r['foreground_motion_energy'] for r in fg_bg_results])),
                            'background_motion_energy_mean': float(np.mean([r['background_motion_energy'] for r in fg_bg_results])),
                            'ratio_foreground_background_mean': float(np.mean([r['ratio_foreground_background_flow'] for r in fg_bg_results])),
                            'foreground_coverage_mean': float(np.mean([r['foreground_coverage_ratio'] for r in fg_bg_results]))
                        }
                        results['foreground_background_motion'] = {
                            'summary': avg_fg_bg,
                            'per_frame_count': len(fg_bg_results)
                        }
                except Exception as e:
                    logger.warning(f"Ошибка FG/BG: {e}")
                    results['foreground_background_motion'] = {'error': str(e)}
            
            # 3. Motion Clusters
            if getattr(self.config, 'enable_clusters', True):
                try:
                    cluster_results = []
                    n_clusters = getattr(self.config, 'motion_clusters_n', 5)
                    for flow in flows[:min(20, len(flows))]:  # Ограничиваем для скорости
                        clusters = MotionClusters.cluster_motion(
                            flow,
                            n_clusters=n_clusters
                        )
                        cluster_results.append(clusters['features'])
                    
                    if cluster_results:
                        avg_clusters = {
                            'num_clusters_mean': float(np.mean([r.get('num_motion_clusters', 0) for r in cluster_results])),
                            'largest_cluster_coverage_mean': float(np.mean([r.get('largest_cluster_coverage', 0) for r in cluster_results])),
                            'cluster_diversity_mean': float(np.mean([r.get('cluster_diversity', 0) for r in cluster_results]))
                        }
                        results['motion_clusters'] = {
                            'summary': avg_clusters,
                            'per_frame_count': len(cluster_results)
                        }
                except Exception as e:
                    logger.warning(f"Ошибка кластеров: {e}")
                    results['motion_clusters'] = {'error': str(e)}
            
            # 4. Smoothness/Jerkiness
            if getattr(self.config, 'enable_smoothness', True):
                try:
                    fps = video_metadata.get('video_properties', {}).get('fps', 25.0)
                    skip = video_metadata.get('processing_parameters', {}).get('frame_skip', 5)
                    smoothness = SmoothnessJerkiness.compute_smoothness_metrics(
                        flows,
                        fps=fps,
                        frame_skip=skip
                    )
                    results['smoothness_jerkiness'] = smoothness
                except Exception as e:
                    logger.warning(f"Ошибка smoothness: {e}")
                    results['smoothness_jerkiness'] = {'error': str(e)}
        
        except Exception as e:
            logger.error(f"Ошибка анализа продвинутых фичей: {e}")
            results['error'] = str(e)
        
        return results
    
    def _compile_results(self, frame_stats_list: List[Dict[str, Any]],
                        spatial_results: Dict[str, Any],
                        temporal_results: Dict[str, Any],
                        video_metadata: Dict[str, Any],
                        flow_dir: Path,
                        camera_motion_results: Optional[Dict[str, Any]] = None,
                        advanced_results: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Компиляция всех результатов."""
        # Сводные метрики
        summary_metrics = self._extract_summary_metrics(frame_stats_list, temporal_results)
        
        # Извлекаем model_type из video_metadata
        model_type = video_metadata.get('processing_parameters', {}).get('model', 'unknown')
        model_name = f"RAFT_{model_type}" if model_type in ['small', 'large'] else f"RAFT_{model_type}"
        
        return {
            'analysis_info': {
                'version': '1.0.0',
                'created_at': datetime.now().isoformat(),
                'timestamp': datetime.now().isoformat(),  # Для обратной совместимости
                'model_type': model_type,
                'model_name': model_name,
            },
            'processing_info': {
                'total_frames_analyzed': len(frame_stats_list),
                'frames_with_spatial_analysis': len(spatial_results),
                'analysis_duration_seconds': None  # Заполняется при сохранении
            },
            'statistics': {
                'frame_statistics': frame_stats_list,
                'spatial_analysis': spatial_results,
                'temporal_analysis': temporal_results,
                'summary_metrics': summary_metrics,
                'camera_motion': camera_motion_results,
                'advanced_features': advanced_results
            }
        }
    
    def _extract_summary_metrics(self, frame_stats_list: List[Dict[str, Any]],
                                temporal_results: Dict[str, Any]) -> Dict[str, Any]:
        """Извлечение ключевых метрик."""
        if not frame_stats_list:
            return {}
        
        df = pd.DataFrame(frame_stats_list)
        moving_keys = [c for c in df.columns if c.startswith('moving_pixels_')]
        moving_key = moving_keys[0] if moving_keys else None
        
        metrics = {
            'overall_magnitude_mean': float(df['magnitude_mean'].mean()),
            'overall_magnitude_std': float(df['magnitude_mean'].std()),
            'activity_variability': float(df[moving_key].std()) if moving_key else 0.0
        }
        
        # Добавляем временные метрики если они есть
        if 'summary' in temporal_results and not isinstance(temporal_results.get('error'), str):
            temp_summary = temporal_results['summary']
            metrics.update({
                'temporal_stability': temp_summary.get('stability_score', 0.0),
                'peak_activity_frames': temp_summary.get('activity_peaks_count', 0),
                'dominant_trend': temporal_results.get('trends', {}).get('magnitude', {}).get('trend_type', 'unknown'),
                'has_periodicity': any(r.get('has_periodicity', False) for r in 
                                      temporal_results.get('periodicity', {}).values()),
                'transition_count': sum(t.get('transition_count', 0) for t in 
                                       temporal_results.get('transitions', {}).values())
            })
        
        return metrics
    
    def _save_results(self, results: Dict[str, Any], output_dir: Path) -> None:
        """Сохранение результатов анализа."""
        # Добавляем время анализа
        start_time = results['analysis_info'].get('timestamp')
        if start_time:
            duration = (datetime.now() - datetime.fromisoformat(start_time)).total_seconds()
            results['processing_info']['analysis_duration_seconds'] = duration
        
        # Сохраняем JSON
        json_path = output_dir / 'statistical_analysis.json'
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False, default=str)
        
        # Сохраняем сводную таблицу
        self._save_summary_table(results, output_dir)
        
        logger.info(f"Результаты сохранены в {output_dir}")
    
    def _save_summary_table(self, results: Dict[str, Any], output_dir: Path) -> None:
        """Сохранение сводной таблицы в CSV."""
        try:
            frame_stats = results['statistics']['frame_statistics']
            if frame_stats:
                df = pd.DataFrame(frame_stats)
                csv_path = output_dir / 'frame_statistics.csv'
                df.to_csv(csv_path, index=False, encoding='utf-8')
        except Exception as e:
            logger.warning(f"Ошибка сохранения CSV: {e}")
    
    @staticmethod
    def _extract_frame_index(filename: str) -> int:
        """Извлечение индекса кадра из имени файла."""
        try:
            # Ожидается формат: flow_000000.pt
            base = filename.split('.')[0]
            return int(base.split('_')[1])
        except:
            return -1
