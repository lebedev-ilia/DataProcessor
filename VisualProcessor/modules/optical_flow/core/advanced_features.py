"""
advanced_features.py - Продвинутые фичи оптического потока
Включает: MEI/MEP, Foreground/Background Motion, Motion Clusters, Smoothness/Jerkiness
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional, Union, Any
from pathlib import Path
import cv2
from scipy import stats
from sklearn.cluster import KMeans
import logging

logger = logging.getLogger(__name__)


class MotionEnergyImage:
    """Класс для вычисления Motion Energy Image (MEI) и Motion History Image (MHI)."""
    
    @staticmethod
    def compute_mei(flow_magnitudes: List[np.ndarray],
                   fps: float = 25.0,
                   frame_skip: int = 1,
                   tau_seconds: float = 1.0) -> Tuple[np.ndarray, Dict[str, float]]:
        """
        Вычисляет Motion Energy Image (MEI) - бинарное изображение движения.
        
        Args:
            flow_magnitudes: Список массивов величины движения [H, W] для каждого кадра
            fps: кадров в секунду
            frame_skip: шаг между кадрами
            tau_seconds: временная константа затухания MHI
            
        Returns:
            (mei, features_dict): MEI изображение и словарь фичей
        """
        if not flow_magnitudes:
            return np.zeros((1, 1)), {}
        
        frame_dt = frame_skip / max(fps, 1e-6)
        decay_factor = float(np.exp(-frame_dt / max(tau_seconds, 1e-6)))
        decay_factor = float(np.clip(decay_factor, 0.0, 1.0))
        
        # Нормализуем все к одному размеру
        h, w = flow_magnitudes[0].shape
        for mag in flow_magnitudes:
            if mag.shape != (h, w):
                mag = cv2.resize(mag, (w, h), interpolation=cv2.INTER_LINEAR)
        
        # Порог для определения движения
        threshold = np.percentile(np.concatenate([m.flatten() for m in flow_magnitudes]), 50)
        
        # MEI: бинарное изображение, где 1 = было движение
        mei = np.zeros((h, w), dtype=np.float32)
        mhi = np.zeros((h, w), dtype=np.float32)  # Motion History Image
        
        for i, mag in enumerate(flow_magnitudes):
            # Бинаризация движения
            motion_mask = mag > threshold
            mei = np.maximum(mei, motion_mask.astype(np.float32))
            
            # MHI: накопление с затуханием
            mhi = np.maximum(mhi * decay_factor, motion_mask.astype(np.float32))
        
        # Вычисляем фичи
        features = {
            'mei_total_energy': float(np.sum(mei)),
            'mei_coverage_ratio': float(np.mean(mei)),
            'mei_max_energy': float(np.max(mei)),
            'mei_std': float(np.std(mei)),
            'mhi_contrast': float((np.max(mhi) - np.min(mhi)) / (np.max(mhi) + 1e-10)),
            'mhi_entropy': float(MotionEnergyImage._compute_entropy(mhi)),
            'mhi_mean': float(np.mean(mhi)),
            'mhi_max': float(np.max(mhi)),
            'motion_persistence': float(np.mean(mhi > 0.5))  # Доля пикселей с устойчивым движением
        }
        
        return mei, features
    
    @staticmethod
    def _compute_entropy(image: np.ndarray, bins: int = 256) -> float:
        """Вычисляет энтропию изображения."""
        hist, _ = np.histogram(image.flatten(), bins=bins, range=(0, 1))
        hist = hist / (hist.sum() + 1e-10)
        hist = hist[hist > 0]
        return float(-np.sum(hist * np.log2(hist + 1e-10)))


class ForegroundBackgroundMotion:
    """Класс для разделения движения переднего и заднего плана."""
    
    @staticmethod
    def separate_motion(flow: np.ndarray, 
                       method: str = 'magnitude_threshold',
                       threshold: float = 0.5,
                       segmentation_mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Разделяет движение на передний и задний план.
        
        Args:
            flow: Оптический поток [H, W, 2] или [2, H, W]
            method: Метод разделения ('magnitude_threshold', 'spatial_clustering', 'segmentation')
            threshold: Порог для magnitude_threshold метода
            segmentation_mask: Опциональная маска сегментации (True = foreground)
            
        Returns:
            Словарь с разделенными потоками и статистиками
        """
        # Нормализация формы потока
        if flow.shape[0] == 2:
            flow = np.transpose(flow, (1, 2, 0))
        
        dx = flow[..., 0]
        dy = flow[..., 1]
        magnitude = np.sqrt(dx**2 + dy**2)
        
        if method == 'segmentation' and segmentation_mask is not None:
            # Используем предоставленную маску сегментации
            fg_mask = segmentation_mask.astype(bool)
            bg_mask = ~fg_mask
        elif method == 'spatial_clustering':
            # Кластеризация по величине и позиции
            fg_mask, bg_mask = ForegroundBackgroundMotion._spatial_clustering(magnitude)
        else:
            # По умолчанию: порог по величине
            fg_mask = magnitude > threshold
            bg_mask = magnitude <= threshold
        
        # Разделяем потоки
        fg_magnitude = magnitude[fg_mask] if np.any(fg_mask) else np.array([])
        bg_magnitude = magnitude[bg_mask] if np.any(bg_mask) else np.array([])
        
        # Вычисляем статистики
        features = {
            'foreground_motion_energy': float(np.sum(fg_magnitude**2)) if len(fg_magnitude) > 0 else 0.0,
            'background_motion_energy': float(np.sum(bg_magnitude**2)) if len(bg_magnitude) > 0 else 0.0,
            'foreground_motion_mean': float(np.mean(fg_magnitude)) if len(fg_magnitude) > 0 else 0.0,
            'background_motion_mean': float(np.mean(bg_magnitude)) if len(bg_magnitude) > 0 else 0.0,
            'foreground_motion_std': float(np.std(fg_magnitude)) if len(fg_magnitude) > 0 else 0.0,
            'background_motion_std': float(np.std(bg_magnitude)) if len(bg_magnitude) > 0 else 0.0,
            'foreground_coverage_ratio': float(np.mean(fg_mask)),
            'background_coverage_ratio': float(np.mean(bg_mask)),
            'ratio_foreground_background_flow': float(
                (np.sum(fg_magnitude**2) / (np.sum(bg_magnitude**2) + 1e-10)) 
                if len(fg_magnitude) > 0 and len(bg_magnitude) > 0 else 0.0
            ),
            'foreground_max': float(np.max(fg_magnitude)) if len(fg_magnitude) > 0 else 0.0,
            'background_max': float(np.max(bg_magnitude)) if len(bg_magnitude) > 0 else 0.0
        }
        
        return {
            'foreground_mask': fg_mask,
            'background_mask': bg_mask,
            'foreground_flow': flow[fg_mask] if np.any(fg_mask) else np.array([]),
            'background_flow': flow[bg_mask] if np.any(bg_mask) else np.array([]),
            'features': features
        }
    
    @staticmethod
    def _spatial_clustering(magnitude: np.ndarray, n_clusters: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """Кластеризация по величине движения для разделения FG/BG."""
        h, w = magnitude.shape
        # Создаем признаки: величина + позиция
        y_coords, x_coords = np.mgrid[0:h, 0:w]
        features = np.stack([
            magnitude.flatten(),
            x_coords.flatten() / w,  # Нормализованные координаты
            y_coords.flatten() / h
        ], axis=1)
        
        # K-means кластеризация
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = kmeans.fit_predict(features)
        
        # Кластер с большей средней величиной = foreground
        cluster_means = [magnitude.flatten()[labels == i].mean() for i in range(n_clusters)]
        fg_cluster = np.argmax(cluster_means)
        
        fg_mask = (labels == fg_cluster).reshape(h, w)
        bg_mask = ~fg_mask
        
        return fg_mask, bg_mask


class MotionClusters:
    """Класс для кластеризации векторов движения."""
    
    @staticmethod
    def cluster_motion(flow: np.ndarray, 
                      n_clusters: int = 5,
                      method: str = 'direction_speed',
                      sample_ratio: float = 0.1) -> Dict[str, Any]:
        """
        Кластеризует векторы движения по направлению и скорости.
        
        Args:
            flow: Оптический поток [H, W, 2] или [2, H, W]
            n_clusters: Количество кластеров
            method: Метод кластеризации ('direction_speed', 'full_vector')
            sample_ratio: Доля пикселей для выборки (для ускорения)
            
        Returns:
            Словарь с результатами кластеризации
        """
        # Нормализация формы
        if flow.shape[0] == 2:
            flow = np.transpose(flow, (1, 2, 0))
        
        dx = flow[..., 0]
        dy = flow[..., 1]
        magnitude = np.sqrt(dx**2 + dy**2)
        direction = np.arctan2(dy, dx)
        
        h, w = magnitude.shape
        
        # Выборка пикселей
        total_pixels = h * w
        n_sample = max(1000, int(total_pixels * sample_ratio))
        
        if method == 'direction_speed':
            # Кластеризация по направлению и скорости
            features = np.stack([
                magnitude.flatten(),
                np.cos(direction.flatten()),
                np.sin(direction.flatten())
            ], axis=1)
        else:
            # Полный вектор
            features = np.stack([
                dx.flatten(),
                dy.flatten(),
                magnitude.flatten()
            ], axis=1)
        
        # Выборка
        if n_sample < total_pixels:
            indices = np.random.choice(total_pixels, n_sample, replace=False)
            features_sample = features[indices]
        else:
            features_sample = features
            indices = np.arange(total_pixels)
        
        # Нормализация признаков
        features_norm = (features_sample - features_sample.mean(axis=0)) / (features_sample.std(axis=0) + 1e-10)
        
        # K-means кластеризация
        try:
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            cluster_labels = kmeans.fit_predict(features_norm)
            
            # Распространяем метки на все пиксели (ближайший кластер)
            if n_sample < total_pixels:
                # Для остальных пикселей находим ближайший кластер
                all_features_norm = (features - features_sample.mean(axis=0)) / (features_sample.std(axis=0) + 1e-10)
                all_labels = kmeans.predict(all_features_norm)
            else:
                all_labels = cluster_labels
            
            # Формируем маску кластеров
            cluster_mask = all_labels.reshape(h, w)
            
            # Статистики по кластерам
            cluster_stats = []
            for i in range(n_clusters):
                cluster_mask_i = cluster_mask == i
                cluster_mag = magnitude[cluster_mask_i]
                
                if len(cluster_mag) > 0:
                    cluster_dir = direction[cluster_mask_i]
                    cluster_stats.append({
                        'cluster_id': i,
                        'size': int(np.sum(cluster_mask_i)),
                        'coverage_ratio': float(np.mean(cluster_mask_i)),
                        'mean_magnitude': float(np.mean(cluster_mag)),
                        'std_magnitude': float(np.std(cluster_mag)),
                        'mean_direction': float(np.arctan2(
                            np.mean(np.sin(cluster_dir)),
                            np.mean(np.cos(cluster_dir))
                        )),
                        'max_magnitude': float(np.max(cluster_mag))
                    })
            
            # Сортируем по размеру
            cluster_stats.sort(key=lambda x: x['size'], reverse=True)
            
            features = {
                'num_motion_clusters': len(cluster_stats),
                'largest_cluster_size': cluster_stats[0]['size'] if cluster_stats else 0,
                'largest_cluster_coverage': cluster_stats[0]['coverage_ratio'] if cluster_stats else 0.0,
                'cluster_size_distribution': [s['size'] for s in cluster_stats],
                'cluster_coverage_distribution': [s['coverage_ratio'] for s in cluster_stats],
                'dominant_cluster_magnitude': cluster_stats[0]['mean_magnitude'] if cluster_stats else 0.0,
                'cluster_diversity': float(len(cluster_stats) / n_clusters)  # Насколько равномерно распределены
            }
            
            return {
                'cluster_mask': cluster_mask,
                'cluster_stats': cluster_stats,
                'features': features
            }
        except Exception as e:
            logger.warning(f"Ошибка кластеризации движения: {e}")
            return {
                'cluster_mask': np.zeros((h, w), dtype=int),
                'cluster_stats': [],
                'features': {
                    'num_motion_clusters': 0,
                    'error': str(e)
                }
            }


class SmoothnessJerkiness:
    """Класс для вычисления метрик плавности и резкости движения."""
    
    @staticmethod
    def compute_smoothness_metrics(flow_sequence: List[np.ndarray],
                                  fps: float = 25.0,
                                  frame_skip: int = 1) -> Dict[str, Any]:
        """
        Вычисляет метрики плавности и резкости движения.
        
        Args:
            flow_sequence: Список оптических потоков
            fps: Кадров в секунду
            frame_skip: Шаг между кадрами
            
        Returns:
            Словарь с метриками
        """
        if len(flow_sequence) < 3:
            return {'error': 'insufficient_frames'}
        
        # Вычисляем производные (скорость изменения потока)
        magnitudes = []
        accelerations = []
        jerks = []  # Вторая производная
        
        for i in range(len(flow_sequence)):
            flow = flow_sequence[i]
            if flow.shape[0] == 2:
                flow = np.transpose(flow, (1, 2, 0))
            
            mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
            magnitudes.append(np.mean(mag))
            
            if i > 0:
                # Ускорение (изменение скорости)
                prev_mag = magnitudes[i-1]
                accel = abs(magnitudes[i] - prev_mag)
                accelerations.append(accel)
                
                if i > 1:
                    # Рывок (изменение ускорения)
                    prev_accel = accelerations[i-2] if len(accelerations) > 1 else 0
                    jerk = abs(accel - prev_accel)
                    jerks.append(jerk)
        
        # Метрики плавности
        if len(accelerations) > 0:
            accel_array = np.array(accelerations)
            smoothness_index = 1.0 / (1.0 + np.mean(accel_array))
            jerkiness_index = np.mean(jerks) if len(jerks) > 0 else 0.0
        else:
            smoothness_index = 1.0
            jerkiness_index = 0.0
        
        # Временная энтропия потока
        if len(magnitudes) > 1:
            mag_array = np.array(magnitudes)
            # Энтропия изменений
            changes = np.diff(mag_array)
            hist, _ = np.histogram(changes, bins=50)
            hist = hist / (hist.sum() + 1e-10)
            hist = hist[hist > 0]
            flow_temporal_entropy = float(-np.sum(hist * np.log2(hist + 1e-10)))
        else:
            flow_temporal_entropy = 0.0
        
        # Стабильность движения
        if len(magnitudes) > 1:
            mag_array = np.array(magnitudes)
            cv = np.std(mag_array) / (np.mean(mag_array) + 1e-10)
            movement_stability = 1.0 / (1.0 + cv)
        else:
            movement_stability = 1.0
        
        return {
            'smoothness_index': float(smoothness_index),
            'jerkiness_index': float(jerkiness_index),
            'flow_temporal_entropy': float(flow_temporal_entropy),
            'movement_stability': float(movement_stability),
            'mean_acceleration': float(np.mean(accelerations)) if len(accelerations) > 0 else 0.0,
            'std_acceleration': float(np.std(accelerations)) if len(accelerations) > 0 else 0.0,
            'max_acceleration': float(np.max(accelerations)) if len(accelerations) > 0 else 0.0,
            'mean_jerk': float(np.mean(jerks)) if len(jerks) > 0 else 0.0,
            'max_jerk': float(np.max(jerks)) if len(jerks) > 0 else 0.0
        }
    
    @staticmethod
    def compute_frame_smoothness(flow_current: np.ndarray,
                                 flow_previous: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        Вычисляет метрики плавности для одного кадра относительно предыдущего.
        
        Args:
            flow_current: Текущий оптический поток
            flow_previous: Предыдущий оптический поток (опционально)
            
        Returns:
            Словарь с метриками
        """
        if flow_current.shape[0] == 2:
            flow_current = np.transpose(flow_current, (1, 2, 0))
        
        mag_current = np.sqrt(flow_current[..., 0]**2 + flow_current[..., 1]**2)
        
        features = {
            'magnitude_mean': float(np.mean(mag_current)),
            'magnitude_std': float(np.std(mag_current)),
            'magnitude_max': float(np.max(mag_current))
        }
        
        if flow_previous is not None:
            if flow_previous.shape[0] == 2:
                flow_previous = np.transpose(flow_previous, (1, 2, 0))
            
            mag_previous = np.sqrt(flow_previous[..., 0]**2 + flow_previous[..., 1]**2)
            
            # Изменение величины
            mag_change = np.abs(mag_current - mag_previous)
            features.update({
                'magnitude_change_mean': float(np.mean(mag_change)),
                'magnitude_change_std': float(np.std(mag_change)),
                'temporal_consistency': float(1.0 / (1.0 + np.mean(mag_change)))
            })
        
        return features

