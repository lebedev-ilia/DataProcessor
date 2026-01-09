"""
Quality validator для проверки качества фичей в NPZ артефактах.

Проверяет:
1. Статистические свойства (распределения, выбросы, NaN)
2. Семантическую валидность (разумные диапазоны значений)
3. Согласованность между связанными фичами
4. Качество данных (не все нули, не все одинаковые)
"""

from __future__ import annotations

import os
import glob
import json
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path


@dataclass
class QualityIssue:
    """Проблема качества фичи."""
    component: str
    feature_name: str
    severity: str  # "error", "warning", "info"
    message: str
    value: Any = None
    expected_range: Optional[Tuple[float, float]] = None


class QualityValidator:
    """Валидатор качества фичей."""
    
    # Семантические ограничения для фичей
    SEMANTIC_RANGES = {
        # Audio
        "tempo_bpm": (40.0, 220.0),  # разумный диапазон BPM
        "tempo_bpm_mean": (40.0, 220.0),
        "tempo_bpm_median": (40.0, 220.0),
        "loudness_lufs": (-70.0, 0.0),  # LUFS обычно -70 to 0 dB
        "loudness_dbfs": (-60.0, 0.0),  # dBFS обычно -60 to 0 dB
        "loudness_rms": (0.0, 1.0),  # RMS нормализованный
        "loudness_peak": (0.0, 1.0),  # Peak нормализованный
        # Visual
        "shot_quality_score": (0.0, 1.0),
        "aesthetic_score": (0.0, 1.0),
        "luxury_score": (0.0, 1.0),
        "confidence": (0.0, 1.0),
        # CLAP embeddings
        "clap_norm": (0.8, 1.2),  # нормализованный embedding должен быть близок к 1.0
    }
    
    def __init__(self, run_dir: str):
        """
        Args:
            run_dir: Путь к директории run (содержит manifest.json и компоненты)
        """
        self.run_dir = Path(run_dir)
        self.manifest_path = self.run_dir / "manifest.json"
        self.issues: List[QualityIssue] = []
        
    def validate_all(self) -> Tuple[bool, List[QualityIssue], Dict[str, Any]]:
        """
        Валидирует все артефакты в run.
        
        Returns:
            (ok, issues, summary)
        """
        self.issues = []
        
        if not self.manifest_path.exists():
            self.issues.append(QualityIssue(
                component="system",
                feature_name="manifest",
                severity="error",
                message="manifest.json not found"
            ))
            return False, self.issues, {}
        
        # Загружаем manifest
        try:
            with open(self.manifest_path, 'r') as f:
                manifest = json.load(f)
        except Exception as e:
            self.issues.append(QualityIssue(
                component="system",
                feature_name="manifest",
                severity="error",
                message=f"Failed to load manifest.json: {e}"
            ))
            return False, self.issues, {}
        
        # Находим все NPZ артефакты
        npz_paths = sorted(glob.glob(str(self.run_dir / "**" / "*.npz"), recursive=True))
        
        summary = {
            "total_artifacts": len(npz_paths),
            "components_checked": {},
            "statistics": {},
            "feature_statistics": {},  # Статистики по фичам для анализа
        }
        
        # Валидируем каждый артефакт
        for npz_path in npz_paths:
            component_name = self._extract_component_name(npz_path)
            if component_name:
                stats = self._validate_artifact(npz_path, component_name, manifest)
                if component_name not in summary["components_checked"]:
                    summary["components_checked"][component_name] = 0
                summary["components_checked"][component_name] += 1
                if stats:
                    summary["feature_statistics"][component_name] = stats
        
        # Кросс-компонентные проверки согласованности
        self._validate_cross_component_consistency(npz_paths, manifest)
        
        # Подсчитываем статистику
        error_count = sum(1 for i in self.issues if i.severity == "error")
        warning_count = sum(1 for i in self.issues if i.severity == "warning")
        info_count = sum(1 for i in self.issues if i.severity == "info")
        
        summary["issues"] = {
            "error": error_count,
            "warning": warning_count,
            "info": info_count,
            "total": len(self.issues),
        }
        
        ok = error_count == 0
        return ok, self.issues, summary
    
    def _extract_component_name(self, npz_path: Path) -> Optional[str]:
        """Извлекает имя компонента из пути к NPZ."""
        rel_path = Path(npz_path).relative_to(self.run_dir)
        parts = rel_path.parts
        if len(parts) >= 1:
            return parts[0]
        return None
    
    def _validate_artifact(self, npz_path: Path, component_name: str, manifest: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Валидирует один NPZ артефакт.
        
        Returns:
            Статистики по фичам или None
        """
        stats = {}
        try:
            data = np.load(npz_path, allow_pickle=True)
        except Exception as e:
            self.issues.append(QualityIssue(
                component=component_name,
                feature_name="file",
                severity="error",
                message=f"Failed to load NPZ: {e}"
            ))
            return None
        
        # Проверяем meta
        meta = data.get("meta")
        if meta is None:
            self.issues.append(QualityIssue(
                component=component_name,
                feature_name="meta",
                severity="error",
                message="Missing 'meta' field"
            ))
        else:
            # Unbox scalar object array if needed
            if isinstance(meta, np.ndarray) and meta.dtype == object and meta.ndim == 0:
                meta = meta.item()
            
            if isinstance(meta, dict):
                status = meta.get("status")
                if status == "error":
                    error_msg = meta.get("error", "unknown error")
                    self.issues.append(QualityIssue(
                        component=component_name,
                        feature_name="status",
                        severity="error",
                        message=f"Component reported error: {error_msg}"
                    ))
        
        # Компонент-специфичные проверки
        if component_name == "tempo_extractor":
            self._validate_tempo(data, component_name)
        elif component_name == "loudness_extractor":
            self._validate_loudness(data, component_name)
        elif component_name == "clap_extractor":
            self._validate_clap(data, component_name)
        elif component_name.startswith("core_") or component_name in ["cut_detection", "shot_quality", "video_pacing", "story_structure", "similarity_metrics", "text_scoring", "uniqueness", "scene_classification"]:
            self._validate_visual_component(data, component_name)
        
        # Общие проверки для всех компонентов
        self._validate_general_quality(data, component_name)
        
        # Собираем статистики по ключевым фичам
        stats = self._collect_feature_statistics(data, component_name)
        return stats
    
    def _validate_tempo(self, data: np.ndarray, component_name: str) -> None:
        """Валидация tempo_extractor."""
        # Проверяем feature_values
        feature_values = data.get("feature_values")
        feature_names = data.get("feature_names")
        
        if feature_values is None or feature_names is None:
            self.issues.append(QualityIssue(
                component=component_name,
                feature_name="feature_values",
                severity="error",
                message="Missing feature_values or feature_names"
            ))
            return
        
        # Unbox если нужно
        if isinstance(feature_names, np.ndarray) and feature_names.dtype == object:
            feature_names = feature_names.tolist()
        if isinstance(feature_values, np.ndarray):
            feature_values = np.asarray(feature_values, dtype=np.float32)
        
        # Создаем словарь для удобного доступа
        features = {}
        if isinstance(feature_names, (list, np.ndarray)) and len(feature_names) == len(feature_values):
            for name, value in zip(feature_names, feature_values):
                features[str(name)] = float(value) if not np.isnan(value) else None
        
        # Проверяем tempo_bpm
        tempo_bpm = features.get("tempo_bpm")
        if tempo_bpm is not None:
            if not (40.0 <= tempo_bpm <= 220.0):
                self.issues.append(QualityIssue(
                    component=component_name,
                    feature_name="tempo_bpm",
                    severity="warning",
                    message=f"Tempo BPM out of typical range: {tempo_bpm:.2f}",
                    value=tempo_bpm,
                    expected_range=(40.0, 220.0)
                ))
            elif tempo_bpm < 60 or tempo_bpm > 180:
                self.issues.append(QualityIssue(
                    component=component_name,
                    feature_name="tempo_bpm",
                    severity="info",
                    message=f"Tempo BPM unusual but valid: {tempo_bpm:.2f}",
                    value=tempo_bpm
                ))
        
        # Проверяем confidence
        confidence = features.get("tempo_confidence")
        if confidence is not None:
            if not (0.0 <= confidence <= 1.0):
                self.issues.append(QualityIssue(
                    component=component_name,
                    feature_name="tempo_confidence",
                    severity="error",
                    message=f"Confidence out of [0,1] range: {confidence:.3f}",
                    value=confidence,
                    expected_range=(0.0, 1.0)
                ))
            elif confidence < 0.3:
                self.issues.append(QualityIssue(
                    component=component_name,
                    feature_name="tempo_confidence",
                    severity="warning",
                    message=f"Low confidence: {confidence:.3f}",
                    value=confidence
                ))
        
        # Проверяем tempo_estimates
        tempo_estimates = data.get("tempo_estimates")
        if tempo_estimates is not None:
            tempo_estimates = np.asarray(tempo_estimates, dtype=np.float32)
            if tempo_estimates.size > 0:
                valid_estimates = tempo_estimates[(tempo_estimates >= 40) & (tempo_estimates <= 220)]
                if len(valid_estimates) < len(tempo_estimates) * 0.8:
                    self.issues.append(QualityIssue(
                        component=component_name,
                        feature_name="tempo_estimates",
                        severity="warning",
                        message=f"Many tempo estimates out of range: {len(valid_estimates)}/{len(tempo_estimates)} valid"
                    ))
    
    def _validate_loudness(self, data: np.ndarray, component_name: str) -> None:
        """Валидация loudness_extractor."""
        feature_values = data.get("feature_values")
        feature_names = data.get("feature_names")
        
        if feature_values is None or feature_names is None:
            return
        
        if isinstance(feature_names, np.ndarray) and feature_names.dtype == object:
            feature_names = feature_names.tolist()
        if isinstance(feature_values, np.ndarray):
            feature_values = np.asarray(feature_values, dtype=np.float32)
        
        features = {}
        if isinstance(feature_names, (list, np.ndarray)) and len(feature_names) == len(feature_values):
            for name, value in zip(feature_names, feature_values):
                features[str(name)] = float(value) if not np.isnan(value) else None
        
        # Проверяем LUFS (может быть NaN если pyloudnorm недоступен)
        lufs = features.get("loudness_lufs")
        lufs_present = data.get("lufs_present")
        if lufs_present is not None:
            lufs_present = bool(np.asarray(lufs_present).item() if isinstance(lufs_present, np.ndarray) else lufs_present)
            if lufs_present and lufs is not None:
                if not (-70.0 <= lufs <= 0.0):
                    self.issues.append(QualityIssue(
                        component=component_name,
                        feature_name="loudness_lufs",
                        severity="warning",
                        message=f"LUFS out of typical range: {lufs:.2f}",
                        value=lufs,
                        expected_range=(-70.0, 0.0)
                    ))
        
        # Проверяем dBFS
        dbfs = features.get("loudness_dbfs")
        if dbfs is not None:
            if not (-60.0 <= dbfs <= 0.0):
                self.issues.append(QualityIssue(
                    component=component_name,
                    feature_name="loudness_dbfs",
                    severity="warning",
                    message=f"dBFS out of typical range: {dbfs:.2f}",
                    value=dbfs,
                    expected_range=(-60.0, 0.0)
                ))
        
        # Проверяем RMS (должен быть > 0 для не-тихого аудио)
        rms = features.get("loudness_rms")
        if rms is not None:
            if rms < 1e-6:
                self.issues.append(QualityIssue(
                    component=component_name,
                    feature_name="loudness_rms",
                    severity="warning",
                    message=f"RMS extremely low (possibly silent audio): {rms:.6f}",
                    value=rms
                ))
    
    def _validate_clap(self, data: np.ndarray, component_name: str) -> None:
        """Валидация clap_extractor."""
        embedding = data.get("embedding")
        embedding_present = data.get("embedding_present")
        
        if embedding_present is not None:
            embedding_present = bool(np.asarray(embedding_present).item() if isinstance(embedding_present, np.ndarray) else embedding_present)
            if embedding_present and embedding is not None:
                emb = np.asarray(embedding, dtype=np.float32)
                if emb.size > 0:
                    # Проверяем размерность (ожидаем 512 для CLAP)
                    if emb.shape[0] != 512:
                        self.issues.append(QualityIssue(
                            component=component_name,
                            feature_name="embedding_dim",
                            severity="warning",
                            message=f"CLAP embedding dimension unexpected: {emb.shape[0]} (expected 512)",
                            value=emb.shape[0]
                        ))
                    
                    # Проверяем норму (должна быть близка к 1.0 если нормализован)
                    norm = float(np.linalg.norm(emb))
                    if norm < 0.1 or norm > 10.0:
                        self.issues.append(QualityIssue(
                            component=component_name,
                            feature_name="embedding_norm",
                            severity="warning",
                            message=f"CLAP embedding norm unusual: {norm:.3f} (expected ~1.0 if normalized)",
                            value=norm
                        ))
                    
                    # Проверяем на все нули
                    if np.allclose(emb, 0.0):
                        self.issues.append(QualityIssue(
                            component=component_name,
                            feature_name="embedding",
                            severity="error",
                            message="CLAP embedding is all zeros"
                        ))
                    
                    # Проверяем на NaN/Inf
                    if np.any(np.isnan(emb)) or np.any(np.isinf(emb)):
                        nan_count = np.sum(np.isnan(emb))
                        inf_count = np.sum(np.isinf(emb))
                        self.issues.append(QualityIssue(
                            component=component_name,
                            feature_name="embedding",
                            severity="error",
                            message=f"CLAP embedding contains NaN/Inf: {nan_count} NaN, {inf_count} Inf"
                        ))
        
        # Проверяем feature_values
        feature_values = data.get("feature_values")
        feature_names = data.get("feature_names")
        if feature_values is not None and feature_names is not None:
            if isinstance(feature_names, np.ndarray) and feature_names.dtype == object:
                feature_names = feature_names.tolist()
            if isinstance(feature_values, np.ndarray):
                feature_values = np.asarray(feature_values, dtype=np.float32)
            
            features = {}
            if isinstance(feature_names, (list, np.ndarray)) and len(feature_names) == len(feature_values):
                for name, value in zip(feature_names, feature_values):
                    features[str(name)] = float(value) if not np.isnan(value) else None
            
            clap_norm = features.get("clap_norm")
            if clap_norm is not None:
                if not (0.8 <= clap_norm <= 1.2):
                    self.issues.append(QualityIssue(
                        component=component_name,
                        feature_name="clap_norm",
                        severity="warning",
                        message=f"CLAP norm out of expected range: {clap_norm:.3f}",
                        value=clap_norm,
                        expected_range=(0.8, 1.2)
                    ))
    
    def _validate_visual_component(self, data: np.ndarray, component_name: str) -> None:
        """Валидация visual компонентов."""
        # Проверяем frame_indices если есть
        frame_indices = data.get("frame_indices")
        if frame_indices is not None:
            fi = np.asarray(frame_indices, dtype=np.int32)
            if fi.size > 0:
                # Проверяем сортировку
                if not np.all(fi[:-1] <= fi[1:]):
                    self.issues.append(QualityIssue(
                        component=component_name,
                        feature_name="frame_indices",
                        severity="error",
                        message="frame_indices not sorted"
                    ))
                
                # Проверяем уникальность
                if len(np.unique(fi)) != len(fi):
                    self.issues.append(QualityIssue(
                        component=component_name,
                        feature_name="frame_indices",
                        severity="error",
                        message="frame_indices contains duplicates"
                    ))
                
                # Проверяем на отрицательные значения
                if np.any(fi < 0):
                    self.issues.append(QualityIssue(
                        component=component_name,
                        feature_name="frame_indices",
                        severity="error",
                        message="frame_indices contains negative values"
                    ))
        
        # Для core_clip проверяем embeddings
        if component_name == "core_clip":
            frame_embeddings = data.get("frame_embeddings")
            if frame_embeddings is not None:
                emb = np.asarray(frame_embeddings, dtype=np.float32)
                if emb.size > 0:
                    # Проверяем размерность (ожидаем 512 для ViT-B/32)
                    if emb.ndim == 2 and emb.shape[1] != 512:
                        self.issues.append(QualityIssue(
                            component=component_name,
                            feature_name="frame_embeddings_dim",
                            severity="warning",
                            message=f"CLIP embedding dimension unexpected: {emb.shape[1]} (expected 512 for ViT-B/32)",
                            value=emb.shape[1]
                        ))
                    
                    # Проверяем на все нули
                    if np.allclose(emb, 0.0):
                        self.issues.append(QualityIssue(
                            component=component_name,
                            feature_name="frame_embeddings",
                            severity="error",
                            message="CLIP embeddings are all zeros"
                        ))
        
        # Для shot_quality проверяем scores
        if component_name == "shot_quality":
            quality_probs = data.get("quality_probs")
            if quality_probs is None:
                self.issues.append(QualityIssue(
                    component=component_name,
                    feature_name="quality_probs",
                    severity="warning",
                    message="Missing quality_probs (expected (N,P) float16)"
                ))
            else:
                qp = np.asarray(quality_probs, dtype=np.float32)
                if qp.ndim != 2 or qp.shape[0] <= 0 or qp.shape[1] <= 0:
                    self.issues.append(QualityIssue(
                        component=component_name,
                        feature_name="quality_probs",
                        severity="error",
                        message=f"quality_probs has invalid shape: {list(qp.shape)}"
                    ))
                else:
                    row_sums = np.sum(qp, axis=1)
                    if np.any(row_sums < 0.95) or np.any(row_sums > 1.05):
                        self.issues.append(QualityIssue(
                            component=component_name,
                            feature_name="quality_probs",
                            severity="warning",
                            message=f"quality_probs rows not close to 1.0 (min_sum={row_sums.min():.3f}, max_sum={row_sums.max():.3f})"
                        ))
    
    def _validate_general_quality(self, data: np.ndarray, component_name: str) -> None:
        """Общие проверки качества для всех компонентов."""
        # Проверяем все числовые массивы на NaN/Inf
        for key in data.keys():
            if key == "meta":
                continue
            arr = data.get(key)
            if isinstance(arr, np.ndarray) and np.issubdtype(arr.dtype, np.floating):
                arr = np.asarray(arr, dtype=np.float32)
                if arr.size > 0:
                    nan_count = np.sum(np.isnan(arr))
                    inf_count = np.sum(np.isinf(arr))
                    if nan_count > 0 or inf_count > 0:
                        # NaN допустимы для missing values, но проверяем что не все
                        # Исключаем опциональные фичи где NaN нормальны
                        optional_features = ["hands_landmarks", "pose_landmarks", "face_landmarks"]
                        is_optional = any(opt in key for opt in optional_features)
                        
                        if nan_count == arr.size and not is_optional:
                            self.issues.append(QualityIssue(
                                component=component_name,
                                feature_name=key,
                                severity="error",
                                message=f"All values are NaN in {key}"
                            ))
                        elif nan_count > arr.size * 0.5 and not is_optional:
                            self.issues.append(QualityIssue(
                                component=component_name,
                                feature_name=key,
                                severity="warning",
                                message=f"More than 50% NaN in {key}: {nan_count}/{arr.size}"
                            ))
                    
                    if inf_count > 0:
                        self.issues.append(QualityIssue(
                            component=component_name,
                            feature_name=key,
                            severity="error",
                            message=f"Inf values in {key}: {inf_count}"
                        ))
                    
                    # Проверяем на все нули (может быть проблемой для некоторых фичей)
                    # Исключаем опциональные фичи где NaN нормальны
                    optional_features = ["hands_landmarks", "pose_landmarks", "face_landmarks"]  # NaN нормальны если нет объектов
                    if key not in optional_features:
                        if np.any(~np.isnan(arr)):
                            if np.allclose(arr[~np.isnan(arr)], 0.0):
                                if key not in ["frame_indices", "warnings"]:  # frame_indices могут быть нулями, warnings - это список
                                    self.issues.append(QualityIssue(
                                        component=component_name,
                                        feature_name=key,
                                        severity="info",
                                        message=f"All non-NaN values are zero in {key}"
                                    ))
    
    def _validate_cross_component_consistency(self, npz_paths: List[Path], manifest: Dict[str, Any]) -> None:
        """Проверки согласованности между компонентами."""
        # Загружаем все артефакты для проверки
        artifacts = {}
        for npz_path in npz_paths:
            component_name = self._extract_component_name(npz_path)
            if component_name:
                try:
                    data = np.load(npz_path, allow_pickle=True)
                    artifacts[component_name] = data
                except Exception:
                    continue
        
        # Проверяем согласованность frame_indices между visual компонентами
        visual_frame_indices = {}
        for name, data in artifacts.items():
            if name.startswith("core_") or name in ["cut_detection", "shot_quality", "video_pacing", "story_structure", "similarity_metrics", "text_scoring", "uniqueness", "scene_classification"]:
                fi = data.get("frame_indices")
                if fi is not None:
                    fi = np.asarray(fi, dtype=np.int32)
                    if fi.size > 0:
                        visual_frame_indices[name] = set(fi.tolist())
        
        # Проверяем что все visual компоненты используют подмножество union
        if len(visual_frame_indices) > 1:
            # Находим union (должен быть в metadata или можно собрать из всех)
            all_indices = set()
            for indices in visual_frame_indices.values():
                all_indices.update(indices)
            
            # Проверяем что каждый компонент использует подмножество union
            for name, indices in visual_frame_indices.items():
                if not indices.issubset(all_indices):
                    self.issues.append(QualityIssue(
                        component=name,
                        feature_name="frame_indices",
                        severity="error",
                        message=f"frame_indices contains values not in union domain"
                    ))
        
        # Проверяем согласованность audio_present
        audio_present_flags = {}
        for name, data in artifacts.items():
            if name in ["clap_extractor", "tempo_extractor", "loudness_extractor"]:
                meta = data.get("meta")
                if meta is not None:
                    if isinstance(meta, np.ndarray) and meta.dtype == object and meta.ndim == 0:
                        meta = meta.item()
                    if isinstance(meta, dict):
                        audio_present = meta.get("audio_present")
                        if audio_present is not None:
                            audio_present_flags[name] = bool(audio_present)
        
        # Все audio extractors должны иметь одинаковый audio_present
        if len(audio_present_flags) > 1:
            values = list(audio_present_flags.values())
            if not all(v == values[0] for v in values):
                self.issues.append(QualityIssue(
                    component="audio",
                    feature_name="audio_present",
                    severity="error",
                    message=f"Inconsistent audio_present flags: {audio_present_flags}"
                ))
        
        # Проверяем что если audio_present=False, то фичи должны быть empty или иметь empty_reason
        for name, data in artifacts.items():
            if name in ["clap_extractor", "tempo_extractor", "loudness_extractor"]:
                meta = data.get("meta")
                if meta is not None:
                    if isinstance(meta, np.ndarray) and meta.dtype == object and meta.ndim == 0:
                        meta = meta.item()
                    if isinstance(meta, dict):
                        audio_present = meta.get("audio_present", True)
                        status = meta.get("status")
                        empty_reason = meta.get("empty_reason")
                        
                        if not audio_present:
                                if status != "empty" or empty_reason is None:
                                    self.issues.append(QualityIssue(
                                        component=name,
                                        feature_name="status",
                                        severity="warning",
                                        message=f"audio_present=False but status={status}, empty_reason={empty_reason}"
                                    ))
    
    def _collect_feature_statistics(self, data: np.ndarray, component_name: str) -> Dict[str, Any]:
        """Собирает статистики по ключевым фичам для анализа."""
        stats = {}
        
        # Для audio extractors собираем статистики из feature_values
        if component_name in ["tempo_extractor", "loudness_extractor", "clap_extractor"]:
            feature_values = data.get("feature_values")
            feature_names = data.get("feature_names")
            if feature_values is not None and feature_names is not None:
                if isinstance(feature_names, np.ndarray) and feature_names.dtype == object:
                    feature_names = feature_names.tolist()
                if isinstance(feature_values, np.ndarray):
                    feature_values = np.asarray(feature_values, dtype=np.float32)
                
                if isinstance(feature_names, (list, np.ndarray)) and len(feature_names) == len(feature_values):
                    for name, value in zip(feature_names, feature_values):
                        name_str = str(name)
                        if not np.isnan(value):
                            stats[name_str] = {
                                "value": float(value),
                                "is_valid": True
                            }
        
        # Для CLAP проверяем embedding
        if component_name == "clap_extractor":
            embedding = data.get("embedding")
            if embedding is not None:
                emb = np.asarray(embedding, dtype=np.float32)
                if emb.size > 0:
                    stats["embedding"] = {
                        "shape": list(emb.shape),
                        "norm": float(np.linalg.norm(emb)),
                        "mean": float(np.mean(emb)),
                        "std": float(np.std(emb)),
                        "min": float(np.min(emb)),
                        "max": float(np.max(emb)),
                    }
        
        # Для visual компонентов собираем статистики по frame_indices
        if component_name.startswith("core_") or component_name in ["cut_detection", "shot_quality", "video_pacing", "story_structure", "similarity_metrics", "text_scoring", "uniqueness", "scene_classification"]:
            frame_indices = data.get("frame_indices")
            if frame_indices is not None:
                fi = np.asarray(frame_indices, dtype=np.int32)
                if fi.size > 0:
                    stats["frame_indices"] = {
                        "count": int(fi.size),
                        "min": int(np.min(fi)),
                        "max": int(np.max(fi)),
                        "unique": int(len(np.unique(fi))),
                    }
        
        return stats


def validate_quality(run_dir: str) -> Tuple[bool, List[QualityIssue], Dict[str, Any]]:
    """
    Валидирует качество фичей в run.
    
    Args:
        run_dir: Путь к директории run
        
    Returns:
        (ok, issues, summary)
    """
    validator = QualityValidator(run_dir)
    return validator.validate_all()


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: quality_validator.py <run_dir>")
        sys.exit(1)
    
    run_dir = sys.argv[1]
    ok, issues, summary = validate_quality(run_dir)
    
    print(f"\n=== Quality Validation Report ===")
    print(f"Run directory: {run_dir}")
    print(f"Status: {'✅ PASS' if ok else '❌ FAIL'}")
    print(f"\nSummary:")
    print(f"  Total artifacts: {summary.get('total_artifacts', 0)}")
    print(f"  Components checked: {len(summary.get('components_checked', {}))}")
    print(f"  Issues: {summary.get('issues', {})}")
    
    if issues:
        print(f"\n=== Issues by Severity ===")
        for severity in ["error", "warning", "info"]:
            sev_issues = [i for i in issues if i.severity == severity]
            if sev_issues:
                print(f"\n{severity.upper()} ({len(sev_issues)}):")
                for issue in sev_issues[:20]:  # Показываем первые 20
                    print(f"  [{issue.component}] {issue.feature_name}: {issue.message}")
                    if issue.value is not None:
                        print(f"    Value: {issue.value}")
                    if issue.expected_range is not None:
                        print(f"    Expected: {issue.expected_range}")
                if len(sev_issues) > 20:
                    print(f"  ... and {len(sev_issues) - 20} more")
    
    # Выводим статистики по ключевым фичам
    feature_stats = summary.get("feature_statistics", {})
    if feature_stats:
        print(f"\n=== Feature Statistics ===")
        for component, stats in sorted(feature_stats.items()):
            if stats:
                print(f"\n[{component}]:")
                for key, value in sorted(stats.items()):
                    if isinstance(value, dict):
                        print(f"  {key}:")
                        for k, v in sorted(value.items()):
                            if isinstance(v, float):
                                print(f"    {k}: {v:.4f}")
                            else:
                                print(f"    {k}: {v}")
                    else:
                        print(f"  {key}: {value}")
    
    sys.exit(0 if ok else 1)

