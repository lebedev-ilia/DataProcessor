"""
Рефакторинг DetalizeFaceExtractor с использованием модульной архитектуры.

Это демонстрация того, как можно рефакторить оригинальный класс для использования модулей.
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence, Tuple
from pathlib import Path
from collections import defaultdict, deque

import cv2
import mediapipe as mp
import numpy as np

from _modules import MODULE_REGISTRY
from _modules.base_module import FaceModule
from _utils import (
    landmarks_to_ndarray,
    validate_face_landmarks,
    compute_bbox,
    extract_roi,
)
from _utils.landmarks_utils import LANDMARKS

from utils.logger import get_logger
logger = get_logger("DetalizeFaceExtractorRefactored")

_FACE_MESH = mp.solutions.face_mesh

class DetalizeFaceExtractorRefactored():
    """
    Рефакторинг DetalizeFaceExtractor с использованием модульной архитектуры.
    
    Использует модули для извлечения различных типов фич лица.
    """

    def __init__(
        self,
        *,
        modules: Optional[List[str]] = None,
        module_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        max_faces: int = 10,
        refine_landmarks: bool = True,
        visualize: bool = False,
        visualize_dir: Optional[str] = None,
        show_landmarks: bool = False,
        # Quality filtering parameters
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.7,
        min_face_size: int = 30,
        max_face_size_ratio: float = 0.8,
        min_aspect_ratio: float = 0.6,
        max_aspect_ratio: float = 1.4,
        validate_landmarks: bool = True,
    ) -> None:
        """
        :param modules: список имен модулей для загрузки (если None - загружаются все)
        :param module_configs: конфигурации для конкретных модулей
        :param max_faces: maximum number of faces to detect per frame
        :param refine_landmarks: use refined landmarks (468 points)
        :param kwargs: дополнительные параметры для BaseExtractor
        """

        self.frames_with_face = self.frames_with_face_load("auto")

        self.max_faces = max_faces
        self.refine_landmarks = refine_landmarks

        logger.info(f"DetalizeFaceExtractorRefactored | init | face_mesh params | max_faces = {max_faces}")
        logger.info(f"                                                          | refine_landmarks = {refine_landmarks}")
        logger.info(f"                                                          | min_detection_confidence = {min_detection_confidence}")
        logger.info(f"                                                          | min_tracking_confidence = {min_tracking_confidence}")

        self.face_mesh = _FACE_MESH.FaceMesh(
            max_num_faces=max_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        # Quality filtering parameters
        self.min_face_size = max(10, min_face_size)
        self.max_face_size_ratio = np.clip(max_face_size_ratio, 0.1, 1.0)
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio
        self.validate_landmarks = validate_landmarks

        # Visualization settings
        self.visualize = visualize
        self.show_landmarks = show_landmarks and visualize
        if visualize:
            self.visualize_dir = Path(visualize_dir) if visualize_dir else Path("./face_visualizations")
            self.visualize_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"DetalizeFaceExtractorRefactored | init | Visualization enabled. Saving frames to: {self.visualize_dir}")

        logger.info(f"DetalizeFaceExtractorRefactored | init | load modules...")

        # Загружаем модули
        modules_to_load = modules or list(MODULE_REGISTRY.keys())
        self.modules: List[FaceModule] = []

        for module_name in modules_to_load:
            if module_name not in MODULE_REGISTRY:
                logger.info(f"DetalizeFaceExtractorRefactored | init | Модуль '{module_name}' не найден в registry, пропускаем")
                continue

            module_class = MODULE_REGISTRY[module_name]
            module_config = (module_configs or {}).get(module_name, {})

            try:
                module = module_class(config=module_config)
                self.modules.append(module)
                logger.info(f"DetalizeFaceExtractorRefactored | init | Загружен модуль: {module_name}")
            except Exception as e:
                logger.error(f"DetalizeFaceExtractorRefactored | init | Ошибка при загрузке модуля '{module_name}': {e}")

        if not self.modules:
            raise ValueError("DetalizeFaceExtractorRefactored | init | Не удалось загрузить ни одного модуля")

        # Инициализируем модули
        for module in self.modules:
            try:
                module.initialize()
            except Exception as e:
                logger.error(f"DetalizeFaceExtractorRefactored | init | Ошибка при инициализации модуля '{module.module_name}': {e}")

        # Tracking для мульти-лица (tracking_id для каждого лица)
        self._face_tracking: Dict[int, Dict[str, Any]] = defaultdict(dict)  # frame_idx -> {face_idx -> tracking_id}
        self._tracking_counter = 0
        self._track_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=30))  # tracking_id -> history

    def frames_with_face_load(self, filename):
        import os, json
        p = f"{os.path.dirname(os.path.dirname(os.path.dirname(__file__)))}/result_store/face_detection"
        try:
            if filename == "auto":
                filename = os.listdir(p)[-1]
            with open(f"{p}/{filename}", "r") as f:
                return json.load(f)["frames_with_face"]
        except Exception as e:
            logger.error(f"VideoEmotionProcessor | frames_with_face_load | Error: {e}")
            raise

    def extract(self, frame_manager) -> List[List[Dict[str, Any]]]:
        """
        Processes a sequence of OpenCV BGR frames and returns a list where each
        element corresponds to the facial feature set per frame.
        """
        outputs = {}

        for frame_idx in self.frames_with_face:

            frame = frame_manager.get(frame_idx)

            frame_results = self._process_frame(frame, frame_idx)
            outputs[frame_idx] = frame_results

            # Visualize frame if enabled and faces detected
            if self.visualize and frame_results:
                self._visualize_frame(frame, frame_idx, frame_results)

        return outputs

    __call__ = extract

    def _compute_iou(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Вычисляет IoU между двумя bbox."""
        x1_min, y1_min, x1_max, y1_max = bbox1
        x2_min, y2_min, x2_max, y2_max = bbox2
        
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max <= inter_x_min or inter_y_max <= inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        area1 = (x1_max - x1_min) * (y1_max - y1_min)
        area2 = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = area1 + area2 - inter_area
        
        return inter_area / max(union_area, 1e-6)

    def _assign_tracking_id(self, frame_idx: int, face_idx: int, bbox: np.ndarray, 
                           detection_confidence: float) -> int:
        """
        Назначает tracking_id для лица на основе IoU с предыдущими кадрами.
        """
        # Ищем совпадения в предыдущих кадрах (последние 3 кадра)
        best_match_id = None
        best_iou = 0.3  # Порог IoU для совпадения
        
        for prev_frame_idx in range(max(0, frame_idx - 3), frame_idx):
            if prev_frame_idx in self._face_tracking:
                for prev_face_idx, prev_tracking_id in self._face_tracking[prev_frame_idx].items():
                    # Получаем bbox из истории
                    if prev_tracking_id in self._track_history:
                        hist = list(self._track_history[prev_tracking_id])
                        if len(hist) > 0:
                            prev_bbox = hist[-1].get("bbox")
                            if prev_bbox is not None:
                                iou = self._compute_iou(bbox, np.array(prev_bbox))
                                if iou > best_iou:
                                    best_iou = iou
                                    best_match_id = prev_tracking_id
        
        if best_match_id is not None:
            return best_match_id
        else:
            # Новый track
            self._tracking_counter += 1
            return self._tracking_counter

    def _process_frame(self, frame: np.ndarray, frame_idx: int) -> List[Dict[str, Any]]:
        """
        Обрабатывает один кадр, используя модульную архитектуру.
        """
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return []

        # Собираем все лица с их bbox для определения primary face
        faces_with_bbox = []
        for face_idx, landmark_list in enumerate(results.multi_face_landmarks):
            coords = landmarks_to_ndarray(landmark_list, width, height)
            bbox = compute_bbox(coords, width, height)
            
            # Валидация
            if not validate_face_landmarks(
                bbox, coords, width, height,
                min_face_size=self.min_face_size,
                max_face_size_ratio=self.max_face_size_ratio,
                min_aspect_ratio=self.min_aspect_ratio,
                max_aspect_ratio=self.max_aspect_ratio,
                validate_landmarks=self.validate_landmarks,
            ):
                continue
            
            # Вычисляем площадь для определения primary face
            bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            faces_with_bbox.append((face_idx, landmark_list, coords, bbox, bbox_area))
        
        # Сортируем по площади (primary face = наибольшая площадь)
        faces_with_bbox.sort(key=lambda x: x[4], reverse=True)
        
        # Определяем primary face (индекс 0) и secondary faces
        frame_features: List[Dict[str, Any]] = []
        self._face_tracking[frame_idx] = {}
        
        for idx, (face_idx, landmark_list, coords, bbox, bbox_area) in enumerate(faces_with_bbox):
            # Detection confidence (MediaPipe не предоставляет напрямую, используем эвристику)
            # Можно использовать стабильность landmarks или другие метрики
            detection_confidence = 0.9  # По умолчанию, можно улучшить
            
            # Назначаем tracking_id
            tracking_id = self._assign_tracking_id(frame_idx, face_idx, bbox, detection_confidence)
            self._face_tracking[frame_idx][face_idx] = tracking_id
            
            # Обновляем историю трека
            if tracking_id not in self._track_history:
                self._track_history[tracking_id] = deque(maxlen=30)
            
            self._track_history[tracking_id].append({
                "bbox": bbox.tolist(),
                "frame_idx": frame_idx,
                "detection_confidence": detection_confidence,
            })

            # 4. Извлекаем ROI
            roi = extract_roi(frame, bbox)

            # 5. Конвертируем coords в систему координат ROI
            bbox_x_min, bbox_y_min = bbox[0], bbox[1]
            coords_roi = coords.copy()
            coords_roi[:, 0] -= bbox_x_min
            coords_roi[:, 1] -= bbox_y_min

            # 6. Подготавливаем shared_data для модулей
            shared_data = {
                "coords": coords,
                "coords_roi": coords_roi,
                "bbox": bbox,
                "roi": roi,
                "frame_shape": frame.shape,
                "face_idx": face_idx,
                "tracking_id": tracking_id,
                "detection_confidence": detection_confidence,
                "is_primary_face": (idx == 0),  # Первое лицо = primary
            }

            # 7. Обрабатываем через модули
            face_feature = {
                "frame_index": frame_idx,
                "face_index": face_idx,
                "bbox": bbox.tolist(),
                "detection_confidence": detection_confidence,
                "tracking_id": tracking_id,
                "is_primary_face": (idx == 0),
            }

            # Обрабатываем модули в порядке загрузки
            for module in self.modules:
                if not module.can_process(shared_data):
                    continue

                try:
                    module_result = module.process(shared_data)
                    face_feature.update(module_result)

                    # Обновляем shared_data для зависимостей между модулями
                    # (например, professional зависит от quality, eyes, motion, pose, lip_reading)
                    for key, value in module_result.items():
                        # Добавляем все результаты модулей в shared_data для зависимостей
                        shared_data[key] = value

                except Exception as e:
                    logger.error(
                        f"Ошибка в модуле '{module.module_name}' на кадре {frame_idx}, лицо {face_idx}: {e}",
                        exc_info=True,
                    )

            # Сохраняем landmarks для визуализации
            if self.visualize:
                face_feature["_landmarks_coords"] = coords.tolist()

            frame_features.append(face_feature)

        return frame_features

    def _visualize_frame(
        self, frame: np.ndarray, frame_idx: int, frame_results: List[Dict[str, Any]]
    ) -> None:
        """Visualize faces with bounding boxes and optionally landmarks."""
        vis_frame = frame.copy()
        
        for face_result in frame_results:
            bbox = face_result["bbox"]
            face_idx = face_result.get("face_index", 0)
            
            # Draw bounding box
            x_min, y_min, x_max, y_max = [int(coord) for coord in bbox]
            color = (0, 255, 0)  # Green
            cv2.rectangle(vis_frame, (x_min, y_min), (x_max, y_max), color, 2)
            
            # Draw face index label
            label = f"Face {face_idx}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(
                vis_frame,
                (x_min, y_min - label_size[1] - 5),
                (x_min + label_size[0], y_min),
                color,
                -1,
            )
            cv2.putText(
                vis_frame,
                label,
                (x_min, y_min - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),
                1,
            )
            
            # Draw landmarks if enabled
            if self.show_landmarks and "_landmarks_coords" in face_result:
                coords = np.array(face_result["_landmarks_coords"], dtype=np.float32)
                for point in coords:
                    x, y = int(point[0]), int(point[1])
                    cv2.circle(vis_frame, (x, y), 1, (0, 0, 255), -1)
                
                # Draw key landmarks with labels
                for name, idx in LANDMARKS.items():
                    if idx < len(coords):
                        x, y = int(coords[idx][0]), int(coords[idx][1])
                        cv2.circle(vis_frame, (x, y), 3, (255, 0, 0), -1)
        
        # Save frame
        output_path = self.visualize_dir / f"frame_{frame_idx:05d}.jpg"
        cv2.imwrite(str(output_path), vis_frame)
        
        if frame_idx == 0 or (frame_idx + 1) % 10 == 0:
            logger.info(f"Saved visualization: {output_path}")