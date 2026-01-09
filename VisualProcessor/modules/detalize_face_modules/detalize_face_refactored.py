"""
Все TODO выполнены:
    1. ✅ Интеграция с внешними зависимостями через BaseModule (core_face_landmarks)
    2. ✅ Использование результатов core провайдеров вместо прямых вызовов моделей
    3. ✅ Интеграция с BaseModule через класс DetalizeFaceModule
    4. ✅ Единый формат вывода для сохранения в npz
"""
from __future__ import annotations

import os
import sys
import json
from typing import Any, Dict, List, Optional, Tuple
from pathlib import Path
from collections import defaultdict, deque

import cv2 # type: ignore
import numpy as np # type: ignore

# Добавляем путь для импорта BaseModule
_MODULE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if _MODULE_PATH not in sys.path:
    sys.path.append(_MODULE_PATH)

from modules.base_module import BaseModule
from utils.frame_manager import FrameManager

from _modules import MODULE_REGISTRY
from _modules.base_module import FaceModule
from _utils import (
    validate_face_landmarks,
    compute_bbox,
    extract_roi,
)
from _utils.landmarks_utils import LANDMARKS
from utils.logger import get_logger

NAME = "DetalizeFaceExtractorRefactored"
VERSION = "1.0"
logger = get_logger(NAME)

def _load_core_face_landmarks(rs_path: Optional[str]):
    """
    Пытается загрузить предрасчитанные Mediapipe‑landmarks из core_face_landmarks.
    Формат: result_store/core_face_landmarks/landmarks.json
    
    Используется для обратной совместимости. В DetalizeFaceModule используется
    load_core_provider() из BaseModule.
    """
    if not rs_path:
        return None

    core_path = os.path.join(rs_path, "core_face_landmarks", "landmarks.json")
    if not os.path.isfile(core_path):
        return None

    try:
        with open(core_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        frames = data.get("frames") or []
        # Преобразуем в dict[frame_index] -> frame_payload
        return {int(f["frame_index"]): f for f in frames if "frame_index" in f}
    except Exception as e:
        logger.warning(f"DetalizeFaceExtractorRefactored | _load_core_face_landmarks | error: {e}")
        return None

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
        # core‑данные
        rs_path: Optional[str] = None,
    ) -> None:
        """
        :param modules: список имен модулей для загрузки (если None - загружаются все)
        :param module_configs: конфигурации для конкретных модулей
        :param max_faces: maximum number of faces to detect per frame
        :param refine_landmarks: use refined landmarks (468 points)
        :param kwargs: дополнительные параметры для BaseExtractor
        """

        self.rs_path = rs_path
        
        # Загружаем frames_with_face (будет переопределено в DetalizeFaceModule)
        self.frames_with_face = self.frames_with_face_load("auto", rs_path=rs_path)
        
        # Загружаем core_face_landmarks - обязательное требование
        self.core_landmarks = _load_core_face_landmarks(self.rs_path)
        if not self.core_landmarks:
            raise RuntimeError(
                f"DetalizeFaceExtractorRefactored | init | core_face_landmarks не найдены. "
                f"Убедитесь, что core провайдер core_face_landmarks запущен перед этим модулем. "
                f"rs_path: {self.rs_path}"
            )

        self.max_faces = max_faces
        self.refine_landmarks = refine_landmarks

        logger.info(f"DetalizeFaceExtractorRefactored | init | Используем core_face_landmarks (max_faces = {max_faces})")
        
        # Mediapipe face_mesh удалён - используем только core_face_landmarks

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

    def frames_with_face_load(self, filename, rs_path: Optional[str] = None):
        """
        Возвращает список кадров с лицами на основе `core_face_landmarks`.
        
        Args:
            filename: Имя файла или "auto" для автоматического поиска
            rs_path: Путь к хранилищу результатов (если None, использует self.rs_path)
            
        Returns:
            Список индексов кадров с лицами
        """
        rs_path = rs_path or self.rs_path
        if not rs_path:
            logger.warning("DetalizeFaceExtractorRefactored | frames_with_face_load | rs_path не указан, возвращаем пустой список")
            return []
        
        # face_detection module was removed; we rely on core_face_landmarks only.
        try:
            if not isinstance(self.core_landmarks, dict) or not self.core_landmarks:
                return []
            frames_with_face = sorted(int(k) for k in self.core_landmarks.keys())
            logger.info(
                f"DetalizeFaceExtractorRefactored | frames_with_face_load | using core_face_landmarks frames: {len(frames_with_face)}"
            )
            return frames_with_face
        except Exception as e:
            logger.error(f"DetalizeFaceExtractorRefactored | frames_with_face_load | Error: {e}")
            return []

    def extract(self, frame_manager) -> List[List[Dict[str, Any]]]:
        """
        Processes a sequence of OpenCV BGR frames and returns a list where each
        element corresponds to the facial feature set per frame.
        """
        outputs = {}

        for frame_idx in self.frames_with_face:
            frame = frame_manager.get(frame_idx)

            # Используем только core_face_landmarks
            if int(frame_idx) not in self.core_landmarks:
                logger.warning(f"DetalizeFaceExtractorRefactored | extract | Frame {frame_idx} отсутствует в core_face_landmarks, пропускаем")
                continue

            core_frame = self.core_landmarks[int(frame_idx)]
            frame_results = self._process_with_core(frame, frame_idx, core_frame)
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

    def _process_with_core(
        self,
        frame: np.ndarray,
        frame_idx: int,
        core_frame: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Обработка кадра на основе уже готовых core_face_landmarks.
        По сути повторяет _process_frame, но вместо вызова Mediapipe
        использует координаты из core‑слоя.
        """
        height, width = frame.shape[:2]

        faces_with_bbox: List[Tuple[int, Any, np.ndarray, np.ndarray, float]] = []

        # core_frame["face_landmarks"] — список лиц, каждое лицо — список точек [{x,y,z}, ...]
        for face_idx, face_points in enumerate(core_frame.get("face_landmarks", []) or []):
            if not face_points:
                continue

            # Восстанавливаем coords в пиксельных координатах, совместимых с MediaPipe‑пайплайном
            coords = np.zeros((len(face_points), 3), dtype=np.float32)
            for i, p in enumerate(face_points):
                coords[i, 0] = float(p.get("x", 0.0)) * float(width)
                coords[i, 1] = float(p.get("y", 0.0)) * float(height)
                coords[i, 2] = float(p.get("z", 0.0))

            bbox = compute_bbox(coords, width, height)

            if not validate_face_landmarks(
                bbox,
                coords,
                width,
                height,
                min_face_size=self.min_face_size,
                max_face_size_ratio=self.max_face_size_ratio,
                min_aspect_ratio=self.min_aspect_ratio,
                max_aspect_ratio=self.max_aspect_ratio,
                validate_landmarks=self.validate_landmarks,
            ):
                continue

            bbox_area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            faces_with_bbox.append((face_idx, None, coords, bbox, bbox_area))

        # Если ничего не осталось после валидации
        if not faces_with_bbox:
            return []

        # Сортировка по площади для определения primary face
        faces_with_bbox.sort(key=lambda x: x[4], reverse=True)

        frame_features: List[Dict[str, Any]] = []
        self._face_tracking[frame_idx] = {}

        for idx, (face_idx, _unused_landmarks, coords, bbox, bbox_area) in enumerate(faces_with_bbox):
            detection_confidence = 0.9  # эвристика: core‑данные считаем надёжными

            tracking_id = self._assign_tracking_id(frame_idx, face_idx, bbox, detection_confidence)
            self._face_tracking[frame_idx][face_idx] = tracking_id

            if tracking_id not in self._track_history:
                self._track_history[tracking_id] = deque(maxlen=30)

            self._track_history[tracking_id].append(
                {
                    "bbox": bbox.tolist(),
                    "frame_idx": frame_idx,
                    "detection_confidence": detection_confidence,
                }
            )

            roi = extract_roi(frame, bbox)

            bbox_x_min, bbox_y_min = bbox[0], bbox[1]
            coords_roi = coords.copy()
            coords_roi[:, 0] -= bbox_x_min
            coords_roi[:, 1] -= bbox_y_min

            shared_data = {
                "coords": coords,
                "coords_roi": coords_roi,
                "bbox": bbox,
                "roi": roi,
                "frame_shape": frame.shape,
                "face_idx": face_idx,
                "tracking_id": tracking_id,
                "detection_confidence": detection_confidence,
                "is_primary_face": (idx == 0),
            }

            face_feature: Dict[str, Any] = {
                "frame_index": frame_idx,
                "face_index": face_idx,
                "bbox": bbox.tolist(),
                "detection_confidence": detection_confidence,
                "tracking_id": tracking_id,
                "is_primary_face": (idx == 0),
            }

            for module in self.modules:
                if not module.can_process(shared_data):
                    continue

                try:
                    module_result = module.process(shared_data)
                    face_feature.update(module_result)

                    for key, value in module_result.items():
                        shared_data[key] = value
                except Exception as e:
                    logger.error(
                        f"DetalizeFaceExtractorRefactored | core | Ошибка в модуле '{module.module_name}' "
                        f"на кадре {frame_idx}, лицо {face_idx}: {e}",
                        exc_info=True,
                    )

            if self.visualize:
                face_feature["_landmarks_coords"] = coords.tolist()

            frame_features.append(face_feature)

        return frame_features

    # Метод _process_frame удалён - используем только core_face_landmarks через _process_with_core

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


def _load_core_face_landmarks_from_data(data: Dict[str, Any]) -> Optional[Dict[int, Dict[str, Any]]]:
    """
    Загружает core_face_landmarks из данных, загруженных через BaseModule.
    
    Args:
        data: Данные из load_core_provider("core_face_landmarks")
        
    Returns:
        dict[frame_index] -> frame_payload или None
    """
    if not data:
        return None
    
    try:
        # Вариант 1: данные в формате JSON (если загружены через _load_json)
        if "frames" in data:
            frames = data.get("frames", [])
            if isinstance(frames, list):
                return {int(f["frame_index"]): f for f in frames if "frame_index" in f}
        
        # Вариант 2: данные в формате npz (если загружены через _load_npz)
        # Проверяем, есть ли ключ с landmarks
        for key in ["landmarks", "frames", "face_landmarks"]:
            if key in data:
                landmarks_data = data[key]
                if isinstance(landmarks_data, np.ndarray) and landmarks_data.dtype == object:
                    # object array - возможно, это список словарей
                    try:
                        frames = landmarks_data.item() if landmarks_data.size == 1 else landmarks_data.tolist()
                        if isinstance(frames, list):
                            return {int(f["frame_index"]): f for f in frames if "frame_index" in f}
                    except Exception:
                        pass
        
        # Вариант 3: прямой доступ к структуре данных
        if isinstance(data, dict):
            # Ищем вложенные структуры
            for value in data.values():
                if isinstance(value, (list, tuple)):
                    try:
                        result = {int(f["frame_index"]): f for f in value if isinstance(f, dict) and "frame_index" in f}
                        if result:
                            return result
                    except Exception:
                        continue
        
        return None
    except Exception as e:
        logger.warning(f"DetalizeFaceModule | _load_core_face_landmarks_from_data | error: {e}")
        return None


class DetalizeFaceModule(BaseModule):
    """
    Модуль для детального извлечения фичей лица.
    
    Наследуется от BaseModule для интеграции с системой зависимостей и единым форматом вывода.
    Использует DetalizeFaceExtractorRefactored для обработки кадров.
    
    Зависимости:
    - core_face_landmarks (обязательная) - landmarks лиц (и face presence)
    """
    
    def __init__(
        self,
        rs_path: Optional[str] = None,
        modules: Optional[List[str]] = None,
        module_configs: Optional[Dict[str, Dict[str, Any]]] = None,
        max_faces: int = 10,
        refine_landmarks: bool = True,
        visualize: bool = False,
        visualize_dir: Optional[str] = None,
        show_landmarks: bool = False,
        min_detection_confidence: float = 0.7,
        min_tracking_confidence: float = 0.7,
        min_face_size: int = 30,
        max_face_size_ratio: float = 0.8,
        min_aspect_ratio: float = 0.6,
        max_aspect_ratio: float = 1.4,
        validate_landmarks: bool = True,
        use_face_detection: bool = True,
        **kwargs: Any
    ):
        """
        Инициализация DetalizeFaceModule.
        
        Args:
            rs_path: Путь к хранилищу результатов
            modules: Список имен модулей для загрузки (если None - загружаются все)
            module_configs: Конфигурации для конкретных модулей
            max_faces: Максимальное количество лиц на кадр
            refine_landmarks: Использовать уточненные landmarks (468 точек)
            visualize: Включить визуализацию
            visualize_dir: Директория для визуализаций
            show_landmarks: Показывать landmarks на визуализации
            min_detection_confidence: Минимальная уверенность детекции
            min_tracking_confidence: Минимальная уверенность трекинга
            min_face_size: Минимальный размер лица в пикселях
            max_face_size_ratio: Максимальное отношение размера лица к размеру кадра
            min_aspect_ratio: Минимальное соотношение сторон лица
            max_aspect_ratio: Максимальное соотношение сторон лица
            validate_landmarks: Валидировать landmarks
            use_face_detection: Устаревший флаг. Теперь фильтрация делается по core_face_landmarks, face_detection удалён.
            **kwargs: Дополнительные параметры для BaseModule
        """
        super().__init__(rs_path=rs_path, **kwargs)
        
        # Backward-compat flag: keep it, but it no longer loads face_detection.
        self.use_face_detection = bool(use_face_detection)
        
        # Инициализируем extractor
        self.extractor = DetalizeFaceExtractorRefactored(
            modules=modules,
            module_configs=module_configs,
            max_faces=max_faces,
            refine_landmarks=refine_landmarks,
            visualize=visualize,
            visualize_dir=visualize_dir,
            show_landmarks=show_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
            min_face_size=min_face_size,
            max_face_size_ratio=max_face_size_ratio,
            min_aspect_ratio=min_aspect_ratio,
            max_aspect_ratio=max_aspect_ratio,
            validate_landmarks=validate_landmarks,
            rs_path=rs_path,
        )
        
        # Загружаем core_face_landmarks через BaseModule
        self._load_core_landmarks()
        
        # Always derive frames_with_face from core_face_landmarks (face_detection удалён).
        if self.extractor.core_landmarks:
            self.extractor.frames_with_face = sorted(self.extractor.core_landmarks.keys())
        else:
            self.extractor.frames_with_face = []
    
    def required_dependencies(self) -> List[str]:
        """
        Возвращает список зависимостей модуля.
        
        Обязательные:
        - core_face_landmarks: landmarks лиц
        
        Опциональные:
        - (none) — face_detection удалён
        """
        return ["core_face_landmarks"]
    
    def _load_core_landmarks(self) -> None:
        """Загружает core_face_landmarks через BaseModule."""
        try:
            landmarks_data = self.load_core_provider("core_face_landmarks", file_name="landmarks.json")
            if landmarks_data:
                # Пытаемся преобразовать в нужный формат
                core_landmarks = _load_core_face_landmarks_from_data(landmarks_data)
                if core_landmarks:
                    self.extractor.core_landmarks = core_landmarks
                    self.logger.info(
                        f"DetalizeFaceModule | Загружены core_face_landmarks "
                        f"({len(core_landmarks)} кадров)"
                    )
                    return
            
            # Fallback: используем старый метод
            core_landmarks = _load_core_face_landmarks(self.rs_path)
            if core_landmarks:
                self.extractor.core_landmarks = core_landmarks
                self.logger.info(
                    f"DetalizeFaceModule | Загружены core_face_landmarks (fallback, "
                    f"{len(core_landmarks)} кадров)"
                )
            else:
                raise RuntimeError(
                    "DetalizeFaceModule | core_face_landmarks не найдены. "
                    "Убедитесь, что core провайдер core_face_landmarks запущен."
                )
        except Exception as e:
            self.logger.exception(f"DetalizeFaceModule | Ошибка загрузки core_face_landmarks: {e}")
            raise
    
    def _load_frames_with_face(self) -> None:
        """
        Deprecated: face_detection removed. Kept for compatibility; derives frames from core_face_landmarks.
        """
        if self.extractor.core_landmarks:
            self.extractor.frames_with_face = sorted(self.extractor.core_landmarks.keys())
        else:
            self.extractor.frames_with_face = []
    
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
            config: Конфигурация модуля (не используется, параметры заданы в __init__)
            
        Returns:
            Словарь с результатами в формате для сохранения в npz:
            - frames: dict[frame_index] -> list[face_features]
            - summary: метаданные обработки
        """
        # Фильтруем frame_indices по frames_with_face, если нужно
        if self.use_face_detection and self.extractor.frames_with_face:
            filtered_indices = sorted(set(frame_indices) & set(self.extractor.frames_with_face))
            if not filtered_indices:
                self.logger.warning(
                    f"DetalizeFaceModule | Нет пересечения между frame_indices "
                    f"({len(frame_indices)}) и frames_with_face ({len(self.extractor.frames_with_face)})"
                )
                return self._empty_result(len(frame_indices))
        else:
            filtered_indices = frame_indices
        
        # Обновляем frames_with_face в extractor для обработки
        original_frames_with_face = self.extractor.frames_with_face
        self.extractor.frames_with_face = filtered_indices
        
        try:
            # Обрабатываем через extractor
            outputs = self.extractor.extract(frame_manager=frame_manager)
            
            # Преобразуем результаты в единый формат для npz
            result = self._format_results_for_npz(outputs, len(frame_indices), len(filtered_indices))
            
            self.logger.info(
                f"DetalizeFaceModule | Обработка завершена: "
                f"обработано {len(filtered_indices)} кадров, "
                f"найдено лиц: {sum(len(faces) for faces in outputs.values())}"
            )
            
            return result
            
        finally:
            # Восстанавливаем оригинальный список
            self.extractor.frames_with_face = original_frames_with_face
    
    def _format_results_for_npz(
        self,
        outputs: Dict[int, List[Dict[str, Any]]],
        total_frames: int,
        processed_frames: int
    ) -> Dict[str, Any]:
        """
        Преобразует результаты extractor в формат для сохранения в npz.
        
        Args:
            outputs: Результаты из extractor.extract()
            total_frames: Общее количество кадров
            processed_frames: Количество обработанных кадров
            
        Returns:
            Словарь в формате для npz
        """
        # Подготавливаем данные для сохранения
        frames_data = {}
        all_face_features = []
        
        for frame_idx, face_features in outputs.items():
            # Преобразуем каждое лицо в numpy-совместимый формат
            frame_faces = []
            for face_feature in face_features:
                face_clean = {}
                
                for key, value in face_feature.items():
                    # Пропускаем служебные ключи
                    if key.startswith("_"):
                        continue
                    
                    # Преобразуем значения в numpy-совместимые типы
                    if isinstance(value, (int, float, bool)):
                        face_clean[key] = float(value) if isinstance(value, bool) else value
                    elif isinstance(value, (list, tuple)):
                        try:
                            # Пытаемся преобразовать в numpy массив
                            arr = np.asarray(value, dtype=np.float32)
                            face_clean[key] = arr
                        except Exception:
                            # Если не получается - сохраняем как object array
                            face_clean[key] = np.asarray(value, dtype=object)
                    elif isinstance(value, np.ndarray):
                        face_clean[key] = value
                    elif isinstance(value, dict):
                        # Вложенные словари сохраняем как object
                        face_clean[key] = np.asarray(value, dtype=object)
                    else:
                        # Остальное - как object
                        face_clean[key] = np.asarray(value, dtype=object)
                
                frame_faces.append(face_clean)
                all_face_features.append(face_clean)
            
            # Сохраняем как object array для каждого кадра
            if frame_faces:
                frames_data[int(frame_idx)] = np.asarray(frame_faces, dtype=object)
        
        # Формируем summary
        total_faces = sum(len(faces) for faces in outputs.values())
        primary_faces = sum(
            1 for faces in outputs.values()
            for face in faces
            if face.get("is_primary_face", False)
        )
        
        summary = {
            "total_frames": total_frames,
            "processed_frames": processed_frames,
            "frames_with_faces": len(outputs),
            "total_faces": total_faces,
            "primary_faces": primary_faces,
            "avg_faces_per_frame": float(total_faces / len(outputs)) if outputs else 0.0,
        }
        
        # Формируем итоговый результат
        result = {
            "frames": np.asarray(frames_data, dtype=object) if frames_data else np.array([], dtype=object),
            "summary": summary,
        }
        
        return result
    
    def _empty_result(self, total_frames: int) -> Dict[str, Any]:
        """Возвращает пустой результат в правильном формате."""
        return {
            "frames": np.array([], dtype=object),
            "summary": {
                "total_frames": total_frames,
                "processed_frames": 0,
                "frames_with_faces": 0,
                "total_faces": 0,
                "primary_faces": 0,
                "avg_faces_per_frame": 0.0,
            },
        }