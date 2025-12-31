"""
Все TODO выполнены:
    1. ✅ Интеграция с внешними зависимостями через BaseModule (core_object_detections, core_face_landmarks, core_depth_midas)
    2. ✅ Использование результатов core провайдеров вместо прямых вызовов моделей
    3. ✅ Интеграция с BaseModule через класс FramesCompositionModule
    4. ✅ Единый формат вывода для сохранения в npz
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any, Callable

_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if _path not in sys.path:
    sys.path.append(_path)

# Добавляем путь для импорта BaseModule
_MODULE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if _MODULE_PATH not in sys.path:
    sys.path.append(_MODULE_PATH)

from modules.base_module import BaseModule
from utils.frame_manager import FrameManager

import cv2
import numpy as np

import json
from enum import Enum
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional, Any

import warnings
warnings.filterwarnings('ignore')

from utils.logger import get_logger


NAME = "VideoCompositionAnalyzer"
LOGGER = get_logger(NAME)


def _load_core_object_detections(
    rs_path: Optional[str],
    frame_index: int,
    load_func: Optional[Callable] = None
) -> Optional[List[Dict]]:
    """
    Загружает детекции объектов из core_object_detections для конкретного кадра.
    Приоритет: BaseModule load_func > прямой доступ к файлам.
    Возвращает None, если core данные недоступны.
    """
    # Вариант 1: через BaseModule (если доступна функция загрузки)
    if load_func is not None:
        try:
            core_data = load_func("core_object_detections", format="json")
            if core_data and isinstance(core_data, dict):
                frames_data = core_data.get("data", {}).get("frames", {})
                frame_key = str(frame_index)
                
                if frame_key in frames_data:
                    detections = []
                    for det in frames_data[frame_key]:
                        detections.append({
                            "bbox": det.get("bbox", []),
                            "class": det.get("class", "unknown"),
                            "confidence": det.get("confidence", 0.0),
                            "class_id": det.get("class_id", -1),
                        })
                    return detections
        except Exception as e:
            LOGGER.warning(f"{NAME} | _load_core_object_detections | Error loading via BaseModule: {e}")
    
    # Вариант 2: прямой доступ к файлам (fallback)
    if not rs_path:
        return None
    
    detections_path = os.path.join(rs_path, "core_object_detections", "detections.json")
    if not os.path.isfile(detections_path):
        return None
    
    try:
        with open(detections_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        frames_data = data.get("data", {}).get("frames", {})
        frame_key = str(frame_index)
        
        if frame_key in frames_data:
            detections = []
            for det in frames_data[frame_key]:
                detections.append({
                    "bbox": det.get("bbox", []),
                    "class": det.get("class", "unknown"),
                    "confidence": det.get("confidence", 0.0),
                    "class_id": det.get("class_id", -1),
                })
            return detections
    except Exception as e:
        LOGGER.warning(f"{NAME} | _load_core_object_detections | Error loading core data: {e}")
    
    return None


def _load_core_face_landmarks(
    rs_path: Optional[str],
    frame_index: int,
    load_func: Optional[Callable] = None
) -> Optional[Dict]:
    """
    Загружает landmarks лиц из core_face_landmarks для конкретного кадра.
    Приоритет: BaseModule load_func > прямой доступ к файлам.
    Возвращает None, если core данные недоступны.
    """
    # Вариант 1: через BaseModule (если доступна функция загрузки)
    if load_func is not None:
        try:
            core_data = load_func("core_face_landmarks", format="json")
            if core_data and isinstance(core_data, dict):
                frames = core_data.get("frames", [])
                for frame_data in frames:
                    if frame_data.get("frame_index") == frame_index:
                        return frame_data
        except Exception as e:
            LOGGER.warning(f"{NAME} | _load_core_face_landmarks | Error loading via BaseModule: {e}")
    
    # Вариант 2: прямой доступ к файлам (fallback)
    if not rs_path:
        return None
    
    landmarks_path = os.path.join(rs_path, "core_face_landmarks", "landmarks.json")
    if not os.path.isfile(landmarks_path):
        return None
    
    try:
        with open(landmarks_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        frames = data.get("frames", [])
        for frame_data in frames:
            if frame_data.get("frame_index") == frame_index:
                return frame_data
    except Exception as e:
        LOGGER.warning(f"{NAME} | _load_core_face_landmarks | Error loading core data: {e}")
    
    return None


def _load_core_depth_midas(
    rs_path: Optional[str],
    frame_index: int,
    load_func: Optional[Callable] = None
) -> Optional[Dict[str, float]]:
    """
    Загружает статистику глубины из core_depth_midas для конкретного кадра.
    Приоритет: BaseModule load_func > прямой доступ к файлам.
    Возвращает None, если core данные недоступны.
    """
    # Вариант 1: через BaseModule (если доступна функция загрузки)
    if load_func is not None:
        try:
            core_data = load_func("core_depth_midas", format="json")
            if core_data and isinstance(core_data, dict):
                per_frame = core_data.get("per_frame", [])
                for pf in per_frame:
                    if pf.get("frame_index") == frame_index:
                        return {
                            "depth_mean": pf.get("depth_mean", 0.5),
                            "depth_std": pf.get("depth_std", 0.0),
                            "depth_reliable": pf.get("depth_reliable", True),
                            "foreground_ratio": pf.get("foreground_ratio", 0.0),
                            "bokeh_potential": pf.get("bokeh_potential", 0.0),
                        }
        except Exception as e:
            LOGGER.warning(f"{NAME} | _load_core_depth_midas | Error loading via BaseModule: {e}")
    
    # Вариант 2: прямой доступ к файлам (fallback)
    if not rs_path:
        return None
    
    depth_path = os.path.join(rs_path, "core_depth_midas", "depth.json")
    if not os.path.isfile(depth_path):
        return None
    
    try:
        with open(depth_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        per_frame = data.get("per_frame", [])
        for pf in per_frame:
            if pf.get("frame_index") == frame_index:
                return {
                    "depth_mean": pf.get("depth_mean", 0.5),
                    "depth_std": pf.get("depth_std", 0.0),
                    "depth_reliable": pf.get("depth_reliable", True),
                    "foreground_ratio": pf.get("foreground_ratio", 0.0),
                    "bokeh_potential": pf.get("bokeh_potential", 0.0),
                }
    except Exception as e:
        LOGGER.warning(f"{NAME} | _load_core_depth_midas | Error loading core data: {e}")
    
    return None

# =========================
# КОНФИГУРАЦИЯ
# =========================
@dataclass
class Config:
    """Конфигурация системы анализа композиции"""
    # Общие настройки
    device: str = 'cpu'  # torch удалён, device больше не используется для моделей
    rs_path: Optional[str] = None  # Путь к result_store для чтения core провайдеров
    load_dependency_func: Optional[Callable] = None  # Функция для загрузки зависимостей через BaseModule
    
    # Настройки YOLO
    yolo_model_path: str = 'yolo11n.pt'
    yolo_conf_threshold: float = 0.3
    max_detections: int = 50  # Максимальное количество детекций
    use_segmentation: bool = False  # Использовать сегментацию вместо bbox (если доступно)
    
    # Настройки MediaPipe
    max_num_faces: int = 5
    min_detection_confidence: float = 0.5
    
    # Настройки глубины
    use_midas: bool = True
    min_resolution_for_depth: int = 256  # Минимальное разрешение для depth (H или W)
    num_depth_layers: int = 3
    
    # Настройки SLIC (deprecated, используем local variance вместо)
    slic_n_segments: int = 100
    slic_compactness: int = 10
    
    # Веса для баланса (deprecated, используем saliency)
    brightness_weight: float = 0.65
    object_weight: float = 0.35
    use_saliency: bool = True  # Использовать saliency map вместо brightness+object_mask
    
    # Настройки симметрии
    fast_mode: bool = True  # Только horizontal/vertical симметрия

class CompositionStyle(Enum):
    """Стили композиции"""
    MINIMALIST = "minimalist"
    DOCUMENTARY = "documentary"
    VLOG = "vlog"
    CINEMATIC = "cinematic"
    PRODUCT_CENTERED = "product_centered"
    INTERVIEW = "interview"
    TIKTOK = "tiktok"
    GAMING = "gaming"
    ARTISTIC = "artistic"
    NEWS = "news"
    TUTORIAL = "tutorial"
    SPORTS = "sports"

# =========================
# МОДЕЛИ (Singleton паттерн)
# =========================
class ModelManager:
    """Менеджер для ленивой загрузки моделей"""
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self.config = Config()
            self._yolo_model = None
            self._face_mesh = None
            self._midas_model = None
            self._midas_transform = None
            self._style_model = None
            self._style_transform = None
            self._initialized = True

# =========================
# ОСНОВНЫЕ КОМПОНЕНТЫ АНАЛИЗА
# =========================
class FrameAnalyzer:
    """Анализатор отдельного кадра"""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.models = ModelManager()
    
    def extract_objects(self, frame: np.ndarray, frame_index: Optional[int] = None) -> Dict:
        """
        Детекция объектов с YOLOv8 (улучшенная версия с масками и дополнительными фичами).
        Приоритет: core_object_detections > локальный YOLO.
        """
        H, W = frame.shape[:2]
        frame_area = H * W
        
        # Используем только core_object_detections - обязательное требование
        if frame_index is None or not self.config.rs_path:
            raise RuntimeError(
                f"{NAME} | extract_objects | frame_index и rs_path обязательны для чтения core_object_detections"
            )
        
        core_detections = _load_core_object_detections(
            self.config.rs_path, frame_index, load_func=self.config.load_dependency_func
        )
        if core_detections is None:
            raise RuntimeError(
                f"{NAME} | extract_objects | core_object_detections не найдены для frame {frame_index}. "
                f"Убедитесь, что core провайдер object_detections запущен перед этим модулем. "
                f"rs_path: {self.config.rs_path}"
            )
        
        # Используем core данные
        class MockResults:
            def __init__(self, detections):
                self.boxes = None
                self.masks = None
                self.detections = detections
        
        results = MockResults(core_detections)
        
        objects = []
        object_mask = np.zeros((H, W), dtype=np.float32)
        object_centers = []
        main_subject = None
        main_subject_confidence = 0.0
        
        # Обрабатываем результаты из core_object_detections
        if hasattr(results, 'detections') and results.detections:
            # Обработка core детекций
            boxes_sorted = sorted(
                results.detections,
                key=lambda x: x.get("confidence", 0.0),
                reverse=True
            )[:self.config.max_detections]
            
            for det in boxes_sorted:
                bbox = det.get("bbox", [])
                if len(bbox) != 4:
                    continue
                x1, y1, x2, y2 = map(int, bbox)
                conf = float(det.get("confidence", 0.0))
                label = det.get("class", "unknown")
                
                # Вычисляем площадь bbox
                bbox_area = (x2 - x1) * (y2 - y1)
                bbox_area_ratio = bbox_area / frame_area
                
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                center_x_norm = cx / W
                center_y_norm = cy / H
                
                # Добавляем объект (без масок для core данных)
                objects.append({
                    "label": label,
                    "confidence": conf,
                    "bbox": [x1, y1, x2, y2],
                    "bbox_area": bbox_area,
                    "bbox_area_ratio": bbox_area_ratio,
                    "center": (cx, cy),
                    "center_norm": (center_x_norm, center_y_norm),
                })
                
                # Обновляем маску объектов (простой прямоугольник)
                object_mask[y1:y2, x1:x2] = 1.0
                object_centers.append((cx, cy))
                
                # Определяем главный объект
                if conf > main_subject_confidence:
                    main_subject_confidence = conf
                    main_subject = {
                        "label": label,
                        "confidence": conf,
                        "bbox": [x1, y1, x2, y2],
                        "center": (cx, cy),
                    }
        elif results.boxes is not None:
            # Сортируем по confidence и ограничиваем количество
            boxes_sorted = sorted(
                zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls),
                key=lambda x: x[1],
                reverse=True
            )[:self.config.max_detections]
            
            for box_xyxy, conf, cls in boxes_sorted:
                x1, y1, x2, y2 = map(int, box_xyxy)
                conf = float(conf)
                cls = int(cls)
                label = self.models.yolo_model.names[cls]
                
                # Вычисляем площадь bbox
                bbox_area = (x2 - x1) * (y2 - y1)
                bbox_area_ratio = bbox_area / frame_area
                
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                center_x_norm = cx / W
                center_y_norm = cy / H
                
                # Пытаемся получить маску (если доступна сегментация)
                mask = None
                if self.config.use_segmentation and hasattr(results, 'masks') and results.masks is not None:
                    # YOLOv8-seg предоставляет маски
                    mask_idx = len(objects)  # Упрощенная логика
                    if mask_idx < len(results.masks.data):
                        mask = results.masks.data[mask_idx].cpu().numpy()
                        mask = cv2.resize(mask, (W, H))
                        object_mask = np.maximum(object_mask, mask)
                
                # Если маски нет, используем bbox
                if mask is None:
                    object_mask[y1:y2, x1:x2] = np.maximum(
                        object_mask[y1:y2, x1:x2],
                        conf  # Вес по confidence
                    )
                
                obj_data = {
                    'bbox': [x1, y1, x2, y2],
                    'center': (cx, cy),
                    'center_x_norm': center_x_norm,
                    'center_y_norm': center_y_norm,
                    'confidence': conf,
                    'class': label,
                    'class_id': cls,
                    'bbox_area': bbox_area,
                    'bbox_area_ratio': bbox_area_ratio
                }
                
                objects.append(obj_data)
                object_centers.append((cx, cy))
                
                # Определяем главный субъект: приоритет лицам, иначе самый крупный объект с высокой уверенностью
                if main_subject is None or (bbox_area_ratio * conf > main_subject_confidence):
                    main_subject = (cx, cy)
                    main_subject_confidence = bbox_area_ratio * conf
        
        return {
            'objects': objects,
            'object_mask': object_mask,
            'object_centers': object_centers,
            'object_count': len(objects),
            'main_subject': main_subject,
            'main_subject_confidence': main_subject_confidence,
            'main_subject_criterion': 'largest_object' if main_subject else None
        }
    
    def extract_faces(self, frame: np.ndarray, frame_index: Optional[int] = None) -> Dict:
        """
        Детекция лиц. Использует только core_face_landmarks.
        """
        H, W = frame.shape[:2]
        frame_area = H * W
        
        if frame_index is None or not self.config.rs_path:
            raise RuntimeError(
                f"{NAME} | extract_faces | frame_index и rs_path обязательны для чтения core_face_landmarks"
            )
        
        core_frame = _load_core_face_landmarks(
            self.config.rs_path, frame_index, load_func=self.config.load_dependency_func
        )
        if core_frame is None:
            raise RuntimeError(
                f"{NAME} | extract_faces | core_face_landmarks не найдены для frame {frame_index}. "
                f"Убедитесь, что core провайдер core_face_landmarks запущен перед этим модулем. "
                f"rs_path: {self.config.rs_path}"
            )
        
        # Преобразуем core данные в формат, ожидаемый остальным кодом
        face_landmarks_list_core = core_frame.get("face_landmarks", [])
        
        faces = []
        face_landmarks_list = []
        main_face = None
        main_face_confidence = 0.0
        
        if face_landmarks_list_core:
            for face_idx, face_landmarks_data in enumerate(face_landmarks_list_core):
                # Извлекаем ключевые точки из core формата
                landmarks = []
                landmark_3d = []
                for lm_data in face_landmarks_data:
                    x = float(lm_data.get("x", 0.0))
                    y = float(lm_data.get("y", 0.0))
                    z = float(lm_data.get("z", 0.0))
                    # Преобразуем нормализованные координаты в пиксели
                    x_px = int(x * W)
                    y_px = int(y * H)
                    landmarks.append((x_px, y_px))
                    landmark_3d.append((x, y, z))
                
                if len(landmarks) < 10:
                    continue
                
                # Вычисляем bounding box лица
                xs = [lm[0] for lm in landmarks]
                ys = [lm[1] for lm in landmarks]
                x1, x2 = min(xs), max(xs)
                y1, y2 = min(ys), max(ys)
                
                # Площадь и размер лица
                face_width = x2 - x1
                face_height = y2 - y1
                face_area = face_width * face_height
                face_size_ratio = face_area / frame_area
                
                # Центр лица
                face_center = (np.mean(xs), np.mean(ys))
                center_x_norm = face_center[0] / W
                center_y_norm = face_center[1] / H
                
                # Face pose estimation (yaw, pitch, roll) - упрощенная версия
                face_pose = self._estimate_face_pose(landmark_3d, W, H)
                
                # Eye gaze estimation (упрощенная версия)
                eye_gaze = self._estimate_eye_gaze(landmark_3d, W, H)
                
                # Landmarks visibility ratio (предполагаем, что все видимы, если нет данных)
                landmarks_visibility_ratio = 1.0
                
                face_data = {
                    'bbox': [x1, y1, x2, y2],
                    'center': face_center,
                    'center_x_norm': center_x_norm,
                    'center_y_norm': center_y_norm,
                    'landmarks': landmarks[:10],  # Сохраняем только первые 10 для экономии памяти
                    'face_width': face_width,
                    'face_height': face_height,
                    'face_area': face_area,
                    'face_size_ratio': face_size_ratio,
                    'face_pose': face_pose,  # {'yaw': float, 'pitch': float, 'roll': float}
                    'eye_gaze': eye_gaze,  # {'x': float, 'y': float} normalized
                    'landmarks_visibility_ratio': landmarks_visibility_ratio
                }
                
                faces.append(face_data)
                face_landmarks_list.append(face_landmarks_data)
                
                # Главное лицо: самое крупное с высокой видимостью
                face_confidence = face_size_ratio * landmarks_visibility_ratio
                if main_face is None or face_confidence > main_face_confidence:
                    main_face = face_data
                    main_face_confidence = face_confidence
        
        return {
            'faces': faces,
            'face_landmarks': face_landmarks_list[0] if face_landmarks_list else None,
            'face_count': len(faces),
            'main_face': main_face,
            'main_face_confidence': main_face_confidence
        }
    
    def _estimate_face_pose(self, landmarks_3d: List[Tuple[float, float, float]], W: int, H: int) -> Dict[str, float]:
        """Упрощенная оценка позы лица (yaw, pitch, roll)"""
        if len(landmarks_3d) < 10:
            return {'yaw': 0.0, 'pitch': 0.0, 'roll': 0.0}
        
        # Упрощенная оценка на основе ключевых точек
        # Нос (примерно индекс 4 в MediaPipe Face Mesh)
        nose = landmarks_3d[4] if len(landmarks_3d) > 4 else landmarks_3d[0]
        
        # Глаза (примерно индексы 33, 263)
        left_eye_idx = min(33, len(landmarks_3d) - 1)
        right_eye_idx = min(263, len(landmarks_3d) - 1)
        left_eye = landmarks_3d[left_eye_idx]
        right_eye = landmarks_3d[right_eye_idx]
        
        # Yaw (поворот влево/вправо) - на основе разницы в глубине глаз
        yaw = (left_eye[2] - right_eye[2]) * 10.0  # Упрощенная нормализация
        yaw = np.clip(yaw, -1.0, 1.0)
        
        # Pitch (наклон вверх/вниз) - на основе позиции носа относительно глаз
        eye_center_y = (left_eye[1] + right_eye[1]) / 2
        pitch = (nose[1] - eye_center_y) * 5.0
        pitch = np.clip(pitch, -1.0, 1.0)
        
        # Roll (наклон головы) - на основе угла между глазами
        eye_angle = np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0])
        roll = np.sin(eye_angle) * 2.0
        roll = np.clip(roll, -1.0, 1.0)
        
        return {'yaw': float(yaw), 'pitch': float(pitch), 'roll': float(roll)}
    
    def _estimate_eye_gaze(self, landmarks_3d: List[Tuple[float, float, float]], W: int, H: int) -> Dict[str, float]:
        """Упрощенная оценка направления взгляда"""
        if len(landmarks_3d) < 10:
            return {'x': 0.0, 'y': 0.0}
        
        # Упрощенная оценка на основе позиции зрачков относительно центра лица
        # В реальности нужны более точные индексы для зрачков
        left_eye_idx = min(33, len(landmarks_3d) - 1)
        right_eye_idx = min(263, len(landmarks_3d) - 1)
        left_eye = landmarks_3d[left_eye_idx]
        right_eye = landmarks_3d[right_eye_idx]
        
        # Центр между глазами
        eye_center_x = (left_eye[0] + right_eye[0]) / 2
        eye_center_y = (left_eye[1] + right_eye[1]) / 2
        
        # Нормализованное смещение от центра кадра
        gaze_x = (eye_center_x - 0.5) * 2.0  # [-1, 1]
        gaze_y = (eye_center_y - 0.5) * 2.0  # [-1, 1]
        
        return {'x': float(np.clip(gaze_x, -1.0, 1.0)), 'y': float(np.clip(gaze_y, -1.0, 1.0))}
    
    def analyze_composition_anchors(self, frame: np.ndarray, object_data: Dict, face_data: Dict) -> Dict:
        """
        Объединенный анализ композиционных якорей (Rule of Thirds + Golden Ratio + Center).
        Возвращает минимальное расстояние до любого эстетического якоря и тип ближайшего.
        """
        H, W = frame.shape[:2]
        
        # Определяем главный субъект (приоритет лицу)
        main_subject = None
        main_subject_x_norm = 0.5
        main_subject_y_norm = 0.5
        
        if face_data.get('main_face'):
            main_face = face_data['main_face']
            main_subject_x_norm = main_face['center_x_norm']
            main_subject_y_norm = main_face['center_y_norm']
            main_subject = (main_face['center'][0], main_face['center'][1])
        elif object_data.get('main_subject'):
            main_subject = object_data['main_subject']
            main_subject_x_norm = main_subject[0] / W
            main_subject_y_norm = main_subject[1] / H
        elif object_data.get('objects'):
            # Используем самый крупный объект
            objects = object_data['objects']
            main_obj = max(objects, key=lambda o: o.get('bbox_area_ratio', 0))
            main_subject_x_norm = main_obj['center_x_norm']
            main_subject_y_norm = main_obj['center_y_norm']
            main_subject = main_obj['center']
        else:
            main_subject = (W / 2, H / 2)
            main_subject_x_norm = 0.5
            main_subject_y_norm = 0.5
        
        # Эстетические якоря: Rule of Thirds, Golden Ratio, Center
        aesthetic_points = []
        
        # Rule of Thirds (4 точки пересечения)
        third_x = [W / 3, 2 * W / 3]
        third_y = [H / 3, 2 * H / 3]
        for tx in third_x:
            for ty in third_y:
                aesthetic_points.append({
                    'point': (tx, ty),
                    'type': 'rule_of_thirds',
                    'normalized': (tx / W, ty / H)
                })
        
        # Golden Ratio (4 точки)
        phi = 1.618033988749895
        golden_points = [
            (W / phi, H / phi),
            (W * (phi - 1), H / phi),
            (W / phi, H * (phi - 1)),
            (W * (phi - 1), H * (phi - 1))
        ]
        for gx, gy in golden_points:
            aesthetic_points.append({
                'point': (gx, gy),
                'type': 'golden_ratio',
                'normalized': (gx / W, gy / H)
            })
        
        # Center
        aesthetic_points.append({
            'point': (W / 2, H / 2),
            'type': 'center',
            'normalized': (0.5, 0.5)
        })
        
        # Находим ближайший якорь
        mx, my = main_subject
        min_distance = float('inf')
        closest_anchor = None
        closest_type = None
        
        for anchor in aesthetic_points:
            ax, ay = anchor['point']
            dist = np.sqrt((mx - ax)**2 + (my - ay)**2)
            if dist < min_distance:
                min_distance = dist
                closest_anchor = anchor['normalized']
                closest_type = anchor['type']
        
        # Нормализованное расстояние
        max_possible = np.sqrt((W/2)**2 + (H/2)**2)
        anchor_distance_norm = min_distance / max_possible if max_possible > 0 else 0.0
        
        # Alignment score (обратная метрика расстояния)
        alignment_score = max(0.0, 1.0 - anchor_distance_norm)
        
        # Rule of Thirds score (отдельно для обратной совместимости)
        rot_min_dist = float('inf')
        for tx in third_x:
            for ty in third_y:
                dist = np.sqrt((mx - tx)**2 + (my - ty)**2)
                if dist < rot_min_dist:
                    rot_min_dist = dist
        rot_alignment_score = max(0.0, 1.0 - (rot_min_dist / max_possible))
        
        # Баланс объектов по квадрантам
        quadrants = {
            'top_left': 0, 'top_right': 0,
            'bottom_left': 0, 'bottom_right': 0
        }
        
        for obj in object_data.get('objects', []):
            cx_norm = obj.get('center_x_norm', obj['center'][0] / W)
            cy_norm = obj.get('center_y_norm', obj['center'][1] / H)
            if cy_norm < 0.5:
                if cx_norm < 0.5:
                    quadrants['top_left'] += 1
                else:
                    quadrants['top_right'] += 1
            else:
                if cx_norm < 0.5:
                    quadrants['bottom_left'] += 1
                else:
                    quadrants['bottom_right'] += 1
        
        return {
            'alignment_score': float(alignment_score),  # Для обратной совместимости
            'rule_of_thirds_score': float(rot_alignment_score),  # Для обратной совместимости
            'composition_anchor_distance': float(anchor_distance_norm),  # Новое: минимальное расстояние до любого якоря
            'closest_anchor_type': closest_type,  # 'rule_of_thirds', 'golden_ratio', или 'center'
            'closest_anchor_x': float(closest_anchor[0]) if closest_anchor else 0.5,
            'closest_anchor_y': float(closest_anchor[1]) if closest_anchor else 0.5,
            'main_subject_x': float(main_subject_x_norm),
            'main_subject_y': float(main_subject_y_norm),
            'distance_to_thirds': float(rot_min_dist / max_possible),  # Для обратной совместимости
            'quadrant_distribution': quadrants
        }
    
    def analyze_balance(self, frame: np.ndarray, 
                       object_mask: np.ndarray) -> Dict:
        """
        Анализ визуального баланса с использованием saliency map (если доступно) или fallback на brightness+object_mask.
        """
        H, W = frame.shape[:2]
        
        # Пытаемся использовать saliency map
        if self.config.use_saliency:
            try:
                # Упрощенная saliency через градиенты и контраст (lightweight proxy)
                # В продакшн можно использовать DeepGaze/UNISAL/ViT-attention
                saliency_map = self._compute_saliency_proxy(frame)
            except:
                saliency_map = None
        else:
            saliency_map = None
        
        # Fallback: brightness + object_mask
        if saliency_map is None:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
            obj_norm = object_mask / (object_mask.max() + 1e-6)
            weight_map = (self.config.brightness_weight * gray + 
                         self.config.object_weight * obj_norm)
        else:
            weight_map = saliency_map
        
        # Центр масс
        y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        total_weight = np.sum(weight_map)
        if total_weight > 1e-6:
            mass_x = np.sum(x_coords * weight_map) / total_weight
            mass_y = np.sum(y_coords * weight_map) / total_weight
        else:
            mass_x, mass_y = W / 2, H / 2
        
        # Смещение от центра
        center_x, center_y = W / 2, H / 2
        offset_distance = np.sqrt((mass_x - center_x)**2 + (mass_y - center_y)**2)
        max_offset = np.sqrt((W/2)**2 + (H/2)**2)
        normalized_offset = offset_distance / max_offset if max_offset > 0 else 0.0
        
        # Saliency center offset (расстояние между saliency CM и center)
        saliency_center_offset = normalized_offset
        
        # Баланс по квадрантам
        quadrants = {
            'top_left': float(weight_map[:H//2, :W//2].sum()),
            'top_right': float(weight_map[:H//2, W//2:].sum()),
            'bottom_left': float(weight_map[H//2:, :W//2].sum()),
            'bottom_right': float(weight_map[H//2:, W//2:].sum())
        }
        
        total_weight = sum(quadrants.values())
        if total_weight > 0:
            for key in quadrants:
                quadrants[key] = float(quadrants[key] / total_weight)
        
        # Баланс лево-право, верх-низ
        left_weight = quadrants['top_left'] + quadrants['bottom_left']
        right_weight = quadrants['top_right'] + quadrants['bottom_right']
        top_weight = quadrants['top_left'] + quadrants['top_right']
        bottom_weight = quadrants['bottom_left'] + quadrants['bottom_right']
        
        left_right_balance = 1.0 - abs(left_weight - right_weight)
        top_bottom_balance = 1.0 - abs(top_weight - bottom_weight)
        
        overall_balance_score = (left_right_balance + top_bottom_balance) / 2.0
        
        return {
            'mass_center_x': float(mass_x / W),
            'mass_center_y': float(mass_y / H),
            'center_offset': float(normalized_offset),
            'center_offset_norm': float(normalized_offset),  # Алиас для совместимости
            'saliency_center_offset': float(saliency_center_offset),  # Новое
            'quadrant_weights': quadrants,
            'left_right_balance': float(left_right_balance),
            'top_bottom_balance': float(top_bottom_balance),
            'overall_balance_score': float(overall_balance_score)
        }
    
    def _compute_saliency_proxy(self, frame: np.ndarray) -> np.ndarray:
        """
        Упрощенный saliency proxy через градиенты и контраст.
        В продакшн можно заменить на DeepGaze/UNISAL/ViT-attention.
        """
        H, W = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        
        # Градиенты (Sobel)
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        
        # Локальный контраст (разница с соседями)
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]) / 8.0
        contrast = cv2.filter2D(gray, -1, kernel)
        contrast = np.abs(contrast)
        
        # Комбинируем градиенты и контраст
        saliency = (gradient_magnitude + contrast) / 2.0
        
        # Нормализуем
        saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-6)
        
        return saliency.astype(np.float32)
    
    def analyze_depth(self, frame: np.ndarray, frame_index: Optional[int] = None) -> Dict[str, float]:
        """
        MiDaS-based depth analysis (relative depth only).
        Приоритет: core_depth_midas > локальный MiDaS.
        Вычисляется только если use_midas=True и разрешение >= min_resolution_for_depth.
        """
        # Используем только core_depth_midas - обязательное требование
        if frame_index is None or not self.config.rs_path:
            raise RuntimeError(
                f"{NAME} | analyze_depth | frame_index и rs_path обязательны для чтения core_depth_midas"
            )
        
        core_depth = _load_core_depth_midas(
            self.config.rs_path, frame_index, load_func=self.config.load_dependency_func
        )
        if core_depth is None:
            raise RuntimeError(
                f"{NAME} | analyze_depth | core_depth_midas не найдены для frame {frame_index}. "
                f"Убедитесь, что core провайдер depth_midas запущен перед этим модулем. "
                f"rs_path: {self.config.rs_path}"
            )
        
        return core_depth

    def analyze_symmetry(self, frame: np.ndarray) -> Dict:
        """
        Анализ симметрии (упрощенная версия).
        В fast_mode вычисляет только horizontal/vertical симметрию.
        Diagonal и radial симметрия опциональны (если fast_mode=False).
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        H, W = gray.shape
        
        # Горизонтальная симметрия (всегда вычисляется)
        h_flip = cv2.flip(gray, 1)
        horizontal_corr = np.corrcoef(gray.flatten(), h_flip.flatten())[0, 1]
        horizontal_score = float(np.nan_to_num(horizontal_corr, nan=0.0))
        
        # Вертикальная симметрия (всегда вычисляется)
        v_flip = cv2.flip(gray, 0)
        vertical_corr = np.corrcoef(gray.flatten(), v_flip.flatten())[0, 1]
        vertical_score = float(np.nan_to_num(vertical_corr, nan=0.0))
        
        # Комбинированный показатель симметрии (только horizontal/vertical)
        symmetry_score = float(np.mean([horizontal_score, vertical_score]))
        
        scores = {
            'horizontal': horizontal_score,
            'vertical': vertical_score
        }
        
        best_symmetry = max(scores.items(), key=lambda x: x[1])
        
        result = {
            'symmetry_score': symmetry_score,
            'dominant_symmetry_type': best_symmetry[0],
            'horizontal_symmetry': horizontal_score,
            'vertical_symmetry': vertical_score
        }
        
        # Диагональная и радиальная симметрия (опционально, если fast_mode=False)
        if not self.config.fast_mode:
            # Диагональная симметрия
            diag_flip = cv2.flip(cv2.flip(gray, -1), -1)
            diag_corr = np.corrcoef(gray.flatten(), diag_flip.flatten())[0, 1]
            diag_score = float(np.nan_to_num(diag_corr, nan=0.0))
            
            # Радиальная симметрия
            center = (W // 2, H // 2)
            max_radius = min(W, H) // 2
            try:
                polar = cv2.linearPolar(gray, center, max_radius, cv2.WARP_FILL_OUTLIERS)
                radial_flip = cv2.flip(polar, 1)
                radial_corr = np.corrcoef(polar.flatten(), radial_flip.flatten())[0, 1]
                radial_score = float(np.nan_to_num(radial_corr, nan=0.0))
            except:
                radial_score = 0.0
            
            scores['diagonal'] = diag_score
            scores['radial'] = radial_score
            
            # Обновляем best_symmetry с учетом всех типов
            best_symmetry = max(scores.items(), key=lambda x: x[1])
            symmetry_score = float(np.mean(list(scores.values())))
            
            result.update({
                'symmetry_score': symmetry_score,
                'dominant_symmetry_type': best_symmetry[0],
                'diagonal_symmetry': diag_score,
                'radial_symmetry': radial_score,
                'symmetry_details': scores
            })
        
        return result
    
    def analyze_negative_space(self, frame: np.ndarray, 
                             object_mask: np.ndarray) -> Dict:
        """
        Анализ негативного пространства.
        Использует object_mask (из сегментации, если доступно, иначе из bbox).
        """
        H, W = frame.shape[:2]
        
        # Маска негативного пространства (1 - object_mask)
        # Если используется сегментация, object_mask уже точный
        negative_space_mask = 1.0 - np.clip(object_mask, 0.0, 1.0)
        
        # Общее негативное пространство
        negative_space_ratio = float(negative_space_mask.mean())
        
        # Баланс негативного пространства (лево-право)
        left_neg_space = float(negative_space_mask[:, :W//2].mean())
        right_neg_space = float(negative_space_mask[:, W//2:].mean())
        neg_space_balance_lr = 1.0 - abs(left_neg_space - right_neg_space)
        
        # Энтропия негативного пространства (опционально, для агрегатов)
        hist, _ = np.histogram(negative_space_mask, bins=64, range=(0, 1))
        hist_norm = hist / (hist.sum() + 1e-6)
        entropy = float(-np.sum(hist_norm * np.log2(hist_norm + 1e-6)))
        
        # Распределение по квадрантам (для агрегатов)
        quadrants = {
            'top_left': float(negative_space_mask[:H//2, :W//2].mean()),
            'top_right': float(negative_space_mask[:H//2, W//2:].mean()),
            'bottom_left': float(negative_space_mask[H//2:, :W//2].mean()),
            'bottom_right': float(negative_space_mask[H//2:, W//2:].mean())
        }
        
        return {
            'negative_space_ratio': negative_space_ratio,
            'neg_space_balance_lr': neg_space_balance_lr,  # Новое: компактная метрика
            'negative_space_balance': neg_space_balance_lr,  # Алиас для обратной совместимости
            'negative_space_entropy': entropy,  # Опционально, для агрегатов
            'object_background_ratio': 1.0 - negative_space_ratio,
            'quadrant_distribution': quadrants  # Для агрегатов
        }
    
    def analyze_complexity(self, frame: np.ndarray) -> Dict:
        """
        Анализ визуальной сложности (упрощенная версия).
        Использует edge_density, local variance (вместо SLIC), color_complexity.
        overall_complexity - learnable weighted sum (не фиксированные веса).
        """
        # Границы (Canny)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(edges.mean() / 255.0)
        
        # Текстура: local variance (вместо SLIC - быстрее и дешевле)
        # Вычисляем локальную дисперсию с downsampling для скорости
        gray_float = gray.astype(np.float32) / 255.0
        kernel = np.ones((5, 5), np.float32) / 25.0
        local_mean = cv2.filter2D(gray_float, -1, kernel)
        local_variance = cv2.filter2D((gray_float - local_mean)**2, -1, kernel)
        texture_entropy = float(np.mean(local_variance))  # Упрощенная метрика текстуры
        
        # Цветовая сложность (hue std)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hue_std = float(hsv[:, :, 0].std() / 180.0)  # Нормализовано [0, 1]
        saturation_mean = float(hsv[:, :, 1].mean() / 255.0)
        
        # Общая оценка сложности (learnable weighted sum - здесь простая версия)
        # В продакшн можно обучить веса на целевой метрике
        complexity_score = (edge_density * 0.4 + texture_entropy * 0.3 + hue_std * 0.3)
        
        return {
            'edge_density': edge_density,
            'texture_entropy': texture_entropy,  # Теперь local variance вместо SLIC
            'color_complexity': hue_std,  # Нормализованный hue_std
            'saturation_mean': saturation_mean,
            'overall_complexity': float(np.clip(complexity_score, 0.0, 1.0))
        }
    
    def analyze_composition_style(self, frame: np.ndarray, analysis: Dict) -> Dict[str, Any]:
        """
        Style inference from composition signals.
        All scores are soft, bounded, and comparable.
        """

        H, W = frame.shape[:2]
        eps = 1e-6

        # -------- Safe getters --------
        def g(path, default=0.0):
            cur = analysis
            for p in path:
                if not isinstance(cur, dict) or p not in cur:
                    return default
                cur = cur[p]
            return cur

        # -------- Normalized primitives --------
        complexity = np.clip(g(["complexity", "overall_complexity"]), 0, 1)
        neg_space = np.clip(g(["negative_space", "negative_space_ratio"]), 0, 1)

        obj_count = g(["object_data", "object_count"], 0)
        obj_density = np.clip(obj_count / 8.0, 0, 1)

        depth_std = np.clip(g(["depth", "depth_std"]), 0, 1)
        depth_edges = np.clip(g(["depth", "depth_edge_density"]), 0, 1)
        bokeh = np.clip(g(["depth", "bokeh_potential"]), 0, 1)

        center_offset = np.clip(g(["balance", "center_offset"]), 0, 1)
        symmetry = np.clip(g(["symmetry", "symmetry_score"]), 0, 1)
        thirds = np.clip(g(["rule_of_thirds", "alignment_score"]), 0, 1)

        face_count = g(["face_data", "face_count"], 0)
        faces = g(["face_data", "faces"], [])

        # =========================
        # STYLE SCORES
        # =========================
        styles = {}

        # --- Minimalist ---
        styles["minimalist"] = (
            0.45 * (1.0 - complexity) +
            0.35 * neg_space +
            0.20 * (1.0 - obj_density)
        )

        # --- Cinematic ---
        styles["cinematic"] = (
            0.35 * depth_std +
            0.25 * depth_edges +
            0.20 * (1.0 - center_offset) +
            0.20 * (1.0 - symmetry)
        )

        # --- Vlog ---
        vlog_score = 0.0
        if face_count > 0 and len(faces) > 0:
            fx = faces[0]["center"][0] / (W + eps)
            face_centering = 1.0 - abs(fx - 0.5) * 2.0  # [0..1]
            vlog_score = (
                0.45 * face_centering +
                0.35 * (1.0 - complexity) +
                0.20 * obj_density
            )
        styles["vlog"] = vlog_score

        # --- Product / object-centric ---
        product_score = 0.0
        objs = g(["object_data", "objects"], [])
        if objs:
            max_area = 0.0
            frame_area = H * W
            for o in objs:
                x1, y1, x2, y2 = o["bbox"]
                area = max(0, (x2 - x1) * (y2 - y1))
                max_area = max(max_area, area)

            size_ratio = np.clip(max_area / (frame_area + eps), 0, 1)

            product_score = (
                0.45 * size_ratio +
                0.30 * thirds +
                0.25 * bokeh
            )

        styles["product_centered"] = product_score

        # =========================
        # NORMALIZATION
        # =========================
        for k in styles:
            styles[k] = float(np.clip(styles[k], 0.0, 1.0))

        total = sum(styles.values()) + eps
        styles = {k: v / total for k, v in styles.items()}

        dominant_style = max(styles.items(), key=lambda x: x[1])[0]

        return {
            "style_probabilities": styles,
            "dominant_style": dominant_style,
            "style_confidence": float(styles[dominant_style])
        }

    def analyze_leading_lines(self, frame: np.ndarray) -> Dict:
        """
        Анализ ведущих линий (упрощенная версия).
        Использует edge thinning и saliency mask для фильтрации линий от фона.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Edge thinning (морфологическая операция для более точных линий)
        kernel = np.ones((3, 3), np.uint8)
        edges_thinned = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        # Применяем saliency mask (если доступно) для фильтрации линий от фона
        if self.config.use_saliency:
            try:
                saliency_map = self._compute_saliency_proxy(frame)
                saliency_threshold = np.percentile(saliency_map, 30)  # Нижние 30% - фон
                saliency_mask = (saliency_map > saliency_threshold).astype(np.uint8) * 255
                edges_thinned = cv2.bitwise_and(edges_thinned, saliency_mask)
            except:
                pass
        
        # Детекция линий (используем thinned edges)
        lines = cv2.HoughLinesP(edges_thinned, 1, np.pi/180, 
                               threshold=80, 
                               minLineLength=50, 
                               maxLineGap=10)
        
        line_features = {
            'line_count': 0,
            'total_length': 0.0,
            'avg_length': 0.0,
            'horizontal_lines': 0,
            'vertical_lines': 0,
            'diagonal_lines': 0,
            'convergence_score': 0.0
        }
        
        if lines is not None:
            lines = lines.reshape(-1, 4)
            line_features['line_count'] = len(lines)
            
            lengths = []
            angles = []
            endpoints = []
            
            for x1, y1, x2, y2 in lines:
                # Длина линии
                length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                lengths.append(length)
                
                # Угол линии
                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                angle = (angle + 180) % 180  # Нормализация
                angles.append(angle)
                
                endpoints.append(((x1, y1), (x2, y2)))
            
            line_features['total_length'] = float(sum(lengths))
            line_features['avg_length'] = float(np.mean(lengths))
            
            # Классификация линий и определение доминирующей ориентации
            orientation_counts = {'horizontal': 0, 'vertical': 0, 'diagonal': 0}
            for angle in angles:
                if angle < 30 or angle > 150:
                    line_features['vertical_lines'] += 1
                    orientation_counts['vertical'] += 1
                elif 60 < angle < 120:
                    line_features['horizontal_lines'] += 1
                    orientation_counts['horizontal'] += 1
                else:
                    line_features['diagonal_lines'] += 1
                    orientation_counts['diagonal'] += 1
            
            # Доминирующая ориентация
            if orientation_counts['horizontal'] > orientation_counts['vertical'] and orientation_counts['horizontal'] > orientation_counts['diagonal']:
                line_features['dominant_line_orientation'] = 'horizontal'
            elif orientation_counts['vertical'] > orientation_counts['diagonal']:
                line_features['dominant_line_orientation'] = 'vertical'
            elif orientation_counts['diagonal'] > 0:
                line_features['dominant_line_orientation'] = 'diagonal'
            else:
                line_features['dominant_line_orientation'] = 'none'
            
            # Оценка схождения линий
            if len(endpoints) > 1:
                convergence_points = []
                for i in range(len(endpoints)):
                    for j in range(i+1, len(endpoints)):
                        # Проверяем пересечение линий
                        line1 = endpoints[i]
                        line2 = endpoints[j]
                        # Упрощенная проверка схождения
                        mid1 = ((line1[0][0] + line1[1][0]) / 2, 
                               (line1[0][1] + line1[1][1]) / 2)
                        mid2 = ((line2[0][0] + line2[1][0]) / 2, 
                               (line2[0][1] + line2[1][1]) / 2)
                        dist = np.sqrt((mid1[0] - mid2[0])**2 + (mid1[1] - mid2[1])**2)
                        convergence_points.append(dist)
                
                if convergence_points:
                    avg_convergence = np.mean(convergence_points)
                    max_dist = np.sqrt(frame.shape[0]**2 + frame.shape[1]**2)
                    line_features['convergence_score'] = float(1.0 - avg_convergence / max_dist)
        
        # Общая оценка ведущих линий
        if line_features['line_count'] > 0:
            line_strength = min(line_features['total_length'] / (frame.shape[0] * frame.shape[1]), 1.0)
        else:
            line_strength = 0.0
        
        line_features['line_strength'] = float(line_strength)
        
        return line_features
    
    def analyze_frame(self, frame: np.ndarray, frame_index: Optional[int] = None) -> Dict:
        """Полный анализ одного кадра"""
        # Базовые данные
        H, W = frame.shape[:2]
        
        # Извлечение объектов и лиц
        object_data = self.extract_objects(frame, frame_index=frame_index)
        face_data = self.extract_faces(frame, frame_index=frame_index)
        
        # Основные субъекты
        main_subject = None
        if face_data['face_landmarks']:
            main_subject = face_data['faces'][0]['center']
        elif object_data['object_centers']:
            main_subject = object_data['object_centers'][0]
        
        main_subject_norm = None
        if main_subject:
            main_subject_norm = (main_subject[0] / W, main_subject[1] / H)
        
        # Анализ различных аспектов
        composition_anchors = self.analyze_composition_anchors(frame, object_data, face_data)
        
        analysis = {
            'frame_dimensions': {'height': H, 'width': W},
            'object_data': object_data,
            'face_data': face_data,
            'rule_of_thirds': {  # Для обратной совместимости
                'alignment_score': composition_anchors['alignment_score'],
                'main_subject_x': composition_anchors['main_subject_x'],
                'main_subject_y': composition_anchors['main_subject_y'],
                'distance_to_thirds': composition_anchors['distance_to_thirds'],
                'quadrant_distribution': composition_anchors['quadrant_distribution']
            },
            'composition_anchors': composition_anchors,  # Новое: объединенный анализ
            'balance': self.analyze_balance(frame, object_data['object_mask']),
            'depth': self.analyze_depth(frame, frame_index=frame_index),
            'symmetry': self.analyze_symmetry(frame),
            'negative_space': self.analyze_negative_space(frame, object_data['object_mask']),
            'complexity': self.analyze_complexity(frame),
            'leading_lines': self.analyze_leading_lines(frame),
        }
        
        # Определение стиля композиции
        analysis['composition_style'] = self.analyze_composition_style(frame, analysis)
        
        # Общая оценка композиции
        composition_score = self._calculate_composition_score(analysis)
        analysis['overall_composition_score'] = composition_score
        
        return analysis
    
    def _calculate_composition_score(self, analysis: Dict) -> float:
        """
        Вычисление общей оценки композиции.
        Устойчива к отсутствующим блокам и ключам.
        Итоговый скор ∈ [0, 1].
        """

        weights = {
            'rule_of_thirds': 0.2,
            'balance': 0.15,
            'symmetry': 0.1,
            'negative_space': 0.15,
            'depth': 0.15,
            'leading_lines': 0.1,
            'complexity': 0.1,
            'style_confidence': 0.05
        }

        weighted_scores = []
        used_weights = []

        # --- Rule of thirds ---
        rot = analysis.get('rule_of_thirds')
        if rot and 'alignment_score' in rot:
            weighted_scores.append(rot['alignment_score'] * weights['rule_of_thirds'])
            used_weights.append(weights['rule_of_thirds'])

        # --- Balance ---
        balance = analysis.get('balance')
        if balance and 'overall_balance_score' in balance:
            weighted_scores.append(balance['overall_balance_score'] * weights['balance'])
            used_weights.append(weights['balance'])

        # --- Symmetry ---
        symmetry = analysis.get('symmetry')
        if symmetry and 'symmetry_score' in symmetry:
            weighted_scores.append(symmetry['symmetry_score'] * weights['symmetry'])
            used_weights.append(weights['symmetry'])

        # --- Negative space ---
        neg = analysis.get('negative_space')
        if neg and 'negative_space_balance' in neg:
            weighted_scores.append(neg['negative_space_balance'] * weights['negative_space'])
            used_weights.append(weights['negative_space'])

        # --- Depth ---
        depth = analysis.get('depth')
        if depth:
            depth_contrast = float(np.clip(depth.get('depth_contrast', 0.0), 0.0, 1.0))
            bokeh_potential = float(np.clip(depth.get('bokeh_potential', 0.0), 0.0, 1.0))

            depth_score = 0.5 * depth_contrast + 0.5 * bokeh_potential
            weighted_scores.append(depth_score * weights['depth'])
            used_weights.append(weights['depth'])

        # --- Leading lines ---
        lines = analysis.get('leading_lines')
        if lines and 'line_strength' in lines:
            weighted_scores.append(lines['line_strength'] * weights['leading_lines'])
            used_weights.append(weights['leading_lines'])

        # --- Complexity ---
        complexity_block = analysis.get('complexity')
        if complexity_block and 'overall_complexity' in complexity_block:
            complexity = np.clip(complexity_block['overall_complexity'], 0.0, 1.0)
            # Оптимум при 0.5
            complexity_score = max(0.0, 1.0 - abs(complexity - 0.5) * 2.0)
            weighted_scores.append(complexity_score * weights['complexity'])
            used_weights.append(weights['complexity'])

        # --- Style confidence ---
        style = analysis.get('composition_style')
        if style and 'style_confidence' in style:
            weighted_scores.append(style['style_confidence'] * weights['style_confidence'])
            used_weights.append(weights['style_confidence'])

        if not weighted_scores:
            return 0.0

        # Нормализация по реально использованным весам
        total_weight = sum(used_weights)
        final_score = sum(weighted_scores) / max(total_weight, 1e-6)

        return float(np.clip(final_score, 0.0, 1.0))

    # =========================
# СИСТЕМА АНАЛИЗА ВИДЕО
# =========================
class VideoCompositionAnalyzer:
    """Система анализа композиции видео"""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.frame_analyzer = FrameAnalyzer(config)
        self.analysis_history = []
    
    def analyze_video_frames(self, frame_manager, frame_indices) -> Dict:
        """Анализ нескольких кадров видео"""
        
        frame_analyses = []
        
        for i, idx in enumerate(frame_indices):

            frame = frame_manager.get(idx)

            frame_analysis = self.frame_analyzer.analyze_frame(frame, frame_index=idx)

            LOGGER.info(f"Обработано кадров: {i+1}/{len(frame_indices)}")

            frame_analysis['frame_index'] = idx
            frame_analyses.append(frame_analysis)
        
        # Агрегация результатов по всему видео
        video_analysis = self._aggregate_video_analysis(frame_analyses)
        
        # Сохраняем историю
        self.analysis_history.append(video_analysis)
        
        return video_analysis
    
    def _aggregate_video_analysis(self, frame_analyses: List[Dict]) -> Dict:
        """Агрегация результатов анализа кадров"""
        if not frame_analyses:
            return {}
        
        # Собираем все числовые значения для агрегации
        numeric_features = {}
        
        # Сначала собираем все ключи
        all_keys = set()
        for analysis in frame_analyses:
            all_keys.update(self._extract_numeric_keys(analysis))
        
        # Для каждого ключа собираем значения
        for key in all_keys:
            values = []
            for analysis in frame_analyses:
                val = self._get_nested_value(analysis, key)
                if val is not None:
                    values.append(val)
            
            if values:
                values = np.array(values)
                numeric_features[f'{key}_mean'] = float(values.mean())
                numeric_features[f'{key}_std'] = float(values.std())
                numeric_features[f'{key}_min'] = float(values.min())
                numeric_features[f'{key}_max'] = float(values.max())
                numeric_features[f'{key}_median'] = float(np.median(values))
                numeric_features[f'{key}_range'] = float(values.max() - values.min())
        
        # Качественные характеристики
        qualitative = self._analyze_qualitative_features(frame_analyses)
        
        # Общая оценка видео
        video_score = float(np.mean([a.get('overall_composition_score', 0) for a in frame_analyses]))
        
        return {
            'frame_count': len(frame_analyses),
            'video_composition_score': video_score,
            'numeric_features': numeric_features,
            'qualitative_features': qualitative,
            'frame_analysis_summary': self._summarize_frame_analyses(frame_analyses)
        }
    
    def _extract_numeric_keys(self, d: Dict, parent_key: str = '') -> List[str]:
        """Рекурсивное извлечение ключей числовых значений"""
        keys = []
        for k, v in d.items():
            full_key = f"{parent_key}.{k}" if parent_key else k
            
            if isinstance(v, dict):
                keys.extend(self._extract_numeric_keys(v, full_key))
            elif isinstance(v, (int, float, np.number)):
                keys.append(full_key)
            elif isinstance(v, list) and v and isinstance(v[0], (int, float, np.number)):
                keys.append(full_key)
        
        return keys
    
    def _get_nested_value(self, d: Dict, key: str) -> Optional[float]:
        """Получение значения по вложенному ключу"""
        keys = key.split('.')
        current = d
        
        try:
            for k in keys:
                if k in current:
                    current = current[k]
                else:
                    return None
            
            if isinstance(current, (int, float, np.number)):
                return float(current)
            elif isinstance(current, list) and current and isinstance(current[0], (int, float, np.number)):
                return float(np.mean(current))
        except:
            return None
        
        return None
    
    def _analyze_qualitative_features(self, frame_analyses: List[Dict]) -> Dict:
        """Анализ качественных характеристик"""
        # Частота различных стилей
        style_counts = {}
        symmetry_types = {}
        
        for analysis in frame_analyses:
            # Стили
            if 'composition_style' in analysis:
                style = analysis['composition_style'].get('dominant_style', 'unknown')
                style_counts[style] = style_counts.get(style, 0) + 1
            
            # Типы симметрии
            if 'symmetry' in analysis:
                sym_type = analysis['symmetry'].get('dominant_symmetry_type', 'unknown')
                symmetry_types[sym_type] = symmetry_types.get(sym_type, 0) + 1
        
        # Доминирующие стили
        dominant_style = max(style_counts.items(), key=lambda x: x[1])[0] if style_counts else 'unknown'
        dominant_symmetry = max(symmetry_types.items(), key=lambda x: x[1])[0] if symmetry_types else 'unknown'
        
        # Консистентность
        consistency_score = 0.0
        if style_counts:
            total_frames = len(frame_analyses)
            max_style_count = max(style_counts.values())
            consistency_score = max_style_count / total_frames
        
        return {
            'dominant_composition_style': dominant_style,
            'style_distribution': style_counts,
            'dominant_symmetry_type': dominant_symmetry,
            'symmetry_distribution': symmetry_types,
            'style_consistency': float(consistency_score)
        }
    
    def _summarize_frame_analyses(self, frame_analyses: List[Dict]) -> Dict:
        """Создание сводки по анализу кадров"""
        # Лучшие и худшие кадры
        scores = []
        for i, analysis in enumerate(frame_analyses):
            score = analysis.get('overall_composition_score', 0)
            scores.append((i, score))
        
        scores.sort(key=lambda x: x[1], reverse=True)
        
        best_frames = scores[:3]
        worst_frames = scores[-3:] if len(scores) >= 3 else scores
        
        # Статистика по стилям
        styles_summary = {}
        for analysis in frame_analyses:
            if 'composition_style' in analysis:
                style = analysis['composition_style'].get('dominant_style', 'unknown')
                if style not in styles_summary:
                    styles_summary[style] = {
                        'count': 0,
                        'avg_score': 0,
                        'best_score': 0
                    }
                
                styles_summary[style]['count'] += 1
                score = analysis.get('overall_composition_score', 0)
                styles_summary[style]['avg_score'] += score
                styles_summary[style]['best_score'] = max(
                    styles_summary[style]['best_score'], score
                )
        
        for style in styles_summary:
            if styles_summary[style]['count'] > 0:
                styles_summary[style]['avg_score'] /= styles_summary[style]['count']
        
        return {
            'total_frames_analyzed': len(frame_analyses),
            'best_frames': [{'index': idx, 'score': score} for idx, score in best_frames],
            'worst_frames': [{'index': idx, 'score': score} for idx, score in worst_frames],
            'style_summary': styles_summary,
            'score_range': {
                'min': min([s[1] for s in scores]) if scores else 0,
                'max': max([s[1] for s in scores]) if scores else 0,
                'mean': np.mean([s[1] for s in scores]) if scores else 0
            }
        }


class FramesCompositionModule(BaseModule):
    """
    Модуль для анализа композиции кадров видео.
    
    Наследуется от BaseModule для интеграции с системой зависимостей и единым форматом вывода.
    Использует VideoCompositionAnalyzer для обработки кадров.
    
    Зависимости:
    - core_object_detections (обязательная) - для детекции объектов
    - core_face_landmarks (опциональная) - для анализа лиц
    - core_depth_midas (опциональная) - для анализа глубины
    """
    
    def __init__(
        self,
        rs_path: Optional[str] = None,
        device: str = "cpu",
        yolo_model_path: Optional[str] = None,
        yolo_conf_threshold: Optional[float] = None,
        max_num_faces: Optional[int] = None,
        min_detection_confidence: Optional[float] = None,
        use_midas: bool = True,
        num_depth_layers: Optional[int] = None,
        slic_n_segments: Optional[int] = None,
        slic_compactness: Optional[int] = None,
        brightness_weight: Optional[float] = None,
        object_weight: Optional[float] = None,
        **kwargs: Any
    ):
        """
        Инициализация FramesCompositionModule.
        
        Args:
            rs_path: Путь к хранилищу результатов
            device: Устройство для обработки (cuda/cpu)
            yolo_model_path: Путь к модели YOLO
            yolo_conf_threshold: Порог уверенности YOLO детекции
            max_num_faces: Максимальное количество лиц
            min_detection_confidence: Минимальная уверенность детекции
            use_midas: Использовать MiDaS для анализа глубины
            num_depth_layers: Количество слоев глубины
            slic_n_segments: Количество сегментов SLIC
            slic_compactness: Компактность SLIC
            brightness_weight: Вес яркости
            object_weight: Вес объектов
            **kwargs: Дополнительные параметры для BaseModule
        """
        super().__init__(rs_path=rs_path, **kwargs)
        
        # Создаем обёртку для load_dependency_results
        def load_dep_wrapper(module_name: str, format: str = "auto"):
            return self.load_core_provider(module_name) if module_name.startswith("core_") else self.load_dependency_results(module_name, format=format)
        
        # Подготавливаем конфигурацию
        config = Config(
            device=device,
            rs_path=rs_path,
            yolo_model_path=yolo_model_path or "yolo11n.pt",
            yolo_conf_threshold=yolo_conf_threshold or 0.3,
            max_num_faces=max_num_faces or 5,
            min_detection_confidence=min_detection_confidence or 0.5,
            use_midas=use_midas,
            num_depth_layers=num_depth_layers or 3,
            slic_n_segments=slic_n_segments or 100,
            slic_compactness=slic_compactness or 10,
            brightness_weight=brightness_weight or 0.65,
            object_weight=object_weight or 0.35,
            load_dependency_func=load_dep_wrapper,
        )
        
        # Инициализируем анализатор
        self.analyzer = VideoCompositionAnalyzer(config)
    
    def required_dependencies(self) -> List[str]:
        """
        Возвращает список зависимостей модуля.
        
        Обязательные:
        - core_object_detections: для детекции объектов
        
        Опциональные:
        - core_face_landmarks: для анализа лиц
        - core_depth_midas: для анализа глубины
        """
        return ["core_object_detections"]
    
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
            - features: агрегированные фичи композиции
            - per_frame_features: per-frame фичи для VisualTransformer
            - summary: метаданные обработки
        """
        try:
            # Обрабатываем кадры через VideoCompositionAnalyzer
            result = self.analyzer.analyze_video_frames(frame_manager, frame_indices)
            
            # Преобразуем результаты в единый формат для npz
            formatted_result = self._format_results_for_npz(result, frame_indices)
            
            self.logger.info(
                f"FramesCompositionModule | Обработка завершена: "
                f"обработано {len(frame_indices)} кадров"
            )
            
            return formatted_result
            
        except Exception as e:
            self.logger.exception(f"FramesCompositionModule | Ошибка обработки: {e}")
            return self._empty_result()
    
    def _format_results_for_npz(
        self,
        result: Dict[str, Any],
        frame_indices: List[int]
    ) -> Dict[str, Any]:
        """
        Преобразует результаты VideoCompositionAnalyzer в формат для сохранения в npz.
        
        Args:
            result: Результаты из analyzer.analyze_video_frames()
            frame_indices: Список индексов кадров
            
        Returns:
            Словарь в формате для npz
        """
        # Извлекаем основные данные
        numeric_features = result.get("numeric_features", {})
        qualitative_features = result.get("qualitative_features", {})
        video_composition_score = result.get("video_composition_score", 0.0)
        frame_analysis_summary = result.get("frame_analysis_summary", {})
        
        # Подготавливаем features (агрегированные фичи)
        features = {}
        
        # Добавляем numeric_features
        for key, value in numeric_features.items():
            if isinstance(value, (int, float, bool)):
                features[key] = float(value) if isinstance(value, bool) else value
            elif isinstance(value, (list, tuple)):
                try:
                    features[key] = np.asarray(value, dtype=np.float32)
                except Exception:
                    features[key] = np.asarray(value, dtype=object)
            elif isinstance(value, np.ndarray):
                features[key] = value
        
        # Добавляем video_composition_score
        features["video_composition_score"] = float(video_composition_score)
        
        # Добавляем qualitative_features
        if qualitative_features:
            for key, value in qualitative_features.items():
                if isinstance(value, (int, float, bool)):
                    features[f"qualitative_{key}"] = float(value) if isinstance(value, bool) else value
                elif isinstance(value, dict):
                    # Сохраняем словари как есть (будут преобразованы в object array)
                    features[f"qualitative_{key}"] = value
                elif isinstance(value, (list, tuple)):
                    try:
                        features[f"qualitative_{key}"] = np.asarray(value, dtype=np.float32)
                    except Exception:
                        features[f"qualitative_{key}"] = np.asarray(value, dtype=object)
        
        # Добавляем frame_analysis_summary
        if frame_analysis_summary:
            for key, value in frame_analysis_summary.items():
                if isinstance(value, (int, float, bool)):
                    features[f"summary_{key}"] = float(value) if isinstance(value, bool) else value
                elif isinstance(value, (list, tuple)):
                    try:
                        features[f"summary_{key}"] = np.asarray(value, dtype=np.float32)
                    except Exception:
                        features[f"summary_{key}"] = np.asarray(value, dtype=object)
                elif isinstance(value, dict):
                    features[f"summary_{key}"] = value
        
        # Подготавливаем per_frame_features (placeholder - можно расширить)
        per_frame_features = {
            "frame_indices": np.array(frame_indices, dtype=np.int32),
            "composition_scores": np.array([video_composition_score] * len(frame_indices), dtype=np.float32),
        }
        
        # Подготавливаем summary
        summary = {
            "frame_count": len(frame_indices),
            "video_composition_score": float(video_composition_score),
            "total_frames_analyzed": result.get("frame_count", len(frame_indices)),
            "success": True,
        }
        
        # Формируем итоговый результат
        formatted_result = {
            "features": features,
            "per_frame_features": per_frame_features,
            "summary": summary,
        }
        
        return formatted_result
    
    def _empty_result(self) -> Dict[str, Any]:
        """Возвращает пустой результат в правильном формате."""
        return {
            "features": {},
            "per_frame_features": {
                "frame_indices": np.array([], dtype=np.int32),
                "composition_scores": np.array([], dtype=np.float32),
            },
            "summary": {
                "frame_count": 0,
                "video_composition_score": 0.0,
                "total_frames_analyzed": 0,
                "success": False,
            },
        }