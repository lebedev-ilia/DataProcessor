import enum
import cv2
import numpy as np
import torch
import json
from PIL import Image
from torchvision import models, transforms
from ultralytics import YOLO
import mediapipe as mp
from skimage.segmentation import slic
from skimage.measure import shannon_entropy
from sklearn.cluster import KMeans
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import warnings
warnings.filterwarnings('ignore')

import os
import sys
_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

if _path not in sys.path:
    sys.path.append(_path)

name = "VideoCompositionAnalyzer"

from utils.logger import get_logger
logger = get_logger(name)

# =========================
# КОНФИГУРАЦИЯ
# =========================
@dataclass
class Config:
    """Конфигурация системы анализа композиции"""
    # Общие настройки
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Настройки YOLO
    yolo_model_path: str = 'yolo11n.pt'
    yolo_conf_threshold: float = 0.3
    
    # Настройки MediaPipe
    max_num_faces: int = 5
    min_detection_confidence: float = 0.5
    
    # Настройки глубины
    use_midas: bool = True
    num_depth_layers: int = 3
    
    # Настройки SLIC
    slic_n_segments: int = 100
    slic_compactness: int = 10
    
    # Веса для баланса
    brightness_weight: float = 0.65
    object_weight: float = 0.35

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
    
    @property
    def yolo_model(self):
        if self._yolo_model is None:
            logger.info("Загрузка YOLOv8...")

            if "/" not in self.config.yolo_model_path:
                self.config.yolo_model_path = f"{os.path.dirname(__file__)}/{self.config.yolo_model_path}"

            self._yolo_model = YOLO(self.config.yolo_model_path)
            self._yolo_model.to(self.config.device)
        return self._yolo_model
    
    @property
    def face_mesh(self):
        if self._face_mesh is None:
            logger.info("Загрузка MediaPipe Face Mesh...")
            mp_face = mp.solutions.face_mesh
            self._face_mesh = mp_face.FaceMesh(
                static_image_mode=True,
                max_num_faces=self.config.max_num_faces,
                min_detection_confidence=self.config.min_detection_confidence
            )
        return self._face_mesh
    
    @property
    def midas(self):
        """
        Lazy-loading MiDaS model + transform
        """
        if getattr(self, "_midas", None) is None:
            if not self.config.use_midas:
                raise RuntimeError("MiDaS disabled in config")

            import torch

            torch.hub.set_dir("./models")

            model = torch.hub.load(
                "intel-isl/MiDaS",
                "MiDaS_small",
                pretrained=True,
                trust_repo=True,
                verbose=False
            ).to(self.config.device).eval()

            transforms = torch.hub.load(
                "intel-isl/MiDaS",
                "transforms",
                trust_repo=True,
                verbose=False
            )

            self._midas = {
                "model": model,
                "transform": transforms.small_transform
            }

        return self._midas["model"], self._midas["transform"]
    
    @property
    def style_model(self):
        if self._style_model is None:
            logger.info("Загрузка ResNet50 для классификации стилей...")
            self._style_model = models.resnet50(pretrained=True).to(self.config.device)
            self._style_model.eval()
            self._style_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                )
            ])
        return self._style_model, self._style_transform

# =========================
# ОСНОВНЫЕ КОМПОНЕНТЫ АНАЛИЗА
# =========================
class FrameAnalyzer:
    """Анализатор отдельного кадра"""
    
    def __init__(self, config: Config = None):
        self.config = config or Config()
        self.models = ModelManager()
    
    def extract_objects(self, frame: np.ndarray) -> Dict:
        """Детекция объектов с YOLOv8"""
        H, W = frame.shape[:2]
        results = self.models.yolo_model(
            frame, 
            conf=self.config.yolo_conf_threshold,
            verbose=False
        )[0]
        
        objects = []
        object_mask = np.zeros((H, W), dtype=np.float32)
        object_centers = []
        
        if results.boxes is not None:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                label = self.models.yolo_model.names[cls]
                
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                
                objects.append({
                    'bbox': [x1, y1, x2, y2],
                    'center': (cx, cy),
                    'confidence': conf,
                    'class': label,
                    'class_id': cls
                })
                
                object_centers.append((cx, cy))
                object_mask[y1:y2, x1:x2] = 1.0
        
        return {
            'objects': objects,
            'object_mask': object_mask,
            'object_centers': object_centers,
            'object_count': len(objects)
        }
    
    def extract_faces(self, frame: np.ndarray) -> Dict:
        """Детекция лиц с MediaPipe"""
        H, W = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.models.face_mesh.process(rgb_frame)
        
        faces = []
        face_landmarks_list = []
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Извлекаем ключевые точки
                landmarks = []
                for landmark in face_landmarks.landmark:
                    x, y = int(landmark.x * W), int(landmark.y * H)
                    landmarks.append((x, y))
                
                # Вычисляем bounding box лица
                xs = [lm[0] for lm in landmarks]
                ys = [lm[1] for lm in landmarks]
                x1, x2 = min(xs), max(xs)
                y1, y2 = min(ys), max(ys)
                
                # Центр лица
                face_center = (np.mean(xs), np.mean(ys))
                
                faces.append({
                    'bbox': [x1, y1, x2, y2],
                    'center': face_center,
                    'landmarks': landmarks[:10]  # Сохраняем только первые 10 для экономии памяти
                })
                
                face_landmarks_list.append(face_landmarks.landmark)
        
        return {
            'faces': faces,
            'face_landmarks': face_landmarks_list[0] if face_landmarks_list else None,
            'face_count': len(faces)
        }
    
    def analyze_rule_of_thirds(self, frame: np.ndarray, 
                              object_data: Dict, 
                              face_data: Dict) -> Dict:
        """Анализ правила третей"""
        H, W = frame.shape[:2]
        
        # Сетка третей
        third_x = [W / 3, 2 * W / 3]
        third_y = [H / 3, 2 * H / 3]
        
        # Находим главный субъект (лицо или самый крупный объект)
        main_subject = None
        
        if face_data['faces']:
            # Используем первое лицо как главный субъект
            main_subject = face_data['faces'][0]['center']
        elif object_data['objects']:
            # Используем самый большой объект
            objects = object_data['objects']
            areas = [(obj['bbox'][2] - obj['bbox'][0]) * 
                    (obj['bbox'][3] - obj['bbox'][1]) 
                    for obj in objects]
            main_idx = np.argmax(areas)
            main_subject = objects[main_idx]['center']
        else:
            main_subject = (W / 2, H / 2)
        
        mx, my = main_subject
        
        # Расстояние до ближайшей точки пересечения третей
        min_dist = float('inf')
        best_point = None
        
        for tx in third_x:
            for ty in third_y:
                dist = np.sqrt((mx - tx)**2 + (my - ty)**2)
                if dist < min_dist:
                    min_dist = dist
                    best_point = (tx, ty)
        
        # Нормализованная метрика выравнивания
        max_dist = np.sqrt((W/2)**2 + (H/2)**2)
        alignment_score = max(0, 1.0 - (min_dist / max_dist))
        
        # Позиция субъекта в сетке
        grid_x = 1 if mx < third_x[0] else 2 if mx < third_x[1] else 3
        grid_y = 1 if my < third_y[0] else 2 if my < third_y[1] else 3
        grid_position = f"{grid_x}-{grid_y}"
        
        # Баланс объектов по квадрантам
        quadrants = {
            'top_left': 0, 'top_right': 0,
            'bottom_left': 0, 'bottom_right': 0
        }
        
        for obj in object_data['objects']:
            cx, cy = obj['center']
            if cy < H/2:
                if cx < W/2:
                    quadrants['top_left'] += 1
                else:
                    quadrants['top_right'] += 1
            else:
                if cx < W/2:
                    quadrants['bottom_left'] += 1
                else:
                    quadrants['bottom_right'] += 1
        
        # Вычисляем баланс
        total_objs = sum(quadrants.values())
        balance_score = 1.0
        if total_objs > 0:
            quadrant_balance = [quadrants['top_left'] + quadrants['bottom_right'],
                              quadrants['top_right'] + quadrants['bottom_left']]
            balance_score = 1.0 - abs(quadrant_balance[0] - quadrant_balance[1]) / total_objs
        
        return {
            'alignment_score': float(alignment_score),
            'main_subject_position': grid_position,
            'main_subject_x': float(mx / W),
            'main_subject_y': float(my / H),
            'balance_score': float(balance_score),
            'quadrant_distribution': quadrants,
            'distance_to_thirds': float(min_dist / max_dist)
        }
    
    def analyze_golden_ratio(self, frame: np.ndarray,
                           main_subject_pos: Tuple[float, float]) -> Dict:
        """Анализ золотого сечения"""
        H, W = frame.shape[:2]
        phi = 1.618033988749895
        
        # Точки золотого сечения
        golden_points = [
            (W / phi, H / phi),          # Верхний левый
            (W * (phi - 1), H / phi),    # Верхний правый
            (W / phi, H * (phi - 1)),    # Нижний левый
            (W * (phi - 1), H * (phi - 1))  # Нижний правый
        ]
        
        # Расстояние от субъекта до ближайшей точки золотого сечения
        mx, my = main_subject_pos
        mx, my = mx * W, my * H
        
        distances = []
        for gx, gy in golden_points:
            dist = np.sqrt((mx - gx)**2 + (my - gy)**2)
            distances.append(dist)
        
        min_dist = min(distances)
        max_possible = np.sqrt(W**2 + H**2)
        golden_score = max(0, 1.0 - (min_dist / max_possible))
        
        # Определяем ближайшую спираль (ориентацию)
        orientations = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
        closest_idx = np.argmin(distances)
        
        return {
            'golden_ratio_score': float(golden_score),
            'closest_orientation': orientations[closest_idx],
            'min_distance_normalized': float(min_dist / max_possible)
        }
    
    def analyze_balance(self, frame: np.ndarray, 
                       object_mask: np.ndarray) -> Dict:
        """Анализ визуального баланса"""
        H, W = frame.shape[:2]
        
        # Карта яркости
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        
        # Нормализованная карта объектов
        obj_norm = object_mask / (object_mask.max() + 1e-6)
        
        # Комбинированная карта значимости
        weight_map = (self.config.brightness_weight * gray + 
                     self.config.object_weight * obj_norm)
        
        # Центр масс
        y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        mass_x = np.sum(x_coords * weight_map) / (np.sum(weight_map) + 1e-6)
        mass_y = np.sum(y_coords * weight_map) / (np.sum(weight_map) + 1e-6)
        
        # Смещение от центра
        center_x, center_y = W / 2, H / 2
        offset_distance = np.sqrt((mass_x - center_x)**2 + (mass_y - center_y)**2)
        max_offset = np.sqrt((W/2)**2 + (H/2)**2)
        normalized_offset = offset_distance / max_offset
        
        # Баланс по квадрантам
        quadrants = {
            'top_left': weight_map[:H//2, :W//2].sum(),
            'top_right': weight_map[:H//2, W//2:].sum(),
            'bottom_left': weight_map[H//2:, :W//2].sum(),
            'bottom_right': weight_map[H//2:, W//2:].sum()
        }
        
        total_weight = sum(quadrants.values())
        if total_weight > 0:
            for key in quadrants:
                quadrants[key] = float(quadrants[key] / total_weight)
        
        # Баланс лево-право, верх-низ
        left_right_balance = abs(quadrants['top_left'] + quadrants['bottom_left'] - 
                               quadrants['top_right'] - quadrants['bottom_right'])
        top_bottom_balance = abs(quadrants['top_left'] + quadrants['top_right'] - 
                               quadrants['bottom_left'] - quadrants['bottom_right'])
        
        overall_balance_score = 1.0 - (left_right_balance + top_bottom_balance) / 2.0
        
        return {
            'mass_center_x': float(mass_x / W),
            'mass_center_y': float(mass_y / H),
            'center_offset': float(normalized_offset),
            'quadrant_weights': quadrants,
            'left_right_balance': float(1.0 - left_right_balance),
            'top_bottom_balance': float(1.0 - top_bottom_balance),
            'overall_balance_score': float(overall_balance_score)
        }
    
    def analyze_depth(self, frame: np.ndarray) -> Dict[str, float]:
        """
        MiDaS-based depth analysis (relative depth only)
        """

        H, W = frame.shape[:2]

        mm = ModelManager()

        model, transform = mm.midas

        # --- Preprocess ---
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = transform(rgb).to(self.config.device)

        # --- Inference ---
        with torch.no_grad():
            depth = model(input_tensor)          # [1, h, w]
            depth = depth.unsqueeze(1)           # [1, 1, h, w]
            depth = F.interpolate(
                depth,
                size=(H, W),
                mode="bicubic",
                align_corners=False
            ).squeeze()

        depth = depth.cpu().numpy()

        # --- Normalize (relative depth only) ---
        d_min, d_max = depth.min(), depth.max()
        depth = (depth - d_min) / (d_max - d_min + 1e-6)

        # =========================
        # STATISTICS
        # =========================
        mean = float(depth.mean())
        std = float(depth.std())
        p10, p50, p90 = np.percentile(depth, [10, 50, 90])

        dynamic_range = float(p90 - p10)

        # =========================
        # DEPTH LAYERS (percentiles)
        # =========================
        fg_mask = depth <= p10
        bg_mask = depth >= p90
        mg_mask = (~fg_mask) & (~bg_mask)

        fg_ratio = float(fg_mask.mean())
        mg_ratio = float(mg_mask.mean())
        bg_ratio = float(bg_mask.mean())

        # =========================
        # DEPTH EDGES / CONTRAST
        # =========================
        depth_uint8 = (depth * 255).astype(np.uint8)
        edges = cv2.Canny(depth_uint8, 50, 150)
        depth_edge_density = float(edges.mean() / 255.0)

        # =========================
        # DEPTH ENTROPY
        # =========================
        hist, _ = np.histogram(depth, bins=64, range=(0, 1))
        prob = hist / (hist.sum() + 1e-8)
        entropy = float(-np.sum(prob * np.log2(prob + 1e-8)))

        # =========================
        # BOKEH POTENTIAL
        # =========================
        bokeh_potential = float(np.clip(std * 2.0, 0.0, 1.0))

        return {
            "depth_mean": mean,
            "depth_std": std,
            "depth_p10": float(p10),
            "depth_p50": float(p50),
            "depth_p90": float(p90),
            "depth_dynamic_range": dynamic_range,
            "depth_entropy": entropy,
            "depth_edge_density": depth_edge_density,
            "foreground_ratio": fg_ratio,
            "midground_ratio": mg_ratio,
            "background_ratio": bg_ratio,
            "bokeh_potential": bokeh_potential,
            "depth_method": "midas_small"
        }

    def analyze_symmetry(self, frame: np.ndarray) -> Dict:
        """Анализ симметрии"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
        H, W = gray.shape
        
        # Горизонтальная симметрия
        h_flip = cv2.flip(gray, 1)
        horizontal_corr = np.corrcoef(gray.flatten(), h_flip.flatten())[0, 1]
        horizontal_score = float(np.nan_to_num(horizontal_corr, nan=0.0))
        
        # Вертикальная симметрия
        v_flip = cv2.flip(gray, 0)
        vertical_corr = np.corrcoef(gray.flatten(), v_flip.flatten())[0, 1]
        vertical_score = float(np.nan_to_num(vertical_corr, nan=0.0))
        
        # Диагональная симметрия
        diag_flip = cv2.flip(cv2.flip(gray, -1), -1)
        diag_corr = np.corrcoef(gray.flatten(), diag_flip.flatten())[0, 1]
        diag_score = float(np.nan_to_num(diag_corr, nan=0.0))
        
        # Радиальная симметрия
        center = (W // 2, H // 2)
        max_radius = min(W, H) // 2
        
        # Создаем полярные координаты
        polar = cv2.linearPolar(gray, center, max_radius, cv2.WARP_FILL_OUTLIERS)
        radial_flip = cv2.flip(polar, 1)
        radial_corr = np.corrcoef(polar.flatten(), radial_flip.flatten())[0, 1]
        radial_score = float(np.nan_to_num(radial_corr, nan=0.0))
        
        # Определяем тип симметрии
        scores = {
            'horizontal': horizontal_score,
            'vertical': vertical_score,
            'diagonal': diag_score,
            'radial': radial_score
        }
        
        best_symmetry = max(scores.items(), key=lambda x: x[1])
        
        # Комбинированный показатель симметрии
        symmetry_score = float(np.mean([horizontal_score, vertical_score, 
                                       diag_score, radial_score]))
        
        return {
            'symmetry_score': symmetry_score,
            'dominant_symmetry_type': best_symmetry[0],
            'horizontal_symmetry': horizontal_score,
            'vertical_symmetry': vertical_score,
            'diagonal_symmetry': diag_score,
            'radial_symmetry': radial_score,
            'symmetry_details': scores
        }
    
    def analyze_negative_space(self, frame: np.ndarray, 
                             object_mask: np.ndarray) -> Dict:
        """Анализ негативного пространства"""
        H, W = frame.shape[:2]
        
        # Маска негативного пространства
        negative_space_mask = 1.0 - object_mask
        
        # Общее негативное пространство
        negative_space_ratio = float(negative_space_mask.mean())
        
        # Распределение по квадрантам
        quadrants = {
            'top_left': negative_space_mask[:H//2, :W//2].mean(),
            'top_right': negative_space_mask[:H//2, W//2:].mean(),
            'bottom_left': negative_space_mask[H//2:, :W//2].mean(),
            'bottom_right': negative_space_mask[H//2:, W//2:].mean()
        }
        
        # Баланс негативного пространства
        left_balance = abs(quadrants['top_left'] + quadrants['bottom_left'] - 
                          quadrants['top_right'] - quadrants['bottom_right'])
        negative_space_balance = 1.0 - left_balance
        
        # Энтропия негативного пространства
        hist, _ = np.histogram(negative_space_mask, bins=256, range=(0, 1))
        hist_norm = hist / (hist.sum() + 1e-6)
        entropy = float(-np.sum(hist_norm * np.log2(hist_norm + 1e-6)))
        
        # Соотношение объект/фон
        object_background_ratio = 1.0 - negative_space_ratio
        
        return {
            'negative_space_ratio': negative_space_ratio,
            'negative_space_balance': float(negative_space_balance),
            'negative_space_entropy': entropy,
            'object_background_ratio': float(object_background_ratio),
            'quadrant_distribution': {k: float(v) for k, v in quadrants.items()}
        }
    
    def analyze_complexity(self, frame: np.ndarray) -> Dict:
        """Анализ сложности сцены"""
        # Границы
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = float(edges.mean() / 255.0)
        
        # Текстура (сегментация SLIC)
        try:
            segments = slic(frame, n_segments=self.config.slic_n_segments,
                          compactness=self.config.slic_compactness,
                          start_label=1)
            texture_entropy = float(shannon_entropy(segments))
        except:
            texture_entropy = 0.0
        
        # Цветовая сложность
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hue_std = float(hsv[:, :, 0].std())
        saturation_mean = float(hsv[:, :, 1].mean() / 255.0)
        
        # Общая оценка сложности
        complexity_score = (edge_density + texture_entropy / 10.0 + 
                          hue_std / 180.0) / 3.0
        
        return {
            'edge_density': edge_density,
            'texture_entropy': texture_entropy,
            'color_complexity': hue_std,
            'saturation_level': saturation_mean,
            'overall_complexity': float(complexity_score)
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
        """Анализ ведущих линий"""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        
        # Детекция линий
        lines = cv2.HoughLinesP(edges, 1, np.pi/180, 
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
            
            # Классификация линий
            for angle in angles:
                if angle < 30 or angle > 150:
                    line_features['vertical_lines'] += 1
                elif 60 < angle < 120:
                    line_features['horizontal_lines'] += 1
                else:
                    line_features['diagonal_lines'] += 1
            
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
    
    def analyze_frame(self, frame: np.ndarray) -> Dict:
        """Полный анализ одного кадра"""
        # Базовые данные
        H, W = frame.shape[:2]
        
        # Извлечение объектов и лиц
        object_data = self.extract_objects(frame)
        face_data = self.extract_faces(frame)
        
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
        analysis = {
            'frame_dimensions': {'height': H, 'width': W},
            'object_data': object_data,
            'face_data': face_data,
            'rule_of_thirds': self.analyze_rule_of_thirds(frame, object_data, face_data),
            'balance': self.analyze_balance(frame, object_data['object_mask']),
            'depth': self.analyze_depth(frame),
            'symmetry': self.analyze_symmetry(frame),
            'negative_space': self.analyze_negative_space(frame, object_data['object_mask']),
            'complexity': self.analyze_complexity(frame),
            'leading_lines': self.analyze_leading_lines(frame),
        }
        
        # Анализ золотого сечения (если есть главный субъект)
        if main_subject_norm:
            analysis['golden_ratio'] = self.analyze_golden_ratio(frame, main_subject_norm)
        
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

            frame_analysis = self.frame_analyzer.analyze_frame(frame)

            logger.info(f"Обработано кадров: {i+1}/{len(frame_indices)}")

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
        video_score = float(np.mean([a.get('overall_composition_score', 0) 
                                   for a in frame_analyses]))
        
        # Рекомендации
        recommendations = self._generate_recommendations(frame_analyses)
        
        return {
            'frame_count': len(frame_analyses),
            'video_composition_score': video_score,
            'numeric_features': numeric_features,
            'qualitative_features': qualitative,
            'recommendations': recommendations,
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
    
    def _generate_recommendations(self, frame_analyses: List[Dict]) -> List[str]:
        """Генерация рекомендаций по улучшению композиции"""
        recommendations = []
        
        # Анализ средних показателей
        avg_scores = {}
        for key in ['rule_of_thirds.alignment_score',
                   'balance.overall_balance_score',
                   'symmetry.symmetry_score',
                   'depth.depth_contrast']:
            values = []
            for analysis in frame_analyses:
                val = self._get_nested_value(analysis, key)
                if val is not None:
                    values.append(val)
            
            if values:
                avg_scores[key] = np.mean(values)
        
        # Генерация рекомендаций на основе анализа
        if avg_scores.get('rule_of_thirds.alignment_score', 0) < 0.5:
            recommendations.append("Улучшите выравнивание по правилу третей. Размещайте главные объекты на пересечениях линий третей.")
        
        if avg_scores.get('balance.overall_balance_score', 0) < 0.6:
            recommendations.append("Обратите внимание на баланс кадра. Распределите визуальный вес равномернее.")
        
        if avg_scores.get('depth.depth_contrast', 0) < 0.3:
            recommendations.append("Добавьте глубины в кадр. Используйте передний, средний и задний планы.")
        
        # Анализ негативного пространства
        negative_space_vals = []
        for analysis in frame_analyses:
            if 'negative_space' in analysis:
                negative_space_vals.append(analysis['negative_space']['negative_space_ratio'])
        
        if negative_space_vals:
            avg_negative_space = np.mean(negative_space_vals)
            if avg_negative_space > 0.7:
                recommendations.append("Слишком много негативного пространства. Рассмотрите возможность кадрирования или добавления объектов.")
            elif avg_negative_space < 0.2:
                recommendations.append("Мало негативного пространства. Кадр может казаться перегруженным.")
        
        return recommendations
    
    def export_analysis(self, analysis: Dict, format: str = 'json', 
                       filepath: str = None) -> Optional[str]:
        """Экспорт анализа в указанный формат"""
        try:
            # Удаляем несериализуемые объекты
            serializable_analysis = self._make_serializable(analysis)
            
            if format.lower() == 'json':
                result = json.dumps(serializable_analysis, indent=2, ensure_ascii=False)
                if filepath:
                    with open(filepath, 'w', encoding='utf-8') as f:
                        f.write(result)
                return result
                
            elif format.lower() == 'dict':
                return serializable_analysis
                
            else:
                logger.info(f"Формат {format} не поддерживается")
                return None
                
        except Exception as e:
            logger.info(f"Ошибка при экспорте: {e}")
            return None
    
    def _make_serializable(self, obj):
        """Рекурсивное преобразование объекта в сериализуемый формат"""
        if isinstance(obj, dict):
            return {k: self._make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_serializable(item) for item in obj]
        elif isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (str, int, float, bool, type(None))):
            return obj
        else:
            return str(obj)

# =========================
# ИНТЕРФЕЙС ДЛЯ РАБОТЫ С ВИДЕО
# =========================
class VideoProcessor:
    """Обработчик видео для извлечения кадров"""
    
    @staticmethod
    def extract_frames(video_path: str, max_frames: int = 100, 
                      sample_rate: int = 10) -> List[np.ndarray]:
        """Извлечение кадров из видео"""
        frames = []
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Не удалось открыть видео: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        logger.info(f"Видео: {video_path}")
        logger.info(f"Всего кадров: {total_frames}, FPS: {fps}")
        
        frame_count = 0
        extracted_count = 0
        
        while extracted_count < max_frames:
            ret, frame = cap.read()
            
            if not ret:
                break
            
            if frame_count % sample_rate == 0:
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                extracted_count += 1
            
            frame_count += 1
        
        cap.release()
        logger.info(f"Извлечено {len(frames)} кадров")
        return frames