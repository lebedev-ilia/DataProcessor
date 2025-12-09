"""
Рефакторинг DetalizeFaceExtractor с использованием модульной архитектуры.

Это демонстрация того, как можно рефакторить оригинальный класс для использования модулей.
"""
from __future__ import annotations



import argparse
import logging
from typing import Any, Dict, List, Optional, Sequence
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np

from modules import MODULE_REGISTRY
from modules.base_module import FaceModule
from utils import (
    landmarks_to_ndarray,
    validate_face_landmarks,
    compute_bbox,
    extract_roi,
)
from utils.landmarks_utils import LANDMARKS

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
        max_faces: int = 4,
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

        self.max_faces = max_faces
        self.refine_landmarks = refine_landmarks
        
        # Initialize logger
        self.logger = logging.getLogger(self.__class__.__name__)
        if not self.logger.handlers:
            handler = logging.StreamHandler()
            handler.setFormatter(logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            ))
            self.logger.addHandler(handler)
            self.logger.setLevel(logging.WARNING)

        print(f"DetalizeFaceExtractorRefactored | init | face_mesh params | max_faces = {max_faces}")
        print(f"                                                          | refine_landmarks = {refine_landmarks}")
        print(f"                                                          | min_detection_confidence = {min_detection_confidence}")
        print(f"                                                          | min_tracking_confidence = {min_tracking_confidence}")

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
            print(f"DetalizeFaceExtractorRefactored | init | Visualization enabled. Saving frames to: {self.visualize_dir}")

        print(f"DetalizeFaceExtractorRefactored | init | load modules...")

        # Загружаем модули
        modules_to_load = modules or list(MODULE_REGISTRY.keys())
        self.modules: List[FaceModule] = []

        for module_name in modules_to_load:
            if module_name not in MODULE_REGISTRY:
                print(f"DetalizeFaceExtractorRefactored | init | Модуль '{module_name}' не найден в registry, пропускаем")
                continue

            module_class = MODULE_REGISTRY[module_name]
            module_config = (module_configs or {}).get(module_name, {})

            try:
                module = module_class(config=module_config)
                self.modules.append(module)
                print(f"DetalizeFaceExtractorRefactored | init | Загружен модуль: {module_name}")
            except Exception as e:
                self.logger.error(f"DetalizeFaceExtractorRefactored | init | Ошибка при загрузке модуля '{module_name}': {e}")

        if not self.modules:
            raise ValueError("DetalizeFaceExtractorRefactored | init | Не удалось загрузить ни одного модуля")

        # Инициализируем модули
        for module in self.modules:
            try:
                module.initialize()
            except Exception as e:
                self.logger.error(f"DetalizeFaceExtractorRefactored | init | Ошибка при инициализации модуля '{module.module_name}': {e}")

    def extract(self, frames: Sequence[np.ndarray]) -> List[List[Dict[str, Any]]]:
        """
        Processes a sequence of OpenCV BGR frames and returns a list where each
        element corresponds to the facial feature set per frame.
        """
        if not frames:
            return []

        outputs: List[List[Dict[str, Any]]] = []

        for frame_idx, frame in enumerate(frames):
            if frame is None:
                outputs.append([])
                continue

            frame_results = self._process_frame(frame, frame_idx)
            outputs.append(frame_results)

            # Visualize frame if enabled and faces detected
            if self.visualize and frame_results:
                self._visualize_frame(frame, frame_idx, frame_results)

        return outputs

    __call__ = extract

    def _process_frame(self, frame: np.ndarray, frame_idx: int) -> List[Dict[str, Any]]:
        """
        Обрабатывает один кадр, используя модульную архитектуру.
        """
        height, width = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return []

        frame_features: List[Dict[str, Any]] = []
        for face_idx, landmark_list in enumerate(results.multi_face_landmarks):
            # 1. Извлекаем landmarks
            coords = landmarks_to_ndarray(landmark_list, width, height)

            # 2. Вычисляем bbox
            bbox = compute_bbox(coords, width, height)

            # 3. Валидация лица
            if not validate_face_landmarks(
                bbox, coords, width, height,
                min_face_size=self.min_face_size,
                max_face_size_ratio=self.max_face_size_ratio,
                min_aspect_ratio=self.min_aspect_ratio,
                max_aspect_ratio=self.max_aspect_ratio,
                validate_landmarks=self.validate_landmarks,
            ):
                self.logger.debug(
                    f"Frame {frame_idx}: Skipping invalid face detection (face_idx={face_idx})"
                )
                continue

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
            }

            # 7. Обрабатываем через модули
            face_feature = {
                "frame_index": frame_idx,
                "face_index": face_idx,
                "bbox": bbox.tolist(),
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
                    self.logger.error(
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
            print(f"Saved visualization: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DetalizeFaceExtractorRefactored - модульная система извлечения фич лица",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/Output arguments
    parser.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Путь к входному видео файлу или директории с изображениями"
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default="face_features.json",
        help="Путь к выходному JSON файлу с результатами"
    )
    
    # Module configuration
    parser.add_argument(
        "--modules",
        type=str,
        nargs="+",
        default=None,
        help="Список модулей для загрузки (по умолчанию - все доступные)"
    )
    parser.add_argument(
        "--module-config",
        type=str,
        default=None,
        help="Путь к JSON файлу с конфигурацией модулей"
    )
    
    # Face detection parameters
    parser.add_argument(
        "--max-faces",
        type=int,
        default=4,
        help="Максимальное количество лиц для детекции на кадр"
    )
    parser.add_argument(
        "--refine-landmarks",
        action="store_true",
        default=True,
        help="Использовать уточненные landmarks (468 точек)"
    )
    parser.add_argument(
        "--no-refine-landmarks",
        action="store_false",
        dest="refine_landmarks",
        help="Не использовать уточненные landmarks"
    )
    
    # Quality filtering parameters
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.7,
        help="Минимальная уверенность детекции лица"
    )
    parser.add_argument(
        "--min-tracking-confidence",
        type=float,
        default=0.7,
        help="Минимальная уверенность трекинга лица"
    )
    parser.add_argument(
        "--min-face-size",
        type=int,
        default=30,
        help="Минимальный размер лица в пикселях"
    )
    parser.add_argument(
        "--max-face-size-ratio",
        type=float,
        default=0.8,
        help="Максимальное отношение размера лица к размеру кадра"
    )
    parser.add_argument(
        "--min-aspect-ratio",
        type=float,
        default=0.6,
        help="Минимальное соотношение сторон лица"
    )
    parser.add_argument(
        "--max-aspect-ratio",
        type=float,
        default=1.4,
        help="Максимальное соотношение сторон лица"
    )
    parser.add_argument(
        "--no-validate-landmarks",
        action="store_false",
        dest="validate_landmarks",
        default=True,
        help="Отключить валидацию landmarks"
    )
    
    # Visualization parameters
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Включить визуализацию результатов"
    )
    parser.add_argument(
        "--visualize-dir",
        type=str,
        default="./face_visualizations",
        help="Директория для сохранения визуализаций"
    )
    parser.add_argument(
        "--show-landmarks",
        action="store_true",
        help="Показывать landmarks на визуализации"
    )
    
    # Processing parameters
    parser.add_argument(
        "--frame-skip",
        type=int,
        default=1,
        help="Обрабатывать каждый N-й кадр (для ускорения)"
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Максимальное количество кадров для обработки"
    )
    
    args = parser.parse_args()
    
    # Load module configuration if provided
    module_configs = None
    if args.module_config:
        import json
        try:
            with open(args.module_config, 'r', encoding='utf-8') as f:
                module_configs = json.load(f)
            print(f"Загружена конфигурация модулей из: {args.module_config}")
        except Exception as e:
            print(f"Ошибка при загрузке конфигурации модулей: {e}")
            exit(1)
    
    # Initialize extractor
    try:
        extractor = DetalizeFaceExtractorRefactored(
            modules=args.modules,
            module_configs=module_configs,
            max_faces=args.max_faces,
            refine_landmarks=args.refine_landmarks,
            visualize=args.visualize,
            visualize_dir=args.visualize_dir,
            show_landmarks=args.show_landmarks,
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
            min_face_size=args.min_face_size,
            max_face_size_ratio=args.max_face_size_ratio,
            min_aspect_ratio=args.min_aspect_ratio,
            max_aspect_ratio=args.max_aspect_ratio,
            validate_landmarks=args.validate_landmarks,
        )
    except Exception as e:
        print(f"Ошибка при инициализации экстрактора: {e}")
        exit(1)
    
    # Load and process input
    input_path = Path(args.input)
    frames = []
    
    if input_path.is_file():
        # Process video file
        print(f"Загрузка видео: {input_path}")
        cap = cv2.VideoCapture(str(input_path))
        
        if not cap.isOpened():
            print(f"Ошибка: не удалось открыть видео файл {input_path}")
            exit(1)
        
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_count % args.frame_skip == 0:
                frames.append(frame)
                
                if args.max_frames and len(frames) >= args.max_frames:
                    break
            
            frame_count += 1
        
        cap.release()
        print(f"Загружено {len(frames)} кадров из {frame_count} (skip={args.frame_skip})")
        
    elif input_path.is_dir():
        # Process image directory
        print(f"Загрузка изображений из директории: {input_path}")
        image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(input_path.glob(f"*{ext}"))
            image_files.extend(input_path.glob(f"*{ext.upper()}"))
        
        image_files.sort()
        
        for i, img_path in enumerate(image_files):
            if i % args.frame_skip == 0:
                frame = cv2.imread(str(img_path))
                if frame is not None:
                    frames.append(frame)
                    
                    if args.max_frames and len(frames) >= args.max_frames:
                        break
        
        print(f"Загружено {len(frames)} изображений")
    else:
        print(f"Ошибка: {input_path} не является файлом или директорией")
        exit(1)
    
    if not frames:
        print("Ошибка: не удалось загрузить ни одного кадра")
        exit(1)
    
    # Process frames
    print("Начинаем обработку...")
    try:
        results = extractor.extract(frames)
        print(f"Обработка завершена. Обработано {len(results)} кадров")
        
        # Count total faces
        total_faces = sum(len(frame_result) for frame_result in results)
        print(f"Всего обнаружено лиц: {total_faces}")
        
    except Exception as e:
        print(f"Ошибка при обработке: {e}")
        exit(1)
    
    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        import json
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"Результаты сохранены в: {output_path}")
        
    except Exception as e:
        print(f"Ошибка при сохранении результатов: {e}")
        exit(1)
    
    print("Готово!")