#!/usr/bin/env python3
"""
CLI интерфейс для модуля анализа композиции кадров.

Использует VideoCompositionAnalyzer для анализа композиции кадров видео.
"""

from __future__ import annotations

import os
import sys
import argparse
import json
from typing import Optional, List, Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from modules.frames_composition.balance_composition import Config, VideoCompositionAnalyzer
from utils.frame_manager import FrameManager
from utils.results_store import ResultsStore
from utils.logger import get_logger

MODULE_NAME = "frames_composition"
logger = get_logger(MODULE_NAME)


def _load_json(path: str) -> Dict[str, Any]:
    """Загружает JSON файл."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_pipeline(
    frames_dir: str,
    rs_path: str,
    device: Optional[str] = None,
    yolo_model_path: Optional[str] = None,
    yolo_conf_threshold: Optional[float] = None,
    max_num_faces: Optional[int] = None,
    min_detection_confidence: Optional[float] = None,
    use_midas: bool = False,
    num_depth_layers: Optional[int] = None,
    slic_n_segments: Optional[int] = None,
    slic_compactness: Optional[int] = None,
    brightness_weight: Optional[float] = None,
    object_weight: Optional[float] = None,
) -> str:
    """
    Основная логика обработки frames_composition.
    
    Возвращает путь к сохраненному файлу результатов.
    """
    if not rs_path:
        raise ValueError(f"{MODULE_NAME} | rs_path не указан")

    rs = ResultsStore(rs_path)

    meta_path = os.path.join(frames_dir, "metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"{MODULE_NAME} | metadata.json не найден в {frames_dir}")

    metadata = _load_json(meta_path)

    frame_manager = FrameManager(
        frames_dir,
        chunk_size=int(metadata.get("chunk_size", 32)),
        cache_size=int(metadata.get("cache_size", 2)),
    )

    # Получаем frame_indices из метаданных
    module_section = metadata.get(MODULE_NAME, {})
    frame_indices = module_section.get("frame_indices")
    
    if frame_indices is None:
        # Fallback: используем все кадры
        total_frames = int(metadata.get("total_frames") or metadata.get("num_frames") or 0)
        if total_frames > 0:
            frame_indices = list(range(total_frames))
            logger.info(f"{MODULE_NAME} | Используем все кадры (fallback): {total_frames}")
        else:
            raise ValueError(f"{MODULE_NAME} | Не удалось определить frame_indices")

    logger.info(f"VisualProcessor | {MODULE_NAME} | main | Инициализация VideoCompositionAnalyzer")

    # Подготавливаем конфигурацию
    config = Config(
        device=device if device else "cpu",
        rs_path=rs_path,
        yolo_model_path=yolo_model_path,
        yolo_conf_threshold=yolo_conf_threshold,
        max_num_faces=max_num_faces,
        min_detection_confidence=min_detection_confidence,
        use_midas=use_midas,
        num_depth_layers=num_depth_layers,
        slic_n_segments=slic_n_segments,
        slic_compactness=slic_compactness,
        brightness_weight=brightness_weight,
        object_weight=object_weight,
    )

    analyzer = VideoCompositionAnalyzer(config)

    try:
        # Обрабатываем кадры
        logger.info(f"VisualProcessor | {MODULE_NAME} | main | Обработка {len(frame_indices)} кадров")
        result = analyzer.analyze_video_frames(frame_manager, frame_indices=frame_indices)

        # Сохраняем результаты через ResultsStore
        rs.store(result, name=MODULE_NAME)

        logger.info(
            f"VisualProcessor | {MODULE_NAME} | main | Обработка завершена. "
            f"Результаты сохранены в {rs_path}/{MODULE_NAME}"
        )

        return os.path.join(rs_path, MODULE_NAME)

    finally:
        try:
            frame_manager.close()
        except Exception as e:  # noqa: BLE001
            logger.exception(
                f"VisualProcessor | {MODULE_NAME} | main | Ошибка при закрытии FrameManager: {e}"
            )


def main(argv: Optional[List[str]] = None) -> int:
    """Главная функция CLI."""
    parser = argparse.ArgumentParser(
        prog=f"run_{MODULE_NAME}",
        description="Анализ композиции кадров видео — CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--frames-dir",
        type=str,
        required=True,
        help="Директория с кадрами (должна содержать metadata.json)",
    )
    parser.add_argument(
        "--rs-path",
        type=str,
        required=True,
        help="Путь к директории ResultsStore для сохранения результатов",
    )
    parser.add_argument("--device", type=str, help="Устройство для обработки (cuda/cpu)")
    parser.add_argument("--yolo-model-path", type=str, help="Путь к модели YOLO")
    parser.add_argument(
        "--yolo-conf-threshold",
        type=float,
        help="Порог уверенности YOLO детекции",
    )
    parser.add_argument("--max-num-faces", type=int, help="Максимальное количество лиц")
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        help="Минимальная уверенность детекции",
    )
    parser.add_argument(
        "--use-midas",
        action="store_true",
        help="Использовать MiDaS для анализа глубины",
    )
    parser.add_argument("--num-depth-layers", type=int, help="Количество слоев глубины")
    parser.add_argument("--slic-n-segments", type=int, help="Количество сегментов SLIC")
    parser.add_argument("--slic-compactness", type=int, help="Компактность SLIC")
    parser.add_argument("--brightness-weight", type=float, help="Вес яркости")
    parser.add_argument("--object-weight", type=float, help="Вес объектов")

    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Уровень логирования (DEBUG/INFO/WARN/ERROR)",
    )

    args = parser.parse_args(argv)

    # Настройка уровня логирования
    try:
        import logging as _logging

        _logging.getLogger().setLevel(getattr(_logging, args.log_level.upper(), _logging.INFO))
    except Exception:
        logger.warning("Не удалось установить log-level: %s", args.log_level)

    try:
        saved_path = run_pipeline(
            frames_dir=args.frames_dir,
            rs_path=args.rs_path,
            device=args.device,
            yolo_model_path=args.yolo_model_path,
            yolo_conf_threshold=args.yolo_conf_threshold,
            max_num_faces=args.max_num_faces,
            min_detection_confidence=args.min_detection_confidence,
            use_midas=args.use_midas,
            num_depth_layers=args.num_depth_layers,
            slic_n_segments=args.slic_n_segments,
            slic_compactness=args.slic_compactness,
            brightness_weight=args.brightness_weight,
            object_weight=args.object_weight,
        )

        logger.info(f"Обработка завершена. Результаты сохранены: {saved_path}")
        return 0

    except FileNotFoundError as e:
        logger.error("Файл не найден: %s", e)
        return 2
    except ValueError as e:
        logger.error("Некорректные данные: %s", e)
        return 3
    except Exception as e:
        logger.exception("Fatal error в %s: %s", MODULE_NAME, e)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
