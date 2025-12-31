#!/usr/bin/env python3
"""
CLI интерфейс для модуля детекции лиц в видео.

Использует FaceDetector для обнаружения лиц в кадрах видео.
"""

from __future__ import annotations

import os
import sys
import argparse
import json
from typing import Optional, List, Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from modules.face_detection.face_detector import FaceDetector
from utils.frame_manager import FrameManager
from utils.results_store import ResultsStore
from utils.logger import get_logger

MODULE_NAME = "face_detection"
logger = get_logger(MODULE_NAME)


def _load_json(path: str) -> Dict[str, Any]:
    """Загружает JSON файл."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_pipeline(
    frames_dir: str,
    rs_path: str,
    det_size: Optional[str] = None,
    threshold: Optional[float] = None,
) -> str:
    """
    Основная логика обработки face_detection.
    
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

    logger.info(f"VisualProcessor | {MODULE_NAME} | main | Инициализация FaceDetector")

    # Парсим det_size
    if det_size:
        det_size_tuple = tuple(int(x.strip()) for x in det_size.split(","))
        if len(det_size_tuple) != 2:
            raise ValueError(f"{MODULE_NAME} | det_size должен быть в формате 'width,height'")
    else:
        det_size_tuple = (640, 640)

    # Инициализируем детектор
    detector = FaceDetector(
        detect_thr=threshold if threshold is not None else 0.3,
        det_size=det_size_tuple
    )

    try:
        # Обрабатываем кадры
        logger.info(f"VisualProcessor | {MODULE_NAME} | main | Обработка {len(frame_indices)} кадров")
        timeline = detector.run(frame_manager=frame_manager, frame_indices=frame_indices)

        logger.info(f"FaceDetector | Обнаружено лиц: {len(timeline.get('frames_with_face', []))}")

        # Сохраняем результаты через ResultsStore
        rs.store(timeline, name=MODULE_NAME)

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
        description="Детекция лиц в видео — CLI",
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
    parser.add_argument(
        "--det-size",
        type=str,
        default="640,640",
        help="Размер детекции в формате 'width,height' (по умолчанию: 640,640)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Порог уверенности детекции лица (по умолчанию: 0.3)",
    )
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
            det_size=args.det_size,
            threshold=args.threshold,
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
