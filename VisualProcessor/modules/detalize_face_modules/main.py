"""
CLI интерфейс для модуля detalize_face_modules (DetalizeFaceExtractorRefactored).

Приведён к единому формату, аналогичному `modules/action_recognition/main.py`:
- отдельная функция `main(argv)` для удобного вызова
- единая схема аргументов (`--frames-dir`, `--rs-path`, и т.п.)
- аккуратное логирование и работа с `FrameManager` / `ResultsStore`.
"""

from __future__ import annotations

import os
import sys
import json
import argparse
from typing import Optional, List, Dict, Any


_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if _PATH not in sys.path:
    sys.path.append(_PATH)

from modules.detalize_face_modules.detalize_face_refactored import (  # type: ignore
    DetalizeFaceExtractorRefactored,
)
from utils.frame_manager import FrameManager
from utils.results_store import ResultsStore
from utils.logger import get_logger


MODULE_NAME = "detalize_face_modules"
logger = get_logger(MODULE_NAME)


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_pipeline(
    frames_dir: str,
    rs_path: str,
    modules: Optional[List[str]] = None,
    max_faces: int = 4,
    refine_landmarks: bool = True,
    min_detection_confidence: float = 0.7,
    min_tracking_confidence: float = 0.7,
    min_face_size: int = 30,
    max_face_size_ratio: float = 0.8,
    min_aspect_ratio: float = 0.6,
    max_aspect_ratio: float = 1.4,
    validate_landmarks: bool = True,
    visualize: bool = False,
    visualize_dir: str = "./face_visualizations",
    show_landmarks: bool = False,
) -> str:
    """
    Основная логика запуска DetalizeFaceExtractorRefactored через CLI.

    Возвращает путь к сохранённому npz/json (через ResultsStore).
    """
    if not rs_path:
        raise ValueError(f"{MODULE_NAME} | rs_path не указан")

    meta_path = os.path.join(frames_dir, "metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"{MODULE_NAME} | metadata.json не найден в {frames_dir}")

    metadata = _load_json(meta_path)

    frame_manager: Optional[FrameManager] = None
    try:
        frame_manager = FrameManager(
            frames_dir=frames_dir,
            chunk_size=int(metadata.get("chunk_size", 32)),
            cache_size=int(metadata.get("cache_size", 2)),
        )

        logger.info(
            "VisualProcessor | %s | main | FrameManager создан (chunk_size=%s, cache_size=%s)",
            MODULE_NAME,
            metadata.get("chunk_size", 32),
            metadata.get("cache_size", 2),
        )

        # Инициализируем ResultsStore и extractor
        rs = ResultsStore(rs_path)

        extractor = DetalizeFaceExtractorRefactored(
            modules=modules,
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

        logger.info(
            "VisualProcessor | %s | main | Запуск extractor (modules=%s)",
            MODULE_NAME,
            ",".join(modules) if modules else "auto",
        )

        results = extractor.extract(frame_manager=frame_manager)

        saved_path = rs.store(results, name=MODULE_NAME)
        logger.info(
            "VisualProcessor | %s | main | Результаты сохранены: %s",
            MODULE_NAME,
            saved_path,
        )
        return saved_path

    finally:
        if frame_manager is not None:
            try:
                frame_manager.close()
            except Exception as e:  # noqa: BLE001
                logger.exception(
                    "VisualProcessor | %s | main | Ошибка при закрытии FrameManager: %s",
                    MODULE_NAME,
                    e,
                )


def main(argv: Optional[List[str]] = None) -> int:
    """CLI-вход для detalize_face_modules (аналогично action_recognition.main)."""

    parser = argparse.ArgumentParser(
        prog=f"run_{MODULE_NAME}",
        description="DetalizeFaceExtractorRefactored - модульная система извлечения фич лица",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Input/Output arguments
    parser.add_argument(
        "--frames-dir",
        type=str,
        required=True,
        help="Путь к директории с кадрами (должна содержать metadata.json)",
    )
    parser.add_argument(
        "--rs-path",
        type=str,
        required=True,
        help="Путь к директории ResultsStore для сохранения результатов",
    )

    # Module configuration
    parser.add_argument(
        "--modules",
        type=str,
        default=None,
        help="Список модулей через запятую (по умолчанию - все из MODULE_REGISTRY)",
    )

    # Face detection / landmarks quality parameters
    parser.add_argument(
        "--max-faces",
        type=int,
        default=4,
        help="Максимальное количество лиц для детекции на кадр (см. core_face_landmarks)",
    )
    parser.add_argument(
        "--refine-landmarks",
        action="store_true",
        default=True,
        help="Использовать уточненные landmarks (468 точек), если доступны",
    )
    parser.add_argument(
        "--min-detection-confidence",
        type=float,
        default=0.7,
        help="Минимальная уверенность детекции лица",
    )
    parser.add_argument(
        "--min-tracking-confidence",
        type=float,
        default=0.7,
        help="Минимальная уверенность трекинга лица",
    )
    parser.add_argument(
        "--min-face-size",
        type=int,
        default=30,
        help="Минимальный размер лица в пикселях",
    )
    parser.add_argument(
        "--max-face-size-ratio",
        type=float,
        default=0.8,
        help="Максимальное отношение размера лица к размеру кадра",
    )
    parser.add_argument(
        "--min-aspect-ratio",
        type=float,
        default=0.6,
        help="Минимальное соотношение сторон лица",
    )
    parser.add_argument(
        "--max-aspect-ratio",
        type=float,
        default=1.4,
        help="Максимальное соотношение сторон лица",
    )
    parser.add_argument(
        "--no-validate-landmarks",
        action="store_false",
        dest="validate_landmarks",
        default=True,
        help="Отключить валидацию landmarks",
    )

    # Visualization parameters
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Включить визуализацию результатов",
    )
    parser.add_argument(
        "--visualize-dir",
        type=str,
        default="./face_visualizations",
        help="Директория для сохранения визуализаций",
    )
    parser.add_argument(
        "--show-landmarks",
        action="store_true",
        help="Показывать landmarks на визуализации",
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

        _logging.getLogger().setLevel(
            getattr(_logging, args.log_level.upper(), _logging.INFO)
        )
    except Exception:  # noqa: BLE001
        logger.warning("Не удалось установить log-level: %s", args.log_level)

    modules_list: Optional[List[str]]
    if args.modules:
        modules_list = [m.strip() for m in args.modules.split(",") if m.strip()]
    else:
        modules_list = None

    try:
        run_pipeline(
            frames_dir=args.frames_dir,
            rs_path=args.rs_path,
            modules=modules_list,
            max_faces=args.max_faces,
            refine_landmarks=args.refine_landmarks,
            min_detection_confidence=args.min_detection_confidence,
            min_tracking_confidence=args.min_tracking_confidence,
            min_face_size=args.min_face_size,
            max_face_size_ratio=args.max_face_size_ratio,
            min_aspect_ratio=args.min_aspect_ratio,
            max_aspect_ratio=args.max_aspect_ratio,
            validate_landmarks=args.validate_landmarks,
            visualize=args.visualize,
            visualize_dir=args.visualize_dir,
            show_landmarks=args.show_landmarks,
        )
        return 0
    except FileNotFoundError as e:
        logger.error("Файл не найден: %s", e)
        return 2
    except ValueError as e:
        logger.error("Некорректные данные: %s", e)
        return 3
    except Exception as e:  # noqa: BLE001
        logger.exception("Fatal error в %s: %s", MODULE_NAME, e)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())