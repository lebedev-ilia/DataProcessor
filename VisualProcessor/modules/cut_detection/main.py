#!/usr/bin/env python3
"""
CLI интерфейс для модуля детекции переходов и анализа стиля монтажа.

Использует BaseModule для автоматизации работы с метаданными и FrameManager.
"""

from __future__ import annotations

import os
import sys
import argparse
from typing import Optional, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 0=all, 1=INFO, 2=WARNING, 3=ERROR

from modules.cut_detection.cut_detection import CutDetectionPipeline
from utils.logger import get_logger

MODULE_NAME = "cut_detection"
logger = get_logger(MODULE_NAME)


def main(argv: Optional[List[str]] = None) -> int:
    """Главная функция CLI."""
    parser = argparse.ArgumentParser(
        prog=f"run_{MODULE_NAME}",
        description="Детекция переходов и анализ стиля монтажа — CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--frames-dir",
        required=True,
        help="Директория с кадрами (FrameManager ожидает metadata.json внутри)"
    )
    parser.add_argument(
        "--rs-path",
        required=True,
        help="Папка для результирующего стора (ResultsStore и npz)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default='auto',
        choices=['auto', 'cpu', 'cuda'],
        help="Устройство для обработки (auto/cpu/cuda)"
    )
    parser.add_argument(
        "--use-clip",
        action='store_true',
        help="Использовать CLIP для классификации переходов"
    )
    parser.add_argument(
        "--use-deep-features",
        action='store_true',
        help="Использовать глубокие признаки для детекции"
    )
    parser.add_argument(
        "--audio-path",
        type=str,
        default=None,
        help="Путь к аудио файлу для аудио-анализа (опционально)"
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Уровень логирования (DEBUG/INFO/WARN/ERROR)"
    )

    args = parser.parse_args(argv)

    # Настройка уровня логирования
    try:
        import logging as _logging
        _logging.getLogger().setLevel(
            getattr(_logging, args.log_level.upper(), _logging.INFO)
        )
    except Exception:
        logger.warning("Не удалось установить log-level: %s", args.log_level)

    try:
        pipeline = CutDetectionPipeline(
            rs_path=args.rs_path,
            fps=30.0,  # actual fps is taken from frame_manager.meta in process()
            device=args.device,
            clip_zero_shot=args.use_clip,
            use_deep_features=args.use_deep_features,
            use_adaptive_thresholds=True,
            use_semantic_clustering=True,
        )

        config = {}
        if args.audio_path:
            config["audio_path"] = args.audio_path

        saved_path = pipeline.run(frames_dir=args.frames_dir, config=config)
        logger.info("Обработка завершена. Результаты сохранены: %s", saved_path)
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