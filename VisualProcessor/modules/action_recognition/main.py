#!/usr/bin/env python3
"""
CLI интерфейс для модуля распознавания действий в видео (SlowFast).

Использует BaseModule для автоматизации работы с метаданными и FrameManager.
"""

from __future__ import annotations

import os
import sys
import argparse
from typing import Optional, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from action_recognition_slowfast import SlowFastActionRecognizer
from utils.logger import get_logger

MODULE_NAME = "action_recognition"
logger = get_logger(MODULE_NAME)


def main(argv: Optional[List[str]] = None) -> int:
    """Главная функция CLI."""
    parser = argparse.ArgumentParser(
        prog=f"run_{MODULE_NAME}",
        description="Распознавание действий (SlowFast) — CLI",
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
        "--clip-len",
        type=int,
        default=16,
        help="Длина клипа в кадрах"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Batch size для inference"
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
        # Инициализация модуля
        recognizer = SlowFastActionRecognizer(
            rs_path=args.rs_path,
            clip_len=args.clip_len,
            batch_size=args.batch_size
        )

        # Загрузка метаданных
        metadata = recognizer.load_metadata(args.frames_dir)
        total_frames = int(metadata.get("total_frames") or metadata.get("num_frames") or 0)

        # Получение индексов кадров
        frame_indices = recognizer.get_frame_indices(metadata)

        # Создание FrameManager
        frame_manager = None
        try:
            frame_manager = recognizer.create_frame_manager(args.frames_dir, metadata)

            # Обработка
            logger.info(f"Начинаем обработку {len(frame_indices)} кадров")
            results = recognizer.process(
                frame_manager=frame_manager,
                frame_indices=frame_indices,
            )

            # Подготовка метаданных для сохранения
            save_metadata = {
                "total_frames": total_frames,
                "clip_len": args.clip_len,
                "batch_size": args.batch_size,
                "processed_tracks": len(results),
            }

            # Сохранение результатов (per-track с эмбеддингами)
            saved_path = recognizer.save_results(
                results=results,
                metadata=save_metadata,
                use_compressed=True,
                embeddings_key="embedding_normed_256d"
            )

            logger.info(
                f"Обработка завершена. Обработано треков: {len(results)}. "
                f"Результаты сохранены: {saved_path}"
            )

            return 0

        finally:
            # Гарантированное закрытие FrameManager
            if frame_manager is not None:
                try:
                    frame_manager.close()
                except Exception as e:
                    logger.exception("Ошибка при закрытии FrameManager: %s", e)

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
