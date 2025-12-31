#!/usr/bin/env python3
"""
CLI интерфейс для модуля анализа цвета и освещения видео.

Использует BaseModule для автоматизации работы с метаданными и FrameManager.
"""

from __future__ import annotations

import os
import sys
import argparse
from typing import Optional, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from modules.color_light.processor import ColorLightProcessor
from utils.logger import get_logger

MODULE_NAME = "color_light"
logger = get_logger(MODULE_NAME)


def main(argv: Optional[List[str]] = None) -> int:
    """Главная функция CLI."""
    parser = argparse.ArgumentParser(
        prog=f"run_{MODULE_NAME}",
        description="Анализ цвета и освещения видео — CLI",
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
        "--max-frames-per-scene",
        type=int,
        default=350,
        help="Максимальное количество кадров для обработки на сцену"
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=5,
        help="Шаг для выборки кадров"
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
        processor = ColorLightProcessor(
            rs_path=args.rs_path,
            max_frames_per_scene=args.max_frames_per_scene,
            stride=args.stride
        )

        # Загрузка метаданных
        metadata = processor.load_metadata(args.frames_dir)
        total_frames = int(metadata.get("total_frames") or metadata.get("num_frames") or 0)

        # Получение индексов кадров
        frame_indices = processor.get_frame_indices(metadata)

        # Создание FrameManager
        frame_manager = None
        try:
            frame_manager = processor.create_frame_manager(args.frames_dir, metadata)

            # Обработка
            logger.info(f"Начинаем обработку {len(frame_indices)} кадров")
            results = processor.process(
                frame_manager=frame_manager,
                frame_indices=frame_indices,
                config={}  # Пустой config, параметры передаются через __init__
            )

            # Подготовка метаданных для сохранения
            save_metadata = {
                "total_frames": total_frames,
                "processed_frames": len(frame_indices),
                "max_frames_per_scene": args.max_frames_per_scene,
                "stride": args.stride,
            }

            # Сохранение результатов
            saved_path = processor.save_results(
                results=results,
                metadata=save_metadata,
                use_compressed=False  # Не используем compressed формат для сложных структур
            )

            logger.info(
                f"Обработка завершена. Результаты сохранены: {saved_path}"
            )
            
            # Выводим некоторые ключевые метрики
            if isinstance(results, dict) and 'video_features' in results:
                vf = results['video_features']
                logger.info("Ключевые метрики:")
                if 'cinematic_lighting_score' in vf:
                    logger.info(f"  - Cinematic Lighting Score: {vf['cinematic_lighting_score']:.3f}")
                if 'professional_look_score' in vf:
                    logger.info(f"  - Professional Look Score: {vf['professional_look_score']:.3f}")
                if 'style_teal_orange_prob' in vf:
                    logger.info(f"  - Teal & Orange Style: {vf['style_teal_orange_prob']:.3f}")
                if 'color_distribution_entropy' in vf:
                    logger.info(f"  - Color Distribution Entropy: {vf['color_distribution_entropy']:.3f}")

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

