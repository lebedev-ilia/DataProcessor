#!/usr/bin/env python3
"""
CLI интерфейс для модуля анализа эмоций на лицах в видео.

Использует VideoEmotionProcessor для обработки видео и извлечения эмоций.
"""

from __future__ import annotations

import os
import sys
import argparse
import json
from typing import Optional, List, Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from modules.emotion_face.core.video_processor import VideoEmotionProcessor
from utils.frame_manager import FrameManager
from utils.results_store import ResultsStore
from utils.logger import get_logger

MODULE_NAME = "emotion_face"
logger = get_logger(MODULE_NAME)


def _load_json(path: str) -> Dict[str, Any]:
    """Загружает JSON файл."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def run_pipeline(
    frames_dir: str,
    rs_path: str,
    min_frames_ratio: Optional[float] = None,
    min_keyframes: Optional[int] = None,
    min_transitions: Optional[int] = None,
    min_diversity_threshold: Optional[float] = None,
    quality_threshold: Optional[float] = None,
    memory_threshold_low: Optional[int] = None,
    batch_load_low: Optional[int] = None,
    batch_process_low: Optional[int] = None,
    memory_threshold_medium: Optional[int] = None,
    batch_load_medium: Optional[int] = None,
    batch_process_medium: Optional[int] = None,
    memory_threshold_high: Optional[int] = None,
    batch_load_high: Optional[int] = None,
    batch_process_high: Optional[int] = None,
    batch_load_very_high: Optional[int] = None,
    batch_process_very_high: Optional[int] = None,
    enable_structured_metrics: bool = False,
    min_faces_threshold: Optional[int] = None,
    target_length: Optional[int] = None,
    max_retries: Optional[int] = None,
    transition_threshold: Optional[float] = None,
    max_gap_seconds: Optional[float] = None,
    max_samples_per_segment: Optional[int] = None,
    emo_path: Optional[str] = None,
    device: Optional[str] = None,
) -> str:
    """
    Основная логика обработки emotion_face.
    
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

    logger.info(f"VisualProcessor | {MODULE_NAME} | main | Инициализация VideoEmotionProcessor")

    # Подготавливаем параметры для VideoEmotionProcessor
    processor_kwargs = {}
    if min_frames_ratio is not None:
        processor_kwargs["min_frames_ratio"] = min_frames_ratio
    if min_keyframes is not None:
        processor_kwargs["min_keyframes"] = min_keyframes
    if min_transitions is not None:
        processor_kwargs["min_transitions"] = min_transitions
    if min_diversity_threshold is not None:
        processor_kwargs["min_diversity_threshold"] = min_diversity_threshold
    if quality_threshold is not None:
        processor_kwargs["quality_threshold"] = quality_threshold
    if memory_threshold_low is not None:
        processor_kwargs["memory_threshold_low"] = memory_threshold_low
    if batch_load_low is not None:
        processor_kwargs["batch_load_low"] = batch_load_low
    if batch_process_low is not None:
        processor_kwargs["batch_process_low"] = batch_process_low
    if memory_threshold_medium is not None:
        processor_kwargs["memory_threshold_medium"] = memory_threshold_medium
    if batch_load_medium is not None:
        processor_kwargs["batch_load_medium"] = batch_load_medium
    if batch_process_medium is not None:
        processor_kwargs["batch_process_medium"] = batch_process_medium
    if memory_threshold_high is not None:
        processor_kwargs["memory_threshold_high"] = memory_threshold_high
    if batch_load_high is not None:
        processor_kwargs["batch_load_high"] = batch_load_high
    if batch_process_high is not None:
        processor_kwargs["batch_process_high"] = batch_process_high
    if batch_load_very_high is not None:
        processor_kwargs["batch_load_very_high"] = batch_load_very_high
    if batch_process_very_high is not None:
        processor_kwargs["batch_process_very_high"] = batch_process_very_high
    if enable_structured_metrics:
        processor_kwargs["enable_structured_metrics"] = True
    if min_faces_threshold is not None:
        processor_kwargs["min_faces_threshold"] = min_faces_threshold
    if target_length is not None:
        processor_kwargs["target_length"] = target_length
    if max_retries is not None:
        processor_kwargs["max_retries"] = max_retries
    if transition_threshold is not None:
        processor_kwargs["transition_threshold"] = transition_threshold
    if max_gap_seconds is not None:
        processor_kwargs["max_gap_seconds"] = max_gap_seconds
    if max_samples_per_segment is not None:
        processor_kwargs["max_samples_per_segment"] = max_samples_per_segment
    if emo_path is not None:
        processor_kwargs["emo_path"] = emo_path
    if device is not None:
        processor_kwargs["device"] = device

    processor = VideoEmotionProcessor(**processor_kwargs)

    logger.info(f"VisualProcessor | {MODULE_NAME} | main | VideoEmotionProcessor инициализирован")

    try:
        save_path = os.path.join(rs_path, MODULE_NAME)
        result = processor.process(frame_manager, save_path)

        # Сохраняем результаты через ResultsStore
        rs.store(result, name=MODULE_NAME)

        logger.info(
            f"VisualProcessor | {MODULE_NAME} | main | Обработка завершена. "
            f"Результаты сохранены в {save_path}"
        )

        return save_path

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
        description="Анализ эмоций на лицах в видео — CLI",
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

    # Validation parameters
    parser.add_argument("--min-frames-ratio", type=float, help="Минимальное соотношение кадров")
    parser.add_argument("--min-keyframes", type=int, help="Минимальное количество ключевых кадров")
    parser.add_argument("--min-transitions", type=int, help="Минимальное количество переходов")
    parser.add_argument(
        "--min-diversity-threshold",
        type=float,
        help="Минимальный порог разнообразия",
    )
    parser.add_argument("--quality-threshold", type=float, help="Порог качества")

    # Performance parameters
    parser.add_argument("--memory-threshold-low", type=int, help="Порог памяти (низкий)")
    parser.add_argument("--batch-load-low", type=int, help="Размер батча загрузки (низкий)")
    parser.add_argument("--batch-process-low", type=int, help="Размер батча обработки (низкий)")
    parser.add_argument(
        "--memory-threshold-medium",
        type=int,
        help="Порог памяти (средний)",
    )
    parser.add_argument("--batch-load-medium", type=int, help="Размер батча загрузки (средний)")
    parser.add_argument(
        "--batch-process-medium",
        type=int,
        help="Размер батча обработки (средний)",
    )
    parser.add_argument("--memory-threshold-high", type=int, help="Порог памяти (высокий)")
    parser.add_argument("--batch-load-high", type=int, help="Размер батча загрузки (высокий)")
    parser.add_argument("--batch-process-high", type=int, help="Размер батча обработки (высокий)")
    parser.add_argument(
        "--batch-load-very-high",
        type=int,
        help="Размер батча загрузки (очень высокий)",
    )
    parser.add_argument(
        "--batch-process-very-high",
        type=int,
        help="Размер батча обработки (очень высокий)",
    )

    # Processing parameters
    parser.add_argument(
        "--enable-structured-metrics",
        action="store_true",
        help="Включить структурированные метрики",
    )
    parser.add_argument("--min-faces-threshold", type=int, help="Минимальный порог лиц")
    parser.add_argument("--target-length", type=int, help="Целевая длина последовательности")
    parser.add_argument("--max-retries", type=int, help="Максимальное количество повторных попыток")
    parser.add_argument("--transition-threshold", type=float, help="Порог перехода")
    parser.add_argument("--max-gap-seconds", type=float, help="Максимальный разрыв в секундах")
    parser.add_argument(
        "--max-samples-per-segment",
        type=int,
        help="Максимальное количество сэмплов на сегмент",
    )

    # Model parameters
    parser.add_argument("--emo-path", type=str, help="Путь к модели EmoNet")
    parser.add_argument("--device", type=str, help="Устройство для обработки (cuda/cpu)")

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
            min_frames_ratio=args.min_frames_ratio,
            min_keyframes=args.min_keyframes,
            min_transitions=args.min_transitions,
            min_diversity_threshold=args.min_diversity_threshold,
            quality_threshold=args.quality_threshold,
            memory_threshold_low=args.memory_threshold_low,
            batch_load_low=args.batch_load_low,
            batch_process_low=args.batch_process_low,
            memory_threshold_medium=args.memory_threshold_medium,
            batch_load_medium=args.batch_load_medium,
            batch_process_medium=args.batch_process_medium,
            memory_threshold_high=args.memory_threshold_high,
            batch_load_high=args.batch_load_high,
            batch_process_high=args.batch_process_high,
            batch_load_very_high=args.batch_load_very_high,
            batch_process_very_high=args.batch_process_very_high,
            enable_structured_metrics=args.enable_structured_metrics,
            min_faces_threshold=args.min_faces_threshold,
            target_length=args.target_length,
            max_retries=args.max_retries,
            transition_threshold=args.transition_threshold,
            max_gap_seconds=args.max_gap_seconds,
            max_samples_per_segment=args.max_samples_per_segment,
            emo_path=args.emo_path,
            device=args.device,
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
