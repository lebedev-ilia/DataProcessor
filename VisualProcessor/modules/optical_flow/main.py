#!/usr/bin/env python3
"""
CLI интерфейс для модуля оптического потока.

Использует OpticalFlowProcessor и FlowStatisticsAnalyzer для анализа оптического потока в видео.
"""

from __future__ import annotations

import os
import sys
import argparse
from typing import Optional, List

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from modules.optical_flow.core.flow_statistics import FlowStatisticsAnalyzer
from modules.optical_flow.core.optical_flow import OpticalFlowProcessor
from modules.optical_flow.core.config import (
    FlowPipelineConfig,
    FlowStatisticsConfig,
    DEFAULT_GRID_SIZES,
    DEFAULT_MOTION_THRESHOLDS,
)
from utils.frame_manager import FrameManager
from utils.results_store import ResultsStore
from utils.logger import get_logger
from utils.utilites import load_metadata

MODULE_NAME = "optical_flow"
logger = get_logger(MODULE_NAME)


def run_pipeline(
    frames_dir: str,
    rs_path: str,
    model: str = "small",
    max_dim: int = 256,
    use_overlay: bool = False,
    run_stats: bool = False,
    device: Optional[str] = None,
    enable_forward_backward: bool = False,
    save_backward_flow: bool = False,
    fb_error_threshold: Optional[float] = None,
    occlusion_error_threshold: Optional[float] = None,
    grid_size: Optional[str] = None,
    motion_thresholds: Optional[str] = None,
    direction_bins: Optional[int] = None,
    spatial_sample_rate: Optional[int] = None,
    top_regions_count: Optional[int] = None,
    savgol_window: Optional[int] = None,
    min_frames_for_temporal: Optional[int] = None,
    peak_detection_height: Optional[float] = None,
    enable_camera_motion: bool = False,
    enable_advanced_features: bool = False,
    enable_mei: bool = False,
    enable_fg_bg: bool = False,
    enable_clusters: bool = False,
    enable_smoothness: bool = False,
    fg_bg_method: Optional[str] = None,
    fg_bg_threshold: Optional[float] = None,
    motion_clusters_n: Optional[int] = None,
) -> str:
    """
    Основная логика обработки optical_flow.
    
    Возвращает путь к сохраненному файлу результатов.
    """
    if not rs_path:
        raise ValueError(f"{MODULE_NAME} | rs_path не указан")

    # Создаем директорию для результатов
    output_dir = os.path.join(rs_path, MODULE_NAME)
    os.makedirs(output_dir, exist_ok=True)

    # Загружаем метаданные
    try:
        metadata = load_metadata(f"{frames_dir}/metadata.json", MODULE_NAME)
        logger.info(f"{MODULE_NAME} | main | Загружена метадата")
    except Exception as e:
        logger.error(f"{MODULE_NAME} | main | Ошибка загрузки метадаты: {e}")
        raise

    frame_manager = FrameManager(
        frames_dir=frames_dir,
        chunk_size=metadata["chunk_size"],
        cache_size=metadata["cache_size"],
    )

    frame_indices = metadata[MODULE_NAME]["frame_indices"]

    # Настройка конфигурации flow pipeline
    flow_config = FlowPipelineConfig(
        model_type=model,
        max_dimension=max_dim,
        save_overlay=use_overlay,
        device=device,
        enable_forward_backward=enable_forward_backward,
        save_backward_flow=save_backward_flow,
        fb_error_threshold=fb_error_threshold,
        occlusion_error_threshold=occlusion_error_threshold,
    )
    flow_config.output_dir = output_dir

    # Настройка конфигурации статистики
    grid_size_tuple = (
        tuple([int(i) for i in grid_size.split(",")])
        if grid_size
        else DEFAULT_GRID_SIZES[1]
    )
    motion_thresholds_tuple = (
        tuple([float(i) for i in motion_thresholds.split(",")])
        if motion_thresholds
        else DEFAULT_MOTION_THRESHOLDS
    )

    stats_config = FlowStatisticsConfig(
        grid_size=grid_size_tuple,
        grid_sizes=DEFAULT_GRID_SIZES,
        motion_thresholds=motion_thresholds_tuple,
        direction_bins=direction_bins,
        spatial_sample_rate=spatial_sample_rate,
        top_regions_count=top_regions_count,
        savgol_window=savgol_window,
        min_frames_for_temporal=min_frames_for_temporal,
        peak_detection_height=peak_detection_height,
        enable_camera_motion=enable_camera_motion,
        enable_advanced_features=enable_advanced_features,
        enable_mei=enable_mei,
        enable_fg_bg=enable_fg_bg,
        enable_clusters=enable_clusters,
        enable_smoothness=enable_smoothness,
        fg_bg_method=fg_bg_method,
        fg_bg_threshold=fg_bg_threshold,
        motion_clusters_n=motion_clusters_n,
    )

    rs = ResultsStore(root_path=rs_path)

    try:
        # Обработка оптического потока
        flow_processor = OpticalFlowProcessor(flow_config)
        flow_results = flow_processor.process_video(
            frame_manager=frame_manager, frame_indices=frame_indices
        )

        # Статистический анализ (если включен)
        stats_results = None
        if run_stats:
            stats_analyzer = FlowStatisticsAnalyzer(stats_config)
            stats_results = stats_analyzer.analyze_video(
                flow_results["flow_dir"], flow_results["metadata"]
            )

        # Формируем итоговый результат
        result = {
            "flow_results": flow_results,
        }
        if stats_results is not None:
            result["stats_results"] = stats_results

        # Сохраняем результаты через ResultsStore
        rs.store(result, name=MODULE_NAME)

        logger.info(
            f"VisualProcessor | {MODULE_NAME} | main | Обработка завершена. "
            f"Результаты сохранены в {output_dir}"
        )

        return output_dir

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
        description="Анализ оптического потока в видео — CLI",
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

    # Flow pipeline параметры
    parser.add_argument(
        "--model",
        type=str,
        default="small",
        choices=["small", "large"],
        help="Модель RAFT",
    )
    parser.add_argument(
        "--max-dim",
        type=int,
        default=256,
        help="Максимальный размер стороны",
    )
    parser.add_argument(
        "--use-overlay",
        action="store_true",
        help="Сохранять overlay визуализацию",
    )
    parser.add_argument(
        "--run-stats",
        action="store_true",
        help="Запустить статистический анализ",
    )
    parser.add_argument("--device", type=str, help="Устройство для обработки (cuda/cpu)")
    parser.add_argument(
        "--enable-forward-backward",
        action="store_true",
        help="Включить forward-backward проверку",
    )
    parser.add_argument(
        "--save-backward-flow",
        action="store_true",
        help="Сохранять обратный поток",
    )
    parser.add_argument(
        "--fb-error-threshold",
        type=float,
        help="Порог ошибки forward-backward",
    )
    parser.add_argument(
        "--occlusion-error-threshold",
        type=float,
        help="Порог ошибки окклюзии",
    )

    # Statistics параметры
    parser.add_argument("--grid-size", type=str, help="Размер сетки (формат: 'x,y')")
    parser.add_argument(
        "--motion-thresholds",
        type=str,
        help="Пороги движения (формат: 'min,max')",
    )
    parser.add_argument("--direction-bins", type=int, help="Количество бинов направления")
    parser.add_argument(
        "--spatial-sample-rate",
        type=int,
        help="Частота пространственной выборки",
    )
    parser.add_argument(
        "--top-regions-count",
        type=int,
        help="Количество топовых регионов",
    )
    parser.add_argument("--savgol-window", type=int, help="Окно Savitzky-Golay")
    parser.add_argument(
        "--min-frames-for-temporal",
        type=int,
        help="Минимальное количество кадров для временного анализа",
    )
    parser.add_argument(
        "--peak-detection-height",
        type=float,
        help="Высота для детекции пиков",
    )
    parser.add_argument(
        "--enable-camera-motion",
        action="store_true",
        help="Включить анализ движения камеры",
    )
    parser.add_argument(
        "--enable-advanced-features",
        action="store_true",
        help="Включить расширенные фичи",
    )
    parser.add_argument("--enable-mei", action="store_true", help="Включить MEI")
    parser.add_argument(
        "--enable-fg-bg",
        action="store_true",
        help="Включить анализ переднего/заднего плана",
    )
    parser.add_argument(
        "--enable-clusters",
        action="store_true",
        help="Включить кластеризацию движения",
    )
    parser.add_argument(
        "--enable-smoothness",
        action="store_true",
        help="Включить анализ плавности",
    )
    parser.add_argument("--fg-bg-method", type=str, help="Метод разделения переднего/заднего плана")
    parser.add_argument(
        "--fg-bg-threshold",
        type=float,
        help="Порог для разделения переднего/заднего плана",
    )
    parser.add_argument(
        "--motion-clusters-n",
        type=int,
        help="Количество кластеров движения",
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
            model=args.model,
            max_dim=args.max_dim,
            use_overlay=args.use_overlay,
            run_stats=args.run_stats,
            device=args.device,
            enable_forward_backward=args.enable_forward_backward,
            save_backward_flow=args.save_backward_flow,
            fb_error_threshold=args.fb_error_threshold,
            occlusion_error_threshold=args.occlusion_error_threshold,
            grid_size=args.grid_size,
            motion_thresholds=args.motion_thresholds,
            direction_bins=args.direction_bins,
            spatial_sample_rate=args.spatial_sample_rate,
            top_regions_count=args.top_regions_count,
            savgol_window=args.savgol_window,
            min_frames_for_temporal=args.min_frames_for_temporal,
            peak_detection_height=args.peak_detection_height,
            enable_camera_motion=args.enable_camera_motion,
            enable_advanced_features=args.enable_advanced_features,
            enable_mei=args.enable_mei,
            enable_fg_bg=args.enable_fg_bg,
            enable_clusters=args.enable_clusters,
            enable_smoothness=args.enable_smoothness,
            fg_bg_method=args.fg_bg_method,
            fg_bg_threshold=args.fg_bg_threshold,
            motion_clusters_n=args.motion_clusters_n,
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
