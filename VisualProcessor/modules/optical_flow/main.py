import os
import sys

_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

if _path not in sys.path:
    sys.path.append(_path)

from core.flow_statistics import FlowStatisticsAnalyzer
from core.optical_flow import OpticalFlowProcessor
from core.config import (
    FlowPipelineConfig,
    FlowStatisticsConfig,
    DEFAULT_GRID_SIZES,
    DEFAULT_MOTION_THRESHOLDS
)

from utils.frame_manager import FrameManager
from utils.results_store import ResultsStore
from utils.logger import get_logger
from utils.utilites import load_metadata

name = "optical_flow"
logger = get_logger(name)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Production пайплайн обработки видео с RAFT')
    
    parser.add_argument('--frames-dir', type=str, help='')
    parser.add_argument('--rs-path',    type=str, help='')
    
    parser.add_argument('--model',                type=str, default='small', choices=['small', 'large'], help='Модель RAFT')
    parser.add_argument('--max-dim',              type=int, default=256, help='Максимальный размер стороны')
    parser.add_argument('--no-overlay',           action='store_true', help='Не сохранять overlay')
    parser.add_argument('--run-stats',            action='store_true', help='Запустить статистический анализ')
    
    parser.add_argument('--grid-size',                type=str, help='')
    parser.add_argument('--motion-thresholds',        type=str, help='')
    parser.add_argument('--direction-bins',           type=int, help='')
    parser.add_argument('--spatial-sample-rate',      type=int, help='')
    parser.add_argument('--top-regions-count',        type=int, help='')
    parser.add_argument('--savgol-window',            type=int, help='')
    parser.add_argument('--min-frames-for-temporal',  type=int, help='')
    parser.add_argument('--peak-detection-height',    type=float, help='')
    parser.add_argument('--enable-camera-motion',     action="store_true", help='')
    parser.add_argument('--enable-advanced-features', action="store_true", help='')
    parser.add_argument('--enable-mei',               action="store_true", help='')
    parser.add_argument('--enable-fg-bg',             action="store_true", help='')
    parser.add_argument('--enable-clusters',          action="store_true", help='')
    parser.add_argument('--enable-smoothness',        action="store_true", help='')
    parser.add_argument('--fg-bg-method',             type=str, help='')
    parser.add_argument('--fg-bg-threshold',          type=float, help='')
    parser.add_argument('--motion-clusters-n',        type=int, help='')
    
    args = parser.parse_args()
    
    flow_config = FlowPipelineConfig(
        model_type=args.model,
        max_dimension=args.max_dim,
        save_overlay=not args.no_overlay,
        save_flow_tensors=True
    )

    p = f"{os.path.dirname(os.path.dirname(os.path.dirname(__file__)))}/result_store/{name}"
    os.makedirs(p, exist_ok=True)

    flow_config.output_dir = p

    grid_size = tuple([int(i) for i in args.grid_size.split(",")]) if args.grid_size else DEFAULT_GRID_SIZES[1]
    motion_thresholds = tuple([float(i) for i in args.motion_thresholds.split(",")]) if args.motion_thresholds else DEFAULT_MOTION_THRESHOLDS
    
    stats_config = FlowStatisticsConfig(
        grid_size = grid_size,
        grid_sizes = DEFAULT_GRID_SIZES,
        motion_thresholds = motion_thresholds,
        direction_bins = args.direction_bins,
        spatial_sample_rate = args.spatial_sample_rate,
        top_regions_count = args.top_regions_count,
        savgol_window = args.savgol_window,
        min_frames_for_temporal = args.min_frames_for_temporal,
        peak_detection_height = args.peak_detection_height,
        enable_camera_motion = args.enable_camera_motion,
        enable_advanced_features = args.enable_advanced_features,
        enable_mei = args.enable_mei,
        enable_fg_bg = args.enable_fg_bg,
        enable_clusters = args.enable_clusters,
        enable_smoothness = args.enable_smoothness,
        fg_bg_method = args.fg_bg_method,
        fg_bg_threshold = args.fg_bg_threshold,
        motion_clusters_n = args.motion_clusters_n
    )

    flow_processor = OpticalFlowProcessor(flow_config)
    stats_analyzer = FlowStatisticsAnalyzer(stats_config)
    
    try:
        metadata = load_metadata(f"{args.frames_dir}/metadata.json", name)
        logger.info(f"{name} | main | Загружена метадата")
    except Exception as e:
        logger.error(f"{name} | main | Ошибка загрузки метадаты: {e}")
        raise
    
    frame_manager = FrameManager(frames_dir=args.frames_dir, chunk_size=metadata["chunk_size"], cache_size=metadata["cache_size"])
    
    rs = ResultsStore(root_path=args.rs_path)
    
    flow_results = flow_processor.process_video(frame_manager=frame_manager, frame_indices=metadata[name]["frame_indices"])
    stats_results = stats_analyzer.analyze_video(flow_results['flow_dir'], flow_results['metadata'])
    
    # result = {"flow_results":flow_results, "stats_results":stats_results}
