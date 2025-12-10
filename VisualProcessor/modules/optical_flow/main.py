import json

from core.flow_statistics import FlowStatisticsAnalyzer
from core.optical_flow import OpticalFlowProcessor
from core.config import FlowPipelineConfig, FlowStatisticsConfig

from utils.frame_manager import FrameManager
from utils.results_store import ResultsStore

name = "object_detection"


def load_metadata(meta_path):
    try:
        with open(meta_path, "r") as f:
            return json.load(f)
    except:
        raise f"{name} | main | load_metadata | Ошибка при открытии метадаты"


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Production пайплайн обработки видео с RAFT')
    
    parser.add_argument('--frames-dir', type=str, help='')
    parser.add_argument('--rs-path',    type=str, help='')
    
    parser.add_argument('--model',                type=str, default='small', choices=['small', 'large'], help='Модель RAFT')
    parser.add_argument('--max_dim',              type=int, default=256, help='Максимальный размер стороны')
    parser.add_argument('--no_overlay',           action='store_true', help='Не сохранять overlay')
    parser.add_argument('--run_stats',            action='store_true', help='Запустить статистический анализ')
    
    parser.add_argument('--grid-size',                help='')
    parser.add_argument('--motion-thresholds',        help='')
    parser.add_argument('--direction-bins',           help='')
    parser.add_argument('--spatial-sample-rate',      help='')
    parser.add_argument('--top-regions-count',        help='')
    parser.add_argument('--savgol-window',            help='')
    parser.add_argument('--min-frames-for-temporal',  help='')
    parser.add_argument('--peak-detection-height',    help='')
    parser.add_argument('--enable-camera-motion',     help='')
    parser.add_argument('--enable-advanced-features', help='')
    parser.add_argument('--enable-mei',               help='')
    parser.add_argument('--enable-fg-bg',             help='')
    parser.add_argument('--enable-clusters',          help='')
    parser.add_argument('--enable-smoothness',        help='')
    parser.add_argument('--fg-bg-method',             help='')
    parser.add_argument('--fg-bg-threshold',          help='')
    parser.add_argument('--motion-clusters-n',        help='')
    
    args = parser.parse_args()
    
    flow_config = FlowPipelineConfig(
        model_type=args.model,
        max_dimension=args.max_dim,
        save_overlay=not args.no_overlay,
        save_flow_tensors=True
    )
    
    stats_config = FlowStatisticsConfig(
        grid_size = args.grid_size,
        motion_thresholds = args.motion_thresholds,
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
    
    stats_config.enable_camera_motion = args.run_camera_motion
    stats_config.enable_advanced_features = not args.no_advanced_features

    flow_processor = OpticalFlowProcessor(flow_config)
    stats_analyzer = FlowStatisticsAnalyzer(stats_config)
    
    metadata = load_metadata(f"{args.frames_dir}/metadata.json")
    
    frame_manager = FrameManager(frames_dir=args.frames_dir, chunk_size=metadata["chunk_size"], cache_size=metadata["cache_size"])
    
    rs = ResultsStore(root_path=args.rs_path)
    
    flow_results = flow_processor.process_video(frame_manager=frame_manager, frame_indices=metadata[name]["frame_indices"])
    stats_results = stats_analyzer.analyze_video(flow_results['flow_dir'], flow_results['metadata'])
    
    result = {"flow_results":flow_results, "stats_results":stats_results}
    
    rs.store(result, name=name)
