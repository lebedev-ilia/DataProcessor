"""
config.py - Единая конфигурация для всей системы
"""
import torch
from dataclasses import dataclass, field
from typing import Tuple, List, Dict

@dataclass
class FlowPipelineConfig:
    """Конфигурация пайплайна обработки оптического потока."""
    output_dir: str = "raft_output"
    model_type: str = "small"  # "small" или "large"
    max_dimension: int = 512
    save_overlay: bool = True
    save_flow_tensors: bool = True
    create_summary_video: bool = False
    device: str = "auto"  # "auto", "cuda", "cpu"
    # Контроль качества потока
    enable_forward_backward: bool = True
    save_backward_flow: bool = False
    fb_error_threshold: float = 1.0  # px
    occlusion_error_threshold: float = 1.5  # px
    
    def __post_init__(self):
        if self.device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

# Константы
DEFAULT_GRID_SIZES = [(1, 1), (4, 4), (8, 8)]
DEFAULT_MOTION_THRESHOLDS = [1.0]  # в px/sec
DIRECTION_BINS = 36
MAX_SAVGOL_WINDOW = 11

# Camera motion defaults
CAMERA_MOTION_CONFIG = {
    "mag_bg_thresh": 0.6,         # background threshold in px
    "zoom_eps": 1e-3,             # minimal scale delta to count zoom
    "sharp_angle_thresh_deg": 15  # reserved for future heuristics
}

@dataclass
class FlowStatisticsConfig:
    """Конфигурация для статистического анализа."""
    grid_size: List[int] = field(default_factory=tuple)  # устаревшее поле для обратной совместимости
    grid_sizes: List[tuple] = field(default_factory=lambda: DEFAULT_GRID_SIZES.copy())
    motion_thresholds: List[float] = field(default_factory=lambda: DEFAULT_MOTION_THRESHOLDS.copy())  # px/sec
    moving_threshold_k: float = 3.0  # median + k * MAD
    noise_floor: float = 0.05  # px/frame
    direction_bins: int = DIRECTION_BINS
    spatial_sample_rate: int = 10
    top_regions_count: int = 3
    savgol_window: int = MAX_SAVGOL_WINDOW
    min_frames_for_temporal: int = 10
    peak_detection_height: float = 0.1
    enable_camera_motion: bool = False
    camera_motion_config: Dict[str, float] = field(default_factory=lambda: CAMERA_MOTION_CONFIG.copy())
    # Продвинутые фичи
    enable_advanced_features: bool = True
    enable_mei: bool = True
    enable_fg_bg: bool = True
    enable_clusters: bool = True
    enable_smoothness: bool = True
    fg_bg_method: str = 'magnitude_threshold'  # 'magnitude_threshold', 'spatial_clustering', 'segmentation'
    fg_bg_threshold: float = 0.5
    motion_clusters_n: int = 5

@dataclass
class SystemConfig:
    """Общая конфигурация системы."""
    # Параметры обработки
    default_model: str = "small"  # "small" или "large"
    default_max_dim: int = 512
    default_frame_skip: int = 5
    
    # Параметры хранения
    output_structure: str = "hierarchical"  # "hierarchical" или "flat"
    save_intermediate: bool = True
    compression_level: int = 1  # 0-9
    
    # Параметры производительности
    use_gpu: bool = True
    batch_size: int = 1
    num_workers: int = 0
    
    # Параметры анализа
    enable_statistics: bool = True
    enable_classification: bool = False
    enable_clustering: bool = False

# Предустановки для разных типов видео
VIDEO_PRESETS = {
    'talk_show': {
        'max_dim': 384,
        'skip': 10,
        'model': 'small'
    },
    'gaming': {
        'max_dim': 512,
        'skip': 3,
        'model': 'small'
    },
    'tutorial': {
        'max_dim': 256,
        'skip': 15,
        'model': 'small'
    },
    'sports': {
        'max_dim': 512,
        'skip': 2,
        'model': 'large'
    },
    'vlog': {
        'max_dim': 384,
        'skip': 5,
        'model': 'small'
    }
}