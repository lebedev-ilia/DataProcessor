import os
import sys
_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

if _path not in sys.path:
    sys.path.append(_path)

import json
import argparse

from modules.scene_classification.scene_classification import Places365SceneClassifier
from utils.frame_manager import FrameManager
from utils.results_store import ResultsStore

name = "scene_classification"

from utils.logger import get_logger
logger = get_logger(name)

def load_metadata(meta_path):
    try:
        with open(meta_path, "r") as f:
            return json.load(f)
    except:
        raise f"{name} | main | load_metadata | Ошибка при открытии метадаты"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--model-arch',                 type=str,              default=None, help='')
    parser.add_argument('--use-timm',                   action="store_true",   default=None, help='')
    parser.add_argument('--min-scene-length',           type=int,   default=None, help='')
    parser.add_argument('--batch-size',                 type=int,              default=None, help='')
    parser.add_argument('--device',                     type=str,              default=None, help='')
    parser.add_argument('--categories-path',            type=str,              default=None, help='')
    parser.add_argument('--cache-dir',                  type=str,              default=None, help='')
    parser.add_argument('--gpu-memory-threshold',       type=float,            default=None, help='')
    parser.add_argument('--log-metrics-every-n-frames', type=int,              default=None, help='')
    parser.add_argument('--input-size',                 type=int,              default=None, help='')
    parser.add_argument('--use-tta',                    action="store_true",   default=None, help='')
    parser.add_argument('--use-multi-crop',             action="store_true",   default=None, help='')
    parser.add_argument('--temporal-smoothing',         action="store_true",   default=None, help='')
    parser.add_argument('--smoothing-window',           type=int,              default=None, help='')
    parser.add_argument('--enable-advanced-features',   action="store_true",   default=None, help='')
    parser.add_argument('--use-clip-for-semantics',     action="store_true",   default=None, help='')
    parser.add_argument('--frames-dir',                 type=str,              default=None, help='')
    parser.add_argument('--rs-path',                    type=str,              default=None, help='')
    
    args = parser.parse_args()
    
    frames_dir = args.frames_dir
    rs_path    = args.rs_path
    
    metadata   = load_metadata(f"{frames_dir}/metadata.json")
    
    model_arch                 = args.model_arch
    use_timm                   = args.use_timm
    min_scene_length           = args.min_scene_length
    batch_size                 = args.batch_size
    device                     = args.device
    categories_path            = args.categories_path
    cache_dir                  = args.cache_dir
    gpu_memory_threshold       = args.gpu_memory_threshold
    log_metrics_every_n_frames = args.log_metrics_every_n_frames
    input_size                 = args.input_size
    use_tta                    = args.use_tta
    use_multi_crop             = args.use_multi_crop
    temporal_smoothing         = args.temporal_smoothing
    smoothing_window           = args.smoothing_window
    enable_advanced_features   = args.enable_advanced_features
    use_clip_for_semantics     = args.use_clip_for_semantics
    
    classifier = Places365SceneClassifier(
        model_arch=model_arch,
        use_timm=use_timm,
        min_scene_length=min_scene_length,
        batch_size=batch_size,
        device=device,
        categories_path=categories_path,
        cache_dir=cache_dir,
        gpu_memory_threshold=gpu_memory_threshold,
        log_metrics_every_n_frames=log_metrics_every_n_frames,
        input_size=input_size,
        use_tta=use_tta,
        use_multi_crop=use_multi_crop,
        temporal_smoothing=temporal_smoothing,
        smoothing_window=smoothing_window,
        enable_advanced_features=enable_advanced_features,
        use_clip_for_semantics=use_clip_for_semantics,
        rs_path=rs_path
    )
    
    rs = ResultsStore(root_path=rs_path)
    
    frame_manager = FrameManager(frames_dir=frames_dir, chunk_size=metadata["chunk_size"], cache_size=metadata["cache_size"])
    frame_indices = metadata[name]["frame_indices"]
    
    result = classifier.classify_with_advanced_features(frame_manager=frame_manager, frame_indices=frame_indices)
    
    rs.store(result, name=name)
    
    
    
    