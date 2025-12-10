import json
import argparse

from modules.scene_classification.scene_classification import Places365SceneClassifier
from utils.frame_manager import FrameManager
from utils.results_store import ResultsStore

name = "scene_classification"

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
    
    parser.add_argument('--model_arch',                 type=str,   default=None, help='')
    parser.add_argument('--use_timm',                   type=str,   default=None, help='')
    parser.add_argument('--top_k',                      type=str,   default=None, help='')
    parser.add_argument('--batch_size',                 type=float, default=None, help='')
    parser.add_argument('--device',                     type=str,   default=None, help='')
    parser.add_argument('--categories_path',            type=str,   default=None, help='')
    parser.add_argument('--cache_dir',                  type=str,   default=None, help='')
    parser.add_argument('--gpu_memory_threshold',       type=str,   default=None, help='')
    parser.add_argument('--log_metrics_every_n_frames', type=str,   default=None, help='')
    parser.add_argument('--input_size',                 type=str,   default=None, help='')
    parser.add_argument('--use_tta',                    type=str,   default=None, help='')
    parser.add_argument('--use_multi_crop',             type=str,   default=None, help='')
    parser.add_argument('--temporal_smoothing',         type=str,   default=None, help='')
    parser.add_argument('--smoothing_window',           type=str,   default=None, help='')
    parser.add_argument('--enable_advanced_features',   type=str,   default=None, help='')
    parser.add_argument('--use_clip_for_semantics',     type=str,   default=None, help='')
    parser.add_argument('--frames-dir',    type=str,   default=None, help='')
    parser.add_argument('--rs-path',       type=str,   default=None, help='')
    
    args = parser.parse_args()
    
    frames_dir = args.frames_dir
    rs_path    = args.rs_path
    
    metadata   = load_metadata(f"{frames_dir}/metadata.json")
    
    model_arch                 = args.model_arch
    use_timm                   = args.use_timm
    top_k                      = args.top_k
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
        top_k=top_k,
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
        use_clip_for_semantics=use_clip_for_semantics
    )
    
    rs = ResultsStore(root_path=rs_path)
    
    frame_manager = FrameManager(frames_dir=frames_dir, chunk_size=metadata["chunk_size"], cache_size=metadata["cache_size"])
    frame_indices = metadata[name]["frame_indices"]
    
    result = classifier.classify_with_advanced_features(frame_manager=frame_manager, frame_indices=frame_indices)
    
    rs.store(result, name=name)
    
    
    
    