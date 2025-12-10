import json
import argparse

from modules.object_detection.object_detection import ObjectDetectionModule
from modules.object_detection.config import Config

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
    parser = argparse.ArgumentParser(
        description='',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--model-name',    type=str,   default=None, help='')
    parser.add_argument('--model-family',  type=str,   default=None, help='')
    parser.add_argument('--device',        type=str,   default=None, help='')
    parser.add_argument('--box-threshold', type=float, default=None, help='')
    parser.add_argument('--frames-dir',    type=str,   default=None, help='')
    parser.add_argument('--meta-path',     type=str,   default=None, help='')
    parser.add_argument('--rs-path',       type=str,   default=None, help='')
    
    args = parser.parse_args()
    
    cfg = Config()
    
    model_name    = args.model_name    or cfg.model_name
    model_family  = args.model_family  or cfg.model_family
    device        = args.device        or cfg.device
    box_threshold = args.box_threshold or cfg.box_threshold
    
    frames_dir    = args.frames_dir    or cfg.frames_dir
    meta_path     = args.meta_path     or cfg.meta_path
    rs_path       = args.rs_path       or cfg.rs_path
    
    metadata = load_metadata(meta_path)
    
    rs = ResultsStore()
    
    detector = ObjectDetectionModule(
        model_name=model_name,
        model_family=model_family, 
        device=device,  
        box_threshold=box_threshold
    )
    
    frame_manager = FrameManager(frames_dir=frames_dir, chunk_size=metadata["chunk_size"], cache_size=metadata["cache_size"])
    frame_indices = metadata[name]["frame_indices"]
    
    result = detector.run(frame_manager=frame_manager, frame_indices=frame_indices)
    
    rs.store(result, name=name)
    
    
    