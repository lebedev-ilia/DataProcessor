import json
import argparse

from balance_composition import Config, VideoCompositionAnalyzer

import os
import sys
_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

if _path not in sys.path:
    sys.path.append(_path)

from utils.frame_manager import FrameManager
from utils.results_store import ResultsStore

name = "frames_composition"

def load_metadata(meta_path):
    try:
        with open(meta_path, "r") as f:
            return json.load(f)
    except:
        raise f"{name} | main | load_metadata | Ошибка при открытии метадаты"

def main():
    parser = argparse.ArgumentParser(description='Frame Composition Analysis')
    parser.add_argument('--frames-dir',      type=str,            help='Path to input video file')
    parser.add_argument('--rs-path',         type=str, default=None, help='Output JSON file path (default: video_name.json)')
    parser.add_argument('--device',         type=str, default=None)
    parser.add_argument('--yolo-model-path',         type=str, default=None)
    parser.add_argument('--yolo-conf-threshold',         type=float, default=None)
    parser.add_argument('--max-num-faces',         type=int, default=None)
    parser.add_argument('--min-detection-confidence',         type=float, default=None)
    parser.add_argument('--use-midas',         action="store_true", default=None)
    parser.add_argument('--num-depth-layers',         type=int, default=None)
    parser.add_argument('--slic-n-segments',         type=int, default=None)
    parser.add_argument('--slic-compactness',         type=int, default=None)
    parser.add_argument('--brightness-weight',         type=float, default=None)
    parser.add_argument('--object-weight',         type=float, default=None)
    
    args = parser.parse_args()

    rs = ResultsStore(args.rs_path)

    metadata = load_metadata(f"{args.frames_dir}/metadata.json")

    frame_manager = FrameManager(args.frames_dir)

    config = Config(
        device = args.device,
        yolo_model_path = args.yolo_model_path,
        yolo_conf_threshold = args.yolo_conf_threshold ,
        max_num_faces = args.max_num_faces ,
        min_detection_confidence = args.min_detection_confidence ,
        use_midas = args.use_midas ,
        num_depth_layers = args.num_depth_layers ,
        slic_n_segments = args.slic_n_segments ,
        slic_compactness = args.slic_compactness ,
        brightness_weight = args.brightness_weight ,
        object_weight = args.object_weight,
    )
    
    analyzer = VideoCompositionAnalyzer(config)

    result = analyzer.analyze_video_frames(frame_manager, frame_indices=metadata[name]["frame_indices"])

    rs.store(result, name=name)

if __name__=="__main__":
    main()