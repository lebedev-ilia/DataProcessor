import os
import sys
_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

if _path not in sys.path:
    sys.path.append(_path)

import json
import argparse

from modules.face_detection.face_detector import FaceDetector
from utils.frame_manager import FrameManager
from utils.results_store import ResultsStore

name = "face_detection"

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
    
    parser.add_argument('--frames-dir',    type=str,   default=None, help='')
    parser.add_argument('--rs-path',    type=str,   default=None, help='')
    parser.add_argument('--det-size',    type=str,   default=None, help='')
    parser.add_argument('--threshold',    type=float,   default=None, help='')
    
    args = parser.parse_args()      
    
    metadata = load_metadata(f"{args.frames_dir}/metadata.json")
    rs = ResultsStore(args.rs_path)

    frame_manager = FrameManager(frames_dir=args.frames_dir, chunk_size=metadata["chunk_size"], cache_size=metadata["cache_size"])
    frame_indices = metadata[name]["frame_indices"]

    det_size = args.det_size.split(",")

    detector = FaceDetector(args.threshold, det_size)

    timeline = detector.run(frame_manager=frame_manager, frame_indices=frame_indices)

    logger.info(f"FaceDetector | Обнаружено лиц: {len(timeline)}")
    
    rs.store(timeline, name=name)
    
    