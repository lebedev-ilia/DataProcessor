import os
import sys
_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

if _path not in sys.path:
    sys.path.append(_path)

import logging
import json
import argparse

from utils.frame_manager import FrameManager
from utils.results_store import ResultsStore

name = "object_detection"

from utils.logger import get_logger
logger = get_logger(name)

class ObjectDetectException:
    pass

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
    
    parser.add_argument('--model',    type=str,   default=None, help='')
    parser.add_argument('--batch-size',    type=int,   default=None, help='')
    parser.add_argument('--use-queries',   action='store_true',   default=None, help='')
    parser.add_argument('--default-categories',    type=str,   default=None, help='')
    parser.add_argument('--model-family',  type=str,   default=None, help='')
    parser.add_argument('--device',        type=str,   default=None, help='')
    parser.add_argument('--box-threshold', type=float, default=None, help='')
    parser.add_argument('--frames-dir',    type=str,   default=None, help='')
    parser.add_argument('--rs-path',       type=str,   default=None, help='')
    
    args = parser.parse_args()      
    
    metadata = load_metadata(f"{args.frames_dir}/metadata.json")
    rs = ResultsStore(args.rs_path)

    frame_manager = FrameManager(frames_dir=args.frames_dir, chunk_size=metadata["chunk_size"], cache_size=metadata["cache_size"])
    frame_indices = metadata[name]["frame_indices"]

    if args.use_queries:
        if args.default_categories:
            default_categories = args.default_categories.split(",")
        else:
            raise ObjectDetectException("default_categories должен быть заполнен при использовании use-queries")

        from modules.object_detection.object_detection_owl import ObjectDetectionModule

        detector = ObjectDetectionModule(
            model_name=args.model,
            model_family=args.model_family, 
            device=args.device,  
            box_threshold=args.box_threshold,
            default_categories=default_categories
        )

    else:

        from modules.object_detection.object_detection_yolo import ObjectDetectionYOLO

        detector = ObjectDetectionYOLO(args.model, args.box_threshold, args.batch_size)

    logger.info(f"VisualProcessor | {name} | main | Запущен detector.run (use_queries=True)")

    result = detector.run(frame_manager=frame_manager, frame_indices=frame_indices)

    logger.info(f"VisualProcessor | {name} | main | Получен результат | result len: {len(result)}")
    
    rs.store(result, name=name)
    
    
    