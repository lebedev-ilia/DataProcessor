import json
import argparse
from pathlib import Path

import os
import sys
_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

if _path not in sys.path:
    sys.path.append(_path)

from modules.behavioral.behavior_analyzer import BehaviorAnalyzer
from utils.frame_manager import FrameManager
from utils.results_store import ResultsStore

name = "behavioral"

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

from utils.logger import get_logger
logger = get_logger(name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Анализ поведения людей в видео',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--frames-dir',    type=str, required=True, help='Путь к входному видео файлу')
    parser.add_argument('--rs-path',       type=str, help='')
    parser.add_argument('--visualize',     action='store_true', help='Сохранять визуализацию результатов на кадрах')
    parser.add_argument('--visualize-dir', type=str, default='./behavior_visualizations', help='Директория для сохранения визуализаций')
    parser.add_argument('--pose-static-image-mode', action="store_true")
    parser.add_argument('--pose-model-complexity', type=int)
    parser.add_argument('--pose-smooth-landmarks', action="store_true")
    parser.add_argument('--pose-min-detection-confidence', type=float)
    parser.add_argument('--pose-min-tracking-confidence', type=float)
    parser.add_argument('--hands-static-image-mode', action="store_true")
    parser.add_argument('--hands-max-num-hands', type=int)
    parser.add_argument('--hands-model-complexity', type=int)
    parser.add_argument('--hands-min-detection-confidence', type=float)
    parser.add_argument('--hands-min-tracking-confidence', type=float)
    parser.add_argument('--face-static-image-mode', action="store_true")
    parser.add_argument('--face-max-num-faces', type=int)
    parser.add_argument('--face-refine-landmarks', action="store_true")
    parser.add_argument('--face-min-detection-confidence', type=float)
    parser.add_argument('--face-min-tracking-confidence', type=float)
    
    args = parser.parse_args()
    
    if args.visualize:
        visualize_dir = Path(args.visualize_dir)
        visualize_dir.mkdir(parents=True, exist_ok=True)
    
    analyzer = BehaviorAnalyzer(
        pose_static_image_mode=args.pose_static_image_mode,
        pose_model_complexity=args.pose_model_complexity,
        pose_smooth_landmarks=args.pose_smooth_landmarks,
        pose_min_detection_confidence=args.pose_min_detection_confidence,
        pose_min_tracking_confidence=args.pose_min_tracking_confidence,
        hands_static_image_mode=args.hands_static_image_mode,
        hands_max_num_hands=args.hands_max_num_hands,
        hands_model_complexity=args.hands_model_complexity,
        hands_min_detection_confidence=args.hands_min_detection_confidence,
        hands_min_tracking_confidence=args.hands_min_tracking_confidence,
        face_static_image_mode=args.face_static_image_mode,
        face_max_num_faces=args.face_max_num_faces,
        face_refine_landmarks=args.face_refine_landmarks,
        face_min_detection_confidence=args.face_min_detection_confidence,
        face_min_tracking_confidence=args.face_min_tracking_confidence
    )
    
    metadata = load_json(f"{args.frames_dir}/metadata.json")
    
    frame_manager = FrameManager(frames_dir=args.frames_dir, chunk_size=metadata["chunk_size"], cache_size=metadata["cache_size"])
    rs = ResultsStore(args.rs_path)
    
    results = analyzer.process_video(frame_manager=frame_manager, frame_indices=metadata[name]["frame_indices"])
    
    rs.store(results, name=name)

    logger.info("Behavioral | main | Результат сохранен")