import json
import argparse

import os
import sys
_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

if _path not in sys.path:
    sys.path.append(_path)

from modules.detalize_face_modules.detalize_face_refactored import DetalizeFaceExtractorRefactored
from utils.frame_manager import FrameManager
from utils.results_store import ResultsStore

name = "detalize_face_modules"

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="DetalizeFaceExtractorRefactored - модульная система извлечения фич лица",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/Output arguments
    parser.add_argument("--frames-dir",               type=str, required=True, help="Путь к входному видео файлу или директории с изображениями")
    parser.add_argument("--rs-path",                  type=str)
    # Module configuration
    parser.add_argument("--modules",                  type=str, default=None)
    # Face detection parameters
    parser.add_argument("--max-faces",                type=int, default=4, help="Максимальное количество лиц для детекции на кадр")
    parser.add_argument("--refine-landmarks",         action="store_true", default=True, help="Использовать уточненные landmarks (468 точек)")
    parser.add_argument( "--no-refine-landmarks",     action="store_false", dest="refine_landmarks", help="Не использовать уточненные landmarks")
    # Quality filtering parameters
    parser.add_argument("--min-detection-confidence", type=float, default=0.7, help="Минимальная уверенность детекции лица")
    parser.add_argument("--min-tracking-confidence",  type=float, default=0.7, help="Минимальная уверенность трекинга лица")
    parser.add_argument("--min-face-size",            type=int,   default=30,  help="Минимальный размер лица в пикселях")
    parser.add_argument("--max-face-size-ratio",      type=float, default=0.8, help="Максимальное отношение размера лица к размеру кадра")
    parser.add_argument("--min-aspect-ratio",         type=float, default=0.6, help="Минимальное соотношение сторон лица")
    parser.add_argument("--max-aspect-ratio",         type=float, default=1.4, help="Максимальное соотношение сторон лица")
    parser.add_argument("--no-validate-landmarks",    action="store_false", dest="validate_landmarks", default=True, help="Отключить валидацию landmarks")
    # Visualization parameters
    parser.add_argument("--visualize",                action="store_true", help="Включить визуализацию результатов")
    parser.add_argument("--visualize-dir",            type=str, default="./face_visualizations", help="Директория для сохранения визуализаций")
    parser.add_argument("--show-landmarks",           action="store_true", help="Показывать landmarks на визуализации")
    
    args = parser.parse_args()
        
    modules = args.modules.split(",")
    
    extractor = DetalizeFaceExtractorRefactored(
        modules=modules,
        max_faces=args.max_faces,
        refine_landmarks=args.refine_landmarks,
        visualize=args.visualize,
        visualize_dir=args.visualize_dir,
        show_landmarks=args.show_landmarks,
        min_detection_confidence=args.min_detection_confidence,
        min_tracking_confidence=args.min_tracking_confidence,
        min_face_size=args.min_face_size,
        max_face_size_ratio=args.max_face_size_ratio,
        min_aspect_ratio=args.min_aspect_ratio,
        max_aspect_ratio=args.max_aspect_ratio,
        validate_landmarks=args.validate_landmarks,
    )
    
    metadata = load_json(f"{args.frames_dir}/metadata.json")
    
    rs = ResultsStore(args.rs_path)
    frame_manager = FrameManager(args.frames_dir, chunk_size=metadata["chunk_size"], cache_size=metadata["cache_size"])
    
    results = extractor.extract(frame_manager=frame_manager, frame_indices=metadata[name]["frame_indices"])
    
    rs.store(results, name=name)