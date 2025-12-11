import json
import argparse
import os, sys

_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

if _path not in sys.path:
    sys.path.append(_path)

from modules.emotion_face.core.video_processor import VideoEmotionProcessor
from utils.frame_manager import FrameManager
from utils.results_store import ResultsStore

name = "emotion_face"

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

from utils.logger import get_logger
logger = get_logger(name)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--frames-dir",               type=str, required=True, help="")
    parser.add_argument("--rs-path",                  type=str)

    parser.add_argument("--min-frames-ratio",         type=float)
    parser.add_argument("--min-keyframes",            type=int)
    parser.add_argument("--min-transitions",          type=int)
    parser.add_argument("--min-diversity-threshold",  type=float)
    parser.add_argument("--quality-threshold",        type=float)
    parser.add_argument("--memory-threshold-low",     type=int)
    parser.add_argument("--batch-load-low",           type=int)
    parser.add_argument("--batch-process-low",        type=int)
    parser.add_argument("--memory-threshold-medium",  type=int)
    parser.add_argument("--batch-load-medium",        type=int)
    parser.add_argument("--batch-process-medium",     type=int)
    parser.add_argument("--memory-threshold-high",    type=int)
    parser.add_argument("--batch-load-high",          type=int)
    parser.add_argument("--batch-process-high",       type=int)
    parser.add_argument("--batch-load-very-high",     type=int)
    parser.add_argument("--batch-process-very-high",  type=int)
    parser.add_argument("--enable-structured-metrics",action="store_true")
    parser.add_argument("--min-faces-threshold",      type=int)
    parser.add_argument("--target-length",            type=int)
    parser.add_argument("--max-retries",              type=int)
    parser.add_argument("--transition-threshold",     type=float)
    parser.add_argument("--max-gap-seconds",          type=float)
    parser.add_argument("--max-samples-per-segment",  type=int)
    parser.add_argument("--emo-path",                 type=str)
    parser.add_argument("--device",                   type=str)
    
    args = parser.parse_args()
        
    processor = VideoEmotionProcessor(
        min_frames_ratio = args.min_frames_ratio,
        min_keyframes = args.min_keyframes,
        min_transitions = args.min_transitions,
        min_diversity_threshold = args.min_diversity_threshold,
        quality_threshold = args.quality_threshold,
        memory_threshold_low = args.memory_threshold_low,
        batch_load_low = args.batch_load_low,
        batch_process_low = args.batch_process_low,
        memory_threshold_medium = args.memory_threshold_medium,
        batch_load_medium = args.batch_load_medium,
        batch_process_medium = args.batch_process_medium,
        memory_threshold_high = args.memory_threshold_high,
        batch_load_high = args.batch_load_high,
        batch_process_high = args.batch_process_high,
        batch_load_very_high = args.batch_load_very_high,
        batch_process_very_high = args.batch_process_very_high,
        enable_structured_metrics = args.enable_structured_metrics,
        min_faces_threshold = args.min_faces_threshold,
        target_length = args.target_length,
        max_retries = args.max_retries,
        transition_threshold = args.transition_threshold,
        max_gap_seconds = args.max_gap_seconds,
        max_samples_per_segment = args.max_samples_per_segment,
        emo_path = args.emo_path,
        device = args.device
    )

    logger.info(f"VisualProcessor | {name} | main | VideoEmotionProcessor инициализирован")
    
    metadata = load_json(f"{args.frames_dir}/metadata.json")

    frame_manager = FrameManager(args.frames_dir, chunk_size=metadata["chunk_size"], cache_size=metadata["cache_size"])
    
    save_path = f"{args.rs_path}/{name}"

    result = processor.process(frame_manager, args.rs_path)
    
