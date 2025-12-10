import json
import argparse

from modules.emotion_face.core.video_processor import VideoEmotionProcessor
from utils.frame_manager import FrameManager
from utils.results_store import ResultsStore

name = "emotion_face"

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument("--frames-dir",               type=str, required=True, help="")
    parser.add_argument("--rs-path",                  type=str)

    parser.add_argument("--ttl-enabled",              type=str)
    parser.add_argument("--ttl-seconds",              type=str)
    parser.add_argument("--cache-size-limit",         type=str)
    parser.add_argument("--min-frames-ratio",         type=str)
    parser.add_argument("--min-keyframes",            type=str)
    parser.add_argument("--min-transitions",          type=str)
    parser.add_argument("--min-diversity-threshold",  type=str)
    parser.add_argument("--quality-threshold",        type=str)
    parser.add_argument("--enable-structured-metrics",type=str)
    parser.add_argument("--log-memory-usage",         type=str)
    parser.add_argument("--min-faces-threshold",      type=str)
    parser.add_argument("--target-length",            type=str)
    parser.add_argument("--max-retries",              type=str)
    parser.add_argument("--default-threshold",        type=str)
    parser.add_argument("--transition-threshold",     type=str)
    parser.add_argument("--max-gap-seconds",          type=str)
    parser.add_argument("--max-samples-per-segment",  type=str)
    parser.add_argument("--det-size",                 type=str)
    parser.add_argument("--emo-path",                 type=str)
    parser.add_argument("--device",                   type=str)
    
    args = parser.parse_args()
        
    processor = VideoEmotionProcessor(
        ttl_enabled = args.ttl_enabled,
        ttl_seconds = args.ttl_seconds,
        cache_size_limit = args.cache_size_limit,
        min_frames_ratio = args.min_frames_ratio,
        min_keyframes = args.min_keyframes,
        min_transitions = args.min_transitions,
        min_diversity_threshold = args.min_diversity_threshold,
        quality_threshold = args.quality_threshold,
        enable_structured_metrics = args.enable_structured_metrics,
        log_memory_usage = args.log_memory_usage,
        min_faces_threshold = args.min_faces_threshold,
        target_length = args.target_length,
        max_retries = args.max_retries,
        default_threshold = args.default_threshold,
        transition_threshold = args.transition_threshold,
        max_gap_seconds = args.max_gap_seconds,
        max_samples_per_segment = args.max_samples_per_segment,
        det_size = args.det_size,
        emo_path = args.emo_path,
        device = args.device
    )
    
    metadata = load_json(f"{args.frames_dir}/metadata.json")

    rs = ResultsStore(args.rs_path)
    frame_manager = FrameManager(args.frames_dir, chunk_size=metadata["chunk_size"], cache_dir=metadata["cache_dir"])
    
    result = processor.process(frame_manager)
    
    rs.store(result, name=name)
    
