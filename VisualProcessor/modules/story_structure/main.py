import argparse
import json

import os
import sys
_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

if _path not in sys.path:
    sys.path.append(_path)

from utils.frame_manager import FrameManager
from utils.results_store import ResultsStore

from story_structure import StoryStructurePipelineOptimized

name = "story_structure"

def load_json(meta_path):
    try:
        with open(meta_path, "r") as f:
            return json.load(f)
    except:
        raise f"{name} | main | load_json | Ошибка при открытии метадаты"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Video Pacing Module - Extracts pacing metrics from video (visual + optional audio/person/object data)'
    )

    parser.add_argument('--frames-dir', type=str, required=True, help='')
    parser.add_argument('--rs-path', type=str, default=None, help='')
    parser.add_argument('--clip-model', type=str, default=None, help='')
    parser.add_argument('--sentence-model', type=str, default=None, help='')
    parser.add_argument('--subtitles', type=str, default=None, help='')

    args = parser.parse_args()

    subtitles = args.subtitles.split(",")

    rs = ResultsStore(args.rs_path)

    metadata = load_json(f"{args.frames_dir}/metadata.json")

    frame_manager = FrameManager(args.frames_dir, chunk_size=metadata["chunk_size"], cache_size=metadata["cache_size"])
    frame_indices = metadata[name]["frame_indices"]

    pipeline = StoryStructurePipelineOptimized(
        frame_manager=frame_manager,
        frame_indices=frame_indices,
        clip_model=args.clip_model,
        sentence_model=args.sentence_model,
        rs_path=args.rs_path,
    )

    result = pipeline.extract_all_features(subtitles=subtitles)

    rs.store(result, name=name)
