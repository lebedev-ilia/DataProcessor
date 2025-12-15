if __name__ == "__main__":
    import argparse
    import json

    from cut_detection import CutDetectionPipeline

    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)

    import os
    import sys
    _path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    if _path not in sys.path:
        sys.path.append(_path)

    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # 0=all, 1=INFO, 2=WARNING, 3=ERROR

    from utils.frame_manager import FrameManager
    from utils.results_store import ResultsStore

    name = "cut_detection"

    def load_metadata(meta_path):
        try:
            with open(meta_path, "r") as f:
                return json.load(f)
        except:
            raise f"{name} | main | load_metadata | Ошибка при открытии метадаты"
    
    parser = argparse.ArgumentParser(description='Cut Detection Module - Detects cuts, transitions, and analyzes editing style')

    parser.add_argument('--frames-dir', type=str, required=True, help='Путь к входному видео файлу')
    parser.add_argument('--rs-path', type=str, default=None, help='Путь к выходному JSON файлу с результатами (по умолчанию: input_name_shot_quality.json)')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'], help='Device to use (default: auto)')
    parser.add_argument('--use-clip', action='store_true', help='Disable CLIP-based transition classification')
    parser.add_argument('--use-deep-features', action='store_true', help='Disable deep feature extraction')

    args = parser.parse_args()

    rs = ResultsStore(args.rs_path)

    metadata = load_metadata(f"{args.frames_dir}/metadata.json")

    pipeline = CutDetectionPipeline(
        fps=metadata["fps"],
        device=args.device, 
        clip_zero_shot=args.use_clip,
        use_deep_features=args.use_deep_features,
        use_adaptive_thresholds=True,
        use_semantic_clustering=True
    )

    frame_manager = FrameManager(args.frames_dir, chunk_size=metadata["chunk_size"], cache_size=metadata["cache_size"])
    frame_indices = metadata[name]["frame_indices"]
    
    seg_root = "/".join(args.frames_dir.split("/")[:-1])

    audio_path = f"{seg_root}/audio/NSumhkOwSg.wav"

    result = pipeline.process_video_frames(frame_manager, frame_indices, audio_path=audio_path)

    rs.store(result, name=name)