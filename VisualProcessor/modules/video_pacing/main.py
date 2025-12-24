if __name__ == "__main__":
    import argparse
    import json
    
    import os
    import sys
    _path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

    if _path not in sys.path:
        sys.path.append(_path)

    from utils.frame_manager import FrameManager
    from utils.results_store import ResultsStore

    from video_pacing import VideoPacingPipelineVisualOptimized

    name = "video_pacing"

    def load_json(meta_path):
        try:
            with open(meta_path, "r") as f:
                return json.load(f)
        except:
            raise f"{name} | main | load_json | Ошибка при открытии метадаты"

    parser = argparse.ArgumentParser(
        description='Video Pacing Module - Extracts pacing metrics from video (visual + optional audio/person/object data)'
    )

    parser.add_argument('--frames-dir', type=str, required=True, help='')
    parser.add_argument('--rs-path', type=str, default=None, help='')
    parser.add_argument('--batch-size', type=int, default=None, help='')
    parser.add_argument('--downscale-factor', type=float, default=None, help='')
    
    args = parser.parse_args()

    rs = ResultsStore(args.rs_path)

    metadata = load_json(f"{args.frames_dir}/metadata.json")

    frame_manager = FrameManager(args.frames_dir, chunk_size=metadata["chunk_size"], cache_size=metadata["cache_size"])
    frame_indices = metadata[name]["frame_indices"]

#    # Загружаем дополнительные данные
#     person_tracks = load_json(args.person_tracks) if args.person_tracks else None
#     person_keypoints = load_json(args.person_keypoints) if args.person_keypoints else None
#     object_detections = load_json(args.object_detections) if args.object_detections else None

    # Инициализация pipeline
    pipeline = VideoPacingPipelineVisualOptimized(
        frame_manager=frame_manager,
        frame_indices=frame_indices,
        batch_size=args.batch_size,
        downscale_factor=args.downscale_factor,
        rs_path=args.rs_path,
    )

    # Извлечение всех метрик
    result = pipeline.extract_all_features()

    rs.store(result, name=name)
 