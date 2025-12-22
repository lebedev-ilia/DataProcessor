if __name__ == "__main__":
    import argparse
    import json
    import os
    import sys
    import numpy as np
    from PIL import Image
    
    _path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    
    if _path not in sys.path:
        sys.path.append(_path)
    
    from utils.frame_manager import FrameManager
    from utils.results_store import ResultsStore
    from hl_semantic import HighLevelSemanticsOptimized
    
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    
    name = "high_level_semantic"
    
    def load_json(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise Exception(f"{name} | main | load_json | Ошибка при открытии файла {path}: {e}")
    
    from utils.logger import get_logger
    logger = get_logger(name)
    
    parser = argparse.ArgumentParser(
        description='High-Level Semantic Module - Extracts high-level semantic features from video',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--frames-dir', type=str, required=True, help='Path to frames directory')
    parser.add_argument('--rs-path', type=str, default=None, help='Path to results store directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--clip-model-name', type=str, default='ViT-B/32', help='CLIP model name')
    parser.add_argument('--clip-batch-size', type=int, default=64, help='Batch size for CLIP processing')
    parser.add_argument('--use-face-data', action='store_true', help='Use face emotion data from emotion_face module')
    parser.add_argument('--use-audio-data', action='store_true', help='Use audio data from audio processor')
    parser.add_argument('--use-cut-data', action='store_true', help='Use cut detection data for scene boundaries')
    parser.add_argument('--class-prompts', type=str, default=None, help='Comma-separated list of class prompts for zero-shot classification')
    
    args = parser.parse_args()
    
    rs = ResultsStore(args.rs_path)
    
    metadata = load_json(f"{args.frames_dir}/metadata.json")
    
    frame_manager = FrameManager(
        args.frames_dir, 
        chunk_size=metadata["chunk_size"], 
        cache_size=metadata["cache_size"]
    )
    
    fps = metadata.get("fps", 30)
    
    logger.info(f"VisualProcessor | {name} | main | Initializing HighLevelSemanticsOptimized")
    
    processor = HighLevelSemanticsOptimized(
        device=args.device,
        clip_model_name=args.clip_model_name,
        clip_batch_size=args.clip_batch_size,
        fps=fps
    )
    
    # Try to load additional data from other modules if available
    seg_root = "/".join(args.frames_dir.split("/")[:-1])
    face_emotion_curve = None
    audio_energy_curve = None
    scene_boundary_frames = None
    
    if args.use_face_data:
        try:
            face_results_path = f"{args.rs_path}/emotion_face"
            if os.path.exists(face_results_path):
                face_files = [f for f in os.listdir(face_results_path) if f.endswith('.json')]
                if face_files:
                    face_data = load_json(f"{face_results_path}/{sorted(face_files)[-1]}")
                    if 'emotion_curve' in face_data:
                        face_emotion_curve = np.array(face_data['emotion_curve'])
                    logger.info(f"VisualProcessor | {name} | main | Loaded face emotion data")
        except Exception as e:
            logger.warning(f"VisualProcessor | {name} | main | Could not load face data: {e}")
    
    if args.use_audio_data:
        try:
            audio_path = f"{seg_root}/audio"
            if os.path.exists(audio_path):
                # Audio processing would go here
                # For now, we'll skip it
                logger.info(f"VisualProcessor | {name} | main | Audio data path found but not processed yet")
        except Exception as e:
            logger.warning(f"VisualProcessor | {name} | main | Could not load audio data: {e}")
    
    if args.use_cut_data:
        try:
            cut_results_path = f"{args.rs_path}/cut_detection"
            if os.path.exists(cut_results_path):
                cut_files = [f for f in os.listdir(cut_results_path) if f.endswith('.json')]
                if cut_files:
                    cut_data = load_json(f"{cut_results_path}/{sorted(cut_files)[-1]}")
                    # Extract scene boundaries if available
                    if 'scene_boundaries' in cut_data:
                        scene_boundary_frames = cut_data['scene_boundaries']
                    elif 'cuts' in cut_data:
                        # Extract frame indices from cuts
                        scene_boundary_frames = [cut.get('frame', 0) for cut in cut_data['cuts']]
                    logger.info(f"VisualProcessor | {name} | main | Loaded cut detection data")
        except Exception as e:
            logger.warning(f"VisualProcessor | {name} | main | Could not load cut data: {e}")
    
    # Get scene frames - use cut boundaries if available, otherwise sample uniformly
    logger.info(f"VisualProcessor | {name} | main | Extracting scene frames")
    
    total_frames = metadata["total_frames"]
    
    if scene_boundary_frames and len(scene_boundary_frames) > 0:
        # Use cut boundaries to define scenes
        scene_boundary_frames = sorted(set([0] + scene_boundary_frames + [total_frames - 1]))
        scene_frames = []
        for i in range(len(scene_boundary_frames) - 1):
            start_frame = scene_boundary_frames[i]
            end_frame = scene_boundary_frames[i + 1]
            # Take middle frame of each scene
            mid_frame = (start_frame + end_frame) // 2
            frame = frame_manager.get_frame(mid_frame)
            scene_frames.append(Image.fromarray(frame))
    else:
        # Sample uniformly - take one frame per ~2 seconds
        sample_rate = max(1, int(fps * 2))
        scene_frames = []
        for frame_idx in range(0, total_frames, sample_rate):
            frame = frame_manager.get_frame(frame_idx)
            scene_frames.append(Image.fromarray(frame))
    
    logger.info(f"VisualProcessor | {name} | main | Extracted {len(scene_frames)} scene frames")
    
    # Parse class prompts
    class_prompts = None
    if args.class_prompts:
        class_prompts = [p.strip() for p in args.class_prompts.split(',')]
    
    # Extract features
    logger.info(f"VisualProcessor | {name} | main | Extracting high-level semantic features")
    
    result = processor.extract_all(
        scene_frames=scene_frames,
        scene_embeddings=None,
        face_emotion_curve=face_emotion_curve,
        audio_energy_curve=audio_energy_curve,
        pose_activity_curve=None,
        text_features=None,
        topic_vectors=None,
        class_prompts=class_prompts,
        scene_boundary_frames=scene_boundary_frames
    )
    
    # Add metadata
    result['metadata'] = {
        'total_frames': total_frames,
        'fps': fps,
        'n_scenes': len(scene_frames),
        'device': args.device,
        'clip_model': args.clip_model_name
    }
    
    rs.store(result, name=name)
    logger.info(f"VisualProcessor | {name} | main | Results stored successfully")

