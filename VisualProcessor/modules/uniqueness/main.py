if __name__ == "__main__":
    import argparse
    import json
    import os
    import sys
    import numpy as np
    
    _path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    
    if _path not in sys.path:
        sys.path.append(_path)
    
    from utils.frame_manager import FrameManager
    from utils.results_store import ResultsStore
    from uniqueness import UniquenessModule
    
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    
    name = "uniqueness"
    
    def load_json(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise Exception(f"{name} | main | load_json | Ошибка при открытии файла {path}: {e}")
    
    from utils.logger import get_logger
    logger = get_logger(name)
    
    parser = argparse.ArgumentParser(
        description='Uniqueness Module - Computes uniqueness/novelty scores for video',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--frames-dir', type=str, required=True, help='Path to frames directory')
    parser.add_argument('--rs-path', type=str, default=None, help='Path to results store directory')
    parser.add_argument('--top-n', type=int, default=100, help='Number of top videos for comparison')
    parser.add_argument('--reference-embeddings-path', type=str, default=None, help='Path to reference video embeddings file (JSON)')
    parser.add_argument('--use-high-level-semantic', action='store_true', help='Use high-level semantic module results for video embedding')
    parser.add_argument('--use-similarity-metrics', action='store_true', help='Use similarity_metrics module results for similarity scores')
    parser.add_argument('--use-visual-features', action='store_true', help='Use visual features from other modules')
    parser.add_argument('--use-pacing-features', action='store_true', help='Use pacing features from video_pacing module')
    parser.add_argument('--use-text-features', action='store_true', help='Use text features from text_scoring module')
    parser.add_argument('--use-audio-features', action='store_true', help='Use audio features from audio processor')
    parser.add_argument('--use-behavior-features', action='store_true', help='Use behavior features from behavioral module')
    
    args = parser.parse_args()
    
    rs = ResultsStore(args.rs_path)
    
    metadata = load_json(f"{args.frames_dir}/metadata.json")
    
    frame_manager = FrameManager(
        args.frames_dir, 
        chunk_size=metadata["chunk_size"], 
        cache_size=metadata["cache_size"]
    )
    
    logger.info(f"VisualProcessor | {name} | main | Initializing UniquenessModule")
    
    uniqueness = UniquenessModule(top_n=args.top_n)
    
    # Try to load current video embedding and features from other modules
    video_embedding = None
    video_topics = None
    video_visual_features = None
    video_pacing_features = None
    video_text_features = None
    video_audio_features = None
    video_behavior_features = None
    video_events = None
    similarity_scores = None
    
    if args.use_high_level_semantic:
        try:
            hl_results_path = f"{args.rs_path}/high_level_semantic"
            if os.path.exists(hl_results_path):
                hl_files = [f for f in os.listdir(hl_results_path) if f.endswith('.json')]
                if hl_files:
                    hl_data = load_json(f"{hl_results_path}/{sorted(hl_files)[-1]}")
                    if 'video_embeddings' in hl_data and 'weighted_mean_embedding' in hl_data['video_embeddings']:
                        video_embedding = np.array(hl_data['video_embeddings']['weighted_mean_embedding'])
                    if 'features' in hl_data and 'topic_probabilities' in hl_data['features']:
                        video_topics = hl_data['features']['topic_probabilities']
                    if 'events' in hl_data:
                        video_events = hl_data['events'].get('event_timestamps', [])
                    logger.info(f"VisualProcessor | {name} | main | Loaded high-level semantic data")
        except Exception as e:
            logger.warning(f"VisualProcessor | {name} | main | Could not load high-level semantic data: {e}")
    
    if args.use_similarity_metrics:
        try:
            sim_results_path = f"{args.rs_path}/similarity_metrics"
            if os.path.exists(sim_results_path):
                sim_files = [f for f in os.listdir(sim_results_path) if f.endswith('.json')]
                if sim_files:
                    sim_data = load_json(f"{sim_results_path}/{sorted(sim_files)[-1]}")
                    if 'features' in sim_data:
                        similarity_scores = sim_data['features']
                    logger.info(f"VisualProcessor | {name} | main | Loaded similarity metrics data")
        except Exception as e:
            logger.warning(f"VisualProcessor | {name} | main | Could not load similarity metrics data: {e}")
    
    if args.use_visual_features:
        try:
            # Try to load from color_light or frames_composition modules
            color_results_path = f"{args.rs_path}/color_light"
            if os.path.exists(color_results_path):
                color_files = [f for f in os.listdir(color_results_path) if f.endswith('.json')]
                if color_files:
                    color_data = load_json(f"{color_results_path}/{sorted(color_files)[-1]}")
                    video_visual_features = {
                        'color_histogram': color_data.get('color_histogram', None),
                        'lighting_features': color_data.get('lighting_features', None),
                    }
                    logger.info(f"VisualProcessor | {name} | main | Loaded visual features")
        except Exception as e:
            logger.warning(f"VisualProcessor | {name} | main | Could not load visual features: {e}")
    
    if args.use_pacing_features:
        try:
            pacing_results_path = f"{args.rs_path}/video_pacing"
            if os.path.exists(pacing_results_path):
                pacing_files = [f for f in os.listdir(pacing_results_path) if f.endswith('.json')]
                if pacing_files:
                    pacing_data = load_json(f"{pacing_results_path}/{sorted(pacing_files)[-1]}")
                    video_pacing_features = {
                        'cut_rate': pacing_data.get('cut_rate', None),
                        'pacing_curve': pacing_data.get('pacing_curve', None),
                        'shot_duration_distribution': pacing_data.get('shot_duration_distribution', None),
                    }
                    logger.info(f"VisualProcessor | {name} | main | Loaded pacing features")
        except Exception as e:
            logger.warning(f"VisualProcessor | {name} | main | Could not load pacing features: {e}")
    
    if args.use_text_features:
        try:
            text_results_path = f"{args.rs_path}/text_scoring"
            if os.path.exists(text_results_path):
                text_files = [f for f in os.listdir(text_results_path) if f.endswith('.json')]
                if text_files:
                    text_data = load_json(f"{text_results_path}/{sorted(text_files)[-1]}")
                    video_text_features = {
                        'ocr_embedding': text_data.get('ocr_embedding', None),
                        'text_layout': text_data.get('text_layout', None),
                    }
                    logger.info(f"VisualProcessor | {name} | main | Loaded text features")
        except Exception as e:
            logger.warning(f"VisualProcessor | {name} | main | Could not load text features: {e}")
    
    # Load reference embeddings
    reference_embeddings = []
    reference_topics_list = []
    reference_visual_features_list = []
    reference_pacing_features_list = []
    reference_text_features_list = []
    reference_audio_features_list = []
    reference_behavior_features_list = []
    reference_events_list = []
    reference_videos_metadata = None
    
    if args.reference_embeddings_path and os.path.exists(args.reference_embeddings_path):
        try:
            ref_data = load_json(args.reference_embeddings_path)
            if isinstance(ref_data, list):
                reference_embeddings = [np.array(emb) for emb in ref_data]
            elif isinstance(ref_data, dict) and 'embeddings' in ref_data:
                reference_embeddings = [np.array(emb) for emb in ref_data['embeddings']]
            logger.info(f"VisualProcessor | {name} | main | Loaded {len(reference_embeddings)} reference embeddings")
        except Exception as e:
            logger.warning(f"VisualProcessor | {name} | main | Could not load reference embeddings: {e}")
    
    # If no video embedding available, create a dummy one (this should be replaced with actual embedding)
    if video_embedding is None:
        logger.warning(f"VisualProcessor | {name} | main | No video embedding found, creating dummy embedding")
        video_embedding = np.random.randn(512)  # Dummy embedding
    
    # If no reference embeddings, create empty list (novelty scores will be 1.0)
    if len(reference_embeddings) == 0:
        logger.warning(f"VisualProcessor | {name} | main | No reference embeddings found, uniqueness scores will be maximum")
    
    # Extract uniqueness metrics
    logger.info(f"VisualProcessor | {name} | main | Computing uniqueness metrics")
    
    result = uniqueness.extract_all(
        video_embedding=video_embedding,
        reference_embeddings=reference_embeddings,
        video_topics=video_topics,
        reference_topics_list=reference_topics_list if reference_topics_list else None,
        video_visual_features=video_visual_features,
        reference_visual_features_list=reference_visual_features_list if reference_visual_features_list else None,
        video_pacing_features=video_pacing_features,
        reference_pacing_features_list=reference_pacing_features_list if reference_pacing_features_list else None,
        video_audio_features=video_audio_features,
        reference_audio_features_list=reference_audio_features_list if reference_audio_features_list else None,
        video_text_features=video_text_features,
        reference_text_features_list=reference_text_features_list if reference_text_features_list else None,
        video_behavior_features=video_behavior_features,
        reference_behavior_features_list=reference_behavior_features_list if reference_behavior_features_list else None,
        video_events=video_events,
        reference_events_list=reference_events_list if reference_events_list else None,
        video_metadata=None,
        reference_videos_metadata=reference_videos_metadata,
        similarity_scores=similarity_scores
    )
    
    # Add metadata
    result['metadata'] = {
        'total_frames': metadata["total_frames"],
        'top_n': args.top_n,
        'reference_videos_count': len(reference_embeddings)
    }
    
    rs.store(result, name=name)
    logger.info(f"VisualProcessor | {name} | main | Results stored successfully")

