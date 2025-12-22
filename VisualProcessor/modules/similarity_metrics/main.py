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
    from similarity_metrics import SimilarityMetrics
    
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    
    name = "similarity_metrics"
    
    def load_json(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise Exception(f"{name} | main | load_json | Ошибка при открытии файла {path}: {e}")
    
    from utils.logger import get_logger
    logger = get_logger(name)
    
    parser = argparse.ArgumentParser(
        description='Similarity Metrics Module - Computes similarity metrics between current video and reference videos',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--frames-dir', type=str, required=True, help='Path to frames directory')
    parser.add_argument('--rs-path', type=str, default=None, help='Path to results store directory')
    parser.add_argument('--top-n', type=int, default=10, help='Number of top videos for averaging metrics')
    parser.add_argument('--reference-embeddings-path', type=str, default=None, help='Path to reference video embeddings file (JSON)')
    parser.add_argument('--use-high-level-semantic', action='store_true', help='Use high-level semantic module results for video embedding')
    parser.add_argument('--use-text-scoring', action='store_true', help='Use text scoring module results for text features')
    parser.add_argument('--use-visual-features', action='store_true', help='Use visual features from other modules')
    
    args = parser.parse_args()
    
    rs = ResultsStore(args.rs_path)
    
    metadata = load_json(f"{args.frames_dir}/metadata.json")
    
    frame_manager = FrameManager(
        args.frames_dir, 
        chunk_size=metadata["chunk_size"], 
        cache_size=metadata["cache_size"]
    )
    
    logger.info(f"VisualProcessor | {name} | main | Initializing SimilarityMetrics")
    
    similarity = SimilarityMetrics(top_n=args.top_n)
    
    # Try to load current video embedding from high-level semantic module
    video_embedding = None
    video_topics = None
    video_visual_features = None
    video_text_features = None
    
    if args.use_high_level_semantic:
        try:
            hl_results_path = f"{args.rs_path}/high_level_semantic"
            if os.path.exists(hl_results_path):
                hl_files = [f for f in os.listdir(hl_results_path) if f.endswith('.json')]
                if hl_files:
                    hl_data = load_json(f"{hl_results_path}/{sorted(hl_files)[-1]}")
                    if 'video_embeddings' in hl_data and 'weighted_mean_embedding' in hl_data['video_embeddings']:
                        video_embedding = np.array(hl_data['video_embeddings']['weighted_mean_embedding'])
                    elif 'features' in hl_data and 'video_embedding_norm_weighted' in hl_data['features']:
                        # Fallback: try to reconstruct from features
                        logger.warning(f"VisualProcessor | {name} | main | Using fallback embedding reconstruction")
                    if 'features' in hl_data and 'topic_probabilities' in hl_data['features']:
                        video_topics = hl_data['features']['topic_probabilities']
                    logger.info(f"VisualProcessor | {name} | main | Loaded high-level semantic data")
        except Exception as e:
            logger.warning(f"VisualProcessor | {name} | main | Could not load high-level semantic data: {e}")
    
    if args.use_text_scoring:
        try:
            text_results_path = f"{args.rs_path}/text_scoring"
            if os.path.exists(text_results_path):
                text_files = [f for f in os.listdir(text_results_path) if f.endswith('.json')]
                if text_files:
                    text_data = load_json(f"{text_results_path}/{sorted(text_files)[-1]}")
                    # Extract text features
                    video_text_features = {
                        'text_timing': text_data.get('text_switch_rate', 0.0),
                        'text_layout': None,  # Would need to extract from OCR data
                    }
                    logger.info(f"VisualProcessor | {name} | main | Loaded text scoring data")
        except Exception as e:
            logger.warning(f"VisualProcessor | {name} | main | Could not load text scoring data: {e}")
    
    # Load reference embeddings
    reference_embeddings = []
    reference_topics_list = []
    
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
    
    # If no reference embeddings, create empty list (metrics will return zeros)
    if len(reference_embeddings) == 0:
        logger.warning(f"VisualProcessor | {name} | main | No reference embeddings found, similarity metrics will be zero")
    
    # Extract similarity metrics
    logger.info(f"VisualProcessor | {name} | main | Computing similarity metrics")
    
    result = similarity.extract_all(
        video_embedding=video_embedding,
        reference_embeddings=reference_embeddings,
        video_topics=video_topics,
        reference_topics_list=reference_topics_list if reference_topics_list else None,
        video_visual_features=video_visual_features,
        reference_visual_features_list=None,
        video_text_features=video_text_features,
        reference_text_features_list=None,
        video_audio_features=None,
        reference_audio_features_list=None,
        video_emotion_features=None,
        reference_emotion_features_list=None,
        video_pacing_features=None,
        reference_pacing_features_list=None,
        reference_videos_metadata=None
    )
    
    # Add metadata
    result['metadata'] = {
        'total_frames': metadata["total_frames"],
        'top_n': args.top_n,
        'reference_videos_count': len(reference_embeddings)
    }
    
    rs.store(result, name=name)
    logger.info(f"VisualProcessor | {name} | main | Results stored successfully")

