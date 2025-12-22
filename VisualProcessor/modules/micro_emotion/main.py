if __name__ == "__main__":
    import argparse
    import json
import os
import sys
import numpy as np
    import cv2
from pathlib import Path
    import tempfile
    
    _path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
    
    if _path not in sys.path:
        sys.path.append(_path)
    
    from utils.frame_manager import FrameManager
    from utils.results_store import ResultsStore
    
    # Import OpenFaceAnalyzer from a separate module file
    # Check if openface_analyzer.py exists, otherwise use the class from main.py
    _module_path = os.path.dirname(__file__)
    openface_file = os.path.join(_module_path, "openface_analyzer.py")
    
    if os.path.exists(openface_file):
        from openface_analyzer import OpenFaceAnalyzer
    else:
        # Fallback: import from main.py using importlib to avoid circular import
        import importlib.util
        spec = importlib.util.spec_from_file_location("openface_module", os.path.join(_module_path, "main.py"))
        openface_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(openface_module)
        OpenFaceAnalyzer = openface_module.OpenFaceAnalyzer
    
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    
    name = "micro_emotion"
    
    def load_json(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise Exception(f"{name} | main | load_json | Ошибка при открытии файла {path}: {e}")
    
    from utils.logger import get_logger
    logger = get_logger(name)
    
    parser = argparse.ArgumentParser(
        description='Micro Emotion Module - Extracts micro-expressions and Action Units using OpenFace',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--frames-dir', type=str, required=True, help='Path to frames directory')
    parser.add_argument('--rs-path', type=str, default=None, help='Path to results store directory')
    parser.add_argument('--features', type=str, default='all', choices=['all', 'basic', 'au', 'pose', 'gaze'], help='Which features to extract')
    parser.add_argument('--batch-size', type=int, default=50, help='Batch size for processing frames')
    parser.add_argument('--use-face-detection', action='store_true', help='Use face detection results to filter frames')
    parser.add_argument('--docker-image', type=str, default='openface/openface:latest', help='Docker image for OpenFace')
    
    args = parser.parse_args()
    
    rs = ResultsStore(args.rs_path)
    
    metadata = load_json(f"{args.frames_dir}/metadata.json")
    
    frame_manager = FrameManager(
        args.frames_dir, 
        chunk_size=metadata["chunk_size"], 
        cache_size=metadata["cache_size"]
    )
    
    logger.info(f"VisualProcessor | {name} | main | Initializing OpenFaceAnalyzer")
    
    analyzer = OpenFaceAnalyzer(docker_image=args.docker_image)
        
    # Get frame indices - use face detection results if available
    frame_indices = list(range(metadata["total_frames"]))
    
    if args.use_face_detection:
        try:
            face_results_path = f"{args.rs_path}/face_detection"
            if os.path.exists(face_results_path):
                face_files = [f for f in os.listdir(face_results_path) if f.endswith('.json')]
                if face_files:
                    face_data = load_json(f"{face_results_path}/{sorted(face_files)[-1]}")
                    # Filter frames that have faces
                    if 'frames' in face_data:
                        frames_with_faces = [int(k) for k, v in face_data['frames'].items() if v and len(v) > 0]
                        frame_indices = sorted(set(frame_indices) & set(frames_with_faces))
                        logger.info(f"VisualProcessor | {name} | main | Filtered to {len(frame_indices)} frames with faces")
        except Exception as e:
            logger.warning(f"VisualProcessor | {name} | main | Could not load face detection data: {e}")
    
    # Process frames in batches
    logger.info(f"VisualProcessor | {name} | main | Processing {len(frame_indices)} frames in batches of {args.batch_size}")
        
    all_results = []
        
    for batch_start in range(0, len(frame_indices), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(frame_indices))
        batch_indices = frame_indices[batch_start:batch_end]
        
        logger.info(f"VisualProcessor | {name} | main | Processing batch {batch_start // args.batch_size + 1}/{(len(frame_indices) + args.batch_size - 1) // args.batch_size}")
        
        # Get frames
        frames = []
        for idx in batch_indices:
            try:
                frame = frame_manager.get_frame(idx)
                # Convert RGB to BGR for OpenCV
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                frames.append(frame_bgr)
            except Exception as e:
                logger.warning(f"VisualProcessor | {name} | main | Error loading frame {idx}: {e}")
                continue
        
        if len(frames) == 0:
            continue
        
        # Analyze frames using OpenFaceAnalyzer
        try:
            batch_results = analyzer.analyze_frames(
                frames=frames,
                frame_indices=batch_indices[:len(frames)],
                output_prefix=f"batch_{batch_start}"
            )
            
            if batch_results:
                all_results.extend(batch_results)
                logger.info(f"VisualProcessor | {name} | main | Processed {len(batch_results)} frames in batch")
        except Exception as e:
            logger.error(f"VisualProcessor | {name} | main | Error processing batch: {e}")
            continue
    
    logger.info(f"VisualProcessor | {name} | main | Processed {len(all_results)} frames total")
    
    # Try to use optimized processor if DataFrame is available
    result = None
    try:
        from micro_emotion_processor import MicroEmotionProcessor
        import pandas as pd
        
        # Try to get DataFrame from results
        df = None
        csv_paths = []
        
        # Collect CSV paths from all batch results
        for res in all_results:
            if isinstance(res, dict):
                if 'csv_path' in res and res['csv_path']:
                    csv_paths.append(res['csv_path'])
                elif 'dataframe' in res and res['dataframe'] is not None:
                    if df is None:
                        df = res['dataframe']
                    else:
                        df = pd.concat([df, res['dataframe']], ignore_index=True)
        
        # Try to load from CSV if DataFrame not available
        if df is None and csv_paths:
            # Use the last CSV (should contain all frames if OpenFace concatenates)
            csv_path = csv_paths[-1]
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                logger.info(f"VisualProcessor | {name} | main | Loaded DataFrame from {csv_path}, shape: {df.shape}")
        
        # If we have DataFrame, use optimized processor
        if df is not None and len(df) > 0:
            logger.info(f"VisualProcessor | {name} | main | Using optimized MicroEmotionProcessor")
            processor = MicroEmotionProcessor(fps=metadata.get('fps', 30))
            processed = processor.process_openface_dataframe(df, fit_models=True)
            
            # Create result structure
            frames_with_face = int(df['success'].sum()) if 'success' in df.columns else len(df)
            result = {
                'success': processed['success'] and frames_with_face > 0,
                'face_count': frames_with_face,
                'success_rate': float(df['success'].mean()) if 'success' in df.columns else 1.0,
                'features': processed['features'],
                'per_frame_vectors': processed['per_frame_vectors'].tolist(),
                'reliability_flags': processed['reliability_flags'],
                'microexpr_features': processed['microexpr_features'],
                'summary': {
                    'total_frames': len(frame_indices),
                    'frames_processed': len(df),
                    'frames_with_face': frames_with_face,
                    'au_count': len([k for k in processed['features'].keys() if k.startswith('AU')]),
                    'landmarks_2d_count': 68,
                    'landmarks_3d_count': 68,
                },
                'metadata': {
                    'features_extracted': args.features,
                    'batch_size': args.batch_size,
                    'docker_image': args.docker_image,
                    'processing_mode': 'optimized',
                }
            }
    except Exception as e:
        logger.warning(f"VisualProcessor | {name} | main | Could not use optimized processor: {e}, falling back to original")
        result = None
    
    # Fallback to original aggregation if optimized processor failed
    if result is None:
        # Aggregate results (original code)
        if len(all_results) == 0:
        logger.warning(f"VisualProcessor | {name} | main | No results extracted")
        result = {
            'success': False,
            'face_count': 0,
            'action_units': {},
            'pose': {},
            'gaze': {},
            'facial_landmarks_2d': [],
            'facial_landmarks_3d': [],
            'summary': {
                'total_frames': len(frame_indices),
                'frames_with_face': 0,
                'au_count': 0,
                'landmarks_2d_count': 0,
                'landmarks_3d_count': 0
            }
        }
    else:
        # Aggregate action units
        action_units = {}
        pose_data = {}
        gaze_data = {}
        landmarks_2d_all = []
        landmarks_3d_all = []
        
        frames_with_face = 0
        
        for res in all_results:
            if res.get('success', False):
                frames_with_face += 1
                
                # Aggregate AU
                if 'action_units' in res:
                    for au_name, au_data in res['action_units'].items():
                        if au_name not in action_units:
                            action_units[au_name] = {
                                'intensity_mean': [],
                                'intensity_std': [],
                                'presence_mean': [],
                                'presence_std': []
                            }
                        action_units[au_name]['intensity_mean'].append(au_data.get('intensity_mean', 0))
                        action_units[au_name]['intensity_std'].append(au_data.get('intensity_std', 0))
                        action_units[au_name]['presence_mean'].append(au_data.get('presence_mean', 0))
                        action_units[au_name]['presence_std'].append(au_data.get('presence_std', 0))
                
                # Aggregate pose
                if 'pose' in res:
                    for pose_key, pose_val in res['pose'].items():
                        if pose_key not in pose_data:
                            pose_data[pose_key] = {'mean': [], 'std': [], 'min': [], 'max': []}
                        pose_data[pose_key]['mean'].append(pose_val.get('mean', 0))
                        pose_data[pose_key]['std'].append(pose_val.get('std', 0))
                        pose_data[pose_key]['min'].append(pose_val.get('min', 0))
                        pose_data[pose_key]['max'].append(pose_val.get('max', 0))
                
                # Aggregate gaze
                if 'gaze' in res:
                    for gaze_key, gaze_val in res['gaze'].items():
                        if gaze_key not in gaze_data:
                            gaze_data[gaze_key] = {'mean': [], 'std': []}
                        gaze_data[gaze_key]['mean'].append(gaze_val.get('mean', 0))
                        gaze_data[gaze_key]['std'].append(gaze_val.get('std', 0))
        
        # Compute final aggregated values
        for au_name in action_units:
            action_units[au_name] = {
                'intensity_mean': float(np.mean(action_units[au_name]['intensity_mean'])) if action_units[au_name]['intensity_mean'] else 0.0,
                'intensity_std': float(np.mean(action_units[au_name]['intensity_std'])) if action_units[au_name]['intensity_std'] else 0.0,
                'presence_mean': float(np.mean(action_units[au_name]['presence_mean'])) if action_units[au_name]['presence_mean'] else 0.0,
                'presence_std': float(np.mean(action_units[au_name]['presence_std'])) if action_units[au_name]['presence_std'] else 0.0
            }
        
        for pose_key in pose_data:
            pose_data[pose_key] = {
                'mean': float(np.mean(pose_data[pose_key]['mean'])) if pose_data[pose_key]['mean'] else 0.0,
                'std': float(np.mean(pose_data[pose_key]['std'])) if pose_data[pose_key]['std'] else 0.0,
                'min': float(np.min(pose_data[pose_key]['min'])) if pose_data[pose_key]['min'] else 0.0,
                'max': float(np.max(pose_data[pose_key]['max'])) if pose_data[pose_key]['max'] else 0.0
            }
        
        for gaze_key in gaze_data:
            gaze_data[gaze_key] = {
                'mean': float(np.mean(gaze_data[gaze_key]['mean'])) if gaze_data[gaze_key]['mean'] else 0.0,
                'std': float(np.mean(gaze_data[gaze_key]['std'])) if gaze_data[gaze_key]['std'] else 0.0
            }
        
        result = {
            'success': frames_with_face > 0,
            'face_count': frames_with_face,
            'success_rate': float(frames_with_face / len(all_results)) if len(all_results) > 0 else 0.0,
            'action_units': action_units,
            'pose': pose_data,
            'gaze': gaze_data,
            'facial_landmarks_2d': landmarks_2d_all[:68] if landmarks_2d_all else [],
            'facial_landmarks_3d': landmarks_3d_all[:68] if landmarks_3d_all else [],
            'summary': {
                'total_frames': len(frame_indices),
                'frames_processed': len(all_results),
                'frames_with_face': frames_with_face,
                'au_count': len(action_units),
                'landmarks_2d_count': len(landmarks_2d_all),
                'landmarks_3d_count': len(landmarks_3d_all)
            },
            'metadata': {
                'features_extracted': args.features,
                'batch_size': args.batch_size,
                'docker_image': args.docker_image
            }
        }
    
    rs.store(result, name=name)
    logger.info(f"VisualProcessor | {name} | main | Results stored successfully")