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
    from text_scoring import TextVideoInteractionPipeline
    
    import warnings
    warnings.filterwarnings("ignore", category=UserWarning)
    
    name = "text_scoring"
    
    def load_json(path):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            raise Exception(f"{name} | main | load_json | Ошибка при открытии файла {path}: {e}")
    
    from utils.logger import get_logger
    logger = get_logger(name)
    
    parser = argparse.ArgumentParser(
        description='Text Scoring Module - Extracts text features from video frames using OCR',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--frames-dir', type=str, required=True, help='Path to frames directory')
    parser.add_argument('--rs-path', type=str, default=None, help='Path to results store directory')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use (cuda/cpu)')
    parser.add_argument('--ocr-model', type=str, default='easyocr', choices=['easyocr', 'pytesseract'], help='OCR model to use')
    parser.add_argument('--batch-size', type=int, default=8, help='Batch size for OCR processing')
    parser.add_argument('--use-motion-data', action='store_true', help='Use motion data from optical_flow module')
    parser.add_argument('--use-face-data', action='store_true', help='Use face data from emotion_face module')
    parser.add_argument('--use-audio-data', action='store_true', help='Use audio data from audio processor')
    
    args = parser.parse_args()
    
    rs = ResultsStore(args.rs_path)
    
    metadata = load_json(f"{args.frames_dir}/metadata.json")
    
    frame_manager = FrameManager(
        args.frames_dir, 
        chunk_size=metadata["chunk_size"], 
        cache_size=metadata["cache_size"]
    )
    
    fps = metadata.get("fps", 30)
    
    logger.info(f"VisualProcessor | {name} | main | Initializing TextVideoInteractionPipeline")
    
    pipeline = TextVideoInteractionPipeline(video_fps=fps)
    
    # Try to load additional data from other modules if available
    seg_root = "/".join(args.frames_dir.split("/")[:-1])

    # -------- core‑данные: motion / face / audio ----------
    motion_peaks = None
    face_peaks = None
    audio_peaks = None

    def _load_core_optical_flow(rs_path: str) -> list | None:
        """
        Загружает motion данные из core_optical_flow - обязательное требование.
        """
        if not rs_path:
            return None
        
        # Используем optical_flow модуль (который работает как core провайдер через fallback в main.py)
        stats_path = os.path.join(rs_path, "optical_flow", "statistical_analysis.json")
        if not os.path.isfile(stats_path):
            return None
        
        try:
                motion_data = load_json(stats_path)
                frame_stats = (motion_data.get("statistics") or {}).get("frame_statistics") or []
                if frame_stats:
                peaks = []
                    for fs in frame_stats:
                        v = (
                            fs.get("magnitude_mean_px_sec_norm")
                            if "magnitude_mean_px_sec_norm" in fs
                            else fs.get("magnitude_mean_px_sec", fs.get("magnitude_mean", 0.0))
                        )
                    peaks.append(float(v))
                logger.info(f"VisualProcessor | {name} | main | Loaded motion data from optical_flow")
                return peaks
        except Exception as e:
            logger.warning(f"VisualProcessor | {name} | main | Could not load from optical_flow: {e}")
        
        return None

    def _load_core_face_landmarks(rs_path: str) -> list | None:
        """
        Загружает face данные из core_face_landmarks.
        Извлекает простую метрику присутствия лиц (количество лиц на кадр).
        """
        if not rs_path:
            return None
        
        core_path = os.path.join(rs_path, "core_face_landmarks", "landmarks.json")
        if not os.path.isfile(core_path):
            return None
        
        try:
            data = load_json(core_path)
            frames = data.get("frames") or []
            if not frames:
                return None
            
            # Извлекаем метрику: количество лиц на кадр (нормализованное)
            peaks = []
            max_faces = 0
            for f in frames:
                face_count = len(f.get("face_landmarks", []))
                max_faces = max(max_faces, face_count)
                peaks.append(float(face_count))
            
            # Нормализуем к 0..1
            if max_faces > 0:
                peaks = [p / float(max_faces) for p in peaks]
            
            logger.info(f"VisualProcessor | {name} | main | Loaded face data from core_face_landmarks")
            return peaks
        except Exception as e:
            logger.warning(f"VisualProcessor | {name} | main | Could not load from core_face_landmarks: {e}")
            return None

    if args.use_motion_data:
        motion_peaks = _load_core_optical_flow(args.rs_path)
        if motion_peaks is None:
            raise RuntimeError(
                f"VisualProcessor | {name} | main | core_optical_flow не найден. "
                f"Убедитесь, что core провайдер optical_flow запущен перед этим модулем. "
                f"rs_path: {args.rs_path}"
            )

    if args.use_face_data:
        # Используем только core_face_landmarks - обязательное требование
        face_peaks = _load_core_face_landmarks(args.rs_path)
        if face_peaks is None:
            raise RuntimeError(
                f"VisualProcessor | {name} | main | core_face_landmarks не найдены. "
                f"Убедитесь, что core провайдер core_face_landmarks запущен перед этим модулем. "
                f"rs_path: {args.rs_path}"
            )

    if args.use_audio_data:
        # core_audio_embeddings ещё не реализован; оставляем заглушку под будущий провайдер.
        try:
            audio_path = os.path.join(seg_root, "audio")
            if os.path.exists(audio_path):
                logger.info(f"VisualProcessor | {name} | main | Audio data path found (core_audio_embeddings TBD)")
        except Exception as e:
            logger.warning(f"VisualProcessor | {name} | main | Could not load audio data: {e}")
    
    # Extract OCR data from frames
    logger.info(f"VisualProcessor | {name} | main | Extracting OCR data from frames")
    
    try:
        import easyocr
        reader = easyocr.Reader(['en', 'ru'], gpu=(args.device == 'cuda'))
    except ImportError:
        logger.error("VisualProcessor | {name} | main | easyocr not installed. Install with: pip install easyocr")
        raise
    
    ocr_data = []
    frame_indices = list(range(metadata["total_frames"]))
    
    # Process frames in batches
    for batch_start in range(0, len(frame_indices), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(frame_indices))
        batch_indices = frame_indices[batch_start:batch_end]
        
        frames = []
        for idx in batch_indices:
            frame = frame_manager.get_frame(idx)
            frames.append(frame)
        
        # Run OCR on batch
        for i, (frame, idx) in enumerate(zip(frames, batch_indices)):
            try:
                results = reader.readtext(frame)
                for detection in results:
                    bbox = detection[0]  # [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
                    text = detection[1]
                    confidence = detection[2]
                    
                    # Convert bbox to (x1, y1, x2, y2) format
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[1] for point in bbox]
                    x1, y1 = min(x_coords), min(y_coords)
                    x2, y2 = max(x_coords), max(y_coords)
                    
                    # Simple CTA detection
                    cta_keywords = ['subscribe', 'follow', 'like', 'link in bio', 'click', 'watch']
                    is_cta = any(keyword in text.lower() for keyword in cta_keywords)
                    
                    ocr_data.append({
                        "frame": idx,
                        "bbox": (int(x1), int(y1), int(x2), int(y2)),
                        "text": text,
                        "confidence": float(confidence),
                        "is_cta": is_cta
                    })
            except Exception as e:
                logger.warning(f"VisualProcessor | {name} | main | Error processing frame {idx}: {e}")
                continue
        
        if (batch_start // args.batch_size + 1) % 10 == 0:
            logger.info(f"VisualProcessor | {name} | main | Processed {batch_end}/{len(frame_indices)} frames")
    
    logger.info(f"VisualProcessor | {name} | main | Extracted {len(ocr_data)} OCR detections")
    
    # Prepare motion/face/audio peaks (default to zeros if not available)
    if motion_peaks is None:
        motion_peaks = [0.0] * metadata["total_frames"]
    if face_peaks is None:
        face_peaks = [0.0] * metadata["total_frames"]
    if audio_peaks is None:
        audio_peaks = None
    
    # Extract features
    logger.info(f"VisualProcessor | {name} | main | Extracting text scoring features")
    result = pipeline.extract_features(
        ocr_data=ocr_data,
        motion_peaks=motion_peaks,
        face_peaks=face_peaks,
        audio_peaks=audio_peaks
    )
    
    # Add metadata
    result['metadata'] = {
        'total_frames': metadata["total_frames"],
        'fps': fps,
        'ocr_detections_count': len(ocr_data),
        'device': args.device
    }
    
    rs.store(result, name=name)
    logger.info(f"VisualProcessor | {name} | main | Results stored successfully")

