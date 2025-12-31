

1. object_detection

    requirements:

        from typing import Any, Dict, List, Optional, Union, Tuple
        import cv2
        import numpy as np
        import torch
        from PIL import Image
        from collections import defaultdict
        from dataclasses import dataclass
        from transformers import OwlViTProcessor, OwlViTForObjectDetection, Owlv2Processor, Owlv2ForObjectDetection
        from sklearn.cluster import KMeans

    input:

        model-name    -
        model-family  -  
        device        -        
        box-threshold -
        frames-dir    -    
        meta-path     -     
        rs-path       -       

    output:

        {
            "frames": all_detections,  # Per-frame detections
            "frame_metadata": frame_metadata,  # Density, overlapping per frame
            "summary": {
                "total_detections"    : total_detections,
                "unique_categories"   : len(object_counts),
                "category_counts"     : object_counts,
                "semantic_tag_counts" : dict(semantic_tag_counts),
                "brand_detections"    : brand_detections,
                "avg_density"         : float(avg_density),
                "avg_overlap_ratio"   : float(avg_overlap_ratio),
            },
            "tracking"                : tracking_metrics if tracking_metrics else None,
            "tracks": [
                {
                    "track_id"        : track.track_id,
                    "label"           : track.label,
                    "first_frame"     : track.first_frame,
                    "last_frame"      : track.last_frame,
                    "duration_frames" : track.last_frame - track.first_frame + 1,
                    "duration_seconds": (track.last_frame - track.first_frame + 1) * frame_time,
                    "num_detections"  : len(track.frames),
                    "semantic_tags"   : track.semantic_tags or [],
                    "colors": [
                        {
                            "B": int(c[0]), 
                            "G": int(c[1]), 
                            "R": int(c[2])
                        } for c in (track.colors or [])
                    ]
                }
                for track in tracks
            ] if tracks else [],
            "frame_count": len(frame_indices),
        }

2. scene_classification

    requirements:

        from __future__ import annotations
        from pathlib import Path
        from typing import Any, Dict, List, Optional, Sequence, Union, Tuple
        from collections import defaultdict
        import cv2
        import numpy as np
        import requests
        import torch
        import torch.nn.functional as F
        from PIL import Image
        from torchvision import models, transforms
        import timm
        from transformers import CLIPProcessor, CLIPModel

    input:

        model_arch
        use_timm
        top_k
        batch_size
        device
        categories_path
        cache_dir
        gpu_memory_threshold
        log_metrics_every_n_frames
        input_size
        use_tta
        use_multi_crop
        temporal_smoothing
        smoothing_window
        enable_advanced_features
        use_clip_for_semantics

3. detalize_face_modules

    requirements:

        from __future__ import annotations
        import logging
        from typing import Any, Dict, List, Optional, Sequence
        from pathlib import Path
        import cv2
        import mediapipe as mp
        import numpy as np
        from modules import MODULE_REGISTRY
        from modules.base_module import FaceModule
        from utils import landmarks_to_ndarray, validate_face_landmarks, compute_bbox, extract_roi
        from utils.landmarks_utils import LANDMARKS
        import math
        from abc import ABC, abstractmethod
        from collections import defaultdict, deque

    input:

        modules
        max_faces
        refine_landmarks
        visualize
        visualize_dir
        show_landmarks
        min_detection_confidence
        min_tracking_confidence
        min_face_size
        max_face_size_ratio
        min_aspect_ratio
        max_aspect_ratio
        validate_landmarks

4. emotion_face

    requirements:

        import numpy as np
        from typing import List, Dict, Any, Tuple, Optional, TypeVar, Generic, Callable
        import math
        import gc
        import torch
        from contextlib import contextmanager
        from functools import wraps
        import sys
        import time
        import json
        from dataclasses import dataclass, field
        from pathlib import Path
        import traceback
        from collections import OrderedDict

        from insightface.app import FaceAnalysis
        from models.emonet.emonet.models.emonet import EmoNet

    input:

        ttl_enabled
        ttl_seconds
        cache_size_limit
        min_frames_ratio
        min_keyframes
        min_transitions
        min_diversity_threshold
        quality_threshold
        enable_structured_metrics
        log_memory_usage
        min_faces_threshold
        target_length
        max_retries
        default_threshold
        transition_threshold
        max_gap_seconds
        max_samples_per_segment
        det_size
        emo_path
        device