from utils import print_memory_usage

import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np
import cv2
import sys
import gc
import os
import shutil
import json
from pathlib import Path
from typing import List, Tuple, Optional
from numpy.lib.format import open_memmap

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CHUNK_SIZE = 64

# Для segmentation
DEFAULT_MAX_GAP_SECONDS = 0.5  # 500 мс

# Для select_from_segments
DEFAULT_SHORT_THRESHOLD_SEC = 1.0    # < 1 сек - короткий
DEFAULT_MEDIUM_THRESHOLD_SEC = 5.0   # 1-5 сек - средний, >5 - длинный
DEFAULT_MAX_SAMPLES_PER_SEGMENT = 10

# Для detect_keyframes
DEFAULT_TRANSITION_THRESHOLD = 0.3
DEFAULT_MIN_DISTANCE_BETWEEN_KEYFRAMES = 3

# Для нормализации
TARGET_LENGTH = 256
MIN_FACES_THRESHOLD = 20
MAX_RETRIES = 1


# ------------------------
# Face detection (InsightFace)
# -------------------------
def init_face_app(det_size: Tuple[int,int]=(640,640)):
    try:
        from insightface.app import FaceAnalysis
    except Exception as e:
        raise RuntimeError("insightface not installed or failed to import") from e
    # try GPU, fallback CPU
    try:
        app = FaceAnalysis(providers=["CUDAExecutionProvider"])
        app.prepare(ctx_id=0, det_size=det_size)
    except Exception:
        app = FaceAnalysis(providers=["CPUExecutionProvider"])
        app.prepare(ctx_id=-1, det_size=det_size)
    return app

def load_emonet(path: str, n_expression: int = 8):
    from models.emonet.emonet.models.emonet import EmoNet
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    model = EmoNet(n_expression=n_expression).to(DEVICE)
    model.load_state_dict(state, strict=False)
    model.eval()
    return model


if __name__ == "__main__":

    video_path = "videos/-RRm7U8ZSA0.mp4"
    model = load_emonet("models/emonet/pretrained/emonet_8.pth")
    face_app = init_face_app()

    from core.video_processor import VideoEmotionProcessor
        
    # Используем новый класс для обработки
    processor = VideoEmotionProcessor()
    result = processor.process(video_path, model, face_app, TARGET_LENGTH, CHUNK_SIZE)
    
    # Преобразуем результат в старый формат для обратной совместимости
    if result.get("success"):
        print("success")
    else:
        # Возвращаем результат в старом формате при ошибке
        print({
            "success": False,
            "error": result.get("error", "Failed to process video"),
            "attempts": result.get("attempts", 0),
            "final_params": result.get("final_params", {})
        })