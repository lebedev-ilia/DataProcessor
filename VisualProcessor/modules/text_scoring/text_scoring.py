# text_video_pipeline.py

import numpy as np
from scipy.ndimage import gaussian_filter1d
from collections import defaultdict
from typing import List, Dict, Any, Tuple

class TextVideoInteractionPipeline:
    """
    Пайплайн для извлечения фичей взаимодействия текста и видео.
    Вход: кадры, OCR с bbox, motion/face/audio пики
    Выход: словарь фичей на видео
    """

    def __init__(self, video_fps: int = 30):
        self.video_fps = video_fps

    @staticmethod
    def _iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
        """Intersection over union для bbox"""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou

    @staticmethod
    def _normalize_signal(signal: np.ndarray) -> np.ndarray:
        """Нормализация и сглаживание сигнала для уменьшения шумов"""
        if len(signal) == 0:
            return np.array([])
        signal = np.array(signal)
        signal = gaussian_filter1d(signal, sigma=1)
        return signal / (signal.max() + 1e-6)

    def extract_features(
        self,
        ocr_data: List[Dict[str, Any]],
        motion_peaks: List[float],
        face_peaks: List[float],
        audio_peaks: List[float] = None
    ) -> Dict[str, Any]:
        """
        ocr_data: List[Dict] = [
            {"frame": 10, "bbox": (x1,y1,x2,y2), "text": "...", "confidence": 0.95, "is_cta": False},
            ...
        ]
        motion_peaks: List[float] = motion intensity per frame
        face_peaks: List[float] = face detection peaks per frame
        audio_peaks: List[float] = audio energy per frame (optional)
        """
        features = defaultdict(float)

        num_frames = max([d["frame"] for d in ocr_data], default=0) + 1
        motion_signal = np.array(motion_peaks[:num_frames])
        face_signal = np.array(face_peaks[:num_frames])
        audio_signal = np.array(audio_peaks[:num_frames]) if audio_peaks is not None else np.zeros(num_frames)

        motion_signal = self._normalize_signal(motion_signal)
        face_signal = self._normalize_signal(face_signal)
        audio_signal = self._normalize_signal(audio_signal)

        # ---------- 1. Text → Action / Motion ----------
        text_action_scores = []
        text_motion_align_scores = []
        text_durations = defaultdict(list)  # frame indices per unique text
        cta_times = []
        cta_strengths = []

        for entry in ocr_data:
            frame_idx = entry["frame"]
            bbox = entry["bbox"]
            text = entry["text"]
            is_cta = entry.get("is_cta", False)

            # Text → Action / Motion correlation
            motion_peak_val = motion_signal[frame_idx] if frame_idx < len(motion_signal) else 0
            face_peak_val = face_signal[frame_idx] if frame_idx < len(face_signal) else 0
            audio_peak_val = audio_signal[frame_idx] if frame_idx < len(audio_signal) else 0

            # Простая корреляция: пересечение bbox с motion (0..1)
            # Тут можно расширить до региональной корреляции, но оставим как упрощенный сигнал
            action_score = motion_peak_val  # сильное движение совпадает с текстом
            text_action_scores.append(action_score)

            # Alignment score: motion + face + audio peaks
            multimodal_score = 0.4*motion_peak_val + 0.3*face_peak_val + 0.3*audio_peak_val
            text_motion_align_scores.append(multimodal_score)

            # Длительность текста на экране
            text_durations[text].append(frame_idx)

            # CTA detection
            if is_cta:
                cta_times.append(frame_idx)
                cta_strengths.append(multimodal_score)

        # ---------- Aggregate features ----------
        features["text_action_sync_score"] = np.mean(text_action_scores) if text_action_scores else 0
        features["text_motion_alignment"] = np.mean(text_motion_align_scores) if text_motion_align_scores else 0
        features["multimodal_attention_boost_score"] = np.max(text_motion_align_scores) if text_motion_align_scores else 0

        # Text duration and switch rate
        durations = [len(frames)/self.video_fps for frames in text_durations.values()]
        features["text_on_screen_continuity"] = np.mean(durations) if durations else 0
        features["text_switch_rate"] = len(text_durations)/(num_frames/self.video_fps + 1e-6) if num_frames > 0 else 0

        # CTA
        features["cta_presence"] = 1 if cta_times else 0
        features["cta_timestamp"] = np.mean(cta_times)/self.video_fps if cta_times else None
        features["cta_strength"] = np.mean(cta_strengths) if cta_strengths else 0

        # Optional: per-frame emphasis flags
        threshold = np.percentile(text_motion_align_scores, 85) if text_motion_align_scores else 1.0
        features["text_emphasis_peak_flags"] = [i for i, s in enumerate(text_motion_align_scores) if s >= threshold]

        return dict(features)
