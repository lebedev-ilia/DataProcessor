# segmenter.py
"""
Segmenter: подготовка фреймов, аудио и метаданных для экстракторов.

Функціонал:
- process_video: сохраняет фреймы батчами (batch_{id}.npy) и возвращает metadata.json
- extract_audio: извлекает аудио через ffmpeg -> wav, собирает метаданные (duration, sr, samples)
- create_extractor_metadata: формирует per-extractor метаданные:
    - для video: список frame_indices
    - для audio: список сегментов в ms и в сэмплах
- helper'ы: load_batch, read_metadata

Требования:
- opencv (cv2), numpy, ffmpeg (cli)
- ffprobe (обычно вместе с ffmpeg)
"""
from __future__ import annotations
import os
import json
import math
import subprocess
from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import cv2


def _log(logger, *args, **kwargs):
    if logger is None:
        print(*args, **kwargs)
    else:
        # поддерживаем .info или .log
        if hasattr(logger, "info"):
            logger.info(" ".join(map(str, args)))
        elif hasattr(logger, "log"):
            logger.log(" ".join(map(str, args)))
        else:
            print(*args, **kwargs)

# -----------------------
# Video processing
# -----------------------
def process_video(
    video_path: str,
    out_dir: str,
    chunk_size: int = 512,
    cache_size = 2,
    overwrite: bool = False,
    logger = None
) -> Dict[str, Any]:
    """
    Сохраняет фреймы видео батчами в out_dir и возвращает метаданные.
    Формат батча: np.save(os.path.join(out_dir, "batch_{id:05d}.npy"), np.array(frames_batch, dtype=np.uint8))
    Создаёт metadata.json c полями:
      total_frames, fps, height, width, channels, chunk_size, batches: [{batch_index, path, start_frame, end_frame}, ...]
    """
    output = f"{out_dir}/video"

    os.makedirs(output, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video '{video_path}'")

    # Попытка взять fps и приближенное количество фреймов
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if fps <= 0:
        fps = 30.0  # fallback
    approx_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    meta: Dict[str, Any] = {
        "video_path": os.path.abspath(video_path),
        "total_frames": 0,
        "approx_frame_count": approx_frame_count,
        "chunk_size": chunk_size,
        "cache_size":cache_size,
        "fps": float(fps),
        "batches": []
    }

    batch_frames: List[np.ndarray] = []
    batch_id = 0
    frame_idx = 0
    H = W = C = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if H is None:
            H, W, C = frame.shape
            meta["height"] = int(H)
            meta["width"] = int(W)
            meta["channels"] = int(C)

        # иногда cv2 возвращает другой размер — приводим к первому
        if frame.shape != (H, W, C):
            frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)

        batch_frames.append(frame.astype(np.uint8))
        frame_idx += 1
        meta["total_frames"] = frame_idx

        if len(batch_frames) >= chunk_size:
            fname = f"batch_{batch_id:05d}.npy"
            path = os.path.join(output, fname)
            np.save(path, np.stack(batch_frames, axis=0))
            _log(logger, f"[process_video] saved batch {batch_id} frames {frame_idx - len(batch_frames)}..{frame_idx-1} -> {fname}")

            meta["batches"].append({
                "batch_index": batch_id,
                "path": fname,
                "start_frame": frame_idx - len(batch_frames),
                "end_frame": frame_idx - 1
            })

            batch_id += 1
            batch_frames = []

    # Последний неполный батч
    if len(batch_frames) > 0:
        fname = f"batch_{batch_id:05d}.npy"
        path = os.path.join(output, fname)
        np.save(path, np.stack(batch_frames, axis=0))
        _log(logger, f"[process_video] saved final batch {batch_id} frames {frame_idx - len(batch_frames)}..{frame_idx-1} -> {fname}")

        meta["batches"].append({
            "batch_index": batch_id,
            "path": fname,
            "start_frame": frame_idx - len(batch_frames),
            "end_frame": frame_idx - 1
        })

    cap.release()
    return meta

def load_batch(batch_path: str) -> np.ndarray:
    """
    Загружает .npy батч и возвращает np.ndarray shape [N, H, W, C]
    """
    return np.load(batch_path, mmap_mode=None)

# -----------------------
# Audio extraction
# -----------------------
def _run_cmd(cmd: List[str]) -> Tuple[int, str, str]:
    """Выполняет subprocess команду, возвращает (retcode, stdout, stderr)."""
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return p.returncode, p.stdout, p.stderr

def extract_audio(
    video_path: str,
    out_dir: str,
    target_sr: int = 16000,
    mono: bool = True,
    overwrite: bool = False,
    logger = None
) -> Dict[str, Any]:
    """
    Извлекает аудио из видео в WAV (PCM S16) через ffmpeg.
    Возвращает аудио-мета: {audio_path, duration_sec, sample_rate, total_samples}
    Требует ffmpeg/ffprobe в PATH.
    """
    output = f"{out_dir}/audio"
    os.makedirs(output, exist_ok=True)
    base = os.path.splitext(os.path.basename(video_path))[0]
    audio_fname = f"{base}.wav"
    audio_path = os.path.join(output, audio_fname)

    if os.path.exists(audio_path) and not overwrite:
        _log(logger, f"[extract_audio] audio already exists: {audio_path}")
    else:
        # ffmpeg команда: извлечь аудио, привести к нужной частоте и моно
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", str(target_sr),
            "-ac", "1" if mono else "2",
            audio_path
        ]
        code, out, err = _run_cmd(cmd)
        if code != 0:
            raise RuntimeError(f"ffmpeg failed: {err.strip()}")

    # получить инфо через ffprobe (duration, sr)
    duration = None
    sample_rate = None
    total_samples = None

    # duration
    cmd_dur = ["ffprobe", "-v", "error", "-show_entries", "format=duration",
               "-of", "default=noprint_wrappers=1:nokey=1", audio_path]
    code, out, err = _run_cmd(cmd_dur)
    if code == 0 and out.strip():
        try:
            duration = float(out.strip())
        except Exception:
            duration = None

    # sample rate
    cmd_sr = ["ffprobe", "-v", "error", "-select_streams", "a:0",
              "-show_entries", "stream=sample_rate",
              "-of", "default=noprint_wrappers=1:nokey=1", audio_path]
    code, out, err = _run_cmd(cmd_sr)
    if code == 0 and out.strip():
        try:
            sample_rate = int(float(out.strip()))
        except Exception:
            sample_rate = None

    if duration is not None and sample_rate is not None:
        total_samples = int(math.floor(duration * sample_rate))

    audio_meta = {
        "audio_path": os.path.abspath(audio_path),
        "duration_sec": duration,
        "sample_rate": sample_rate,
        "total_samples": total_samples
    }

    # сохраняем мету
    with open(os.path.join(out_dir, "audio_metadata.json"), "w") as f:
        json.dump(audio_meta, f, indent=2)

    _log(logger, f"[extract_audio] saved audio metadata -> {audio_meta}")
    return audio_meta

# -----------------------
# Extractor metadata creation
# -----------------------
def create_extractor_metadata(
    output,
    frames_meta: Dict[str, Any],
    audio_meta: Optional[Dict[str, Any]],
    extractor_configs: List[Dict[str, Any]],
    logger = None
) -> List[Dict[str, Any]]:
    """
    Формирует для каждого экстрактора список индексов фреймов или аудио-сегментов.

    Формат extractor_config (пример):
    {
      "name": "EmotionExtractor",
      "modality": "video",
      # либо явно: "frame_indices": [0,3,6,9]
      # либо задать шаг: "frame_step": 3, "start_frame": 0, "max_frames": None
      "frame_step": 3,
      "start_frame": 0,
      "max_frames": None
    }

    Для audio:
    {
      "name": "AudioSpec",
      "modality": "audio",
      # segment_ms: длина сегмента в миллисекундах
      # step_ms: шаг (может быть равен segment_ms)
      "segment_ms": 1000,
      "step_ms": 500
    }

    Возвращает список dict'ов:
      {
       "name": ..,
       "modality": "video" | "audio",
       "frame_indices": [...],  # если video
       "audio_segments_ms": [{"start_ms":..,"end_ms":..,"start_sample":..,"end_sample":..}, ...]  # если audio
      }
    """
    total_frames = int(frames_meta.get("total_frames", 0))
    fps = float(frames_meta.get("fps", 30.0))
    video_duration = total_frames / fps if fps > 0 else None

    for cfg in extractor_configs:
        name = cfg.get("name", "unnamed")
        mod = cfg.get("modality", "video")
        out: Dict[str, Any] = {"modality": mod}

        if mod == "video":
            # explicit indices
            if "frame_indices" in cfg and cfg["frame_indices"] is not None:
                indices = [int(i) for i in cfg["frame_indices"] if 0 <= int(i) < total_frames]
            else:
                start = int(cfg.get("start_frame", 0))
                step = int(cfg.get("frame_step", 1))
                maxf = cfg.get("max_frames", None)
                indices = list(range(start, total_frames, step))
                if maxf is not None:
                    indices = indices[:int(maxf)]
            out["frame_indices"] = indices
            out["num_indices"] = len(indices)

            frames_meta.update({name:out})

            _log(logger, f"[create_extractor_metadata] {name} -> {len(indices)} frames (modality=video)")

        elif mod == "audio":
            if audio_meta is None or audio_meta.get("duration_sec") is None or audio_meta.get("sample_rate") is None:
                _log(logger, f"[create_extractor_metadata] warning: no audio_meta or incomplete audio_meta for extractor {name}")
                out["audio_segments_ms"] = []
                out["num_segments"] = 0
            else:
                dur_ms = int(round(audio_meta["duration_sec"] * 1000.0))
                sr = int(audio_meta["sample_rate"])
                segment_ms = int(cfg.get("segment_ms", 1000))
                step_ms = int(cfg.get("step_ms", segment_ms))
                segments = []
                start_ms = 0
                while start_ms < dur_ms:
                    end_ms = min(start_ms + segment_ms, dur_ms)
                    # convert to samples
                    start_sample = int(math.floor(start_ms * sr / 1000.0))
                    end_sample = int(math.floor(end_ms * sr / 1000.0))
                    segments.append({
                        "start_ms": int(start_ms),
                        "end_ms": int(end_ms),
                        "start_sample": start_sample,
                        "end_sample": end_sample
                    })
                    start_ms += step_ms
                out["audio_segments_ms"] = segments
                out["num_segments"] = len(segments)

                audio_meta.update({name:out})

                _log(logger, f"[create_extractor_metadata] {name} -> {len(segments)} audio segments (modality=audio)")
        else:
            _log(logger, f"[create_extractor_metadata] unknown modality '{mod}' for extractor {name}")
            out["note"] = "unknown modality"

    with open(f"{output}/video/metadata.json", "w") as f:
        json.dump(frames_meta, f, indent=2)

    with open(f"{output}/audio/metadata.json", "w") as f:
        json.dump(audio_meta, f, indent=2)

    return True

# -----------------------
# High-level orchestrator
# -----------------------
class Segmenter:
    """
    Высокоуровневый интерфейс — делает процессинг видео + аудио + формирование extractor metadata.
    """
    def __init__(self, out_dir: str, chunk_size: int = 512, logger = None):
        self.out_dir = out_dir
        self.chunk_size = chunk_size
        self.logger = logger
        os.makedirs(self.out_dir, exist_ok=True)

    def run(
        self,
        video_path: str,
        extractor_configs: List[Dict[str, Any]],
        overwrite: bool = False
    ) -> Dict[str, Any]:
        """
        Выполняет:
          - процессинг фреймов -> frames_metadata.json (в self.out_dir)
          - извлечение аудио -> audio_metadata.json (в self.out_dir)
          - создание extractor metadata (возвращается)
        Возвращает dict с keys: frames_meta, audio_meta, extractor_meta
        """
        _log(self.logger, f"[Segmenter.run] starting processing {video_path}")
        frames_meta = process_video(video_path, self.out_dir, chunk_size=self.chunk_size, overwrite=overwrite, logger=self.logger)
        audio_meta = extract_audio(video_path, self.out_dir, overwrite=overwrite, logger=self.logger)
        status = create_extractor_metadata(self.out_dir, frames_meta, audio_meta, extractor_configs, logger=self.logger)

        _log(self.logger, f"[Segmenter.run] finished. manifest saved -> segmenter_manifest.json")

# -----------------------
# Example usage (if run as script)
# -----------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--video-path", help="path to video")
    parser.add_argument("--output", default="data", help="out dir")
    parser.add_argument("--chunk-size", type=int, default=64)
    args = parser.parse_args()

    seg = Segmenter(out_dir=args.output, chunk_size=int(args.chunk_size), logger=None)

    extractor_configs = [
        {"name": "object_detection", "modality": "video", "frame_step": 3},
        {"name": "scene_classification", "modality": "video", "frame_step": 5},
        {"name": "face_detection", "modality": "video", "frame_step": 5},
        {"name": "detalize_face_modules", "modality": "video", "frame_step": 5},
        {"name": "emotion_face", "modality": "video", "frame_step": 5},
    ]

    seg.run(args.video_path, extractor_configs)

