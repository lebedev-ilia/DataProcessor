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
import yaml


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
def _utc_iso_now() -> str:
    import time
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _compute_uniform_indices(total_frames: int, n: int) -> List[int]:
    """
    Uniformly sample n indices from [0..total_frames-1], inclusive endpoints.
    Returns sorted unique indices.
    """
    total_frames = int(total_frames)
    n = int(n)
    if total_frames <= 0 or n <= 0:
        return []
    if n >= total_frames:
        return list(range(total_frames))
    # Use linspace for stability across lengths.
    idx = np.linspace(0, total_frames - 1, num=n)
    idx = np.unique(np.rint(idx).astype(np.int64))
    idx.sort()
    return [int(i) for i in idx.tolist()]


def _build_default_component_budgets() -> Dict[str, Dict[str, int]]:
    """
    Start budgets (min/target/max). Can be moved to config later.
    """
    return {
        "cut_detection": {"min": 400, "target": 800, "max": 1500},
        "core_clip": {"min": 200, "target": 400, "max": 800},
        "core_optical_flow": {"min": 200, "target": 400, "max": 800},
        "core_depth_midas": {"min": 120, "target": 200, "max": 400},
        "core_face_landmarks": {"min": 200, "target": 400, "max": 800},
        "core_object_detections": {"min": 200, "target": 400, "max": 800},
        "shot_quality": {"min": 200, "target": 500, "max": 1000},
        # reasonable defaults for remaining modules
        "scene_classification": {"min": 120, "target": 250, "max": 600},
        "video_pacing": {"min": 200, "target": 500, "max": 1200},
        "uniqueness": {"min": 200, "target": 500, "max": 1200},
        "story_structure": {"min": 120, "target": 250, "max": 600},
        "similarity_metrics": {"min": 120, "target": 250, "max": 600},
        "text_scoring": {"min": 120, "target": 250, "max": 600},
    }


def _canonical_component_name(name: str) -> str:
    # Directory name aliases → canonical metadata keys expected by core providers.
    if name == "object_detections":
        return "core_object_detections"
    if name == "depth_midas":
        return "core_depth_midas"
    return name


def _build_visual_extractor_configs_from_visual_cfg(
    visual_cfg_path: str,
    logger=None,
) -> List[Dict[str, Any]]:
    """
    Reads VisualProcessor/config.yaml and builds extractor_configs for enabled core providers + modules.
    Uses budgets (min/target/max) to generate `target_frames`.
    """
    with open(visual_cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    budgets = _build_default_component_budgets()

    enabled: List[str] = []
    core_cfg = (cfg.get("core_providers") or {})
    for name, enabled_flag in core_cfg.items():
        if enabled_flag:
            enabled.append(_canonical_component_name(str(name)))

    modules_cfg = (cfg.get("modules") or {})
    for name, enabled_flag in modules_cfg.items():
        if enabled_flag:
            enabled.append(_canonical_component_name(str(name)))

    # Deduplicate while preserving order
    seen = set()
    enabled_unique = []
    for n in enabled:
        if n not in seen:
            enabled_unique.append(n)
            seen.add(n)

    extractor_configs: List[Dict[str, Any]] = []
    for comp in enabled_unique:
        b = budgets.get(comp, {"min": 120, "target": 250, "max": 600})

        # Optional per-component overrides from VisualProcessor config.
        # Supports either:
        #   <component>:
        #     sampling:
        #       min_frames: ...
        #       target_frames: ...
        #       max_frames: ...
        # or direct keys on the component config for convenience.
        comp_cfg = cfg.get(comp) or {}
        sampling_cfg = (comp_cfg.get("sampling") or {}) if isinstance(comp_cfg, dict) else {}
        def _pick_int(key: str, default: int) -> int:
            if isinstance(sampling_cfg, dict) and key in sampling_cfg and sampling_cfg[key] is not None:
                return int(sampling_cfg[key])
            if isinstance(comp_cfg, dict) and key in comp_cfg and comp_cfg[key] is not None:
                return int(comp_cfg[key])
            return int(default)

        extractor_configs.append(
            {
                "name": comp,
                "modality": "video",
                "min_frames": _pick_int("min_frames", int(b["min"])),
                "target_frames": _pick_int("target_frames", int(b["target"])),
                "max_frames": _pick_int("max_frames", int(b["max"])),
            }
        )
    _log(logger, f"[Segmenter] built {len(extractor_configs)} video extractor configs from {visual_cfg_path}")
    return extractor_configs


def process_video_union(
    vid: str,
    video_path: str,
    out_dir: str,
    union_source_indices: List[int],
    chunk_size: int = 512,
    cache_size=2,
    overwrite: bool = False,
    logger=None,
    analysis_width: Optional[int] = None,
    analysis_height: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Extracts ONLY frames whose source indices are in union_source_indices, saves them in batches,
    and returns metadata for FrameManager (union-domain indexing).

    IMPORTANT: saved frames are RGB and stored in union order.
    """
    output = f"{out_dir}/{vid}/video"
    os.makedirs(output, exist_ok=True)

    # Requested source indices (may include frames beyond actual readable range; we will keep ONLY captured frames in metadata)
    requested_union_source_indices = sorted({int(i) for i in union_source_indices if int(i) >= 0})
    union_set = set(requested_union_source_indices)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video '{video_path}'")

    source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if source_fps <= 0:
        source_fps = 30.0
    approx_frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    meta: Dict[str, Any] = {
        "video_path": os.path.abspath(video_path),
        "source_fps": float(source_fps),
        "fps": float(source_fps),  # legacy field used by some codepaths; analysis_fps can be added later
        "approx_frame_count": approx_frame_count,
        # storage batch size (FrameManager supports batch_size or chunk_size)
        "chunk_size": int(chunk_size),
        "batch_size": int(chunk_size),
        "cache_size": int(cache_size),
        "color_space": "RGB",
        "batches": [],
        "total_frames": 0,
        # union mapping (filled from captured frames only)
        "union_frame_indices_source": [],
        "union_timestamps_sec": [],
    }

    batch_frames: List[np.ndarray] = []
    batch_id = 0
    union_pos = 0
    frame_idx = 0
    H = W = C = None
    captured_source_indices: List[int] = []
    captured_timestamps_sec: List[float] = []

    # Determine output resolution
    out_H = out_W = None
    if analysis_width is not None and analysis_height is not None:
        out_W = int(analysis_width)
        out_H = int(analysis_height)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if H is None:
            H, W, C = frame.shape
            # default output resolution = source
            if out_W is None or out_H is None:
                out_H, out_W = int(H), int(W)
            meta["height"] = int(out_H)
            meta["width"] = int(out_W)
            meta["channels"] = 3  # RGB

        if frame_idx in union_set:
            # normalize size (cv2 may vary)
            if frame.shape != (H, W, C):
                frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)

            if (int(out_W), int(out_H)) != (int(W), int(H)):
                frame = cv2.resize(frame, (int(out_W), int(out_H)), interpolation=cv2.INTER_AREA)

            # cv2.VideoCapture gives BGR; store RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            batch_frames.append(frame_rgb.astype(np.uint8))
            captured_source_indices.append(int(frame_idx))
            captured_timestamps_sec.append(float(frame_idx) / float(source_fps))
            union_pos += 1
            meta["total_frames"] = union_pos

            if len(batch_frames) >= chunk_size:
                fname = f"batch_{batch_id:05d}.npy"
                path = os.path.join(output, fname)
                np.save(path, np.stack(batch_frames, axis=0))

                start_frame = union_pos - len(batch_frames)
                end_frame = union_pos - 1
                meta["batches"].append(
                    {
                        "batch_index": batch_id,
                        "path": fname,
                        "start_frame": int(start_frame),
                        "end_frame": int(end_frame),
                    }
                )
                _log(logger, f"[process_video_union] saved batch {batch_id} union_frames {start_frame}..{end_frame} -> {fname}")
                batch_id += 1
                batch_frames = []

        frame_idx += 1

        # Optional early stop: if we've already captured all requested frames and indices are increasing.
        if union_pos >= len(requested_union_source_indices):
            # we've captured all frames in union
            break

    # final partial batch
    if len(batch_frames) > 0:
        fname = f"batch_{batch_id:05d}.npy"
        path = os.path.join(output, fname)
        np.save(path, np.stack(batch_frames, axis=0))
        start_frame = union_pos - len(batch_frames)
        end_frame = union_pos - 1
        meta["batches"].append(
            {
                "batch_index": batch_id,
                "path": fname,
                "start_frame": int(start_frame),
                "end_frame": int(end_frame),
            }
        )
        _log(logger, f"[process_video_union] saved final batch {batch_id} union_frames {start_frame}..{end_frame} -> {fname}")

    cap.release()

    # actual source frames read can be smaller than requested; keep union mapping strictly in captured union-domain.
    meta["union_frame_indices_source"] = captured_source_indices
    meta["union_timestamps_sec"] = captured_timestamps_sec
    meta["source_total_frames_read"] = int(frame_idx)
    meta["created_at"] = _utc_iso_now()
    return meta

def process_video(
    vid: str,
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
    output = f"{out_dir}/{vid}/video"

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
        "chunk_size": int(chunk_size),
        "batch_size": int(chunk_size),
        "cache_size":cache_size,
        "fps": float(fps),
        # Важно: все кадры сохраняем в RGB (а не BGR как отдаёт cv2).
        "color_space": "RGB",
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

        # cv2.VideoCapture отдаёт BGR; приводим к RGB, чтобы downstream всегда работал в одном цветовом пространстве.
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        batch_frames.append(frame_rgb.astype(np.uint8))
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
    vid: str,
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
    output = f"{out_dir}/{vid}/audio"
    os.makedirs(output, exist_ok=True)
    base = os.path.splitext(os.path.basename(video_path))[0]
    audio_fname = f"{base}.wav"
    audio_path = os.path.join(output, audio_fname)

    if os.path.exists(audio_path) and not overwrite:
        _log(logger, f"[extract_audio] audio already exists: {audio_path}")
    else:
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

    duration = None
    sample_rate = None
    total_samples = None

    cmd_dur = ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", audio_path]
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
        overwrite: bool = False,
        legacy_full_extract: bool = False,
        analysis_width: Optional[int] = None,
        analysis_height: Optional[int] = None,
        run_meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Выполняет:
          - процессинг фреймов -> frames_metadata.json (в self.out_dir)
          - извлечение аудио -> audio_metadata.json (в self.out_dir)
          - создание extractor metadata (возвращается)
        Возвращает dict с keys: frames_meta, audio_meta, extractor_meta
        """
        _log(self.logger, f"[Segmenter.run] starting processing {video_path}")

        # IMPORTANT: directory identity should follow canonical video_id when provided (not file basename).
        vid = None
        if run_meta and isinstance(run_meta.get("video_id"), str) and run_meta.get("video_id"):
            vid = str(run_meta["video_id"])
        if not vid:
            vid = os.path.splitext(os.path.basename(video_path))[0]
        
        if legacy_full_extract:
            frames_meta = process_video(
                vid, video_path, self.out_dir, chunk_size=self.chunk_size, overwrite=overwrite, logger=self.logger
            )
            audio_meta = extract_audio(vid, video_path, self.out_dir, overwrite=overwrite, logger=self.logger)
            create_extractor_metadata(self.out_dir, frames_meta, audio_meta, extractor_configs, logger=self.logger)
            return {"frames_meta": frames_meta, "audio_meta": audio_meta}

        # --- Union-sampled mode (new default) ---
        # 1) Estimate total frames from container (best effort)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video '{video_path}'")
        source_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or 30.0
        total_frames_source = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        cap.release()

        # If frame count unknown, we still can sample by step later; for now use a safe fallback.
        if total_frames_source <= 0:
            total_frames_source = 1

        # 2) Compute per-component SOURCE indices using budgets or explicit frame_indices
        per_component_source: Dict[str, List[int]] = {}
        for cfg in extractor_configs:
            name = str(cfg.get("name") or "").strip()
            if not name:
                continue
            if cfg.get("modality", "video") != "video":
                continue

            if cfg.get("frame_indices") is not None:
                indices = [int(i) for i in cfg["frame_indices"] if 0 <= int(i) < total_frames_source]
            else:
                target = int(cfg.get("target_frames") or 0)
                min_n = int(cfg.get("min_frames") or 0)
                max_n = int(cfg.get("max_frames") or 0)
                if target <= 0:
                    # fallback to step-based configs if provided
                    step = int(cfg.get("frame_step") or 1)
                    indices = list(range(0, total_frames_source, max(1, step)))
                else:
                    n = target
                    if min_n > 0:
                        n = max(n, min_n)
                    if max_n > 0:
                        n = min(n, max_n)
                    indices = _compute_uniform_indices(total_frames_source, n)

            per_component_source[name] = indices

        union_source_indices: List[int] = sorted({i for v in per_component_source.values() for i in v})

        # 3) Extract only union frames
        frames_meta = process_video_union(
            vid=vid,
            video_path=video_path,
            out_dir=self.out_dir,
            union_source_indices=union_source_indices,
            chunk_size=self.chunk_size,
            overwrite=overwrite,
            logger=self.logger,
            analysis_width=analysis_width,
            analysis_height=analysis_height,
        )

        # 4) Build source->union mapping and write per-component indices in UNION domain
        source_to_union = {src: idx for idx, src in enumerate(frames_meta["union_frame_indices_source"])}
        for comp, src_idx in per_component_source.items():
            union_idx = [int(source_to_union[i]) for i in src_idx if i in source_to_union]
            frames_meta[comp] = {
                "modality": "video",
                "frame_indices": union_idx,
                "num_indices": int(len(union_idx)),
                # debug-only mapping
                "source_frame_indices": src_idx,
                "num_source_indices": int(len(src_idx)),
            }

        # 4.5) Self-check: ensure union-domain indices are valid for FrameManager.get()
        total_union = int(frames_meta.get("total_frames") or 0)
        for comp, payload in list(frames_meta.items()):
            if not isinstance(payload, dict):
                continue
            if payload.get("modality") != "video":
                continue
            fi = payload.get("frame_indices")
            if fi is None:
                continue
            if not isinstance(fi, list):
                raise TypeError(f"[Segmenter] {comp}.frame_indices must be a list, got {type(fi).__name__}")
            # ints, sorted, unique, within range
            ints = [int(x) for x in fi]
            if ints != sorted(ints):
                raise ValueError(f"[Segmenter] {comp}.frame_indices not sorted")
            if len(ints) != len(set(ints)):
                raise ValueError(f"[Segmenter] {comp}.frame_indices not unique")
            if any((x < 0 or x >= total_union) for x in ints):
                raise ValueError(f"[Segmenter] {comp}.frame_indices out of range for union total_frames={total_union}")
            # write back normalized ints to avoid accidental numpy scalars
            payload["frame_indices"] = ints

        # 5) Add run meta (best effort)
        if run_meta:
            frames_meta.update({k: v for k, v in run_meta.items() if v is not None})

        # 6) Save metadata.json (video)
        video_meta_path = os.path.join(self.out_dir, vid, "video", "metadata.json")
        with open(video_meta_path, "w", encoding="utf-8") as f:
            json.dump(frames_meta, f, indent=2, ensure_ascii=False)

        # audio is unchanged (saved for completeness)
        audio_meta = extract_audio(vid, video_path, self.out_dir, overwrite=overwrite, logger=self.logger)
        audio_meta_path = os.path.join(self.out_dir, vid, "audio", "metadata.json")
        with open(audio_meta_path, "w", encoding="utf-8") as f:
            json.dump(audio_meta, f, indent=2, ensure_ascii=False)

        _log(self.logger, f"[Segmenter.run] union mode done: union_frames={frames_meta.get('total_frames')} -> {video_meta_path}")
        return {"frames_meta": frames_meta, "audio_meta": audio_meta}

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
    parser.add_argument("--legacy-full-extract", action="store_true", help="extract ALL frames (legacy, expensive)")
    parser.add_argument("--visual-cfg-path", type=str, default=None, help="Path to VisualProcessor/config.yaml (to build per-component budgets)")
    parser.add_argument("--analysis-width", type=int, default=None, help="Optional resize width for analysis timeline")
    parser.add_argument("--analysis-height", type=int, default=None, help="Optional resize height for analysis timeline")
    parser.add_argument("--platform-id", type=str, default="youtube")
    parser.add_argument("--video-id", type=str, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--sampling-policy-version", type=str, default="v1")
    parser.add_argument("--config-hash", type=str, default=None, help="Optional config hash propagated by DataProcessor")
    args = parser.parse_args()

    seg = Segmenter(out_dir=args.output, chunk_size=int(args.chunk_size), logger=None)

    if args.visual_cfg_path:
        extractor_configs = _build_visual_extractor_configs_from_visual_cfg(args.visual_cfg_path, logger=None)
    else:
        # Simple default for manual runs (legacy behavior)
        extractor_configs = [
            {"name": "core_clip", "modality": "video", "target_frames": 400, "min_frames": 200, "max_frames": 800},
            {"name": "cut_detection", "modality": "video", "target_frames": 800, "min_frames": 400, "max_frames": 1500},
            {"name": "shot_quality", "modality": "video", "target_frames": 500, "min_frames": 200, "max_frames": 1000},
        ]

    # Ensure ids are populated (so output folder matches orchestrator expectations)
    _vid = args.video_id or os.path.splitext(os.path.basename(args.video_path))[0]
    _run_id = args.run_id or None
    run_meta = {
        "platform_id": args.platform_id,
        "video_id": _vid,
        "run_id": _run_id,
        "sampling_policy_version": args.sampling_policy_version,
        "config_hash": args.config_hash,
    }

    seg.run(
        args.video_path,
        extractor_configs,
        legacy_full_extract=bool(args.legacy_full_extract),
        analysis_width=args.analysis_width,
        analysis_height=args.analysis_height,
        run_meta=run_meta,
    )

