#!/usr/bin/env python3
"""
Run split CPU/GPU extractor tests for three videos.

- CPU extractors: (hpss, speaker_diarization, clap, rhythmic, chroma, video_audio,
  spectral, loudness, voice_quality, band_energy, tempo, onset, mel,
  spectral_entropy, quality, key, mfcc)
- GPU extractors: (pitch, source_separation, speech_analysis, asr)

Saves outputs into tests/output_cpu/<video_id> and tests/output_gpu/<video_id>.
Writes clean transcription text files for ASR. Records timing per extractor,
per device, and combined, producing a summary JSON.
"""
import os
import sys
import json
import gc
import time
import logging
from pathlib import Path
from typing import Dict, Any, List, Tuple
import warnings

# Silence most warnings/logging as early as possible
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=UserWarning, module=r"webrtcvad")
warnings.filterwarnings("ignore", category=UserWarning, module=r"torchaudio(\..*)?")
logging.disable(logging.CRITICAL)

# Suppress deprecation warning emitted by webrtcvad importing pkg_resources
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module=r"webrtcvad",
)


CPU_EXTRACTORS: List[str] = [
    "hpss",
    "clap",
    "rhythmic",
    "chroma",
    "video_audio",
    "spectral",
    "loudness",
    "voice_quality",
    "band_energy",
    "tempo",
    "onset",
    "mel",
    "spectral_entropy",
    "quality",
    "key",
    "mfcc",
]

GPU_EXTRACTORS: List[str] = [
    "source_separation",
    "speech_analysis",
    # Виртуальные экстракторы, реализуемые внутри speech_analysis
    "pitch",
    "asr",
    "speaker_diarization",
    "emotion_diarization",
]


def _setup_logging() -> None:
    # No-op: logging is silenced globally
    return None


def _clean_transcription(payload: Dict[str, Any]) -> str:
    if not isinstance(payload, dict):
        return ""
    text = payload.get("transcription") or payload.get("text") or ""
    if not isinstance(text, str):
        return ""
    # Basic cleanup: strip whitespace and collapse internal whitespace
    cleaned = " ".join(text.strip().split())
    return cleaned


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def main() -> int:
    root = Path(__file__).parent
    src = root / "src"
    sys.path.insert(0, str(src))

    # Hint CPU threads
    try:
        import torch  # noqa: WPS433
        torch.set_num_threads(min(4, os.cpu_count() or 4))
        cuda_available = torch.cuda.is_available()
    except Exception:
        cuda_available = False

    from core.main_processor import MainProcessor

    videos_dir = root / "tests"
    video_files = [
        videos_dir / "-69HDT6DZEM.mp4",
        videos_dir / "-JuF2ivdnAg.mp4",
        videos_dir / "-niwQ0xGEGk.mp4",
    ]

    cpu_out_root = videos_dir / "output_cpu"
    gpu_out_root = videos_dir / "output_gpu"
    cpu_out_root.mkdir(parents=True, exist_ok=True)
    gpu_out_root.mkdir(parents=True, exist_ok=True)

    # ВАЖНО: инициализируем процессоры последовательно, чтобы не держать два одновременно
    proc_cpu = None
    proc_gpu = None

    global_t0 = time.time()
    overall_summary: Dict[str, Any] = {
        "device_totals": {"cpu_total_s": 0.0, "gpu_total_s": 0.0, "combined_total_s": 0.0},
        "videos": {},
        "cuda_available": bool(cuda_available),
        "wall_clock": {
            "start_ts": global_t0,
            "start_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(global_t0)),
            "end_ts": None,
            "end_iso": None,
            "elapsed_s": None,
        },
        "timeline": [],  # list of {event, device, video, ts}
    }

    # ===== CPU pass =====
    proc_cpu = MainProcessor(device="cpu", max_workers=2, enabled_extractors=CPU_EXTRACTORS)
    for vf in video_files:
        if not vf.exists():
            continue

        vid = vf.stem
        cpu_out = cpu_out_root / vid
        gpu_out = gpu_out_root / vid
        cpu_out.mkdir(parents=True, exist_ok=True)
        gpu_out.mkdir(parents=True, exist_ok=True)

        per_video: Dict[str, Any] = {
            "cpu": {"time_s": 0.0, "per_extractor": {}, "wall_clock": {}},
            "gpu": {"time_s": 0.0, "per_extractor": {}, "wall_clock": {}},
            "combined_time_s": 0.0,
            "wall_clock": {},
        }

        # CPU run
        t0 = time.time()
        overall_summary["timeline"].append({"event": "cpu_start", "device": "cpu", "video": vid, "ts": t0})
        res_cpu = proc_cpu.process_video(
            video_path=str(vf),
            output_dir=str(cpu_out),
            extractor_names=CPU_EXTRACTORS,
            extract_audio=True,
        )
        t1 = time.time()
        overall_summary["timeline"].append({"event": "cpu_end", "device": "cpu", "video": vid, "ts": t1})
        cpu_time = res_cpu.get("processing_time", t1 - t0)
        per_video["cpu"]["time_s"] = float(cpu_time)
        per_video["cpu"]["wall_clock"] = {
            "start_ts": t0,
            "start_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
            "end_ts": t1,
            "end_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t1)),
            "elapsed_s": float(t1 - t0),
        }
        for name, r in (res_cpu.get("extractor_results") or {}).items():
            per_video["cpu"]["per_extractor"][name] = float(r.get("processing_time", 0.0) or 0.0)

        # Пока GPU не запускали; per_video запишем после GPU pass
        overall_summary["videos"][vid] = per_video
        overall_summary["device_totals"]["cpu_total_s"] += per_video["cpu"]["time_s"]

    # Освобождаем CPU процессор до инициализации GPU
    try:
        del proc_cpu
    except Exception:
        pass
    try:
        import torch  # noqa: WPS433
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    gc.collect()

    # ===== GPU pass (если доступен) =====
    if cuda_available:
        proc_gpu = MainProcessor(device="cuda", max_workers=2, enabled_extractors=GPU_EXTRACTORS)
        for vf in video_files:
            if not vf.exists():
                continue

            vid = vf.stem
            gpu_out = gpu_out_root / vid
            gpu_out.mkdir(parents=True, exist_ok=True)

            t2 = time.time()
            overall_summary["timeline"].append({"event": "gpu_start", "device": "gpu", "video": vid, "ts": t2})
            res_gpu = proc_gpu.process_video(
                video_path=str(vf),
                output_dir=str(gpu_out),
                extractor_names=GPU_EXTRACTORS,
                extract_audio=True,
            )
            t3 = time.time()
            overall_summary["timeline"].append({"event": "gpu_end", "device": "gpu", "video": vid, "ts": t3})
            gpu_time = res_gpu.get("processing_time", t3 - t2)

            # Обновляем уже созданную запись per_video
            per_video = overall_summary["videos"].setdefault(vid, {"cpu": {}, "gpu": {}, "combined_time_s": 0.0, "wall_clock": {}})
            per_video.setdefault("gpu", {})
            per_video["gpu"]["time_s"] = float(gpu_time)
            per_video["gpu"]["wall_clock"] = {
                "start_ts": t2,
                "start_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t2)),
                "end_ts": t3,
                "end_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t3)),
                "elapsed_s": float(t3 - t2),
            }
            per_video["gpu"].setdefault("per_extractor", {})
            for name, r in (res_gpu.get("extractor_results") or {}).items():
                per_video["gpu"]["per_extractor"][name] = float(r.get("processing_time", 0.0) or 0.0)

            # Combined + wall clock
            cpu_ts = per_video.get("cpu", {}).get("time_s", 0.0) or 0.0
            per_video["combined_time_s"] = float(cpu_ts + per_video["gpu"]["time_s"])

            vid_start_ts = per_video.get("cpu", {}).get("wall_clock", {}).get("start_ts") or per_video["gpu"]["wall_clock"].get("start_ts")
            vid_end_ts = per_video["gpu"]["wall_clock"].get("end_ts") or per_video.get("cpu", {}).get("wall_clock", {}).get("end_ts")
            if per_video.get("cpu", {}).get("wall_clock") and per_video.get("gpu", {}).get("wall_clock"):
                vid_start_ts = min(per_video["cpu"]["wall_clock"]["start_ts"], per_video["gpu"]["wall_clock"]["start_ts"])
                vid_end_ts = max(per_video["cpu"]["wall_clock"]["end_ts"], per_video["gpu"]["wall_clock"]["end_ts"])
            per_video["wall_clock"] = {
                "start_ts": vid_start_ts,
                "start_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(vid_start_ts)) if vid_start_ts else None,
                "end_ts": vid_end_ts,
                "end_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(vid_end_ts)) if vid_end_ts else None,
                "elapsed_s": float((vid_end_ts - vid_start_ts)) if (vid_start_ts and vid_end_ts) else None,
            }

            overall_summary["device_totals"]["gpu_total_s"] += per_video["gpu"]["time_s"]
            overall_summary["device_totals"]["combined_total_s"] += per_video["gpu"]["time_s"]  # CPU уже был учтен
    else:
        pass

    # Close overall wall clock
    global_t1 = time.time()
    overall_summary["wall_clock"]["end_ts"] = global_t1
    overall_summary["wall_clock"]["end_iso"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(global_t1))
    overall_summary["wall_clock"]["elapsed_s"] = float(global_t1 - global_t0)

    # Write overall summary JSON next to outputs
    summary_path = videos_dir / "cpu_gpu_split_summary.json"
    _write_json(summary_path, overall_summary)

    # Console output is silenced; summary JSON is written above
    try:
        print("\n===== SUMMARY =====")
        wc = overall_summary["wall_clock"]["elapsed_s"]
        print(f"Wall-clock total: {wc:.3f}s | CUDA available: {overall_summary['cuda_available']}")
        dt = overall_summary["device_totals"]
        print(f"CPU total: {dt['cpu_total_s']:.3f}s | GPU total: {dt['gpu_total_s']:.3f}s | Combined: {dt['combined_total_s']:.3f}s")
        for vid, per in overall_summary["videos"].items():
            cpu_t = per.get("cpu", {}).get("time_s", 0.0) or 0.0
            gpu_t = per.get("gpu", {}).get("time_s", 0.0) or 0.0
            comb = per.get("combined_time_s", cpu_t + gpu_t)
            print(f" - {vid}: CPU {cpu_t:.3f}s, GPU {gpu_t:.3f}s, Combined {comb:.3f}s")
        print("===================\n")
    except Exception:
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())


