#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import uuid
import tempfile
import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np


def _utc_iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

def _sha256_text(s: str) -> str:
    import hashlib
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _timestamp_now() -> str:
    # time.strftime does not support %f (microseconds); use datetime.
    return datetime.datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S-%f")


def _short_uuid() -> str:
    return uuid.uuid4().hex[:8]


def _atomic_save_npz(path: str, **arrays: Any) -> None:
    target_dir = os.path.dirname(path)
    os.makedirs(target_dir, exist_ok=True)
    tmp_fd, tmp_path = tempfile.mkstemp(prefix=Path(path).name + ".", suffix=".npz", dir=target_dir)
    os.close(tmp_fd)
    try:
        np.savez_compressed(tmp_path, **arrays)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise


def _meta(
    *,
    producer: str,
    producer_version: str,
    status: str,
    schema_version: str,
    extra: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    d = {
        "producer": producer,
        "producer_version": producer_version,
        "schema_version": schema_version,
        "status": status,
        "created_at": _utc_iso_now(),
    }
    if extra:
        d.update(extra)
    # PR-3: model system baseline
    try:
        # Import lazily: sys.path is prepared inside main()
        from src.utils.meta_builder import apply_models_meta  # type: ignore

        d = apply_models_meta(d, models_used=d.get("models_used"))
    except Exception:
        # Best-effort: do not crash saving path on missing helper.
        d.setdefault("models_used", [])
        d.setdefault("model_signature", "")
    return np.asarray(d, dtype=object)


def _as_float(v: Any) -> float:
    try:
        if v is None:
            return float("nan")
        return float(v)
    except Exception:
        return float("nan")


def _as_int(v: Any) -> int:
    try:
        if v is None:
            return -1
        return int(v)
    except Exception:
        return -1


def _save_component_npz(
    *,
    run_rs_path: str,
    component_name: str,
    payload: Optional[Dict[str, Any]],
    status: str,
    error: Optional[str],
    empty_reason: Optional[str],
    producer_version: str,
    schema_version: str,
    extra_meta: Optional[Dict[str, Any]] = None,
) -> str:
    comp_dir = os.path.join(run_rs_path, component_name)
    out_path = os.path.join(comp_dir, f"{_timestamp_now()}_{_short_uuid()}.npz")

    payload = payload or {}

    # Store a flexible "feature vector" representation:
    # - feature_names: object array of strings
    # - feature_values: float32 array aligned with names
    feature_names: list[str] = []
    feature_values: list[float] = []

    def add(name: str, value: Any):
        feature_names.append(name)
        feature_values.append(_as_float(value))

    # Component-specific payload parsing
    if component_name == "tempo_extractor":
        add("tempo_bpm", payload.get("tempo_bpm"))
        add("tempo_bpm_mean", payload.get("tempo_bpm_mean"))
        add("tempo_bpm_median", payload.get("tempo_bpm_median"))
        add("tempo_bpm_std", payload.get("tempo_bpm_std"))
        add("tempo_confidence", payload.get("confidence"))
        add("duration_sec", payload.get("duration"))
        add("sample_rate", payload.get("sample_rate"))
        add("tempo_windowed_bpm_mean", (payload.get("windowed_bpm") or {}).get("bpm_mean"))
        add("tempo_windowed_bpm_median", (payload.get("windowed_bpm") or {}).get("bpm_median"))
        add("tempo_windowed_bpm_std", (payload.get("windowed_bpm") or {}).get("bpm_std"))
        add("segments_count", payload.get("segments_count"))

        tempo_estimates = payload.get("tempo_estimates")
        if tempo_estimates is None:
            tempo_estimates_arr = np.zeros((0,), dtype=np.float32)
        else:
            tempo_estimates_arr = np.asarray(tempo_estimates, dtype=np.float32).reshape(-1)

        windowed = payload.get("windowed_bpm") or {}
        if isinstance(windowed, dict) and windowed:
            w_times = np.asarray(windowed.get("times_sec") or [], dtype=np.float32).reshape(-1)
            w_bpm = np.asarray(windowed.get("bpm") or [], dtype=np.float32).reshape(-1)
        else:
            w_times = np.zeros((0,), dtype=np.float32)
            w_bpm = np.zeros((0,), dtype=np.float32)

        _atomic_save_npz(
            out_path,
            feature_names=np.asarray(feature_names, dtype=object),
            feature_values=np.asarray(feature_values, dtype=np.float32),
            tempo_estimates=tempo_estimates_arr,
            windowed_times_sec=w_times,
            windowed_bpm=w_bpm,
            warnings=np.asarray(payload.get("warnings") or [], dtype=object),
            meta=_meta(
                producer=component_name,
                producer_version=producer_version,
                schema_version=schema_version,
                status=status,
                extra={
                    **(extra_meta or {}),
                    **({"error": error} if error else {}),
                    "empty_reason": empty_reason,
                },
            ),
        )
        return out_path

    if component_name == "loudness_extractor":
        # LUFS is optional. Store NaN + a flag.
        lufs = payload.get("lufs")
        lufs_present = bool(isinstance(lufs, (int, float)) and np.isfinite(float(lufs)))
        add("loudness_rms", payload.get("rms"))
        add("loudness_peak", payload.get("peak"))
        add("loudness_dbfs", payload.get("dbfs"))
        add("loudness_lufs", lufs if lufs_present else float("nan"))
        add("duration_sec", payload.get("duration"))
        add("sample_rate", payload.get("sample_rate"))
        add("frame_rms_mean", payload.get("frame_rms_mean"))
        add("frame_rms_std", payload.get("frame_rms_std"))
        add("frame_rms_median", payload.get("frame_rms_median"))
        add("frame_rms_p10", payload.get("frame_rms_p10"))
        add("frame_rms_p90", payload.get("frame_rms_p90"))
        add("frames_count", payload.get("frames_count"))
        add("segments_count", payload.get("segments_count"))
        add("segment_rms_mean", payload.get("segment_rms_mean"))
        add("segment_rms_std", payload.get("segment_rms_std"))
        add("segment_rms_median", payload.get("segment_rms_median"))
        add("segment_rms_p10", payload.get("segment_rms_p10"))
        add("segment_rms_p90", payload.get("segment_rms_p90"))

        _atomic_save_npz(
            out_path,
            feature_names=np.asarray(feature_names, dtype=object),
            feature_values=np.asarray(feature_values, dtype=np.float32),
            lufs_present=np.asarray(lufs_present, dtype=np.bool_),
            segment_centers_sec=np.asarray(payload.get("segment_centers_sec") or [], dtype=np.float32).reshape(-1),
            segment_rms=np.asarray(payload.get("segment_rms") or [], dtype=np.float32).reshape(-1),
            segment_peak=np.asarray(payload.get("segment_peak") or [], dtype=np.float32).reshape(-1),
            segment_dbfs=np.asarray(payload.get("segment_dbfs") or [], dtype=np.float32).reshape(-1),
            segment_lufs=np.asarray(payload.get("segment_lufs") or [], dtype=np.float32).reshape(-1),
            meta=_meta(
                producer=component_name,
                producer_version=producer_version,
                schema_version=schema_version,
                status=status,
                extra={
                    **(extra_meta or {}),
                    **({"error": error} if error else {}),
                    "empty_reason": empty_reason,
                },
            ),
        )
        return out_path

    if component_name == "clap_extractor":
        # Embeddings can be provided directly (preferred) or as a .npy path (legacy).
        emb_present = False
        emb = np.zeros((0,), dtype=np.float32)
        emb_seq = np.zeros((0, 0), dtype=np.float32)
        seg_centers = np.zeros((0,), dtype=np.float32)

        if payload.get("embedding") is not None:
            try:
                emb = np.asarray(payload.get("embedding"), dtype=np.float32).reshape(-1)
                emb_present = emb.size > 0
            except Exception:
                emb_present = False
                emb = np.zeros((0,), dtype=np.float32)
        else:
            emb_path = payload.get("clap_embeddings_npy")
            if isinstance(emb_path, str) and emb_path and os.path.exists(emb_path):
                try:
                    emb = np.asarray(np.load(emb_path), dtype=np.float32).reshape(-1)
                    emb_present = emb.size > 0
                except Exception:
                    emb_present = False
                    emb = np.zeros((0,), dtype=np.float32)

        if payload.get("embedding_sequence") is not None:
            try:
                emb_seq = np.asarray(payload.get("embedding_sequence"), dtype=np.float32)
                if emb_seq.ndim != 2:
                    emb_seq = emb_seq.reshape(emb_seq.shape[0], -1) if emb_seq.size else np.zeros((0, 0), dtype=np.float32)
            except Exception:
                emb_seq = np.zeros((0, 0), dtype=np.float32)
        if payload.get("segment_centers_sec") is not None:
            try:
                seg_centers = np.asarray(payload.get("segment_centers_sec"), dtype=np.float32).reshape(-1)
            except Exception:
                seg_centers = np.zeros((0,), dtype=np.float32)

        add("embedding_dim", payload.get("embedding_dim"))
        add("sample_rate", payload.get("sample_rate"))
        add("clap_norm", payload.get("clap_norm"))
        add("clap_magnitude_mean", payload.get("clap_magnitude_mean"))
        add("clap_magnitude_std", payload.get("clap_magnitude_std"))
        add("clap_non_zero_count", payload.get("clap_non_zero_count"))
        add("segments_count", payload.get("segments_count"))

        _atomic_save_npz(
            out_path,
            feature_names=np.asarray(feature_names, dtype=object),
            feature_values=np.asarray(feature_values, dtype=np.float32),
            embedding=emb,
            embedding_present=np.asarray(emb_present, dtype=np.bool_),
            embedding_sequence=emb_seq,
            segment_centers_sec=seg_centers,
            meta=_meta(
                producer=component_name,
                producer_version=producer_version,
                schema_version=schema_version,
                status=status,
                extra={
                    **(extra_meta or {}),
                    **({"error": error} if error else {}),
                    "empty_reason": empty_reason,
                },
            ),
        )
        return out_path

    if component_name == "asr_extractor":
        # Token IDs from a shared tokenizer (no raw text stored).
        token_ids_by_segment = payload.get("token_ids_by_segment") or []
        if not isinstance(token_ids_by_segment, list):
            token_ids_by_segment = []
        # Store as object array of int32 vectors (variable lengths).
        tok_obj = np.asarray(
            [np.asarray(x, dtype=np.int32).reshape(-1) for x in token_ids_by_segment],
            dtype=object,
        )
        seg_st = np.asarray(payload.get("segment_start_sec") or [], dtype=np.float32).reshape(-1)
        seg_en = np.asarray(payload.get("segment_end_sec") or [], dtype=np.float32).reshape(-1)
        seg_center = np.asarray(payload.get("segment_center_sec") or [], dtype=np.float32).reshape(-1)
        lang_ids = np.asarray(payload.get("lang_id_by_segment") or [], dtype=np.int32).reshape(-1)

        add("segments_count", payload.get("segments_count"))
        add("tokenizer_model", payload.get("tokenizer_model_name"))
        add("whisper_model", payload.get("whisper_model_name"))
        add("sample_rate", payload.get("sample_rate"))

        _atomic_save_npz(
            out_path,
            feature_names=np.asarray(feature_names, dtype=object),
            feature_values=np.asarray(feature_values, dtype=np.float32),
            token_ids_by_segment=tok_obj,
            segment_start_sec=seg_st,
            segment_end_sec=seg_en,
            segment_center_sec=seg_center,
            lang_id_by_segment=lang_ids,
            meta=_meta(
                producer=component_name,
                producer_version=producer_version,
                schema_version=schema_version,
                status=status,
                extra={
                    **(extra_meta or {}),
                    **({"error": error} if error else {}),
                    "empty_reason": empty_reason,
                },
            ),
        )
        return out_path

    if component_name == "speaker_diarization_extractor":
        speaker_segments = payload.get("speaker_segments") or []
        if not isinstance(speaker_segments, list):
            speaker_segments = []
        speaker_ids = np.asarray(payload.get("speaker_ids") or [], dtype=np.int32).reshape(-1)
        emb = payload.get("speaker_embeddings_mean")
        if emb is None:
            emb_arr = np.zeros((0, 0), dtype=np.float32)
        else:
            emb_arr = np.asarray(emb, dtype=np.float32)
            if emb_arr.ndim != 2:
                emb_arr = emb_arr.reshape(emb_arr.shape[0], -1) if emb_arr.size else np.zeros((0, 0), dtype=np.float32)

        add("speaker_count", payload.get("speaker_count"))
        add("duration_sec", payload.get("duration"))
        add("segments_count", payload.get("segments_count"))
        add("sample_rate", payload.get("sample_rate"))
        add("rms", payload.get("rms"))
        add("peak", payload.get("peak"))
        add("model_name", payload.get("model_name"))

        _atomic_save_npz(
            out_path,
            feature_names=np.asarray(feature_names, dtype=object),
            feature_values=np.asarray(feature_values, dtype=np.float32),
            speaker_segments=np.asarray(speaker_segments, dtype=object),
            speaker_ids=speaker_ids,
            speaker_embeddings_mean=emb_arr,
            speaker_stats=np.asarray(payload.get("speaker_stats") or {}, dtype=object),
            segment_start_sec=np.asarray(payload.get("segment_start_sec") or [], dtype=np.float32).reshape(-1),
            segment_end_sec=np.asarray(payload.get("segment_end_sec") or [], dtype=np.float32).reshape(-1),
            segment_center_sec=np.asarray(payload.get("segment_center_sec") or [], dtype=np.float32).reshape(-1),
            meta=_meta(
                producer=component_name,
                producer_version=producer_version,
                schema_version=schema_version,
                status=status,
                extra={
                    **(extra_meta or {}),
                    **({"error": error} if error else {}),
                    "empty_reason": empty_reason,
                },
            ),
        )
        return out_path

    if component_name == "emotion_diarization_extractor":
        probs = payload.get("emotion_probs")
        if probs is None:
            probs_arr = np.zeros((0, 0), dtype=np.float32)
        else:
            probs_arr = np.asarray(probs, dtype=np.float32)
            if probs_arr.ndim != 2:
                probs_arr = probs_arr.reshape(probs_arr.shape[0], -1) if probs_arr.size else np.zeros((0, 0), dtype=np.float32)

        emo_id = np.asarray(payload.get("emotion_id") or [], dtype=np.int32).reshape(-1)
        emo_conf = np.asarray(payload.get("emotion_confidence") or [], dtype=np.float32).reshape(-1)
        mean_probs = np.asarray(payload.get("emotion_mean_probs") or [], dtype=np.float32).reshape(-1)
        labels = payload.get("emotion_labels") or []
        if not isinstance(labels, list):
            labels = []

        add("segments_count", payload.get("segments_count"))
        add("sample_rate", payload.get("sample_rate"))
        add("emotion_entropy", payload.get("emotion_entropy"))
        add("dominant_emotion_id", payload.get("dominant_emotion_id"))
        add("dominant_emotion_prob", payload.get("dominant_emotion_prob"))
        add("rms", payload.get("rms"))
        add("peak", payload.get("peak"))
        add("model_name", payload.get("model_name"))

        _atomic_save_npz(
            out_path,
            feature_names=np.asarray(feature_names, dtype=object),
            feature_values=np.asarray(feature_values, dtype=np.float32),
            emotion_probs=probs_arr,
            emotion_id=emo_id,
            emotion_confidence=emo_conf,
            emotion_mean_probs=mean_probs,
            emotion_labels=np.asarray(labels, dtype=object),
            segment_start_sec=np.asarray(payload.get("segment_start_sec") or [], dtype=np.float32).reshape(-1),
            segment_end_sec=np.asarray(payload.get("segment_end_sec") or [], dtype=np.float32).reshape(-1),
            segment_center_sec=np.asarray(payload.get("segment_center_sec") or [], dtype=np.float32).reshape(-1),
            meta=_meta(
                producer=component_name,
                producer_version=producer_version,
                schema_version=schema_version,
                status=status,
                extra={
                    **(extra_meta or {}),
                    **({"error": error} if error else {}),
                    "empty_reason": empty_reason,
                },
            ),
        )
        return out_path

    if component_name == "source_separation_extractor":
        share_seq = payload.get("share_sequence")
        if share_seq is None:
            share_seq_arr = np.zeros((0, 4), dtype=np.float32)
        else:
            share_seq_arr = np.asarray(share_seq, dtype=np.float32)
            if share_seq_arr.ndim != 2:
                share_seq_arr = share_seq_arr.reshape(share_seq_arr.shape[0], -1) if share_seq_arr.size else np.zeros((0, 4), dtype=np.float32)
        energy_seq = payload.get("energy_sequence")
        if energy_seq is None:
            energy_seq_arr = np.zeros((0, 4), dtype=np.float32)
        else:
            energy_seq_arr = np.asarray(energy_seq, dtype=np.float32)
            if energy_seq_arr.ndim != 2:
                energy_seq_arr = energy_seq_arr.reshape(energy_seq_arr.shape[0], -1) if energy_seq_arr.size else np.zeros((0, 4), dtype=np.float32)

        share_mean = np.asarray(payload.get("share_mean") or [], dtype=np.float32).reshape(-1)
        share_std = np.asarray(payload.get("share_std") or [], dtype=np.float32).reshape(-1)
        src_order = payload.get("source_order") or ["vocals", "drums", "bass", "other"]
        if not isinstance(src_order, list):
            src_order = ["vocals", "drums", "bass", "other"]

        # Flatten mean shares into feature vector for compatibility
        if share_mean.size >= 4:
            add("share_vocals_mean", float(share_mean[0]))
            add("share_drums_mean", float(share_mean[1]))
            add("share_bass_mean", float(share_mean[2]))
            add("share_other_mean", float(share_mean[3]))
        add("segments_count", payload.get("segments_count"))
        add("sample_rate", payload.get("sample_rate"))
        add("model_name", payload.get("model_name"))

        _atomic_save_npz(
            out_path,
            feature_names=np.asarray(feature_names, dtype=object),
            feature_values=np.asarray(feature_values, dtype=np.float32),
            share_sequence=share_seq_arr,
            energy_sequence=energy_seq_arr,
            share_mean=share_mean,
            share_std=share_std,
            source_order=np.asarray(src_order, dtype=object),
            segment_start_sec=np.asarray(payload.get("segment_start_sec") or [], dtype=np.float32).reshape(-1),
            segment_end_sec=np.asarray(payload.get("segment_end_sec") or [], dtype=np.float32).reshape(-1),
            segment_center_sec=np.asarray(payload.get("segment_center_sec") or [], dtype=np.float32).reshape(-1),
            meta=_meta(
                producer=component_name,
                producer_version=producer_version,
                schema_version=schema_version,
                status=status,
                extra={
                    **(extra_meta or {}),
                    **({"error": error} if error else {}),
                    "empty_reason": empty_reason,
                },
            ),
        )
        return out_path

    if component_name == "speech_analysis_extractor":
        add("duration_sec", payload.get("duration_sec"))
        add("sample_rate", payload.get("sample_rate"))
        add("speaker_count", payload.get("speaker_count"))
        add("dominant_speaker_share", payload.get("dominant_speaker_share"))
        add("asr_segments_count", payload.get("asr_segments_count"))
        add("asr_token_total", payload.get("asr_token_total"))
        add("asr_token_mean", payload.get("asr_token_mean"))
        add("asr_token_std", payload.get("asr_token_std"))
        add("asr_token_density_per_sec", payload.get("asr_token_density_per_sec"))
        add("diar_segments_count", payload.get("diar_segments_count"))
        add("pitch_enabled", payload.get("pitch_enabled"))
        add("pitch_f0_mean", payload.get("pitch_f0_mean"))
        add("pitch_f0_std", payload.get("pitch_f0_std"))

        _atomic_save_npz(
            out_path,
            feature_names=np.asarray(feature_names, dtype=object),
            feature_values=np.asarray(feature_values, dtype=np.float32),
            asr_lang_id_by_segment=np.asarray(payload.get("asr_lang_id_by_segment") or [], dtype=np.int32).reshape(-1),
            speaker_ids=np.asarray(payload.get("speaker_ids") or [], dtype=np.int32).reshape(-1),
            meta=_meta(
                producer=component_name,
                producer_version=producer_version,
                schema_version=schema_version,
                status=status,
                extra={
                    **(extra_meta or {}),
                    **({"error": error} if error else {}),
                    "empty_reason": empty_reason,
                },
            ),
        )
        return out_path

    # Generic fallback: dump scalars into feature vector and store raw payload as object
    for k, v in payload.items():
        if isinstance(v, (int, float, np.integer, np.floating)) or v is None:
            add(str(k), v)
    _atomic_save_npz(
        out_path,
        feature_names=np.asarray(feature_names, dtype=object),
        feature_values=np.asarray(feature_values, dtype=np.float32),
        payload=np.asarray(payload, dtype=object),
        meta=_meta(
            producer=component_name,
            producer_version=producer_version,
            schema_version=schema_version,
            status=status,
            extra={
                **(extra_meta or {}),
                **({"error": error} if error else {}),
                "empty_reason": empty_reason,
            },
        ),
    )
    return out_path


def _parse_extractors_arg(s: str) -> Tuple[list[str], list[str]]:
    requested = [x.strip() for x in (s or "").split(",") if x.strip()]
    if not requested:
        requested = ["clap", "tempo", "loudness"]

    # AudioProcessor internal registry keys -> canonical component names
    key_to_component = {
        "clap": "clap_extractor",
        "tempo": "tempo_extractor",
        "loudness": "loudness_extractor",
        "asr": "asr_extractor",
        "speaker_diarization": "speaker_diarization_extractor",
        "emotion_diarization": "emotion_diarization_extractor",
        "source_separation": "source_separation_extractor",
        "speech_analysis": "speech_analysis_extractor",
    }
    keys: list[str] = []
    comps: list[str] = []
    for k in requested:
        if k not in key_to_component:
            raise ValueError(f"Unknown audio extractor key: {k}. Expected one of: {sorted(key_to_component.keys())}")
        keys.append(k)
        comps.append(key_to_component[k])
    return keys, comps


def main() -> int:
    ap_root = Path(__file__).resolve().parent
    repo_root = ap_root.parent

    # AudioProcessor/src imports
    src_path = ap_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    # VisualProcessor utils imports (manifest + validator)
    vp_root = repo_root / "VisualProcessor"
    if str(vp_root) not in sys.path:
        sys.path.insert(0, str(vp_root))

    from src.core.main_processor import MainProcessor  # type: ignore
    from utils.manifest import RunManifest, ManifestComponent  # type: ignore
    from utils.artifact_validator import validate_npz  # type: ignore

    parser = argparse.ArgumentParser(description="AudioProcessor CLI (per-run NPZ artifacts)")
    parser.add_argument("--video-path", type=str, required=True, help="Video path (legacy). If --frames-dir is provided, audio is taken from frames-dir/audio/audio.wav")
    parser.add_argument("--frames-dir", type=str, default=None, help="Segmenter output dir for this video: <Segmenter/output>/<video_id>")
    parser.add_argument("--rs-base", type=str, required=True, help="Base result_store path (will create per-run subdir)")
    parser.add_argument(
        "--run-rs-path",
        type=str,
        default=None,
        help="Explicit per-run result_store directory (overrides --rs-base/platform/video/run). "
             "Expected: <rs_base>/<platform_id>/<video_id>/<run_id>",
    )
    parser.add_argument("--platform-id", type=str, default="youtube")
    parser.add_argument("--video-id", type=str, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--sampling-policy-version", type=str, default="v1")
    parser.add_argument(
        "--config-hash",
        type=str,
        default=None,
        help="Optional config hash propagated by DataProcessor (for idempotency). If not provided, will be derived from CLI args.",
    )
    parser.add_argument("--dataprocessor-version", type=str, default="unknown")

    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument(
        "--extractors",
        type=str,
        default="clap,tempo,loudness",
        help=(
            "Comma-separated keys. Baseline Tier-0: clap,tempo,loudness. "
            "Additional: asr,speaker_diarization,emotion_diarization,source_separation,speech_analysis"
        ),
    )
    parser.add_argument("--asr-model-size", type=str, default="small", choices=["small", "medium", "large"], help="Whisper model size (Triton-backed via ModelManager)")
    parser.add_argument("--diarization-model-size", type=str, default="small", choices=["small", "large"], help="Speaker diarization embedding model size (Triton via ModelManager)")
    parser.add_argument("--emotion-model-size", type=str, default="small", choices=["small", "large"], help="Emotion diarization model size (Triton via ModelManager)")
    parser.add_argument("--source-separation-model-size", type=str, default="small", choices=["small", "medium", "large"], help="Source separation model size (Triton via ModelManager)")
    parser.add_argument("--speech-analysis-pitch", action="store_true", help="Enable pitch inside speech_analysis (full-audio, CPU-heavy)")
    parser.add_argument("--write-legacy-manifest", action="store_true", help="Also write legacy AudioProcessor JSON manifest into tmp dir")
    args = parser.parse_args()

    video_id = args.video_id or os.path.splitext(os.path.basename(args.video_path))[0]
    run_id = args.run_id or uuid.uuid4().hex[:12]

    config_hash = args.config_hash
    if not config_hash:
        # Best-effort: stabilize idempotency for audio-only runs.
        cfg_dump = json.dumps(
            {
                "device": args.device,
                "extractors": args.extractors,
                "asr_model_size": args.asr_model_size,
                "diarization_model_size": args.diarization_model_size,
                "emotion_model_size": args.emotion_model_size,
                "source_separation_model_size": args.source_separation_model_size,
                "sampling_policy_version": args.sampling_policy_version,
                "speech_analysis_pitch": bool(args.speech_analysis_pitch),
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        config_hash = _sha256_text(cfg_dump)[:16]

    run_rs_path = os.path.abspath(args.run_rs_path) if args.run_rs_path else os.path.join(os.path.abspath(args.rs_base), args.platform_id, video_id, run_id)
    os.makedirs(run_rs_path, exist_ok=True)

    manifest_path = os.path.join(run_rs_path, "manifest.json")
    manifest = RunManifest(
        path=manifest_path,
        run_meta={
            "platform_id": args.platform_id,
            "video_id": video_id,
            "run_id": run_id,
            "config_hash": config_hash,
            "sampling_policy_version": args.sampling_policy_version,
            "dataprocessor_version": str(args.dataprocessor_version),
            "created_at": _utc_iso_now(),
        },
    )

    extractor_keys, component_names = _parse_extractors_arg(args.extractors)

    # Resolve CLAP model meta via ModelManager (if available) for reproducibility.
    clap_model_used = None
    try:
        from dp_models import get_global_model_manager  # type: ignore

        mm = get_global_model_manager()
        spec = mm.get_spec(model_name="laion_clap")
        device, precision, runtime, engine, weights_digest, _ = mm.resolve(spec)
        clap_model_used = {
            "model_name": str(spec.model_name),
            "model_version": str(getattr(spec, "model_version", "unknown") or "unknown"),
            "weights_digest": str(weights_digest or "unknown"),
            "runtime": str(runtime),
            "engine": str(engine),
            "precision": str(precision),
            # device in ModelManager is auto-picked; we still report actual extractor device for execution.
            "device": str(device),
        }
    except Exception:
        clap_model_used = None

    # Resolve Whisper(triton) + shared tokenizer meta for ASR reproducibility.
    asr_model_used = None
    tokenizer_model_used = None
    try:
        from dp_models import get_global_model_manager  # type: ignore

        mm = get_global_model_manager()
        whisper_spec_name = f"whisper_{str(args.asr_model_size).strip().lower()}_triton"
        wspec = mm.get_spec(model_name=whisper_spec_name)
        device, precision, runtime, engine, weights_digest, _ = mm.resolve(wspec)
        asr_model_used = {
            "model_name": str(wspec.model_name),
            "model_version": str(getattr(wspec, "model_version", "unknown") or "unknown"),
            "weights_digest": str(weights_digest or "unknown"),
            "runtime": str(runtime),
            "engine": str(engine),
            "precision": str(precision),
            "device": str(device),
        }
        tspec = mm.get_spec(model_name="shared_tokenizer_v1")
        _d2, _p2, rt2, eng2, wd2, _arts = mm.resolve(tspec)
        tokenizer_model_used = {
            "model_name": str(tspec.model_name),
            "model_version": str(getattr(tspec, "model_version", "unknown") or "unknown"),
            "weights_digest": str(wd2 or "unknown"),
            "runtime": str(rt2),
            "engine": str(eng2),
            "precision": str(getattr(tspec, "precision", "unknown") or "unknown"),
            "device": "cpu",
        }
    except Exception:
        asr_model_used = None
        tokenizer_model_used = None

    # Resolve speaker diarization model meta via ModelManager.
    diar_model_used = None
    try:
        from dp_models import get_global_model_manager  # type: ignore

        mm = get_global_model_manager()
        diar_spec_name = f"speaker_diarization_{str(args.diarization_model_size).strip().lower()}_triton"
        dspec = mm.get_spec(model_name=diar_spec_name)
        device, precision, runtime, engine, weights_digest, _ = mm.resolve(dspec)
        diar_model_used = {
            "model_name": str(dspec.model_name),
            "model_version": str(getattr(dspec, "model_version", "unknown") or "unknown"),
            "weights_digest": str(weights_digest or "unknown"),
            "runtime": str(runtime),
            "engine": str(engine),
            "precision": str(precision),
            "device": str(device),
        }
    except Exception:
        diar_model_used = None

    # Resolve emotion diarization model meta via ModelManager.
    emo_model_used = None
    try:
        from dp_models import get_global_model_manager  # type: ignore

        mm = get_global_model_manager()
        emo_spec_name = f"emotion_diarization_{str(args.emotion_model_size).strip().lower()}_triton"
        espec = mm.get_spec(model_name=emo_spec_name)
        device, precision, runtime, engine, weights_digest, _ = mm.resolve(espec)
        emo_model_used = {
            "model_name": str(espec.model_name),
            "model_version": str(getattr(espec, "model_version", "unknown") or "unknown"),
            "weights_digest": str(weights_digest or "unknown"),
            "runtime": str(runtime),
            "engine": str(engine),
            "precision": str(precision),
            "device": str(device),
        }
    except Exception:
        emo_model_used = None

    # Resolve source separation model meta via ModelManager.
    sep_model_used = None
    try:
        from dp_models import get_global_model_manager  # type: ignore

        mm = get_global_model_manager()
        sep_spec_name = f"source_separation_{str(args.source_separation_model_size).strip().lower()}_triton"
        sspec = mm.get_spec(model_name=sep_spec_name)
        device, precision, runtime, engine, weights_digest, _ = mm.resolve(sspec)
        sep_model_used = {
            "model_name": str(sspec.model_name),
            "model_version": str(getattr(sspec, "model_version", "unknown") or "unknown"),
            "weights_digest": str(weights_digest or "unknown"),
            "runtime": str(runtime),
            "engine": str(engine),
            "precision": str(precision),
            "device": str(device),
        }
    except Exception:
        sep_model_used = None

    # Keep all AudioProcessor internal temp outputs inside the run folder (debuggable, but not source-of-truth).
    tmp_dir = os.path.join(run_rs_path, "_tmp_audio")
    os.makedirs(tmp_dir, exist_ok=True)

    processor = MainProcessor(
        device=args.device,
        asr_model_size=str(args.asr_model_size),
        diarization_model_size=str(args.diarization_model_size),
        emotion_model_size=str(args.emotion_model_size),
        source_separation_model_size=str(args.source_separation_model_size),
        speech_analysis_pitch_enabled=bool(args.speech_analysis_pitch),
        enabled_extractors=extractor_keys,
        save_debug_results=False,
        write_legacy_manifest=bool(args.write_legacy_manifest),
    )

    # Resolve Segmenter-produced audio + segments (new contract).
    frames_dir = os.path.abspath(args.frames_dir) if args.frames_dir else None
    audio_path = None
    segments_json = None
    segments_payload = None
    if frames_dir:
        audio_path = os.path.join(frames_dir, "audio", "audio.wav")
        segments_json = os.path.join(frames_dir, "audio", "segments.json")
        if not os.path.exists(audio_path):
            raise RuntimeError(f"AudioProcessor | missing required audio file: {audio_path}")
        if not os.path.exists(segments_json):
            raise RuntimeError(f"AudioProcessor | missing required segments.json: {segments_json}")
        with open(segments_json, "r", encoding="utf-8") as f:
            segments_payload = json.load(f) or {}

        if str(segments_payload.get("schema_version")) != "audio_segments_v1":
            raise RuntimeError(f"AudioProcessor | unsupported segments.json schema: {segments_payload.get('schema_version')}")

        families = segments_payload.get("families") or {}
        primary = ((families.get("primary") or {}) if isinstance(families, dict) else {}) or {}
        tempo_f = ((families.get("tempo") or {}) if isinstance(families, dict) else {}) or {}
        asr_f = ((families.get("asr") or {}) if isinstance(families, dict) else {}) or {}
        diar_f = ((families.get("diarization") or {}) if isinstance(families, dict) else {}) or {}
        emo_f = ((families.get("emotion") or {}) if isinstance(families, dict) else {}) or {}
        sep_f = ((families.get("source_separation") or {}) if isinstance(families, dict) else {}) or {}
        # speech_analysis uses both ASR and diarization families
        primary_segments = primary.get("segments") or []
        tempo_segments = tempo_f.get("segments") or []
        asr_segments = asr_f.get("segments") or []
        diar_segments = diar_f.get("segments") or []
        emo_segments = emo_f.get("segments") or []
        sep_segments = sep_f.get("segments") or []
        if not isinstance(primary_segments, list) or not primary_segments:
            raise RuntimeError("AudioProcessor | segments.json missing families.primary.segments (no-fallback)")
        if not isinstance(tempo_segments, list) or not tempo_segments:
            raise RuntimeError("AudioProcessor | segments.json missing families.tempo.segments (no-fallback)")
        if ("asr" in extractor_keys) and (not isinstance(asr_segments, list) or not asr_segments):
            raise RuntimeError("AudioProcessor | segments.json missing families.asr.segments (no-fallback)")
        if ("speaker_diarization" in extractor_keys) and (not isinstance(diar_segments, list) or not diar_segments):
            raise RuntimeError("AudioProcessor | segments.json missing families.diarization.segments (no-fallback)")
        if ("emotion_diarization" in extractor_keys) and (not isinstance(emo_segments, list) or not emo_segments):
            raise RuntimeError("AudioProcessor | segments.json missing families.emotion.segments (no-fallback)")
        if ("source_separation" in extractor_keys) and (not isinstance(sep_segments, list) or not sep_segments):
            raise RuntimeError("AudioProcessor | segments.json missing families.source_separation.segments (no-fallback)")
    else:
        # Legacy mode: AudioProcessor will extract audio from video itself.
        audio_path = args.video_path

    started_at = _utc_iso_now()
    t0 = time.time()
    if frames_dir:
        # New mode: run extractors directly on Segmenter audio, using Segmenter segments.
        extractor_results = {}
        for key in extractor_keys:
            extractor = processor.extractors.get(key)
            if extractor is None:
                extractor_results[key] = {"success": False, "payload": None, "error": "extractor_not_available", "device_used": "unknown"}
                continue
            try:
                if key == "clap" and segments_payload is not None:
                    r = extractor.run_segments(audio_path, tmp_dir, primary_segments)  # type: ignore[attr-defined]
                elif key == "loudness" and segments_payload is not None:
                    r = extractor.run_segments(audio_path, tmp_dir, primary_segments)  # type: ignore[attr-defined]
                elif key == "tempo" and segments_payload is not None:
                    r = extractor.run_segments(audio_path, tmp_dir, tempo_segments)  # type: ignore[attr-defined]
                elif key == "asr" and segments_payload is not None:
                    r = extractor.run_segments(audio_path, tmp_dir, asr_segments)  # type: ignore[attr-defined]
                elif key == "speaker_diarization" and segments_payload is not None:
                    r = extractor.run_segments(audio_path, tmp_dir, diar_segments)  # type: ignore[attr-defined]
                elif key == "emotion_diarization" and segments_payload is not None:
                    r = extractor.run_segments(audio_path, tmp_dir, emo_segments)  # type: ignore[attr-defined]
                elif key == "source_separation" and segments_payload is not None:
                    r = extractor.run_segments(audio_path, tmp_dir, sep_segments)  # type: ignore[attr-defined]
                elif key == "speech_analysis" and segments_payload is not None:
                    # Needs multiple families; call bundle API.
                    r = extractor.run_bundle(audio_path, tmp_dir, asr_segments=asr_segments, diar_segments=diar_segments)  # type: ignore[attr-defined]
                else:
                    r = extractor.run(audio_path, tmp_dir)
                extractor_results[key] = {
                    "success": bool(r.success),
                    "payload": r.payload if isinstance(r.payload, dict) else None,
                    "error": r.error,
                    "processing_time": r.processing_time,
                    "device_used": r.device_used,
                }
            except Exception as e:
                extractor_results[key] = {"success": False, "payload": None, "error": str(e), "device_used": getattr(extractor, "device", "unknown")}

        results = {
            "extractor_results": extractor_results,
            "extracted_audio_path": audio_path,
        }
    else:
        results = processor.process_video(args.video_path, tmp_dir, extractor_names=extractor_keys, extract_audio=True)
    finished_at = _utc_iso_now()
    duration_ms = int((time.time() - t0) * 1000)

    extractor_results = (results or {}).get("extractor_results") or {}
    extracted_audio_path = (results or {}).get("extracted_audio_path")
    audio_present = bool(isinstance(extracted_audio_path, str) and extracted_audio_path and os.path.exists(extracted_audio_path))
    audio_empty_reason = None if audio_present else "audio_missing_or_extract_failed"
    # Map internal keys -> payloads for saving.
    key_to_component = dict(zip(extractor_keys, component_names))

    overall_ok = True

    for key, component_name in key_to_component.items():
        r = extractor_results.get(key) or {}
        success = bool(r.get("success"))
        payload = r.get("payload") if isinstance(r.get("payload"), dict) else None
        err = r.get("error")

        # Normalize missing result as error.
        if key not in extractor_results:
            success = False
            err = f"missing extractor result for key={key}"

        if not audio_present:
            # Missing audio is a normal empty case (e.g., silent videos).
            status = "empty"
            empty_reason = audio_empty_reason
        else:
            # Allow extractors to declare valid empty outputs explicitly.
            if success and isinstance(payload, dict) and str(payload.get("status") or "") == "empty":
                status = "empty"
                empty_reason = str(payload.get("empty_reason") or "empty")
            else:
                status = "ok" if success else "error"
                empty_reason = None

        # Save NPZ artifact regardless (ok or error) so downstream can see status in meta.
        producer_version = getattr(processor.extractors.get(key), "version", None) or "unknown"
        artifact_path = _save_component_npz(
            run_rs_path=run_rs_path,
            component_name=component_name,
            payload=payload,
            status=status,
            error=str(err) if err else None,
            empty_reason=empty_reason,
            producer_version=str(producer_version),
            schema_version="audio_npz_v2",
            extra_meta={
                # Required run identity fields (baseline contract)
                "platform_id": args.platform_id,
                "video_id": video_id,
                "run_id": run_id,
                "config_hash": config_hash,
                "sampling_policy_version": args.sampling_policy_version,
                "dataprocessor_version": str(args.dataprocessor_version),
                "device_used": r.get("device_used", args.device),
                "source_video_path": os.path.abspath(args.video_path),
                "audio_present": audio_present,
                "audio_segments_schema": (segments_payload.get("schema_version") if isinstance(segments_payload, dict) else None),
                # PR-3
                "models_used": (
                    [
                        {
                            **(clap_model_used or {
                                "model_name": "laion_clap",
                                "model_version": "unknown",
                                "weights_digest": "unknown",
                                "runtime": "inprocess",
                                "engine": "torch",
                                "precision": "fp32",
                                "device": str(args.device),
                            }),
                            # actual execution device
                            "device": str(r.get("device_used", args.device)),
                        }
                    ]
                    if component_name == "clap_extractor"
                    else (
                        (
                            ([m for m in [asr_model_used, tokenizer_model_used] if isinstance(m, dict)] if component_name == "asr_extractor" else
                             ([m for m in [diar_model_used] if isinstance(m, dict)] if component_name == "speaker_diarization_extractor" else
                              ([m for m in [emo_model_used] if isinstance(m, dict)] if component_name == "emotion_diarization_extractor" else
                               ([m for m in [sep_model_used] if isinstance(m, dict)] if component_name == "source_separation_extractor" else []))))
                        )
                    ),
                ),
            },
        )

        v_ok, issues, meta = validate_npz(artifact_path)
        notes = None
        if not v_ok:
            status = "error"
            notes = "artifact validation failed: " + "; ".join(i.message for i in issues[:5])
            overall_ok = False
        # empty is acceptable (do not fail the run)
        if status == "error":
            overall_ok = False

        error_code = None
        if status == "error":
            error_code = "artifact_validation_failed" if not v_ok else "exception"

        warnings = None
        if notes:
            warnings = [notes]

        manifest.upsert_component(
            ManifestComponent(
                name=component_name,
                kind="audio",
                status=status,
                started_at=started_at,
                finished_at=finished_at,
                duration_ms=duration_ms,
                artifacts=[{"path": artifact_path, "type": "npz"}],
                error=str(err) if err else None,
                error_code=error_code,
                warnings=warnings,
                notes=notes,
                device_used=(meta or {}).get("device_used") if isinstance(meta, dict) else r.get("device_used", args.device),
                producer_version=(meta or {}).get("producer_version") if isinstance(meta, dict) else None,
                schema_version=(meta or {}).get("schema_version") if isinstance(meta, dict) else None,
            )
        )

    return 0 if overall_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())


