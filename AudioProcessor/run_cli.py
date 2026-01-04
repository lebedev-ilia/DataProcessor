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

        _atomic_save_npz(
            out_path,
            feature_names=np.asarray(feature_names, dtype=object),
            feature_values=np.asarray(feature_values, dtype=np.float32),
            lufs_present=np.asarray(lufs_present, dtype=np.bool_),
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
        # Expect embeddings saved by the extractor into a .npy file.
        emb_path = payload.get("clap_embeddings_npy")
        emb_present = False
        emb = np.zeros((0,), dtype=np.float32)
        if isinstance(emb_path, str) and emb_path and os.path.exists(emb_path):
            try:
                emb = np.asarray(np.load(emb_path), dtype=np.float32).reshape(-1)
                emb_present = emb.size > 0
            except Exception:
                emb_present = False
                emb = np.zeros((0,), dtype=np.float32)

        add("embedding_dim", payload.get("embedding_dim"))
        add("sample_rate", payload.get("sample_rate"))
        add("clap_norm", payload.get("clap_norm"))
        add("clap_magnitude_mean", payload.get("clap_magnitude_mean"))
        add("clap_magnitude_std", payload.get("clap_magnitude_std"))
        add("clap_non_zero_count", payload.get("clap_non_zero_count"))

        _atomic_save_npz(
            out_path,
            feature_names=np.asarray(feature_names, dtype=object),
            feature_values=np.asarray(feature_values, dtype=np.float32),
            embedding=emb,
            embedding_present=np.asarray(emb_present, dtype=np.bool_),
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
    parser.add_argument("--video-path", type=str, required=True)
    parser.add_argument("--rs-base", type=str, required=True, help="Base result_store path (will create per-run subdir)")
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

    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--extractors", type=str, default="clap,tempo,loudness", help="Comma-separated keys: clap,tempo,loudness")
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
                "sampling_policy_version": args.sampling_policy_version,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        config_hash = _sha256_text(cfg_dump)[:16]

    run_rs_path = os.path.join(os.path.abspath(args.rs_base), args.platform_id, video_id, run_id)
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
            "created_at": _utc_iso_now(),
        },
    )

    extractor_keys, component_names = _parse_extractors_arg(args.extractors)

    # Keep all AudioProcessor internal temp outputs inside the run folder (debuggable, but not source-of-truth).
    tmp_dir = os.path.join(run_rs_path, "_tmp_audio")
    os.makedirs(tmp_dir, exist_ok=True)

    processor = MainProcessor(
        device=args.device,
        enabled_extractors=extractor_keys,
        save_debug_results=False,
        write_legacy_manifest=bool(args.write_legacy_manifest),
    )

    started_at = _utc_iso_now()
    t0 = time.time()
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
            schema_version="audio_npz_v1",
            extra_meta={
                # Required run identity fields (baseline contract)
                "platform_id": args.platform_id,
                "video_id": video_id,
                "run_id": run_id,
                "config_hash": config_hash,
                "sampling_policy_version": args.sampling_policy_version,
                "device_used": r.get("device_used", args.device),
                "source_video_path": os.path.abspath(args.video_path),
                "audio_present": audio_present,
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
                notes=notes,
                producer_version=(meta or {}).get("producer_version") if isinstance(meta, dict) else None,
                schema_version=(meta or {}).get("schema_version") if isinstance(meta, dict) else None,
            )
        )

    return 0 if overall_ok else 2


if __name__ == "__main__":
    raise SystemExit(main())


