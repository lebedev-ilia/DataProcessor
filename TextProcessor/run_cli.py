#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import uuid
import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _utc_iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _timestamp_now() -> str:
    return datetime.datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S-%f")


def _short_uuid() -> str:
    return uuid.uuid4().hex[:8]


def _sha256_text(s: str) -> str:
    import hashlib
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def _atomic_save_npz(path: str, **arrays: Any) -> None:
    target_dir = os.path.dirname(path)
    os.makedirs(target_dir, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(prefix=Path(path).name + ".", suffix=".npz", dir=target_dir)
    os.close(fd)
    try:
        np.savez_compressed(tmp_path, **arrays)
        os.replace(tmp_path, path)
    except Exception:
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        raise


def _meta(*, producer_version: str, status: str, schema_version: str, extra: Dict[str, Any]) -> np.ndarray:
    d = {
        "producer": "text_processor",
        "producer_version": producer_version,
        "schema_version": schema_version,
        "status": status,
        "created_at": _utc_iso_now(),
        **(extra or {}),
    }
    return np.asarray(d, dtype=object)


def _flatten_scalars(d: Any, prefix: str = "") -> Dict[str, float]:
    """
    Extract numeric scalars into a flat dict. Non-numeric values are ignored.
    """
    out: Dict[str, float] = {}
    if isinstance(d, dict):
        for k, v in d.items():
            key = f"{prefix}.{k}" if prefix else str(k)
            out.update(_flatten_scalars(v, key))
        return out
    if isinstance(d, list):
        return out
    if d is None:
        return out
    if isinstance(d, bool):
        out[prefix] = 1.0 if d else 0.0
        return out
    if isinstance(d, (int, float, np.integer, np.floating)):
        try:
            out[prefix] = float(d)
        except Exception:
            pass
        return out
    return out


def main() -> int:
    tp_root = Path(__file__).resolve().parent
    repo_root = tp_root.parent

    # TextProcessor/src imports
    src_path = tp_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    # VisualProcessor utils imports (manifest + validator)
    vp_root = repo_root / "VisualProcessor"
    if str(vp_root) not in sys.path:
        sys.path.insert(0, str(vp_root))

    from src.core.main_processor import MainProcessor, load_document_from_json  # type: ignore
    from utils.manifest import RunManifest, ManifestComponent  # type: ignore
    from utils.artifact_validator import validate_npz  # type: ignore

    parser = argparse.ArgumentParser(description="TextProcessor CLI (per-run NPZ artifacts)")
    parser.add_argument("--input-json", type=str, required=True, help="Path to VideoDocument JSON")
    parser.add_argument("--rs-base", type=str, required=True, help="Base result_store path (will create per-run subdir)")
    parser.add_argument("--platform-id", type=str, default="youtube")
    parser.add_argument("--video-id", type=str, default=None)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--sampling-policy-version", type=str, default="v1")
    parser.add_argument("--config-hash", type=str, default=None)

    parser.add_argument(
        "--enable-embeddings",
        action="store_true",
        help="If set, will also run GPU embedders (slower/heavier). Default is CPU-only extractors.",
    )
    args = parser.parse_args()

    video_id = args.video_id or os.path.splitext(os.path.basename(args.input_json))[0]
    run_id = args.run_id or uuid.uuid4().hex[:12]
    config_hash = args.config_hash
    if not config_hash:
        cfg_dump = json.dumps(
            {"enable_embeddings": bool(args.enable_embeddings), "sampling_policy_version": args.sampling_policy_version},
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

    # Default devices config: CPU-only by default (baseline-safe).
    devices_config: Dict[str, List[str]] = {
        "cpu": ["LexicalStatsExtractor", "TagsExtractor", "ASRTextProxyExtractor"],
    }
    if args.enable_embeddings:
        devices_config["gpu"] = [
            "TitleEmbedder",
            "DescriptionEmbedder",
            "TranscriptChunkEmbedder",
            "CommentsEmbedder",
            "HashtagEmbedder",
        ]
        devices_config["cpu2"] = [
            "TranscriptAggregatorExtractor",
            "CommentsAggregationExtractor",
            "EmbeddingStatsExtractor",
            "CosineMetricsExtractor",
            "EmbeddingSourceIdExtractor",
        ]

    started_at = _utc_iso_now()
    t0 = time.time()
    status = "ok"
    empty_reason = None
    err: Optional[str] = None
    payload: Dict[str, Any] = {}
    try:
        doc = load_document_from_json(os.path.abspath(args.input_json))
        processor = MainProcessor(devices_config=devices_config)
        payload = processor.run(doc) or {}
    except Exception as e:
        status = "error"
        err = str(e)
    finished_at = _utc_iso_now()
    duration_ms = int((time.time() - t0) * 1000)

    scalars = _flatten_scalars(payload)
    feature_names = np.asarray(sorted(scalars.keys()), dtype=object)
    feature_values = np.asarray([float(scalars[k]) for k in feature_names.tolist()], dtype=np.float32)

    component_name = "text_processor"
    comp_dir = os.path.join(run_rs_path, component_name)
    out_path = os.path.join(comp_dir, f"{_timestamp_now()}_{_short_uuid()}.npz")

    _atomic_save_npz(
        out_path,
        feature_names=feature_names,
        feature_values=feature_values,
        payload=np.asarray(payload, dtype=object),
        meta=_meta(
            producer_version=str(payload.get("version") or "unknown"),
            schema_version="text_npz_v1",
            status=status,
            extra={
                "platform_id": args.platform_id,
                "video_id": video_id,
                "run_id": run_id,
                "config_hash": config_hash,
                "sampling_policy_version": args.sampling_policy_version,
                "empty_reason": empty_reason,
                "error": err,
                "input_json": os.path.abspath(args.input_json),
                "enable_embeddings": bool(args.enable_embeddings),
            },
        ),
    )

    v_ok, issues, meta = validate_npz(out_path)
    notes = None
    if not v_ok:
        status = "error"
        notes = "artifact validation failed: " + "; ".join(i.message for i in issues[:5])

    manifest.upsert_component(
        ManifestComponent(
            name=component_name,
            kind="text",
            status=status,
            started_at=started_at,
            finished_at=finished_at,
            duration_ms=duration_ms,
            artifacts=[{"path": out_path, "type": "npz"}],
            error=err,
            notes=notes,
            producer_version=(meta or {}).get("producer_version") if isinstance(meta, dict) else None,
            schema_version=(meta or {}).get("schema_version") if isinstance(meta, dict) else None,
        )
    )

    return 0 if status != "error" else 2


if __name__ == "__main__":
    raise SystemExit(main())


