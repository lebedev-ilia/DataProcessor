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
    # PR-3: model system baseline
    try:
        from src.utils.meta_builder import apply_models_meta  # type: ignore

        d = apply_models_meta(d, models_used=d.get("models_used"))
    except Exception:
        d.setdefault("models_used", [])
        d.setdefault("model_signature", "")
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


def _safe_text(s: Any, limit: int = 20000) -> str:
    """
    Best-effort string normalization for hashing/summaries.
    Never returns raw content into artifacts unless explicitly allowed by flags.
    """
    try:
        txt = str(s or "")
    except Exception:
        txt = ""
    txt = " ".join(txt.split())
    if len(txt) > limit:
        txt = txt[:limit]
    return txt


def _content_hash_for_document(doc: Any) -> str:
    """
    Privacy-safe stable hash for document content (title/description/transcripts/comments).
    Raw text is NOT stored; only the hash.
    """
    try:
        title = _safe_text(getattr(doc, "title", ""))
        desc = _safe_text(getattr(doc, "description", ""))
        transcripts = getattr(doc, "transcripts", {}) or {}
        transcript_join = ""
        if isinstance(transcripts, dict):
            transcript_join = " ".join(_safe_text(v) for v in transcripts.values() if v)
        comments = getattr(doc, "comments", []) or []
        c_join = ""
        if isinstance(comments, list):
            parts = []
            for c in comments[:2000]:
                if isinstance(c, dict):
                    parts.append(_safe_text(c.get("text")))
                else:
                    parts.append(_safe_text(getattr(c, "text", c)))
            c_join = " ".join([p for p in parts if p])
        payload = "\n".join([title, desc, transcript_join, c_join]).strip()
        return _sha256_text(payload)
    except Exception:
        return ""


def _payload_summary(*, doc: Any, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Produce a privacy-safe payload summary for NPZ.
    This must not contain raw user text by default.
    """
    results_by_extractor = payload.get("results_by_extractor") if isinstance(payload.get("results_by_extractor"), dict) else {}
    timings_by_extractor = payload.get("timings_by_extractor") if isinstance(payload.get("timings_by_extractor"), dict) else {}
    systems_by_extractor = payload.get("systems_by_extractor") if isinstance(payload.get("systems_by_extractor"), dict) else {}

    # Minimal input stats (no raw).
    try:
        title_len = len(_safe_text(getattr(doc, "title", "")))
        desc_len = len(_safe_text(getattr(doc, "description", "")))
        transcripts = getattr(doc, "transcripts", {}) or {}
        transcript_len = len(_safe_text(" ".join(str(v or "") for v in transcripts.values()))) if isinstance(transcripts, dict) else 0
        comments = getattr(doc, "comments", []) or []
        comments_count = len(comments) if isinstance(comments, list) else 0
    except Exception:
        title_len, desc_len, transcript_len, comments_count = 0, 0, 0, 0

    return {
        "schema_version": "text_payload_summary_v1",
        "version": payload.get("version"),
        "device": payload.get("device"),
        "content_hash": _content_hash_for_document(doc),
        "input_stats": {
            "title_len_chars": int(title_len),
            "description_len_chars": int(desc_len),
            "transcript_len_chars": int(transcript_len),
            "comments_count": int(comments_count),
        },
        "extractors": {
            "results_keys": sorted(list(results_by_extractor.keys())),
            "timings_keys": sorted(list(timings_by_extractor.keys())),
            "systems_keys": sorted(list(systems_by_extractor.keys())),
        },
        # keep timings only (safe); omit detailed per-extractor results to avoid leaking text.
        "timings_by_extractor": timings_by_extractor,
    }


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
    parser.add_argument(
        "--rs-base",
        type=str,
        default="./_runs/result_store",
        help="Base result_store path (per-run subdir will be created). Default matches DataProcessor baseline layout.",
    )
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
    parser.add_argument("--config-hash", type=str, default=None)
    parser.add_argument("--dataprocessor-version", type=str, default="unknown")

    parser.add_argument(
        "--enable-embeddings",
        action="store_true",
        help="If set, will also run GPU embedders (slower/heavier). Default is CPU-only extractors.",
    )
    parser.add_argument(
        "--store-raw-payload",
        action="store_true",
        help="Debug-only: store raw TextProcessor payload under _tmp_text/. NOT for production (privacy).",
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

    run_rs_path = (
        os.path.abspath(args.run_rs_path)
        if args.run_rs_path
        else os.path.join(os.path.abspath(args.rs_base), args.platform_id, video_id, run_id)
    )
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
    empty_reason: Optional[str] = None
    err: Optional[str] = None
    payload: Dict[str, Any] = {}
    doc = None
    try:
        doc = load_document_from_json(os.path.abspath(args.input_json))
        processor = MainProcessor(devices_config=devices_config)
        payload = processor.run(doc) or {}
    except Exception as e:
        status = "error"
        err = str(e)
    finished_at = _utc_iso_now()
    duration_ms = int((time.time() - t0) * 1000)

    # Empty semantics: valid "no text available" should be `status=empty`, not `ok`.
    if status == "ok":
        try:
            has_any_text = False
            if doc is not None:
                if _safe_text(getattr(doc, "title", "")) or _safe_text(getattr(doc, "description", "")):
                    has_any_text = True
                transcripts = getattr(doc, "transcripts", {}) or {}
                if isinstance(transcripts, dict) and any(_safe_text(v) for v in transcripts.values()):
                    has_any_text = True
                comments = getattr(doc, "comments", []) or []
                if isinstance(comments, list) and any(_safe_text(getattr(c, "text", "")) for c in comments):
                    has_any_text = True
            if not has_any_text:
                status = "empty"
                empty_reason = "no_text_available"
        except Exception:
            # If empty detection failed, keep ok (do not mask real errors).
            pass

    scalars = _flatten_scalars(payload)
    feature_names = np.asarray(sorted(scalars.keys()), dtype=object)
    feature_values = np.asarray([float(scalars[k]) for k in feature_names.tolist()], dtype=np.float32)

    component_name = "text_processor"
    comp_dir = os.path.join(run_rs_path, component_name)
    # Canonical single per-run artifact for TextProcessor (docs/contracts/ARTIFACTS_AND_SCHEMAS.md).
    # Deterministic filename avoids orphaned old artifacts on re-run.
    out_path = os.path.join(comp_dir, "text_features.npz")

    # Privacy: do not store raw payload by default.
    payload_summary = _payload_summary(doc=doc, payload=payload) if doc is not None else {"schema_version": "text_payload_summary_v1", "error": "doc_missing"}

    raw_payload_path = None
    if bool(args.store_raw_payload):
        try:
            tmp_dir = os.path.join(run_rs_path, "_tmp_text")
            os.makedirs(tmp_dir, exist_ok=True)
            raw_payload_path = os.path.join(tmp_dir, "raw_payload.json")
            tmp_path = raw_payload_path + ".tmp"
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, ensure_ascii=False, indent=2)
            os.replace(tmp_path, raw_payload_path)
        except Exception:
            raw_payload_path = None

    # Model meta (PR-10): resolve from ModelManager (local-only).
    models_used: list[dict] = []
    if bool(args.enable_embeddings):
        try:
            from dp_models import get_global_model_manager  # type: ignore

            mm = get_global_model_manager()
            spec = mm.get_spec(model_name="sentence-transformers/all-MiniLM-L6-v2")
            _dev, _prec, rt, eng, weights_digest, _arts = mm.resolve(spec)
            models_used = [
                {
                    "model_name": str(spec.model_name),
                    "model_version": str(getattr(spec, "model_version", "unknown") or "unknown"),
                    "weights_digest": str(weights_digest or "unknown"),
                    "runtime": str(rt),
                    "engine": str(eng),
                    "precision": "fp16",
                    "device": "cuda",
                }
            ]
        except Exception:
            models_used = []

    _atomic_save_npz(
        out_path,
        feature_names=feature_names,
        feature_values=feature_values,
        payload=np.asarray(payload_summary, dtype=object),
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
                "dataprocessor_version": str(args.dataprocessor_version),
                "empty_reason": empty_reason,
                "error": err,
                "input_json_basename": os.path.basename(str(args.input_json or "")),
                "enable_embeddings": bool(args.enable_embeddings),
                # PR-3
                "models_used": models_used,
                "raw_payload_path": os.path.abspath(raw_payload_path) if raw_payload_path else None,
            },
        ),
    )

    v_ok, issues, meta = validate_npz(out_path)
    notes = None
    if not v_ok:
        status = "error"
        notes = "artifact validation failed: " + "; ".join(i.message for i in issues[:5])

    error_code = None
    if status == "error":
        error_code = "artifact_validation_failed" if not v_ok else "exception"

    device_used = "cpu" if not args.enable_embeddings else "cuda"

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
            error_code=error_code,
            notes=notes,
            device_used=device_used,
            producer_version=(meta or {}).get("producer_version") if isinstance(meta, dict) else None,
            schema_version=(meta or {}).get("schema_version") if isinstance(meta, dict) else None,
        )
    )

    return 0 if status != "error" else 2


if __name__ == "__main__":
    raise SystemExit(main())


