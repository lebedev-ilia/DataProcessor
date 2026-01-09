from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Optional


def model_used(
    *,
    model_name: str,
    model_version: str = "unknown",
    weights_digest: str = "unknown",
    runtime: str = "inprocess",
    engine: str = "unknown",
    precision: str = "unknown",
    device: str = "unknown",
) -> Dict[str, Any]:
    """
    Canonical `models_used[]` entry compatible with current repo-wide schema
    (see `common/meta_builder.py`).
    """
    return {
        "model_name": str(model_name),
        "model_version": str(model_version),
        "weights_digest": str(weights_digest),
        "runtime": str(runtime),
        "engine": str(engine),
        "precision": str(precision),
        "device": str(device),
    }


def _canonicalize_models_used(models_used: Any) -> List[Dict[str, Any]]:
    if models_used is None:
        return []
    if not isinstance(models_used, list):
        return []

    out: List[Dict[str, Any]] = []
    for m in models_used:
        if not isinstance(m, dict):
            continue
        out.append(
            model_used(
                model_name=str(m.get("model_name") or "unknown"),
                model_version=str(m.get("model_version") or "unknown"),
                weights_digest=str(m.get("weights_digest") or "unknown"),
                runtime=str(m.get("runtime") or "unknown"),
                engine=str(m.get("engine") or "unknown"),
                precision=str(m.get("precision") or "unknown"),
                device=str(m.get("device") or "unknown"),
            )
        )

    # Stable ordering for determinism.
    out.sort(
        key=lambda d: (
            str(d.get("model_name") or ""),
            str(d.get("model_version") or ""),
            str(d.get("weights_digest") or ""),
            str(d.get("engine") or ""),
            str(d.get("precision") or ""),
            str(d.get("device") or ""),
            str(d.get("runtime") or ""),
        )
    )
    return out


def compute_model_signature(models_used: Any) -> str:
    """
    Deterministic signature derived from canonicalized `models_used[]`.
    IMPORTANT: must not include dynamic params like batch_size.
    """
    canon = _canonicalize_models_used(models_used)
    payload = json.dumps(canon, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def apply_models_meta(meta: Dict[str, Any], *, models_used: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Returns a copy of meta with:
      - models_used (canonicalized list)
      - model_signature (sha256 of canonical models_used)
    """
    out = dict(meta or {})
    canon = _canonicalize_models_used(models_used if models_used is not None else out.get("models_used"))
    out["models_used"] = canon
    out["model_signature"] = compute_model_signature(canon)
    return out


