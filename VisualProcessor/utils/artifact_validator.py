from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


@dataclass
class ValidationIssue:
    level: str  # "error" | "warning"
    message: str


def _extract_meta_dict(npz: np.lib.npyio.NpzFile) -> Optional[Dict[str, Any]]:
    if "meta" not in npz.files:
        return None
    meta_arr = npz["meta"]
    # We store meta as object array with a dict inside.
    try:
        if isinstance(meta_arr, np.ndarray) and meta_arr.dtype == object:
            # common case: shape=(), or shape=(1,)
            if meta_arr.shape == ():
                meta = meta_arr.item()
            else:
                meta = meta_arr.flat[0].item() if hasattr(meta_arr.flat[0], "item") else meta_arr.flat[0]
            if isinstance(meta, dict):
                return meta
    except Exception:
        return None
    return None


def validate_frame_indices(arr: np.ndarray) -> List[ValidationIssue]:
    issues: List[ValidationIssue] = []
    if not isinstance(arr, np.ndarray):
        issues.append(ValidationIssue("error", "frame_indices is not a numpy array"))
        return issues
    if arr.ndim != 1:
        issues.append(ValidationIssue("error", f"frame_indices must be 1D, got ndim={arr.ndim}"))
        return issues
    if not np.issubdtype(arr.dtype, np.integer):
        issues.append(ValidationIssue("error", f"frame_indices dtype must be int, got {arr.dtype}"))
        return issues
    if arr.size == 0:
        issues.append(ValidationIssue("warning", "frame_indices is empty"))
        return issues
    if not np.all(arr[1:] >= arr[:-1]):
        issues.append(ValidationIssue("error", "frame_indices is not sorted non-decreasing"))
    if np.unique(arr).size != arr.size:
        issues.append(ValidationIssue("error", "frame_indices has duplicates"))
    return issues


def validate_npz(path: str, required_meta_keys: Optional[List[str]] = None) -> Tuple[bool, List[ValidationIssue], Dict[str, Any]]:
    """
    Returns: (ok, issues, extracted_meta)
    """
    issues: List[ValidationIssue] = []
    meta: Dict[str, Any] = {}
    if not os.path.exists(path):
        return False, [ValidationIssue("error", f"file does not exist: {path}")], meta

    try:
        npz = np.load(path, allow_pickle=True)
    except Exception as e:
        return False, [ValidationIssue("error", f"failed to load npz: {e}")], meta

    try:
        meta_dict = _extract_meta_dict(npz)
        if meta_dict is None:
            issues.append(ValidationIssue("error", "missing or invalid `meta` in npz"))
        else:
            meta = dict(meta_dict)
            # Baseline contract (strict-by-default).
            # Note: values may be None; we only require presence of keys.
            keys = required_meta_keys or [
                "producer",
                "producer_version",
                "schema_version",
                "created_at",
                "platform_id",
                "video_id",
                "run_id",
                "config_hash",
                "sampling_policy_version",
                "status",
                "empty_reason",
                # PR-3: model system baseline
                "models_used",
                "model_signature",
            ]
            for k in keys:
                if k not in meta:
                    issues.append(ValidationIssue("error", f"meta missing required key: {k}"))

        # Optional frame_indices key validation (if present)
        if "frame_indices" in npz.files:
            try:
                issues.extend(validate_frame_indices(npz["frame_indices"]))
            except Exception as e:
                issues.append(ValidationIssue("error", f"failed to validate frame_indices: {e}"))
    finally:
        try:
            npz.close()
        except Exception:
            pass

    ok = not any(i.level == "error" for i in issues)
    return ok, issues, meta


