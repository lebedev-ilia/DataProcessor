#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np


def _safe_float(x: Any) -> float:
    try:
        if x is None:
            return float("nan")
        return float(x)
    except Exception:
        return float("nan")


def _meta_from_npz(npz: np.lib.npyio.NpzFile) -> Dict[str, Any]:
    if "meta" not in npz.files:
        return {}
    arr = npz["meta"]
    try:
        if isinstance(arr, np.ndarray) and arr.dtype == object:
            if arr.shape == ():
                v = arr.item()
            else:
                first = arr.flat[0]
                v = first.item() if hasattr(first, "item") else first
            return dict(v) if isinstance(v, dict) else {}
    except Exception:
        return {}
    return {}


def _summarize_numeric_array(prefix: str, arr: np.ndarray) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if arr.size == 0:
        out[f"{prefix}__empty"] = 1.0
        return out

    x = arr
    if x.dtype == bool:
        # fraction true
        out[f"{prefix}__mean"] = float(np.mean(x.astype(np.float32)))
        return out

    if np.issubdtype(x.dtype, np.integer) or np.issubdtype(x.dtype, np.floating):
        xf = x.astype(np.float32, copy=False).reshape(-1)
        xf = xf[np.isfinite(xf)]
        if xf.size == 0:
            out[f"{prefix}__all_non_finite"] = 1.0
            return out
        out[f"{prefix}__mean"] = float(np.mean(xf))
        out[f"{prefix}__std"] = float(np.std(xf))
        out[f"{prefix}__min"] = float(np.min(xf))
        out[f"{prefix}__max"] = float(np.max(xf))
        out[f"{prefix}__p50"] = float(np.percentile(xf, 50))
        out[f"{prefix}__p90"] = float(np.percentile(xf, 90))
        out[f"{prefix}__count"] = float(xf.size)
    return out


def _flatten_obj_dict(prefix: str, d: Dict[str, Any]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    for k, v in d.items():
        key = f"{prefix}__{k}"
        if isinstance(v, (int, float, np.integer, np.floating)) or v is None:
            out[key] = _safe_float(v)
        elif isinstance(v, (list, tuple, np.ndarray)):
            try:
                arr = np.asarray(v)
                if arr.dtype == object:
                    continue
                out.update(_summarize_numeric_array(key, arr))
            except Exception:
                continue
        # ignore strings / nested dicts for baseline v0
    return out


def _extract_features_from_npz(component: str, path: str) -> Tuple[Dict[str, float], Dict[str, Any]]:
    feats: Dict[str, float] = {}
    meta: Dict[str, Any] = {}
    with np.load(path, allow_pickle=True) as npz:
        meta = _meta_from_npz(npz)

        # Preferred: feature_names + feature_values
        if "feature_names" in npz.files and "feature_values" in npz.files:
            try:
                names = npz["feature_names"]
                vals = npz["feature_values"]
                names_list = [str(x) for x in (names.tolist() if isinstance(names, np.ndarray) else list(names))]
                vals_arr = np.asarray(vals, dtype=np.float32).reshape(-1)
                for n, v in zip(names_list, vals_arr):
                    feats[f"{component}__{n}"] = float(v)
                return feats, meta
            except Exception:
                # fall through to generic path
                pass

        # Common: features dict saved as object array
        if "features" in npz.files:
            try:
                f_arr = npz["features"]
                if isinstance(f_arr, np.ndarray) and f_arr.dtype == object:
                    f_dict = f_arr.item() if f_arr.shape == () else f_arr.flat[0].item()
                    if isinstance(f_dict, dict):
                        feats.update(_flatten_obj_dict(component, f_dict))
            except Exception:
                pass

        # Generic: summarize numeric arrays / scalars
        for k in npz.files:
            if k in ("meta", "features", "feature_names", "feature_values"):
                continue
            arr = npz[k]
            # Skip huge embeddings-like tensors by name
            if k in ("embeddings", "embedding", "ocr_raw", "ocr_unique_elements"):
                continue
            if isinstance(arr, np.ndarray):
                if arr.dtype == object:
                    continue
                if arr.ndim == 0:
                    feats[f"{component}__{k}"] = _safe_float(arr.item())
                else:
                    feats.update(_summarize_numeric_array(f"{component}__{k}", arr))
    return feats, meta


def _iter_run_manifests(rs_base: str) -> Iterable[Tuple[str, Dict[str, Any]]]:
    base = Path(rs_base)
    if not base.exists():
        return
    # rs_base/<platform>/<video>/<run>/manifest.json
    for manifest_path in base.glob("*/*/*/manifest.json"):
        try:
            with open(manifest_path, "r", encoding="utf-8") as f:
                payload = json.load(f) or {}
            yield str(manifest_path), payload
        except Exception:
            continue


@dataclass
class Row:
    keys: Dict[str, Any]
    feats: Dict[str, float]


def build_rows(rs_base: str) -> List[Row]:
    rows: List[Row] = []
    for manifest_path, manifest in _iter_run_manifests(rs_base):
        run = manifest.get("run") or {}
        platform_id = run.get("platform_id") or ""
        video_id = run.get("video_id") or ""
        run_id = run.get("run_id") or ""
        config_hash = run.get("config_hash") or ""
        sampling_policy_version = run.get("sampling_policy_version") or ""

        keys = {
            "platform_id": platform_id,
            "video_id": video_id,
            "run_id": run_id,
            "config_hash": config_hash,
            "sampling_policy_version": sampling_policy_version,
            "manifest_path": manifest_path,
        }

        feats: Dict[str, float] = {}

        # component statuses
        comps = manifest.get("components") or []
        if isinstance(comps, list):
            for c in comps:
                if not isinstance(c, dict):
                    continue
                name = c.get("name")
                status = c.get("status")
                if not isinstance(name, str) or not name:
                    continue
                val = -1.0
                if status == "ok":
                    val = 1.0
                elif status == "empty":
                    val = 0.0
                feats[f"component_status__{name}"] = val

                # Try to load the first NPZ artifact listed
                arts = c.get("artifacts") or []
                npz_path = None
                if isinstance(arts, list):
                    for a in arts:
                        if isinstance(a, dict) and isinstance(a.get("path"), str) and str(a.get("path")).lower().endswith(".npz"):
                            npz_path = a.get("path")
                            break
                if npz_path and os.path.exists(npz_path):
                    comp_feats, _meta = _extract_features_from_npz(name, npz_path)
                    feats.update(comp_feats)

        rows.append(Row(keys=keys, feats=feats))
    return rows


def write_csv(rows: List[Row], out_csv: str) -> None:
    # Determine columns
    key_cols: List[str] = []
    feat_cols: List[str] = []
    for r in rows:
        for k in r.keys.keys():
            if k not in key_cols:
                key_cols.append(k)
        for k in r.feats.keys():
            if k not in feat_cols:
                feat_cols.append(k)

    fieldnames = key_cols + sorted(feat_cols)
    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)

    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            row = {**r.keys}
            row.update({k: r.feats.get(k, "") for k in feat_cols})
            w.writerow(row)


def main() -> int:
    p = argparse.ArgumentParser(description="Build training table (features) from per-run NPZ artifacts")
    p.add_argument("--rs-base", type=str, required=True, help="Base result_store directory")
    p.add_argument("--out-csv", type=str, required=True, help="Output CSV path")
    args = p.parse_args()

    rows = build_rows(args.rs_base)
    write_csv(rows, args.out_csv)
    print(f"[ok] wrote {len(rows)} rows -> {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


