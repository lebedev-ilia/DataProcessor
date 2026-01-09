#!/usr/bin/env python3
"""
Convert scripts/baseline/run_cut_detection_matrix.py output (matrix_results.json)
into docs/models_docs/resource_costs/*_costs_v1.json format.

This keeps the existing JSON schema used in this repo:
- resource_costs_cut_detection_v1
- resource_costs_cut_detection_soft_v1
- resource_costs_cut_detection_motion_v1
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime
from typing import Any, Dict, List, Optional


def _read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.write("\n")


def _git_repo_info(repo_root: str) -> Dict[str, Any]:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=repo_root,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        commit = "unknown"

    try:
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain"],
                cwd=repo_root,
                stderr=subprocess.DEVNULL,
                text=True,
            ).strip()
        )
    except Exception:
        dirty = False
    return {"commit": commit, "dirty": dirty}


def _schema_version_for(metric_key: str) -> str:
    mk = str(metric_key or "").lower()
    if "detect_soft_cuts" in mk:
        return "resource_costs_cut_detection_soft_v1"
    if "detect_motion_based_cuts" in mk:
        return "resource_costs_cut_detection_motion_v1"
    return "resource_costs_cut_detection_v1"


def _unit_for(metric_key: str) -> str:
    # These cut detectors operate on transitions between frames.
    return "frame_pair"


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser("render resource_costs JSON from matrix_results.json")
    ap.add_argument("--matrix-json", required=True, help="Path to matrix_results.json produced by run_cut_detection_matrix.py")
    ap.add_argument("--out", required=True, help="Output JSON path (e.g. docs/models_docs/resource_costs/cut_detection_costs_v1.json)")
    ap.add_argument("--repo-root", default=None, help="Repo root (auto by default).")
    args = ap.parse_args(argv)

    matrix_json = os.path.abspath(str(args.matrix_json))
    out_path = os.path.abspath(str(args.out))
    repo_root = os.path.abspath(str(args.repo_root)) if args.repo_root else os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))

    payload = _read_json(matrix_json)
    mat = payload.get("matrix") or {}
    results = payload.get("results") or []
    metric_key = str(mat.get("metric_key") or "")
    if not metric_key:
        raise RuntimeError("matrix_results.json missing matrix.metric_key")

    schema_version = _schema_version_for(metric_key)
    unit = _unit_for(metric_key)

    out: Dict[str, Any] = {
        "schema_version": schema_version,
        "generated_at": datetime.utcnow().isoformat(),
        "repo": _git_repo_info(repo_root),
        "generated_from": matrix_json,
        "matrix": mat,
        "costs": [],
    }

    for row in results:
        if not isinstance(row, dict):
            continue
        pr = row.get("preset") or {}
        pr_name = str(pr.get("name") or "unknown")
        knobs: Dict[str, Any] = {}
        # Keep knobs sparse and component-specific.
        if "ssim_max_side" in pr:
            knobs["ssim_max_side"] = int(pr.get("ssim_max_side"))
        if "flow_max_side" in pr:
            knobs["flow_max_side"] = int(pr.get("flow_max_side"))
        if "hard_cuts_cascade" in pr:
            knobs["hard_cuts_cascade"] = bool(pr.get("hard_cuts_cascade"))
        if "hard_cuts_cascade_keep_top_p" in pr:
            knobs["hard_cuts_cascade_keep_top_p"] = float(pr.get("hard_cuts_cascade_keep_top_p"))
        if "hard_cuts_cascade_hist_margin" in pr:
            knobs["hard_cuts_cascade_hist_margin"] = float(pr.get("hard_cuts_cascade_hist_margin"))

        met = row.get("metric") or {}
        metrics: Dict[str, Any] = {}
        # Keep the same metric keys used in existing resource_costs files.
        if "latency_ms_mean_stable_per_unit" in met:
            metrics["latency_ms_mean_stable_per_unit"] = met.get("latency_ms_mean_stable_per_unit")
        if "cpu_rss_peak_mb" in met:
            metrics["cpu_rss_peak_mb"] = met.get("cpu_rss_peak_mb")
        if "gpu_vram_peak_mb" in met:
            metrics["gpu_vram_peak_mb"] = met.get("gpu_vram_peak_mb")
        if "spikes" in met:
            metrics["spikes"] = bool(met.get("spikes"))

        out["costs"].append(
            {
                "component": metric_key,
                "unit": unit,
                "aspect": row.get("aspect"),
                "short_side": row.get("short_side"),
                "width": row.get("width"),
                "height": row.get("height"),
                "preset": pr_name,
                "knobs": knobs,
                "metrics": metrics,
                "provenance": {"out_json": row.get("out_json")},
            }
        )

    _write_json(out_path, out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


