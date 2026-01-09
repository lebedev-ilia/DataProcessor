#!/usr/bin/env python3
"""
Render markdown table rows from a checklist_micro_results.json.

Why:
- Avoid "append rows" drift/duplicates in docs.
- Make results reproducible and regeneratable from JSON artifacts.

This script converts model-level micro measurements into the canonical table format:
component/module × model_branch × source_resolution (routing) × selected_branch.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional


def _read_json(p: str) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def _fmt_float(x: Any, nd: int = 3) -> str:
    try:
        if x is None:
            return ""
        v = float(x)
        if v != v:  # nan
            return ""
        return f"{v:.{nd}f}"
    except Exception:
        return ""


def _source_str(inp: Dict[str, Any]) -> str:
    try:
        w = int(inp.get("width"))
        h = int(inp.get("height"))
        fmt = str(inp.get("format") or "")
        s = int(inp.get("short_side"))
        return f"{w}×{h} ({fmt}, S={s})"
    except Exception:
        return ""


def _family_rows(payload: Dict[str, Any], *, src_filter: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    visual_inputs = (payload.get("visual") or {}).get("inputs") or []
    models = payload.get("models_triton") or {}
    out: List[Dict[str, Any]] = []

    # Map family -> (component_name, model_name_builder)
    def _model_name(family: str, s: int) -> str:
        if family == "clip_image":
            return f"clip_image_{s}"
        if family == "places365":
            return f"places365_resnet50_{s}"
        if family == "midas":
            return f"midas_{s}"
        if family == "raft":
            return f"raft_{s}"
        if family == "yolo":
            return f"yolo11x_{s}"
        raise KeyError(family)

    comp_by_family = {
        "clip_image": "core_clip",
        "places365": "scene_classification",
        "midas": "core_depth_midas",
        "raft": "core_optical_flow",
        "yolo": "core_object_detections",
    }

    families = ["clip_image", "places365", "midas", "raft", "yolo"]

    for inp in visual_inputs:
        if src_filter:
            if "format" in src_filter and str(inp.get("format")) != str(src_filter["format"]):
                continue
            if "short_side" in src_filter and int(inp.get("short_side")) != int(src_filter["short_side"]):
                continue
            if "width" in src_filter and int(inp.get("width")) != int(src_filter["width"]):
                continue
            if "height" in src_filter and int(inp.get("height")) != int(src_filter["height"]):
                continue

        selected = (inp.get("selected") or {})
        for fam in families:
            s = selected.get(fam)
            if s is None:
                continue
            try:
                s_int = int(s)
            except Exception:
                continue
            mn = _model_name(fam, s_int)
            r = models.get(mn) or {}
            out.append(
                {
                    "component_or_module": comp_by_family.get(fam, ""),
                    "model_branch": mn,
                    "source_resolution": _source_str(inp),
                    "selected_branch": str(s_int),
                    "latency_ms_mean_stable": _fmt_float(r.get("latency_ms_mean_stable"), 3),
                    "p95_ms": _fmt_float(((r.get("latency_stats") or {}) if isinstance(r.get("latency_stats"), dict) else {}).get("p95"), 3),
                    "p99_ms": _fmt_float(((r.get("latency_stats") or {}) if isinstance(r.get("latency_stats"), dict) else {}).get("p99"), 3),
                    "spikes": str(bool(r.get("spikes"))).lower(),
                    "spike_fraction": _fmt_float(r.get("spike_fraction"), 3),
                    "rss_peak_mb": _fmt_float(r.get("cpu_rss_peak_mb"), 3),
                    "vram_triton_delta_run_mb": _fmt_float(r.get("vram_triton_delta_run_mb"), 1),
                    "status": str(r.get("status") or ""),
                    "notes": "",
                }
            )
    return out


def _write_md(out_path: str, *, rows: List[Dict[str, Any]], title: str, source_note: str) -> None:
    Path(os.path.dirname(out_path) or ".").mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(f"## {title}\n\n")
        if source_note:
            f.write(f"{source_note}\n\n")
        f.write("| component_or_module | model_branch | source_resolution | selected_branch | latency_ms_mean_stable | p95_ms | p99_ms | spikes | spike_fraction | rss_peak_mb | vram_triton_delta_run_mb |\n")
        f.write("|---|---|---:|---:|---:|---:|---:|:---:|---:|---:|---:|\n")
        for r in rows:
            f.write(
                "| "
                + " | ".join(
                    [
                        str(r.get("component_or_module", "")),
                        str(r.get("model_branch", "")),
                        str(r.get("source_resolution", "")),
                        str(r.get("selected_branch", "")),
                        str(r.get("latency_ms_mean_stable", "")),
                        str(r.get("p95_ms", "")),
                        str(r.get("p99_ms", "")),
                        str(r.get("spikes", "")),
                        str(r.get("spike_fraction", "")),
                        str(r.get("rss_peak_mb", "")),
                        str(r.get("vram_triton_delta_run_mb", "")),
                    ]
                )
                + " |\n"
            )


def main() -> None:
    ap = argparse.ArgumentParser("render checklist results to markdown")
    ap.add_argument("--input-json", type=str, required=True, help="Path to checklist_micro_results.json")
    ap.add_argument("--out-md", type=str, required=True, help="Output markdown path")
    ap.add_argument("--format", type=str, default="", choices=["", "16:9", "9:16"])
    ap.add_argument("--short-side", type=int, default=0)
    ap.add_argument("--width", type=int, default=0)
    ap.add_argument("--height", type=int, default=0)
    args = ap.parse_args()

    payload = _read_json(args.input_json)
    src_filter: Dict[str, Any] = {}
    if str(args.format).strip():
        src_filter["format"] = str(args.format).strip()
    if int(args.short_side) > 0:
        src_filter["short_side"] = int(args.short_side)
    if int(args.width) > 0:
        src_filter["width"] = int(args.width)
    if int(args.height) > 0:
        src_filter["height"] = int(args.height)

    rows = _family_rows(payload, src_filter=(src_filter or None))
    in_dir = os.path.dirname(os.path.abspath(args.input_json))
    note = f"- source json: `{args.input_json}`\n- out_dir: `{in_dir}`"
    _write_md(
        args.out_md,
        rows=rows,
        title="Baseline component/model results (rendered from JSON)",
        source_note=note,
    )
    print(args.out_md)


if __name__ == "__main__":
    main()


