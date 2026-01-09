#!/usr/bin/env python3
"""
Run cut_detection micro-benchmarks over a matrix of:
- resolutions (short-side buckets)
- aspect ratios (16:9, 9:16)
- quality presets (ssim_max_side, flow_max_side)

Outputs:
- <out_dir>/matrix_results.json (aggregated)
- per-run folders under <out_dir>/runs/...

This script uses scripts/baseline/run_checklist_components_micro.py with --frames-dir
to reuse the same synthetic frames per resolution across multiple quality presets.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import subprocess
import sys
from datetime import datetime
from typing import Any, Dict, List, Tuple

import logging


LOGGER = logging.getLogger("baseline.cut_detection_matrix")


def _setup_logging(level: str) -> None:
    lvl = str(level or "INFO").strip().upper()
    if lvl not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        lvl = "INFO"
    logging.basicConfig(
        level=getattr(logging, lvl, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )


def _ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def _parse_csv_ints(s: str) -> List[int]:
    out: List[int] = []
    for part in (s or "").split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out


def _parse_presets(items: List[str]) -> List[Dict[str, Any]]:
    """
    preset syntax:
      - name:ssim_max_side:flow_max_side
      - name:ssim_max_side:flow_max_side:cascade_enabled[:keep_top_p[:hist_margin]]
    Examples:
      - default:512:320
      - fast:384:256:1:0.25:0.0
    """
    out: List[Dict[str, Any]] = []
    for it in items:
        it = str(it).strip()
        if not it:
            continue
        parts = it.split(":")
        if len(parts) < 3:
            raise ValueError(f"Invalid preset {it!r}. Expected at least name:ssim:flow")
        name = parts[0].strip()
        ssim = int(parts[1])
        flow = int(parts[2])
        cascade = False
        keep_top_p = 0.25
        hist_margin = 0.0
        if len(parts) >= 4:
            cascade = bool(int(str(parts[3]).strip() or "0"))
        if len(parts) >= 5:
            keep_top_p = float(parts[4])
        if len(parts) >= 6:
            hist_margin = float(parts[5])
        out.append(
            {
                "name": name,
                "ssim_max_side": ssim,
                "flow_max_side": flow,
                "hard_cuts_cascade": bool(cascade),
                "hard_cuts_cascade_keep_top_p": float(keep_top_p),
                "hard_cuts_cascade_hist_margin": float(hist_margin),
            }
        )
    if not out:
        raise ValueError("No presets provided")
    return out


def _wh_from_short_side(short_side: int, aspect: str) -> Tuple[int, int]:
    s = int(short_side)
    a = str(aspect).strip()
    if a == "16:9":
        h = s
        w = int(round(s * 16.0 / 9.0))
        return w, h
    if a == "9:16":
        w = s
        h = int(round(s * 16.0 / 9.0))
        return w, h
    raise ValueError(f"Unknown aspect: {aspect!r}")


def _run_cmd(cmd: List[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\n\n{p.stdout}")


def _read_json(p: str) -> Any:
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    ap = argparse.ArgumentParser("cut_detection matrix runner")
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--resume", action="store_true", help="Resume: reuse existing frames/ and skip already completed runs (still aggregates).")
    ap.add_argument("--short-sides", type=str, default="160,224,256,320,384,448,512,640,720,768,896,1080")
    ap.add_argument("--aspects", type=str, default="16:9,9:16", help="Comma-separated: 16:9,9:16")
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--n-frames", type=int, default=64)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--sampler-interval-ms", type=float, default=50.0)
    ap.add_argument("--only", type=str, default="cut_detection_split", help="cut_detection or cut_detection_split")
    ap.add_argument(
        "--metric-key",
        type=str,
        default=None,
        help="Result key to extract from checklist_components_micro_results.json (auto if omitted).",
    )
    ap.add_argument(
        "--preset",
        action="append",
        default=[],
        help="Repeatable. Format: name:ssim_max_side:flow_max_side. Example: default:512:320",
    )
    ap.add_argument("--log-level", type=str, default="INFO")
    ap.add_argument("--log-every", type=int, default=0)
    args = ap.parse_args()

    _setup_logging(str(args.log_level))

    # Default presets if none provided
    preset_items = list(args.preset or [])
    if not preset_items:
        preset_items = [
            "quality:640:384:0",
            "default:512:320:0",
            "fast:384:256:1:0.25:0.0",
        ]
    presets = _parse_presets(preset_items)

    out_dir = args.out_dir or os.path.join("docs", "baseline", "out", f"cut_detection-matrix-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}")
    out_dir = os.path.abspath(out_dir)
    _ensure_dir(out_dir)
    runs_root = os.path.join(out_dir, "runs")
    frames_root = os.path.join(out_dir, "frames")
    _ensure_dir(runs_root)
    _ensure_dir(frames_root)

    short_sides = _parse_csv_ints(str(args.short_sides))
    aspects = [x.strip() for x in str(args.aspects).split(",") if x.strip()]

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    runner = os.path.join(repo_root, "scripts", "baseline", "run_checklist_components_micro.py")
    py = sys.executable or "python3"

    only_mode = str(args.only or "").strip()
    metric_key = str(args.metric_key).strip() if args.metric_key else ""
    if not metric_key:
        if only_mode in ("cut_detection", "cut_detection_split"):
            metric_key = "cut_detection.detect_hard_cuts(cpu_no_deep)"
        elif only_mode == "cut_detection_soft":
            metric_key = "cut_detection.detect_soft_cuts(cpu)"
        elif only_mode == "cut_detection_motion":
            metric_key = "cut_detection.detect_motion_based_cuts(cpu)"
        else:
            metric_key = "cut_detection.detect_hard_cuts(cpu_no_deep)"
    LOGGER.info("Metric key: %s", metric_key)

    agg: Dict[str, Any] = {
        "created_at": datetime.utcnow().isoformat(),
        "matrix": {
            "short_sides": short_sides,
            "aspects": aspects,
            "presets": presets,
            "fps": float(args.fps),
            "n_frames": int(args.n_frames),
            "batch_size": int(args.batch_size),
            "warmup": int(args.warmup),
            "repeats": int(args.repeats),
            "runs": int(args.runs),
            "sampler_interval_ms": float(args.sampler_interval_ms),
            "only": str(args.only),
            "metric_key": metric_key,
        },
        "results": [],
    }

    # 1) Create frames for each resolution once.
    LOGGER.info(
        "Matrix start: aspects=%s short_sides=%d presets=%d only=%s out_dir=%s",
        aspects,
        int(len(short_sides)),
        int(len(presets)),
        str(args.only),
        out_dir,
    )
    frames_index: Dict[str, str] = {}
    for aspect in aspects:
        for s in short_sides:
            w, h = _wh_from_short_side(int(s), aspect)
            key = f"{aspect}|{w}x{h}"
            if key in frames_index:
                continue
            fr_out = os.path.join(frames_root, aspect.replace(":", "x"), f"{w}x{h}")
            _ensure_dir(fr_out)
            LOGGER.info("Frames: generating %s (S=%s) -> %dx%d", aspect, int(s), int(w), int(h))
            frames_dir = os.path.join(fr_out, "frames")
            meta_path = os.path.join(frames_dir, "metadata.json")
            if bool(args.resume) and os.path.isdir(frames_dir) and os.path.isfile(meta_path):
                # reuse existing frames
                pass
            else:
                cmd = [
                    py,
                    runner,
                    "--only",
                    "cut_detection",  # fast generate + single bench; we only need frames dir
                    "--out-dir",
                    fr_out,
                    "--width",
                    str(w),
                    "--height",
                    str(h),
                    "--fps",
                    str(float(args.fps)),
                    "--n-frames",
                    str(int(args.n_frames)),
                    "--batch-size",
                    str(int(args.batch_size)),
                    "--warmup",
                    "0",
                    "--repeats",
                    "1",
                    "--runs",
                    "1",
                    "--log-level",
                    "ERROR",
                    "--log-every",
                    "0",
                ]
                _run_cmd(cmd)
                if not os.path.isdir(frames_dir):
                    raise RuntimeError(f"Expected frames dir not created: {frames_dir}")
            frames_index[key] = frames_dir

    # 2) Run matrix using --frames-dir (reusing frames for presets).
    total = len(aspects) * len(short_sides) * len(presets)
    done = 0
    for aspect in aspects:
        for s in short_sides:
            w, h = _wh_from_short_side(int(s), aspect)
            key = f"{aspect}|{w}x{h}"
            frames_dir = frames_index[key]
            for pr in presets:
                done += 1
                run_name = f"{aspect.replace(':','x')}_{w}x{h}_{pr['name']}_ssim{pr['ssim_max_side']}_flow{pr['flow_max_side']}"
                run_out = os.path.join(runs_root, run_name)
                _ensure_dir(run_out)
                out_json = os.path.join(run_out, "checklist_components_micro_results.json")

                if bool(args.resume) and os.path.isfile(out_json):
                    # already computed -> aggregate from existing json and continue
                    try:
                        payload = _read_json(out_json)
                        main = (payload.get("results") or {}).get(metric_key) or {}
                        agg["results"].append(
                            {
                                "aspect": aspect,
                                "short_side": int(s),
                                "width": int(w),
                                "height": int(h),
                                "preset": pr,
                                "out_json": out_json,
                                "metric": {
                                    "latency_ms_mean_stable_per_unit": main.get("latency_ms_mean_stable_per_unit"),
                                    "latency_ms_mean_stable_total_call": main.get("latency_ms_mean_stable_total_call"),
                                    "gpu_vram_peak_mb": main.get("gpu_vram_peak_mb"),
                                    "cpu_rss_peak_mb": main.get("cpu_rss_peak_mb"),
                                    "spikes": main.get("spikes"),
                                },
                            }
                        )
                        continue
                    except Exception:
                        # fall through and re-run if read failed
                        pass

                if int(args.log_every) > 0 and (done == 1 or done % int(args.log_every) == 0 or done == total):
                    LOGGER.info(
                        "Run %d/%d: %s S=%d (%dx%d) preset=%s (ssim=%s flow=%s)",
                        int(done),
                        int(total),
                        aspect,
                        int(s),
                        int(w),
                        int(h),
                        str(pr["name"]),
                        int(pr["ssim_max_side"]),
                        int(pr["flow_max_side"]),
                    )

                cmd = [
                    py,
                    runner,
                    "--only",
                    str(args.only),
                    "--out-dir",
                    run_out,
                    "--frames-dir",
                    frames_dir,
                    "--batch-size",
                    str(int(args.batch_size)),
                    "--warmup",
                    str(int(args.warmup)),
                    "--repeats",
                    str(int(args.repeats)),
                    "--runs",
                    str(int(args.runs)),
                    "--sampler-interval-ms",
                    str(float(args.sampler_interval_ms)),
                    "--ssim-max-side",
                    str(int(pr["ssim_max_side"])),
                    "--flow-max-side",
                    str(int(pr["flow_max_side"])),
                    "--log-level",
                    str(args.log_level),
                    "--log-every",
                    str(int(args.log_every)),
                ]
                # Optional cascade mode for hard_cuts (fast preset)
                if bool(pr.get("hard_cuts_cascade")):
                    cmd += [
                        "--hard-cuts-cascade",
                        "--hard-cuts-cascade-keep-top-p",
                        str(float(pr.get("hard_cuts_cascade_keep_top_p", 0.25))),
                        "--hard-cuts-cascade-hist-margin",
                        str(float(pr.get("hard_cuts_cascade_hist_margin", 0.0))),
                    ]
                _run_cmd(cmd)
                payload = _read_json(out_json)
                # Extract the main metric for quick indexing
                main = (payload.get("results") or {}).get(metric_key) or {}
                agg["results"].append(
                    {
                        "aspect": aspect,
                        "short_side": int(s),
                        "width": int(w),
                        "height": int(h),
                        "preset": pr,
                        "out_json": out_json,
                        "metric": {
                            "latency_ms_mean_stable_per_unit": main.get("latency_ms_mean_stable_per_unit"),
                            "latency_ms_mean_stable_total_call": main.get("latency_ms_mean_stable_total_call"),
                            "gpu_vram_peak_mb": main.get("gpu_vram_peak_mb"),
                            "spikes": main.get("spikes"),
                            "spikes_mad": main.get("spikes_mad"),
                            "spikes_stable_filter": main.get("spikes_stable_filter"),
                            "cpu_rss_peak_mb": main.get("cpu_rss_peak_mb"),
                        },
                    }
                )

    out_json = os.path.join(out_dir, "matrix_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(agg, f, ensure_ascii=False, indent=2)
    LOGGER.info("Matrix done: %s", out_json)
    print(out_json)


if __name__ == "__main__":
    main()


