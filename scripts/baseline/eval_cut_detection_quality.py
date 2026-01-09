#!/usr/bin/env python3
"""
Quality evaluation for cut_detection presets on real videos.

Flow:
1) Run Segmenter to build frames_dir (union-sampled) for the video.
2) Load per-component union-domain frame_indices for cut_detection.
3) Pick a stable subset of indices (uniform over the list) to control runtime.
4) Run a selected cut_detection task with different presets:
   - hard: detect_hard_cuts(cpu_no_deep) with (ssim_max_side, flow_max_side)
   - soft: detect_soft_cuts with (flow_max_side)
   - motion: detect_motion_based_cuts with (flow_max_side)
5) Compare predicted event times (union_timestamps_sec) vs reference using a time tolerance.

Outputs:
- <out_dir>/quality_report.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import cv2  # type: ignore
import logging


LOGGER = logging.getLogger("baseline.cut_detection_quality")


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


def _run(cmd: List[str]) -> None:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}\n\n{p.stdout}")


def _read_json(p: str) -> Dict[str, Any]:
    with open(p, "r", encoding="utf-8") as f:
        d = json.load(f)
    if not isinstance(d, dict):
        raise RuntimeError(f"Expected dict json: {p}")
    return d


def _uniform_subset(indices: List[int], n: int) -> List[int]:
    if n <= 0:
        return indices
    if len(indices) <= n:
        return indices
    pos = np.linspace(0, len(indices) - 1, num=n)
    pos = np.unique(np.rint(pos).astype(np.int64))
    pos.sort()
    out = [int(indices[int(i)]) for i in pos.tolist()]
    # ensure strictly increasing unique (FrameManager contract)
    out = sorted(set(out))
    return out


def _write_frames_dir_from_frames(out_dir: str, frames_rgb: List[np.ndarray], fps: float) -> str:
    """
    Create a minimal Segmenter-like frames_dir for FrameManager:
    - frames_dir/metadata.json
    - frames_dir/batch*.npy (NHWC uint8 RGB)
    """
    frames_dir = os.path.join(out_dir, "frames")
    _ensure_dir(frames_dir)
    batch_size = 32
    n = len(frames_rgb)
    batches: List[Dict[str, Any]] = []
    for bi in range(0, n, batch_size):
        chunk = np.stack(frames_rgb[bi : bi + batch_size], axis=0).astype(np.uint8)
        fname = f"batch{bi // batch_size}.npy"
        np.save(os.path.join(frames_dir, fname), chunk)
        start = int(bi)
        end = int(min(n - 1, bi + batch_size - 1))
        batches.append(
            {
                "batch_index": int(bi // batch_size),
                "path": fname,
                "start_frame": start,
                "end_frame": end,
            }
        )

    times = [float(i) / float(max(1e-6, fps)) for i in range(n)]
    meta = {
        "producer": "eval_cut_detection_quality",
        "created_at": datetime.utcnow().isoformat(),
        "color_space": "RGB",
        "width": int(frames_rgb[0].shape[1]) if frames_rgb else 0,
        "height": int(frames_rgb[0].shape[0]) if frames_rgb else 0,
        "channels": 3,
        "fps": float(fps),
        "total_frames": int(n),
        "batch_size": int(batch_size),
        "cache_size": 2,
        "batches": batches,
        "union_timestamps_sec": times,
        # Minimal per-component indices in union domain
        "cut_detection": {"frame_indices": list(range(n))},
    }
    with open(os.path.join(frames_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    return frames_dir


def _read_video_frames(video_path: str, start_frame: int, count: int) -> List[np.ndarray]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(start_frame))
        out: List[np.ndarray] = []
        while len(out) < int(count):
            ok, bgr = cap.read()
            if not ok:
                break
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            out.append(rgb.astype(np.uint8))
        return out
    finally:
        cap.release()


def _median_dt(times: np.ndarray) -> float:
    if times.size < 2:
        return 0.0
    dt = np.diff(times)
    dt = dt[np.isfinite(dt)]
    if dt.size == 0:
        return 0.0
    return float(np.median(dt))


def _match_with_tol(ref: List[float], pred: List[float], tol_s: float) -> Dict[str, Any]:
    """
    Greedy matching by time distance within tolerance.
    Returns precision/recall/f1 + matches list.
    """
    ref = sorted([float(x) for x in ref])
    pred = sorted([float(x) for x in pred])
    used = [False] * len(ref)
    matches: List[Tuple[int, int, float]] = []
    for pi, pt in enumerate(pred):
        best_j = None
        best_d = None
        for rj, rt in enumerate(ref):
            if used[rj]:
                continue
            d = abs(pt - rt)
            if d <= tol_s and (best_d is None or d < best_d):
                best_d = d
                best_j = rj
        if best_j is not None:
            used[best_j] = True
            matches.append((pi, best_j, float(best_d)))
    tp = len(matches)
    fp = len(pred) - tp
    fn = len(ref) - tp
    prec = tp / max(1, tp + fp)
    rec = tp / max(1, tp + fn)
    f1 = 0.0 if (prec + rec) == 0 else (2 * prec * rec / (prec + rec))
    return {
        "tol_s": float(tol_s),
        "ref_n": int(len(ref)),
        "pred_n": int(len(pred)),
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "matches": matches[:50],  # cap for readability
    }


def main() -> None:
    ap = argparse.ArgumentParser("cut_detection quality eval on real videos")
    ap.add_argument("--videos", type=str, required=True, help="Comma-separated list of video paths (mp4/mkv/...)")
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--segmenter-out", type=str, default=None, help="Where to write Segmenter outputs (defaults under out-dir).")
    ap.add_argument(
        "--visual-cfg-path",
        type=str,
        default="",
        help="Optional VisualProcessor/config.yaml to build per-component budgets. If empty, Segmenter uses its default extractor list (includes cut_detection).",
    )
    ap.add_argument("--subset-n", type=int, default=250, help="Subsample cut_detection frame_indices to this many points (controls runtime).")
    ap.add_argument("--device", type=str, default="cpu")
    ap.add_argument("--log-level", type=str, default="INFO")
    ap.add_argument("--analysis-width", type=int, default=None, help="Pass-through to Segmenter: analysis timeline width (resize).")
    ap.add_argument("--analysis-height", type=int, default=None, help="Pass-through to Segmenter: analysis timeline height (resize).")
    ap.add_argument("--analysis-fps", type=float, default=None, help="Pass-through to Segmenter: analysis fps (sampling).")
    ap.add_argument(
        "--task",
        type=str,
        default="hard",
        choices=["hard", "soft", "motion"],
        help="Which cut_detection task to evaluate: hard|soft|motion.",
    )
    ap.add_argument(
        "--ref",
        type=str,
        default="ref_nodownscale",
        help="Reference preset name to compare against: ref_nodownscale|quality|default|fast. Use quality/default for faster runs.",
    )
    ap.add_argument(
        "--also-stitched",
        action="store_true",
        help="Additionally evaluate on a 'stitched' sequence made from chunks of the provided videos to ensure there are hard cuts.",
    )
    ap.add_argument("--stitched-chunk-frames", type=int, default=30, help="Frames per chunk in stitched evaluation.")
    ap.add_argument("--stitched-chunks", type=int, default=6, help="Number of chunks in stitched evaluation.")
    args = ap.parse_args()
    _setup_logging(str(args.log_level))

    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    out_dir = os.path.abspath(args.out_dir or os.path.join("docs", "baseline", "out", f"cut_detection-quality-{datetime.utcnow().strftime('%Y%m%d-%H%M%S')}"))
    _ensure_dir(out_dir)
    # Cache Segmenter outputs per analysis settings to avoid accidental reuse of mismatched configs.
    seg_base = os.path.abspath(args.segmenter_out or os.path.join(out_dir, "segmenter_out"))
    sig = f"aw{args.analysis_width or 'src'}_ah{args.analysis_height or 'src'}_afps{args.analysis_fps or 'src'}"
    seg_out = os.path.join(seg_base, sig)
    _ensure_dir(seg_out)

    # Make VisualProcessor imports work
    vp_root = os.path.join(repo_root, "VisualProcessor")
    if vp_root not in sys.path:
        sys.path.insert(0, vp_root)
    from utils.frame_manager import FrameManager  # type: ignore
    from modules.cut_detection.cut_detection import detect_hard_cuts, detect_soft_cuts, detect_motion_based_cuts  # type: ignore

    videos = [x.strip() for x in str(args.videos).split(",") if x.strip()]
    if not videos:
        raise ValueError("No videos provided")
    LOGGER.info("Quality eval start: videos=%d subset_n=%d device=%s out_dir=%s", len(videos), int(args.subset_n), str(args.device), out_dir)

    presets = [
        {"name": "ref_nodownscale", "ssim_max_side": 0, "flow_max_side": 0},
        {"name": "quality", "ssim_max_side": 640, "flow_max_side": 384},
        {"name": "default", "ssim_max_side": 512, "flow_max_side": 320},
        {"name": "fast", "ssim_max_side": 384, "flow_max_side": 256},
    ]
    ref_name = str(args.ref or "").strip()
    if ref_name not in {p["name"] for p in presets}:
        raise ValueError(f"--ref must be one of: {[p['name'] for p in presets]}, got {ref_name!r}")
    # Speed optimization: if the chosen reference is NOT the expensive "no downscale" run,
    # skip computing it entirely.
    if ref_name != "ref_nodownscale":
        presets = [p for p in presets if p["name"] != "ref_nodownscale"]

    report: Dict[str, Any] = {
        "created_at": datetime.utcnow().isoformat(),
        "videos": [],
        "stitched": None,
        "presets": presets,
        "notes": {
            "task": str(args.task),
            "comparison": "events compared by time (union_timestamps_sec) with tolerance = 1.5 * median_dt(union) on the evaluated subset",
            "detector": "cut_detection task via direct call (no deep features).",
        },
    }

    def _pos_to_time(pos: int, subset: List[int], times: np.ndarray) -> Optional[float]:
        try:
            p = int(pos)
            if p < 0 or p >= len(subset):
                return None
            u = int(subset[p])
            if u < 0 or u >= int(times.size):
                return None
            return float(times[u])
        except Exception:
            return None

    def _soft_events_to_times(events: Any, subset: List[int], times: np.ndarray) -> List[float]:
        out: List[float] = []
        if not isinstance(events, list):
            return out
        for e in events:
            if not isinstance(e, dict):
                continue
            s = e.get("start")
            en = e.get("end")
            if s is None or en is None:
                continue
            try:
                mid = int(round((int(s) + int(en)) / 2.0))
            except Exception:
                continue
            t = _pos_to_time(mid, subset, times)
            if t is not None:
                out.append(t)
        return sorted(out)

    for vpath in videos:
        vid = os.path.splitext(os.path.basename(vpath))[0]
        LOGGER.info("Segmenter: video_id=%s path=%s", vid, os.path.abspath(vpath))
        frames_dir = os.path.join(seg_out, vid, "video")
        meta_path = os.path.join(frames_dir, "metadata.json")
        if os.path.isfile(meta_path):
            LOGGER.info("Segmenter: reuse existing frames_dir=%s", frames_dir)
        else:
            seg_cmd = [
                sys.executable or "python3",
                os.path.join(repo_root, "Segmenter", "segmenter.py"),
                "--video-path",
                os.path.abspath(vpath),
                "--output",
                seg_out,
                f"--video-id={vid}",
                "--platform-id=local",
            ]
            visual_cfg = str(args.visual_cfg_path or "").strip()
            if visual_cfg:
                seg_cmd.extend(["--visual-cfg-path", os.path.join(repo_root, visual_cfg)])
            if args.analysis_width is not None:
                seg_cmd.extend(["--analysis-width", str(int(args.analysis_width))])
            if args.analysis_height is not None:
                seg_cmd.extend(["--analysis-height", str(int(args.analysis_height))])
            if args.analysis_fps is not None:
                seg_cmd.extend(["--analysis-fps", str(float(args.analysis_fps))])
            _run(seg_cmd)
        meta = _read_json(os.path.join(frames_dir, "metadata.json"))
        block = meta.get("cut_detection") or {}
        fi = block.get("frame_indices")
        if not isinstance(fi, list) or not fi:
            raise RuntimeError(f"Segmenter output missing cut_detection.frame_indices for {vid}")
        frame_indices = [int(x) for x in fi]
        frame_indices_sub = _uniform_subset(frame_indices, int(args.subset_n))
        LOGGER.info("Frames: video_id=%s union_total=%s cut_detection_n=%d subset_n=%d", vid, meta.get("total_frames"), len(frame_indices), len(frame_indices_sub))

        fm = FrameManager(frames_dir=frames_dir, chunk_size=int(meta.get("chunk_size", 64) or 64), cache_size=2)
        times = np.asarray(meta.get("union_timestamps_sec") or [], dtype=np.float32)
        if times.size == 0:
            raise RuntimeError(f"Segmenter output missing union_timestamps_sec for {vid}")

        # Evaluate: collect cut times (seconds) for each preset.
        preset_results: Dict[str, Any] = {}
        for pr in presets:
            LOGGER.info("Eval: video_id=%s preset=%s (ssim=%s flow=%s)", vid, pr["name"], pr["ssim_max_side"], pr["flow_max_side"])
            task = str(args.task)
            flow_ms = int(pr["flow_max_side"])
            ssim_ms = int(pr["ssim_max_side"])
            if task == "hard":
                cut_pos, _strengths = detect_hard_cuts(
                    fm,
                    frame_indices_sub,
                    use_deep_features=False,
                    use_adaptive_thresholds=True,
                    temporal_smoothing=True,
                    ssim_max_side=ssim_ms,
                    flow_max_side=flow_ms,
                    device=str(args.device),
                )
                pos_list = [int(x) for x in (cut_pos or [])]
                event_times = [t for t in (_pos_to_time(p, frame_indices_sub, times) for p in pos_list) if t is not None]
            elif task == "soft":
                events = detect_soft_cuts(
                    fm,
                    frame_indices_sub,
                    fps=float(meta.get("fps") or (args.analysis_fps or 30.0) or 30.0),
                    flow_max_side=flow_ms,
                )
                event_times = _soft_events_to_times(events, frame_indices_sub, times)
            else:
                spike_idxs, _intensities, _types = detect_motion_based_cuts(
                    fm,
                    frame_indices_sub,
                    flow_max_side=flow_ms,
                )
                pos_list = [int(x) for x in (spike_idxs or [])]
                event_times = [t for t in (_pos_to_time(p, frame_indices_sub, times) for p in pos_list) if t is not None]

            preset_results[str(pr["name"])] = {
                "events_n": int(len(event_times)),
                "event_times_s": sorted([float(x) for x in event_times]),
                "task": task,
            }

        # Compare against selected reference (speed knob).
        ref = preset_results[ref_name]["event_times_s"]
        dt_med = _median_dt(times[np.asarray(frame_indices_sub, dtype=np.int32)])
        tol = max(0.05, 1.5 * float(dt_med))
        comparisons: Dict[str, Any] = {}
        for pr in presets:
            name = str(pr["name"])
            if name == ref_name:
                continue
            pred = preset_results[name]["event_times_s"]
            comparisons[name] = _match_with_tol(ref, pred, tol_s=tol)
        LOGGER.info("Compare: video_id=%s tol_s=%.3f ref_n=%d", vid, float(tol), len(ref))
        for name, cmp in comparisons.items():
            LOGGER.info("  vs_ref %s: P=%.3f R=%.3f F1=%.3f tp=%d fp=%d fn=%d", name, cmp["precision"], cmp["recall"], cmp["f1"], cmp["tp"], cmp["fp"], cmp["fn"])

        report["videos"].append(
            {
                "video_id": vid,
                "video_path": os.path.abspath(vpath),
                "frames_dir": frames_dir,
                "subset_n": int(len(frame_indices_sub)),
                "subset_policy": "uniform over cut_detection.frame_indices",
                "tol_s": float(tol),
                "ref": ref_name,
                "preset_results": preset_results,
                "comparisons_vs_ref": comparisons,
            }
        )

    if bool(args.also_stitched) and str(args.task) == "hard":
        LOGGER.info("Stitched eval start: chunk_frames=%d chunks=%d", int(args.stitched_chunk_frames), int(args.stitched_chunks))
        # Build stitched frames: alternating chunks from different videos/time offsets.
        chunk = int(args.stitched_chunk_frames)
        n_chunks = int(args.stitched_chunks)
        if chunk <= 0 or n_chunks < 2:
            raise ValueError("Invalid stitched config")
        frames_all: List[np.ndarray] = []
        gt_cut_indices: List[int] = []
        # Use deterministic offsets spaced across each video.
        for ci in range(n_chunks):
            vpath = videos[ci % len(videos)]
            # pick a start frame proportional to chunk index to vary content
            start = int(100 + ci * 60)
            chunk_frames = _read_video_frames(os.path.abspath(vpath), start_frame=start, count=chunk)
            if len(chunk_frames) < chunk:
                # fallback: try from start
                chunk_frames = _read_video_frames(os.path.abspath(vpath), start_frame=0, count=chunk)
            if len(chunk_frames) < chunk:
                raise RuntimeError(f"Not enough frames for stitched chunk from {vpath}")
            if frames_all:
                gt_cut_indices.append(len(frames_all))  # boundary at start of this chunk
            frames_all.extend(chunk_frames)

        stitched_root = os.path.join(out_dir, "stitched")
        _ensure_dir(stitched_root)
        frames_dir = _write_frames_dir_from_frames(stitched_root, frames_all, fps=30.0)
        meta = _read_json(os.path.join(frames_dir, "metadata.json"))
        fm = FrameManager(frames_dir=frames_dir, chunk_size=int(meta.get("chunk_size", 32) or 32), cache_size=2)
        frame_indices = list(range(int(meta.get("total_frames", len(frames_all)) or len(frames_all))))
        times = np.asarray(meta.get("union_timestamps_sec") or [], dtype=np.float32)
        dt_med = _median_dt(times)
        tol = max(0.05, 1.5 * float(dt_med))

        preset_results: Dict[str, Any] = {}
        for pr in presets:
            LOGGER.info("Stitched eval: preset=%s (ssim=%s flow=%s)", pr["name"], pr["ssim_max_side"], pr["flow_max_side"])
            cut_pos, _strengths = detect_hard_cuts(
                fm,
                frame_indices,
                use_deep_features=False,
                use_adaptive_thresholds=True,
                temporal_smoothing=True,
                ssim_max_side=int(pr["ssim_max_side"]),
                flow_max_side=int(pr["flow_max_side"]),
                device=str(args.device),
            )
            cut_pos = [int(x) for x in (cut_pos or [])]
            cut_union = [int(frame_indices[i]) for i in cut_pos if 0 <= int(i) < len(frame_indices)]
            cut_times = [float(times[u]) for u in cut_union if 0 <= int(u) < int(times.size)]
            preset_results[str(pr["name"])] = {"events_n": int(len(cut_times)), "event_times_s": cut_times, "task": "hard"}

        ref = preset_results[ref_name]["event_times_s"]
        comparisons: Dict[str, Any] = {}
        for pr in presets:
            name = str(pr["name"])
            if name == ref_name:
                continue
            comparisons[name] = _match_with_tol(ref, preset_results[name]["event_times_s"], tol_s=tol)

        # Ground truth cut times for stitched sequence
        gt_times = [float(times[i]) for i in gt_cut_indices if 0 <= int(i) < int(times.size)]
        gt_cmp: Dict[str, Any] = {}
        for pr in presets:
            name = str(pr["name"])
            gt_cmp[name] = _match_with_tol(gt_times, preset_results[name]["event_times_s"], tol_s=tol)
        LOGGER.info("Stitched GT: tol_s=%.3f gt_n=%d ref_n=%d", float(tol), len(gt_times), len(ref))
        for name, cmp in gt_cmp.items():
            LOGGER.info("  vs_gt %s: P=%.3f R=%.3f F1=%.3f tp=%d fp=%d fn=%d", name, cmp["precision"], cmp["recall"], cmp["f1"], cmp["tp"], cmp["fp"], cmp["fn"])

        report["stitched"] = {
            "frames_dir": frames_dir,
            "total_frames": int(len(frame_indices)),
            "gt_cut_indices": gt_cut_indices,
            "gt_cut_times_s": gt_times,
            "tol_s": float(tol),
            "ref": ref_name,
            "preset_results": preset_results,
            "comparisons_vs_ref": comparisons,
            "comparisons_vs_ground_truth": gt_cmp,
        }

    out_json = os.path.join(out_dir, "quality_report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    LOGGER.info("Quality eval done: %s", out_json)
    print(out_json)


if __name__ == "__main__":
    main()


