#!/usr/bin/env python3
"""
Component-level micro benchmarks (MVP).

Goal:
- Measure time + RSS peak for "big" VisualProcessor components in isolation.
- Provide a framework to later split components (e.g., cut_detection) into logical parts.

Notes:
- Many VisualProcessor modules need Segmenter-produced `frames_dir` (FrameManager + metadata.json).
  Here we generate a minimal synthetic `frames_dir` with a single `.npy` batch.
- Unit of work is still "per-frame" where possible; for algorithms that naturally
  operate on frame pairs/sequence (e.g., cut detection), we report both total and per-unit.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import time

# Reuse benchmark primitives from the model-level runner.
from run_checklist_micro import (  # noqa: E402
    PeakSampler,
    _detect_spikes_mad,
    _ensure_dir,
    _git_info,
    _repo_root,
    _stable_mean_ms,
    _stats_latency_ms,
    _system_info,
    _utc_ts,
)

LOGGER = logging.getLogger("baseline.components")


def _setup_logging(level: str) -> None:
    lvl = str(level or "INFO").strip().upper()
    if lvl not in ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"):
        lvl = "INFO"
    logging.basicConfig(
        level=getattr(logging, lvl, logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    )


def _write_json(p: str, payload: Dict[str, Any]) -> None:
    with open(p, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _load_metadata_json(meta_path: str) -> Dict[str, Any]:
    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)
    if not isinstance(meta, dict):
        raise RuntimeError(f"metadata.json must be a dict: {meta_path}")
    return meta


def _make_synth_frames_dir(
    *,
    out_dir: str,
    width: int,
    height: int,
    n_frames: int,
    fps: float,
    batch_size: int,
    seed: int = 0,
) -> str:
    frames_dir = os.path.join(out_dir, "frames")
    _ensure_dir(frames_dir)

    LOGGER.info("Generating synthetic frames: %dx%d n_frames=%d fps=%.3f batch_size=%d", int(width), int(height), int(n_frames), float(fps), int(batch_size))
    rng = np.random.RandomState(int(seed))
    frames = rng.randint(0, 256, size=(int(n_frames), int(height), int(width), 3), dtype=np.uint8)

    # Write as Segmenter-like batches: batch0.npy, batch1.npy, ...
    bs = int(max(1, int(batch_size)))
    batches: List[Dict[str, Any]] = []
    n = int(n_frames)
    bidx = 0
    for start in range(0, n, bs):
        end = min(n, start + bs)
        chunk = frames[start:end]
        fname = f"batch{bidx}.npy"
        np.save(os.path.join(frames_dir, fname), chunk)
        batches.append(
            {
                "batch_index": int(bidx),
                "start_frame": int(start),
                "end_frame": int(end - 1),
                "path": fname,
            }
        )
        bidx += 1
    LOGGER.info("Wrote %d batch files under %s", int(len(batches)), frames_dir)

    run_id = uuid.uuid4().hex
    meta: Dict[str, Any] = {
        "total_frames": int(n_frames),
        "batch_size": int(batch_size),
        "batches": batches,
        "height": int(height),
        "width": int(width),
        "channels": 3,
        "fps": float(fps),
        "color_space": "RGB",
        # Segmenter contract (best-effort)
        "union_timestamps_sec": [float(i) / float(fps) for i in range(int(n_frames))],
        # Required identity keys (BaseModule contract)
        "platform_id": "local",
        "video_id": "synthetic",
        "run_id": str(run_id),
        "sampling_policy_version": "baseline_synth_v1",
        "config_hash": "none",
        "dataprocessor_version": "unknown",
        "analysis_fps": float(fps),
        "analysis_width": int(width),
        "analysis_height": int(height),
    }
    with open(os.path.join(frames_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    return frames_dir


def _bench_call(
    *,
    fn,
    name: str,
    warmup: int,
    repeats: int,
    runs: int,
    units_per_call: int,
    sampler_interval_sec: float,
    spike_mad_k: float,
    record_series: bool,
    gpu_index: int,
    log_every: int,
) -> Dict[str, Any]:
    status = "ok"
    err: Optional[str] = None
    all_samples_ms: List[float] = []
    run_details: List[Dict[str, Any]] = []

    LOGGER.info(
        "Benchmark start: %s | warmup=%d repeats=%d runs=%d units_per_call=%d sampler_interval_ms=%.1f record_series=%s",
        str(name),
        int(warmup),
        int(repeats),
        int(runs),
        int(units_per_call),
        float(sampler_interval_sec) * 1000.0,
        str(bool(record_series)).lower(),
    )

    for run_idx in range(max(1, int(runs))):
        LOGGER.info("Run %d/%d: warmup...", int(run_idx) + 1, int(max(1, int(runs))))
        sampler = PeakSampler(
            interval_sec=float(sampler_interval_sec),
            gpu_index=int(gpu_index),
            record_series=bool(record_series),
            match_process_substrings=[],  # component-level bench is usually non-triton
        )
        lat_ms: List[float] = []
        sampler.start()
        try:
            for wi in range(max(0, int(warmup))):
                _ = fn()
                if int(log_every) > 0 and (wi + 1) % int(log_every) == 0:
                    LOGGER.info("Run %d/%d: warmup %d/%d done", int(run_idx) + 1, int(max(1, int(runs))), int(wi) + 1, int(max(0, int(warmup))))

            LOGGER.info("Run %d/%d: measure...", int(run_idx) + 1, int(max(1, int(runs))))
            for ri in range(max(1, int(repeats))):
                t0 = time.perf_counter()
                _ = fn()
                dt_ms = (time.perf_counter() - t0) * 1000.0
                lat_ms.append(float(dt_ms))
                if int(log_every) > 0 and (ri + 1) % int(log_every) == 0:
                    try:
                        last = float(lat_ms[-1])
                    except Exception:
                        last = float("nan")
                    LOGGER.info("Run %d/%d: repeat %d/%d last_ms=%.3f", int(run_idx) + 1, int(max(1, int(runs))), int(ri) + 1, int(max(1, int(repeats))), float(last))
        except Exception as e:  # noqa: BLE001
            status = "error"
            err = str(e)
            LOGGER.exception("Benchmark error in %s: %s", str(name), str(e))
        finally:
            sampler.stop()

        all_samples_ms.extend(lat_ms)
        spikes = _detect_spikes_mad(lat_ms, k=float(spike_mad_k))
        stats = _stats_latency_ms(lat_ms)
        stats["spike_fraction"] = float(spikes["spike_fraction"])
        stats["spike_rule"] = str(spikes["rule"])
        run_details.append(
            {
                "run_idx": int(run_idx),
                "latency_ms_samples": lat_ms,
                "latency_stats": stats,
                "spikes_mad": bool(spikes["spikes"]),
                "spike_fraction_mad": float(spikes["spike_fraction"]),
                "spikes_rule_mad": str(spikes["rule"]),
                "cpu_rss_peak_mb": sampler.rss_peak_mb,
                "gpu_vram_peak_mb": sampler.gpu_peak_mb,
                "series": sampler.series,
            }
        )
        if status != "ok":
            break

    LOGGER.info("Benchmark done: %s | status=%s samples=%d", str(name), str(status), int(len(all_samples_ms)))

    mean_stable, spikes_stable_filter = _stable_mean_ms(all_samples_ms)
    spikes_all = _detect_spikes_mad(all_samples_ms, k=float(spike_mad_k))
    stats_all = _stats_latency_ms(all_samples_ms)
    stats_all["spike_fraction"] = float(spikes_all["spike_fraction"])
    stats_all["spike_rule"] = str(spikes_all["rule"])

    # Peak RSS/GPU across runs
    try:
        rss_peak = max(float(r.get("cpu_rss_peak_mb")) for r in run_details if r.get("cpu_rss_peak_mb") is not None)
    except Exception:
        rss_peak = None
    try:
        gpu_peak = max(float(r.get("gpu_vram_peak_mb")) for r in run_details if r.get("gpu_vram_peak_mb") is not None)
    except Exception:
        gpu_peak = None

    per_unit_mean_stable = None
    try:
        per_unit_mean_stable = float(mean_stable) / float(max(1, int(units_per_call)))
    except Exception:
        per_unit_mean_stable = None

    return {
        "status": status,
        "error": err,
        "warmup": int(warmup),
        "repeats": int(repeats),
        "runs": int(max(1, int(runs))),
        "units_per_call": int(max(1, int(units_per_call))),
        "latency_ms_samples_total_call": all_samples_ms,
        "latency_stats_total_call": stats_all,
        "latency_ms_mean_stable_total_call": mean_stable,
        "latency_ms_mean_stable_per_unit": per_unit_mean_stable,
        "spikes_mad": bool(spikes_all["spikes"]),
        "spike_fraction_mad": float(spikes_all["spike_fraction"]),
        "spikes_rule_mad": str(spikes_all["rule"]),
        "spikes_stable_filter": bool(spikes_stable_filter),
        "spikes": bool(bool(spikes_stable_filter) or bool(spikes_all["spikes"])),
        "cpu_rss_peak_mb": rss_peak,
        "gpu_vram_peak_mb": gpu_peak,
        "runs_detail": run_details,
    }


def main() -> None:
    ap = argparse.ArgumentParser("baseline component checklist micro runner")
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument(
        "--frames-dir",
        type=str,
        default=None,
        help="Use an existing frames_dir (with metadata.json and batch*.npy) instead of generating synthetic frames.",
    )
    ap.add_argument("--gpu-index", type=int, default=0)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument("--n-frames", type=int, default=32, help="Synthetic frame count for sequence-based components.")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--sampler-interval-ms", type=float, default=50.0)
    ap.add_argument("--record-series", action="store_true")
    ap.add_argument("--spike-mad-k", type=float, default=4.0)
    ap.add_argument("--ssim-max-side", type=int, default=512, help="Max side for SSIM computation (0=no downscale).")
    ap.add_argument("--flow-max-side", type=int, default=320, help="Max side for Farneback flow computation (0=no downscale).")
    ap.add_argument("--hard-cuts-cascade", action="store_true", help="Enable histogram-gated cascade in detect_hard_cuts (default: off).")
    ap.add_argument("--hard-cuts-cascade-keep-top-p", type=float, default=0.25, help="Cascade keep-top-p by hist (0..1).")
    ap.add_argument("--hard-cuts-cascade-hist-margin", type=float, default=0.0, help="Cascade hist margin: keep hist>=hist_thresh-margin.")
    ap.add_argument("--log-level", type=str, default="INFO", help="DEBUG|INFO|WARNING|ERROR")
    ap.add_argument("--log-every", type=int, default=1, help="Print progress every N iterations (0=disable).")
    ap.add_argument(
        "--only",
        type=str,
        default="cut_detection",
        help=(
            "Comma-separated component parts to run. Supported: "
            "cut_detection (hard cuts), cut_detection_split (hard cuts + substeps), "
            "cut_detection_soft (soft cuts), cut_detection_motion (motion-based cuts)."
        ),
    )
    args = ap.parse_args()

    _setup_logging(str(args.log_level))
    repo_root = _repo_root()
    out_dir = args.out_dir or os.path.join("docs", "baseline", "out", f"checklist-components-{_utc_ts()}")
    _ensure_dir(out_dir)
    LOGGER.info("out_dir=%s", out_dir)

    frames_dir_arg = str(args.frames_dir).strip() if args.frames_dir else ""
    if frames_dir_arg:
        frames_dir = frames_dir_arg
        meta_path = os.path.join(frames_dir, "metadata.json")
        if not os.path.isfile(meta_path):
            raise RuntimeError(f"--frames-dir provided but metadata.json not found: {meta_path}")
        meta = _load_metadata_json(meta_path)
        # metadata.json is the source of truth; CLI values are advisory.
        try:
            args.width = int(meta.get("width", args.width))
            args.height = int(meta.get("height", args.height))
            args.fps = float(meta.get("fps", args.fps))
            args.n_frames = int(meta.get("total_frames", args.n_frames))
        except Exception:
            pass
        LOGGER.info(
            "Using existing frames_dir=%s (meta: %dx%d fps=%.3f n_frames=%d)",
            frames_dir,
            int(args.width),
            int(args.height),
            float(args.fps),
            int(args.n_frames),
        )
    else:
        frames_dir = _make_synth_frames_dir(
            out_dir=out_dir,
            width=int(args.width),
            height=int(args.height),
            n_frames=int(args.n_frames),
            fps=float(args.fps),
            batch_size=int(args.batch_size),
            seed=0,
        )

    # Make VisualProcessor imports work (modules expect "utils.*" imports).
    vp_root = os.path.join(repo_root, "VisualProcessor")
    if vp_root not in sys.path:
        sys.path.insert(0, vp_root)
    LOGGER.info("VisualProcessor import root: %s", vp_root)

    from utils.frame_manager import FrameManager  # type: ignore  # noqa: E402

    fm = FrameManager(frames_dir=frames_dir, chunk_size=int(args.batch_size), cache_size=2)
    frame_indices = list(range(int(args.n_frames)))
    LOGGER.info("FrameManager ready: frames_dir=%s total_frames=%d", frames_dir, int(getattr(fm, "total_frames", len(frame_indices))))

    sampler_interval_sec = float(max(1.0, float(args.sampler_interval_ms)) / 1000.0)

    only = {x.strip() for x in str(args.only or "").split(",") if x.strip()}
    results: Dict[str, Any] = {}

    if "cut_detection" in only or "cut_detection_split" in only:
        # Benchmark hard-cut detector (CPU-only baseline: no deep features).
        LOGGER.info("Importing cut_detection.detect_hard_cuts ...")
        from modules.cut_detection.cut_detection import (  # type: ignore  # noqa: E402
            detect_hard_cuts,
            frame_histogram_diff,
            frame_ssim,
            optical_flow_magnitude,
            _resize_gray_max_side,
        )

        def _run_cut_detection_hard() -> Any:
            return detect_hard_cuts(
                fm,
                frame_indices,
                use_deep_features=False,
                use_adaptive_thresholds=True,
                temporal_smoothing=True,
                ssim_max_side=int(args.ssim_max_side),
                flow_max_side=int(args.flow_max_side),
                embed_model=None,
                transform=None,
                device="cpu",
                cascade_enabled=bool(args.hard_cuts_cascade),
                cascade_keep_top_p=float(args.hard_cuts_cascade_keep_top_p),
                cascade_hist_margin=float(args.hard_cuts_cascade_hist_margin),
            )

        units = max(1, len(frame_indices) - 1)  # pairs (transitions)
        name_e2e = "cut_detection.detect_hard_cuts(cpu_no_deep)"
        results[name_e2e] = _bench_call(
            fn=_run_cut_detection_hard,
            name=name_e2e,
            warmup=int(args.warmup),
            repeats=int(args.repeats),
            runs=int(args.runs),
            units_per_call=int(units),
            sampler_interval_sec=float(sampler_interval_sec),
            spike_mad_k=float(args.spike_mad_k),
            record_series=bool(args.record_series),
            gpu_index=int(args.gpu_index),
            log_every=int(args.log_every),
        )

        if "cut_detection_split" in only:
            LOGGER.info("Preparing cut_detection split benchmarks (feature extraction vs postprocess)...")

            # -------- Substep 1: frame loading only (FrameManager IO + numpy copies)
            def _load_only() -> Any:
                for i in range(1, len(frame_indices)):
                    _ = fm.get(frame_indices[i - 1])
                    _ = fm.get(frame_indices[i])
                return None

            name_load = "cut_detection.substep.load_frames_only"
            results[name_load] = _bench_call(
                fn=_load_only,
                name=name_load,
                warmup=int(args.warmup),
                repeats=int(args.repeats),
                runs=int(args.runs),
                units_per_call=int(units),
                sampler_interval_sec=float(sampler_interval_sec),
                spike_mad_k=float(args.spike_mad_k),
                record_series=bool(args.record_series),
                gpu_index=int(args.gpu_index),
                log_every=int(args.log_every),
            )

            # -------- Substep 2: histogram diffs only
            def _hist_only() -> Any:
                out: List[float] = []
                for i in range(1, len(frame_indices)):
                    fA = fm.get(frame_indices[i - 1])
                    fB = fm.get(frame_indices[i])
                    out.append(float(frame_histogram_diff(fA, fB)))
                return out

            name_hist = "cut_detection.substep.feature_histogram_diff_only"
            results[name_hist] = _bench_call(
                fn=_hist_only,
                name=name_hist,
                warmup=int(args.warmup),
                repeats=int(args.repeats),
                runs=int(args.runs),
                units_per_call=int(units),
                sampler_interval_sec=float(sampler_interval_sec),
                spike_mad_k=float(args.spike_mad_k),
                record_series=bool(args.record_series),
                gpu_index=int(args.gpu_index),
                log_every=int(args.log_every),
            )

            # -------- Substep 3: SSIM only
            def _ssim_only() -> Any:
                out: List[float] = []
                for i in range(1, len(frame_indices)):
                    fA = fm.get(frame_indices[i - 1])
                    fB = fm.get(frame_indices[i])
                    # Match module policy: SSIM on downscaled grayscale capped by ssim_max_side.
                    import cv2

                    gA = cv2.cvtColor(fA, cv2.COLOR_RGB2GRAY)
                    gB = cv2.cvtColor(fB, cv2.COLOR_RGB2GRAY)
                    gA = _resize_gray_max_side(gA, int(args.ssim_max_side))
                    gB = _resize_gray_max_side(gB, int(args.ssim_max_side))
                    try:
                        from skimage.metrics import structural_similarity as _ssim

                        dr = float(gB.max() - gB.min())
                        dr = dr if dr > 1e-9 else 1.0
                        out.append(float(1.0 - _ssim(gA, gB, data_range=dr)))
                    except Exception:
                        out.append(0.0)
                return out

            name_ssim = "cut_detection.substep.feature_ssim_only"
            results[name_ssim] = _bench_call(
                fn=_ssim_only,
                name=name_ssim,
                warmup=int(args.warmup),
                repeats=int(args.repeats),
                runs=int(args.runs),
                units_per_call=int(units),
                sampler_interval_sec=float(sampler_interval_sec),
                spike_mad_k=float(args.spike_mad_k),
                record_series=bool(args.record_series),
                gpu_index=int(args.gpu_index),
                log_every=int(args.log_every),
            )

            # -------- Substep 4: Farneback optical flow magnitude only
            def _farneback_only() -> Any:
                import cv2

                out: List[float] = []
                prev_gray = cv2.cvtColor(fm.get(frame_indices[0]), cv2.COLOR_RGB2GRAY)
                prev_gray = _resize_gray_max_side(prev_gray, int(args.flow_max_side))
                for i in range(1, len(frame_indices)):
                    fB = fm.get(frame_indices[i])
                    gray = cv2.cvtColor(fB, cv2.COLOR_RGB2GRAY)
                    gray = _resize_gray_max_side(gray, int(args.flow_max_side))
                    mag, _mag, _ang = optical_flow_magnitude(prev_gray, gray)
                    out.append(float(mag))
                    prev_gray = gray
                return out

            name_flow = "cut_detection.substep.feature_farneback_flowmag_only"
            results[name_flow] = _bench_call(
                fn=_farneback_only,
                name=name_flow,
                warmup=int(args.warmup),
                repeats=int(args.repeats),
                runs=int(args.runs),
                units_per_call=int(units),
                sampler_interval_sec=float(sampler_interval_sec),
                spike_mad_k=float(args.spike_mad_k),
                record_series=bool(args.record_series),
                gpu_index=int(args.gpu_index),
                log_every=int(args.log_every),
            )

            # -------- Substep 5: postprocess only (use cached features once)
            # Precompute features once to isolate thresholding+smoothing+cleanup overhead.
            LOGGER.info("Precomputing cut_detection features once (for postprocess-only bench)...")
            import cv2
            import numpy as _np
            from scipy.ndimage import gaussian_filter1d
            from scipy.signal import medfilt
            from modules.cut_detection.cut_detection import morphological_clean_cuts  # type: ignore

            hdiffs: List[float] = []
            ssim_diffs: List[float] = []
            flow_mags: List[float] = []
            prev_gray = cv2.cvtColor(fm.get(frame_indices[0]), cv2.COLOR_RGB2GRAY)
            prev_gray_flow = _resize_gray_max_side(prev_gray, int(args.flow_max_side))
            prev_gray_ssim = _resize_gray_max_side(prev_gray, int(args.ssim_max_side))
            for i in range(1, len(frame_indices)):
                fA = fm.get(frame_indices[i - 1])
                fB = fm.get(frame_indices[i])
                hdiffs.append(float(frame_histogram_diff(fA, fB)))
                gray = cv2.cvtColor(fB, cv2.COLOR_RGB2GRAY)
                gray_flow = _resize_gray_max_side(gray, int(args.flow_max_side))
                gray_ssim = _resize_gray_max_side(gray, int(args.ssim_max_side))
                try:
                    from skimage.metrics import structural_similarity as _ssim

                    dr = float(gray_ssim.max() - gray_ssim.min())
                    dr = dr if dr > 1e-9 else 1.0
                    ssim_diffs.append(float(1.0 - _ssim(prev_gray_ssim, gray_ssim, data_range=dr)))
                except Exception:
                    ssim_diffs.append(0.0)
                mag, _mag, _ang = optical_flow_magnitude(prev_gray_flow, gray_flow)
                flow_mags.append(float(mag))
                prev_gray = gray
                prev_gray_flow = gray_flow
                prev_gray_ssim = gray_ssim

            hdiffs_np = _np.asarray(hdiffs, dtype=_np.float32)
            ssim_np = _np.asarray(ssim_diffs, dtype=_np.float32)
            flow_np = _np.asarray(flow_mags, dtype=_np.float32)
            deep_np = _np.zeros_like(hdiffs_np, dtype=_np.float32)

            def _postprocess_only() -> Any:
                # This mirrors detect_hard_cuts postprocessing logic (score + smoothing + cleanup).
                # Keep this in sync with cut_detection.detect_hard_cuts when you change it.
                n = int(len(frame_indices))
                if n < 2:
                    return [], []
                # Adaptive thresholds (same formulas)
                hist_thresh = float(_np.median(hdiffs_np) + 2.0 * _np.std(hdiffs_np))
                ssim_thresh = float(_np.median(ssim_np) + 1.5 * _np.std(ssim_np))
                flow_thresh = float(_np.median(flow_np) + 2.0 * _np.std(flow_np))
                deep_thresh = 0.0

                scores = _np.zeros(hdiffs_np.shape[0], dtype=_np.float32)
                scores += (hdiffs_np > hist_thresh).astype(_np.float32)
                scores += (ssim_np > ssim_thresh).astype(_np.float32)
                scores += (flow_np > flow_thresh).astype(_np.float32)
                scores += (deep_np > deep_thresh).astype(_np.float32)

                cut_candidates: List[Any] = []
                if scores.size > 3:
                    scores_median = medfilt(scores.astype(float), kernel_size=3)
                    scores_smooth = gaussian_filter1d(scores_median, sigma=1.0)
                    for i in range(1, int(len(scores_smooth)) - 1):
                        if scores_smooth[i] > scores_smooth[i - 1] and scores_smooth[i] > scores_smooth[i + 1]:
                            if scores_smooth[i] >= 2.0:
                                cut_candidates.append((i + 1, float(scores_smooth[i])))
                else:
                    cut_candidates = [(i + 1, float(s)) for i, s in enumerate(scores) if float(s) >= 2.0]

                cut_flag_array = _np.zeros(int(len(frame_indices)) - 1, dtype=int)
                for idx, _ in cut_candidates:
                    if int(idx) - 1 < int(len(cut_flag_array)):
                        cut_flag_array[int(idx) - 1] = 1
                cut_flag_array = morphological_clean_cuts(cut_flag_array, min_neighbors=0)

                cleaned_candidates = [(i + 1, float(scores[i])) for i in range(int(len(cut_flag_array))) if int(cut_flag_array[i]) == 1]
                cut_idxs: List[int] = []
                strengths: List[float] = []
                for idx, strength in cleaned_candidates:
                    if not cut_idxs or int(idx) - int(cut_idxs[-1]) > 5:
                        cut_idxs.append(int(idx))
                        strengths.append(float(strength))
                return cut_idxs, strengths

            name_post = "cut_detection.substep.postprocess_only(cached_features)"
            results[name_post] = _bench_call(
                fn=_postprocess_only,
                name=name_post,
                warmup=int(args.warmup),
                repeats=int(args.repeats),
                runs=int(args.runs),
                units_per_call=int(units),
                sampler_interval_sec=float(sampler_interval_sec),
                spike_mad_k=float(args.spike_mad_k),
                record_series=bool(args.record_series),
                gpu_index=int(args.gpu_index),
                log_every=int(args.log_every),
            )

    if "cut_detection_soft" in only:
        LOGGER.info("Importing cut_detection.detect_soft_cuts ...")
        from modules.cut_detection.cut_detection import detect_soft_cuts  # type: ignore  # noqa: E402

        def _run_soft() -> Any:
            # Soft cuts are sequence-based; treat units as transitions (frame pairs) for cost modeling.
            return detect_soft_cuts(
                fm,
                frame_indices,
                fps=float(args.fps),
                flow_max_side=int(args.flow_max_side),
            )

        units = max(1, len(frame_indices) - 1)
        name_soft = "cut_detection.detect_soft_cuts(cpu)"
        results[name_soft] = _bench_call(
            fn=_run_soft,
            name=name_soft,
            warmup=int(args.warmup),
            repeats=int(args.repeats),
            runs=int(args.runs),
            units_per_call=int(units),
            sampler_interval_sec=float(sampler_interval_sec),
            spike_mad_k=float(args.spike_mad_k),
            record_series=bool(args.record_series),
            gpu_index=int(args.gpu_index),
            log_every=int(args.log_every),
        )

    if "cut_detection_motion" in only:
        LOGGER.info("Importing cut_detection.detect_motion_based_cuts ...")
        from modules.cut_detection.cut_detection import detect_motion_based_cuts  # type: ignore  # noqa: E402

        def _run_motion() -> Any:
            return detect_motion_based_cuts(
                fm,
                frame_indices,
                flow_max_side=int(args.flow_max_side),
            )

        units = max(1, len(frame_indices) - 1)
        name_motion = "cut_detection.detect_motion_based_cuts(cpu)"
        results[name_motion] = _bench_call(
            fn=_run_motion,
            name=name_motion,
            warmup=int(args.warmup),
            repeats=int(args.repeats),
            runs=int(args.runs),
            units_per_call=int(units),
            sampler_interval_sec=float(sampler_interval_sec),
            spike_mad_k=float(args.spike_mad_k),
            record_series=bool(args.record_series),
            gpu_index=int(args.gpu_index),
            log_every=int(args.log_every),
        )

    payload = {
        "created_at": datetime.utcnow().isoformat(),
        "repo": _git_info(repo_root),
        "system": _system_info(),
        "protocol": {
            "mode": "micro_components",
            "warmup": int(args.warmup),
            "repeats": int(args.repeats),
            "runs": int(args.runs),
            "sampler_interval_ms": float(args.sampler_interval_ms),
            "record_series": bool(args.record_series),
            "spike_mad_k": float(args.spike_mad_k),
            "gpu_index": int(args.gpu_index),
        },
        "synthetic_input": {"frames_dir": frames_dir, "width": int(args.width), "height": int(args.height), "fps": float(args.fps), "n_frames": int(args.n_frames)},
        "results": results,
    }

    out_json = os.path.join(out_dir, "checklist_components_micro_results.json")
    _write_json(out_json, payload)
    LOGGER.info("Wrote JSON: %s", out_json)
    print(out_json)


if __name__ == "__main__":
    main()


