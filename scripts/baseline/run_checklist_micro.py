#!/usr/bin/env python3
"""
Baseline checklist micro-runner.

What it measures (MVP):
- Triton model branches used in baseline (latency + CPU RSS peak + GPU VRAM peak)
- Audio extractors (clap/loudness/tempo) per Segmenter-like segment duration

Design:
- "Micro" = per item (one frame / one segment) with warmup + N repeats.
- GPU peak is measured via `nvidia-smi` polling (pynvml is not required).
- Results are written to JSON (+ a compact Markdown summary).

Notes:
- This runner is intentionally *model-level / extractor-level* for reproducibility.
- Component-level (full VisualProcessor modules) is tracked by the checklist doc and can be added later.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import os
import platform
import sys
import subprocess
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _repo_root() -> str:
    # scripts/baseline/run_checklist_micro.py lives at <repo>/scripts/baseline/...
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))


# Ensure repo-root packages are importable (dp_models, dp_triton, etc.)
_ROOT = _repo_root()
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _utc_ts() -> str:
    return datetime.utcnow().strftime("%Y%m%d-%H%M%S")


def _ensure_dir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def _run(cmd: List[str]) -> Tuple[int, str, str]:
    p = subprocess.run(cmd, check=False, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return int(p.returncode), str(p.stdout), str(p.stderr)

def _run_cwd(cmd: List[str], cwd: str) -> Tuple[int, str, str]:
    p = subprocess.run(cmd, check=False, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    return int(p.returncode), str(p.stdout), str(p.stderr)


def _git_info(repo_root: str) -> Dict[str, Any]:
    """
    Best-effort repo version manifest. Never fails hard.
    """
    info: Dict[str, Any] = {"commit": None, "dirty": None}
    try:
        code, out, _ = _run_cwd(["git", "rev-parse", "HEAD"], cwd=repo_root)
        if code == 0:
            info["commit"] = out.strip()
    except Exception:
        pass
    try:
        code, out, _ = _run_cwd(["git", "status", "--porcelain"], cwd=repo_root)
        if code == 0:
            info["dirty"] = bool(out.strip())
    except Exception:
        pass
    return info


def _system_info() -> Dict[str, Any]:
    """
    Best-effort runtime manifest to make runs reproducible.
    """
    info: Dict[str, Any] = {
        "python": sys.version.replace("\n", " "),
        "platform": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
        },
        "nvidia_smi": None,
    }
    try:
        code, out, err = _run(["nvidia-smi"])
        if code == 0:
            info["nvidia_smi"] = out.strip()
        else:
            info["nvidia_smi"] = (out + "\n" + err).strip()
    except Exception:
        pass
    return info


def _try_import_pynvml():
    try:
        # pynvml emits a FutureWarning in some environments; it's noisy for baseline runners.
        import warnings
        warnings.filterwarnings("ignore", category=FutureWarning)
        import pynvml  # type: ignore
        return pynvml
    except Exception:
        return None


def _proc_name(pid: int) -> Optional[str]:
    """
    Best-effort Linux process name for PID (comm/cmdline).
    """
    try:
        p = f"/proc/{int(pid)}/comm"
        if os.path.exists(p):
            with open(p, "r", encoding="utf-8") as f:
                return f.read().strip()
    except Exception:
        pass
    try:
        p = f"/proc/{int(pid)}/cmdline"
        if os.path.exists(p):
            with open(p, "rb") as f:
                raw = f.read().split(b"\x00")
            s = " ".join(x.decode("utf-8", errors="ignore") for x in raw if x)
            s = s.strip()
            return s or None
    except Exception:
        pass
    return None


def _now_ms() -> float:
    return float(time.perf_counter() * 1000.0)


def _percentiles(xs: List[float], ps: List[float]) -> Dict[str, float]:
    if not xs:
        return {f"p{int(p)}": float("nan") for p in ps}
    a = np.asarray(xs, dtype=np.float64)
    out: Dict[str, float] = {}
    for p in ps:
        out[f"p{int(p)}"] = float(np.percentile(a, float(p)))
    return out


def _stats_latency_ms(samples_ms: List[float]) -> Dict[str, Any]:
    if not samples_ms:
        return {
            "n": 0,
            "mean": float("nan"),
            "std": float("nan"),
            "min": float("nan"),
            "max": float("nan"),
            "p50": float("nan"),
            "p90": float("nan"),
            "p95": float("nan"),
            "p99": float("nan"),
            "cv": float("nan"),
            "spike_fraction": 0.0,
            "spike_rule": None,
        }
    a = np.asarray(samples_ms, dtype=np.float64)
    mean = float(np.mean(a))
    std = float(np.std(a))
    cv = float(std / (mean + 1e-12))
    pct = _percentiles(samples_ms, [50, 90, 95, 99])

    return {
        "n": int(a.size),
        "mean": mean,
        "std": std,
        "min": float(np.min(a)),
        "max": float(np.max(a)),
        **pct,
        "cv": cv,
        "spike_fraction": 0.0,
        "spike_rule": None,
    }


def _detect_spikes_mad(samples_ms: List[float], *, k: float = 4.0) -> Dict[str, Any]:
    """
    Deterministic spikes detector:
    spike if sample > median + k*MAD (MAD over raw samples).
    """
    if not samples_ms:
        return {"spikes": False, "spike_fraction": 0.0, "rule": f"median + {k}*MAD"}
    a = np.asarray(samples_ms, dtype=np.float64)
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med)))
    if mad <= 1e-12:
        return {"spikes": False, "spike_fraction": 0.0, "rule": f"median + {k}*MAD"}
    thr = med + float(k) * mad
    m = a > thr
    frac = float(np.mean(m))
    return {"spikes": bool(np.any(m)), "spike_fraction": frac, "rule": f"x > median({med:.3f}) + {k}*MAD({mad:.3f}) = {thr:.3f}"}


class VramProbe:
    """
    VRAM reader with NVML (preferred) fallback to nvidia-smi.
    Tracks:
    - total GPU used (MB)
    - per-process used sum (MB) for a given match set (by pid->procname, or process_name via nvidia-smi)
    """

    def __init__(self, gpu_index: int = 0):
        self.gpu_index = int(gpu_index)
        self._pynvml = _try_import_pynvml()
        self._nvml_handle = None
        if self._pynvml is not None:
            try:
                self._pynvml.nvmlInit()
                self._nvml_handle = self._pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            except Exception:
                self._nvml_handle = None

    def backend(self) -> str:
        return "nvml" if self._nvml_handle is not None else "nvidia-smi"

    def total_used_mb(self) -> Optional[float]:
        if self._nvml_handle is not None:
            try:
                mi = self._pynvml.nvmlDeviceGetMemoryInfo(self._nvml_handle)
                return float(mi.used) / (1024.0 * 1024.0)
            except Exception:
                return None
        return _nvidia_smi_total_used_mb(self.gpu_index)

    def process_used_mb(self, *, match_substrings: List[str]) -> Optional[float]:
        subs = [str(s).lower() for s in (match_substrings or []) if str(s).strip()]
        if not subs:
            return None

        # NVML: iterate compute procs and match by /proc/<pid> name/cmdline
        if self._nvml_handle is not None:
            try:
                procs = self._pynvml.nvmlDeviceGetComputeRunningProcesses_v3(self._nvml_handle)
            except Exception:
                try:
                    procs = self._pynvml.nvmlDeviceGetComputeRunningProcesses(self._nvml_handle)
                except Exception:
                    procs = []
            total = 0.0
            found = False
            for pr in procs or []:
                try:
                    pid = int(getattr(pr, "pid"))
                    used_b = int(getattr(pr, "usedGpuMemory"))
                except Exception:
                    continue
                pname = _proc_name(pid)
                pname_l = str(pname or "").lower()
                if any(s in pname_l for s in subs):
                    total += float(used_b) / (1024.0 * 1024.0)
                    found = True
            return float(total) if found else 0.0

        # Fallback: nvidia-smi compute apps output (name+pid if supported)
        return _nvidia_smi_process_used_mb(match_substrings=subs, gpu_index=self.gpu_index)

def _nvidia_smi_total_used_mb(gpu_index: int = 0) -> Optional[float]:
    try:
        code, out, _err = _run(
            [
                "nvidia-smi",
                f"--query-gpu=memory.used",
                "--format=csv,noheader,nounits",
                f"--id={int(gpu_index)}",
            ]
        )
        if code != 0:
            return None
        s = out.strip().splitlines()[0].strip()
        return float(s)
    except Exception:
        return None


def _nvidia_smi_process_used_mb(*, match_substrings: List[str], gpu_index: int = 0) -> Optional[float]:
    """
    Sum used_memory for compute apps whose process_name contains any of match_substrings.
    This is much closer to "VRAM used by Triton" than total GPU used.
    """
    try:
        # Try to include pid for better debugging when supported.
        code, out, _err = _run(
            [
                "nvidia-smi",
                "--query-compute-apps=pid,process_name,used_memory",
                "--format=csv,noheader,nounits",
            ]
        )
        if code != 0:
            return None
        subs = [str(s).lower() for s in (match_substrings or []) if str(s).strip()]
        if not subs:
            return None
        total = 0.0
        found = False
        for line in out.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) < 2:
                continue
            # Format: pid, process_name, used_memory
            if len(parts) >= 3:
                _pid, pname, mem = parts[0], parts[1], parts[2]
            else:
                # Legacy fallback: process_name, used_memory
                pname, mem = parts[0], parts[1]
            pname = str(pname).lower()
            if any(s in pname for s in subs):
                try:
                    total += float(mem)
                    found = True
                except Exception:
                    pass
        return float(total) if found else 0.0
    except Exception:
        return None


def _rss_mb() -> Optional[float]:
    try:
        import psutil  # type: ignore

        p = psutil.Process(os.getpid())
        return float(p.memory_info().rss) / (1024.0 * 1024.0)
    except Exception:
        # Fallback: resource.getrusage (peak RSS of the process).
        # On Linux: ru_maxrss is in kilobytes. On macOS: bytes.
        try:
            import resource

            r = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            if sys.platform.startswith("linux"):
                return r / 1024.0
            # best-effort for non-linux
            # If it's already MB-scale, return as-is; else assume bytes.
            return r if r < 10_000.0 else (r / (1024.0 * 1024.0))
        except Exception:
            return None


@dataclasses.dataclass
class PeakSampler:
    """
    Polls GPU VRAM (nvidia-smi) + CPU RSS in a background thread and stores peaks.
    """

    interval_sec: float = 0.05
    gpu_index: int = 0
    record_series: bool = False
    match_process_substrings: List[str] = dataclasses.field(default_factory=lambda: ["tritonserver"])
    _stop: threading.Event = dataclasses.field(default_factory=threading.Event)
    _thread: Optional[threading.Thread] = None

    _probe: Optional[VramProbe] = None

    gpu_peak_mb: Optional[float] = None
    gpu_peak_triton_mb: Optional[float] = None
    rss_peak_mb: Optional[float] = None
    series: Optional[Dict[str, List[float]]] = None

    def start(self) -> None:
        self._stop.clear()
        self.gpu_peak_mb = None
        self.gpu_peak_triton_mb = None
        self.rss_peak_mb = None
        self.series = {"t_ms": [], "gpu_used_mb": [], "gpu_triton_mb": [], "rss_mb": []} if self.record_series else None
        self._probe = VramProbe(gpu_index=self.gpu_index)

        def _loop() -> None:
            while not self._stop.is_set():
                g = self._probe.total_used_mb() if self._probe else _nvidia_smi_total_used_mb(self.gpu_index)
                gt = (
                    self._probe.process_used_mb(match_substrings=self.match_process_substrings)
                    if self._probe
                    else _nvidia_smi_process_used_mb(match_substrings=self.match_process_substrings, gpu_index=self.gpu_index)
                )
                r = _rss_mb()
                if g is not None:
                    self.gpu_peak_mb = g if self.gpu_peak_mb is None else max(self.gpu_peak_mb, g)
                if gt is not None:
                    self.gpu_peak_triton_mb = gt if self.gpu_peak_triton_mb is None else max(self.gpu_peak_triton_mb, gt)
                if r is not None:
                    self.rss_peak_mb = r if self.rss_peak_mb is None else max(self.rss_peak_mb, r)
                if self.series is not None:
                    self.series["t_ms"].append(_now_ms())
                    self.series["gpu_used_mb"].append(float(g) if g is not None else float("nan"))
                    self.series["gpu_triton_mb"].append(float(gt) if gt is not None else float("nan"))
                    self.series["rss_mb"].append(float(r) if r is not None else float("nan"))
                time.sleep(self.interval_sec)

        self._thread = threading.Thread(target=_loop, name="peak_sampler", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._thread = None


def _stable_mean_ms(samples_ms: List[float]) -> Tuple[float, bool]:
    """
    Robust-ish "stable mean":
    - compute median + MAD
    - drop points outside median ± 4*MAD (if MAD>0)
    - mean of remaining
    Returns: (mean, spikes_present)
    """
    if not samples_ms:
        return float("nan"), False
    a = np.asarray(samples_ms, dtype=np.float32)
    med = float(np.median(a))
    mad = float(np.median(np.abs(a - med)))
    if mad <= 1e-9:
        return float(np.mean(a)), False
    lo = med - 4.0 * mad
    hi = med + 4.0 * mad
    keep = (a >= lo) & (a <= hi)
    spikes = bool(np.any(~keep))
    aa = a[keep]
    if aa.size == 0:
        return float(np.mean(a)), True
    return float(np.mean(aa)), spikes


def _append_results_md(*, md_path: str, rows: List[Dict[str, Any]]) -> None:
    """
    Append markdown table rows into docs/baseline/BASELINE_COMPONENT_MODEL_RESULTS.md.
    We keep it simple: append rows at the end of the file under the table.
    """
    if not rows:
        return
    p = str(md_path)
    if not p:
        return
    # Ensure file exists
    if not os.path.exists(p):
        _ensure_dir(os.path.dirname(p) or ".")
        with open(p, "w", encoding="utf-8") as f:
            f.write("## Baseline component/model results (data)\n\n")
            f.write("(auto-generated)\n\n")
            f.write("| component_or_module | model_branch | source_resolution | selected_branch | latency_ms_mean_stable | spikes | rss_peak_mb | vram_triton_delta_run_mb | status | notes | out_dir |\n")
            f.write("|---|---|---:|---:|---:|:---:|---:|---:|---|---|---|\n")
    # Append
    with open(p, "a", encoding="utf-8") as f:
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
                        str(r.get("spikes", "")),
                        str(r.get("rss_peak_mb", "")),
                        str(r.get("vram_triton_delta_run_mb", "")),
                        str(r.get("status", "")),
                        str(r.get("notes", "")),
                        str(r.get("out_dir", "")),
                    ]
                )
                + " |\n"
            )


def _resolve_model_mm(model_spec_name: str) -> Tuple[dict, Any, dict]:
    from dp_models import get_global_model_manager  # type: ignore

    mm = get_global_model_manager()
    rm = mm.get(model_name=str(model_spec_name))
    handle = rm.handle or {}
    rp = rm.spec.runtime_params or {}
    if not isinstance(handle, dict) or "client" not in handle:
        raise RuntimeError(f"ModelManager returned empty Triton handle for: {model_spec_name}")
    if not isinstance(rp, dict) or not rp:
        raise RuntimeError(f"ModelManager returned empty runtime_params for: {model_spec_name}")
    return rm.models_used_entry, handle["client"], rp


def _measure_triton_model(
    *,
    name: str,
    model_spec_name: str,
    kind: str,
    warmup: int,
    repeats: int,
    runs: int,
    sampler_interval_sec: float,
    spike_mad_k: float,
    record_series: bool,
    gpu_index: int,
    warn_triton_drift_mb: float,
    tensor_spec: dict,
    tensor_spec1: Optional[dict] = None,
) -> Dict[str, Any]:
    models_used_entry, client, rp = _resolve_model_mm(model_spec_name)
    triton_model_name = str(rp.get("triton_model_name") or "")
    triton_model_version = rp.get("triton_model_version")
    triton_model_version = str(triton_model_version) if triton_model_version not in (None, "") else None

    input_name = str(rp.get("triton_input_name") or "INPUT__0")
    output_name = str(rp.get("triton_output_name") or "OUTPUT__0")
    input_dt = str(rp.get("triton_input_datatype") or "FP32")

    input0_name = str(rp.get("triton_input0_name") or "INPUT0__0")
    input1_name = str(rp.get("triton_input1_name") or "INPUT1__0")

    def make_tensor(spec: dict) -> np.ndarray:
        dt = str(spec.get("dtype") or "FP32").upper()
        shape = spec.get("shape")
        if not isinstance(shape, list) or not shape:
            raise ValueError(f"{name} invalid shape: {shape}")
        shape_t = tuple(int(x) for x in shape)
        kindv = str(spec.get("value") or "random_normal").strip().lower()
        if dt == "UINT8":
            if kindv != "random_int":
                kindv = "random_int"
            lo, hi = 0, 256
            r = spec.get("int_range")
            if isinstance(r, list) and len(r) == 2:
                lo, hi = int(r[0]), int(r[1])
            x = np.random.randint(lo, hi, size=shape_t, dtype=np.uint8)
            return x
        if dt in ("FP32", "FP16"):
            lo, hi = 0.0, 1.0
            r = spec.get("float_range")
            if isinstance(r, list) and len(r) == 2:
                lo, hi = float(r[0]), float(r[1])
            x = (lo + (hi - lo) * np.random.rand(*shape_t)).astype(np.float32)
            return x.astype(np.float16) if dt == "FP16" else x
        if dt == "INT64":
            lo, hi = 0, 100
            r = spec.get("int_range")
            if isinstance(r, list) and len(r) == 2:
                lo, hi = int(r[0]), int(r[1])
            x = np.random.randint(lo, hi, size=shape_t, dtype=np.int64)
            return x
        raise ValueError(f"{name} unsupported dtype: {dt}")

    x0 = make_tensor(tensor_spec)
    x1 = make_tensor(tensor_spec1) if tensor_spec1 is not None else None

    status = "ok"
    err = None
    lat_ms_all: List[float] = []
    run_blocks: List[Dict[str, Any]] = []

    # Probe used for baseline readings (outside sampler loop too).
    probe = VramProbe(gpu_index=int(gpu_index))

    # Baseline before all runs (best-effort).
    vram_before_all_triton = probe.process_used_mb(match_substrings=["tritonserver"])

    for run_idx in range(max(1, int(runs))):
        sampler = PeakSampler(
            interval_sec=float(sampler_interval_sec),
            gpu_index=int(gpu_index),
            record_series=bool(record_series),
            match_process_substrings=["tritonserver"],
        )
        lat_ms: List[float] = []

        vram_before_run_triton = probe.process_used_mb(match_substrings=["tritonserver"])
        sampler.start()
        try:
            for _ in range(max(0, int(warmup))):
                if kind == "two_inputs":
                    _ = client.infer_two_inputs(
                        model_name=triton_model_name,
                        model_version=triton_model_version,
                        input0_name=input0_name,
                        input0_tensor=x0,
                        input1_name=input1_name,
                        input1_tensor=x1,
                        output_name=output_name,
                        datatype=input_dt,
                    )
                else:
                    _ = client.infer(
                        model_name=triton_model_name,
                        model_version=triton_model_version,
                        input_name=input_name,
                        input_tensor=x0,
                        output_name=output_name,
                        datatype=input_dt,
                    )

            for _ in range(max(1, int(repeats))):
                t0 = time.perf_counter()
                if kind == "two_inputs":
                    _ = client.infer_two_inputs(
                        model_name=triton_model_name,
                        model_version=triton_model_version,
                        input0_name=input0_name,
                        input0_tensor=x0,
                        input1_name=input1_name,
                        input1_tensor=x1,
                        output_name=output_name,
                        datatype=input_dt,
                    )
                else:
                    _ = client.infer(
                        model_name=triton_model_name,
                        model_version=triton_model_version,
                        input_name=input_name,
                        input_tensor=x0,
                        output_name=output_name,
                        datatype=input_dt,
                    )
                lat_ms.append(float((time.perf_counter() - t0) * 1000.0))
        except Exception as e:
            status = "error"
            err = str(e)
        finally:
            sampler.stop()

        # Best-effort "after" reading. Useful for drift diagnostics.
        # NOTE: Triton/ORT pools may keep memory, so after may be > before even if peak is small.
        vram_after_run_triton = probe.process_used_mb(match_substrings=["tritonserver"])

        lat_ms_all.extend(lat_ms)

        spikes_block = _detect_spikes_mad(lat_ms, k=float(spike_mad_k))
        stats_block = _stats_latency_ms(lat_ms)
        stats_block["spike_fraction"] = float(spikes_block["spike_fraction"])
        stats_block["spike_rule"] = str(spikes_block["rule"])

        # VRAM delta_run (peak-before) for this run.
        try:
            vram_peak_triton = float(sampler.gpu_peak_triton_mb) if sampler.gpu_peak_triton_mb is not None else None
        except Exception:
            vram_peak_triton = None
        try:
            vram_delta_run = (float(vram_peak_triton) - float(vram_before_run_triton)) if (vram_peak_triton is not None and vram_before_run_triton is not None) else None
        except Exception:
            vram_delta_run = None

        run_blocks.append(
            {
                "run_idx": int(run_idx),
                "latency_ms_samples": lat_ms,
                "latency_stats": stats_block,
                # Keep both explicit spike detectors.
                "spikes_mad": bool(spikes_block["spikes"]),
                "spike_fraction_mad": float(spikes_block["spike_fraction"]),
                "spikes_rule_mad": str(spikes_block["rule"]),
                "cpu_rss_peak_mb": sampler.rss_peak_mb,
                "vram_probe_backend": sampler._probe.backend() if sampler._probe else probe.backend(),
                "vram_triton_before_mb": vram_before_run_triton,
                "vram_triton_peak_mb": vram_peak_triton,
                "vram_triton_after_mb": vram_after_run_triton,
                "vram_triton_delta_run_mb": vram_delta_run,
                "series": sampler.series,
            }
        )
        if status != "ok":
            break

    mean_stable, spikes_stable_filter = _stable_mean_ms(lat_ms_all)
    spikes_mad = _detect_spikes_mad(lat_ms_all, k=float(spike_mad_k))
    stats_all = _stats_latency_ms(lat_ms_all)
    stats_all["spike_fraction"] = float(spikes_mad["spike_fraction"])
    stats_all["spike_rule"] = str(spikes_mad["rule"])

    # Derive summary VRAM: take max across runs (more conservative).
    vram_delta_run_max = None
    vram_peak_max = None
    rss_peak_max = None
    try:
        peaks = [rb.get("vram_triton_peak_mb") for rb in run_blocks if rb.get("vram_triton_peak_mb") is not None]
        if peaks:
            vram_peak_max = float(max(float(x) for x in peaks))  # type: ignore[arg-type]
    except Exception:
        pass
    try:
        deltas = [rb.get("vram_triton_delta_run_mb") for rb in run_blocks if rb.get("vram_triton_delta_run_mb") is not None]
        if deltas:
            vram_delta_run_max = float(max(float(x) for x in deltas))  # type: ignore[arg-type]
    except Exception:
        pass
    try:
        rss = [rb.get("cpu_rss_peak_mb") for rb in run_blocks if rb.get("cpu_rss_peak_mb") is not None]
        if rss:
            rss_peak_max = float(max(float(x) for x in rss))  # type: ignore[arg-type]
    except Exception:
        pass

    # Drift diagnostics: how much Triton's process memory "sticks" after the run.
    vram_after_all_triton = probe.process_used_mb(match_substrings=["tritonserver"])
    try:
        vram_drift_mb = (
            (float(vram_after_all_triton) - float(vram_before_all_triton))
            if (vram_after_all_triton is not None and vram_before_all_triton is not None)
            else None
        )
    except Exception:
        vram_drift_mb = None

    restart_recommended = False
    restart_reason = None
    try:
        if vram_drift_mb is not None and float(vram_drift_mb) >= float(warn_triton_drift_mb):
            restart_recommended = True
            restart_reason = f"triton vram drift {float(vram_drift_mb):.1f}MB >= {float(warn_triton_drift_mb):.1f}MB"
    except Exception:
        pass

    return {
        "name": name,
        "model_spec_name": model_spec_name,
        "triton_model_name": triton_model_name,
        "status": status,
        "error": err,
        "warmup": int(warmup),
        "repeats": int(repeats),
        "runs": int(max(1, int(runs))),
        "sampler_interval_sec": float(sampler_interval_sec),
        "latency_ms_samples": lat_ms_all,
        "latency_stats": stats_all,
        "latency_ms_mean_stable": mean_stable,
        # Explicit, non-confusing flags:
        "spikes_mad": bool(spikes_mad["spikes"]),
        "spike_fraction_mad": float(spikes_mad["spike_fraction"]),
        "spikes_rule_mad": str(spikes_mad["rule"]),
        "spikes_stable_filter": bool(spikes_stable_filter),
        # Back-compat aggregate:
        "spikes": bool(bool(spikes_stable_filter) or bool(spikes_mad["spikes"])),
        "cpu_rss_peak_mb": rss_peak_max,
        "vram_triton_before_all_mb": vram_before_all_triton,
        "vram_triton_peak_mb": vram_peak_max,
        "vram_triton_after_all_mb": vram_after_all_triton,
        "vram_triton_drift_mb": vram_drift_mb,
        "restart_recommended": bool(restart_recommended),
        "restart_reason": restart_reason,
        # Triton-aware VRAM: keep delta_run (= peak-before). We store max across runs.
        "vram_triton_delta_run_mb": vram_delta_run_max,
        "runs_detail": run_blocks,
        "models_used": models_used_entry,
    }


def _classify_error(err: Optional[str]) -> str:
    """
    Best-effort classification for reporting / restart guidance.
    """
    if not err:
        return "none"
    s = str(err).lower()
    if "connection reset by peer" in s:
        return "triton_crash"
    if "connection refused" in s:
        return "triton_down"
    # Common ORT CUDA OOM signatures
    if "failed to allocate memory" in s:
        return "oom"
    if "cudaerrormemoryallocation" in s:
        return "oom"
    if "bfc_arena" in s and "allocate" in s:
        return "oom"
    if "cublasstatus" in s:
        return "oom_or_cuda"
    return "other"


def _linspace(a: float, b: float, n: int) -> List[float]:
    if n <= 1:
        return [float(b)]
    step = (b - a) / float(n - 1)
    return [float(a + i * step) for i in range(int(n))]


def _visual_grid(short_sides: List[int]) -> List[dict]:
    out = []
    for s in short_sides:
        s = int(s)
        if s <= 0:
            continue
        w16 = int(round(float(s) * 16.0 / 9.0))
        h16 = int(s)
        out.append({"format": "16:9", "short_side": s, "width": w16, "height": h16})
        w9 = int(s)
        h9 = int(round(float(s) * 16.0 / 9.0))
        out.append({"format": "9:16", "short_side": s, "width": w9, "height": h9})
    return out


def _select_branch_s(max_dim: int) -> str:
    d = int(max_dim)
    if d <= 320:
        return "small"
    if d <= 448:
        return "medium"
    return "large"


def main() -> None:
    ap = argparse.ArgumentParser("baseline checklist micro runner")
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--triton-http-url", type=str, default=None)
    ap.add_argument("--dp-models-root", type=str, default=None)
    ap.add_argument("--gpu-index", type=int, default=0, help="CUDA device index to sample VRAM from (default 0).")
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=10)
    ap.add_argument("--runs", type=int, default=1, help="Repeat full warmup+repeats block N times for stability.")
    ap.add_argument("--sampler-interval-ms", type=float, default=50.0, help="Resource sampler polling interval (ms).")
    ap.add_argument("--record-series", action="store_true", help="Record per-sample time series (debug spikes).")
    ap.add_argument("--spike-mad-k", type=float, default=4.0, help="Spikes rule: sample > median + k*MAD.")
    ap.add_argument(
        "--warn-triton-drift-mb",
        type=float,
        default=256.0,
        help="If Triton per-process VRAM drift (after-before) exceeds this, mark restart_recommended=true.",
    )
    ap.add_argument(
        "--profile",
        type=str,
        default="fast",
        choices=["fast", "stable", "debug"],
        help="Preset knobs. fast: warmup=1 repeats=10 runs=1. stable: warmup=3 repeats=30 runs=3. debug: fast + record-series.",
    )
    ap.add_argument(
        "--only-models",
        type=str,
        default="",
        help="Comma-separated model job names to run (e.g., clip_image_448). Empty = run selected groups.",
    )
    ap.add_argument(
        "--results-md",
        type=str,
        default="",
        help="If set, append a single-row summary into this markdown results table (e.g., docs/baseline/BASELINE_COMPONENT_MODEL_RESULTS.md).",
    )
    ap.add_argument(
        "--source-format",
        type=str,
        default="16:9",
        choices=["16:9", "9:16"],
        help="Which source format to use when writing results-md row (table is per source_resolution).",
    )
    ap.add_argument(
        "--model-groups",
        type=str,
        default="all",
        help="Comma-separated model groups to run: all,clip_places,midas_raft,yolo (useful for 6GB: run one group, restart Triton, then next).",
    )
    ap.add_argument(
        "--stop-on-oom",
        action="store_true",
        help="Stop model loop on first OOM-like error and write NEXT_STEPS.md with resume instructions.",
    )
    ap.add_argument(
        "--skip-audio",
        action="store_true",
        help="Skip audio extractors measurement (useful when focusing on Triton VRAM stability).",
    )
    ap.add_argument(
        "--clap-device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda", "auto"],
        help="Device for CLAP extractor. Default cpu to avoid competing with Triton VRAM on 6GB GPUs.",
    )
    ap.add_argument(
        "--visual-short-sides",
        type=str,
        default="128,160,224,256,280,320,384,448,512,640,720,768,896,960,1080",
    )
    ap.add_argument("--audio-min-sec", type=float, default=0.1)
    ap.add_argument("--audio-max-sec", type=float, default=20.0)
    ap.add_argument("--audio-n", type=int, default=25)
    args = ap.parse_args()

    if args.triton_http_url:
        os.environ["TRITON_HTTP_URL"] = str(args.triton_http_url)
    if args.dp_models_root:
        os.environ["DP_MODELS_ROOT"] = str(args.dp_models_root)

    out_dir = args.out_dir or os.path.join("docs", "baseline", "out", f"checklist-{_utc_ts()}")
    _ensure_dir(out_dir)

    # Apply preset profile defaults (explicit CLI flags still take precedence).
    prof = str(args.profile or "fast").strip().lower()
    warmup = int(args.warmup)
    repeats = int(args.repeats)
    runs = int(args.runs)
    record_series = bool(args.record_series)
    if prof == "stable":
        if "--warmup" not in sys.argv:
            warmup = 3
        if "--repeats" not in sys.argv:
            repeats = 30
        if "--runs" not in sys.argv:
            runs = 3
    elif prof == "debug":
        record_series = True

    sampler_interval_sec = float(max(1.0, float(args.sampler_interval_ms)) / 1000.0)
    spike_mad_k = float(args.spike_mad_k)
    gpu_index = int(args.gpu_index)
    warn_triton_drift_mb = float(args.warn_triton_drift_mb)

    # ---------- Visual grid (source resolutions)
    short_sides = [int(x) for x in str(args.visual_short_sides).split(",") if x.strip()]
    visual_inputs = _visual_grid(short_sides)

    # ---------- Model branches to benchmark once per fixed-shape model (Triton)
    model_jobs: List[Dict[str, Any]] = [
        # CLIP image
        {
            "name": "clip_image_224",
            "spec": "clip_image_224_triton",
            "kind": "single_input",
            "input": {"dtype": "UINT8", "shape": [1, 224, 224, 3], "value": "random_int", "int_range": [0, 256]},
        },
        {
            "name": "clip_image_336",
            "spec": "clip_image_336_triton",
            "kind": "single_input",
            "input": {"dtype": "UINT8", "shape": [1, 336, 336, 3], "value": "random_int", "int_range": [0, 256]},
        },
        {
            "name": "clip_image_448",
            "spec": "clip_image_448_triton",
            "kind": "single_input",
            "input": {"dtype": "UINT8", "shape": [1, 448, 448, 3], "value": "random_int", "int_range": [0, 256]},
        },
        # CLIP text
        {
            "name": "clip_text",
            "spec": "clip_text_triton",
            "kind": "single_input",
            "input": {"dtype": "INT64", "shape": [1, 77], "value": "random_int", "int_range": [0, 49408]},
        },
        # Places365
        {
            "name": "places365_resnet50_224",
            "spec": "places365_resnet50_224_triton",
            "kind": "single_input",
            "input": {"dtype": "UINT8", "shape": [1, 224, 224, 3], "value": "random_int", "int_range": [0, 256]},
        },
        {
            "name": "places365_resnet50_336",
            "spec": "places365_resnet50_336_triton",
            "kind": "single_input",
            "input": {"dtype": "UINT8", "shape": [1, 336, 336, 3], "value": "random_int", "int_range": [0, 256]},
        },
        {
            "name": "places365_resnet50_448",
            "spec": "places365_resnet50_448_triton",
            "kind": "single_input",
            "input": {"dtype": "UINT8", "shape": [1, 448, 448, 3], "value": "random_int", "int_range": [0, 256]},
        },
        # MiDaS
        {
            "name": "midas_256",
            "spec": "midas_256_triton",
            "kind": "single_input",
            "input": {"dtype": "UINT8", "shape": [1, 256, 256, 3], "value": "random_int", "int_range": [0, 256]},
        },
        {
            "name": "midas_384",
            "spec": "midas_384_triton",
            "kind": "single_input",
            "input": {"dtype": "UINT8", "shape": [1, 384, 384, 3], "value": "random_int", "int_range": [0, 256]},
        },
        {
            "name": "midas_512",
            "spec": "midas_512_triton",
            "kind": "single_input",
            "input": {"dtype": "UINT8", "shape": [1, 512, 512, 3], "value": "random_int", "int_range": [0, 256]},
        },
        # RAFT
        {
            "name": "raft_256",
            "spec": "raft_256_triton",
            "kind": "two_inputs",
            "input0": {"dtype": "UINT8", "shape": [1, 256, 256, 3], "value": "random_int", "int_range": [0, 256]},
            "input1": {"dtype": "UINT8", "shape": [1, 256, 256, 3], "value": "random_int", "int_range": [0, 256]},
        },
        {
            "name": "raft_384",
            "spec": "raft_384_triton",
            "kind": "two_inputs",
            "input0": {"dtype": "UINT8", "shape": [1, 384, 384, 3], "value": "random_int", "int_range": [0, 256]},
            "input1": {"dtype": "UINT8", "shape": [1, 384, 384, 3], "value": "random_int", "int_range": [0, 256]},
        },
        {
            "name": "raft_512",
            "spec": "raft_512_triton",
            "kind": "two_inputs",
            "input0": {"dtype": "UINT8", "shape": [1, 512, 512, 3], "value": "random_int", "int_range": [0, 256]},
            "input1": {"dtype": "UINT8", "shape": [1, 512, 512, 3], "value": "random_int", "int_range": [0, 256]},
        },
        # YOLO
        {
            "name": "yolo11x_320",
            "spec": "yolo11x_320_triton",
            "kind": "single_input",
            "input": {"dtype": "FP32", "shape": [1, 3, 320, 320], "value": "random_uniform", "float_range": [0.0, 1.0]},
        },
        {
            "name": "yolo11x_640",
            "spec": "yolo11x_640_triton",
            "kind": "single_input",
            "input": {"dtype": "FP32", "shape": [1, 3, 640, 640], "value": "random_uniform", "float_range": [0.0, 1.0]},
        },
        {
            "name": "yolo11x_960",
            "spec": "yolo11x_960_triton",
            "kind": "single_input",
            "input": {"dtype": "FP32", "shape": [1, 3, 960, 960], "value": "random_uniform", "float_range": [0.0, 1.0]},
        },
    ]

    groups = {x.strip().lower() for x in str(args.model_groups or "all").split(",") if x.strip()}
    if "all" in groups:
        groups = {"clip_places", "midas_raft", "yolo"}

    only_models = {x.strip() for x in str(args.only_models or "").split(",") if x.strip()}

    def _job_group(job_name: str) -> str:
        n = str(job_name)
        if n.startswith("clip_") or n.startswith("places365_"):
            return "clip_places"
        if n.startswith("midas_") or n.startswith("raft_"):
            return "midas_raft"
        if n.startswith("yolo"):
            return "yolo"
        return "other"

    model_results: Dict[str, Any] = {}
    for job in model_jobs:
        if only_models and str(job.get("name")) not in only_models:
            continue
        if _job_group(str(job.get("name"))) not in groups:
            continue
        kind = str(job["kind"])
        if kind == "two_inputs":
            res = _measure_triton_model(
                name=str(job["name"]),
                model_spec_name=str(job["spec"]),
                kind="two_inputs",
                warmup=warmup,
                repeats=repeats,
                runs=runs,
                sampler_interval_sec=sampler_interval_sec,
                spike_mad_k=spike_mad_k,
                record_series=record_series,
                gpu_index=gpu_index,
                warn_triton_drift_mb=warn_triton_drift_mb,
                tensor_spec=dict(job["input0"]),
                tensor_spec1=dict(job["input1"]),
            )
        else:
            res = _measure_triton_model(
                name=str(job["name"]),
                model_spec_name=str(job["spec"]),
                kind="single_input",
                warmup=warmup,
                repeats=repeats,
                runs=runs,
                sampler_interval_sec=sampler_interval_sec,
                spike_mad_k=spike_mad_k,
                record_series=record_series,
                gpu_index=gpu_index,
                warn_triton_drift_mb=warn_triton_drift_mb,
                tensor_spec=dict(job["input"]),
            )
        res["error_kind"] = _classify_error(res.get("error"))
        model_results[str(job["name"])] = res

        # incremental save (so we don't lose partial results on OOM/crash)
        try:
            partial_path = os.path.join(out_dir, "checklist_micro_results.partial.json")
            with open(partial_path, "w", encoding="utf-8") as f:
                json.dump({"models_triton": model_results}, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

        if bool(args.stop_on_oom) and res.get("status") != "ok" and str(res.get("error_kind")) in ("oom", "oom_or_cuda"):
            break

    # ---------- Expand to per-source resolution routing table (Visual)
    def _pick_s(family: str, branch: str) -> Optional[int]:
        m = {
            "clip_image": {"small": 224, "medium": 336, "large": 448},
            "places365": {"small": 224, "medium": 336, "large": 448},
            "midas": {"small": 256, "medium": 384, "large": 512},
            "raft": {"small": 256, "medium": 384, "large": 512},
            "yolo": {"small": 320, "medium": 640, "large": 960},
        }
        return int(m[family][branch])

    visual_rows: List[Dict[str, Any]] = []
    for inp in visual_inputs:
        w = int(inp["width"])
        h = int(inp["height"])
        d = int(max(w, h))
        branch = _select_branch_s(d)
        row = dict(inp)
        row["max_dim"] = d
        row["routing_bucket"] = branch
        # selected model branches (fixed S)
        row["selected"] = {
            "clip_image": _pick_s("clip_image", branch),
            "places365": _pick_s("places365", branch),
            "midas": _pick_s("midas", branch),
            "raft": _pick_s("raft", branch),
            "yolo": _pick_s("yolo", branch),
        }
        visual_rows.append(row)

    # ---------- Audio grid + extractor micro measurement
    audio_results: Dict[str, Any] = {"skipped": bool(args.skip_audio)}
    if bool(args.skip_audio):
        # placeholder only; skip heavy deps / GPU contention
        audio_results["note"] = "skipped by --skip-audio"
        audio_results["by_extractor"] = {}
        audio_results["durations_sec"] = []
    else:
    # Use a single 20s WAV and vary segment end_sample.
        audio_durs = _linspace(float(args.audio_min_sec), float(args.audio_max_sec), int(args.audio_n))
        audio_sr = 22050
        audio_path = os.path.join(out_dir, "tmp_audio_20s.wav")
        try:
            import soundfile as sf  # type: ignore
            sf_ok = True
        except Exception:
            sf_ok = False

    # write WAV via stdlib if soundfile missing
        n_total = int(math.ceil(float(args.audio_max_sec) * audio_sr))
        rng = np.random.RandomState(0)
        wav = (0.02 * rng.randn(n_total)).astype(np.float32)  # quiet noise
        if sf_ok:
            sf.write(audio_path, wav, audio_sr)
        else:
            import wave

            with wave.open(audio_path, "wb") as wf:
                wf.setnchannels(1)
                wf.setsampwidth(2)
                wf.setframerate(audio_sr)
                # float32 -> int16
                ints = np.clip(wav, -1.0, 1.0)
                ints = (ints * 32767.0).astype(np.int16)
                wf.writeframes(ints.tobytes())

        segments_by_dur: Dict[str, List[dict]] = {}
        for dsec in audio_durs:
            n = int(max(1, round(float(dsec) * audio_sr)))
            segments_by_dur[f"{dsec:.3f}"] = [
                {
                    "index": 0,
                    "start_sec": 0.0,
                    "end_sec": float(dsec),
                    "center_sec": float(0.5 * float(dsec)),
                    "start_sample": 0,
                    "end_sample": n,
                }
            ]

        # Import extractors (AudioProcessor)
        audio_results = {"durations_sec": audio_durs, "by_extractor": {}}
        # Ensure AudioProcessor src is importable
        ap_root = os.path.abspath(os.path.join(_ROOT, "AudioProcessor"))
        if ap_root not in sys.path:
            sys.path.insert(0, ap_root)

        # Reduce noisy warnings/logging from extractors for very short segments.
        try:
            import logging

            lg = logging.getLogger("src.extractors.loudness_extractor")
            lg.setLevel(logging.CRITICAL)
            lg.propagate = False
            # CLAP extractor uses BaseExtractor logger name: "src.core.base_extractor.clap_extractor"
            logging.getLogger("src.core.base_extractor.clap_extractor").setLevel(logging.ERROR)
        except Exception:
            pass

        from src.extractors.loudness_extractor import LoudnessExtractor  # type: ignore
        from src.extractors.tempo_extractor import TempoExtractor  # type: ignore

        clap_dev = str(args.clap_device).lower()
        if clap_dev not in ("cpu", "cuda", "auto"):
            clap_dev = "cpu"

        # IMPORTANT:
        # laion_clap / torch sometimes try to touch CUDA even when we want CPU.
        # To enforce CPU-only CLAP on 6GB systems (and avoid VRAM contention with Triton),
        # hide CUDA devices BEFORE importing CLAPExtractor (which imports torch).
        if clap_dev == "cpu":
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

        from src.extractors.clap_extractor import CLAPExtractor  # type: ignore

        extractors = {
            "loudness_extractor": LoudnessExtractor(device="auto", sample_rate=audio_sr),
            "tempo_extractor": TempoExtractor(device="auto", sample_rate=audio_sr),
            # Default CPU to avoid Triton VRAM contention on 6GB GPUs.
            "clap_extractor": CLAPExtractor(device=clap_dev, sample_rate=48000),
        }

        def _measure_audio_extractor(extractor_name: str, extractor_obj: Any) -> Dict[str, Any]:
            rows = []
            # one-time warmup to ensure model is loaded (especially CLAP)
            try:
                _ = extractor_obj.run_segments(audio_path, out_dir, segments_by_dur[f"{audio_durs[0]:.3f}"])
            except Exception:
                pass
            for key, segs in segments_by_dur.items():
                lat_ms: List[float] = []
                status = "ok"
                err = None
                sampler = PeakSampler(
                    interval_sec=float(sampler_interval_sec),
                    gpu_index=int(gpu_index),
                    record_series=bool(record_series),
                    match_process_substrings=["tritonserver"],
                )
                sampler.start()
                try:
                    for _ in range(max(0, warmup)):
                        _ = extractor_obj.run_segments(audio_path, out_dir, segs)
                    for _ in range(max(1, repeats)):
                        t0 = time.perf_counter()
                        _ = extractor_obj.run_segments(audio_path, out_dir, segs)
                        lat_ms.append(float((time.perf_counter() - t0) * 1000.0))
                except Exception as e:
                    status = "error"
                    err = str(e)
                finally:
                    sampler.stop()
                mean_stable, spikes = _stable_mean_ms(lat_ms)
                spikes_mad = _detect_spikes_mad(lat_ms, k=float(spike_mad_k))
                stats = _stats_latency_ms(lat_ms)
                stats["spike_fraction"] = float(spikes_mad["spike_fraction"])
                stats["spike_rule"] = str(spikes_mad["rule"])
                rows.append(
                    {
                        "duration_sec": float(key),
                        "status": status,
                        "error": err,
                        "error_kind": _classify_error(err),
                        "latency_ms_samples": lat_ms,
                        "latency_stats": stats,
                        "latency_ms_mean_stable": mean_stable,
                        "spikes_mad": bool(spikes_mad["spikes"]),
                        "spike_fraction_mad": float(spikes_mad["spike_fraction"]),
                        "spikes_rule_mad": str(spikes_mad["rule"]),
                        "spikes_stable_filter": bool(spikes),
                        "spikes": bool(bool(spikes) or bool(spikes_mad["spikes"])),
                        "cpu_rss_peak_mb": sampler.rss_peak_mb,
                        "gpu_vram_peak_mb": sampler.gpu_peak_mb,
                        "series": sampler.series,
                    }
                )
            return {"extractor": extractor_name, "warmup": warmup, "repeats": repeats, "rows": rows}

        for ename, eobj in extractors.items():
            audio_results["by_extractor"][ename] = _measure_audio_extractor(ename, eobj)

    # ---------- Write outputs
    payload = {
        "created_at": datetime.utcnow().isoformat(),
        "repo": _git_info(_ROOT),
        "system": _system_info(),
        "protocol": {
            "mode": "micro",
            "profile": prof,
            "warmup": warmup,
            "repeats": repeats,
            "runs": runs,
            "sampler_interval_ms": float(args.sampler_interval_ms),
            "record_series": bool(record_series),
            "spike_mad_k": float(spike_mad_k),
            "gpu_index": int(gpu_index),
            "warn_triton_drift_mb": float(warn_triton_drift_mb),
        },
        "command": {"argv": sys.argv, "env": {"TRITON_HTTP_URL": os.environ.get("TRITON_HTTP_URL"), "DP_MODELS_ROOT": os.environ.get("DP_MODELS_ROOT")}},
        "visual": {"inputs": visual_rows, "note": "inputs are source WxH; model branches are fixed SxS (selected via routing)."},
        "models_triton": model_results,
        "audio": audio_results,
        "components": {
            "visual_core": ["core_clip", "core_depth_midas", "core_optical_flow", "core_object_detections", "core_face_landmarks"],
            "visual_modules": [
                "cut_detection",
                "optical_flow",
                "scene_classification",
                "shot_quality",
                "story_structure",
                "uniqueness",
                "video_pacing",
            ],
            "audio_components": ["clap_extractor", "loudness_extractor", "tempo_extractor"],
            "status": "mvp_models_and_audio_only",
        },
    }
    out_json = os.path.join(out_dir, "checklist_micro_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    # If we stopped early due to OOM, write explicit guidance.
    try:
        failed = [v for v in model_results.values() if v.get("status") != "ok" and v.get("error_kind") in ("oom", "oom_or_cuda")]
        if bool(args.stop_on_oom) and failed:
            p = os.path.join(out_dir, "NEXT_STEPS.md")
            with open(p, "w", encoding="utf-8") as f:
                f.write("## Triton restart required (OOM-like failure)\n\n")
                f.write("One or more Triton models failed with OOM-like errors. On 6GB GPUs this is expected.\n\n")
                f.write("Recommended approach: run groups one-by-one, restarting Triton between them:\n\n")
                f.write("- `--model-groups clip_places`\n")
                f.write("- restart Triton\n")
                f.write("- `--model-groups yolo`\n")
                f.write("- restart Triton\n")
                f.write("- `--model-groups midas_raft`\n")
    except Exception:
        pass

    # Minimal markdown summary
    out_md = os.path.join(out_dir, "SUMMARY.md")
    with open(out_md, "w", encoding="utf-8") as f:
        f.write(f"## Checklist micro summary ({payload['created_at']})\n\n")
        f.write(f"- out_dir: `{out_dir}`\n")
        f.write(f"- profile={prof} warmup={warmup}, repeats={repeats}, runs={runs}\n\n")
        f.write("### Triton models (stable mean ms)\n\n")
        for k, v in model_results.items():
            mean = v.get("latency_ms_mean_stable")
            try:
                mean_s = f"{float(mean):.3f}"
            except Exception:
                mean_s = "nan"
            try:
                delta_s = f"{float(v.get('vram_triton_delta_run_mb')):.1f}"
            except Exception:
                delta_s = "?"
            try:
                p95_s = f"{float((v.get('latency_stats') or {}).get('p95')):.3f}"
            except Exception:
                p95_s = "?"
            try:
                p99_s = f"{float((v.get('latency_stats') or {}).get('p99')):.3f}"
            except Exception:
                p99_s = "?"
            try:
                drift_s = f"{float(v.get('vram_triton_drift_mb')):.1f}"
            except Exception:
                drift_s = "?"
            rr = bool(v.get("restart_recommended"))
            f.write(
                f"- `{k}`: status={v.get('status')} mean_stable_ms={mean_s} spikes={v.get('spikes')} "
                f"| p95={p95_s} p99={p99_s} | vram_delta_run(max)={delta_s}MB drift={drift_s}MB restart={str(rr).lower()}\n"
            )
        f.write("\n### Audio extractors (per duration)\n\n")
        for ename, block in audio_results["by_extractor"].items():
            # show a few points only
            rows = block["rows"]
            ok = sum(1 for r in rows if r.get("status") == "ok")
            f.write(f"- `{ename}`: ok={ok}/{len(rows)}\n")

    print(out_json)
    print(out_md)

    # Optional: append row(s) into the canonical results doc.
    if str(args.results_md).strip():
        fmt = str(args.source_format)
        # pick first matching source row (we usually run a single S in micro)
        src = None
        for rr in visual_rows:
            if rr.get("format") == fmt:
                src = rr
                break
        rows_out: List[Dict[str, Any]] = []
        for model_name, r in model_results.items():
            if r.get("status") is None:
                continue
            # Basic mapping for readability
            comp = "model_only"
            if model_name.startswith("clip_image") or model_name == "clip_text":
                comp = "core_clip"
            elif model_name.startswith("places365_resnet50"):
                comp = "scene_classification"
            elif model_name.startswith("midas_"):
                comp = "core_depth_midas"
            elif model_name.startswith("raft_"):
                comp = "core_optical_flow"
            elif model_name.startswith("yolo"):
                comp = "core_object_detections"

            delta = r.get("vram_triton_delta_run_mb")

            rows_out.append(
                {
                    "component_or_module": comp,
                    "model_branch": model_name,
                    "source_resolution": (f"{src['width']}×{src['height']} ({fmt}, S={src['short_side']})" if src else ""),
                    "selected_branch": (src["selected"].get("clip_image") if src and comp == "core_clip" and model_name.startswith("clip_image") else
                                       src["selected"].get("places365") if src and comp == "scene_classification" else
                                       src["selected"].get("midas") if src and comp == "core_depth_midas" else
                                       src["selected"].get("raft") if src and comp == "core_optical_flow" else
                                       src["selected"].get("yolo") if src and comp == "core_object_detections" else ""),
                    "latency_ms_mean_stable": (f"{float(r.get('latency_ms_mean_stable')):.3f}" if r.get("latency_ms_mean_stable") == r.get("latency_ms_mean_stable") else ""),
                    "spikes": str(bool(r.get("spikes"))).lower(),
                    "rss_peak_mb": (f"{float(r.get('cpu_rss_peak_mb')):.3f}" if r.get("cpu_rss_peak_mb") is not None else ""),
                    "vram_triton_delta_run_mb": delta,
                    "status": str(r.get("status")),
                    "notes": (
                        "restart recommended (oom)"
                        if r.get("error_kind") in ("oom", "oom_or_cuda")
                        else ("restart recommended (vram drift)" if bool(r.get("restart_recommended")) else "")
                    ),
                    "out_dir": f"`{out_dir}`",
                }
            )
        _append_results_md(md_path=str(args.results_md), rows=rows_out)


if __name__ == "__main__":
    main()


