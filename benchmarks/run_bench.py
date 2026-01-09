from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def _utc_ts() -> str:
    return datetime.utcnow().strftime("%Y%m%d-%H%M%S")


def _maybe_import_psutil():
    try:
        import psutil  # type: ignore
    except Exception:
        return None
    return psutil


def _rss_mb() -> Optional[float]:
    psutil = _maybe_import_psutil()
    if psutil is None:
        return None
    try:
        p = psutil.Process(os.getpid())
        return float(p.memory_info().rss) / (1024.0 * 1024.0)
    except Exception:
        return None


def _maybe_import_nvml():
    # Prefer pynvml if installed. Optional dependency.
    try:
        import pynvml  # type: ignore
    except Exception:
        return None
    return pynvml


def _gpu_mem_mb() -> Optional[Dict[str, float]]:
    pynvml = _maybe_import_nvml()
    if pynvml is None:
        return None
    try:
        pynvml.nvmlInit()
        n = pynvml.nvmlDeviceGetCount()
        out: Dict[str, float] = {}
        for i in range(int(n)):
            h = pynvml.nvmlDeviceGetHandleByIndex(i)
            info = pynvml.nvmlDeviceGetMemoryInfo(h)
            out[f"gpu{i}_used_mb"] = float(info.used) / (1024.0 * 1024.0)
            out[f"gpu{i}_total_mb"] = float(info.total) / (1024.0 * 1024.0)
        return out
    except Exception:
        return None
    finally:
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def _load_yaml(path: str) -> dict:
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError(f"PyYAML is required for benchmark specs: {e}") from e
    with open(path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)
    return d if isinstance(d, dict) else {}


def _dtype_from_triton(dt: str):
    t = str(dt or "").upper()
    if t in ("FP16",):
        return np.float16
    if t in ("FP32", "FP64"):
        return np.float32
    if t in ("INT64",):
        return np.int64
    if t in ("INT32",):
        return np.int32
    if t in ("UINT8",):
        return np.uint8
    # default
    return np.float32


def _make_tensor(spec: dict) -> np.ndarray:
    dt = str(spec.get("dtype") or "FP32").upper()
    shape = spec.get("shape")
    if not isinstance(shape, list) or not shape:
        raise ValueError(f"invalid shape in spec: {shape}")
    shape_t = tuple(int(x) for x in shape)
    kind = str(spec.get("value") or "random_normal").strip().lower()
    np_dtype = _dtype_from_triton(dt)

    if kind == "random_int":
        lo, hi = 0, 100
        r = spec.get("int_range")
        if isinstance(r, list) and len(r) == 2:
            lo, hi = int(r[0]), int(r[1])
        x = np.random.randint(lo, hi, size=shape_t, dtype=np.int64)
        return x.astype(np_dtype, copy=False)

    if kind == "random_uniform":
        lo, hi = 0.0, 1.0
        r = spec.get("float_range")
        if isinstance(r, list) and len(r) == 2:
            lo, hi = float(r[0]), float(r[1])
        x = (lo + (hi - lo) * np.random.rand(*shape_t)).astype(np.float32)
        return x.astype(np_dtype, copy=False)

    # default: normal
    x = np.random.randn(*shape_t).astype(np.float32)
    return x.astype(np_dtype, copy=False)


def _is_fixed_batch_shape(shape: List[int]) -> bool:
    """
    Heuristic: treat batch dim as fixed if shape[0] is a positive int (typically 1)
    and spec does not explicitly allow overriding batch.
    """
    if not isinstance(shape, list) or not shape:
        return False
    try:
        b0 = int(shape[0])
    except Exception:
        return False
    return b0 > 0


def _quantiles_ms(samples_ms: List[float]) -> Dict[str, float]:
    a = np.asarray(samples_ms, dtype=np.float32)
    if a.size == 0:
        return {"p50": float("nan"), "p90": float("nan"), "p99": float("nan"), "mean": float("nan")}
    return {
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "p99": float(np.percentile(a, 99)),
        "mean": float(np.mean(a)),
    }


def _resolve_model_via_mm(model_spec_name: str):
    from dp_models import get_global_model_manager  # type: ignore

    mm = get_global_model_manager()
    rm = mm.get(model_name=str(model_spec_name))
    handle = rm.handle or {}
    rp = rm.spec.runtime_params or {}
    if not isinstance(handle, dict) or "client" not in handle:
        raise RuntimeError(f"ModelManager returned empty Triton client handle for: {model_spec_name}")
    if not isinstance(rp, dict) or not rp:
        raise RuntimeError(f"ModelManager returned empty runtime_params for: {model_spec_name}")
    return rm, handle["client"], rp


def main() -> None:
    ap = argparse.ArgumentParser("benchmark harness (Triton via dp_models)")
    ap.add_argument("--spec", type=str, required=True, help="Path to benchmark spec YAML")
    ap.add_argument("--out-dir", type=str, default=None, help="Output directory (default: benchmarks/out/<timestamp>)")
    ap.add_argument("--filter", type=str, default=None, help="Run only models whose name contains this substring")
    ap.add_argument("--warmup", type=int, default=None, help="Warmup iterations (override spec defaults)")
    ap.add_argument("--repeats", type=int, default=None, help="Repeat iterations (override spec defaults)")
    ap.add_argument("--batch-mults", type=str, default="1,2,4,8,16", help="Batch multipliers to try (comma-separated)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--triton-http-url", type=str, default=None, help="Override TRITON_HTTP_URL env var for dp_models specs")
    args = ap.parse_args()

    if args.triton_http_url:
        os.environ["TRITON_HTTP_URL"] = str(args.triton_http_url)

    spec = _load_yaml(str(args.spec))
    defaults = spec.get("defaults") if isinstance(spec.get("defaults"), dict) else {}
    warmup = int(args.warmup if args.warmup is not None else (defaults.get("warmup") or 5))
    repeats = int(args.repeats if args.repeats is not None else (defaults.get("repeats") or 30))

    out_dir = args.out_dir or os.path.join("benchmarks", "out", _utc_ts())
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    results_path = os.path.join(out_dir, "results.jsonl")
    summary_path = os.path.join(out_dir, "summary.json")

    batch_mults = []
    for x in str(args.batch_mults).split(","):
        x = x.strip()
        if not x:
            continue
        batch_mults.append(int(x))
    if not batch_mults:
        batch_mults = [1]

    models = spec.get("models")
    if not isinstance(models, list) or not models:
        raise RuntimeError("bench spec has no models[]")

    plan: List[dict] = []
    for m in models:
        if not isinstance(m, dict):
            continue
        name = str(m.get("name") or "")
        if not name:
            continue
        if args.filter and str(args.filter) not in name:
            continue
        plan.append(m)

    if args.dry_run:
        print(json.dumps({"out_dir": out_dir, "warmup": warmup, "repeats": repeats, "models": [p.get("name") for p in plan]}, indent=2))
        return

    # results.jsonl
    f_out = open(results_path, "w", encoding="utf-8")
    summaries: Dict[str, Any] = {"created_at": datetime.utcnow().isoformat(), "out_dir": out_dir, "items": []}

    try:
        for m in plan:
            name = str(m.get("name"))
            model_spec_name = str(m.get("model_spec_name"))
            kind = str(m.get("kind") or "single_input")

            rm, client, rp = _resolve_model_via_mm(model_spec_name)

            triton_model_name = str(rp.get("triton_model_name") or "")
            triton_model_version = rp.get("triton_model_version")
            triton_model_version = str(triton_model_version) if triton_model_version not in (None, "") else None

            # names (fallback to common defaults)
            input_name = str(rp.get("triton_input_name") or "INPUT__0")
            output_name = str(rp.get("triton_output_name") or "OUTPUT__0")
            input_dt = str(rp.get("triton_input_datatype") or "FP32")

            input0_name = str(rp.get("triton_input0_name") or "INPUT0__0")
            input1_name = str(rp.get("triton_input1_name") or "INPUT1__0")

            base_rss = _rss_mb()
            base_gpu = _gpu_mem_mb()

            for bm in batch_mults:
                # Fixed-shape baseline models are exported with batch=1 (no dynamic axes).
                # By default we DO NOT override batch dim for such specs, because Triton will reject it.
                allow_batch_override = bool(m.get("allow_batch_override") or False)
                # build tensors (override batch dim)
                if kind == "two_inputs":
                    in0_spec = dict(m.get("input0") or {})
                    in1_spec = dict(m.get("input1") or {})
                    base0 = in0_spec.get("shape")
                    base1 = in1_spec.get("shape")
                    if (
                        not allow_batch_override
                        and isinstance(base0, list)
                        and isinstance(base1, list)
                        and _is_fixed_batch_shape(base0)
                        and _is_fixed_batch_shape(base1)
                        and int(bm) != int(base0[0])
                    ):
                        # skip silently (but keep run stable); fixed batch mismatch
                        continue
                    if "shape" in in0_spec and isinstance(in0_spec["shape"], list) and in0_spec["shape"]:
                        in0_spec["shape"] = [int(bm)] + [int(x) for x in in0_spec["shape"][1:]]
                    if "shape" in in1_spec and isinstance(in1_spec["shape"], list) and in1_spec["shape"]:
                        in1_spec["shape"] = [int(bm)] + [int(x) for x in in1_spec["shape"][1:]]
                    x0 = _make_tensor(in0_spec)
                    x1 = _make_tensor(in1_spec)
                else:
                    in_spec = dict(m.get("input") or {})
                    base = in_spec.get("shape")
                    if (
                        not allow_batch_override
                        and isinstance(base, list)
                        and _is_fixed_batch_shape(base)
                        and int(bm) != int(base[0])
                    ):
                        continue
                    if "shape" in in_spec and isinstance(in_spec["shape"], list) and in_spec["shape"]:
                        in_spec["shape"] = [int(bm)] + [int(x) for x in in_spec["shape"][1:]]
                    x = _make_tensor(in_spec)

                # Warmup
                failed = None
                for _ in range(max(0, warmup)):
                    try:
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
                                input_tensor=x,
                                output_name=output_name,
                                datatype=input_dt,
                            )
                    except Exception as e:
                        failed = f"warmup_failed: {e}"
                        break

                if failed is not None:
                    rec = {
                        "ts": datetime.utcnow().isoformat(),
                        "bench": "model_level_triton",
                        "variant": name,
                        "model_spec_name": model_spec_name,
                        "triton_http_url": getattr(client, "base_url", None),
                        "triton_model_name": triton_model_name,
                        "triton_model_version": triton_model_version,
                        "kind": kind,
                        "batch": int(bm),
                        "status": "error",
                        "error": failed,
                        "rss_mb": _rss_mb(),
                        "gpu_mem_mb": _gpu_mem_mb(),
                        "models_used": rm.models_used_entry,
                        "base_rss_mb": base_rss,
                        "base_gpu_mem_mb": base_gpu,
                    }
                    f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f_out.flush()
                    summaries["items"].append(
                        {
                            "variant": name,
                            "batch": int(bm),
                            "status": "error",
                            "error": failed,
                            "samples": 0,
                            "model_spec_name": model_spec_name,
                            "triton_model_name": triton_model_name,
                            "kind": kind,
                        }
                    )
                    continue

                samples_ms: List[float] = []
                out_shape: Optional[List[int]] = None
                for i in range(max(1, repeats)):
                    t0 = time.perf_counter()
                    try:
                        if kind == "two_inputs":
                            res = client.infer_two_inputs(
                                model_name=triton_model_name,
                                model_version=triton_model_version,
                                input0_name=input0_name,
                                input0_tensor=x0,
                                input1_name=input1_name,
                                input1_tensor=x1,
                                output_name=output_name,
                                datatype=input_dt,
                            )
                            out = np.asarray(res.output)
                        else:
                            res = client.infer(
                                model_name=triton_model_name,
                                model_version=triton_model_version,
                                input_name=input_name,
                                input_tensor=x,
                                output_name=output_name,
                                datatype=input_dt,
                            )
                            out = np.asarray(res.output)
                    except Exception as e:
                        failed = f"infer_failed: {e}"
                        break
                    dt_ms = (time.perf_counter() - t0) * 1000.0
                    samples_ms.append(float(dt_ms))
                    if out_shape is None:
                        out_shape = list(out.shape)

                    rec = {
                        "ts": datetime.utcnow().isoformat(),
                        "bench": "model_level_triton",
                        "variant": name,
                        "model_spec_name": model_spec_name,
                        "triton_http_url": getattr(client, "base_url", None),
                        "triton_model_name": triton_model_name,
                        "triton_model_version": triton_model_version,
                        "kind": kind,
                        "batch": int(bm),
                        "status": "ok",
                        "latency_ms": float(dt_ms),
                        "output_shape": list(out.shape),
                        "rss_mb": _rss_mb(),
                        "gpu_mem_mb": _gpu_mem_mb(),
                        "models_used": rm.models_used_entry,
                        "base_rss_mb": base_rss,
                        "base_gpu_mem_mb": base_gpu,
                    }
                    f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f_out.flush()

                if failed is not None:
                    rec = {
                        "ts": datetime.utcnow().isoformat(),
                        "bench": "model_level_triton",
                        "variant": name,
                        "model_spec_name": model_spec_name,
                        "triton_http_url": getattr(client, "base_url", None),
                        "triton_model_name": triton_model_name,
                        "triton_model_version": triton_model_version,
                        "kind": kind,
                        "batch": int(bm),
                        "status": "error",
                        "error": failed,
                        "output_shape": out_shape,
                        "rss_mb": _rss_mb(),
                        "gpu_mem_mb": _gpu_mem_mb(),
                        "models_used": rm.models_used_entry,
                        "base_rss_mb": base_rss,
                        "base_gpu_mem_mb": base_gpu,
                    }
                    f_out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    f_out.flush()
                    summaries["items"].append(
                        {
                            "variant": name,
                            "batch": int(bm),
                            "status": "error",
                            "error": failed,
                            "samples": len(samples_ms),
                            "model_spec_name": model_spec_name,
                            "triton_model_name": triton_model_name,
                            "kind": kind,
                        }
                    )
                else:
                    summaries["items"].append(
                        {
                            "variant": name,
                            "batch": int(bm),
                            "status": "ok",
                            "quantiles_ms": _quantiles_ms(samples_ms),
                            "samples": len(samples_ms),
                            "model_spec_name": model_spec_name,
                            "triton_model_name": triton_model_name,
                            "kind": kind,
                        }
                    )

        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summaries, f, ensure_ascii=False, indent=2)
        print(f"[bench] wrote: {results_path}")
        print(f"[bench] wrote: {summary_path}")
    finally:
        try:
            f_out.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()


