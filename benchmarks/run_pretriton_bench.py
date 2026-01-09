from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np


def _utc_ts() -> str:
    return datetime.utcnow().strftime("%Y%m%d-%H%M%S")


def _load_yaml(path: str) -> dict:
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise RuntimeError(f"PyYAML is required for benchmark specs: {e}") from e
    with open(path, "r", encoding="utf-8") as f:
        d = yaml.safe_load(f)
    return d if isinstance(d, dict) else {}


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


def _make_uint8_nhwc(shape: List[int]) -> np.ndarray:
    st = tuple(int(x) for x in shape)
    return np.random.randint(0, 256, size=st, dtype=np.uint8)


def _torch_device(dev: str) -> str:
    d = str(dev or "cpu").strip().lower()
    if d in ("cuda", "gpu"):
        return "cuda"
    return "cpu"


def _torch_dtype(name: str):
    import torch  # type: ignore

    s = str(name or "fp16").strip().lower()
    if s in ("fp32", "float32"):
        return torch.float32
    return torch.float16


def _torchhub_find_cached_repo_dir(*, torch_home: Optional[str], repo_slug: str) -> Optional[str]:
    """
    torch.hub cache layout typically:
      $TORCH_HOME/hub/<owner>_<repo>_<ref>/
    For intel-isl/MiDaS it is often:
      intel-isl_MiDaS_master/
    We must NOT hit network in offline mode, so we resolve a local repo dir if present.
    """
    if not torch_home:
        return None
    hub_dir = os.path.join(str(torch_home), "hub")
    if not os.path.isdir(hub_dir):
        return None
    # repo_slug like "intel-isl/MiDaS"
    owner_repo = repo_slug.replace("/", "_")
    # Prefer exact known pattern, then fall back to prefix search.
    exact = os.path.join(hub_dir, f"{owner_repo}_master")
    if os.path.isdir(exact):
        return exact
    # Fallback: find any cached dir starting with "<owner>_<repo>_"
    try:
        for name in os.listdir(hub_dir):
            if name.startswith(f"{owner_repo}_"):
                cand = os.path.join(hub_dir, name)
                if os.path.isdir(cand):
                    return cand
    except Exception:
        return None
    return None


def _torchhub_offline_redirect_enable() -> None:
    """
    Monkeypatch torch.hub.load so that any GitHub-style repo slug (owner/repo) is
    transparently redirected to a local cached repo dir under TORCH_HOME/hub, using source='local'.
    This is required for strict offline mode because some upstream hubconfs (e.g. MiDaS)
    call torch.hub.load() internally for backbones (e.g. rwightman/gen-efficientnet-pytorch).
    """
    import torch  # type: ignore

    if getattr(torch.hub, "_dp_offline_redirect_enabled", False):
        return

    orig_load = torch.hub.load

    def patched_load(repo_or_dir, model, *args, **kwargs):  # type: ignore[no-untyped-def]
        try:
            # If caller gave a local dir, ensure source='local' for torch>=2.8
            if isinstance(repo_or_dir, str) and os.path.isdir(repo_or_dir):
                kwargs.setdefault("source", "local")
                return orig_load(repo_or_dir, model, *args, **kwargs)

            if isinstance(repo_or_dir, str) and "/" in repo_or_dir and not repo_or_dir.startswith("/"):
                slug = repo_or_dir.split(":", 1)[0]
                local = _torchhub_find_cached_repo_dir(torch_home=os.environ.get("TORCH_HOME"), repo_slug=slug)
                if local:
                    kwargs.setdefault("source", "local")
                    return orig_load(local, model, *args, **kwargs)
        except Exception:
            # Fall back to original behavior (will be blocked by network_guard if it tries network).
            pass
        return orig_load(repo_or_dir, model, *args, **kwargs)

    torch.hub.load = patched_load  # type: ignore[assignment]
    setattr(torch.hub, "_dp_offline_redirect_enabled", True)


def _prep_nhwc_uint8_to_nchw_float(x_u8: np.ndarray, *, device: str, dtype) -> Any:
    """
    Preprocess policy we plan to put into Triton graph:
    UINT8 NHWC RGB -> float -> NCHW -> normalize.
    For MiDaS/RAFT we use simple [0..1] scaling; exact normalize can be aligned later.
    """
    import torch  # type: ignore

    x = torch.from_numpy(np.asarray(x_u8)).to(device)
    x = x.to(torch.float32) / 255.0
    # NHWC -> NCHW
    x = x.permute(0, 3, 1, 2).contiguous()
    return x.to(dtype)


def _load_midas(*, model_name: str, device: str, dtype):
    import torch  # type: ignore

    # Use cached repo + checkpoints (torch.hub uses local cache if present).
    repo_slug = "intel-isl/MiDaS"
    # In hard-offline mode, torch.hub may still try to hit GitHub to resolve refs.
    # Prefer loading from a local cached repo dir under TORCH_HOME/hub/...
    local_repo = _torchhub_find_cached_repo_dir(torch_home=os.environ.get("TORCH_HOME"), repo_slug=repo_slug)
    if local_repo:
        # torch>=2.8 requires explicit source='local' for absolute paths.
        model = torch.hub.load(local_repo, str(model_name), pretrained=True, trust_repo=True, verbose=False, source="local")
    else:
        model = torch.hub.load(repo_slug, str(model_name), pretrained=True, trust_repo=True, verbose=False)
    model.eval()
    # Ensure model weights dtype matches input dtype to avoid Half/Float mismatches on GPU.
    model.to(device=device, dtype=dtype)
    return model


def _infer_midas(*, model, x_u8: np.ndarray, device: str, dtype) -> Any:
    import torch  # type: ignore

    x = _prep_nhwc_uint8_to_nchw_float(x_u8, device=device, dtype=dtype)
    with torch.inference_mode():
        return model(x)


def _load_raft(*, model_name: str, device: str, dtype):
    import torch  # type: ignore
    import torchvision.models.optical_flow as models  # type: ignore

    name = str(model_name).strip().lower()
    if name == "raft_large":
        m = models.raft_large(weights=models.Raft_Large_Weights.DEFAULT, progress=False)
    else:
        m = models.raft_small(weights=models.Raft_Small_Weights.DEFAULT, progress=False)
    m.eval()
    m.to(device=device, dtype=dtype)
    return m


def _infer_raft(*, model, x0_u8: np.ndarray, x1_u8: np.ndarray, device: str, dtype) -> Any:
    import torch  # type: ignore

    x0 = _prep_nhwc_uint8_to_nchw_float(x0_u8, device=device, dtype=dtype)
    x1 = _prep_nhwc_uint8_to_nchw_float(x1_u8, device=device, dtype=dtype)
    with torch.inference_mode():
        # torchvision RAFT expects list/tuple of two tensors [B,3,H,W]
        out = model(x0, x1)
    return out


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


def _write_json(path: str, obj: Any) -> None:
    # Best-effort atomic-ish write (write tmp then replace).
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)


def main() -> None:
    ap = argparse.ArgumentParser("pre-triton benchmark (torch/torchvision via local caches)")
    ap.add_argument("--spec", type=str, required=True)
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--filter", type=str, default=None)
    ap.add_argument("--warmup", type=int, default=None)
    ap.add_argument("--repeats", type=int, default=None)
    ap.add_argument("--device", type=str, default=None)
    ap.add_argument("--dtype", type=str, default=None)
    ap.add_argument("--batch-mults", type=str, default="1,2,4,8", help="Batch multipliers to try")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--models-root", type=str, default=None, help="DP_MODELS_ROOT-style directory to pin caches into")
    ap.add_argument("--offline", action="store_true", help="Strict no-network guard (requires caches to be populated)")
    args = ap.parse_args()

    spec = _load_yaml(str(args.spec))
    defaults = spec.get("defaults") if isinstance(spec.get("defaults"), dict) else {}

    warmup = int(args.warmup if args.warmup is not None else (defaults.get("warmup") or 5))
    repeats = int(args.repeats if args.repeats is not None else (defaults.get("repeats") or 30))
    device = _torch_device(args.device if args.device is not None else (defaults.get("device") or "cpu"))

    import torch  # type: ignore

    # Optional: pin caches under models_root.
    if args.models_root:
        try:
            from dp_models.offline import pin_cache_env, network_guard  # type: ignore

            pin_cache_env(str(args.models_root), offline=bool(args.offline))
        except Exception as e:
            raise RuntimeError(f"Failed to pin cache env under models_root={args.models_root}: {e}") from e
    net_guard_ctx = None
    if bool(args.offline):
        try:
            from dp_models.offline import network_guard  # type: ignore

            net_guard_ctx = network_guard(enabled=True)
            # Also patch torch.hub to redirect internal hub loads to local cached repos.
            _torchhub_offline_redirect_enable()
        except Exception as e:
            raise RuntimeError(f"Failed to enable network_guard: {e}") from e

    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false")
    dtype = _torch_dtype(args.dtype if args.dtype is not None else (defaults.get("dtype") or "fp16"))

    out_dir = args.out_dir or os.path.join("benchmarks", "out", f"pretriton-{_utc_ts()}")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    results_path = os.path.join(out_dir, "results.jsonl")
    summary_path = os.path.join(out_dir, "summary.json")

    batch_mults = []
    for x in str(args.batch_mults).split(","):
        x = x.strip()
        if x:
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
        print(json.dumps({"out_dir": out_dir, "device": device, "dtype": str(dtype), "models": [p.get("name") for p in plan]}, indent=2))
        return

    f_out = open(results_path, "w", encoding="utf-8")
    summary: Dict[str, Any] = {
        "created_at": datetime.utcnow().isoformat(),
        "out_dir": out_dir,
        "spec": str(args.spec),
        "device": device,
        "dtype_requested": str(dtype),
        "warmup": int(warmup),
        "repeats": int(repeats),
        "batch_mults": list(batch_mults),
        "filter": (str(args.filter) if args.filter else None),
        "items": [],
    }

    try:
        if net_guard_ctx is not None:
            net_guard_ctx.__enter__()
        for m in plan:
            name = str(m.get("name"))
            kind = str(m.get("kind"))
            model_name = str(m.get("model_name") or "")

            # Load model ONCE per variant (critical for meaningful latency + stability).
            model_obj = None
            raft_dtype_used = None
            if kind == "midas":
                model_obj = _load_midas(model_name=model_name, device=device, dtype=dtype)
            elif kind == "raft":
                # RAFT fp16 can fail in some ops (e.g. grid_sample expects input/grid dtypes to match).
                # For benchmarking we force fp32 compute when fp16 is requested.
                dtype_used = dtype
                if str(dtype) == "torch.float16":
                    dtype_used = torch.float32
                raft_dtype_used = dtype_used
                model_obj = _load_raft(model_name=model_name, device=device, dtype=dtype_used)
            else:
                raise RuntimeError(f"Unknown kind: {kind}")

            for bm in batch_mults:
                if kind == "midas":
                    in_spec = m.get("input") or {}
                    shape = list(in_spec.get("shape") or [])
                    if not shape:
                        raise RuntimeError(f"{name} | missing input.shape")
                    shape[0] = int(bm)
                    x = _make_uint8_nhwc(shape)

                    try:
                    # warmup
                    for _ in range(max(0, warmup)):
                            _ = _infer_midas(model=model_obj, x_u8=x, device=device, dtype=dtype)
                        if device == "cuda":
                            torch.cuda.synchronize()

                    samples_ms: List[float] = []
                    for _ in range(max(1, repeats)):
                        t0 = time.perf_counter()
                            _ = _infer_midas(model=model_obj, x_u8=x, device=device, dtype=dtype)
                        if device == "cuda":
                            torch.cuda.synchronize()
                        dt_ms = (time.perf_counter() - t0) * 1000.0
                        samples_ms.append(float(dt_ms))
                    except RuntimeError as e:
                        # Graceful CUDA OOM handling: record and stop increasing batch for this model.
                        msg = str(e).lower()
                        if "out of memory" in msg and "cuda" in msg:
                            if device == "cuda":
                                try:
                                    torch.cuda.empty_cache()
                                except Exception:
                                    pass
                            summary["items"].append({"variant": name, "batch": int(bm), "error": "cuda_oom"})
                            try:
                                _write_json(summary_path, summary)
                            except Exception:
                                pass
                            break
                        raise

                    summary["items"].append({"variant": name, "batch": int(bm), "quantiles_ms": _quantiles_ms(samples_ms)})
                    # Persist partial summary so long runs still produce usable artifacts if interrupted.
                    try:
                        _write_json(summary_path, summary)
                    except Exception:
                        pass

                    for dt_ms in samples_ms:
                        f_out.write(
                            json.dumps(
                                {
                                    "ts": datetime.utcnow().isoformat(),
                                    "bench": "pretriton",
                                    "variant": name,
                                    "kind": kind,
                                    "model_name": model_name,
                                    "batch": int(bm),
                                    "latency_ms": float(dt_ms),
                                    "device": device,
                                    "dtype_requested": str(dtype),
                                    "dtype_used": str(dtype),
                                    "rss_mb": _rss_mb(),
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        f_out.flush()

                elif kind == "raft":
                    in0 = m.get("input0") or {}
                    in1 = m.get("input1") or {}
                    shape0 = list(in0.get("shape") or [])
                    shape1 = list(in1.get("shape") or [])
                    if not shape0 or not shape1:
                        raise RuntimeError(f"{name} | missing input0/input1 shapes")
                    shape0[0] = int(bm)
                    shape1[0] = int(bm)
                    x0 = _make_uint8_nhwc(shape0)
                    x1 = _make_uint8_nhwc(shape1)

                    dtype_used = raft_dtype_used or dtype

                    try:
                    for _ in range(max(0, warmup)):
                            _ = _infer_raft(model=model_obj, x0_u8=x0, x1_u8=x1, device=device, dtype=dtype_used)
                        if device == "cuda":
                            torch.cuda.synchronize()

                    samples_ms = []
                    for _ in range(max(1, repeats)):
                        t0 = time.perf_counter()
                            _ = _infer_raft(model=model_obj, x0_u8=x0, x1_u8=x1, device=device, dtype=dtype_used)
                        if device == "cuda":
                            torch.cuda.synchronize()
                        dt_ms = (time.perf_counter() - t0) * 1000.0
                        samples_ms.append(float(dt_ms))
                    except RuntimeError as e:
                        msg = str(e).lower()
                        if "out of memory" in msg and "cuda" in msg:
                            if device == "cuda":
                                try:
                                    torch.cuda.empty_cache()
                                except Exception:
                                    pass
                            summary["items"].append({"variant": name, "batch": int(bm), "error": "cuda_oom"})
                            try:
                                _write_json(summary_path, summary)
                            except Exception:
                                pass
                            break
                        raise

                    summary["items"].append({"variant": name, "batch": int(bm), "quantiles_ms": _quantiles_ms(samples_ms)})
                    # Persist partial summary so long runs still produce usable artifacts if interrupted.
                    try:
                        _write_json(summary_path, summary)
                    except Exception:
                        pass

                    for dt_ms in samples_ms:
                        f_out.write(
                            json.dumps(
                                {
                                    "ts": datetime.utcnow().isoformat(),
                                    "bench": "pretriton",
                                    "variant": name,
                                    "kind": kind,
                                    "model_name": model_name,
                                    "batch": int(bm),
                                    "latency_ms": float(dt_ms),
                                    "device": device,
                                    "dtype_requested": str(dtype),
                                    "dtype_used": str(dtype_used),
                                    "rss_mb": _rss_mb(),
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        f_out.flush()

        _write_json(summary_path, summary)
        print(f"[pretriton] wrote: {results_path}")
        print(f"[pretriton] wrote: {summary_path}")
    finally:
        # Always try to flush a partial summary (useful on Ctrl+C / exceptions).
        try:
            if summary.get("items"):
                _write_json(summary_path, summary)
        except Exception:
            pass
        if net_guard_ctx is not None:
            try:
                net_guard_ctx.__exit__(None, None, None)
            except Exception:
                pass
        try:
            f_out.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()


