#!/usr/bin/env python3
"""
core_optical_flow

Tier-0 core provider: optical flow motion curve via Triton.

Contract:
- Segmenter provides `metadata["core_optical_flow"]["frame_indices"]` (union-domain indices).
- No sampling fallback is allowed.
- Frames from FrameManager.get(idx) are RGB uint8 (HxWx3).

Output:
- <rs_path>/core_optical_flow/flow.npz
  Keys:
    - frame_indices: int32 (N,)
    - motion_norm_per_sec_mean: float32 (N,)  # mean flow magnitude / dt / max(H,W); 0 for first frame
    - dt_seconds: float32 (N,)                # NaN for first frame
    - meta: object(dict)
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import numpy as np  # type: ignore

_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if _path not in sys.path:
    sys.path.append(_path)
# repo root (needed for dp_triton)
_root = os.path.dirname(_path)
if _root not in sys.path:
    sys.path.append(_root)

from utils.frame_manager import FrameManager
from utils.logger import get_logger
from utils.utilites import load_metadata
from utils.meta_builder import apply_models_meta, model_used

NAME = "core_optical_flow"
VERSION = "2.0"
SCHEMA_VERSION = "core_optical_flow_npz_v1"
LOGGER = get_logger(NAME)


def _load_triton_spec_via_model_manager(model_spec_name: str) -> dict:
    """
    Resolve Triton model spec via dp_models.ModelManager (no-network, reproducible).
    Returns dict with keys:
      - client: TritonHttpClient
      - rp: runtime_params
      - models_used_entry: dict (model_used)
    """
    from dp_models import get_global_model_manager  # type: ignore

    mm = get_global_model_manager()
    rm = mm.get(model_name=str(model_spec_name))
    rp = rm.spec.runtime_params or {}
    handle = rm.handle or {}
    client = None
    if isinstance(handle, dict):
        client = handle.get("client")
    if client is None:
        raise RuntimeError(f"{NAME} | ModelManager returned empty Triton client handle for: {model_spec_name}")
    if not isinstance(rp, dict) or not rp:
        raise RuntimeError(f"{NAME} | ModelManager returned empty runtime_params for: {model_spec_name}")
    return {"client": client, "rp": rp, "models_used_entry": rm.models_used_entry}


def _require_frame_indices(meta: dict, name: str) -> List[int]:
    block = meta.get(name)
    if not isinstance(block, dict) or "frame_indices" not in block:
        raise RuntimeError(
            f"{name} | metadata missing '{name}.frame_indices'. "
            "Segmenter must provide per-provider frame_indices. No fallback is allowed."
        )
    frame_indices = block.get("frame_indices")
    if not isinstance(frame_indices, list) or not frame_indices:
        raise RuntimeError(f"{name} | metadata '{name}.frame_indices' is empty/invalid.")
    return [int(x) for x in frame_indices]


def _get_union_timestamps_sec(frame_manager: FrameManager) -> Optional[np.ndarray]:
    meta = getattr(frame_manager, "meta", None)
    if not isinstance(meta, dict):
        return None
    ts = meta.get("union_timestamps_sec")
    if not isinstance(ts, list) or not ts:
        return None
    try:
        return np.asarray(ts, dtype=np.float32)
    except Exception:
        return None


def _preset_to_input_size(preset: str) -> int:
    p = str(preset or "").strip().lower()
    if p in ("raft_256", "256"):
        return 256
    if p in ("raft_384", "384"):
        return 384
    if p in ("raft_512", "512"):
        return 512
    raise ValueError(f"{NAME} | unknown triton_preprocess_preset: {preset!r}")


def _prep_batch_rgb_uint8(frames: List[np.ndarray], *, input_size: int) -> np.ndarray:
    """
    Minimal client-side formatting for Triton (NOT full preprocessing):
    - resize to (S,S)
    - keep UINT8 NHWC (baseline GPU contract)

    Full preprocessing (normalize/layout conversion to model FP32 NCHW) lives in Triton ensemble.
    """
    import cv2  # type: ignore

    s = int(input_size)
    if s <= 0:
        raise ValueError(f"{NAME} | invalid input_size={input_size}")
    out: List[np.ndarray] = []
    for fr in frames:
        fr_r = cv2.resize(fr, (s, s), interpolation=cv2.INTER_AREA)
        out.append(np.asarray(fr_r, dtype=np.uint8))
    if not out:
        return np.zeros((0, s, s, 3), dtype=np.uint8)
    return np.stack(out, axis=0).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser(description="core_optical_flow (Triton) motion curve extractor")
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--rs-path", required=True)
    # Triton-only policy (prod): local torch engine is removed.
    parser.add_argument("--runtime", type=str, default="triton", choices=["triton"], help="Runtime (prod: triton only)")
    parser.add_argument("--triton-http-url", type=str, default=None)
    # Preferred: resolve Triton params via ModelManager specs (recommended; overrides explicit triton_* args when provided).
    parser.add_argument("--triton-model-spec", type=str, default=None, help="dp_models spec name (e.g., raft_256_triton)")
    parser.add_argument("--triton-model-name", type=str, default=None)
    parser.add_argument("--triton-model-version", type=str, default=None)
    parser.add_argument("--triton-input0-name", type=str, default="INPUT0__0")
    parser.add_argument("--triton-input1-name", type=str, default="INPUT1__0")
    parser.add_argument("--triton-output-name", type=str, default="OUTPUT__0")
    # Triton ensemble expects UINT8 NHWC inputs.
    parser.add_argument("--triton-datatype", type=str, default="UINT8")
    parser.add_argument(
        "--triton-preprocess-preset",
        type=str,
        default="raft_256",
        choices=["raft_256", "raft_384", "raft_512"],
        help="Input preset (square size) for Triton optical-flow model.",
    )
    parser.add_argument("--model-version", type=str, default="unknown")
    parser.add_argument("--weights-digest", type=str, default="unknown")
    parser.add_argument("--precision", type=str, default="fp32")
    args = parser.parse_args()

    runtime = str(args.runtime or "triton").strip().lower()
    if runtime != "triton":
        raise RuntimeError(f"{NAME} | runtime must be triton (no-fallback), got: {runtime}")

    meta = load_metadata(os.path.join(args.frames_dir, "metadata.json"), NAME)
    total_frames = int(meta.get("total_frames", 0))
    frame_indices = _require_frame_indices(meta, NAME)
    LOGGER.info(f"{NAME} | sampled frames: {len(frame_indices)} / total={total_frames}")

    if len(frame_indices) < 2:
        raise RuntimeError(f"{NAME} | frame_indices must contain at least 2 frames (no-fallback)")

    # Triton client (repo-local)
    from dp_triton import TritonHttpClient, TritonError  # type: ignore

    mm_entry = None
    if args.triton_model_spec:
        mm_entry = _load_triton_spec_via_model_manager(str(args.triton_model_spec))
        client = mm_entry["client"]
        rp = mm_entry["rp"]
        args.triton_http_url = str(rp.get("triton_http_url") or args.triton_http_url or "")
        args.triton_model_name = str(rp.get("triton_model_name") or args.triton_model_name or "")
        args.triton_model_version = str(rp.get("triton_model_version") or "") or None
        args.triton_input0_name = str(rp.get("triton_input0_name") or args.triton_input0_name)
        args.triton_input1_name = str(rp.get("triton_input1_name") or args.triton_input1_name)
        args.triton_output_name = str(rp.get("triton_output_name") or args.triton_output_name)
        args.triton_datatype = str(rp.get("triton_input_datatype") or args.triton_datatype)
    else:
        if not args.triton_http_url or not str(args.triton_http_url).strip():
            raise RuntimeError(f"{NAME} | runtime=triton requires --triton-http-url or --triton-model-spec (no-fallback)")
        if not args.triton_model_name or not str(args.triton_model_name).strip():
            raise RuntimeError(f"{NAME} | runtime=triton requires --triton-model-name or --triton-model-spec (no-fallback)")
    client = TritonHttpClient(base_url=str(args.triton_http_url), timeout_sec=10.0)
    if not client.ready():
        raise TritonError(f"{NAME} | Triton is not ready at {args.triton_http_url}", error_code="triton_unavailable")

    frame_manager = FrameManager(
        frames_dir=args.frames_dir,
        chunk_size=meta.get("chunk_size", 32),
        cache_size=meta.get("cache_size", 2),
    )

    union_ts = _get_union_timestamps_sec(frame_manager)
    idx_np = np.asarray(frame_indices, dtype=np.int32)
    n = int(idx_np.size)

    dt_seconds = np.full((n,), np.nan, dtype=np.float32)
    motion_norm_per_sec = np.full((n,), np.nan, dtype=np.float32)

    try:
        input_size = _preset_to_input_size(str(args.triton_preprocess_preset))

        prev_frame: Optional[np.ndarray] = None
        prev_t: Optional[float] = None
        for i, fi in enumerate(idx_np.tolist()):
            frame_rgb = frame_manager.get(int(fi))  # RGB uint8
            cur_t = float(union_ts[int(fi)]) if union_ts is not None and int(fi) < union_ts.size else None

            if prev_frame is None:
                prev_frame = frame_rgb
                prev_t = cur_t
                motion_norm_per_sec[i] = 0.0
                continue

            # dt
            if prev_t is not None and cur_t is not None:
                dt = max(cur_t - prev_t, 1e-6)
            else:
                fps = float(getattr(frame_manager, "fps", 30.0) or 30.0)
                dt = 1.0 / max(fps, 1e-6)

            dt_seconds[i] = float(dt)

            # Triton infer: 2-image inputs (UINT8 NHWC)
            inp0 = _prep_batch_rgb_uint8([prev_frame], input_size=input_size)
            inp1 = _prep_batch_rgb_uint8([frame_rgb], input_size=input_size)
            try:
                out0 = client.infer_two_inputs(
                    model_name=str(args.triton_model_name),
                    model_version=str(args.triton_model_version) if args.triton_model_version else None,
                    input0_name=str(args.triton_input0_name),
                    input0_tensor=inp0,
                    input1_name=str(args.triton_input1_name),
                    input1_tensor=inp1,
                    output_name=str(args.triton_output_name),
                    datatype=str(args.triton_datatype),
                )
            except AttributeError:
                raise RuntimeError(
                    f"{NAME} | dp_triton client missing infer_two_inputs(). "
                    f"Please update dp_triton to support 2-input models."
                )
            except Exception as e:
                raise RuntimeError(f"{NAME} | Triton infer failed: {e}") from e

            flow = np.asarray(out0.output, dtype=np.float32)
            # Expect (B,2,H,W) with B==1
            if flow.ndim != 4 or flow.shape[0] != 1 or flow.shape[1] != 2:
                raise RuntimeError(f"{NAME} | Triton output has invalid shape: {flow.shape}")
            flow = flow[0]  # (2,H,W)
            mag = np.sqrt(np.square(flow[0]) + np.square(flow[1]))
            mag_mean = float(np.mean(mag))
            norm = float(max(flow.shape[1], flow.shape[2], 1))
            val = float((mag_mean / dt) / norm)
            if not np.isfinite(val):
                raise RuntimeError(f"{NAME} | invalid motion value at frame_idx={int(fi)}")
            motion_norm_per_sec[i] = val

            prev_frame = frame_rgb
            prev_t = cur_t if cur_t is not None else (prev_t + dt if prev_t is not None else None)

            if i % 50 == 0:
                LOGGER.info(f"{NAME} | processed {i+1}/{n}")
    finally:
        try:
            frame_manager.close()
        except Exception:
            pass

    out_dir = os.path.join(args.rs_path, NAME)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "flow.npz")

    meta_out: Dict[str, Any] = {
        "producer": NAME,
        "producer_version": VERSION,
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.utcnow().isoformat(),
        "status": "ok",
        "empty_reason": None,
        "model_name": str(args.triton_model_name),
        "runtime": "triton-gpu",
        "device": "cuda",
        "triton_preprocess_preset": str(args.triton_preprocess_preset),
        "total_frames": int(total_frames),
    }
    required_run_keys = ["platform_id", "video_id", "run_id", "sampling_policy_version", "config_hash"]
    missing = [k for k in required_run_keys if not meta.get(k)]
    if missing:
        raise RuntimeError(f"{NAME} | frames metadata missing required run identity keys: {missing}")
    for k in required_run_keys:
        meta_out[k] = meta.get(k)

    # PR-3: model system baseline
    meta_out = apply_models_meta(
        meta_out,
        models_used=[
            model_used(
                model_name=str(args.triton_model_name),
                model_version=str(args.model_version or "unknown"),
                weights_digest=str(args.weights_digest or "unknown"),
                runtime="triton-gpu",
                engine="onnx",
                precision=str(args.precision or "unknown"),
                device="cuda",
            )
        ],
    )

    np.savez_compressed(
        out_path,
        frame_indices=idx_np,
        motion_norm_per_sec_mean=motion_norm_per_sec,
        dt_seconds=dt_seconds,
        meta=np.asarray(meta_out, dtype=object),
    )

    LOGGER.info(f"{NAME} | Saved result: {out_path}")


if __name__ == "__main__":
    main()


