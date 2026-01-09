#!/usr/bin/env python3
"""
Export Places365 (scene_classification) models to ONNX for Triton.

Policy:
- fixed shapes (no dynamic axes)
- offline-only: all weights must be available under DP_MODELS_ROOT (no-network)

Typical usage (local):
  export DP_MODELS_ROOT=/abs/path/to/DataProcessor/dp_models/bundled_models
  PY=VisualProcessor/.vp_venv/bin/python
  "$PY" scripts/model_opt/export_places365_onnx.py \
      --model-spec places365_resnet50 \
      --input-size 224 \
      --out models/optimized/places365/places365_resnet50_224.onnx \
      --offline
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys

import torch  # type: ignore

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser("Export Places365 to ONNX (fixed shapes)")
    ap.add_argument("--model-spec", required=True, help="dp_models spec name (e.g., places365_resnet50)")
    ap.add_argument("--input-size", type=int, default=224, help="Square input size (default: 224)")
    ap.add_argument("--out", required=True, help="Output ONNX path")
    ap.add_argument("--models-root", type=str, default=None, help="DP_MODELS_ROOT path (optional; sets env)")
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--offline", action="store_true", help="Strict no-network guard during load")
    args = ap.parse_args()

    if args.models_root:
        os.environ["DP_MODELS_ROOT"] = os.path.abspath(str(args.models_root))

    # Strict offline: block socket.connect
    net_guard_ctx = None
    if args.offline:
        from dp_models.offline import network_guard  # type: ignore

        net_guard_ctx = network_guard(enabled=True)

    from dp_models import get_global_model_manager  # type: ignore

    mm = get_global_model_manager()
    if net_guard_ctx is not None:
        with net_guard_ctx:
            resolved = mm.get(model_name=str(args.model_spec))
    else:
        resolved = mm.get(model_name=str(args.model_spec))

    model = resolved.handle
    if not isinstance(model, torch.nn.Module):
        raise SystemExit(f"Resolved model handle is not a torch.nn.Module for spec={args.model_spec}")

    # Export stability: force CPU to avoid mixed-device issues with torch.export-based ONNX pipeline.
    model = model.to("cpu")
    model.eval()

    out_path = os.path.abspath(str(args.out))
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    s = int(args.input_size)
    if s <= 0:
        raise SystemExit(f"--input-size must be > 0; got {s}")

    dummy = torch.randn(1, 3, s, s, dtype=torch.float32, device="cpu")
    torch.onnx.export(
        model,
        dummy,
        out_path,
        export_params=True,
        opset_version=int(args.opset),
        do_constant_folding=True,
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes=None,
    )

    print(f"Exported: {out_path}")
    print(f"sha256: {sha256_file(out_path)}")


if __name__ == "__main__":
    main()


