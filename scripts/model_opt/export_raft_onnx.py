#!/usr/bin/env python3
"""
Export torchvision RAFT (small/large) to ONNX for fixed input sizes (NO dynamic axes).

This script is intended for pre-Triton preparation:
- run a pre-Triton bench to choose branch sizes
- export exact ONNX branches for Triton

Notes:
- torchvision will use local cache (~/.cache/torch/hub/checkpoints) if present.
- Network may be used only if weights are missing from cache (avoid in baseline).
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys

import torch  # type: ignore
import torchvision.models.optical_flow as models  # type: ignore

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class WrappedRAFT(torch.nn.Module):
    def __init__(self, m: torch.nn.Module) -> None:
        super().__init__()
        self.m = m

    def forward(self, x0: torch.Tensor, x1: torch.Tensor) -> torch.Tensor:
        # torchvision RAFT returns list of flows; we export the last prediction.
        out = self.m(x0, x1)
        if isinstance(out, (list, tuple)) and len(out) > 0:
            y = out[-1]
        else:
            y = out
        return y


def main() -> None:
    ap = argparse.ArgumentParser("Export RAFT to ONNX (fixed size)")
    ap.add_argument("--model", default="raft_small", choices=["raft_small", "raft_large"])
    ap.add_argument("--out", required=True)
    # torch 2.9 exporter uses opset 18 implementations and may fail converting down to 17.
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--h", type=int, default=256)
    ap.add_argument("--w", type=int, default=256)
    ap.add_argument("--models-root", type=str, default=None, help="Optional DP_MODELS_ROOT (pins TORCH_HOME/HF_HOME caches)")
    ap.add_argument("--offline", action="store_true", help="Strict no-network mode (requires cached weights)")
    args = ap.parse_args()

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    if args.models_root:
        try:
            from dp_models.offline import pin_cache_env, network_guard  # type: ignore
        except Exception as e:
            raise RuntimeError(f"dp_models is required for --models-root/--offline: {e}") from e
        pin_cache_env(str(args.models_root), offline=bool(args.offline))

    net_guard_ctx = None
    if args.offline:
        from dp_models.offline import network_guard  # type: ignore

        net_guard_ctx = network_guard(enabled=True)

    if args.model == "raft_large":
        if net_guard_ctx is not None:
            with net_guard_ctx:
                m = models.raft_large(weights=models.Raft_Large_Weights.DEFAULT, progress=False)
        else:
            m = models.raft_large(weights=models.Raft_Large_Weights.DEFAULT, progress=False)
    else:
        if net_guard_ctx is not None:
            with net_guard_ctx:
                m = models.raft_small(weights=models.Raft_Small_Weights.DEFAULT, progress=False)
        else:
            m = models.raft_small(weights=models.Raft_Small_Weights.DEFAULT, progress=False)
    m.eval()

    wrapped = WrappedRAFT(m)

    # Input: (B,3,H,W) float32 (preprocess is expected to be in Triton graph; this is model-only export).
    x0 = torch.randn(1, 3, int(args.h), int(args.w), dtype=torch.float32)
    x1 = torch.randn(1, 3, int(args.h), int(args.w), dtype=torch.float32)

    torch.onnx.export(
        wrapped,
        (x0, x1),
        out_path,
        export_params=True,
        opset_version=int(args.opset),
        do_constant_folding=True,
        input_names=["input0", "input1"],
        output_names=["flow"],
        dynamic_axes=None,
    )

    digest = sha256_file(out_path)
    print(f"Exported: {out_path}")
    print(f"sha256: {digest}")


if __name__ == "__main__":
    main()


