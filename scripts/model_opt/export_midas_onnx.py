#!/usr/bin/env python3
"""
PR-9: export MiDaS (intel-isl/MiDaS) to ONNX.

Notes:
- This script is intended to be run in an environment that has torch + torchvision installed.
- By default it may use torch.hub caches (and may download on first run if caches are empty).
- For baseline/offline workflows, use `--models-root` + `--offline` to forbid network.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from dataclasses import dataclass
from typing import Tuple

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


class _WrappedMiDaS(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.model(x)
        # normalize to (B,H,W)
        if y.dim() == 4 and y.shape[1] == 1:
            y = y[:, 0, :, :]
        return y


def _torchhub_find_cached_repo_dir(*, torch_home: str | None, repo_slug: str) -> str | None:
    """
    torch.hub cache layout typically:
      $TORCH_HOME/hub/<owner>_<repo>_<ref>/
    For intel-isl/MiDaS it is often:
      intel-isl_MiDaS_master/
    """
    if not torch_home:
        return None
    hub_dir = os.path.join(str(torch_home), "hub")
    if not os.path.isdir(hub_dir):
        return None
    owner_repo = repo_slug.replace("/", "_")
    exact = os.path.join(hub_dir, f"{owner_repo}_master")
    if os.path.isdir(exact):
        return exact
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
    Monkeypatch torch.hub.load so that GitHub-style repo slugs (owner/repo) are
    redirected to local cached repos under TORCH_HOME/hub with source='local'.
    Required because MiDaS hubconf calls torch.hub.load() internally for backbones.
    """
    if getattr(torch.hub, "_dp_offline_redirect_enabled", False):
        return

    orig_load = torch.hub.load

    def patched_load(repo_or_dir, model, *args, **kwargs):  # type: ignore[no-untyped-def]
        try:
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
            pass
        return orig_load(repo_or_dir, model, *args, **kwargs)

    torch.hub.load = patched_load  # type: ignore[assignment]
    setattr(torch.hub, "_dp_offline_redirect_enabled", True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Export MiDaS to ONNX")
    ap.add_argument("--model-name", default="MiDaS_small")
    ap.add_argument("--out", required=True, help="Output ONNX path")
    # torch 2.9 exporter uses opset 18 implementations and may fail converting down to 17.
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--h", type=int, default=256)
    ap.add_argument("--w", type=int, default=256)
    ap.add_argument("--dynamic", action="store_true", help="Enable dynamic axes for batch/height/width")
    ap.add_argument(
        "--models-root",
        type=str,
        default=None,
        help="Optional DP_MODELS_ROOT directory. When set, pins TORCH_HOME/HF_HOME caches under this path.",
    )
    ap.add_argument(
        "--offline",
        action="store_true",
        help="Strict no-network mode (fails if torch.hub tries to connect). Requires caches to be populated.",
    )
    args = ap.parse_args()

    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Optional: pin caches under models_root (useful for one-time bootstrap and reproducible exports).
    if args.models_root:
        try:
            from dp_models.offline import pin_cache_env, network_guard  # type: ignore
        except Exception as e:
            raise RuntimeError(f"dp_models is required for --models-root/--offline: {e}") from e
        pin_cache_env(str(args.models_root), offline=bool(args.offline))

    model_family = "intel-isl/MiDaS"
    # In strict offline mode, torch.hub may still try to hit GitHub to resolve refs.
    # Prefer a local cached repo dir under TORCH_HOME/hub/...
    local_repo = _torchhub_find_cached_repo_dir(torch_home=os.environ.get("TORCH_HOME"), repo_slug=model_family)
    repo_or_dir = local_repo or model_family
    if args.offline:
        # Fail-fast if anything tries to connect.
        from dp_models.offline import network_guard  # type: ignore

        with network_guard(enabled=True):
            _torchhub_offline_redirect_enable()
            if local_repo:
                model = torch.hub.load(local_repo, str(args.model_name), pretrained=True, trust_repo=True, verbose=False, source="local")
            else:
                model = torch.hub.load(model_family, str(args.model_name), pretrained=True, trust_repo=True, verbose=False)
    else:
        if local_repo:
            model = torch.hub.load(local_repo, str(args.model_name), pretrained=True, trust_repo=True, verbose=False, source="local")
        else:
            model = torch.hub.load(model_family, str(args.model_name), pretrained=True, trust_repo=True, verbose=False)
    model.eval()

    wrapped = _WrappedMiDaS(model)

    dummy = torch.randn(1, 3, int(args.h), int(args.w), dtype=torch.float32)

    input_names = ["input"]
    output_names = ["depth"]
    dynamic_axes = None
    if args.dynamic:
        dynamic_axes = {
            "input": {0: "batch", 2: "height", 3: "width"},
            "depth": {0: "batch", 1: "height_out", 2: "width_out"},
        }

    torch.onnx.export(
        wrapped,
        dummy,
        out_path,
        export_params=True,
        opset_version=int(args.opset),
        do_constant_folding=True,
        input_names=input_names,
        output_names=output_names,
        dynamic_axes=dynamic_axes,
    )

    digest = sha256_file(out_path)
    print(f"Exported: {out_path}")
    print(f"sha256: {digest}")


if __name__ == "__main__":
    main()


