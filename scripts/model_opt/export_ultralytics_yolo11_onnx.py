#!/usr/bin/env python3
"""
Export Ultralytics YOLO11 weights to ONNX for Triton (fixed-shape branches).

Policy:
- offline-friendly: weights are loaded from DP_MODELS_ROOT only (no downloads)
- fixed shapes (no dynamic axes)

Default branches for baseline video resolutions (see docs/models_docs/BASELINE_GPU_BRANCHES.md):
- 320 (small), 640 (medium), 960 (large)

Outputs (in --out-dir):
- yolo11x_{S}.onnx
- yolo11x_{S}.onnx.data (if external data is used)
- yolo11x_{S}.meta.json (input/output names + shapes)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from typing import Any, Dict, List, Tuple


_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _load_onnx_io(path: str) -> Dict[str, Any]:
    import onnx  # type: ignore

    m = onnx.load(path)

    def _shape(t) -> List[int]:
        out = []
        for d in t.type.tensor_type.shape.dim:
            out.append(int(getattr(d, "dim_value", 0) or 0))
        return out

    return {
        "inputs": [{"name": i.name, "shape": _shape(i), "dtype": str(i.type.tensor_type.elem_type)} for i in m.graph.input],
        "outputs": [{"name": o.name, "shape": _shape(o), "dtype": str(o.type.tensor_type.elem_type)} for o in m.graph.output],
        "opset_imports": [{"domain": oi.domain, "version": int(oi.version)} for oi in m.opset_import],
    }


def _resolve_weights_path(models_root: str, rel_path: str) -> str:
    p = os.path.abspath(os.path.join(models_root, rel_path))
    if not os.path.exists(p):
        raise FileNotFoundError(f"weights not found under DP_MODELS_ROOT: {p}")
    return p


def main() -> None:
    ap = argparse.ArgumentParser("Export Ultralytics YOLO11 to ONNX (fixed branches)")
    ap.add_argument("--models-root", type=str, required=True, help="DP_MODELS_ROOT (local offline bundle)")
    ap.add_argument("--weights-rel", type=str, default="visual/yolo/yolo11x.pt", help="Path relative to DP_MODELS_ROOT")
    ap.add_argument("--sizes", type=str, default="320,640,960", help="Comma-separated square sizes")
    ap.add_argument("--out-dir", type=str, required=True, help="Output directory")
    ap.add_argument("--opset", type=int, default=18)
    args = ap.parse_args()

    models_root = os.path.abspath(str(args.models_root))
    out_dir = os.path.abspath(str(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)

    weights_path = _resolve_weights_path(models_root, str(args.weights_rel))

    sizes: List[int] = []
    for s in str(args.sizes).split(","):
        s = s.strip()
        if not s:
            continue
        sizes.append(int(s))
    if not sizes:
        raise ValueError("--sizes is empty")

    from ultralytics import YOLO  # type: ignore

    y = YOLO(weights_path)

    for s in sizes:
        # Ultralytics exporter writes into the same folder as weights by default;
        # we export into a temp folder, then move to out_dir with deterministic names.
        tmp_dir = os.path.join(out_dir, f"_tmp_export_{s}")
        os.makedirs(tmp_dir, exist_ok=True)

        exported_path = y.export(
            format="onnx",
            imgsz=int(s),
            dynamic=False,
            simplify=False,
            opset=int(args.opset),
            half=False,
            int8=False,
            device="cpu",
            project=tmp_dir,
            name=f"yolo11x_{s}",
        )
        if not exported_path or not os.path.exists(str(exported_path)):
            raise RuntimeError(f"Ultralytics export failed for size={s}: {exported_path}")

        src_onnx = os.path.abspath(str(exported_path))
        dst_onnx = os.path.join(out_dir, f"yolo11x_{s}.onnx")
        shutil.copy2(src_onnx, dst_onnx)

        # External data (optional)
        src_data = src_onnx + ".data"
        if os.path.exists(src_data):
            shutil.copy2(src_data, dst_onnx + ".data")

        meta = _load_onnx_io(dst_onnx)
        meta_path = os.path.join(out_dir, f"yolo11x_{s}.meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        # cleanup temp dir to keep repo tidy
        try:
            shutil.rmtree(tmp_dir)
        except Exception:
            pass

        print(f"[ok] size={s} -> {dst_onnx}")
        print(f"[ok] meta -> {meta_path}")


if __name__ == "__main__":
    main()


