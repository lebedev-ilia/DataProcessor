#!/usr/bin/env python3
"""
PR-9: optional ONNX quantization helper (dynamic quantization).

This is intentionally lightweight and will fail-fast if onnxruntime is not installed.
"""

from __future__ import annotations

import argparse
import os


def main() -> None:
    ap = argparse.ArgumentParser(description="Quantize ONNX model (dynamic)")
    ap.add_argument("--in", dest="inp", required=True, help="Input ONNX path")
    ap.add_argument("--out", required=True, help="Output ONNX path")
    ap.add_argument("--weight-type", default="QInt8", choices=["QInt8", "QUInt8"])
    args = ap.parse_args()

    inp = os.path.abspath(args.inp)
    outp = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(outp), exist_ok=True)

    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic  # type: ignore
    except Exception as e:
        raise RuntimeError(f"onnxruntime is required for quantization: {e}") from e

    weight_type = QuantType.QInt8 if args.weight_type == "QInt8" else QuantType.QUInt8
    quantize_dynamic(inp, outp, weight_type=weight_type)
    print(f"Quantized: {outp}")


if __name__ == "__main__":
    main()


