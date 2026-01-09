from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from datetime import datetime
from typing import Any, DefaultDict, Dict, List, Tuple

import numpy as np


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


def _iter_jsonl(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if not s:
                continue
            try:
                yield json.loads(s)
            except Exception:
                # Skip malformed lines.
                continue


def main() -> None:
    ap = argparse.ArgumentParser("Summarize benchmark results.jsonl into summary.json")
    ap.add_argument("--in", dest="in_path", required=True, help="Input results.jsonl")
    ap.add_argument("--out", dest="out_path", default=None, help="Output summary.json (default: sibling of input)")
    ap.add_argument("--bench", default=None, help="Optional filter: keep only rows with bench==<value>")
    args = ap.parse_args()

    in_path = os.path.abspath(str(args.in_path))
    if not os.path.isfile(in_path):
        raise SystemExit(f"Input not found: {in_path}")
    out_path = os.path.abspath(args.out_path) if args.out_path else os.path.join(os.path.dirname(in_path), "summary.json")

    buckets: DefaultDict[Tuple[str, int], List[float]] = defaultdict(list)
    meta: Dict[str, Any] = {"bench": set(), "device": set(), "dtype_requested": set(), "dtype_used": set()}

    n_rows = 0
    for row in _iter_jsonl(in_path):
        if not isinstance(row, dict):
            continue
        if args.bench is not None and row.get("bench") != args.bench:
            continue

        variant = row.get("variant")
        batch = row.get("batch")
        latency = row.get("latency_ms")
        if not isinstance(variant, str):
            continue
        try:
            batch_i = int(batch)
            lat_f = float(latency)
        except Exception:
            continue

        buckets[(variant, batch_i)].append(lat_f)
        n_rows += 1

        for k in ("bench", "device", "dtype_requested", "dtype_used"):
            v = row.get(k)
            if isinstance(v, str):
                meta[k].add(v)

    items = []
    for (variant, batch_i), samples in sorted(buckets.items(), key=lambda x: (x[0][0], x[0][1])):
        items.append({"variant": variant, "batch": int(batch_i), "quantiles_ms": _quantiles_ms(samples)})

    def _freeze_set(x: Any) -> Any:
        if isinstance(x, set):
            lst = sorted(x)
            return lst[0] if len(lst) == 1 else lst
        return x

    summary: Dict[str, Any] = {
        "created_at": datetime.utcnow().isoformat(),
        "in_path": in_path,
        "out_path": out_path,
        "rows_used": int(n_rows),
        "meta": {k: _freeze_set(v) for k, v in meta.items()},
        "items": items,
    }

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(f"[summarize_results] wrote: {out_path}")


if __name__ == "__main__":
    main()


