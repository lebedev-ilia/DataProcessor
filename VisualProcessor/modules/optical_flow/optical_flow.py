"""
optical_flow (module)

Production policy:
- This module is a CONSUMER of `core_optical_flow` (NPZ) and MUST NOT compute RAFT itself.
- No JSON artifacts in result_store (NPZ-only).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import numpy as np

from modules.base_module import BaseModule
from utils.frame_manager import FrameManager


def _load_core_optical_flow_npz(rs_path: str) -> Dict[str, Any]:
    p = os.path.join(rs_path, "core_optical_flow", "flow.npz")
    if not os.path.isfile(p):
        raise FileNotFoundError(f"optical_flow | missing dependency: {p}")
    data = np.load(p, allow_pickle=True)
    idx = data.get("frame_indices")
    curve = data.get("motion_norm_per_sec_mean")
    if idx is None or curve is None:
        raise RuntimeError("optical_flow | core_optical_flow/flow.npz missing keys frame_indices/motion_norm_per_sec_mean")
    return {
        "frame_indices": np.asarray(idx, dtype=np.int32),
        "motion_norm_per_sec_mean": np.asarray(curve, dtype=np.float32),
        "meta": data.get("meta"),
    }


class OpticalFlowModule(BaseModule):
    @property
    def module_name(self) -> str:
        return "optical_flow"

    def required_dependencies(self) -> List[str]:
        return ["core_optical_flow"]

    def process(self, frame_manager: FrameManager, frame_indices: List[int], config: Dict[str, Any]) -> Dict[str, Any]:
        if self.rs_path is None:
            raise ValueError("optical_flow | rs_path is required")
        if not frame_indices:
            raise ValueError("optical_flow | frame_indices is empty")

        core = _load_core_optical_flow_npz(self.rs_path)
        core_idx = core["frame_indices"]
        core_curve = core["motion_norm_per_sec_mean"]

        want = np.asarray([int(i) for i in frame_indices], dtype=np.int32)
        mapping = {int(fi): i for i, fi in enumerate(core_idx.tolist())}
        pos = [mapping.get(int(fi), -1) for fi in want.tolist()]
        if any(p < 0 for p in pos):
            raise RuntimeError(
                "optical_flow | core_optical_flow.frame_indices does not cover this module's frame_indices. "
                "Segmenter must produce consistent sampling for dependent components."
            )
        curve = core_curve[np.asarray(pos, dtype=np.int64)]

        # Aggregates (stable, tabular-friendly)
        curve_safe = curve.copy()
        if curve_safe.size and not np.isfinite(curve_safe[0]):
            curve_safe[0] = 0.0

        features: Dict[str, Any] = {
            "motion_curve_mean": float(np.mean(curve_safe)) if curve_safe.size else float("nan"),
            "motion_curve_median": float(np.median(curve_safe)) if curve_safe.size else float("nan"),
            "motion_curve_p90": float(np.percentile(curve_safe, 90)) if curve_safe.size else float("nan"),
            "motion_curve_variance": float(np.var(curve_safe)) if curve_safe.size else float("nan"),
        }

        return {
            "frame_indices": want,
            "motion_norm_per_sec_mean": curve.astype(np.float32),
            "features": np.asarray(features, dtype=object),
        }


