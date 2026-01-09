"""
uniqueness (Tier‑0 baseline)

Baseline mode:
- No reference videos.
- Computes intra-video repetition/diversity proxies from `core_clip` embeddings on sampled frames.

Contract:
- `frame_indices` come strictly from Segmenter metadata (union-domain).
- time-axis strictly from `union_timestamps_sec` (per-second temporal change).
- hard dependency: `core_clip/embeddings.npz` must fully cover this module's `frame_indices` (no-fallback).
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from modules.base_module import BaseModule
from utils.frame_manager import FrameManager

MODULE_NAME = "uniqueness"
VERSION = "1.0"
SCHEMA_VERSION = "uniqueness_npz_v2"


def _unbox_object_scalar(x: Any) -> Any:
    if isinstance(x, np.ndarray) and x.dtype == object and x.shape == ():
        try:
            return x.item()
        except Exception:
            return x
    return x


def _require_union_times_s(frame_manager: FrameManager, frame_indices: np.ndarray) -> np.ndarray:
    meta = getattr(frame_manager, "meta", None)
    if not isinstance(meta, dict):
        raise RuntimeError("uniqueness | FrameManager.meta missing (requires union_timestamps_sec)")
    ts = meta.get("union_timestamps_sec")
    if not isinstance(ts, list) or not ts:
        raise RuntimeError("uniqueness | union_timestamps_sec missing/empty in frames metadata (no-fallback)")
    uts = np.asarray(ts, dtype=np.float32)

    if frame_indices.size == 0:
        raise RuntimeError("uniqueness | frame_indices is empty (no-fallback)")
    if int(np.max(frame_indices)) >= int(uts.shape[0]):
        raise RuntimeError("uniqueness | union_timestamps_sec does not cover frame_indices (no-fallback)")
    times_s = uts[frame_indices.astype(np.int32)]
    if times_s.size >= 2 and np.any(np.diff(times_s) < -1e-3):
        raise RuntimeError("uniqueness | union_timestamps_sec is not monotonic for frame_indices (no-fallback)")
    return times_s.astype(np.float32)


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1, keepdims=True) + 1e-9
    return x / norms


def _load_core_clip_embeddings_aligned(
    rs_path: str, want_frame_indices: np.ndarray
) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Load core_clip embeddings and align to requested frame_indices (union-domain).
    Requires full coverage (no gaps). No fallback.
    Returns (embeddings_aligned, core_clip_models_used_best_effort).
    """
    core_path = os.path.join(rs_path, "core_clip", "embeddings.npz")
    if not os.path.isfile(core_path):
        raise FileNotFoundError(f"uniqueness | missing core_clip embeddings: {core_path}")
    data = np.load(core_path, allow_pickle=True)
    core_idx = data.get("frame_indices")
    core_emb = data.get("frame_embeddings")
    if core_idx is None or core_emb is None:
        raise RuntimeError("uniqueness | core_clip embeddings.npz missing keys frame_indices/frame_embeddings")
    core_idx = np.asarray(core_idx, dtype=np.int32)
    core_emb = np.asarray(core_emb, dtype=np.float32)

    mapping = {int(fi): i for i, fi in enumerate(core_idx.tolist())}
    pos = [mapping.get(int(fi), -1) for fi in want_frame_indices.tolist()]
    if any(p < 0 for p in pos):
        raise RuntimeError(
            "uniqueness | core_clip does not cover requested frame_indices. "
            "Segmenter must provide consistent indices across core_clip and this module."
        )

    # Best-effort: read upstream models_used for reproducibility.
    models_used: List[Dict[str, Any]] = []
    meta = _unbox_object_scalar(data.get("meta"))
    if isinstance(meta, dict):
        mu = meta.get("models_used")
        if isinstance(mu, list):
            models_used = [x for x in mu if isinstance(x, dict)]

    return core_emb[np.asarray(pos, dtype=np.int64)], models_used


class UniquenessBaselineModule(BaseModule):
    """
    Baseline version of uniqueness:
    - No external reference videos.
    - Computes intra-video repetition/diversity proxies using `core_clip` embeddings.
    """

    VERSION = VERSION
    SCHEMA_VERSION = SCHEMA_VERSION

    @property
    def module_name(self) -> str:
        return MODULE_NAME

    def __init__(
        self,
        rs_path: Optional[str] = None,
        repeat_threshold: float = 0.97,
        max_frames: int = 200,
        **kwargs: Any,
    ):
        super().__init__(rs_path=rs_path, logger_name=self.module_name, **kwargs)
        self._repeat_threshold = float(repeat_threshold)
        self._max_frames = int(max_frames)
        self._last_core_clip_models_used: List[Dict[str, Any]] = []

    def required_dependencies(self) -> List[str]:
        return ["core_clip"]

    def get_models_used(self, config: Dict[str, Any], metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        # This module does not run ML models itself; include upstream core_clip model signature for reproducibility.
        return list(self._last_core_clip_models_used or [])

    def process(self, frame_manager: FrameManager, frame_indices: List[int], config: Dict[str, Any]) -> Dict[str, Any]:
        self.initialize()

        if not frame_indices:
            raise ValueError("uniqueness | frame_indices is empty")
        if self.rs_path is None:
            raise ValueError("uniqueness | rs_path is required")

        repeat_thr = float(config.get("repeat_threshold", self._repeat_threshold))
        max_frames = int(config.get("max_frames", self._max_frames))
        fi = np.asarray([int(i) for i in frame_indices], dtype=np.int32)

        if max_frames > 0 and int(fi.size) > int(max_frames):
            raise RuntimeError(
                f"uniqueness | too many frames for NxN similarity: N={int(fi.size)} > max_frames={int(max_frames)}. "
                "Fix Segmenter sampling for uniqueness (no-fallback)."
            )

        times_s = _require_union_times_s(frame_manager, fi)
        emb, core_clip_models_used = _load_core_clip_embeddings_aligned(self.rs_path, fi)
        self._last_core_clip_models_used = core_clip_models_used

        if emb.ndim != 2 or emb.shape[0] != fi.shape[0]:
            raise RuntimeError("uniqueness | invalid embeddings shape after alignment")

        emb_n = _normalize_rows(emb)
        n = int(emb_n.shape[0])

        # Pairwise similarity matrix (N x N)
        sim = emb_n @ emb_n.T
        np.fill_diagonal(sim, -np.inf)

        max_sim_other = np.max(sim, axis=1).astype(np.float32) if n > 0 else np.asarray([], dtype=np.float32)
        repetition_ratio = float(np.mean(max_sim_other >= repeat_thr)) if n > 0 else float("nan")

        if n >= 2:
            iu = np.triu_indices(n, k=1)
            sim_ut = (emb_n @ emb_n.T)[iu].astype(np.float32)
            pairwise_sim_mean = float(np.mean(sim_ut))
            pairwise_sim_p95 = float(np.percentile(sim_ut, 95))
        else:
            sim_ut = np.asarray([], dtype=np.float32)
            pairwise_sim_mean = float("nan")
            pairwise_sim_p95 = float("nan")

        if n >= 2:
            cos_sim_next = np.sum(emb_n[1:] * emb_n[:-1], axis=1).astype(np.float32)
            cos_dist_next = (1.0 - cos_sim_next).astype(np.float32)
            dt = np.diff(times_s).astype(np.float32)
            dt = np.maximum(dt, 1e-3)
            change_rate = (cos_dist_next / dt).astype(np.float32)
            temporal_change_mean = float(np.mean(change_rate))
            temporal_change_std = float(np.std(change_rate))
        else:
            cos_dist_next = np.asarray([], dtype=np.float32)
            temporal_change_mean = float("nan")
            temporal_change_std = float("nan")

        diversity_score = float(
            np.clip(
                1.0 - (pairwise_sim_mean if not np.isnan(pairwise_sim_mean) else 0.0),
                0.0,
                1.0,
            )
        )

        features: Dict[str, Any] = {
            "repeat_threshold": float(repeat_thr),
            "max_frames": int(max_frames),
            "repetition_ratio": float(repetition_ratio),
            "pairwise_sim_mean": float(pairwise_sim_mean),
            "pairwise_sim_p95": float(pairwise_sim_p95),
            "temporal_change_mean": float(temporal_change_mean),
            "temporal_change_std": float(temporal_change_std),
            "diversity_score": float(diversity_score),
            "n_frames": int(n),
        }

        return {
            "frame_indices": fi,
            "max_sim_to_other": max_sim_other,
            "cos_dist_next": cos_dist_next,
            "features": features,
        }


