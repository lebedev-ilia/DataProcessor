"""
story_structure legacy / experimental utilities (NOT Tier‑0 baseline)

This file is kept for future non-baseline extensions:
- topic features from subtitles using SentenceTransformer
- clustering-based segmentation

Policy:
- any ML model MUST be loaded via dp_models.ModelManager (no implicit downloads).
- model artifacts MUST live under DP_MODELS_ROOT and be referenced by dp_models specs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np

from dp_models.manager import get_global_model_manager
from dp_models.errors import ModelManagerError


@dataclass
class TopicModelConfig:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: str = "auto"
    precision: str = "fp16"


def load_sentence_transformer(cfg: TopicModelConfig) -> Any:
    """
    Loads a SentenceTransformer model via ModelManager (local-only).
    Returns the in-process model handle (SentenceTransformer instance).
    """
    mm = get_global_model_manager()
    try:
        resolved = mm.get(model_name=cfg.model_name)
    except ModelManagerError as e:
        raise RuntimeError(f"story_structure legacy | failed to load sentence model via ModelManager: {e}") from e
    return resolved.handle


def encode_subtitles(model: Any, subtitles: List[str]) -> np.ndarray:
    """
    Encodes subtitles to embeddings using SentenceTransformer handle.
    """
    if subtitles is None:
        return np.zeros((0, 1), dtype=np.float32)
    subs = [str(s).strip() for s in subtitles if str(s).strip()]
    if not subs:
        return np.zeros((0, 1), dtype=np.float32)
    emb = model.encode(subs)  # sentence-transformers API
    return np.asarray(emb, dtype=np.float32)


