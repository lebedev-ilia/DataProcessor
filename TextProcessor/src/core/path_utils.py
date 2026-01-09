from __future__ import annotations

import os
from pathlib import Path


def textprocessor_root() -> Path:
    """
    Returns the absolute path to the TextProcessor directory.

    File layout:
      TextProcessor/src/core/path_utils.py -> parents[3] == TextProcessor/
    """
    return Path(__file__).resolve().parents[3]


def default_cache_dir() -> Path:
    """
    Default cache directory for TextProcessor (embeddings caches, etc).
    Overridable via env var for prod/container setups.
    """
    env = os.environ.get("TREND_TEXT_CACHE_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return (textprocessor_root() / ".cache").resolve()


def default_artifacts_dir() -> Path:
    """
    Default artifacts directory for TextProcessor (intermediate .npy artifacts).
    Overridable via env var for prod/container setups.
    """
    env = os.environ.get("TREND_TEXT_ARTIFACTS_DIR")
    if env:
        return Path(env).expanduser().resolve()
    return (textprocessor_root() / ".artifacts").resolve()


