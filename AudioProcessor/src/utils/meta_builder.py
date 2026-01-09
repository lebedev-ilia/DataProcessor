from __future__ import annotations

import os
import sys

# Allow importing from repo-root `common/` even when running AudioProcessor as a standalone script.
_ap_src = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # .../AudioProcessor/src
_repo_root = os.path.dirname(os.path.dirname(_ap_src))  # .../DataProcessor
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from common.meta_builder import apply_models_meta, compute_model_signature, model_used  # noqa: E402

__all__ = ["apply_models_meta", "compute_model_signature", "model_used"]


