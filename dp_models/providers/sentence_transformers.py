from __future__ import annotations

import os
from typing import Any

from ..errors import ModelManagerError
from ..specs import ModelSpec


class SentenceTransformerProvider:
    """
    In-process SentenceTransformers provider.

    Policy:
    - model must be loaded from a **local directory** (no HF id downloads).
    - offline env must be set by ModelManager.
    """

    def supports(self, spec: ModelSpec) -> bool:
        return (spec.runtime == "inprocess") and ("sentence-transformers" in (spec.engine or "").lower())

    def load(
        self,
        *,
        spec: ModelSpec,
        device: str,
        precision: str,
        models_root: str,
        runtime_params: dict | None = None,
    ) -> Any:
        # Pick first directory artifact as the model folder.
        model_dir_rel = None
        for a in spec.local_artifacts:
            if str(a.kind) == "dir":
                model_dir_rel = str(a.path)
                break
        if not model_dir_rel:
            raise ModelManagerError(
                message="SentenceTransformerProvider requires a local_artifacts entry with kind=dir",
                error_code="weights_missing",
                details={"model_name": spec.model_name},
            )
        model_dir = os.path.join(models_root, model_dir_rel) if not os.path.isabs(model_dir_rel) else model_dir_rel
        model_dir = os.path.abspath(model_dir)
        if not os.path.isdir(model_dir):
            raise ModelManagerError(
                message="Local SentenceTransformer directory not found",
                error_code="weights_missing",
                details={"model_name": spec.model_name, "model_dir": model_dir},
            )

        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as e:
            raise ModelManagerError(
                message="sentence_transformers is not installed",
                error_code="dependency_missing",
                details={"import_error": str(e)},
            ) from e

        # Force local path loading. If this fails due to missing files, we want a hard error.
        try:
            model = SentenceTransformer(model_dir, device=device)
        except Exception as e:
            raise ModelManagerError(
                message="Failed to load SentenceTransformer from local directory",
                error_code="model_load_failed",
                details={"model_name": spec.model_name, "model_dir": model_dir, "error": str(e)},
            ) from e

        # Precision best-effort (SentenceTransformer wraps torch model internally).
        if str(precision).lower() == "fp16" and ("cuda" in str(device).lower()):
            try:
                model = model.half()
            except Exception:
                pass
        try:
            model.eval()
        except Exception:
            pass
        return model


