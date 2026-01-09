from __future__ import annotations

import os
from typing import Any

from ..errors import ModelManagerError
from ..specs import ModelSpec


class TorchScriptProvider:
    """
    Loads a TorchScript model from a local `.pt` / `.torchscript` file using `torch.jit.load`.

    This is the safest "generic torch" format because it does not require executing arbitrary
    python factory code and is naturally offline-friendly.
    """

    def supports(self, spec: ModelSpec) -> bool:
        return (spec.runtime == "inprocess") and (str(spec.engine).lower() in ("torchscript", "torch-jit", "jit"))

    def load(
        self,
        *,
        spec: ModelSpec,
        device: str,
        precision: str,
        models_root: str,
        runtime_params: dict | None = None,
    ) -> Any:
        # Pick first file artifact as the model file.
        model_file_rel = None
        for a in spec.local_artifacts:
            if str(a.kind) == "file":
                model_file_rel = str(a.path)
                break
        if not model_file_rel:
            raise ModelManagerError(
                message="TorchScriptProvider requires a local_artifacts entry with kind=file",
                error_code="weights_missing",
                details={"model_name": spec.model_name},
            )

        model_file = os.path.join(models_root, model_file_rel) if not os.path.isabs(model_file_rel) else model_file_rel
        model_file = os.path.abspath(model_file)
        if not os.path.isfile(model_file):
            raise ModelManagerError(
                message="TorchScript model file not found",
                error_code="weights_missing",
                details={"model_name": spec.model_name, "model_file": model_file},
            )

        try:
            import torch  # type: ignore
        except Exception as e:
            raise ModelManagerError(
                message="torch is not installed",
                error_code="dependency_missing",
                details={"import_error": str(e)},
            ) from e

        map_location = "cpu"
        if str(device).lower().startswith("cuda"):
            map_location = device if ":" in str(device) else "cuda"

        try:
            m = torch.jit.load(model_file, map_location=map_location)
        except Exception as e:
            raise ModelManagerError(
                message="Failed to load TorchScript model",
                error_code="model_load_failed",
                details={"model_name": spec.model_name, "model_file": model_file, "error": str(e)},
            ) from e

        try:
            m.eval()
        except Exception:
            pass

        # Precision: best-effort. TorchScript may be already quantized/typed.
        if str(precision).lower() == "fp16" and str(map_location).startswith("cuda"):
            try:
                m = m.half()
            except Exception:
                pass
        return m


