from __future__ import annotations

from typing import Any, Dict, Optional

from ..errors import ModelManagerError
from ..specs import ModelSpec


class TritonHttpProvider:
    """
    Minimal provider for Triton-backed models.

    Notes:
    - This provider only resolves a lightweight client handle and validates required runtime params.
    - It does NOT attempt to infer input/output schemas; those remain component-specific for now.
    """

    def supports(self, spec: ModelSpec) -> bool:
        return str(spec.runtime or "").strip().lower() == "triton"

    def load(
        self,
        *,
        spec: ModelSpec,
        device: str,
        precision: str,
        models_root: str,
        runtime_params: Optional[Dict[str, Any]] = None,
    ) -> Any:
        rp = runtime_params or spec.runtime_params or {}
        if not isinstance(rp, dict):
            rp = {}

        triton_http_url = str(rp.get("triton_http_url") or "").strip()
        triton_model_name = str(rp.get("triton_model_name") or "").strip()
        timeout_sec = float(rp.get("triton_timeout_sec") or 10.0)

        if not triton_http_url:
            raise ModelManagerError(
                message="Triton model spec is missing runtime_params.triton_http_url",
                error_code="model_spec_invalid",
                details={"model_name": spec.model_name},
            )
        if not triton_model_name:
            raise ModelManagerError(
                message="Triton model spec is missing runtime_params.triton_model_name",
                error_code="model_spec_invalid",
                details={"model_name": spec.model_name},
            )

        try:
            from dp_triton import TritonHttpClient  # type: ignore
        except Exception as e:
            raise ModelManagerError(
                message="dp_triton is required for Triton runtime but is not available",
                error_code="triton_unavailable",
                details={"model_name": spec.model_name, "import_error": str(e)},
            ) from e

        client = TritonHttpClient(base_url=triton_http_url, timeout_sec=timeout_sec)
        # Return a small handle; callers can still access full rp via resolved.spec.runtime_params.
        return {"client": client, "triton_http_url": triton_http_url, "triton_model_name": triton_model_name}


