from __future__ import annotations

import importlib
import os
from typing import Any, Callable, Dict, Optional, Tuple

from ..errors import ModelManagerError
from ..specs import ModelSpec


def _import_factory(dotted: str) -> Callable[..., Any]:
    """
    Import a callable from "package.module:attr" or "package.module.attr".
    """
    s = str(dotted or "").strip()
    if not s:
        raise ModelManagerError(message="Empty factory path", error_code="model_spec_invalid")
    if ":" in s:
        mod, attr = s.split(":", 1)
    else:
        parts = s.split(".")
        if len(parts) < 2:
            raise ModelManagerError(
                message="Factory path must be module.attr or module:attr",
                error_code="model_spec_invalid",
                details={"factory": s},
            )
        mod = ".".join(parts[:-1])
        attr = parts[-1]
    try:
        m = importlib.import_module(mod)
    except Exception as e:
        raise ModelManagerError(
            message="Failed to import factory module",
            error_code="model_spec_invalid",
            details={"factory": s, "module": mod, "error": str(e)},
        ) from e
    try:
        fn = getattr(m, attr)
    except Exception as e:
        raise ModelManagerError(
            message="Failed to resolve factory attribute",
            error_code="model_spec_invalid",
            details={"factory": s, "attr": attr, "error": str(e)},
        ) from e
    if not callable(fn):
        raise ModelManagerError(
            message="Factory is not callable",
            error_code="model_spec_invalid",
            details={"factory": s},
        )
    return fn


class TorchStateDictProvider:
    """
    Generic PyTorch provider that:
    - constructs a model using a python factory (dotted import path)
    - loads local `state_dict` from a file

    This is powerful but depends on python code and installed packages (torch/torchvision/etc.).

    Spec requirements (via runtime_params):
    - `factory`: dotted callable (e.g. "torchvision.models.video.slowfast_r50")
    - `factory_kwargs`: dict (must NOT include pretrained=True; forbidden)
    - `checkpoint_relpath`: optional, select which local_artifacts entry is the checkpoint
    - `state_dict_key`: optional, if checkpoint contains nested dict (default "state_dict")
    - `strict`: bool (default True)
    - `strip_prefix`: optional string prefix to strip from all keys (e.g. "module.")
    """

    def supports(self, spec: ModelSpec) -> bool:
        return (spec.runtime == "inprocess") and (str(spec.engine).lower() in ("torch", "pytorch", "torch-state-dict"))

    def load(
        self,
        *,
        spec: ModelSpec,
        device: str,
        precision: str,
        models_root: str,
        runtime_params: dict | None = None,
    ) -> Any:
        rp = runtime_params or spec.runtime_params or {}
        if not isinstance(rp, dict):
            rp = {}

        factory_path = rp.get("factory")
        if not isinstance(factory_path, str) or not factory_path.strip():
            raise ModelManagerError(
                message="TorchStateDictProvider requires runtime_params.factory",
                error_code="model_spec_invalid",
                details={"model_name": spec.model_name},
            )
        factory_kwargs = rp.get("factory_kwargs") if isinstance(rp.get("factory_kwargs"), dict) else {}
        # hard policy: forbid pretrained=True
        if "pretrained" in factory_kwargs and bool(factory_kwargs.get("pretrained")):
            raise ModelManagerError(
                message="pretrained=True is forbidden (no-network). Provide local weights via state_dict.",
                error_code="network_forbidden",
                details={"model_name": spec.model_name, "factory": factory_path},
            )

        state_key = rp.get("state_dict_key")
        state_key = str(state_key) if state_key is not None else "state_dict"
        strict = bool(rp.get("strict", True))
        strip_prefix = rp.get("strip_prefix")
        strip_prefix = str(strip_prefix) if strip_prefix is not None else None

        # Pick checkpoint artifact.
        ckpt_rel = rp.get("checkpoint_relpath")
        ckpt_rel = str(ckpt_rel) if ckpt_rel is not None else None
        if ckpt_rel:
            # ensure it exists in local_artifacts for auditability
            declared = {str(a.path) for a in spec.local_artifacts if str(a.kind) == "file"}
            if ckpt_rel not in declared:
                raise ModelManagerError(
                    message="checkpoint_relpath must point to one of spec.local_artifacts (kind=file)",
                    error_code="model_spec_invalid",
                    details={"model_name": spec.model_name, "checkpoint_relpath": ckpt_rel, "declared_files": sorted(declared)},
                )
        else:
            for a in spec.local_artifacts:
                if str(a.kind) == "file":
                    ckpt_rel = str(a.path)
                    break
        if not ckpt_rel:
            raise ModelManagerError(
                message="TorchStateDictProvider requires a checkpoint file (local_artifacts.kind=file)",
                error_code="weights_missing",
                details={"model_name": spec.model_name},
            )

        ckpt_path = os.path.join(models_root, ckpt_rel) if not os.path.isabs(ckpt_rel) else ckpt_rel
        ckpt_path = os.path.abspath(ckpt_path)
        if not os.path.isfile(ckpt_path):
            raise ModelManagerError(
                message="Checkpoint file not found",
                error_code="weights_missing",
                details={"model_name": spec.model_name, "checkpoint": ckpt_path},
            )

        try:
            import torch  # type: ignore
        except Exception as e:
            raise ModelManagerError(
                message="torch is not installed",
                error_code="dependency_missing",
                details={"import_error": str(e)},
            ) from e

        # device mapping
        map_location = "cpu"
        if str(device).lower().startswith("cuda"):
            map_location = device if ":" in str(device) else "cuda"

        factory = _import_factory(factory_path)

        # Construct model.
        try:
            model = factory(**factory_kwargs)
        except Exception as e:
            raise ModelManagerError(
                message="Failed to construct torch model via factory",
                error_code="model_load_failed",
                details={"model_name": spec.model_name, "factory": factory_path, "error": str(e)},
            ) from e

        # Load checkpoint.
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu")
        except Exception as e:
            raise ModelManagerError(
                message="Failed to load checkpoint file",
                error_code="model_load_failed",
                details={"model_name": spec.model_name, "checkpoint": ckpt_path, "error": str(e)},
            ) from e

        state = ckpt
        if isinstance(ckpt, dict) and state_key in ckpt and isinstance(ckpt[state_key], dict):
            state = ckpt[state_key]
        if not isinstance(state, dict):
            raise ModelManagerError(
                message="Checkpoint does not contain a state_dict dict",
                error_code="model_load_failed",
                details={"model_name": spec.model_name, "checkpoint": ckpt_path, "state_dict_key": state_key},
            )

        # Common checkpoint compatibility: strip a prefix from keys (e.g., "module.").
        if strip_prefix:
            try:
                cleaned = {}
                for k, v in state.items():
                    ks = str(k)
                    if ks.startswith(strip_prefix):
                        ks = ks[len(strip_prefix) :]
                    cleaned[ks] = v
                state = cleaned
            except Exception:
                # if something goes wrong, keep original dict
                pass

        try:
            missing, unexpected = model.load_state_dict(state, strict=strict)
        except Exception as e:
            raise ModelManagerError(
                message="load_state_dict failed",
                error_code="model_load_failed",
                details={"model_name": spec.model_name, "checkpoint": ckpt_path, "error": str(e)},
            ) from e

        # Move to device.
        try:
            model = model.to(map_location)
        except Exception:
            try:
                model = model.to("cpu")
                map_location = "cpu"
            except Exception:
                pass
        try:
            model.eval()
        except Exception:
            pass

        # Precision best-effort.
        if str(precision).lower() == "fp16" and str(map_location).startswith("cuda"):
            try:
                model = model.half()
            except Exception:
                pass

        # If strict=False, still consider surfacing mismatch details via exception in future.
        return model


