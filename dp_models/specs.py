from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

from .errors import ModelManagerError

Runtime = Literal["inprocess", "triton"]

_ENV_RE = re.compile(r"\$\{([^}]+)\}")


def _expand_env_str(s: str) -> str:
    """
    Expand ${VAR} placeholders using current environment.

    IMPORTANT:
    - We intentionally do NOT expand `local_artifacts` paths, to avoid turning them into
      absolute paths and bypassing `models_root` safety checks.
    """
    ss = str(s)
    if "${" not in ss:
        return ss
    return _ENV_RE.sub(lambda m: os.environ.get(m.group(1), ""), ss)


def _expand_env_obj(x: Any) -> Any:
    """
    Recursively expand env placeholders inside JSON/YAML-parsed objects.
    Applies only to strings (and containers of strings).
    """
    if isinstance(x, str):
        return _expand_env_str(x)
    if isinstance(x, list):
        return [_expand_env_obj(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _expand_env_obj(v) for k, v in x.items()}
    return x


@dataclass(frozen=True)
class LocalArtifact:
    path: str  # relative to models_root, or absolute if explicitly allowed
    kind: Literal["file", "dir"] = "file"


@dataclass(frozen=True)
class ModelSpec:
    model_name: str
    model_version: str
    role: str
    runtime: Runtime
    engine: str
    precision: str  # "fp32" | "fp16" ...
    device_policy: str  # "auto" | "cpu" | "cuda" | "cuda:0"
    local_artifacts: List[LocalArtifact]
    weights_digest: str  # "auto" | explicit sha256 | "provided_by_deploy" (for triton)
    preprocess_preset: Optional[str] = None
    runtime_params: Optional[Dict[str, Any]] = None

    @staticmethod
    def from_dict(d: Dict[str, Any]) -> "ModelSpec":
        try:
            la_raw = d.get("local_artifacts") or []
            local_artifacts = [
                LocalArtifact(path=str(x["path"]), kind=str(x.get("kind", "file"))) for x in la_raw
            ]
            rp = d.get("runtime_params")
            runtime_params = _expand_env_obj(rp) if isinstance(rp, dict) else None
            return ModelSpec(
                model_name=str(d["model_name"]),
                model_version=str(d.get("model_version") or "unknown"),
                role=str(d["role"]),
                runtime=str(d.get("runtime") or "inprocess"),
                engine=str(d.get("engine") or "unknown"),
                precision=str(d.get("precision") or "unknown"),
                device_policy=str(d.get("device_policy") or "auto"),
                local_artifacts=local_artifacts,
                weights_digest=str(d.get("weights_digest") or "unknown"),
                preprocess_preset=(str(d["preprocess_preset"]) if d.get("preprocess_preset") is not None else None),
                runtime_params=runtime_params,
            )
        except KeyError as e:
            raise ModelManagerError(
                message=f"ModelSpec missing required field: {e}",
                error_code="model_spec_invalid",
                details={"missing": str(e)},
            ) from e


def _load_yaml(path: str) -> Dict[str, Any]:
    try:
        import yaml  # type: ignore
    except Exception as e:
        raise ModelManagerError(
            message="YAML spec requested but PyYAML is not installed",
            error_code="model_spec_invalid",
            details={"path": path, "import_error": str(e)},
        ) from e
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ModelManagerError(
            message="YAML spec must parse into an object",
            error_code="model_spec_invalid",
            details={"path": path},
        )
    return data


def load_model_spec(path: str) -> ModelSpec:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".json",):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif ext in (".yml", ".yaml"):
        data = _load_yaml(path)
    else:
        raise ModelManagerError(
            message=f"Unsupported model spec extension: {ext}",
            error_code="model_spec_invalid",
            details={"path": path},
        )
    if not isinstance(data, dict):
        raise ModelManagerError(
            message="Model spec must be a JSON/YAML object",
            error_code="model_spec_invalid",
            details={"path": path},
        )
    return ModelSpec.from_dict(data)


