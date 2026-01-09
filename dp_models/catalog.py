from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from .errors import ModelManagerError
from .specs import ModelSpec, load_model_spec


def _iter_spec_files(root: str) -> List[str]:
    out: List[str] = []
    for base, dirs, files in os.walk(root):
        # ignore hidden dirs
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        for fn in files:
            if fn.startswith("."):
                continue
            ext = os.path.splitext(fn)[1].lower()
            if ext not in (".yaml", ".yml", ".json"):
                continue
            out.append(os.path.join(base, fn))
    out.sort()
    return out


@dataclass
class ModelCatalog:
    """
    Loads and indexes ModelSpec files from a directory.
    """

    specs_by_name: Dict[str, ModelSpec]
    specs_by_role: Dict[str, List[ModelSpec]]

    @staticmethod
    def load_from_dir(root: str) -> "ModelCatalog":
        root = os.path.abspath(str(root))
        if not os.path.isdir(root):
            raise ModelManagerError(
                message="Model catalog directory not found",
                error_code="model_catalog_missing",
                details={"root": root},
            )

        by_name: Dict[str, ModelSpec] = {}
        by_role: Dict[str, List[ModelSpec]] = {}
        for path in _iter_spec_files(root):
            spec = load_model_spec(path)
            if spec.model_name in by_name:
                raise ModelManagerError(
                    message="Duplicate model_name in catalog",
                    error_code="model_catalog_invalid",
                    details={"model_name": spec.model_name, "paths": [path]},
                )
            by_name[spec.model_name] = spec
            by_role.setdefault(spec.role, []).append(spec)

        for role, arr in by_role.items():
            arr.sort(key=lambda s: (s.model_name, s.model_version))

        return ModelCatalog(specs_by_name=by_name, specs_by_role=by_role)

    def get_by_name(self, model_name: str) -> ModelSpec:
        spec = self.specs_by_name.get(str(model_name))
        if not spec:
            raise ModelManagerError(
                message="Model spec not found",
                error_code="model_not_found",
                details={"model_name": str(model_name)},
            )
        return spec

    def pick_by_role(self, role: str, *, preferred_name: Optional[str] = None) -> ModelSpec:
        role = str(role)
        arr = self.specs_by_role.get(role) or []
        if not arr:
            raise ModelManagerError(
                message="No models registered for role",
                error_code="model_not_found",
                details={"role": role},
            )
        if preferred_name:
            for s in arr:
                if s.model_name == preferred_name:
                    return s
        # default: first (stable)
        return arr[0]


