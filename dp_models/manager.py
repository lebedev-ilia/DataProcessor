from __future__ import annotations

import os
from collections import OrderedDict
from dataclasses import dataclass
from threading import RLock
from typing import Any, Dict, Optional, Tuple

from .catalog import ModelCatalog
from .digests import sha256_dir, sha256_file
from .errors import ModelManagerError
from .offline import enforce_offline_env
from .providers import ProviderRegistry, SentenceTransformerProvider, TorchScriptProvider, TorchStateDictProvider, TritonHttpProvider
from .providers.base import ResolvedModel
from .signatures import model_used
from .specs import LocalArtifact, ModelSpec


def _default_catalog_root() -> str:
    here = os.path.dirname(__file__)
    return os.path.join(here, "spec_catalog")


def _default_models_root() -> str:
    env = os.environ.get("DP_MODELS_ROOT") or os.environ.get("MODELS_ROOT")
    if env:
        return str(env)
    raise ModelManagerError(
        message="DP_MODELS_ROOT is not set",
        error_code="models_root_missing",
        details={"hint": "export DP_MODELS_ROOT=/path/to/local/models"},
    )


def _pick_device(device_policy: str) -> str:
    p = str(device_policy or "auto").strip().lower()
    if p in ("cpu",):
        return "cpu"
    if p in ("cuda", "gpu", "cuda:0"):
        return "cuda" if p == "cuda" else p
    if p == "auto":
        try:
            import torch  # type: ignore

            if torch.cuda.is_available():
                return "cuda"
        except Exception:
            pass
        return "cpu"
    # allow explicit cuda:N
    if p.startswith("cuda:"):
        return p
    return p


def _safe_join(models_root: str, rel: str) -> str:
    """
    Join models_root with rel and forbid path traversal.
    """
    base = os.path.abspath(str(models_root))
    cand = os.path.abspath(os.path.join(base, rel))
    if not (cand == base or cand.startswith(base + os.sep)):
        raise ModelManagerError(
            message="Local artifact path escapes models_root",
            error_code="model_spec_invalid",
            details={"models_root": base, "path": rel, "resolved": cand},
        )
    return cand


def _validate_local_artifacts(
    *,
    spec: ModelSpec,
    models_root: str,
    allow_absolute_paths: bool,
) -> Dict[str, str]:
    """
    Returns mapping of artifact spec path → resolved absolute path.
    """
    out: Dict[str, str] = {}
    for a in spec.local_artifacts:
        p = str(a.path)
        if os.path.isabs(p):
            if not allow_absolute_paths:
                raise ModelManagerError(
                    message="Absolute local_artifacts paths are forbidden by policy",
                    error_code="model_spec_invalid",
                    details={"model_name": spec.model_name, "path": p},
                )
            abs_path = os.path.abspath(p)
        else:
            abs_path = _safe_join(models_root, p)

        kind = str(a.kind)
        if kind == "file":
            if not os.path.isfile(abs_path):
                raise ModelManagerError(
                    message="Required model file is missing",
                    error_code="weights_missing",
                    details={"model_name": spec.model_name, "path": p, "resolved": abs_path},
                )
        elif kind == "dir":
            if not os.path.isdir(abs_path):
                raise ModelManagerError(
                    message="Required model directory is missing",
                    error_code="weights_missing",
                    details={"model_name": spec.model_name, "path": p, "resolved": abs_path},
                )
        else:
            raise ModelManagerError(
                message="Invalid local_artifacts.kind",
                error_code="model_spec_invalid",
                details={"model_name": spec.model_name, "kind": kind, "path": p},
            )
        out[p] = abs_path
    return out


def _compute_weights_digest(spec: ModelSpec, resolved_artifacts: Dict[str, str]) -> str:
    """
    Digest policy:
    - if spec.weights_digest is explicit and not "auto" → use it
    - if "auto" → derive digest from local artifacts deterministically
    """
    wd = str(spec.weights_digest or "unknown").strip()
    if wd and wd not in ("auto", "unknown"):
        return wd

    # auto digest from artifacts
    if not resolved_artifacts:
        return "unknown"

    items = list(resolved_artifacts.items())
    items.sort(key=lambda t: t[0])

    # single artifact: digest directly
    if len(items) == 1:
        _, path = items[0]
        if os.path.isdir(path):
            return sha256_dir(path)
        return sha256_file(path)

    # multi-artifact: digest of "path\0digest\n" records
    import hashlib

    h = hashlib.sha256()
    for rel, path in items:
        h.update(rel.encode("utf-8"))
        h.update(b"\0")
        d = sha256_dir(path) if os.path.isdir(path) else sha256_file(path)
        h.update(d.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


@dataclass
class ModelManagerConfig:
    models_root: str
    catalog_root: str
    enforce_offline_env: bool = True
    allow_absolute_paths: bool = False
    max_loaded_models: int = 32


class ModelManager:
    """
    Unified model manager (in-process only for now).
    """

    def __init__(self, cfg: Optional[ModelManagerConfig] = None) -> None:
        if cfg is None:
            cfg = ModelManagerConfig(
                models_root=_default_models_root(),
                catalog_root=_default_catalog_root(),
            )
        self.cfg = cfg

        self.models_root = os.path.abspath(str(cfg.models_root))
        if cfg.enforce_offline_env:
            enforce_offline_env(self.models_root)

        self.catalog = ModelCatalog.load_from_dir(cfg.catalog_root)

        # Provider registry
        self.providers = ProviderRegistry()
        self.providers.register(SentenceTransformerProvider())
        self.providers.register(TorchScriptProvider())
        self.providers.register(TorchStateDictProvider())
        # Triton-backed models (HTTP client handle only; actual inference remains component-specific for now).
        self.providers.register(TritonHttpProvider())

        # LRU cache: key -> ResolvedModel
        # IMPORTANT: key MUST include weights_digest to ensure "new model = new cache".
        self._cache: "OrderedDict[Tuple[str, str, str, str, str, str, str], ResolvedModel]" = OrderedDict()
        self._lock = RLock()

    def get_spec(self, *, model_name: Optional[str] = None, role: Optional[str] = None, preferred_name: Optional[str] = None) -> ModelSpec:
        if model_name:
            return self.catalog.get_by_name(model_name)
        if role:
            return self.catalog.pick_by_role(role, preferred_name=preferred_name)
        raise ModelManagerError(message="Either model_name or role must be provided", error_code="model_spec_invalid")

    def resolve(self, spec: ModelSpec) -> Tuple[str, str, str, str, str, Dict[str, str]]:
        """
        Returns (device, precision, runtime, engine, weights_digest, resolved_artifacts).
        """
        device = _pick_device(spec.device_policy)
        precision = str(spec.precision or "unknown").strip().lower()
        runtime = str(spec.runtime or "inprocess").strip().lower()
        engine = str(spec.engine or "unknown").strip()

        if runtime not in ("inprocess", "triton"):
            raise ModelManagerError(
                message="Unsupported runtime",
                error_code="unsupported_runtime",
                details={"runtime": runtime, "model_name": spec.model_name},
            )

        # Validate artifacts now (fail-fast).
        resolved_artifacts = _validate_local_artifacts(
            spec=spec,
            models_root=self.models_root,
            allow_absolute_paths=bool(self.cfg.allow_absolute_paths),
        )
        weights_digest = _compute_weights_digest(spec, resolved_artifacts)

        return device, precision, runtime, engine, weights_digest, resolved_artifacts

    def get(self, *, model_name: Optional[str] = None, role: Optional[str] = None, preferred_name: Optional[str] = None) -> ResolvedModel:
        spec = self.get_spec(model_name=model_name, role=role, preferred_name=preferred_name)
        device, precision, runtime, engine, weights_digest, resolved_artifacts = self.resolve(spec)

        cache_key = (
            spec.model_name,
            device,
            precision,
            runtime,
            engine,
            str(spec.preprocess_preset or ""),
            weights_digest,
        )

        with self._lock:
            hit = self._cache.get(cache_key)
            if hit is not None:
                # refresh LRU
                self._cache.move_to_end(cache_key)
                return hit

        # Load (outside lock to avoid long blocking)
        provider = self.providers.find_provider(spec)
        handle = provider.load(
            spec=spec,
            device=device,
            precision=precision,
            models_root=self.models_root,
            runtime_params=(spec.runtime_params or None),
        )
        mu = model_used(
            model_name=spec.model_name,
            model_version=spec.model_version,
            weights_digest=weights_digest,
            runtime=str(runtime),
            engine=engine,
            precision=precision,
            device=device,
        )
        resolved = ResolvedModel(
            spec=spec,
            device=device,
            precision=precision,
            runtime=str(runtime),
            engine=engine,
            weights_digest=weights_digest,
            resolved_artifacts=resolved_artifacts,
            handle=handle,
            models_used_entry=mu,
        )

        with self._lock:
            self._cache[cache_key] = resolved
            self._cache.move_to_end(cache_key)
            # LRU eviction
            while len(self._cache) > int(self.cfg.max_loaded_models):
                self._cache.popitem(last=False)

        return resolved

    def resolved_mapping_for_manifest(self) -> Dict[str, Any]:
        """
        Deterministic mapping intended to be saved in manifest / DB.
        (No model handles; only resolved params.)
        """
        out: Dict[str, Any] = {}
        # stable order
        for name in sorted(self.catalog.specs_by_name.keys()):
            spec = self.catalog.specs_by_name[name]
            device, precision, runtime, engine, weights_digest, _resolved_artifacts = self.resolve(spec)
            out[name] = {
                "model_name": spec.model_name,
                "model_version": spec.model_version,
                "role": spec.role,
                "runtime": runtime,
                "engine": engine,
                "precision": precision,
                "device": device,
                "weights_digest": weights_digest,
            }
        return out


_GLOBAL_MM: Optional[ModelManager] = None
_GLOBAL_LOCK = RLock()


def get_global_model_manager() -> ModelManager:
    global _GLOBAL_MM
    with _GLOBAL_LOCK:
        if _GLOBAL_MM is None:
            _GLOBAL_MM = ModelManager()
        return _GLOBAL_MM


