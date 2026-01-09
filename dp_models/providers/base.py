from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any, Dict, List, Optional, Protocol

from ..errors import ModelManagerError
from ..specs import ModelSpec


@dataclass(frozen=True)
class ResolvedModel:
    spec: ModelSpec
    device: str
    precision: str
    runtime: str
    engine: str
    weights_digest: str
    resolved_artifacts: Dict[str, str]
    handle: Any
    models_used_entry: Dict[str, Any]


class ModelProvider(Protocol):
    """
    Provider plugin contract.
    """

    def supports(self, spec: ModelSpec) -> bool: ...

    def load(
        self,
        *,
        spec: ModelSpec,
        device: str,
        precision: str,
        models_root: str,
        runtime_params: Optional[Dict[str, Any]] = None,
    ) -> Any: ...


class ProviderRegistry:
    def __init__(self) -> None:
        self._providers: List[ModelProvider] = []
        self._lock = RLock()

    def register(self, provider: ModelProvider) -> None:
        with self._lock:
            self._providers.append(provider)

    def find_provider(self, spec: ModelSpec) -> ModelProvider:
        with self._lock:
            for p in self._providers:
                try:
                    if p.supports(spec):
                        return p
                except Exception:
                    continue
        raise ModelManagerError(
            message="No provider supports this model spec",
            error_code="unsupported_engine",
            details={"model_name": spec.model_name, "engine": spec.engine, "runtime": spec.runtime, "role": spec.role},
        )


