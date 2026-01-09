from __future__ import annotations

from typing import Dict, Tuple

from threading import RLock


_models: Dict[Tuple[str, str, bool], object] = {}
_lock = RLock()


def get_model(model_name: str, device: str, fp16: bool) -> object:
    """
    Return a shared SentenceTransformer instance for (model_name, device, fp16).

    Policy (PR-10):
    - NO network downloads at runtime.
    - Models must be resolved from local artifacts via ModelManager (dp_models).
    """
    global _models
    key = (model_name, device, bool(fp16))
    with _lock:
        if key in _models:
            return _models[key]

        # Fail-fast on invalid requested device.
        if "cuda" in str(device).lower():
            try:
                import torch  # type: ignore

                if not torch.cuda.is_available():
                    raise RuntimeError("CUDA requested but torch.cuda.is_available() is False")
            except Exception as e:
                raise RuntimeError(f"TextProcessor | CUDA requested but unavailable: device={device}") from e

        # Resolve via ModelManager (local-only, offline env enforced).
        try:
            from dp_models import get_global_model_manager  # type: ignore
        except Exception as e:
            raise RuntimeError(
                "TextProcessor | dp_models is required for embedding models (no-network policy). "
                "Ensure DataProcessor repo root is on PYTHONPATH."
            ) from e

        mm = get_global_model_manager()
        # Load model handle via ModelManager providers, but honor requested device/fp16.
        spec = mm.get_spec(model_name=str(model_name))

        # Validate artifacts / get weights digest. (We ignore mm-picked device here; device comes from caller.)
        _d, _p, runtime, engine, weights_digest, _arts = mm.resolve(spec)
        if str(runtime).lower() != "inprocess":
            raise RuntimeError(f"TextProcessor | unsupported runtime for embeddings: {runtime} (model={model_name})")
        precision = "fp16" if bool(fp16) else "fp32"

        provider = mm.providers.find_provider(spec)
        handle = provider.load(
            spec=spec,
            device=str(device),
            precision=str(precision),
            models_root=mm.models_root,
            runtime_params=(spec.runtime_params or None),
        )

        # Best-effort eval mode.
        try:
            handle.eval()  # type: ignore[attr-defined]
        except Exception:
            pass

        # Cache handle. We intentionally cache per (model_name, device, fp16) only;
        # weights_digest changes should be reflected via process restart / run config bump.
        _models[key] = handle
        return handle


def preload(models: Dict[Tuple[str, str, bool], None] | None = None) -> None:
    """
    Optionally preload a list of models given as (model_name, device, fp16) keys.
    """
    if not models:
        return
    for model_name, device, fp16 in models.keys():
        get_model(model_name, device, fp16)


