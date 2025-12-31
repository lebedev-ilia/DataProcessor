from __future__ import annotations

from typing import Dict, Tuple, Optional

from threading import RLock


_models: Dict[Tuple[str, str, bool], object] = {}
_lock = RLock()


def get_model(model_name: str, device: str, fp16: bool) -> object:
    """
    Return a shared SentenceTransformer instance for (model_name, device, fp16).
    Lazily creates and caches the model.
    """
    global _models
    key = (model_name, device, bool(fp16))
    with _lock:
        if key in _models:
            return _models[key]
        # Local import to avoid hard dependency at module import time
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(model_name, device=device)
        if fp16 and ("cuda" in device):
            try:
                model = model.half()
            except Exception:
                pass
        model.eval()
        _models[key] = model
        return model


def preload(models: Dict[Tuple[str, str, bool], None] | None = None) -> None:
    """
    Optionally preload a list of models given as (model_name, device, fp16) keys.
    """
    if not models:
        return
    for model_name, device, fp16 in models.keys():
        get_model(model_name, device, fp16)


