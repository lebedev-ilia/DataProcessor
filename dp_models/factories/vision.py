from __future__ import annotations

from typing import Any


def create_torchvision_resnet(*, model_name: str, num_classes: int) -> Any:
    """
    Create a torchvision ResNet without any pretrained weights.
    Intended to be used from TorchStateDictProvider factory.
    """
    try:
        from torchvision import models  # type: ignore
    except Exception as e:
        raise RuntimeError(f"torchvision is not installed: {e}") from e

    name = str(model_name).strip().lower()
    if not hasattr(models, name):
        raise RuntimeError(f"torchvision.models has no '{name}'")
    constructor = getattr(models, name)

    # Newer torchvision: supports weights=None and num_classes
    try:
        return constructor(weights=None, num_classes=int(num_classes))
    except TypeError:
        m = constructor(weights=None)
        # ensure classifier head is adjusted (ResNet has .fc)
        if not hasattr(m, "fc"):
            raise RuntimeError(f"Model '{name}' does not expose an 'fc' attribute")
        import torch  # type: ignore

        in_features = int(m.fc.in_features)
        m.fc = torch.nn.Linear(in_features, int(num_classes))
        return m


def create_timm_model(*, model_name: str, num_classes: int) -> Any:
    """
    Create a timm model without pretrained weights.
    Intended to be used from TorchStateDictProvider factory.
    """
    try:
        import timm  # type: ignore
    except Exception as e:
        raise RuntimeError(f"timm is not installed: {e}") from e
    return timm.create_model(str(model_name), pretrained=False, num_classes=int(num_classes))


