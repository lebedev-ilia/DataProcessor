from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class ModelManagerError(RuntimeError):
    """
    Standard structured error for model-related failures.
    """

    message: str
    error_code: str = "model_error"
    details: Optional[Dict[str, Any]] = None

    def __str__(self) -> str:  # pragma: no cover
        if self.details:
            return f"{self.error_code}: {self.message} | details={self.details}"
        return f"{self.error_code}: {self.message}"


