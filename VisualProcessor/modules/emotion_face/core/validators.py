"""
Валидаторы для входных данных.
"""
from pathlib import Path
from typing import Optional
from core.exceptions import ConfigurationValidationError


def validate_processing_params(params) -> None:
    """
    Валидирует параметры обработки.
    
    Args:
        params: Параметры обработки.
    
    Raises:
        ConfigurationValidationError: Если параметры некорректны.
    """
    from core.processing_config import ProcessingParams
    
    # Валидация уже выполняется в __post_init__, но можно добавить дополнительные проверки
    if not isinstance(params, ProcessingParams):
        raise ConfigurationValidationError(
            f"params must be ProcessingParams instance, got {type(params)}",
            details={"params_type": str(type(params))}
        )


def validate_target_length(target_length: Optional[int]) -> int:
    """
    Валидирует и нормализует target_length.
    
    Args:
        target_length: Целевая длина последовательности.
        default: Значение по умолчанию.
    
    Returns:
        Валидированное значение target_length.
    
    Raises:
        ConfigurationValidationError: Если значение некорректно.
    """
    if not isinstance(target_length, int):
        raise ConfigurationValidationError(
            f"target_length must be integer, got {type(target_length)}",
            details={"target_length": target_length, "type": str(type(target_length))}
        )
    
    if target_length <= 0:
        raise ConfigurationValidationError(
            f"target_length must be positive, got {target_length}",
            details={"target_length": target_length}
        )
    
    if target_length > 10000:
        raise ConfigurationValidationError(
            f"target_length too large: {target_length} (max 10000)",
            details={"target_length": target_length}
        )
    
    return target_length

