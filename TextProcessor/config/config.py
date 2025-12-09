from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

try:
    from dotenv import load_dotenv  # type: ignore
    _DOTENV_AVAILABLE = True
except Exception:
    _DOTENV_AVAILABLE = False


def _str_to_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _load_env() -> None:
    if _DOTENV_AVAILABLE:
        try:
            load_dotenv()
        except Exception:
            pass


@dataclass
class TitleEmbedderConfig:
    model_name: str = "sentence-transformers/all-mpnet-base-v2"
    cache_dir: str = "/home/ilya/Рабочий стол/DataProcessor/TextProcessor/.cache/embed_cache"
    fp16: bool = False
    batch_size: int = 128
    artifacts_dir: str = "/home/ilya/Рабочий стол/DataProcessor/TextProcessor/.artifacts"


@dataclass
class AppConfig:
    devices_config: Dict[str, Union[str, List[str]]]
    title_embedder: TitleEmbedderConfig
    default_input_path: Optional[str]


def load_config() -> AppConfig:
    _load_env()

    # Devices mapping
    devices_config_raw = os.environ.get("DEVICES_CONFIG", "")
    if devices_config_raw:
        try:
            devices_config: Dict[str, Union[str, List[str]]] = json.loads(devices_config_raw)
        except Exception:
            devices_config = {"gpu": "TitleEmbedder"}
    else:
        devices_config = {"gpu": "TitleEmbedder"}

    # TitleEmbedder params
    model_name = os.environ.get("TITLE_EMBEDDER_MODEL_NAME", "sentence-transformers/all-mpnet-base-v2")
    cache_dir = os.environ.get("TITLE_EMBEDDER_CACHE_DIR", "/home/ilya/Рабочий стол/DataProcessor/TextProcessor/.cache/embed_cache")
    fp16 = _str_to_bool(os.environ.get("TITLE_EMBEDDER_FP16", "false"))
    batch_size = int(os.environ.get("TITLE_EMBEDDER_BATCH_SIZE", "128"))
    artifacts_dir = os.environ.get("ARTIFACTS_DIR", "/home/ilya/Рабочий стол/DataProcessor/TextProcessor/.artifacts")

    default_input_path = os.environ.get("DEFAULT_INPUT_PATH", None)

    return AppConfig(
        devices_config=devices_config,
        title_embedder=TitleEmbedderConfig(
            model_name=model_name,
            cache_dir=cache_dir,
            fp16=fp16,
            batch_size=batch_size,
            artifacts_dir=artifacts_dir,
        ),
        default_input_path=default_input_path,
    )


