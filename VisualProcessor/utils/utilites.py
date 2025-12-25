import os
import sys
import json

def load_metadata(meta_path: str, name) -> dict:
    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise RuntimeError(f"{name} | load_metadata | Ошибка при открытии {meta_path}: {e}")