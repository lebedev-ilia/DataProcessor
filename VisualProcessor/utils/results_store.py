import datetime
import json
import os
import uuid
from typing import Any, List, Optional


class ResultsStore:
    """
    Удобный класс для хранения результатов обработки.
    - Каждый результат сохраняется в отдельный JSON.
    - Файлы именуются timestamp + уникальный id.
    - Можно перечитывать, получать список, удалять старые.
    """

    def __init__(self, root_path: str) -> None:
        self.root_path = root_path
        os.makedirs(self.root_path, exist_ok=True)

    def _build_dir(self, name: str) -> str:
        """Создаёт директорию для указанной группы результатов."""
        path = os.path.join(self.root_path, name)
        os.makedirs(path, exist_ok=True)
        return path

    def _generate_filename(self) -> str:
        """Генерирует файл в формате: 2025-01-13_12-50-03-524150_UUID.json"""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S-%f")
        uid = uuid.uuid4().hex[:8]
        return f"{timestamp}_{uid}.json"

    def _to_json_serializable(self, obj: Any) -> Any:
        """
        Преобразует объект в JSON-совместимый формат.
        Поддержка: numpy, datetime, uuid, списки, dict, NaN, Inf, кастомные объекты.
        """
        import math

        # ---- None, str, bool, int ----
        if obj is None or isinstance(obj, (str, bool, int)):
            return obj

        # ---- float + обработка NaN/Inf ----
        if isinstance(obj, float):
            if math.isnan(obj):
                return None              # JSON-compatible
            if math.isinf(obj):
                return "inf" if obj > 0 else "-inf"
            return obj

        # ---- numpy ----
        try:
            import numpy as np

            if isinstance(obj, (np.integer,)):
                return int(obj)

            if isinstance(obj, (np.floating,)):
                f = float(obj)
                if math.isnan(f):
                    return None
                if math.isinf(f):
                    return "inf" if f > 0 else "-inf"
                return f

            if isinstance(obj, np.ndarray):
                arr = obj.astype(object).tolist()  # сохраним структуру
                return self._to_json_serializable(arr)
        except ImportError:
            pass

        # ---- datetime ----
        if isinstance(obj, (datetime.datetime, datetime.date)):
            return obj.isoformat()

        # ---- uuid ----
        if isinstance(obj, uuid.UUID):
            return str(obj)

        # ---- списки / tuples ----
        if isinstance(obj, (list, tuple)):
            return [self._to_json_serializable(x) for x in obj]

        # ---- dict ----
        if isinstance(obj, dict):
            return {str(k): self._to_json_serializable(v) for k, v in obj.items()}

        # ---- объекты с __dict__ ----
        if hasattr(obj, "__dict__"):
            return {k: self._to_json_serializable(v) for k, v in obj.__dict__.items()}

        # ---- объекты с tolist() ----
        if hasattr(obj, "tolist"):
            try:
                return self._to_json_serializable(obj.tolist())
            except Exception:
                pass

        # ---- fallback ----
        return repr(obj)

    def store(self, result: Any, name: str) -> str:
        """Сохраняет результат в формате JSON и возвращает путь к файлу."""
        directory = self._build_dir(name)
        filename = self._generate_filename()
        filepath = os.path.join(directory, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self._to_json_serializable(result), f, indent=2, ensure_ascii=False)

        return filepath

    def list(self, name: str) -> List[str]:
        """Возвращает список всех файлов в категории."""
        directory = self._build_dir(name)
        return sorted(os.listdir(directory))

    def read(self, name: str, filename: str) -> Optional[Any]:
        """Читает конкретный JSON-файл."""
        path = os.path.join(self.root_path, name, filename)
        if not os.path.exists(path):
            return None
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def cleanup(self, name: str, keep: int = 100) -> None:
        """
        Очищает старые файлы, оставляя только 'keep' последних.
        Например: keep=50 → оставить 50 последних результатов.
        """
        directory = self._build_dir(name)
        files = sorted(os.listdir(directory))

        if len(files) <= keep:
            return

        old_files = files[:-keep]
        for f in old_files:
            try:
                os.remove(os.path.join(directory, f))
            except FileNotFoundError:
                pass
