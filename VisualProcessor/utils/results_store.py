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

    def store(self, result: Any, name: str) -> str:
        """Сохраняет результат в формате JSON и возвращает путь к файлу."""
        directory = self._build_dir(name)
        filename = self._generate_filename()
        filepath = os.path.join(directory, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, ensure_ascii=False)

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
