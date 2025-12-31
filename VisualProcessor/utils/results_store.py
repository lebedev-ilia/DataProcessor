# results_store.py
from __future__ import annotations

import datetime
import json
import os
import uuid
import tempfile
import shutil
import logging
from typing import Any, Dict, List, Optional, Tuple

from pathlib import Path

import numpy as np

LOGGER = logging.getLogger("ResultsStore")
LOGGER.addHandler(logging.NullHandler())


class ResultsStore:
    """
    Production-реализация хранилища результатов.

    - Хранит JSON-артефакты (каждый результат — отдельный файл).
    - Может сохранить агрегированный бинарный .npz (сжатый) через store_compressed.
    - Поддерживает атомарную запись (временный файл -> replace).
    - Предназначен для хранения per-track результатов: results: Dict[int, Dict].
    """

    def __init__(self, root_path: str) -> None:
        self.root_path = os.fspath(root_path)
        os.makedirs(self.root_path, exist_ok=True)

    @staticmethod
    def _timestamp_now() -> str:
        return datetime.datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S-%f")

    @staticmethod
    def _short_uuid() -> str:
        return uuid.uuid4().hex[:8]

    def _build_dir(self, name: str) -> str:
        """Создаёт (если нужно) и возвращает путь к поддиректории для группы результатов."""
        directory = os.path.join(self.root_path, name)
        os.makedirs(directory, exist_ok=True)
        return directory

    def _generate_filename(self, ext: str = "json") -> str:
        """Генерирует имя файла вида: 2025-01-13_12-50-03-524150_<uid>.json"""
        return f"{self._timestamp_now()}_{self._short_uuid()}.{ext}"

    def _atomic_write_text(self, path: str, text: str) -> None:
        """
        Пишем атомарно: записываем во временный файл той же директории, затем replace.
        Это защищает от частичных записей при сбоях.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=path.name, dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(text)
            os.replace(tmp, str(path))
        except Exception:
            try:
                os.remove(tmp)
            except Exception:
                pass
            raise

    def _atomic_write_bytes(self, path: str, data: bytes) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(prefix=path.name, dir=str(path.parent))
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
            os.replace(tmp, str(path))
        except Exception:
            try:
                os.remove(tmp)
            except Exception:
                pass
            raise

    def _to_json_serializable(self, obj: Any) -> Any:
        """
        Рекурсивно преобразует объекты в JSON-совместимые типы.
        Поддерживаются: numpy scalar/ndarray, datetime, uuid, list/tuple, dict,
        объекты с tolist()/__dict__.
        Не ломается на NaN/Inf — заменяет NaN -> None, Inf -> "inf"/"-inf".
        """
        import math

        # простые типы
        if obj is None or isinstance(obj, (str, bool, int)):
            return obj

        # float -> проверяем NaN/Inf
        if isinstance(obj, float):
            if math.isnan(obj):
                return None
            if math.isinf(obj):
                return "inf" if obj > 0 else "-inf"
            return obj

        # numpy
        try:
            import numpy as _np

            if isinstance(obj, (_np.integer,)):
                return int(obj)
            if isinstance(obj, (_np.floating,)):
                f = float(obj)
                if math.isnan(f):
                    return None
                if math.isinf(f):
                    return "inf" if f > 0 else "-inf"
                return f
            if isinstance(obj, _np.ndarray):
                # если массив числовой и небольшой — можем вернуть список;
                # если dtype == object — рекурсивно обрабатываем каждый элемент
                if obj.dtype == object:
                    return [self._to_json_serializable(x) for x in obj.tolist()]
                else:
                    # числовой — приводим к list
                    return obj.tolist()
        except Exception:
            # если numpy недоступен или вышло исключение — идём дальше
            pass

        # datetime
        if isinstance(obj, (datetime.datetime, datetime.date)):
            return obj.isoformat()

        # uuid
        if isinstance(obj, uuid.UUID):
            return str(obj)

        # списки, кортежи
        if isinstance(obj, (list, tuple)):
            return [self._to_json_serializable(x) for x in obj]

        # dict
        if isinstance(obj, dict):
            return {str(k): self._to_json_serializable(v) for k, v in obj.items()}

        # объекты с tolist()
        if hasattr(obj, "tolist"):
            try:
                return self._to_json_serializable(obj.tolist())
            except Exception:
                pass

        # объекты с __dict__
        if hasattr(obj, "__dict__"):
            try:
                return {k: self._to_json_serializable(v) for k, v in vars(obj).items()}
            except Exception:
                pass

        # fallback — string representation
        try:
            return repr(obj)
        except Exception:
            return None

    def store(self, result: Any, name: str) -> str:
        """
        Сохраняет произвольный результат в JSON в директории root_path/name.
        Возвращает путь к сохранённому файлу.
        """
        directory = self._build_dir(name)
        filename = self._generate_filename(ext="json")
        filepath = os.path.join(directory, filename)

        try:
            serial = self._to_json_serializable(result)
            text = json.dumps(serial, ensure_ascii=False, indent=2)
            self._atomic_write_text(filepath, text)
            LOGGER.debug("store: сохранено %s", filepath)
            return filepath
        except Exception as e:
            LOGGER.exception("store: не удалось сохранить JSON в %s: %s", filepath, e)
            raise

    def store_compressed(
        self,
        results: Dict[int, Dict[str, Any]],
        out_path: Optional[str] = None,
        *,
        name: Optional[str] = None,
        embeddings_key: str = "embedding_normed_256d",
        meta: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Сохраняет словарь results (per-track) в сжатый .npz файл.

        Параметры:
         - results: Dict[int, Dict] — ключи: track_id -> результаты (словарь)
         - out_path: путь к .npz файлу. Если None — будет сгенерирован под root_path/name/
         - name: если out_path не указан — используется name для директории
         - embeddings_key: ключ в per-track словаре, где лежат эмбеддинги (list или ndarray)
         - meta: дополнительная metadata (словарь)

        Возвращает путь к сохранённому .npz файлу.
        """

        if not isinstance(results, dict) or not results:
            raise ValueError("store_compressed: ожидается непустой словарь results")

        # определяем директорию/имя файла
        if out_path:
            out_path = str(out_path)
            os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
            target_path = out_path
        else:
            if not name:
                name = "compressed_results"
            directory = self._build_dir(name)
            fname = f"results_{self._timestamp_now()}_{self._short_uuid()}.npz"
            target_path = os.path.join(directory, fname)

        # сортируем track_ids для детерминированности
        track_ids = sorted(int(k) for k in results.keys())

        embeddings_list: List[np.ndarray] = []
        # метрики — гибкий набор. Собираем те, что найдём.
        metrics_accumulators: Dict[str, List] = {}
        json_per_track: List[Dict[str, Any]] = []

        for tid in track_ids:
            r = results[int(tid)]

            # --- эмбеддинги ---
            emb = r.get(embeddings_key, None)
            if emb is None:
                # пустой массив формы (0, embedding_dim) — попробуем угадать dim из параметра
                embedding_dim = int(r.get("embedding_dim", 256))
                emb_arr = np.zeros((0, embedding_dim), dtype=np.float32)
            else:
                # приводим к ndarray float32 если возможно
                try:
                    emb_arr = np.asarray(emb, dtype=np.float32)
                    # если одномерный вектор — превращаем в (T=1, D)
                    if emb_arr.ndim == 1:
                        emb_arr = emb_arr[np.newaxis, :]
                except Exception:
                    # на случай ragged list -> создаём object array позже; временно сериализуем строковым представлением
                    emb_arr = np.asarray([], dtype=np.float32)

            embeddings_list.append(emb_arr)

            # --- собираем все числовые метрики в отдельные массивы ---
            # проходим по всем ключам, если значение скаляр — добавляем в метрику
            for k, v in r.items():
                # пропускаем поле с эмбеддингами — оно отдельно
                if k == embeddings_key:
                    continue
                # принимаем скаляры (int/float/numpy scalars)
                if isinstance(v, (int, float, np.integer, np.floating)):
                    metrics_accumulators.setdefault(k, []).append(v)
                else:
                    # для других типов аккумулируем None, чтобы длина совпадала
                    metrics_accumulators.setdefault(k, []).append(None)

            # --- JSON-представление per-track (удобно для быстрого просмотра) ---
            try:
                json_per_track.append(self._to_json_serializable(r))
            except Exception:
                json_per_track.append({"error": "to_json failed", "track_id": tid})

        # --- Собираем метрики в numpy-массивы (float32 / int32 / object) ---
        npz_dict: Dict[str, Any] = {}
        npz_dict["tracks"] = np.asarray(track_ids, dtype=np.int32)

        # embeddings -> если все emb_arr имеют ndim == 2 и одинаковая D, то можем сохранить 3D padded
        # иначе — сохраняем как object array
        all_have_2d = all((e.ndim == 2) for e in embeddings_list)
        dims = [e.shape[1] if e.ndim == 2 and e.shape[0] > 0 else None for e in embeddings_list]
        unique_dims = set(d for d in dims if d is not None)

        if all_have_2d and (len(unique_dims) == 1 or all(d is None for d in dims)):
            # можем сохранить как object array anyway (универсально) — делаем object array
            emb_obj = np.empty((len(embeddings_list),), dtype=object)
            for i, e in enumerate(embeddings_list):
                emb_obj[i] = e
            npz_dict["embeddings"] = emb_obj
        else:
            # ragged / mixed -> object array
            emb_obj = np.empty((len(embeddings_list),), dtype=object)
            for i, e in enumerate(embeddings_list):
                emb_obj[i] = e
            npz_dict["embeddings"] = emb_obj

        # метрики — приводим к массивам; если значение None -> np.nan
        for metric_name, values in metrics_accumulators.items():
            # пытаемся привести к числовому массиву
            vals_clean: List[float] = []
            is_int = all(isinstance(x, (int, np.integer)) or x is None for x in values)
            if is_int:
                # используем int32, заменяем None -> -1
                arr = np.array([int(x) if x is not None else -1 for x in values], dtype=np.int32)
            else:
                # float array, None -> np.nan
                arr = np.array([float(x) if x is not None else np.nan for x in values], dtype=np.float32)
            # уникальное имя метрики в npz
            sanitized_name = f"metric__{metric_name}"
            npz_dict[sanitized_name] = arr

        # meta
        meta = dict(meta or {})
        meta.setdefault("producer", "ResultsStore")
        meta.setdefault("created_at", datetime.datetime.utcnow().isoformat())
        meta.setdefault("embeddings_key", embeddings_key)
        npz_dict["meta"] = np.asarray(meta, dtype=object)

        # дополнительный JSON-представление каждого track (object array)
        json_arr = np.empty((len(json_per_track),), dtype=object)
        for i, j in enumerate(json_per_track):
            json_arr[i] = j
        npz_dict["results_json"] = json_arr

        # Сохраняем атомарно в .npz
        try:
            # сначала пишем в временную директорию (в том же каталоге), затем replace
            target = Path(target_path)
            target.parent.mkdir(parents=True, exist_ok=True)
            tmp_fd, tmp_path = tempfile.mkstemp(prefix=target.name + ".", suffix=".npz", dir=str(target.parent))
            os.close(tmp_fd)  # будем использовать numpy для записи
            try:
                # numpy.savez_compressed пишет файл полностью
                np.savez_compressed(tmp_path, **npz_dict)
                os.replace(tmp_path, str(target_path))
            except Exception:
                # cleanup tmp and rethrow
                try:
                    os.remove(tmp_path)
                except Exception:
                    pass
                raise
            LOGGER.info("store_compressed: сохранено %s (tracks=%d)", target_path, len(track_ids))
            return target_path
        except Exception as e:
            LOGGER.exception("store_compressed: не удалось сохранить %s: %s", target_path, e)
            raise

    def list(self, name: str) -> List[str]:
        """Возвращает имена файлов (не полные пути) в директории root_path/name."""
        directory = self._build_dir(name)
        try:
            return sorted(os.listdir(directory))
        except FileNotFoundError:
            return []

    def read(self, name: str, filename: str) -> Optional[Any]:
        """Читает JSON-файл из root_path/name/filename и возвращает распарсенный объект."""
        path = os.path.join(self.root_path, name, filename)
        if not os.path.exists(path):
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            LOGGER.exception("read: не удалось прочитать %s", path)
            raise

    def read_compressed(self, path: str) -> Dict[str, Any]:
        """
        Читает .npz файл, возвращает словарь с ключами и десериализованными значениями.
        Для object-массивов возвращает их как списки/ndarray.
        """
        if not os.path.exists(path):
            raise FileNotFoundError(path)
        try:
            loaded = np.load(path, allow_pickle=True)
            out: Dict[str, Any] = {}
            for k in loaded.files:
                out[k] = loaded[k]
            return out
        except Exception:
            LOGGER.exception("read_compressed: не удалось прочитать %s", path)
            raise

    def cleanup(self, name: str, keep: int = 100) -> None:
        """
        Оставляет только 'keep' последних файлов (по имени/таймстемпу) в секции name.
        Удаляет старые.
        """
        directory = self._build_dir(name)
        try:
            files = sorted(os.listdir(directory))
            if len(files) <= keep:
                return
            old_files = files[:-keep]
            for f in old_files:
                p = os.path.join(directory, f)
                try:
                    os.remove(p)
                except FileNotFoundError:
                    pass
        except Exception:
            LOGGER.exception("cleanup: ошибка при очистке %s", directory)
            raise

    def remove(self, name: str, filename: str) -> bool:
        """Удаляет конкретный файл (возвращает True если удалено)."""
        path = os.path.join(self.root_path, name, filename)
        if not os.path.exists(path):
            return False
        try:
            os.remove(path)
            return True
        except Exception:
            LOGGER.exception("remove: не удалось удалить %s", path)
            raise
