"""
Базовый класс для всех модулей VisualProcessor.

Обеспечивает единый интерфейс и стандартизированную работу с:
- Обработкой видео (frame_manager, frame_indices, config)
- Загрузкой результатов других модулей
- Сохранением результатов в np.savez_compressed
- Логированием и обработкой ошибок
"""

from __future__ import annotations

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from abc import ABC, abstractmethod
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
from pathlib import Path

import numpy as np

from utils.frame_manager import FrameManager
from utils.results_store import ResultsStore
from utils.logger import get_logger
from utils.utilites import load_metadata
import json
import uuid
import re


class BaseModule(ABC):
    """
    Базовый класс для всех модулей обработки видео.
    
    Каждый модуль должен:
    1. Наследоваться от BaseModule
    2. Реализовать метод process(frame_manager, frame_indices, config)
    3. Объявить зависимости через required_dependencies() (если нужны)
    4. Вернуть результаты в формате, готовом для сохранения в npz
    """
    
    def __init__(
        self,
        rs_path: Optional[str] = None,
        logger_name: Optional[str] = None,
        **kwargs: Any
    ):
        """
        Инициализация базового модуля.
        
        Args:
            rs_path: Путь к хранилищу результатов (для загрузки данных других модулей)
            logger_name: Имя для logger (по умолчанию - имя класса)
            **kwargs: Дополнительные параметры для конкретных модулей
        """
        self.rs_path = rs_path
        self._explicit_logger_name = logger_name is not None
        self.logger_name = logger_name or self.__class__.__name__
        self.logger = get_logger(self.logger_name)
        self._results_store: Optional[ResultsStore] = None
        self._initialized = False

        self._init_module(**kwargs)
    
    @property
    def module_name(self) -> str:
        """
        Имя модуля (используется для сохранения результатов).
        По умолчанию - имя класса в нижнем регистре без суффикса "Module".
        """
        # 1) Explicit override by subclasses
        forced = getattr(self, "MODULE_NAME", None)
        if isinstance(forced, str) and forced:
            return forced

        # 2) If caller explicitly provided logger_name, treat it as canonical module name (common in this repo).
        if self._explicit_logger_name and isinstance(self.logger_name, str) and self.logger_name:
            return self.logger_name

        # 3) Otherwise: derive snake_case from class name (ShotQualityModule -> shot_quality).
        name = self.__class__.__name__
        for suffix in ("Module", "Processor", "Analyzer", "Pipeline"):
            if name.endswith(suffix):
                name = name[: -len(suffix)]
                break
        s1 = re.sub("(.)([A-Z][a-z]+)", r"\1_\2", name)
        snake = re.sub("([a-z0-9])([A-Z])", r"\1_\2", s1).lower()
        return snake
    
    @property
    def results_store(self) -> ResultsStore:
        """Ленивая инициализация ResultsStore."""
        if self._results_store is None:
            if self.rs_path is None:
                raise ValueError(
                    f"{self.module_name}: rs_path не указан. "
                    "Необходим для сохранения/загрузки результатов."
                )
            self._results_store = ResultsStore(self.rs_path)
        return self._results_store
    
    def _init_module(self, **kwargs: Any) -> None:
        """
        Внутренний метод инициализации модуля.
        Переопределяется в подклассах для загрузки моделей и т.д.
        """
        pass
    
    def initialize(self) -> None:
        """
        Публичный метод инициализации (загрузка моделей и т.д.).
        Вызывается автоматически перед первым использованием process().
        """
        if not self._initialized:
            self._do_initialize()
            self._initialized = True
    
    def _do_initialize(self) -> None:
        """
        Внутренний метод инициализации. Переопределяется в подклассах.
        """
        pass
    
    @abstractmethod
    def process(
        self,
        frame_manager: FrameManager,
        frame_indices: List[int],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Основной метод обработки видео.
        
        Args:
            frame_manager: Менеджер кадров для доступа к кадрам
            frame_indices: Список индексов кадров для обработки
            config: Конфигурация модуля (параметры из YAML/CLI)
            
        Returns:
            Словарь с результатами обработки. Структура должна быть готова
            для сохранения в np.savez_compressed. Может содержать:
            - numpy массивы (embeddings, features)
            - метрики (скаляры, списки)
            - метаданные
            
        Raises:
            ValueError: Если входные данные некорректны
            RuntimeError: Если обработка не удалась
        """
        pass
    
    def required_dependencies(self) -> List[str]:
        """
        Возвращает список имен модулей, результаты которых необходимы для работы.
        
        Примеры:
            - ["core_clip"] - нужны эмбеддинги CLIP
            - ["core_face_landmarks", "core_object_detections"] - нужны landmarks и детекции
            - [] - модуль не зависит от других модулей
            
        Returns:
            Список имен модулей-зависимостей
        """
        return []
    
    def load_dependency_results(
        self,
        module_name: str,
        format: str = "auto"
    ) -> Optional[Dict[str, Any]]:
        """
        Загружает результаты указанного модуля.
        
        Args:
            module_name: Имя модуля-зависимости
            format: Формат загрузки:
                - "auto" - автоматически определяет (npz или json)
                - "npz" - загружает из .npz файла
                - "json" - загружает из .json файла
                
        Returns:
            Словарь с результатами модуля или None, если не найдено
            
        Raises:
            FileNotFoundError: Если модуль обязателен (через required_dependencies)
        """
        if self.rs_path is None:
            self.logger.warning(
                f"{self.module_name} | rs_path не указан, "
                f"не могу загрузить результаты {module_name}"
            )
            return None
        
        module_dir = os.path.join(self.rs_path, module_name)
        if not os.path.exists(module_dir):
            if module_name in self.required_dependencies():
                raise FileNotFoundError(
                    f"{self.module_name} | Обязательная зависимость '{module_name}' "
                    f"не найдена в {module_dir}"
                )
            self.logger.debug(
                f"{self.module_name} | Модуль '{module_name}' не найден (опциональная зависимость)"
            )
            return None
        
        # Автоматическое определение формата
        if format == "auto":
            # Ищем npz файлы
            npz_files = list(Path(module_dir).glob("*.npz"))
            if npz_files:
                # Берем последний по времени модификации
                npz_file = max(npz_files, key=lambda p: p.stat().st_mtime)
                return self._load_npz(str(npz_file))
            
            # Ищем json файлы
            json_files = list(Path(module_dir).glob("*.json"))
            if json_files:
                json_file = max(json_files, key=lambda p: p.stat().st_mtime)
                return self._load_json(str(json_file))
            
            self.logger.warning(
                f"{self.module_name} | Не найдено файлов результатов для {module_name}"
            )
            return None
        
        elif format == "npz":
            npz_files = list(Path(module_dir).glob("*.npz"))
            if not npz_files:
                raise FileNotFoundError(
                    f"{self.module_name} | Не найдено .npz файлов для {module_name}"
                )
            npz_file = max(npz_files, key=lambda p: p.stat().st_mtime)
            return self._load_npz(str(npz_file))
        
        elif format == "json":
            json_files = list(Path(module_dir).glob("*.json"))
            if not json_files:
                raise FileNotFoundError(
                    f"{self.module_name} | Не найдено .json файлов для {module_name}"
                )
            json_file = max(json_files, key=lambda p: p.stat().st_mtime)
            return self._load_json(str(json_file))
        
        else:
            raise ValueError(f"Неизвестный формат: {format}")
    
    def _load_npz(self, path: str) -> Dict[str, Any]:
        """Загружает данные из .npz файла."""
        try:
            loaded = np.load(path, allow_pickle=True)
            result: Dict[str, Any] = {}
            for key in loaded.files:
                value = loaded[key]
                # Unbox scalar object arrays (common for dict-like payloads: detections/features/meta)
                if isinstance(value, np.ndarray) and value.dtype == object and value.shape == ():
                    try:
                        result[key] = value.item()
                    except Exception:
                        result[key] = value
                else:
                    result[key] = value
            self.logger.debug(f"{self.module_name} | Загружен npz: {path}")
            return result
        except Exception as e:
            self.logger.exception(
                f"{self.module_name} | Ошибка загрузки npz {path}: {e}"
            )
            raise
    
    def _load_json(self, path: str) -> Dict[str, Any]:
        """Загружает данные из .json файла."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                result = json.load(f)
            self.logger.debug(f"{self.module_name} | Загружен json: {path}")
            return result
        except Exception as e:
            self.logger.exception(
                f"{self.module_name} | Ошибка загрузки json {path}: {e}"
            )
            raise
    
    def load_core_provider(
        self,
        provider_name: str,
        file_name: Optional[str] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Загружает результаты core провайдера.
        
        Core провайдеры сохраняют результаты в корне rs_path, а не в поддиректории.
        Например: core_object_detections сохраняет в rs_path/detections.npz
        
        Args:
            provider_name: Имя core провайдера (например, "core_object_detections")
            file_name: Имя файла (если None, ищет автоматически: detections.npz, embeddings.npz и т.д.)
            
        Returns:
            Словарь с результатами или None, если не найдено
        """
        if self.rs_path is None:
            self.logger.warning(
                f"{self.module_name} | rs_path не указан, "
                f"не могу загрузить результаты {provider_name}"
            )
            return None
        
        # Production standard: core providers store artifacts under rs_path/<provider_name>/...
        # Legacy: some providers stored in rs_path root. We support both for backward compatibility.
        provider_dir = os.path.join(self.rs_path, provider_name)

        def try_load(path: str) -> Optional[Dict[str, Any]]:
            if not os.path.exists(path):
                return None
            if path.endswith(".npz"):
                return self._load_npz(path)
            if path.endswith(".json"):
                return self._load_json(path)
            return None

        if file_name:
            # Try provider subdir first, then root (legacy)
            out = try_load(os.path.join(provider_dir, file_name))
            if out is not None:
                return out
            out = try_load(os.path.join(self.rs_path, file_name))
            if out is not None:
                return out
        else:
            # Auto-detect: try common names in provider subdir, then root (legacy).
            common_names = [
                "detections.npz",  # core_object_detections
                "embeddings.npz",  # core_clip
                "landmarks.npz",   # core_face_landmarks
                "depth.npz",       # core_depth_midas
            ]
            
            for fname in common_names:
                out = try_load(os.path.join(provider_dir, fname))
                if out is not None:
                    return out

            for fname in common_names:
                out = try_load(os.path.join(self.rs_path, fname))
                if out is not None:
                    return out

            # Last resort: any npz in provider_dir, then root (legacy)
            provider_npz = list(Path(provider_dir).glob("*.npz")) if os.path.exists(provider_dir) else []
            if provider_npz:
                npz_file = max(provider_npz, key=lambda p: p.stat().st_mtime)
                return self._load_npz(str(npz_file))
            
            root_npz = list(Path(self.rs_path).glob("*.npz"))
            if root_npz:
                npz_file = max(root_npz, key=lambda p: p.stat().st_mtime)
                return self._load_npz(str(npz_file))
        
        self.logger.warning(
            f"{self.module_name} | Не найдено результатов для core провайдера {provider_name}"
        )
        return None
    
    def load_all_dependencies(self) -> Dict[str, Optional[Dict[str, Any]]]:
        """
        Загружает результаты всех зависимостей.
        
        Поддерживает как обычные модули, так и core провайдеры.
        
        Returns:
            Словарь {module_name: results} для каждой зависимости
        """
        dependencies = {}
        for module_name in self.required_dependencies():
            try:
                # Проверяем, является ли это core провайдером
                if module_name.startswith("core_"):
                    dependencies[module_name] = self.load_core_provider(module_name)
                else:
                    dependencies[module_name] = self.load_dependency_results(module_name)
            except FileNotFoundError as e:
                self.logger.error(f"{self.module_name} | {e}")
                raise
        return dependencies
    
    def save_results(
        self,
        results: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
        use_compressed: bool = True,
        embeddings_key: Optional[str] = None
    ) -> str:
        """
        Сохраняет результаты модуля в np.savez_compressed.
        
        Args:
            results: Результаты обработки
            metadata: Дополнительные метаданные (total_frames, producer, и т.д.)
            use_compressed: Использовать ResultsStore.store_compressed() для per-track результатов
            embeddings_key: Ключ для эмбеддингов (если use_compressed=True)
            
        Returns:
            Путь к сохраненному файлу
        """
        if self.rs_path is None:
            raise ValueError(f"{self.module_name} | rs_path не указан для сохранения результатов")
        
        # Подготовка метаданных
        save_meta = dict(metadata or {})
        save_meta.setdefault("producer", self.module_name)
        save_meta.setdefault("created_at", datetime.utcnow().isoformat())
        # Baseline meta contract fields (best-effort defaults; run() should populate the identity keys).
        save_meta.setdefault("producer_version", getattr(self, "VERSION", None) or getattr(self, "producer_version", None) or "unknown")
        save_meta.setdefault("schema_version", getattr(self, "SCHEMA_VERSION", None) or f"{self.module_name}_npz_v1")
        save_meta.setdefault("status", "ok")
        save_meta.setdefault("empty_reason", None)
        
        # Определяем, нужно ли использовать store_compressed
        # (для per-track результатов с эмбеддингами)
        if use_compressed and isinstance(results, dict):
            # Проверяем, является ли это per-track результатом
            # (словарь с числовыми ключами и вложенными словарями)
            is_per_track = (
                all(isinstance(k, int) for k in results.keys()) and
                all(isinstance(v, dict) for v in results.values())
            )
            
            if is_per_track and embeddings_key:
                # Используем store_compressed для per-track результатов
                core_dir = os.path.join(self.rs_path, self.module_name)
                os.makedirs(core_dir, exist_ok=True)
                npz_path = os.path.join(core_dir, f"{self.module_name}_emb.npz")
                
                return self.results_store.store_compressed(
                    results=results,
                    out_path=npz_path,
                    embeddings_key=embeddings_key,
                    meta=save_meta
                )
        
        # Стандартное сохранение через np.savez_compressed
        # Production: timestamped artifacts (easier debugging/audit, supports multiple runs).
        core_dir = os.path.join(self.rs_path, self.module_name)
        os.makedirs(core_dir, exist_ok=True)
        ts = datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S-%f")
        uid = uuid.uuid4().hex[:8]
        npz_path = os.path.join(core_dir, f"{self.module_name}_features_{ts}_{uid}.npz")
        
        # Подготовка данных для сохранения
        npz_dict: Dict[str, Any] = {}
        
        # Рекурсивно обрабатываем результаты
        for key, value in results.items():
            if isinstance(value, (np.ndarray, np.generic)):
                npz_dict[key] = value
            elif isinstance(value, (list, tuple)):
                # Преобразуем списки в numpy массивы
                try:
                    npz_dict[key] = np.asarray(value)
                except Exception:
                    # Если не удалось - сохраняем как object array
                    npz_dict[key] = np.asarray(value, dtype=object)
            elif isinstance(value, dict):
                # Вложенные словари сохраняем как object
                npz_dict[key] = np.asarray(value, dtype=object)
            elif isinstance(value, (int, float, str, bool)):
                # Скаляры сохраняем как numpy типы
                npz_dict[key] = np.asarray(value)
            else:
                # Остальное - как object
                npz_dict[key] = np.asarray(value, dtype=object)
        
        # Добавляем метаданные
        npz_dict["meta"] = np.asarray(save_meta, dtype=object)
        
        # Атомарное сохранение
        import tempfile
        tmp_path = None
        try:
            fd, tmp_path = tempfile.mkstemp(
                prefix=f"{Path(npz_path).name}.",
                suffix=".npz",
                dir=core_dir,
            )
            os.close(fd)
            np.savez_compressed(tmp_path, **npz_dict)
            os.replace(tmp_path, npz_path)
        except Exception as e:
            self.logger.exception(
                f"{self.module_name} | Ошибка сохранения результатов: {e}"
            )
            raise
        finally:
            # cleanup tmp if something went wrong before replace
            try:
                if tmp_path and os.path.exists(tmp_path) and tmp_path != npz_path:
                    os.remove(tmp_path)
            except Exception:
                pass

        self.logger.info(f"{self.module_name} | Результаты сохранены: {npz_path}")
        return npz_path
    
    def load_metadata(self, frames_dir: str) -> Dict[str, Any]:
        """
        Загружает метаданные из metadata.json в frames_dir.
        
        Args:
            frames_dir: Директория с кадрами (должна содержать metadata.json)
            
        Returns:
            Словарь с метаданными
            
        Raises:
            FileNotFoundError: Если metadata.json не найден
            ValueError: Если метаданные некорректны
        """
        meta_path = os.path.join(frames_dir, "metadata.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(
                f"{self.module_name} | metadata.json не найден в {frames_dir}"
            )
        
        try:
            metadata = load_metadata(meta_path, self.module_name)
            
            # Валидация обязательных полей
            total_frames = int(metadata.get("total_frames") or metadata.get("num_frames") or 0)
            if total_frames <= 0:
                raise ValueError(
                    f"{self.module_name} | metadata.json не содержит корректного total_frames "
                    f"(значение: {metadata.get('total_frames')})"
                )
            
            return metadata
        except Exception as e:
            self.logger.exception(
                f"{self.module_name} | Ошибка загрузки метаданных из {meta_path}: {e}"
            )
            raise
    
    def create_frame_manager(
        self,
        frames_dir: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> FrameManager:
        """
        Создает и возвращает FrameManager с параметрами из метаданных.
        
        Args:
            frames_dir: Директория с кадрами
            metadata: Метаданные (если None, будут загружены автоматически)
            
        Returns:
            Инициализированный FrameManager
        """
        if metadata is None:
            metadata = self.load_metadata(frames_dir)
        
        chunk_size = int(metadata.get("chunk_size", 32))
        cache_size = int(metadata.get("cache_size", 2))
        
        try:
            frame_manager = FrameManager(
                frames_dir=frames_dir,
                chunk_size=chunk_size,
                cache_size=cache_size
            )
            self.logger.debug(
                f"{self.module_name} | FrameManager создан: "
                f"chunk_size={chunk_size}, cache_size={cache_size}"
            )
            return frame_manager
        except Exception as e:
            self.logger.exception(
                f"{self.module_name} | Ошибка создания FrameManager: {e}"
            )
            raise
    
    def get_frame_indices(
        self,
        metadata: Dict[str, Any],
        fallback_to_all: bool = False
    ) -> List[int]:
        """
        Получает список индексов кадров из метаданных.
        
        Args:
            metadata: Метаданные видео
            fallback_to_all: Если True, возвращает все кадры если нет специфичных индексов
            
        Returns:
            Список индексов кадров для обработки
        """
        # Пытаемся получить индексы из секции модуля
        module_section = metadata.get(self.module_name, {})
        frame_indices = module_section.get("frame_indices")
        
        if frame_indices is not None:
            if isinstance(frame_indices, list):
                return [int(idx) for idx in frame_indices]
            self.logger.error(
                f"{self.module_name} | frame_indices в метаданных не является списком"
            )
            raise TypeError(f"{self.module_name} invalid frame_indices type: {type(frame_indices)}")
        
        # Fallback: все кадры
        if fallback_to_all:
            total_frames = int(metadata.get("total_frames") or metadata.get("num_frames") or 0)
            if total_frames > 0:
                self.logger.info(
                    f"{self.module_name} | Используем все кадры (fallback): {total_frames}"
                )
                return list(range(total_frames))
        
        return []
    
    def run(
        self,
        frames_dir: str,
        config: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Полный цикл обработки: загрузка метаданных, создание FrameManager,
        обработка и сохранение результатов.
        
        Args:
            frames_dir: Директория с кадрами
            config: Конфигурация модуля
            metadata: Метаданные (если None, будут загружены автоматически)
            
        Returns:
            Путь к сохраненному файлу результатов
            
        Raises:
            FileNotFoundError: Если metadata.json не найден
            RuntimeError: Если обработка не удалась
        """
        # Загрузка метаданных
        if metadata is None:
            metadata = self.load_metadata(frames_dir)

        # Enforce run identity fields for baseline reproducibility (Segmenter writes these).
        required_run_keys = ["platform_id", "video_id", "run_id", "sampling_policy_version", "config_hash"]
        missing = [k for k in required_run_keys if not metadata.get(k)]
        if missing:
            raise RuntimeError(f"{self.module_name} | frames metadata missing required run identity keys: {missing}")
        
        # Получение индексов кадров
        frame_indices = self.get_frame_indices(metadata, fallback_to_all=False)
        if not frame_indices:
            raise ValueError(
                f"{self.module_name} | Нет кадров для обработки"
            )
        
        # Создание FrameManager
        frame_manager = None
        try:
            frame_manager = self.create_frame_manager(frames_dir, metadata)
            
            # Обработка
            self.logger.info(
                f"{self.module_name} | Начало обработки {len(frame_indices)} кадров"
            )
            
            results = self.process(
                frame_manager=frame_manager,
                frame_indices=frame_indices,
                config=config
            )
            
            # Подготовка метаданных для сохранения
            save_metadata = {
                "total_frames": metadata.get("total_frames"),
                "processed_frames": len(frame_indices),
                "frames_dir": frames_dir,
            # Baseline run identity (copied into NPZ meta)
            "platform_id": metadata.get("platform_id"),
            "video_id": metadata.get("video_id"),
            "run_id": metadata.get("run_id"),
            "sampling_policy_version": metadata.get("sampling_policy_version"),
            "config_hash": metadata.get("config_hash"),
            }
            
            # Сохранение результатов
            saved_path = self.save_results(
                results=results,
                metadata=save_metadata
            )
            
            self.logger.info(
                f"{self.module_name} | Обработка завершена. Результаты сохранены: {saved_path}"
            )
            
            return saved_path
            
        finally:
            # Гарантированное закрытие FrameManager
            if frame_manager is not None:
                try:
                    frame_manager.close()
                except Exception as e:
                    self.logger.exception(
                        f"{self.module_name} | Ошибка при закрытии FrameManager: {e}"
                    )
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(module_name={self.module_name}, rs_path={self.rs_path})"
