# Стандарты разработки модулей VisualProcessor

## Обзор

Этот документ определяет единые принципы и правила для всех модулей VisualProcessor, обеспечивающие консистентность кода, упрощение поддержки и улучшение взаимодействия между компонентами.

---

## Единые правила и стандарты

### 1. Структура модуля

Каждый модуль должен иметь следующую структуру:

```
modules/<module_name>/
├── main.py                    # Точка входа CLI
├── <module_name>_processor.py # Основной класс процессора (опционально)
├── <module_name>_*.py         # Вспомогательные модули
├── FEATURES_DESCRIPTION.md     # Описание функциональности
└── models/                     # Модели (если нужны)
```

### 2. Стандарт для main.py

#### 2.1 Импорты и настройка пути

```python
#!/usr/bin/env python3
"""
CLI интерфейс для модуля <module_name>.

Особенности:
- безопасное чтение метаданных
- тщательная валидация аргументов
- сохранение результатов через ResultsStore
- подробное логирование ошибок и стадий выполнения
- явное освобождение ресурсов
"""

from __future__ import annotations

import os
import sys
import json
import argparse
import traceback
from typing import Optional, Dict, Any, List

# Добавляем корневую директорию проекта в sys.path
_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if _root not in sys.path:
    sys.path.append(_root)

from utils.frame_manager import FrameManager
from utils.results_store import ResultsStore
from utils.logger import get_logger
from utils.utilites import load_metadata

# Константы модуля
MODULE_NAME = "<module_name>"
logger = get_logger(MODULE_NAME)
```

#### 2.2 Загрузка метаданных

**ВСЕГДА** используйте единую функцию `load_metadata()` из `utils.utilites`:

```python
def load_metadata_safe(meta_path: str) -> dict:
    """Безопасная загрузка метаданных с логированием ошибок."""
    try:
        return load_metadata(meta_path, MODULE_NAME)
    except Exception:
        logger.exception("Не удалось загрузить metadata.json из %s", meta_path)
        raise
```

#### 2.3 Обработка аргументов

```python
def create_parser() -> argparse.ArgumentParser:
    """Создает и настраивает парсер аргументов."""
    parser = argparse.ArgumentParser(
        prog=f"run_{MODULE_NAME}",
        description=f"<Описание модуля> — CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    # Обязательные аргументы (всегда одинаковые)
    parser.add_argument(
        "--frames-dir", 
        required=True, 
        help="Директория с кадрами (FrameManager ожидает metadata.json внутри)"
    )
    parser.add_argument(
        "--rs-path", 
        required=True, 
        help="Папка для результирующего стора (ResultsStore)"
    )
    
    # Специфичные аргументы модуля
    # ...
    
    parser.add_argument(
        "--log-level", 
        type=str, 
        default="INFO", 
        help="Уровень логирования (DEBUG/INFO/WARN/ERROR)"
    )
    
    return parser
```

#### 2.4 Основная функция обработки

```python
def process_video(
    frame_manager: FrameManager,
    metadata: Dict[str, Any],
    rs_path: str,
    **module_specific_args
) -> Dict[str, Any]:
    """
    Обрабатывает видео и возвращает структурированные результаты.
    
    Args:
        frame_manager: Менеджер кадров
        metadata: Метаданные видео
        rs_path: Путь к хранилищу результатов
        **module_specific_args: Специфичные аргументы модуля
        
    Returns:
        Dict с результатами обработки
    """
    # Инициализация процессора
    processor = ModuleProcessor(**module_specific_args)
    
    # Получение индексов кадров из метаданных
    frame_indices = metadata.get(MODULE_NAME, {}).get("frame_indices", list(range(metadata["total_frames"])))
    
    # Обработка
    results = processor.process(
        frame_manager=frame_manager,
        frame_indices=frame_indices
    )
    
    return results
```

#### 2.5 Главная функция main()

```python
def main(argv: Optional[List[str]] = None) -> int:
    """Главная функция CLI. Возвращает код выхода (0 = успех)."""
    parser = create_parser()
    args = parser.parse_args(argv)
    
    # Настройка уровня логирования
    try:
        import logging as _logging
        _logging.getLogger().setLevel(
            getattr(_logging, args.log_level.upper(), _logging.INFO)
        )
    except Exception:
        logger.warning("Не удалось установить log-level: %s", args.log_level)
    
    # Загрузка метаданных
    meta_path = os.path.join(args.frames_dir, "metadata.json")
    try:
        metadata = load_metadata_safe(meta_path)
    except Exception:
        logger.error("Не удалось загрузить metadata.json; завершаем.")
        return 2
    
    # Валидация метаданных
    total_frames = int(metadata.get("total_frames") or metadata.get("num_frames") or 0)
    if total_frames <= 0:
        logger.error(
            "metadata.json не содержит корректного total_frames (значение: %s). Завершаем.",
            metadata.get("total_frames")
        )
        return 3
    
    chunk_size = int(metadata.get("chunk_size", 32))
    cache_size = int(metadata.get("cache_size", 2))
    
    # Инициализация ресурсов
    frame_manager = None
    try:
        frame_manager = FrameManager(
            frames_dir=args.frames_dir,
            chunk_size=chunk_size,
            cache_size=cache_size
        )
        
        # Обработка
        results = process_video(
            frame_manager=frame_manager,
            metadata=metadata,
            rs_path=args.rs_path,
            **vars(args)  # Передаем все аргументы как kwargs
        )
        
        # Сохранение результатов
        rs = ResultsStore(args.rs_path)
        save_results(rs, results, metadata)
        
        logger.info("Обработка завершена успешно")
        return 0
        
    except Exception:
        logger.exception("Fatal error в %s", MODULE_NAME)
        return 4
    finally:
        # Гарантированное освобождение ресурсов
        if frame_manager is not None:
            try:
                frame_manager.close()
            except Exception:
                logger.exception("Ошибка при закрытии FrameManager")


if __name__ == "__main__":
    raise SystemExit(main())
```

### 3. Сохранение результатов

#### 3.1 Единый подход к сохранению

**Правило**: Все модули должны использовать `ResultsStore` для сохранения результатов.

#### 3.2 Выбор метода сохранения

**Используйте `rs.store()` (JSON)** для:
- Структурированных данных (словари, списки)
- Метаданных и метрик
- Результатов без больших массивов

**Используйте `rs.store_compressed()` (NPZ)** для:
- Больших массивов эмбеддингов
- Per-track результатов с эмбеддингами
- Данных, требующих эффективного хранения

#### 3.3 Функция сохранения результатов

```python
def save_results(
    rs: ResultsStore,
    results: Dict[str, Any],
    metadata: Dict[str, Any],
    use_compressed: bool = False,
    embeddings_key: str = "embeddings"
) -> str:
    """
    Сохраняет результаты через ResultsStore.
    
    Args:
        rs: Экземпляр ResultsStore
        results: Результаты обработки
        metadata: Метаданные видео
        use_compressed: Использовать store_compressed вместо store
        embeddings_key: Ключ для эмбеддингов (если use_compressed=True)
        
    Returns:
        Путь к сохраненному файлу
    """
    # Добавление метаданных к результатам
    results_with_meta = {
        **results,
        "metadata": {
            "total_frames": metadata.get("total_frames"),
            "producer": MODULE_NAME,
            "created_at": datetime.utcnow().isoformat(),
        }
    }
    
    if use_compressed:
        # Для per-track результатов с эмбеддингами
        if isinstance(results_with_meta, dict) and any(
            isinstance(v, dict) and embeddings_key in v 
            for v in results_with_meta.values()
        ):
            core_dir = os.path.join(rs.root_path, MODULE_NAME)
            os.makedirs(core_dir, exist_ok=True)
            npz_path = os.path.join(core_dir, f"{MODULE_NAME}_emb.npz")
            
            return rs.store_compressed(
                results=results_with_meta,
                out_path=npz_path,
                embeddings_key=embeddings_key,
                meta=results_with_meta.get("metadata", {})
            )
        else:
            logger.warning(
                "use_compressed=True, но формат результатов не поддерживает store_compressed. "
                "Используется store()"
            )
            use_compressed = False
    
    if not use_compressed:
        # Стандартное сохранение в JSON
        return rs.store(results_with_meta, name=MODULE_NAME)
```

### 4. Работа с FrameManager

#### 4.1 Инициализация

```python
frame_manager = FrameManager(
    frames_dir=args.frames_dir,
    chunk_size=metadata.get("chunk_size", 32),
    cache_size=metadata.get("cache_size", 2)
)
```

#### 4.2 Гарантированное закрытие

**ВСЕГДА** используйте `try-finally` для гарантированного закрытия:

```python
frame_manager = None
try:
    frame_manager = FrameManager(...)
    # ... обработка ...
finally:
    if frame_manager is not None:
        try:
            frame_manager.close()
        except Exception:
            logger.exception("Ошибка при закрытии FrameManager")
```

### 5. Логирование

#### 5.1 Инициализация logger

```python
MODULE_NAME = "<module_name>"
logger = get_logger(MODULE_NAME)
```

#### 5.2 Формат сообщений

Используйте единый формат:
```python
logger.info(f"{MODULE_NAME} | <function_name> | <message>")
logger.error(f"{MODULE_NAME} | <function_name> | <error_message>")
logger.exception(f"{MODULE_NAME} | <function_name> | <error_message>")
```

#### 5.3 Уровни логирования

- **DEBUG**: Детальная информация для отладки
- **INFO**: Основные этапы обработки
- **WARNING**: Предупреждения, не критичные ошибки
- **ERROR**: Ошибки, требующие внимания
- **EXCEPTION**: Критичные ошибки с traceback

### 6. Обработка ошибок

#### 6.1 Валидация входных данных

```python
# Проверка метаданных
total_frames = int(metadata.get("total_frames") or 0)
if total_frames <= 0:
    logger.error("Некорректное значение total_frames: %s", metadata.get("total_frames"))
    return 3  # Код ошибки
```

#### 6.2 Обработка исключений

```python
try:
    # Операция
    result = risky_operation()
except SpecificException as e:
    logger.exception("Ошибка при выполнении операции: %s", e)
    # Fallback или повторная попытка
    result = fallback_operation()
except Exception:
    logger.exception("Неожиданная ошибка")
    raise  # Пробрасываем дальше, если не можем обработать
```

### 7. Типизация

#### 7.1 Использование type hints

```python
from typing import Dict, List, Optional, Any, Tuple

def process_video(
    frame_manager: FrameManager,
    metadata: Dict[str, Any],
    rs_path: str,
    **kwargs: Any
) -> Dict[str, Any]:
    ...
```

#### 7.2 Аннотации возвращаемых значений

Все публичные функции должны иметь аннотации типов.

### 8. Документация

#### 8.1 Docstrings

Все публичные функции и классы должны иметь docstrings:

```python
def process_video(
    frame_manager: FrameManager,
    metadata: Dict[str, Any],
    rs_path: str
) -> Dict[str, Any]:
    """
    Обрабатывает видео и возвращает структурированные результаты.
    
    Args:
        frame_manager: Менеджер кадров для доступа к кадрам
        metadata: Метаданные видео (total_frames, chunk_size, и т.д.)
        rs_path: Путь к хранилищу результатов
        
    Returns:
        Словарь с результатами обработки. Структура зависит от модуля.
        
    Raises:
        ValueError: Если входные данные некорректны
        RuntimeError: Если обработка не удалась
    """
    ...
```

### 9. Константы и конфигурация

#### 9.1 Именование констант

```python
MODULE_NAME = "<module_name>"  # Имя модуля для логирования
DEFAULT_BATCH_SIZE = 32        # Значения по умолчанию
DEFAULT_CHUNK_SIZE = 32
```

#### 9.2 Конфигурация из метаданных

Всегда извлекайте конфигурацию из `metadata.json`:

```python
chunk_size = int(metadata.get("chunk_size", DEFAULT_CHUNK_SIZE))
cache_size = int(metadata.get("cache_size", DEFAULT_CACHE_SIZE))
total_frames = int(metadata.get("total_frames", 0))
```

---

## Миграционный план

### Этап 1: Исправление критических проблем
1. Исправить баг в `action_recognition/main.py` (функция `process_video` не возвращает результат)
2. Унифицировать загрузку метаданных во всех модулях
3. Гарантировать закрытие FrameManager во всех модулях

### Этап 2: Унификация сохранения результатов
1. Определить, какие модули должны использовать `store_compressed()`
2. Привести все модули к единому подходу сохранения
3. Удалить прямое использование `np.savez_compressed()` из core модулей

### Этап 3: Рефакторинг структуры
1. Привести все `main.py` к единому шаблону
2. Унифицировать обработку аргументов
3. Стандартизировать логирование

### Этап 4: Документация и тестирование
1. Добавить docstrings во все публичные функции
2. Обновить документацию модулей
3. Добавить примеры использования

---

## Примеры

### Пример 1: Простой модуль (JSON результаты)

```python
#!/usr/bin/env python3
"""CLI для модуля example_module."""

from __future__ import annotations

import os
import sys
import argparse
from typing import Optional, Dict, Any, List

_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if _root not in sys.path:
    sys.path.append(_root)

from utils.frame_manager import FrameManager
from utils.results_store import ResultsStore
from utils.logger import get_logger
from utils.utilites import load_metadata

MODULE_NAME = "example_module"
logger = get_logger(MODULE_NAME)


def process_video(
    frame_manager: FrameManager,
    metadata: Dict[str, Any],
    rs_path: str,
    **kwargs: Any
) -> Dict[str, Any]:
    """Обрабатывает видео."""
    frame_indices = metadata.get(MODULE_NAME, {}).get(
        "frame_indices", 
        list(range(metadata["total_frames"]))
    )
    
    # Ваша логика обработки
    results = {
        "success": True,
        "processed_frames": len(frame_indices),
        "metrics": {}
    }
    
    return results


def save_results(
    rs: ResultsStore,
    results: Dict[str, Any],
    metadata: Dict[str, Any]
) -> str:
    """Сохраняет результаты."""
    from datetime import datetime
    
    results_with_meta = {
        **results,
        "metadata": {
            "total_frames": metadata.get("total_frames"),
            "producer": MODULE_NAME,
            "created_at": datetime.utcnow().isoformat(),
        }
    }
    
    return rs.store(results_with_meta, name=MODULE_NAME)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog=f"run_{MODULE_NAME}",
        description=f"{MODULE_NAME} — CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--rs-path", required=True)
    parser.add_argument("--log-level", type=str, default="INFO")
    
    args = parser.parse_args(argv)
    
    # Настройка логирования
    try:
        import logging as _logging
        _logging.getLogger().setLevel(
            getattr(_logging, args.log_level.upper(), _logging.INFO)
        )
    except Exception:
        logger.warning("Не удалось установить log-level: %s", args.log_level)
    
    # Загрузка метаданных
    meta_path = os.path.join(args.frames_dir, "metadata.json")
    try:
        metadata = load_metadata(meta_path, MODULE_NAME)
    except Exception:
        logger.exception("Не удалось загрузить metadata.json")
        return 2
    
    # Валидация
    total_frames = int(metadata.get("total_frames") or 0)
    if total_frames <= 0:
        logger.error("Некорректное total_frames: %s", metadata.get("total_frames"))
        return 3
    
    chunk_size = int(metadata.get("chunk_size", 32))
    cache_size = int(metadata.get("cache_size", 2))
    
    # Обработка
    frame_manager = None
    try:
        frame_manager = FrameManager(
            frames_dir=args.frames_dir,
            chunk_size=chunk_size,
            cache_size=cache_size
        )
        
        results = process_video(
            frame_manager=frame_manager,
            metadata=metadata,
            rs_path=args.rs_path,
            **vars(args)
        )
        
        rs = ResultsStore(args.rs_path)
        save_results(rs, results, metadata)
        
        logger.info("Обработка завершена успешно")
        return 0
        
    except Exception:
        logger.exception("Fatal error")
        return 4
    finally:
        if frame_manager is not None:
            try:
                frame_manager.close()
            except Exception:
                logger.exception("Ошибка при закрытии FrameManager")


if __name__ == "__main__":
    raise SystemExit(main())
```

### Пример 2: Модуль с эмбеддингами (NPZ результаты)

Для модулей с большими массивами эмбеддингов используйте `store_compressed()`:

```python
def save_results(
    rs: ResultsStore,
    results: Dict[int, Dict[str, Any]],  # Per-track результаты
    metadata: Dict[str, Any]
) -> str:
    """Сохраняет результаты с эмбеддингами в NPZ."""
    from datetime import datetime
    
    core_dir = os.path.join(rs.root_path, MODULE_NAME)
    os.makedirs(core_dir, exist_ok=True)
    npz_path = os.path.join(core_dir, f"{MODULE_NAME}_emb.npz")
    
    save_meta = {
        "total_frames": metadata.get("total_frames"),
        "producer": MODULE_NAME,
        "created_at": datetime.utcnow().isoformat(),
    }
    
    return rs.store_compressed(
        results=results,
        out_path=npz_path,
        embeddings_key="embedding_normed_256d",
        meta=save_meta
    )
```

---

## Чеклист для проверки модуля

### Для модулей с BaseModule (рекомендуется)

- [ ] Модуль наследуется от `BaseModule`
- [ ] Реализован метод `process(frame_manager, frame_indices, config)`
- [ ] Объявлены зависимости через `required_dependencies()` (если нужны)
- [ ] Результаты сохраняются через `save_results()`
- [ ] `main.py` использует `module.run()` или методы BaseModule для работы с метаданными/FrameManager
- [ ] Все публичные функции имеют type hints и docstrings
- [ ] Используется единый формат логирования

### Для модулей без BaseModule (legacy)

- [ ] Используется единая функция `load_metadata()` из `utils.utilites`
- [ ] FrameManager закрывается в блоке `finally`
- [ ] Результаты сохраняются через `ResultsStore`
- [ ] Выбран правильный метод сохранения (`store()` или `store_compressed()`)
- [ ] Все публичные функции имеют type hints и docstrings
- [ ] Используется единый формат логирования
- [ ] Обработка ошибок с логированием
- [ ] Валидация входных данных
- [ ] Константа `MODULE_NAME` определена и используется
- [ ] Структура `main.py` соответствует шаблону

---

## BaseModule - Базовый класс для модулей

### Обзор

Все модули должны наследоваться от `BaseModule`, который обеспечивает:
- Единый интерфейс обработки (`process(frame_manager, frame_indices, config)`)
- Автоматическую загрузку результатов других модулей
- Стандартизированное сохранение результатов в `np.savez_compressed`
- Управление зависимостями между модулями
- **Автоматическую работу с метаданными и FrameManager**
- Логирование и обработку ошибок

### Основные возможности

#### 1. Работа с метаданными

```python
# Загрузка метаданных
metadata = module.load_metadata(frames_dir)

# Получение индексов кадров из метаданных
frame_indices = module.get_frame_indices(metadata, fallback_to_all=True)
```

#### 2. Работа с FrameManager

```python
# Создание FrameManager с параметрами из метаданных
frame_manager = module.create_frame_manager(frames_dir, metadata)

# Автоматическое закрытие в finally блоке
try:
    # работа с frame_manager
    pass
finally:
    frame_manager.close()
```

#### 3. Полный цикл обработки

```python
# Всё в одном вызове:
# - загружает метаданные
# - создает FrameManager
# - получает frame_indices
# - вызывает process()
# - сохраняет результаты
# - закрывает FrameManager
saved_path = module.run(frames_dir, config)
```

### Основные требования

1. **Обязательный метод `process()`**:
   ```python
   def process(
       self,
       frame_manager: FrameManager,
       frame_indices: List[int],
       config: Dict[str, Any]
   ) -> Dict[str, Any]:
       """Обрабатывает видео и возвращает результаты."""
       pass
   ```

2. **Объявление зависимостей** (если нужны):
   ```python
   def required_dependencies(self) -> List[str]:
       """Возвращает список имен модулей-зависимостей."""
       return ["core_clip", "core_face_landmarks"]
   ```

3. **Сохранение результатов**:
   ```python
   # Автоматически в np.savez_compressed
   saved_path = module.save_results(
       results=results,
       metadata={"total_frames": 1000}
   )
   ```

### Пример использования

#### Простой вариант (рекомендуется)

```python
from modules.base_module import BaseModule
from utils.frame_manager import FrameManager

class MyModule(BaseModule):
    def required_dependencies(self) -> List[str]:
        return ["core_clip"]  # Зависимость от core_clip
    
    def process(
        self,
        frame_manager: FrameManager,
        frame_indices: List[int],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        # Загрузка зависимостей
        dependencies = self.load_all_dependencies()
        clip_results = dependencies["core_clip"]
        
        # Обработка кадров
        features = []
        for frame_idx in frame_indices:
            frame = frame_manager.get(frame_idx)
            feature = self._extract_feature(frame)
            features.append(feature)
        
        return {
            "features": np.array(features),
            "metrics": {"mean": float(np.mean(features))}
        }
```

#### Упрощенный main.py

```python
# main.py - теперь очень простой!
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--rs-path", required=True)
    args = parser.parse_args()
    
    module = MyModule(rs_path=args.rs_path)
    config = {}  # ваша конфигурация
    
    # Всё! BaseModule делает остальное
    saved_path = module.run(frames_dir=args.frames_dir, config=config)
    logger.info(f"Результаты: {saved_path}")
```

#### Детальный контроль (если нужен)

```python
# Если нужен более детальный контроль
def main():
    module = MyModule(rs_path=args.rs_path)
    
    # Загрузка метаданных
    metadata = module.load_metadata(args.frames_dir)
    
    # Получение индексов
    frame_indices = module.get_frame_indices(metadata)
    
    # Создание FrameManager
    frame_manager = None
    try:
        frame_manager = module.create_frame_manager(args.frames_dir, metadata)
        
        # Обработка
        results = module.process(frame_manager, frame_indices, config)
        
        # Сохранение
        saved_path = module.save_results(results, metadata={"total_frames": metadata["total_frames"]})
    finally:
        if frame_manager:
            frame_manager.close()
```

### Преимущества BaseModule

1. **Единообразие**: Все модули имеют одинаковый интерфейс
2. **Автоматизация**: 
   - Загрузка зависимостей и сохранение результатов автоматизированы
   - **Работа с метаданными и FrameManager автоматизирована**
   - **Полный цикл обработки в одном методе `run()`**
3. **Упрощение main.py**: 
   - Не нужно вручную загружать метаданные
   - Не нужно вручную создавать FrameManager
   - Не нужно помнить о закрытии FrameManager
4. **Типобезопасность**: Type hints для всех методов
5. **Обработка ошибок**: Встроенная обработка ошибок и логирование
6. **Гибкость**: Легко расширяется для специфичных нужд модулей

### Миграция существующих модулей

Для миграции существующего модуля:

1. Наследуйтесь от `BaseModule`
2. Переименуйте метод обработки в `process(frame_manager, frame_indices, config)`
3. Объявите зависимости через `required_dependencies()`
4. Используйте `save_results()` вместо прямого сохранения
5. **Упростите `main.py`**:
   - Удалите загрузку метаданных (используйте `module.load_metadata()` или `module.run()`)
   - Удалите создание FrameManager (используйте `module.create_frame_manager()` или `module.run()`)
   - Удалите получение frame_indices (используйте `module.get_frame_indices()` или `module.run()`)
   - Используйте `module.run()` для полного цикла или отдельные методы для детального контроля

**До миграции:**
```python
# main.py - много повторяющегося кода
metadata = load_json(f"{args.frames_dir}/metadata.json")
frame_manager = FrameManager(frames_dir=args.frames_dir, 
                            chunk_size=metadata["chunk_size"], 
                            cache_size=metadata["cache_size"])
frame_indices = metadata[name]["frame_indices"]
# ... обработка ...
try:
    frame_manager.close()
except:
    pass
```

**После миграции:**
```python
# main.py - просто и понятно
module = MyModule(rs_path=args.rs_path)
saved_path = module.run(frames_dir=args.frames_dir, config={})
```

Подробные примеры см. в `modules/base_module_main_example.py`.

---

## Контакты и вопросы

При возникновении вопросов или предложений по улучшению стандартов, создайте issue или обратитесь к команде разработки.

