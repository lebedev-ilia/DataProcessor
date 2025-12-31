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
    
    # Получение индексов кадров из метаданных (Frame Sampling Contract)
    # Важно: никаких fallback. Если Segmenter не положил frame_indices — это ошибка контракта.
    frame_indices = metadata.get(MODULE_NAME, {}).get("frame_indices")
    if frame_indices is None:
        raise KeyError(f"{MODULE_NAME} missing required metadata[{MODULE_NAME}][frame_indices]")
    
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

#### 4.1.1 Цветовое пространство (RGB/BGR)

Полуфинальное правило (контракт):
- Все кадры, доступные через `FrameManager.get()`, должны быть в **RGB** (`HxWx3 uint8`).
- Это фиксируется в `frames_dir/metadata.json` полем `color_space="RGB"`.
- Любые модули, использующие OpenCV, должны явно конвертировать в BGR только если конкретная функция OpenCV этого требует.

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
    # Никаких fallback/эвристик: либо умеем обработать корректно, либо явно падаем.
    raise
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

## Контракт выборки кадров (Frame Sampling Contract) — ОБЯЗАТЕЛЬНО

### TL;DR
- **Segmenter/DataProcessor** отвечает за выборку кадров и записывает её в `metadata.json`.
- **Ни один модуль и ни один core‑провайдер не выбирает кадры сам** (никаких `sample_step` внутри провайдеров).
- Все компоненты обязаны использовать **ровно** `metadata[<component_name>]["frame_indices"]`.
- Если `frame_indices` отсутствуют или пустые — **компонент обязан упасть (raise)**. Никаких fallback/эвристик.

### Почему это важно
Диапазон длины видео сильно варьируется (пример: 120 … 36000 кадров). Фиксированный шаг (например, “каждый 5‑й кадр”) приводит к разной информативности и нестабильному качеству признаков между короткими и длинными видео. Поэтому выборка должна быть **адаптивной** и централизованной (на уровне Segmenter), а модули должны работать строго по входным индексам.

### Требования к `metadata.json`
Для каждого компонента (модуля или core‑провайдера) Segmenter должен добавлять секцию:

```json
{
  "<component_name>": {
    "frame_indices": [0, 5, 10, 15]
  }
}
```

Где `<component_name>` — это:
- **для модулей**: имя модуля (`shot_quality`, `cut_detection`, `scene_classification`, …)
- **для core‑провайдеров**: имя провайдера (`core_clip`, `core_depth_midas`, `core_object_detections`, `core_face_landmarks`, …)

### Рекомендации для “умной” выборки (описание в README каждого модуля)
Каждый модуль обязан документировать в своём README:
- целевое количество кадров \(N\) или диапазон \(N_{min}..N_{max}\)
- принцип распределения кадров по времени/шотам/сценам (например, stratified per‑shot)
- требования к согласованности с другими модулями (например, `shot_quality` ↔ `cut_detection`)

---

## Запрет fallback поведения (No-Fallback Policy)

**Запрещено**:
- “если core_* не найден — посчитаем сами”
- “если нет эмбеддинга — вернём равномерное распределение”
- “если нет frame_indices — возьмём range(total_frames, step)”

**Разрешено**:
- Явно `raise RuntimeError/ValueError` с понятным сообщением, что именно отсутствует (файл/ключ/индексы).

---

## Политика “валидных пустых результатов” (Valid Empty Outputs)

Иногда зависимость является **обязательной**, но по данным она может быть “пустой” (пример: `core_face_landmarks` на видео без лиц).

Это **не ошибка** и не fallback. Это корректный результат провайдера.

Требования:
- core‑провайдер обязан сохранять **явные validity‑маски**:
  - например `face_present (N, max_faces) bool`, `hands_present (N, max_hands) bool`
- значения соответствующих массивов могут оставаться `NaN` (например, `face_landmarks`), но:
  - downstream‑модули должны отличать “провайдер не запускался/не найден” от “провайдер отработал, но данных нет”
- для human-friendly интерпретации “пустоты” провайдер/модуль должен класть в `meta` (object dict):
  - `has_any_<signal>`: bool (например, `has_any_face`)
  - `empty_reason`: str (например, `"no_faces_in_video"`) если данных нет
- если downstream‑модуль не может иметь смысл без этих данных (пример: “модуль только про лица”) — это решение уровня модуля:
  - либо он делает `raise` (если без лиц работа бессмысленна),
  - либо сохраняет `NaN`/пустые массивы и продолжает (если метрики опциональны).

---

## Правила размещения артефактов core‑провайдеров в `rs_path`

Чтобы избежать конфликтов имён и упростить загрузку зависимостей:
- Каждый core‑провайдер сохраняет артефакты в **поддиректорию** `rs_path/<core_provider_name>/`.

Примеры:
- `core_clip` → `rs_path/core_clip/embeddings.npz`
- `core_depth_midas` → `rs_path/core_depth_midas/depth.npz`
- `core_object_detections` → `rs_path/core_object_detections/detections.npz`
- `core_face_landmarks` → `rs_path/core_face_landmarks/landmarks.npz` (или аналогично)

Если по историческим причинам артефакт лежит в корне `rs_path`, загрузчик зависимостей может поддерживать legacy‑путь, но **производственный стандарт** — поддиректории.

---

## Стандарт сохранения артефактов (NPZ artifacts)

### 1) Формат
Проект делает полный переход на:
- **`BaseModule`** как единый интерфейс
- **`NPZ`** как единый формат сохранения результатов

### 2) Имена файлов (production recommendation)
Рекомендуется сохранять **timestamped** артефакты (и выбирать “последний” как актуальный):
- проще отлаживать, сравнивать прогоны и делать аудит
- загрузчики зависимостей могут выбирать последний файл по времени модификации

Пример имени:
`<module_name>_features_YYYY-mm-dd_HH-MM-SS-ffffff_<uid>.npz`

### 3) Retain policy (пока не определена)
На текущем этапе **retain policy не фиксируется** (сколько артефактов хранить). Решение будет принято позже.

