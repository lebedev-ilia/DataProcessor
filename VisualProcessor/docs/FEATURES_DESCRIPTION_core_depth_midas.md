# Core Depth MiDaS Provider

## Описание

Core провайдер для оценки глубины (depth estimation) с помощью модели MiDaS. Запускается один раз на видео в фазе core-провайдеров, сохраняет статистику глубины для использования модулями композиции и качества.

## Расположение

- **Провайдер**: `VisualProcessor/core/model_process/depth_midas/main.py`
- **Выходные данные**: `result_store/core_depth_midas/depth.json`

## Формат выходных данных

### Файл: `depth.json`

JSON-файл со структурой:

```json
{
  "version": "1.0",
  "created_at": "2025-01-15T10:30:00.123456",
  "model_name": "MiDaS_small",
  "total_frames": 1000,
  "frame_indices": [0, 10, 20, ...],
  "per_frame": [
    {
      "frame_index": 0,
      "depth_mean": 0.45,
      "depth_std": 0.12,
      "depth_min": 0.1,
      "depth_max": 0.9,
      "depth_range": 0.8,
      "foreground_depth": 0.3,
      "background_depth": 0.7,
      ...
    },
    ...
  ],
  "aggregates": {
    "depth_mean": {
      "mean": 0.45,
      "std": 0.08,
      "min": 0.2,
      "max": 0.7
    },
    ...
  }
}
```

### Структура данных:

- `version` (str): Версия формата данных
- `created_at` (str): ISO-формат времени создания
- `model_name` (str): Название модели MiDaS (например, "MiDaS_small")
- `total_frames` (int): Общее количество кадров в видео
- `frame_indices` (list[int]): Список обработанных кадров (может быть подвыборка)

- `per_frame` (list): Статистика по каждому кадру
  - `frame_index` (int): Индекс кадра
  - `depth_mean` (float): Средняя глубина (нормализованная 0..1)
  - `depth_std` (float): Стандартное отклонение глубины
  - `depth_min` (float): Минимальная глубина
  - `depth_max` (float): Максимальная глубина
  - `depth_range` (float): Диапазон глубины (max - min)
  - `foreground_depth` (float, optional): Средняя глубина переднего плана
  - `background_depth` (float, optional): Средняя глубина заднего плана
  - Дополнительные поля зависят от реализации

- `aggregates` (dict): Агрегированная статистика по всему видео
  - Для каждого числового поля из `per_frame` вычисляются: `mean`, `std`, `min`, `max`

## Параметры запуска

```bash
python core/model_process/depth_midas/main.py \
    --frames-dir <path_to_frames> \
    --rs-path <result_store_path> \
    --max-frames 100
```

### Параметры

- `--frames-dir` (required): Путь к директории с кадрами
- `--rs-path` (required): Путь к result_store
- `--max-frames` (optional): Ограничение количества кадров для обработки (для ускорения)

## Использование в модулях

Модули, которые используют core_depth_midas:

- `frames_composition` — для анализа композиции с учётом глубины (foreground/background separation)
- `shot_quality` — для оценки качества кадра на основе глубины (depth of field, focus)

### Пример чтения данных

```python
import json
import os

def load_core_depth_midas(rs_path: str):
    """Загружает данные глубины из core_depth_midas провайдера."""
    depth_path = os.path.join(rs_path, "core_depth_midas", "depth.json")
    if not os.path.isfile(depth_path):
        return None
    
    with open(depth_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    per_frame = data.get("per_frame", [])
    
    # Преобразуем в dict[frame_index] -> depth_stats для удобства
    depth_dict = {int(pf["frame_index"]): pf for pf in per_frame if "frame_index" in pf}
    
    return {
        "per_frame": depth_dict,
        "aggregates": data.get("aggregates", {}),
        "model_name": data.get("model_name", "MiDaS_small"),
        "total_frames": data.get("total_frames", 0),
    }
```

## Версионирование

- **Версия 1.0**: Базовый формат со статистикой глубины по кадрам

## Зависимости

- `torch`
- `midas` (MiDaS depth estimation model)
- `numpy`
- `opencv-python`

