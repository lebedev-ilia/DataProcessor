# Core Object Detections Provider

## Описание

Core провайдер для детекции объектов на кадрах видео. Поддерживает YOLO и OWL-ViT/OWLv2 (open-vocabulary detection). Запускается один раз на видео в фазе core-провайдеров, сохраняет детекции объектов для использования модулями композиции и анализа сцен.

## Расположение

- **Провайдер**: `VisualProcessor/core/model_process/object_detections/main.py`
- **Выходные данные**: `result_store/core_object_detections/detections.json`

## Формат выходных данных

### Файл: `detections.json`

JSON-файл со структурой:

```json
{
  "version": "1.0",
  "created_at": "2025-01-15T10:30:00.123456",
  "impl": "yolo",
  "model": "yolo11x.pt",
  "model_family": "yolo",
  "box_threshold": 0.6,
  "total_frames": 1000,
  "frame_indices": [0, 10, 20, ...],
  "data": {
    "frames": {
      "0": [
        {
          "bbox": [100, 150, 200, 250],
          "class": "person",
          "confidence": 0.95,
          "class_id": 0
        },
        ...
      ],
      ...
    },
    "summary": {
      "total_detections": 500,
      "unique_categories": 10,
      "category_counts": {
        "person": 200,
        "car": 150,
        ...
      }
    }
  }
}
```

### Структура данных:

- `version` (str): Версия формата данных
- `created_at` (str): ISO-формат времени создания
- `impl` (str): Реализация ("yolo" или "owl")
- `model` (str): Название модели или путь к файлу
- `model_family` (str): Семейство моделей ("yolo", "owlvit", "owlv2")
- `box_threshold` (float): Порог уверенности для детекций
- `total_frames` (int): Общее количество кадров в видео
- `frame_indices` (list[int]): Список обработанных кадров

- `data.frames` (dict): Детекции по кадрам
  - Ключ: строка с индексом кадра (например, "0", "10")
  - Значение: список детекций
    - `bbox` (list[int]): [x1, y1, x2, y2] в пикселях
    - `class` (str): Название класса объекта
    - `confidence` (float): Уверенность детекции (0..1)
    - `class_id` (int, optional): ID класса (для YOLO)

- `data.summary` (dict): Сводная статистика
  - `total_detections` (int): Общее количество детекций
  - `unique_categories` (int): Количество уникальных классов
  - `category_counts` (dict): Счётчик детекций по классам

## Параметры запуска

### YOLO режим (по умолчанию):

```bash
python core/model_process/object_detections/main.py \
    --frames-dir <path_to_frames> \
    --rs-path <result_store_path> \
    --model "yolo11x.pt" \
    --batch-size 16 \
    --box-threshold 0.6
```

### OWL-ViT/OWLv2 режим (open-vocabulary):

```bash
python core/model_process/object_detections/main.py \
    --frames-dir <path_to_frames> \
    --rs-path <result_store_path> \
    --use-queries \
    --model "google/owlvit-base-patch32" \
    --model-family "owlvit" \
    --default-categories "person,car,bus" \
    --box-threshold 0.3 \
    --device "cuda"
```

### Параметры

- `--frames-dir` (required): Путь к директории с кадрами
- `--rs-path` (required): Путь к result_store
- `--model` (default: "yolo11x.pt"): Название модели или путь к файлу
- `--batch-size` (default: 16): Размер батча для YOLO
- `--use-queries` (flag): Использовать OWL-ViT/OWLv2 (open-vocabulary)
- `--model-family` (default: "owlv2"): Семейство моделей для queries ("owlvit" или "owlv2")
- `--default-categories` (optional): Список категорий через запятую для queries
- `--box-threshold` (default: 0.6): Порог уверенности (для YOLO обычно 0.6, для OWL — 0.3)
- `--device` (optional): Устройство для OWL моделей ("cuda" или "cpu")

## Использование в модулях

Модули, которые используют core_object_detections:

- `frames_composition` — для анализа композиции с учётом объектов
- `object_detection` (сам модуль) — для детального анализа объектов
- `scene_classification` — для улучшения классификации сцен на основе объектов

### Пример чтения данных

```python
import json
import os

def load_core_object_detections(rs_path: str):
    """Загружает детекции объектов из core_object_detections провайдера."""
    detections_path = os.path.join(rs_path, "core_object_detections", "detections.json")
    if not os.path.isfile(detections_path):
        return None
    
    with open(detections_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    frames_data = data.get("data", {}).get("frames", {})
    
    # Преобразуем ключи в int для удобства
    frames_dict = {int(k): v for k, v in frames_data.items()}
    
    return {
        "frames": frames_dict,
        "summary": data.get("data", {}).get("summary", {}),
        "impl": data.get("impl", "yolo"),
        "model": data.get("model", ""),
        "box_threshold": data.get("box_threshold", 0.6),
    }
```

## Версионирование

- **Версия 1.0**: Базовый формат с детекциями по кадрам и сводной статистикой

## Зависимости

### Для YOLO:
- `ultralytics` (YOLOv8/YOLOv11)
- `torch`

### Для OWL-ViT/OWLv2:
- `transformers`
- `torch`
- `torchvision`

