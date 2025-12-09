# Object Detection Module

Модуль для детекции объектов в видео с использованием OWL-ViT (Vision Transformer для Open-Vocabulary Object Detection). Поддерживает детекцию объектов по текстовым запросам без необходимости предобучения на конкретных классах.

## Описание

Модуль `object_detection` использует OWL-ViT (Google) для детекции объектов в видео кадрах. Основное преимущество - возможность детектировать объекты по произвольным текстовым описаниям без необходимости переобучения модели.

## Основные возможности

- **Open-Vocabulary Detection**: Детекция объектов по текстовым запросам
- **Поддержка OWL-ViT и OWL-ViT v2**: Две версии модели
- **Гибкие категории**: Возможность задавать произвольные категории объектов
- **Трекинг объектов**: Отслеживание объектов между кадрами (IoU-based tracker)
- **Длительность присутствия**: Вычисление времени присутствия каждого объекта
- **Object Density**: Метрика плотности объектов в кадре
- **Overlapping Analysis**: Анализ перекрытий между объектами
- **Turnover Rate**: Метрика появления/исчезновения объектов (birth/death events)
- **Детекция брендов**: Автоматическая детекция логотипов популярных брендов
- **Семантические теги**: Классификация объектов по семантике (luxury, danger, cute, etc.)
- **Атрибуты объектов**: Извлечение цвета и других атрибутов
- **Интеграция с BaseModule**: Совместимость с архитектурой проекта
- **Batch processing**: Обработка множества кадров
- **Агрегация результатов**: Подсчет объектов по категориям

## Установка

### Зависимости

```bash
pip install torch torchvision
pip install transformers
pip install opencv-python
pip install pillow
pip install numpy
pip install scikit-learn  # Для извлечения доминирующего цвета (опционально)
```

### Модели

Модели загружаются автоматически из Hugging Face при первом использовании:
- `google/owlvit-base-patch16` (по умолчанию)
- `google/owlv2-base-patch16` (для OWL-ViT v2)

## Использование

### Базовый пример

```python
from object_detection import ObjectDetectionModule

# Инициализация модуля
detector = ObjectDetectionModule(
    model_name="google/owlvit-base-patch16",
    model_family="owlvit",  # или "owlv2"
    device="cuda",  # или "cpu"
    box_threshold=0.3
)

# Подготовка состояния (в соответствии с архитектурой проекта)
state = {
    "video_context": video_context,
    "sampler_result": {
        "global": frame_descriptors
    },
    "frame_reader": frame_reader,
    "model_registry": model_registry,
    "result_store": result_store
}

# Запуск детекции
result = detector.run(state)

if result:
    detections = result["object_detections"]
    print(f"Всего детекций: {detections['summary']['total_detections']}")
    print(f"Уникальных категорий: {detections['summary']['unique_categories']}")
```

### Детекция с кастомными категориями

```python
detector = ObjectDetectionModule(
    default_categories=["person", "car", "bicycle", "dog", "cat"]
)

# Категории можно переопределить при вызове
# (через параметр text_queries в методе _detect_objects_in_frame)
```

### Прямое использование для одного кадра

```python
import cv2

frame = cv2.imread("frame.jpg")
detections = detector._detect_objects_in_frame(
    frame=frame,
    text_queries=["person", "car", "bicycle"]
)

for det in detections:
    print(f"{det['label']}: {det['score']:.3f} at {det['bbox']}")
```

## Формат выходных данных

### Структура результата

```python
{
    "object_detections": {
        "frames": {
            frame_index: [
                {
                    "bbox": [x_min, y_min, x_max, y_max],
                    "score": float,  # Confidence score
                    "label": str,     # Category name
                    "track_id": int,  # ID трека (если включен трекинг)
                    "color": {"B": int, "G": int, "R": int},  # Доминирующий цвет
                    "semantic_tags": ["luxury", "danger", ...]  # Семантические теги
                },
                # ... другие детекции
            ],
            # ... другие кадры
        },
        "frame_metadata": {
            frame_index: {
                "density": float,  # Плотность объектов в кадре
                "num_objects": int,
                "overlapping": {
                    "num_overlapping_pairs": int,
                    "overlap_pairs": [...],
                    "overlap_ratio": float
                }
            }
        },
        "summary": {
            "total_detections": int,
            "unique_categories": int,
            "category_counts": {
                "person": int,
                "car": int,
                # ... другие категории
            },
            "semantic_tag_counts": {
                "luxury": int,
                "danger": int,
                # ... другие теги
            },
            "brand_detections": [
                {
                    "frame": int,
                    "brand": str,
                    "score": float,
                    "bbox": [x_min, y_min, x_max, y_max]
                }
            ],
            "avg_density": float,
            "avg_overlap_ratio": float
        },
        "tracking": {
            "num_tracks": int,
            "track_durations_frames": {track_id: duration},
            "track_durations_seconds": {track_id: duration},
            "avg_duration_frames": float,
            "avg_duration_seconds": float,
            "birth_events": {frame: [track_ids]},
            "death_events": {frame: [track_ids]},
            "total_births": int,
            "total_deaths": int,
            "turnover_rate": float,
            "avg_objects_per_frame": float
        },
        "tracks": [
            {
                "track_id": int,
                "label": str,
                "first_frame": int,
                "last_frame": int,
                "duration_frames": int,
                "duration_seconds": float,
                "num_detections": int,
                "semantic_tags": [str],
                "colors": [{"B": int, "G": int, "R": int}]
            }
        ],
        "frame_count": int
    }
}
```

## Параметры

### Инициализация

- **`model_name`** (str): Имя модели из Hugging Face или локальный путь
- **`model_family`** (str): "owlvit" или "owlv2"
- **`device`** (str): "cuda", "cpu" или None (автоопределение)
- **`default_categories`** (List[str]): Список категорий по умолчанию
- **`box_threshold`** (float): Порог уверенности для bounding boxes (0.0-1.0)
- **`logger`**: Опциональный логгер

### Дополнительные возможности

Модуль автоматически включает следующие функции (можно отключить через атрибуты класса):
- **Трекинг объектов**: `enable_tracking = True`
- **Детекция брендов**: `enable_brand_detection = True`
- **Семантические теги**: `enable_semantic_tags = True`
- **Атрибуты объектов**: `enable_attributes = True`

### Категории по умолчанию

Модуль включает обширный список категорий по умолчанию (80+ категорий COCO):
- Люди: person
- Транспорт: car, truck, bus, motorcycle, bicycle, airplane, boat, train
- Животные: bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe
- Предметы: backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard
- Еда: banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake
- Мебель: chair, couch, bed, dining table, toilet
- Электроника: tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven
- И другие...

## Архитектура

Модуль следует интерфейсу `BaseModule`:

- **`should_run(state)`**: Проверяет, должен ли модуль выполняться
- **`run(state)`**: Основной метод выполнения

### Интеграция с компонентами

- **FrameReader**: Для чтения кадров из видео
- **ModelRegistry**: Для управления моделями и их кэширования
- **ResultStore**: Для сохранения результатов

## Примеры использования

### Детекция конкретных объектов

```python
# Детекция только людей и автомобилей
detector = ObjectDetectionModule(
    default_categories=["person", "car", "truck", "bus"]
)

result = detector.run(state)

# Анализ результатов
for frame_idx, detections in result["object_detections"]["frames"].items():
    people = [d for d in detections if d["label"] == "person"]
    cars = [d for d in detections if d["label"] == "car"]
    print(f"Frame {frame_idx}: {len(people)} people, {len(cars)} cars")
```

### Анализ распределения объектов

```python
result = detector.run(state)
summary = result["object_detections"]["summary"]

print("Распределение объектов:")
for category, count in sorted(
    summary["category_counts"].items(),
    key=lambda x: x[1],
    reverse=True
):
    print(f"  {category}: {count}")
```

### Фильтрация по уверенности

```python
# Использование высокого порога для более точных детекций
detector = ObjectDetectionModule(box_threshold=0.5)

result = detector.run(state)
# Все детекции будут иметь score >= 0.5
```

### Анализ трекинга и метрик

```python
result = detector.run(state)
tracking = result["object_detections"]["tracking"]

print(f"Всего треков: {tracking['num_tracks']}")
print(f"Средняя длительность: {tracking['avg_duration_seconds']:.2f} сек")
print(f"Turnover rate: {tracking['turnover_rate']:.2f}")

# Анализ birth/death events
for frame, track_ids in tracking["birth_events"].items():
    print(f"Frame {frame}: появилось {len(track_ids)} объектов")

# Анализ плотности и перекрытий
summary = result["object_detections"]["summary"]
print(f"Средняя плотность: {summary['avg_density']:.3f}")
print(f"Средний overlap ratio: {summary['avg_overlap_ratio']:.3f}")
```

### Анализ брендов и семантики

```python
result = detector.run(state)
summary = result["object_detections"]["summary"]

# Найденные бренды
print("Обнаруженные бренды:")
for brand_det in summary["brand_detections"]:
    print(f"  {brand_det['brand']} (frame {brand_det['frame']}, score: {brand_det['score']:.2f})")

# Семантические теги
print("\nСемантические теги:")
for tag, count in summary["semantic_tag_counts"].items():
    print(f"  {tag}: {count}")
```

### Анализ атрибутов объектов

```python
result = detector.run(state)

# Получить цвета объектов
for frame_idx, detections in result["object_detections"]["frames"].items():
    for det in detections:
        if "color" in det:
            color = det["color"]
            print(f"{det['label']}: RGB({color['R']}, {color['G']}, {color['B']})")
```

## Производительность

- **Скорость**: ~5-15 FPS (зависит от GPU и количества категорий)
- **Точность**: Высокая точность благодаря Vision Transformer архитектуре
- **Память**: Умеренное использование памяти, оптимизировано для батчевой обработки

## Новые возможности

### 1. Трекинг объектов

Модуль использует IoU-based tracker для отслеживания объектов между кадрами:
- Автоматическое сопоставление объектов по IoU
- Устойчивость к кратковременным пропускам (max_age)
- Вычисление длительности присутствия каждого объекта
- Отслеживание birth/death events (появление/исчезновение)

### 2. Метрики плотности и перекрытий

- **Object Density**: Плотность объектов в кадре (отношение покрытой площади к общей)
- **Overlapping Analysis**: Анализ перекрытий между объектами (IoU-based)
- **Turnover Rate**: Скорость появления/исчезновения объектов

### 3. Детекция брендов

Автоматическая детекция логотипов популярных брендов:
- Технологические: Apple, Samsung, Google, Microsoft, Amazon
- Одежда: Nike, Adidas, Gucci, Prada, Versace, Chanel
- Еда и напитки: Coca Cola, Pepsi, McDonald's, Starbucks
- Автомобили: Tesla, BMW, Mercedes, Audi, Toyota, Ford
- И другие популярные бренды

### 4. Семантическая классификация

Автоматическое добавление семантических тегов к объектам:
- **luxury**: Роскошные предметы (дорогие автомобили, часы, сумки)
- **danger**: Опасные объекты (нож, оружие, острые предметы)
- **cute**: Милые объекты (животные, игрушки)
- **sport**: Спортивное оборудование
- **food**: Еда и напитки
- **technology**: Электронные устройства

### 5. Атрибуты объектов

- **Цвет**: Извлечение доминирующего цвета объекта (K-means или среднее значение)
- Возможность расширения для других атрибутов (стиль, размер, форма)

## Ограничения

- Требует GPU для оптимальной производительности
- Может быть медленным при большом количестве категорий
- Точность зависит от качества текстовых описаний
- Лучше работает с четкими, хорошо освещенными объектами
- Трекинг может терять объекты при сильных окклюзиях или быстром движении
- Детекция брендов зависит от качества и размера логотипов в кадре

## Сравнение OWL-ViT и OWL-ViT v2

- **OWL-ViT**: Оригинальная версия, быстрее
- **OWL-ViT v2**: Улучшенная версия с лучшей точностью, но медленнее

Рекомендуется использовать OWL-ViT v2 для лучшей точности, если производительность не критична.

## Интеграция с другими модулями

Результаты детекции объектов могут быть использованы для:
- Анализа композиции кадра (модуль `frames_composition`)
- Оценки качества кадра (модуль `shot_quality`)
- Семантического анализа (модуль `high_level_semantic`)

## Дополнительные ресурсы

- [OWL-ViT Paper](https://arxiv.org/abs/2205.06230)
- [Hugging Face OWL-ViT](https://huggingface.co/docs/transformers/model_doc/owlvit)
- [COCO Dataset](https://cocodataset.org/)

## Лицензия

См. основной файл лицензии проекта.

