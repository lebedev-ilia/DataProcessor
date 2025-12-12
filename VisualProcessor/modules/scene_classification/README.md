# Scene Classification Module

Модуль для классификации сцен в видео с использованием моделей Places365. Поддерживает различные архитектуры (ResNet, EfficientNet, ConvNeXt, ViT) и продвинутые техники улучшения качества (TTA, multi-crop, temporal smoothing).

## Описание

Модуль `scene_classification` предназначен для классификации сцен в видео кадрах. Использует предобученные модели на датасете Places365 (365 категорий сцен) и поддерживает как оригинальные Places365 модели, так и современные архитектуры через библиотеку timm.

## Основные возможности

- **365 категорий сцен**: Классификация по датасет Places365
- **Множество архитектур**: ResNet, EfficientNet, ConvNeXt, Vision Transformer
- **Test-Time Augmentation (TTA)**: Улучшение точности через аугментации
- **Multi-crop inference**: Анализ нескольких областей кадра
- **Temporal smoothing**: Сглаживание предсказаний для видео
- **Batch processing**: Эффективная обработка множества кадров
- **GPU memory management**: Автоматическое управление памятью GPU
- **Indoor/Outdoor классификация**: Автоматическое определение типа помещения
- **Nature/Urban разделение**: Классификация природных и городских сцен
- **Time of Day Detection**: Определение времени суток (утро, день, вечер, ночь)
- **Aesthetic Score**: Оценка эстетической привлекательности сцены
- **Luxury Score**: Оценка роскошности/премиальности сцены
- **Atmosphere Sentiment**: Определение атмосферы (cozy, scary, epic, neutral)
- **Геометрические фичи**: Openness, clutter, depth cues

## Установка

### Базовые зависимости

```bash
pip install torch torchvision
pip install opencv-python
pip install pillow
pip install numpy
pip install requests
```

### Для современных архитектур (опционально)

```bash
pip install timm
```

### Для продвинутых семантических фичей (опционально)

```bash
pip install transformers
```

Примечание: CLIP используется для более точных aesthetic/luxury scores и atmosphere sentiment. Без него используются эвристические методы.

## Использование

### Базовый пример

```python
from scene_classification import Places365SceneClassifier
import cv2

# Инициализация классификатора
classifier = Places365SceneClassifier(
    model_arch="resnet50",
    use_timm=False,  # Использовать оригинальные Places365 модели
    top_k=5,
    device="cuda"
)

# Загрузка кадров
frames = [cv2.imread(f"frame_{i}.jpg") for i in range(10)]

# Классификация
predictions = classifier.classify(frames, top_k=5)

# Результаты для каждого кадра
for i, frame_preds in enumerate(predictions):
    print(f"Frame {i}:")
    for pred in frame_preds:
        print(f"  {pred['label']}: {pred['score']:.3f}")
```

### Использование современных архитектур (timm)

```python
classifier = Places365SceneClassifier(
    model_arch="efficientnet_b0",
    use_timm=True,  # Использовать timm модели
    top_k=10,
    device="cuda"
)

predictions = classifier(frames)  # __call__ = classify
```

### С Test-Time Augmentation

```python
classifier = Places365SceneClassifier(
    model_arch="resnet50",
    use_tta=True,  # Включить TTA
    top_k=5
)

predictions = classifier.classify(frames)
# Предсказания усредняются по нескольким аугментациям
```

### С Multi-Crop

```python
classifier = Places365SceneClassifier(
    model_arch="resnet50",
    use_multi_crop=True,  # Анализ 5 crops (center + 4 corners)
    top_k=5
)

predictions = classifier.classify(frames)
# Предсказания усредняются по 5 crops
```

### С Temporal Smoothing (для видео)

```python
classifier = Places365SceneClassifier(
    model_arch="resnet50",
    temporal_smoothing=True,
    smoothing_window=5,  # Окно сглаживания (кадры)
    top_k=5
)

# Обработка последовательности кадров
predictions = classifier.classify(video_frames)
# Предсказания сглаживаются по времени
```

### С продвинутыми фичами

```python
classifier = Places365SceneClassifier(
    model_arch="resnet50",
    enable_advanced_features=True,  # Включить продвинутые фичи
    use_clip_for_semantics=True,     # Использовать CLIP для семантики
    top_k=5
)

# Использование метода с продвинутыми фичами
results = classifier.classify_with_advanced_features(frames)

for result in results:
    predictions = result["predictions"]
    advanced = result["advanced_features"]
    
    if advanced:
        print(f"Indoor/Outdoor: {advanced['indoor_outdoor']}")
        print(f"Nature/Urban: {advanced['nature_urban']}")
        print(f"Time of Day: {advanced['time_of_day']}")
        print(f"Aesthetic Score: {advanced['aesthetic_score']:.3f}")
        print(f"Luxury Score: {advanced['luxury_score']:.3f}")
        print(f"Atmosphere: {advanced['atmosphere_sentiment']}")
        print(f"Geometric: {advanced['geometric_features']}")
```

## Параметры

### Инициализация

- **`model_arch`** (str): Архитектура модели
  - Places365: `"resnet18"`, `"resnet50"`
  - timm: `"efficientnet_b0"`, `"convnext_tiny"`, `"vit_base_patch16_224"`, и др.
- **`use_timm`** (bool): Использовать библиотеку timm для современных архитектур
- **`top_k`** (int): Количество топ предсказаний для возврата
- **`batch_size`** (int): Размер батча для обработки
- **`device`** (str): "cuda", "cpu" или None (автоопределение)
- **`input_size`** (int): Размер входного изображения (224, 256, 320, etc.)
- **`use_tta`** (bool): Включить Test-Time Augmentation
- **`use_multi_crop`** (bool): Включить multi-crop inference
- **`temporal_smoothing`** (bool): Включить временное сглаживание
- **`smoothing_window`** (int): Размер окна для temporal smoothing
- **`categories_path`** (str): Путь к файлу категорий (опционально)
- **`cache_dir`** (str): Директория для кэширования файлов
- **`enable_advanced_features`** (bool): Включить продвинутые фичи (indoor/outdoor, time of day, etc.)
- **`use_clip_for_semantics`** (bool): Использовать CLIP для семантических фичей (aesthetic, luxury, atmosphere)

### Поддерживаемые архитектуры (timm)

- **EfficientNet**: `efficientnet_b0`, `efficientnet_b1`, `efficientnet_b2`, `efficientnet_b3`
- **ConvNeXt**: `convnext_tiny`, `convnext_small`, `convnext_base`
- **Vision Transformer**: `vit_base_patch16_224`, `vit_large_patch16_224`
- **RegNet**: `regnetx_002`, `regnetx_004`, `regnetx_006`
- **ResNet**: `resnet50`, `resnet101`

## Формат выходных данных

### Базовый результат (classify)

```python
[
    [  # Предсказания для первого кадра
        {"label": "bedroom", "score": 0.85},
        {"label": "living_room", "score": 0.12},
        {"label": "dining_room", "score": 0.03},
        # ... до top_k предсказаний
    ],
    [  # Предсказания для второго кадра
        {"label": "kitchen", "score": 0.92},
        {"label": "dining_room", "score": 0.06},
        # ...
    ],
    # ... для каждого кадра
]
```

### Расширенный результат (classify_with_advanced_features)

```python
[
    {
        "predictions": [
            {"label": "bedroom", "score": 0.85},
            # ... до top_k предсказаний
        ],
        "advanced_features": {
            "indoor_outdoor": {
                "indoor": 0.9,
                "outdoor": 0.1
            },
            "nature_urban": {
                "nature": 0.2,
                "urban": 0.8
            },
            "time_of_day": {
                "morning": 0.1,
                "day": 0.7,
                "evening": 0.15,
                "night": 0.05
            },
            "aesthetic_score": 0.75,
            "luxury_score": 0.6,
            "atmosphere_sentiment": {
                "cozy": 0.5,
                "scary": 0.1,
                "epic": 0.2,
                "neutral": 0.2
            },
            "geometric_features": {
                "openness": 0.3,
                "clutter": 0.4,
                "depth_cues": 0.5
            }
        }
    },
    # ... для каждого кадра
]
```

## Категории Places365

Модуль поддерживает 365 категорий сцен, включая:

- **Внутренние помещения**: bedroom, living_room, kitchen, bathroom, dining_room, etc.
- **Общественные места**: airport, train_station, bus_station, shopping_mall, etc.
- **Природа**: forest, beach, mountain, desert, lake, etc.
- **Городские сцены**: street, plaza, park, bridge, etc.
- **Специализированные**: hospital, school, office, restaurant, etc.

Полный список категорий загружается автоматически из репозитория Places365.

## Архитектура

Модуль наследуется от `BaseExtractor` и предоставляет:

- **`classify(frames, top_k)`**: Основной метод классификации
- **`__call__(frames)`**: Алиас для `classify`
- Автоматическое управление памятью GPU
- Логирование метрик производительности

### Компоненты

- **Preprocessing**: Нормализация и аугментация изображений
- **Model loading**: Загрузка предобученных весов
- **Inference**: Батчевая обработка с оптимизацией памяти
- **Post-processing**: Усреднение предсказаний (TTA/multi-crop)

## Примеры использования

### Анализ видео

```python
import cv2

# Загрузка видео
cap = cv2.VideoCapture("video.mp4")
frames = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)

cap.release()

# Классификация с временным сглаживанием
classifier = Places365SceneClassifier(
    model_arch="resnet50",
    temporal_smoothing=True,
    smoothing_window=10,
    top_k=3
)

predictions = classifier.classify(frames)

# Анализ доминирующих сцен
scene_counts = {}
for frame_preds in predictions:
    if frame_preds:
        top_scene = frame_preds[0]["label"]
        scene_counts[top_scene] = scene_counts.get(top_scene, 0) + 1

print("Распределение сцен:")
for scene, count in sorted(scene_counts.items(), key=lambda x: x[1], reverse=True):
    print(f"  {scene}: {count} кадров")
```

### Сравнение архитектур

```python
architectures = ["resnet50", "efficientnet_b0", "convnext_tiny"]

for arch in architectures:
    classifier = Places365SceneClassifier(
        model_arch=arch,
        use_timm=(arch != "resnet50"),
        top_k=5
    )
    
    predictions = classifier.classify(frames)
    # Сравнение результатов
```

## Производительность

### Скорость (FPS)

- **ResNet50**: ~30-50 FPS (GPU)
- **EfficientNet-B0**: ~40-60 FPS (GPU)
- **ConvNeXt-Tiny**: ~25-40 FPS (GPU)
- **ViT-Base**: ~15-25 FPS (GPU)

### Точность

- **Places365 ResNet50**: Высокая точность на Places365
- **timm модели**: Модели предобучены на ImageNet, могут требовать fine-tuning на Places365 для оптимальных результатов

### Оптимизация

- **Batch processing**: Увеличьте `batch_size` для лучшей производительности
- **Input size**: Меньший размер (224) быстрее, больший (320) точнее
- **TTA/Multi-crop**: Улучшают точность, но замедляют обработку

## Продвинутые фичи

### 1. Indoor/Outdoor классификация

Автоматически определяет, является ли сцена внутренним помещением или внешней средой на основе Places365 категорий.

### 2. Nature/Urban разделение

Классифицирует сцену как природную или городскую на основе ключевых слов в названии категории.

### 3. Time of Day Detection

Определяет время суток (утро, день, вечер, ночь) на основе:
- Яркости изображения
- Цветовой температуры (теплые/холодные цвета)
- Распределения цветов

### 4. Aesthetic Score

Оценивает эстетическую привлекательность сцены:
- **С CLIP**: Использует семантическое сравнение с текстовыми описаниями
- **Без CLIP**: Использует эвристики (резкость, контраст, цветность, баланс яркости)

### 5. Luxury Score

Оценивает роскошность/премиальность сцены:
- **С CLIP**: Семантическое сравнение с текстами о роскоши
- **Без CLIP**: Анализ качества изображения и ключевых слов

### 6. Atmosphere Sentiment

Определяет атмосферу сцены:
- **Cozy**: Уютная, теплая атмосфера
- **Scary**: Страшная, пугающая атмосфера
- **Epic**: Эпическая, величественная атмосфера
- **Neutral**: Нейтральная атмосфера

### 7. Геометрические фичи

- **Openness**: Мера открытости пространства (видимость неба/горизонта)
- **Clutter**: Мера визуальной сложности/загроможденности
- **Depth Cues**: Признаки глубины (градиенты, перспектива)

## Ограничения

- Модели timm предобучены на ImageNet, не на Places365 (может потребоваться fine-tuning)
- TTA и multi-crop увеличивают время обработки в 2-5 раз
- Требует GPU для оптимальной производительности
- Категории фиксированы (365 классов Places365)
- CLIP увеличивает время обработки, но улучшает точность семантических фичей
- Геометрические фичи используют упрощенные эвристики (не требуют depth estimation моделей)

## Рекомендации

1. **Для скорости**: Используйте `efficientnet_b0` с `use_timm=True`
2. **Для точности**: Используйте `resnet50` с Places365 весами или `convnext_base` с fine-tuning
3. **Для видео**: Включите `temporal_smoothing` для более стабильных результатов
4. **Для сложных сцен**: Используйте `use_tta=True` или `use_multi_crop=True`
5. **Для семантических фичей**: Установите `transformers` и используйте `use_clip_for_semantics=True` для более точных aesthetic/luxury scores и atmosphere detection
6. **Для базовых фичей**: Можно использовать `enable_advanced_features=True` без CLIP - будут использоваться эвристические методы

## Интеграция с другими модулями

Результаты классификации сцен могут быть использованы для:
- Семантического анализа (модуль `high_level_semantic`)
- Оценки качества кадра (модуль `shot_quality`)
- Анализа структуры истории (модуль `story_structure`)

## Дополнительные ресурсы

- [Places365 Dataset](http://places2.csail.mit.edu/)
- [timm Library](https://github.com/rwightman/pytorch-image-models)
- [Places365 Categories](https://github.com/CSAILVision/places365)

## Лицензия

См. основной файл лицензии проекта.

