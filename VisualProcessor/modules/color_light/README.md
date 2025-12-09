# 🎨 Color & Light Processor

## 📌 Описание

Модуль для комплексного анализа цвета и освещения видео. Извлекает покадровые, сценовые и видеоуровневые фичи для последующего использования в моделях машинного обучения.

## 📋 Содержание

- [Установка](#установка)
- [Быстрый старт](#быстрый-старт)
- [Архитектура](#архитектура)
- [API Документация](#api-документация)
- [Описание фич](#описание-фич)
- [Форматы данных](#форматы-данных)
- [Примеры использования](#примеры-использования)
- [Оптимизация](#оптимизация)

## 🚀 Установка

```bash
pip install -r requirements.txt
```

### Зависимости

- `numpy >= 1.21.0`
- `opencv-python >= 4.5.0`
- `scipy >= 1.7.0`
- `scikit-learn >= 1.0.0`

## ⚡ Быстрый старт

### CLI использование

Самый простой способ обработки видео:

```bash
python main.py --input video.mp4 --output results.json
```

**Параметры CLI:**

```bash
python main.py \
    --input video.mp4 \                    # Входное видео (обязательно)
    --output results.json \                 # Выходной JSON файл (опционально)
    --max-frames-per-scene 350 \           # Максимальное количество кадров на сцену
    --scene-duration 5.0 \                 # Длительность сцены в секундах
    --scenes scenes.json \                 # JSON файл со сценами (опционально)
    --frames-dir ./frames \                # Директория с кадрами (если уже извлечены)
    --keep-frames                          # Сохранить извлеченные кадры
```

### Программный интерфейс

```python
from processor import FrameManager, ColorLightProcessor

# Инициализация FrameManager (требует директорию с кадрами и metadata.json)
frame_manager = FrameManager(frames_dir="path/to/frames", cache_size=2)

# Инициализация процессора
processor = ColorLightProcessor(max_frames_per_scene=350)

# Подготовка входных данных
input_data = {
    "total_frames": 1565,
    "scenes": {
        1: [1, 30],      # Сцена 1: кадры с 1 по 30
        2: [31, 120]     # Сцена 2: кадры с 31 по 120
    }
}

# Обработка видео
result = processor.process(
    frame_manager=frame_manager,
    input_data=input_data,
    video_id="video_001"
)

# Результат содержит:
# - result["frames"] - покадровые фичи
# - result["scenes"] - сценовые фичи
# - result["video_features"] - видеоуровневые фичи
# - result["sequence_inputs"] - последовательности для трансформера
```

## 🏗️ Архитектура

Модуль состоит из двух основных классов:

### FrameManager

Управляет доступом к кадрам видео через memory-mapped файлы. Использует LRU кэширование для оптимизации производительности.

**Особенности:**
- Поддержка батчей кадров в raw формате
- LRU кэширование для минимизации I/O операций
- Автоматическое чтение метаданных из `metadata.json`

### ColorLightProcessor

Основной процессор для извлечения фич цвета и освещения.

**Уровни анализа:**
1. **Frame-level** - покадровые фичи (цвет, освещение, палитра)
2. **Scene-level** - временные паттерны и динамика внутри сцен
3. **Video-level** - агрегированные фичи всего видео

## 📚 API Документация

### FrameManager

#### `__init__(frames_dir: str, chunk_size: int = 64, cache_size: int = 2)`

Инициализирует менеджер кадров.

**Параметры:**
- `frames_dir` - путь к директории с кадрами (должна содержать `metadata.json`)
- `chunk_size` - размер батча кадров (по умолчанию 64)
- `cache_size` - количество батчей в LRU кэше (по умолчанию 2)

**Требования к metadata.json:**
```json
{
    "total_frames": 1565,
    "batch_size": 64,
    "height": 720,
    "width": 1280,
    "channels": 3,
    "fps": 30,
    "batches": [
        {
            "batch_index": 0,
            "path": "batch_0.raw",
            "start_frame": 0,
            "end_frame": 63
        }
    ]
}
```

#### `get(idx: int) -> np.ndarray`

Получает кадр по индексу.

**Параметры:**
- `idx` - индекс кадра (0-based)

**Возвращает:**
- `np.ndarray` формы `(height, width, channels)` типа `uint8`

#### `close()`

Очищает кэш. Рекомендуется вызывать после завершения обработки видео.

### ColorLightProcessor

#### `__init__(max_frames_per_scene: int = 350)`

Инициализирует процессор.

**Параметры:**
- `max_frames_per_scene` - максимальное количество кадров для обработки на сцену (для оптимизации)

#### `process(frame_manager: FrameManager, input_data: Dict[str, Any], video_id: Optional[str] = None) -> Dict[str, Any]`

Главный метод обработки видео.

**Параметры:**
- `frame_manager` - экземпляр FrameManager для доступа к кадрам
- `input_data` - словарь с входными данными:
  ```python
  {
      "total_frames": 1565,
      "scenes": {
          1: [start_frame, end_frame],
          2: [start_frame, end_frame]
      }
  }
  ```
- `video_id` - идентификатор видео (опционально)

**Возвращает:**
Словарь с результатами (см. [Форматы данных](#форматы-данных))

#### `extract_frame_features(frame: np.ndarray, frame_idx: int, fps: float) -> Dict[str, Any]`

Извлекает покадровые фичи для одного кадра.

**Параметры:**
- `frame` - кадр как numpy array `(height, width, channels)`
- `frame_idx` - индекс кадра
- `fps` - частота кадров

**Возвращает:**
```python
{
    "timestamp": 1.2,
    "features": {
        # RGB статистики
        "color_mean_r": 125.5,
        "color_std_r": 45.2,
        # ... и т.д.
    }
}
```

#### `extract_scene_features(frame_features: List[Dict], scene_start: int, scene_end: int, fps: float) -> Dict[str, Any]`

Извлекает сценовые фичи из списка покадровых фич.

**Параметры:**
- `frame_features` - список покадровых фич
- `scene_start` - индекс начального кадра сцены
- `scene_end` - индекс конечного кадра сцены
- `fps` - частота кадров

#### `extract_video_features(all_scene_features: List[Dict], all_frame_features: List[Dict]) -> Dict[str, Any]`

Извлекает видеоуровневые агрегированные фичи.

## 🎨 Описание фич

### 1. Frame-level Features (Покадровые фичи)

#### Цветовые статистики (RGB)
Для каждого канала (R, G, B):
- `color_mean_{r/g/b}` - среднее значение
- `color_std_{r/g/b}` - стандартное отклонение
- `color_min_{r/g/b}` - минимальное значение
- `color_max_{r/g/b}` - максимальное значение
- `color_skew_{r/g/b}` - асимметрия (skewness)
- `color_kurt_{r/g/b}` - эксцесс (kurtosis)

**Применение:** Анализ баланса белого, экспозиции, общего освещения.

#### Цветовые пространства

**HSV:**
- `hue_mean`, `hue_std`, `hue_entropy` - статистики оттенка
- `saturation_mean`, `saturation_std` - статистики насыщенности
- `value_mean`, `value_std` - статистики яркости

**LAB:**
- `L_mean` - средняя яркость (lightness)
- `L_contrast` - контраст яркости
- `ab_balance` - баланс теплых/холодных тонов (положительное = тепло)

#### Палитра и гармонии
- `dominant_color` - доминирующий цвет `[R, G, B]`
- `secondary_color` - вторичный цвет `[R, G, B]`
- `tertiary_color` - третичный цвет `[R, G, B]`
- `palette_size` - количество уникальных цветов в палитре
- `colorfulness_index` - индекс цветности (насыщенность цветов)
- `warm_vs_cold_ratio` - соотношение теплых/холодных тонов
- `skin_tone_ratio` - доля пикселей цвета кожи
- `color_palette_entropy` - энтропия цветовой палитры
- `color_harmony_complementary_prob` - вероятность комплементарной гармонии (противоположные цвета)
- `color_harmony_analogous_prob` - вероятность аналогичной гармонии (соседние цвета)
- `color_harmony_triadic_prob` - вероятность триадной гармонии (три равномерно распределенных цвета)
- `color_harmony_split_complementary_prob` - вероятность расщепленной комплементарной гармонии

#### Освещение
- `brightness_mean`, `brightness_std`, `brightness_entropy` - статистики яркости
- `overexposed_pixels` - доля переэкспонированных пикселей (>250)
- `underexposed_pixels` - доля недоэкспонированных пикселей (<5)
- `global_contrast` - глобальный контраст (RMS contrast)
- `local_contrast`, `local_contrast_std` - локальный контраст (среднее и std по окнам)
- `contrast_entropy` - энтропия контраста
- `dynamic_range_db` - динамический диапазон в децибелах (HDR score)
- `highlight_clipping_ratio` - доля обрезанных светлых участков
- `shadow_clipping_ratio` - доля обрезанных теневых участков
- `lighting_uniformity_index` - индекс равномерности освещения (0-1, 1 = равномерное)
- `center_brightness` - яркость центра кадра
- `corner_brightness` - средняя яркость углов кадра
- `vignetting_score` - оценка виньетирования (0-1, 0 = нет виньетирования, 1 = сильное)

#### Направление света
- `light_direction_angle` - угол направления основного источника света (градусы)
- `light_source_count_estimate` - оценка количества источников света
- `soft_light_probability` - вероятность мягкого света
- `hard_light_probability` - вероятность жесткого света

### 2. Scene-level Features (Сценовые фичи)

#### Motion + Lighting
- `brightness_change_speed` - скорость изменения яркости
- `color_change_speed` - скорость изменения цвета
- `scene_flicker_intensity` - интенсивность мерцания сцены
- `flash_events_count` - количество событий вспышки
- `color_transition_variance` - вариативность цветовых переходов

#### Temporal Color Patterns
- `color_stability` - стабильность цвета (обратная метрика изменчивости)
- `color_temporal_entropy` - временная энтропия цвета
- `color_pattern_periodicity` - периодичность цветовых паттернов
- `scene_color_shift_speed` - скорость цветового сдвига

#### Агрегированные показатели
- `{feature}_mean`, `{feature}_std` - среднее и стандартное отклонение для каждой покадровой фичи
- `scene_contrast` - средний контраст сцены
- `dynamic_range` - динамический диапазон яркости

#### Метаданные
- `num_frames` - количество обработанных кадров сцены
- `duration` - длительность сцены в секундах
- `start`, `end` - временные метки начала и конца сцены

### 3. Video-level Features (Видеоуровневые фичи)

#### Агрегация всех сцен
Для каждой сценовой фичи:
- `{feature}_mean`, `{feature}_std`, `{feature}_min`, `{feature}_max`

#### Распределения
- `color_distribution_entropy` - энтропия распределения цветов
- `color_distribution_gini` - коэффициент Джини для распределения цветов

#### Стиль цветокоррекции
Вероятности различных стилей (0.0 - 1.0):
- `style_teal_orange_prob` - Teal & Orange (теплые и холодные тона)
- `style_film_prob` - Film look (низкая насыщенность, мягкие тона)
- `style_desaturated_prob` - Низкая насыщенность
- `style_hyper_saturated_prob` - Высокая насыщенность
- `style_vintage_prob` - Винтажный стиль (сепия-подобные тона)
- `style_tiktok_prob` - TikTok стиль (высокая насыщенность, яркие цвета)

#### Aesthetic & Cinematic Scores
- `nima_mean`, `nima_std` - оценки эстетики (на основе контраста)
- `laion_mean`, `laion_std` - оценки эстетики (на основе цветности)
- `cinematic_lighting_score` - оценка кинематографического освещения
- `professional_look_score` - оценка профессионального вида

#### Глобальная динамика
- `global_brightness_change_speed` - глобальная скорость изменения яркости
- `global_color_change_speed` - глобальная скорость изменения цвета
- `strobe_transition_frequency` - частота стробоскопических переходов
- `global_color_periodicity` - глобальная периодичность цвета
- `global_color_shift` - глобальный цветовой сдвиг

## 📊 Форматы данных

### Входные данные

```python
input_data = {
    "total_frames": 1565,  # Общее количество кадров
    "scenes": {             # Словарь сцен: scene_id -> [start_frame, end_frame]
        1: [1, 30],
        2: [31, 120],
        3: [121, 500]
    }
}
```

### Выходные данные

```python
{
    "video_id": "video_001",
    
    # Покадровые фичи
    "frames": [
        {
            "timestamp": 0.033,  # Время в секундах
            "features": {
                "color_mean_r": 125.5,
                "color_mean_g": 130.2,
                "color_mean_b": 128.9,
                # ... все frame-level фичи
            }
        },
        # ... для каждого обработанного кадра
    ],
    
    # Сценовые фичи
    "scenes": [
        {
            "start": 0.0,      # Начало сцены (секунды)
            "end": 1.0,        # Конец сцены (секунды)
            "num_frames": 30,
            "duration": 1.0,
            "brightness_change_speed": 2.5,
            "color_change_speed": 1.8,
            # ... все scene-level фичи
        },
        # ... для каждой сцены
    ],
    
    # Видеоуровневые фичи
    "video_features": {
        "brightness_change_speed_mean": 2.3,
        "brightness_change_speed_std": 0.5,
        "style_teal_orange_prob": 0.7,
        "cinematic_lighting_score": 0.85,
        # ... все video-level фичи
    },
    
    # Последовательности для трансформера
    "sequence_inputs": {
        "frames": [
            [125.5, 130.2, 128.9, ...],  # Вектор фич для кадра 1
            [126.1, 131.0, 129.5, ...],  # Вектор фич для кадра 2
            # ... [N_frames, D_frame_features]
        ],
        "scenes": [
            [30, 1.0, 2.5, 1.8, ...],    # Вектор фич для сцены 1
            [90, 3.0, 2.1, 1.5, ...],    # Вектор фич для сцены 2
            # ... [N_scenes, D_scene_features]
        ],
        "global": [
            2.3, 0.5, 0.7, 0.85, ...     # Глобальные фичи
            # [D_global_features]
        ]
    }
}
```

## 💡 Примеры использования

### CLI пример

```bash
# Базовое использование
python main.py --input video.mp4 --output results.json

# С настройками
python main.py \
    --input video.mp4 \
    --output results.json \
    --max-frames-per-scene 200 \
    --scene-duration 5.0

# С использованием существующих кадров
python main.py \
    --input video.mp4 \
    --frames-dir ./extracted_frames \
    --output results.json

# С сохранением кадров
python main.py \
    --input video.mp4 \
    --output results.json \
    --keep-frames
```

### Программный интерфейс

```python
from processor import FrameManager, ColorLightProcessor

# Инициализация
frame_manager = FrameManager("frames_dir")
processor = ColorLightProcessor(max_frames_per_scene=200)

# Входные данные
input_data = {
    "total_frames": 1000,
    "scenes": {
        1: [0, 299],
        2: [300, 599],
        3: [600, 999]
    }
}

# Обработка
result = processor.process(frame_manager, input_data, video_id="test_video")

# Использование результатов
print(f"Обработано кадров: {len(result['frames'])}")
print(f"Обработано сцен: {len(result['scenes'])}")
print(f"Cinematic score: {result['video_features']['cinematic_lighting_score']}")

# Очистка
frame_manager.close()
```

### Извлечение конкретных фич

```python
# Получить все покадровые фичи яркости
brightness_values = [
    frame['features']['brightness_mean'] 
    for frame in result['frames']
]

# Получить доминирующие цвета всех кадров
dominant_colors = [
    frame['features']['dominant_color']
    for frame in result['frames']
]

# Получить стиль цветокоррекции
style_probs = {
    'teal_orange': result['video_features']['style_teal_orange_prob'],
    'film': result['video_features']['style_film_prob'],
    'vintage': result['video_features']['style_vintage_prob']
}

# Получить цветовые гармонии (новое)
harmony_probs = {
    'complementary': result['frames'][0]['features'].get('color_harmony_complementary_prob', 0),
    'analogous': result['frames'][0]['features'].get('color_harmony_analogous_prob', 0),
    'triadic': result['frames'][0]['features'].get('color_harmony_triadic_prob', 0),
    'split_complementary': result['frames'][0]['features'].get('color_harmony_split_complementary_prob', 0)
}

# Получить lighting uniformity фичи (новое)
uniformity_features = {
    'uniformity_index': result['frames'][0]['features'].get('lighting_uniformity_index', 0),
    'vignetting_score': result['frames'][0]['features'].get('vignetting_score', 0),
    'center_brightness': result['frames'][0]['features'].get('center_brightness', 0),
    'corner_brightness': result['frames'][0]['features'].get('corner_brightness', 0)
}

# Получить Dynamic Range фичи (улучшенное)
dynamic_range_features = {
    'dynamic_range_db': result['frames'][0]['features'].get('dynamic_range_db', 0),
    'highlight_clipping_ratio': result['frames'][0]['features'].get('highlight_clipping_ratio', 0),
    'shadow_clipping_ratio': result['frames'][0]['features'].get('shadow_clipping_ratio', 0)
}
```

### Использование sequence inputs для трансформера

```python
import numpy as np

# Получить последовательности
frame_sequences = np.array(result['sequence_inputs']['frames'])
scene_sequences = np.array(result['sequence_inputs']['scenes'])
global_features = np.array(result['sequence_inputs']['global'])

# Использовать в модели
# frame_sequences: [N_frames, D_frame_features]
# scene_sequences: [N_scenes, D_scene_features]
# global_features: [D_global_features]
```

### Обработка с ограничением кадров

```python
# Для длинных видео ограничиваем количество кадров на сцену
processor = ColorLightProcessor(max_frames_per_scene=100)

# Процессор автоматически выберет равномерно распределенные кадры
result = processor.process(frame_manager, input_data)
```

## ⚙️ Оптимизация

### Рекомендации по производительности

1. **Ограничение кадров на сцену:**
   - По умолчанию: 350 кадров
   - Для длинных видео: 100-200 кадров
   - Для коротких видео: можно увеличить до 500

2. **Кэширование FrameManager:**
   - `cache_size=2` - для малых видео
   - `cache_size=4-8` - для больших видео (больше RAM)

3. **Обработка по сценам:**
   - Сложные вычисления (палитра, гармонии) выполняются только на frame-level
   - Scene-level и video-level используют агрегацию уже вычисленных фич

4. **Параллелизация:**
   - Можно обрабатывать разные сцены параллельно
   - Каждая сцена независима

### Ограничения

- Покадровые вычисления ограничены до `max_frames_per_scene` кадров на сцену
- Сложные показатели (color harmony, dynamic range) вычисляются только на scene-level
- Для очень больших видео рекомендуется предварительная сегментация на сцены

## 🔧 Расширение функциональности

### Добавление новых фич

Для добавления новых покадровых фич:

```python
def _compute_custom_features(self, frame: np.ndarray) -> Dict[str, float]:
    """Ваши кастомные фичи"""
    features = {}
    # ... вычисления
    return features

# Добавить в extract_frame_features:
def extract_frame_features(self, frame, frame_idx, fps):
    features = {}
    # ... существующие фичи
    features.update(self._compute_custom_features(frame))
    return {"timestamp": frame_idx / fps, "features": features}
```

### Кастомизация стилей

Метод `_compute_color_style_features` можно расширить для добавления новых стилей цветокоррекции.

## 📝 Примечания

- Кадры должны быть в RGB формате (если используются BGR кадры, добавьте конвертацию)
- Все временные метки в секундах
- Индексы кадров начинаются с 0
- Процессор автоматически обрабатывает ошибки при чтении кадров (пропускает проблемные кадры с предупреждением)

## 🐛 Troubleshooting

### Ошибка: "metadata.json not found"
Убедитесь, что в директории с кадрами есть файл `metadata.json` с правильной структурой.

### Ошибка: "Frame index out of bounds"
Проверьте, что индексы кадров в `scenes` не превышают `total_frames - 1`.

### Медленная обработка
- Уменьшите `max_frames_per_scene`
- Увеличьте `cache_size` в FrameManager
- Убедитесь, что кадры хранятся на быстром диске

## 📄 Лицензия

См. основной файл лицензии проекта.
