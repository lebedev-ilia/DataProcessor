# ⭐ Uniqueness Module

## 📌 Описание

Модуль для вычисления степени уникальности видео по различным аспектам. Сравнивает текущее видео с набором референсных/топовых видео и вычисляет novelty scores (метрики новизны) по 8 основным категориям.

**Применение:**
- Определение уникальности контента
- Анализ новизны по сравнению с популярными видео
- Оценка новаторства в нише
- Генерация фичей для ML-моделей предсказания популярности
- Выявление ранних трендов и уникальных паттернов

## 📋 Содержание

- [Установка](#установка)
- [Быстрый старт](#быстрый-старт)
- [Архитектура](#архитектура)
- [API Документация](#api-документация)
- [Категории метрик](#категории-метрик)
- [Форматы данных](#форматы-данных)
- [Примеры использования](#примеры-использования)

## 🚀 Установка

```bash
pip install numpy scipy scikit-learn
```

### Зависимости

- `numpy >= 1.21.0`
- `scipy >= 1.7.0`
- `scikit-learn >= 1.0.0`

## ⚡ Быстрый старт

```python
from uniqueness import UniquenessModule
import numpy as np

# Инициализация
uniqueness = UniquenessModule(top_n=100)

# Embeddings текущего и референсных видео
video_embedding = np.random.randn(512)  # Embedding текущего видео
reference_embeddings = [np.random.randn(512) for _ in range(10)]  # Референсные видео

# Темы
video_topics = ["cooking", "tutorial", "food", "unique_recipe"]
reference_topics_list = [
    ["cooking", "recipe"],
    ["gaming", "tutorial"],
    ["cooking", "food", "tutorial"]
]

# Вычисление метрик уникальности
result = uniqueness.extract_all(
    video_embedding=video_embedding,
    reference_embeddings=reference_embeddings,
    video_topics=video_topics,
    reference_topics_list=reference_topics_list
)

# Результат содержит все метрики уникальности
print(result['features']['semantic_novelty_score'])
print(result['features']['overall_novelty_index'])
print(result['features']['early_adopter_score'])
```

## 🏗️ Архитектура

Модуль реализует 8 категорий метрик уникальности:

### A. Semantic / Content Novelty
Семантическая новизна контента на основе embeddings и тематических концептов

### B. Visual / Style Novelty
Визуальная новизна стиля: цвет, свет, композиция, типы кадров, движение камеры

### C. Editing & Pacing Novelty
Новизна монтажа и ритма: cut rate, shot duration, scene length, pacing patterns

### D. Audio Novelty
Аудио новизна: музыка, голос, звуковые эффекты, паттерны энергии

### E. Text / OCR Novelty
Текстовая новизна: OCR текст, layout, стиль (шрифты, цвета, эффекты)

### F. Behavioral & Motion Novelty
Поведенческая новизна: движения поз, взаимодействия объектов, последовательности действий

### G. Multimodal Novelty
Мультимодальная новизна: комбинация всех модальностей, новые события

### H. Temporal / Trend Novelty
Временная/трендовая новизна: соответствие трендам, историческая схожесть, новаторство

## 📚 API Документация

### UniquenessModule

#### `__init__(top_n: int = 100, novelty_weights: Optional[Dict[str, float]] = None)`

Инициализирует процессор метрик уникальности.

**Параметры:**
- `top_n` - количество топ видео для сравнения (по умолчанию 100)
- `novelty_weights` - веса для различных категорий при вычислении overall_novelty_index:
  ```python
  {
      'semantic': 0.20,
      'visual': 0.15,
      'editing': 0.15,
      'audio': 0.10,
      'text': 0.10,
      'behavioral': 0.10,
      'multimodal': 0.15,
      'temporal': 0.05
  }
  ```

#### `extract_all(...) -> Dict[str, Any]`

Главный метод для вычисления всех метрик уникальности.

**Параметры:**

**Обязательные:**
- `video_embedding` (np.ndarray) - embedding текущего видео (1D array)
- `reference_embeddings` (List[np.ndarray]) - список embeddings референсных видео

**Опциональные (если не переданы, соответствующие метрики будут 0.0 или 1.0):**
- `video_topics` - темы текущего видео (List[str], np.ndarray или Dict[str, float])
- `reference_topics_list` - список тем референсных видео
- `video_visual_features` - визуальные фичи текущего видео (см. [Форматы данных](#форматы-данных))
- `reference_visual_features_list` - список визуальных фичей референсных видео
- `video_pacing_features` - фичи pacing текущего видео
- `reference_pacing_features_list` - список фичей pacing референсных видео
- `video_audio_features` - аудио фичи текущего видео
- `reference_audio_features_list` - список аудио фичей референсных видео
- `video_text_features` - текстовые фичи текущего видео
- `reference_text_features_list` - список текстовых фичей референсных видео
- `video_behavior_features` - фичи поведения текущего видео
- `reference_behavior_features_list` - список фичей поведения референсных видео
- `video_events` - события текущего видео
- `reference_events_list` - список событий референсных видео
- `video_metadata` - метаданные текущего видео
- `reference_videos_metadata` - метаданные референсных видео
- `similarity_scores` - предвычисленные similarity scores (опционально)

**Возвращает:**
```python
{
    'features': {
        # Все метрики уникальности
        'semantic_novelty_score': 0.75,
        'overall_novelty_index': 0.68,
        # ... и т.д.
    },
    'all_metrics': {...}  # То же самое для обратной совместимости
}
```

#### Отдельные методы для каждой категории

Можно вызывать методы для отдельных категорий метрик:

- `compute_semantic_novelty(video_embedding, reference_embeddings, video_topics, reference_topics_list)`
- `compute_visual_novelty(video_visual_features, reference_visual_features_list)`
- `compute_editing_novelty(video_pacing_features, reference_pacing_features_list)`
- `compute_audio_novelty(video_audio_features, reference_audio_features_list)`
- `compute_text_novelty(video_text_features, reference_text_features_list)`
- `compute_behavioral_novelty(video_behavior_features, reference_behavior_features_list)`
- `compute_multimodal_novelty(all_novelty_metrics, video_events, reference_events_list)`
- `compute_temporal_novelty(video_metadata, reference_videos_metadata, similarity_scores)`

## 📊 Категории метрик

### A. Semantic / Content Novelty

**Метрики:**
- `semantic_novelty_score` - новизна на основе embeddings (1 - max similarity)
- `topic_novelty_score` - доля новых концептов, которых нет в популярных видео
- `concept_diversity_score` - разнообразие концептов (количество уникальных / общее количество)

**Использует:** Cosine similarity между видео embeddings (CLIP, VideoCLIP, BLIP), Jaccard similarity для тем

### B. Visual / Style Novelty

**Метрики:**
- `color_palette_novelty` - различие цветовой палитры с топ-видео
- `lighting_style_novelty` - уникальность схем освещения
- `shot_type_novelty` - новизна типов кадров (редкие кадры или нестандартные планы)
- `camera_motion_novelty` - уникальность движений камеры (dolly, gimbal, drone)

**Использует:** Cosine similarity, Wasserstein distance, корреляция

### C. Editing & Pacing Novelty

**Метрики:**
- `cut_rate_novelty` - новизна частоты монтажных склеек
- `shot_duration_distribution_novelty` - новизна распределения длительностей шотов
- `scene_length_novelty` - новизна длительностей сцен
- `pacing_pattern_novelty` - новизна паттернов ритма

**Использует:** Wasserstein distance для распределений, корреляция pacing curves

### D. Audio Novelty

**Метрики:**
- `music_track_novelty` - новизна музыкального трека (новый трек, BPM, стиль)
- `voice_style_novelty` - новизна стиля голоса
- `sound_effects_novelty` - новизна звуковых эффектов
- `audio_energy_pattern_novelty` - новизна паттернов аудио энергии

**Использует:** Cosine similarity между аудио embeddings, корреляция energy patterns

### E. Text / OCR Novelty

**Метрики:**
- `ocr_text_novelty` - новизна OCR текста (новые ключевые слова/фразы)
- `text_layout_novelty` - новизна расположения текста (необычное расположение)
- `text_style_novelty` - новизна стиля текста (уникальные шрифты, цвета, эффекты)

**Использует:** Cosine similarity между текстовыми embeddings

### F. Behavioral & Motion Novelty

**Метрики:**
- `pose_motion_novelty` - новизна движений поз (необычные движения/редкие паттерны)
- `object_interaction_novelty` - новизна взаимодействий объектов
- `action_sequence_novelty` - новизна последовательностей действий

**Использует:** Cosine similarity для pose embeddings, корреляция для action sequences

### G. Multimodal Novelty

**Метрики:**
- `multimodal_novelty_score` - средневзвешенная новизна по всем модальностям
- `novel_event_alignment_score` - доля новых событий, которых нет в топ видео
- `overall_novelty_index` - общий индекс новизны (средневзвешенный по всем модальностям)

**Использует:** Weighted combination всех категорий метрик, анализ событий

### H. Temporal / Trend Novelty

**Метрики:**
- `trend_alignment_score` - насколько видео соответствует текущему тренду (низкий → уникально)
- `historical_similarity_score` - схожесть с прошлым контентом
- `early_adopter_score` - насколько видео новаторское в своей нише

**Использует:** Анализ метаданных, similarity scores, исторические данные

## 📦 Форматы данных

### video_visual_features

```python
{
    'color_histogram': np.ndarray,  # Цветовая гистограмма
    'lighting_features': np.ndarray,  # Фичи освещения
    'shot_type_distribution': np.ndarray,  # Распределение типов кадров
    'camera_motion_features': np.ndarray  # Фичи движения камеры (temporal curve)
}
```

### video_pacing_features

```python
{
    'cut_rate': float,  # Частота монтажных склеек
    'shot_duration_distribution': np.ndarray,  # Распределение длительностей шотов
    'scene_lengths': np.ndarray,  # Массив длительностей сцен
    'pacing_curve': np.ndarray  # Temporal curve pacing
}
```

### video_audio_features

```python
{
    'audio_embedding': np.ndarray,  # Аудио embedding
    'voice_embedding': np.ndarray,  # Embedding голоса (опционально)
    'tempo': float,  # BPM / темп
    'energy_pattern': np.ndarray,  # Temporal curve энергии
    'sound_effects_features': np.ndarray  # Фичи звуковых эффектов (опционально)
}
```

### video_text_features

```python
{
    'ocr_embedding': np.ndarray,  # Embedding OCR текста
    'text_layout': np.ndarray,  # Фичи layout (позиции, размеры)
    'text_style_features': np.ndarray  # Фичи стиля текста (шрифты, цвета, эффекты)
}
```

### video_behavior_features

```python
{
    'pose_motion': np.ndarray,  # Embedding движений поз
    'object_interaction': np.ndarray,  # Embedding взаимодействий объектов
    'action_sequence': np.ndarray  # Temporal curve последовательностей действий
}
```

### video_events

```python
[
    {
        'type': str,  # Тип события ('face', 'audio', 'scene_jump', 'text', 'pose')
        'timestamp': float,  # Время события в секундах
        'strength': float  # Сила события
    },
    ...
]
```

### video_metadata

```python
{
    'created_date': str,  # Дата создания видео (опционально)
    'category': str,  # Категория/ниша (опционально)
    # ... другие метаданные
}
```

## 💡 Примеры использования

### Пример 1: Базовая семантическая новизна

```python
from uniqueness import UniquenessModule
import numpy as np

uniqueness = UniquenessModule(top_n=100)

# Embeddings
video_emb = np.random.randn(512)
ref_embs = [np.random.randn(512) for _ in range(10)]

# Темы
video_topics = ["cooking", "tutorial", "food", "unique_recipe"]
ref_topics = [
    ["cooking", "recipe"],
    ["gaming", "tutorial"],
    ["cooking", "food", "tutorial"]
]

result = uniqueness.extract_all(
    video_embedding=video_emb,
    reference_embeddings=ref_embs,
    video_topics=video_topics,
    reference_topics_list=ref_topics
)

print(f"Semantic novelty: {result['features']['semantic_novelty_score']:.3f}")
print(f"Topic novelty: {result['features']['topic_novelty_score']:.3f}")
print(f"Concept diversity: {result['features']['concept_diversity_score']:.3f}")
```

### Пример 2: Полный анализ с визуальными и монтажными фичами

```python
# Визуальные фичи текущего видео
video_visual = {
    'color_histogram': np.random.rand(256),
    'lighting_features': np.random.rand(10),
    'shot_type_distribution': np.array([0.3, 0.4, 0.2, 0.1]),
    'camera_motion_features': np.random.rand(100)
}

# Референсные визуальные фичи
ref_visuals = [
    {
        'color_histogram': np.random.rand(256),
        'lighting_features': np.random.rand(10),
        'shot_type_distribution': np.array([0.2, 0.5, 0.2, 0.1]),
        'camera_motion_features': np.random.rand(100)
    }
    for _ in range(5)
]

# Pacing фичи
video_pacing = {
    'cut_rate': 2.5,
    'shot_duration_distribution': np.array([0.1, 0.2, 0.3, 0.2, 0.1, 0.1]),
    'scene_lengths': np.array([5.0, 8.0, 6.0, 10.0]),
    'pacing_curve': np.random.rand(50)
}

ref_pacing = [
    {
        'cut_rate': 2.3,
        'shot_duration_distribution': np.array([0.1, 0.3, 0.3, 0.2, 0.1, 0.0]),
        'scene_lengths': np.array([4.0, 7.0, 5.0, 9.0]),
        'pacing_curve': np.random.rand(50)
    }
    for _ in range(5)
]

result = uniqueness.extract_all(
    video_embedding=video_emb,
    reference_embeddings=ref_embs,
    video_visual_features=video_visual,
    reference_visual_features_list=ref_visuals,
    video_pacing_features=video_pacing,
    reference_pacing_features_list=ref_pacing
)

print(f"Color palette novelty: {result['features']['color_palette_novelty']:.3f}")
print(f"Cut rate novelty: {result['features']['cut_rate_novelty']:.3f}")
print(f"Overall novelty: {result['features']['overall_novelty_index']:.3f}")
```

### Пример 3: Анализ с аудио и текстовыми фичами

```python
# Аудио фичи
video_audio = {
    'audio_embedding': np.random.randn(128),
    'tempo': 120.0,
    'energy_pattern': np.random.rand(100)
}

ref_audio = [
    {
        'audio_embedding': np.random.randn(128),
        'tempo': 115.0,
        'energy_pattern': np.random.rand(100)
    }
    for _ in range(5)
]

# Текстовые фичи
video_text = {
    'ocr_embedding': np.random.randn(256),
    'text_layout': np.array([0.2, 0.3, 0.5]),  # top, center, bottom
    'text_style_features': np.random.randn(64)
}

ref_text = [
    {
        'ocr_embedding': np.random.randn(256),
        'text_layout': np.array([0.3, 0.4, 0.3]),
        'text_style_features': np.random.randn(64)
    }
    for _ in range(5)
]

result = uniqueness.extract_all(
    video_embedding=video_emb,
    reference_embeddings=ref_embs,
    video_audio_features=video_audio,
    reference_audio_features_list=ref_audio,
    video_text_features=video_text,
    reference_text_features_list=ref_text
)

print(f"Music track novelty: {result['features']['music_track_novelty']:.3f}")
print(f"OCR text novelty: {result['features']['ocr_text_novelty']:.3f}")
```

### Пример 4: Анализ с событиями и метаданными

```python
# События текущего видео
video_events = [
    {'type': 'face', 'timestamp': 5.0, 'strength': 0.8},
    {'type': 'audio', 'timestamp': 12.0, 'strength': 0.9},
    {'type': 'scene_jump', 'timestamp': 20.0, 'strength': 0.7}
]

# События референсных видео
ref_events = [
    [
        {'type': 'face', 'timestamp': 3.0, 'strength': 0.7},
        {'type': 'audio', 'timestamp': 10.0, 'strength': 0.8}
    ],
    [
        {'type': 'text', 'timestamp': 8.0, 'strength': 0.6}
    ]
]

# Метаданные
video_metadata = {
    'created_date': '2024-01-15',
    'category': 'cooking'
}

ref_metadata = [
    {'created_date': '2024-01-10', 'category': 'cooking', 'is_viral': True},
    {'created_date': '2024-01-12', 'category': 'cooking', 'is_viral': False}
]

result = uniqueness.extract_all(
    video_embedding=video_emb,
    reference_embeddings=ref_embs,
    video_events=video_events,
    reference_events_list=ref_events,
    video_metadata=video_metadata,
    reference_videos_metadata=ref_metadata
)

print(f"Novel event alignment: {result['features']['novel_event_alignment_score']:.3f}")
print(f"Early adopter score: {result['features']['early_adopter_score']:.3f}")
print(f"Trend alignment: {result['features']['trend_alignment_score']:.3f}")
```

### Пример 5: Использование отдельных категорий

```python
# Только семантическая новизна
semantic_novelty = uniqueness.compute_semantic_novelty(
    video_embedding=video_emb,
    reference_embeddings=ref_embs,
    video_topics=video_topics,
    reference_topics_list=ref_topics
)

# Только визуальная новизна
visual_novelty = uniqueness.compute_visual_novelty(
    video_visual_features=video_visual,
    reference_visual_features_list=ref_visuals
)

# Только мультимодальная новизна (требует все предыдущие метрики)
all_metrics = {
    'semantic_novelty_score': semantic_novelty['semantic_novelty_score'],
    'color_palette_novelty': visual_novelty['color_palette_novelty'],
    # ... другие метрики
}

multimodal = uniqueness.compute_multimodal_novelty(
    all_novelty_metrics=all_metrics,
    video_events=video_events,
    reference_events_list=ref_events
)
```

## 🔌 Взаимодействие с другими модулями

| Модуль | Что подаёт |
|--------|-----------|
| `high_level_semantic` | `video_embedding`, `video_topics`, `topic_embeddings` |
| `similarity_metrics` | `similarity_scores` (для temporal novelty) |
| `color_light` | `video_visual_features` (color_histogram, lighting_features) |
| `video_pacing` | `video_pacing_features` (pacing_curve, shot_duration_distribution) |
| `cut_detection` | `cut_rate`, `shot_type_distribution` |
| `optical_flow` | `camera_motion_features` |
| `emotion_face` | `video_events` (face events) |
| `behavior` | `video_behavior_features` (pose_motion, object_interaction) |
| `text_scoring` / OCR | `video_text_features` (ocr_embedding, text_layout, text_style_features) |
| Audio processors | `video_audio_features` (audio_embedding, tempo, energy_pattern) |

## 🎯 Применение

### Для ML-моделей предсказания популярности

Метрики уникальности используются как фичи для:
- Предсказания вирусного потенциала
- Оценки новаторства контента
- Анализа ранних трендов
- Определения уникальности в нише

### Для аналитики контента

- Сравнение с конкурентами
- Выявление уникальных паттернов
- Оценка новаторства
- Анализ соответствия трендам

### Для рекомендательных систем

- Поиск уникального контента
- Выявление ранних трендов
- Группировка по новизне

## ⚙️ Настройка весов

Можно настроить веса для различных категорий при вычислении `overall_novelty_index`:

```python
custom_weights = {
    'semantic': 0.25,  # Увеличиваем важность семантики
    'visual': 0.20,
    'editing': 0.15,
    'audio': 0.10,
    'text': 0.10,
    'behavioral': 0.10,
    'multimodal': 0.08,
    'temporal': 0.02
}

uniqueness = UniquenessModule(
    top_n=100,
    novelty_weights=custom_weights
)
```

## 📝 Примечания

- Если какой-то тип фичей не передан, соответствующие метрики будут равны 0.0 (кроме semantic, где будет 1.0 при отсутствии референсов)
- Все метрики нормализованы в диапазоне [0, 1] где 1.0 = максимальная новизна/уникальность
- `semantic_novelty_score` = 1.0 означает полную уникальность по семантике
- `overall_novelty_index` = 1.0 означает полную уникальность по всем аспектам
- Для `top_n`: если референсных видео меньше, используются все доступные
- Метрики новизны обратны метрикам схожести: `novelty = 1 - similarity`

## 🔄 Связь с Similarity Metrics

Модуль `uniqueness` тесно связан с модулем `similarity_metrics`:
- Novelty scores часто вычисляются как `1 - similarity_score`
- Можно использовать предвычисленные similarity scores для temporal novelty
- Оба модуля используют одинаковые форматы входных данных

**Рекомендация:** Используйте `similarity_metrics` для вычисления схожести, а `uniqueness` для анализа уникальности и новизны.

