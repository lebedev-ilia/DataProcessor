# 📊 Similarity Metrics Module

## 📌 Описание

Модуль для вычисления метрик схожести между видео. Сравнивает текущее видео с набором референсных видео по различным аспектам: семантике, темам, визуальному стилю, тексту, аудио, эмоциям, временному ритму и другим характеристикам.

**Применение:**
- Определение уникальности видео
- Анализ схожести с популярными/вирусными видео
- Оценка соответствия трендам
- Кластеризация и группировка видео
- Генерация фичей для ML-моделей предсказания популярности

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
from similarity_metrics import SimilarityMetrics
import numpy as np

# Инициализация
similarity = SimilarityMetrics(top_n=10)

# Embeddings текущего и референсных видео
video_embedding = np.random.randn(512)  # Embedding текущего видео
reference_embeddings = [np.random.randn(512) for _ in range(5)]  # Референсные видео

# Темы
video_topics = ["cooking", "tutorial", "food"]
reference_topics_list = [
    ["cooking", "recipe"],
    ["gaming", "tutorial"],
    ["cooking", "food", "tutorial"]
]

# Вычисление метрик
result = similarity.extract_all(
    video_embedding=video_embedding,
    reference_embeddings=reference_embeddings,
    video_topics=video_topics,
    reference_topics_list=reference_topics_list
)

# Результат содержит все метрики схожести
print(result['features']['semantic_similarity_mean'])
print(result['features']['topic_overlap_score'])
print(result['features']['overall_similarity_score'])
```

## 🏗️ Архитектура

Модуль реализует 9 категорий метрик схожести:

### A. Semantic Similarity
Семантическая схожесть на основе embeddings (CLIP, VideoCLIP, BLIP)

### B. Topic / Concept Overlap
Тематическое пересечение через Jaccard similarity и topic embeddings

### C. Style & Composition Similarity
Визуальная схожесть: цвет, свет, типы кадров, монтаж, движение

### D. Text & OCR Similarity
Схожесть текста в кадре: semantic embeddings, layout, timing

### E. Audio / Speech Similarity
Аудио схожесть: embeddings, речь, темп, энергия

### F. Emotion & Behavior Similarity
Схожесть эмоций и поведения: кривые эмоций, позы, паттерны активности

### G. Temporal / Pacing Similarity
Временной ритм: pacing curves, shot duration, scene length

### H. High-level Comparative Scores
Высокоуровневые оценки: overall similarity, uniqueness, trend alignment, viral pattern

### I. Group / Batch Metrics
Групповые метрики для батчей видео: кластеризация, variance, outliers

## 📚 API Документация

### SimilarityMetrics

#### `__init__(top_n: int = 10, similarity_weights: Optional[Dict[str, float]] = None)`

Инициализирует процессор метрик схожести.

**Параметры:**
- `top_n` - количество топ видео для усреднения метрик (по умолчанию 10)
- `similarity_weights` - веса для различных категорий при вычислении overall_similarity_score:
  ```python
  {
      'semantic': 0.25,
      'topics': 0.15,
      'visual': 0.15,
      'text': 0.10,
      'audio': 0.15,
      'emotion': 0.10,
      'temporal': 0.10
  }
  ```

#### `extract_all(...) -> Dict[str, Any]`

Главный метод для вычисления всех метрик схожести.

**Параметры:**

**Обязательные:**
- `video_embedding` (np.ndarray) - embedding текущего видео (1D array)
- `reference_embeddings` (List[np.ndarray]) - список embeddings референсных видео

**Опциональные (если не переданы, соответствующие метрики будут 0.0):**
- `video_topics` - темы текущего видео (List[str], np.ndarray или Dict[str, float])
- `reference_topics_list` - список тем референсных видео
- `video_visual_features` - визуальные фичи текущего видео (см. [Форматы данных](#форматы-данных))
- `reference_visual_features_list` - список визуальных фичей референсных видео
- `video_text_features` - текстовые фичи текущего видео
- `reference_text_features_list` - список текстовых фичей референсных видео
- `video_audio_features` - аудио фичи текущего видео
- `reference_audio_features_list` - список аудио фичей референсных видео
- `video_emotion_features` - фичи эмоций/поведения текущего видео
- `reference_emotion_features_list` - список фичей эмоций/поведения референсных видео
- `video_pacing_features` - фичи pacing текущего видео
- `reference_pacing_features_list` - список фичей pacing референсных видео
- `reference_videos_metadata` - метаданные референсных видео (для trend_alignment и viral_pattern)

**Возвращает:**
```python
{
    'features': {
        # Все метрики схожести
        'semantic_similarity_mean': 0.75,
        'semantic_similarity_max': 0.89,
        # ... и т.д.
    },
    'all_metrics': {...}  # То же самое для обратной совместимости
}
```

#### Отдельные методы для каждой категории

Можно вызывать методы для отдельных категорий метрик:

- `compute_semantic_similarity(video_embedding, reference_embeddings)`
- `compute_topic_overlap(video_topics, reference_topics_list)`
- `compute_style_similarity(video_visual_features, reference_visual_features_list)`
- `compute_text_similarity(video_text_features, reference_text_features_list)`
- `compute_audio_similarity(video_audio_features, reference_audio_features_list)`
- `compute_emotion_behavior_similarity(video_emotion_features, reference_emotion_features_list)`
- `compute_temporal_similarity(video_pacing_features, reference_pacing_features_list)`
- `compute_high_level_scores(all_similarity_metrics, reference_videos_metadata)`
- `compute_batch_metrics(video_embeddings, video_features_list)`

## 📊 Категории метрик

### A. Semantic Similarity

**Метрики:**
- `semantic_similarity_mean` - среднее по топ-N видео
- `semantic_similarity_max` - максимальная похожесть
- `semantic_similarity_min` - минимальная похожесть
- `semantic_novelty_score` - уникальность (1 - max_similarity)

**Использует:** Cosine similarity между видео embeddings (CLIP, VideoCLIP, BLIP)

### B. Topic / Concept Overlap

**Метрики:**
- `topic_overlap_score` - доля совпадающих тем (Jaccard similarity)
- `topic_diversity_comparison` - разница в тематическом разнообразии
- `key_concept_match_ratio` - доля совпадающих ключевых концептов

**Использует:** Jaccard similarity по извлечённым ключевым словам и topic embeddings

### C. Style & Composition Similarity

**Метрики:**
- `color_histogram_similarity` - схожесть цветовых гистограмм
- `lighting_pattern_similarity` - схожесть паттернов освещения
- `shot_type_distribution_similarity` - схожесть распределения типов кадров
- `cut_rate_similarity` - схожесть частоты монтажных склеек
- `motion_pattern_similarity` - схожесть паттернов движения

**Использует:** Cosine similarity, Earth Mover's Distance, корреляция

### D. Text & OCR Similarity

**Метрики:**
- `ocr_text_semantic_similarity` - семантическая схожесть OCR текста
- `text_layout_similarity` - схожесть расположения текста (позиции, длина, font size)
- `text_timing_similarity` - схожесть временных паттернов появления текста

**Использует:** Cosine similarity между embeddings, корреляция timing curves

### E. Audio / Speech Similarity

**Метрики:**
- `audio_embedding_similarity` - схожесть аудио embeddings (yamnet, VGGish, OpenL3)
- `speech_content_similarity` - схожесть речевого контента (по ASR)
- `music_tempo_similarity` - схожесть музыкального темпа (BPM)
- `audio_energy_pattern_similarity` - схожесть паттернов аудио энергии

**Использует:** Cosine similarity, корреляция energy patterns

### F. Emotion & Behavior Similarity

**Метрики:**
- `emotion_curve_similarity` - схожесть кривых эмоций (Pearson correlation)
- `pose_motion_similarity` - схожесть движений поз (действия людей)
- `behavior_pattern_similarity` - схожесть паттернов активности объектов

**Использует:** Pearson correlation между кривыми, cosine similarity для pose embeddings

### G. Temporal / Pacing Similarity

**Метрики:**
- `pacing_curve_similarity` - схожесть кривых pacing (корреляция)
- `shot_duration_distribution_similarity` - схожесть распределения длительностей шотов
- `scene_length_similarity` - схожесть длительностей сцен
- `temporal_pattern_novelty` - новизна временных паттернов (1 - mean_similarity)

**Использует:** Pearson correlation, Wasserstein distance для распределений

### H. High-level Comparative Scores

**Метрики:**
- `overall_similarity_score` - взвешенная сумма всех метрик схожести
- `uniqueness_score` - уникальность (1 - overall_similarity)
- `trend_alignment_score` - насколько видео похоже на топ видео в своей нише
- `viral_pattern_score` - схожесть с успешными/вирусными видео

**Использует:** Weighted combination всех категорий метрик

### I. Group / Batch Metrics

**Метрики:**
- `cluster_similarity_mean` - средняя схожесть между всеми парами видео в батче
- `inter_video_variance_topics` - вариативность тем между видео
- `inter_video_variance_emotions` - вариативность эмоций между видео
- `inter_video_variance_editing` - вариативность монтажа (cut rate) между видео
- `inter_video_variance_audio` - вариативность аудио (tempo) между видео

**Применение:** Анализ разнообразия в наборе видео, кластеризация, outlier detection

## 📦 Форматы данных

### video_visual_features

```python
{
    'color_histogram': np.ndarray,  # Цветовая гистограмма
    'lighting_features': np.ndarray,  # Фичи освещения
    'shot_type_distribution': np.ndarray,  # Распределение типов кадров
    'cut_rate': float,  # Частота монтажных склеек
    'motion_pattern': np.ndarray  # Паттерн движения (temporal curve)
}
```

### video_text_features

```python
{
    'ocr_embedding': np.ndarray,  # Embedding OCR текста
    'text_layout': np.ndarray,  # Фичи layout (позиции, размеры)
    'text_timing': np.ndarray  # Temporal curve появления текста
}
```

### video_audio_features

```python
{
    'audio_embedding': np.ndarray,  # Аудио embedding
    'speech_embedding': np.ndarray,  # Embedding речи
    'tempo': float,  # BPM / темп
    'energy_pattern': np.ndarray  # Temporal curve энергии
}
```

### video_emotion_features

```python
{
    'emotion_curve': np.ndarray,  # Temporal curve эмоций
    'pose_motion': np.ndarray,  # Embedding движений поз
    'behavior_pattern': np.ndarray  # Temporal curve активности
}
```

### video_pacing_features

```python
{
    'pacing_curve': np.ndarray,  # Temporal curve pacing
    'shot_duration_distribution': np.ndarray,  # Распределение длительностей
    'scene_lengths': np.ndarray  # Массив длительностей сцен
}
```

### reference_videos_metadata

```python
[
    {
        'is_viral': bool,  # Является ли видео вирусным
        'popularity_score': float,  # Оценка популярности
        'category': str,  # Категория/ниша
        # ... другие метаданные
    },
    ...
]
```

## 💡 Примеры использования

### Пример 1: Базовая семантическая схожесть

```python
from similarity_metrics import SimilarityMetrics
import numpy as np

similarity = SimilarityMetrics(top_n=5)

# Embeddings
video_emb = np.random.randn(512)
ref_embs = [np.random.randn(512) for _ in range(10)]

result = similarity.extract_all(
    video_embedding=video_emb,
    reference_embeddings=ref_embs
)

print(f"Semantic similarity mean: {result['features']['semantic_similarity_mean']:.3f}")
print(f"Novelty score: {result['features']['semantic_novelty_score']:.3f}")
```

### Пример 2: Полный анализ с визуальными фичами

```python
# Визуальные фичи текущего видео
video_visual = {
    'color_histogram': np.random.rand(256),
    'lighting_features': np.random.rand(10),
    'shot_type_distribution': np.array([0.3, 0.4, 0.2, 0.1]),
    'cut_rate': 2.5,
    'motion_pattern': np.random.rand(100)
}

# Референсные визуальные фичи
ref_visuals = [
    {
        'color_histogram': np.random.rand(256),
        'lighting_features': np.random.rand(10),
        'shot_type_distribution': np.array([0.2, 0.5, 0.2, 0.1]),
        'cut_rate': 2.3,
        'motion_pattern': np.random.rand(100)
    }
    for _ in range(5)
]

result = similarity.extract_all(
    video_embedding=video_emb,
    reference_embeddings=ref_embs,
    video_visual_features=video_visual,
    reference_visual_features_list=ref_visuals
)

print(f"Color similarity: {result['features']['color_histogram_similarity']:.3f}")
print(f"Cut rate similarity: {result['features']['cut_rate_similarity']:.3f}")
```

### Пример 3: Анализ уникальности и трендов

```python
# Референсные видео с метаданными
ref_metadata = [
    {'is_viral': True, 'popularity_score': 0.95, 'category': 'cooking'},
    {'is_viral': False, 'popularity_score': 0.60, 'category': 'cooking'},
    {'is_viral': True, 'popularity_score': 0.88, 'category': 'cooking'},
]

result = similarity.extract_all(
    video_embedding=video_emb,
    reference_embeddings=ref_embs,
    reference_videos_metadata=ref_metadata
)

print(f"Overall similarity: {result['features']['overall_similarity_score']:.3f}")
print(f"Uniqueness: {result['features']['uniqueness_score']:.3f}")
print(f"Trend alignment: {result['features']['trend_alignment_score']:.3f}")
print(f"Viral pattern: {result['features']['viral_pattern_score']:.3f}")
```

### Пример 4: Групповые метрики для батча видео

```python
# Embeddings всех видео в батче
batch_embeddings = [np.random.randn(512) for _ in range(20)]

# Фичи всех видео
batch_features = [
    {
        'topic_embedding': np.random.randn(128),
        'emotion_mean': 0.6,
        'cut_rate': 2.5,
        'tempo': 120.0
    }
    for _ in range(20)
]

batch_metrics = similarity.compute_batch_metrics(
    video_embeddings=batch_embeddings,
    video_features_list=batch_features
)

print(f"Cluster similarity: {batch_metrics['cluster_similarity_mean']:.3f}")
print(f"Topics variance: {batch_metrics['inter_video_variance_topics']:.3f}")
```

### Пример 5: Использование отдельных категорий

```python
# Только семантическая схожесть
semantic_metrics = similarity.compute_semantic_similarity(
    video_embedding=video_emb,
    reference_embeddings=ref_embs
)

# Только тематическое пересечение
video_topics = ["cooking", "tutorial"]
ref_topics = [["cooking", "recipe"], ["gaming", "tutorial"]]
topic_metrics = similarity.compute_topic_overlap(
    video_topics=video_topics,
    reference_topics_list=ref_topics
)
```

## 🔌 Взаимодействие с другими модулями

| Модуль | Что подаёт |
|--------|-----------|
| `high_level_semantic` | `video_embedding`, `video_topics`, `topic_embeddings` |
| `color_light` | `video_visual_features` (color_histogram, lighting_features) |
| `video_pacing` | `video_pacing_features` (pacing_curve, shot_duration_distribution) |
| `cut_detection` | `cut_rate`, `shot_type_distribution` |
| `optical_flow` | `motion_pattern` |
| `emotion_face` | `emotion_curve`, `emotion_features` |
| `behavior` | `pose_motion`, `behavior_pattern` |
| `text_scoring` / OCR | `video_text_features` (ocr_embedding, text_layout, text_timing) |
| Audio processors | `video_audio_features` (audio_embedding, tempo, energy_pattern) |

## 🎯 Применение

### Для ML-моделей предсказания популярности

Метрики схожести используются как фичи для:
- Предсказания количества просмотров
- Оценки вирусного потенциала
- Анализа соответствия трендам
- Определения уникальности контента

### Для рекомендательных систем

- Поиск похожих видео
- Кластеризация контента
- Группировка по стилю/тематике

### Для аналитики контента

- Сравнение с конкурентами
- Анализ соответствия трендам
- Оценка уникальности
- Выявление паттернов успешных видео

## ⚙️ Настройка весов

Можно настроить веса для различных категорий при вычислении `overall_similarity_score`:

```python
custom_weights = {
    'semantic': 0.30,  # Увеличиваем важность семантики
    'topics': 0.20,
    'visual': 0.15,
    'text': 0.10,
    'audio': 0.10,
    'emotion': 0.10,
    'temporal': 0.05
}

similarity = SimilarityMetrics(
    top_n=10,
    similarity_weights=custom_weights
)
```

## 📝 Примечания

- Если какой-то тип фичей не передан, соответствующие метрики будут равны 0.0
- Все метрики нормализованы в диапазоне [0, 1] где 1.0 = максимальная схожесть
- `semantic_novelty_score` и `uniqueness_score` = 1.0 означает полную уникальность
- Для `top_n`: если референсных видео меньше, используются все доступные

