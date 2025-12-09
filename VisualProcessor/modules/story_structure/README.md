# 📖 Story Structure Feature Extraction Pipeline

## 📌 Описание

Пайплайн предназначен для извлечения структурных и сюжетных признаков видео на уровне сегментов, шотов и персонажей. Он комбинирует визуальные, семантические и эмбеддинговые сигналы для оценки:

- **Story Segmentation** — сегментация истории
- **Hook качества** — качество зацепки в начале видео
- **Climax / Peak Detection** — детекция кульминации и пиков
- **Character-level dynamics** — динамика персонажей
- **Topic / Semantic structure** — семантическая структура
- **Narrative continuity** — непрерывность повествования
- **Energy curve** — кривая энергии истории

### ✨ Особенности

- ✅ Сглаживание сигналов для уменьшения шумов
- ✅ Комбинированные признаки motion + semantic jumps
- ✅ Нормализация относительно длительности видео
- ✅ Статистика mean + std для динамических сигналов

## ⚡ Установка

```bash
pip install opencv-python-headless tqdm torch torchvision torchaudio mediapipe sentence-transformers git+https://github.com/openai/CLIP.git
```

> **Примечание:** Требуется GPU для ускоренного извлечения CLIP-эмбеддингов.

## 🛠 Используемые модели

| Модуль | Модель / Метод |
|--------|----------------|
| Visual embedding | OpenAI CLIP (ViT-B/32) |
| Text embedding | SentenceTransformer (all-MiniLM-L6-v2) |
| Face detection / tracking | Mediapipe FaceMesh |
| Clustering / Segmentation | Agglomerative Clustering (cosine affinity) |
| Optical flow | OpenCV Farneback dense flow |

## 🚀 Основные функции

### 1. Story Segmentation

**Фичи:**
- `number_of_story_segments` — количество сегментов истории
- `avg_story_segment_duration` — средняя длительность сегмента
- `abrupt_story_transition_count` — количество резких переходов
- `narrative_continuity_score` — оценка непрерывности повествования
- `narrative_continuity_std` — стандартное отклонение непрерывности

### 2. Hook Features (первые 5–10 сек)

**Фичи:**
- `hook_motion_intensity` — интенсивность движения в начале
- `hook_cut_rate` — частота катов в начале
- `hook_motion_spikes` — всплески движения
- `hook_face_presence` — присутствие лиц
- `hook_visual_surprise_score` — визуальная неожиданность
- `hook_visual_surprise_std` — стандартное отклонение
- `hook_brightness_spike` — всплеск яркости
- `hook_saturation_spike` — всплеск насыщенности

### 3. Climax / Peak Detection

**Фичи:**
- `climax_timestamp` — временная метка кульминации
- `climax_strength` — сила кульминации
- `number_of_peaks` — количество пиков
- `climax_duration` — длительность кульминации
- `story_energy_curve` — полный сигнал комбинированной энергии

### 4. Character-level Features

**Фичи:**
- `number_of_speakers` — количество говорящих
- `main_character_screen_time` — экранное время главного персонажа
- `speaker_switch_rate` — частота смены говорящих
- `face_presence_curve` — кривая присутствия лиц

### 5. Topic / Semantic Features

**Фичи:**
- `number_of_topics` — количество тем
- `avg_topic_duration` — средняя длительность темы
- `topic_shift_times` — времена смены тем
- `topic_diversity` — разнообразие тем
- `semantic_coherence_score` — оценка семантической связности

## 🖼 Пример использования

```python
from video_story_structure_optimized import StoryStructurePipelineOptimized

video_path = "example.mp4"
subtitles = [
    "Hello everyone",
    "Welcome to my vlog",
    "Today we will..."
]

# Инициализация
pipeline = StoryStructurePipelineOptimized(video_path)

# Извлечение фичей
features = pipeline.extract_all_features(subtitles=subtitles)

# Вывод результатов
for k, v in features.items():
    print(f"{k}: {v}")
```

## 🔧 Настройки

| Параметр | Описание | По умолчанию |
|----------|----------|--------------|
| `fps` | Частота кадров для анализа | 1 FPS |
| `clip_model` | Модель CLIP | ViT-B/32 |
| `sentence_model` | SentenceTransformer для текста | all-MiniLM-L6-v2 |
| `hook_seconds` | Длительность первых секунд для hook анализа | 5 |

## 💡 Особенности / Улучшения

- **Smooth signals** — уменьшение ложных пиков motion и embedding jumps
- **Hook + Climax** — комбинированные метрики движения и семантических изменений
- **Character-level** — динамика присутствия персонажей и переключения
- **Topic segmentation** — скользящее окно + cosine similarity для плавной кластеризации
- **Story energy curve** — сохранение полного сигнала для анализа или ML
- **Нормализация по длительности** — сравнение между роликами разной длины

## 📊 Выходные фичи

| Категория | Примеры фичей |
|-----------|---------------|
| Story Segmentation | `number_of_story_segments`, `avg_story_segment_duration`, `abrupt_story_transition_count` |
| Hook | `hook_motion_intensity`, `hook_visual_surprise_score`, `hook_face_presence` |
| Climax | `climax_timestamp`, `climax_strength`, `story_energy_curve` |
| Character-level | `number_of_speakers`, `main_character_screen_time`, `speaker_switch_rate` |
| Topic | `number_of_topics`, `avg_topic_duration`, `semantic_coherence_score` |

## 🎯 Применение

- Анализ структуры повествования
- Оценка качества зацепки (hook)
- Детекция кульминационных моментов
- Анализ динамики персонажей
- Семантический анализ контента
