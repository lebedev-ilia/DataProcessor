# 🎬 Video Pacing Pipeline

## 📌 Описание

Пайплайн извлекает метрики темпа видео, включая визуальные метрики и опциональную синхронизацию с аудио, людьми и объектами. Он анализирует шоты, сцены, движение, визуальные изменения, цвет, освещённость и структуру для оценки динамики и ритма видео.

### Фокус

- ✅ **Визуальный анализ** (video-only базовый режим)
- ✅ **Audio-visual sync** (получает аудио данные на вход для синхронизации)
- ✅ **Per-person motion pace** (получает данные о треках людей)
- ✅ **Object change pacing** (получает данные о детекции объектов)
- ✅ **Оптимизирован** для точности и информативности фичей
- ✅ **Подходит** для анализа влогов, рекламных роликов, TikTok/YouTube видео и кино

## ⚡ Установка

```bash
pip install opencv-python-headless numpy torch torchvision scikit-image scipy scenedetect ftfy regex tqdm clip-by-openai

# Опционально для audio-visual sync (если нужна обработка аудио файлов):
pip install librosa
```

## 🚀 Использование

### Python API

```python
from video_pacing.video_pacing import VideoPacingPipelineVisualOptimized
# или если запускается из директории модуля:
# from video_pacing import VideoPacingPipelineVisualOptimized

video_path = "example.mp4"

# Инициализация
pipeline = VideoPacingPipelineVisualOptimized(video_path)

# Базовое извлечение фичей (только визуальные)
features = pipeline.extract_all_features()

# С аудио данными (путь к файлу или предобработанные данные)
audio_data = "audio.wav"  # или словарь с предобработанными данными
features = pipeline.extract_all_features(audio_data=audio_data)

# С данными о людях (треки и ключевые точки)
person_tracks = {0: [0, 1, 2, 3, ...], 1: [10, 11, 12, ...]}  # {person_id: [frame_indices]}
person_keypoints = {0: {0: np.array([...]), 1: np.array([...])}}  # {person_id: {frame_idx: keypoints}}
features = pipeline.extract_all_features(
    person_tracks=person_tracks,
    person_keypoints=person_keypoints
)

# С данными о детекции объектов
object_detections = {
    0: [{"label": "person", "bbox": [10, 20, 100, 200], "score": 0.9}],
    1: [{"label": "car", "bbox": [50, 60, 150, 250], "score": 0.8}]
}
features = pipeline.extract_all_features(object_detections=object_detections)

# Полный анализ со всеми данными
features = pipeline.extract_all_features(
    audio_data=audio_data,
    person_tracks=person_tracks,
    person_keypoints=person_keypoints,
    object_detections=object_detections
)

# Вывод результатов
for k, v in features.items():
    print(f"{k}: {v}")
```

### CLI интерфейс

```bash
# Базовый анализ (только визуальные метрики)
python video_pacing.py --video video.mp4

# С аудио файлом для синхронизации
python video_pacing.py --video video.mp4 --audio audio.wav

# С данными о людях (JSON файл)
python video_pacing.py --video video.mp4 --person-tracks person_tracks.json

# С детекцией объектов (JSON файл)
python video_pacing.py --video video.mp4 --object-detections objects.json

# Полный анализ со всеми данными
python video_pacing.py \
    --video video.mp4 \
    --audio audio.wav \
    --person-tracks person_tracks.json \
    --object-detections objects.json \
    --output results.json

# Вывод краткой статистики
python video_pacing.py --video video.mp4 --summary
```

#### Формат JSON файлов для CLI

**person_tracks.json:**
```json
{
    "0": [0, 1, 2, 3, 4, 5],
    "1": [10, 11, 12, 13, 14]
}
```

**object_detections.json:**
```json
{
    "0": [
        {"label": "person", "bbox": [10, 20, 100, 200], "score": 0.9, "track_id": 0}
    ],
    "1": [
        {"label": "person", "bbox": [15, 25, 105, 205], "score": 0.85, "track_id": 0},
        {"label": "car", "bbox": [50, 60, 150, 250], "score": 0.8, "track_id": 1}
    ]
}
```

## 🏗️ Архитектура пайплайна

### 1. Загрузка кадров

- **OpenCV** — загрузка видео
- **Преобразование в RGB** — нормализация цветового пространства

### 2. Shot / Scene Detection

- **PySceneDetect (ContentDetector)** — детекция сцен
- **Рефайнмент через SSIM** — для точного hard-cut detection

### 3. Shot Features

**Метрики:**
- Средняя, медианная, минимальная, максимальная длительность шотов
- Энтропия длительности
- Cuts per 10s, variance
- Longest / Shortest shot

### 4. Pace Curve

**Анализ:**
- Склонность изменения длительности шотов
- Peaks (всплески длительности)
- Periodicity (автокорреляция)

### 5. Scene Pacing

**Метрики:**
- Средняя длительность сцены
- Scene changes per minute
- Scene duration variance

### 6. Motion / Optical Flow

**Анализ движения:**
- Средняя скорость движения
- Медиана, 90-й перцентиль
- Доля high-motion кадров
- Изменения направления движения

### 7. Content Change Rate

**CLIP embeddings:**
- CLIP embeddings каждого кадра
- Средние/стандартные изменения между кадрами
- High change frames ratio
- Scene embedding jumps
- Сглаживание скользящим окном

### 8. Color & Lighting Pacing

**Анализ цвета и света:**
- deltaE (Lab) для perceptual color pacing
- Изменения насыщенности и яркости
- High-frequency lighting flash (FFT)
- Luminance spikes per minute

### 9. Structural Pacing

**Структурный анализ:**
- Intro / Main / Climax pacing (медиана длительностей шотов)
- Pacing symmetry

### 10. Audio-Visual Pacing Sync (опционально)

**Синхронизация аудио и визуального темпа:**
- Корреляция между визуальной динамикой (optical flow) и аудио энергией
- AV sync score (0 — несоответствие, 1 — хореография)
- AV energy alignment (скользящая корреляция)
- Beats per cut ratio (соответствие битов и катов)

**Входные данные:**
- Путь к аудио файлу (str) или словарь с предобработанными данными:
  ```python
  {
      'energy_curve': np.ndarray,  # Энергия аудио по времени (per frame)
      'beats': List[float],  # Временные метки битов в секундах
      'tempo': float  # BPM (опционально)
  }
  ```

### 11. Per-Person Motion Pace (опционально)

**Темп движения для каждого человека:**
- Средняя скорость движения (на основе keypoints или optical flow)
- Дисперсия скорости
- Bursts of activity per minute (всплески активности)
- Freeze moments count (моменты без движения)

**Входные данные:**
- `person_tracks`: Словарь {person_id: [frame_indices]} — треки людей по кадрам
- `person_keypoints`: Словарь {person_id: {frame_idx: keypoints_array}} — ключевые точки (опционально)

### 12. Object Change Pacing (опционально)

**Темп изменения объектов:**
- New objects per 10s (новые объекты каждые 10 секунд)
- Object entry/exit rate (частота появления/исчезновения объектов)
- Main object switching rate (частота смены главного объекта)

**Входные данные:**
- `object_detections`: Словарь {frame_idx: [detections]} где каждое detection:
  ```python
  {
      'label': str,
      'bbox': [x1, y1, x2, y2],
      'score': float,
      'track_id': int  # опционально
  }
  ```

## 📊 Выходные фичи

### Shot / Cut Features

- `shot_duration_mean`, `shot_duration_median`, `shot_duration_std`
- `shot_duration_entropy`, `cuts_per_10s`, `cuts_variance`
- `longest_shot_duration`, `shortest_shot_duration`

### Pace Curve

- `pace_curve_mean`, `pace_curve_slope`, `pace_curve_peaks`, `pace_curve_periodicity`

### Scene Pacing

- `scene_changes_per_minute`, `average_scene_duration`, `scene_duration_variance`

### Motion Features

- `mean_motion_speed_per_shot`, `motion_speed_median`, `motion_speed_variance`
- `motion_speed_90perc`, `share_of_high_motion_frames`
- `optical_flow_direction_changes_per_second`

### Content Change Rate

- `frame_embedding_diff_mean`, `frame_embedding_diff_std`
- `high_change_frames_ratio`, `scene_embedding_jumps`

### Color & Lighting Pacing

- `color_histogram_diff_mean`, `color_histogram_diff_std`
- `saturation_change_rate`, `brightness_change_rate`
- `luminance_spikes_per_minute`, `high_frequency_flash_ratio`

### Structural Pacing

- `intro_speed`, `main_speed`, `climax_speed`, `pacing_symmetry`

### Audio-Visual Sync (опционально)

- `av_sync_score` — корреляция визуальной и аудио динамики (0-1)
- `av_energy_alignment` — скользящая корреляция энергии (0-1)
- `beats_per_cut_ratio` — доля битов, синхронизированных с катами

### Per-Person Motion Pace (опционально)

- `avg_pose_speed` — средняя скорость движения всех людей
- `pose_speed_variance` — дисперсия скорости движения
- `bursts_of_activity_per_minute` — всплески активности в минуту
- `freeze_moments_count` — количество моментов без движения
- `per_person_motion_pace` — детальные метрики для каждого человека:
  ```python
  {
      person_id: {
          "avg_pose_speed": float,
          "pose_speed_variance": float,
          "bursts_of_activity_per_minute": float,
          "freeze_moments_count": int
      }
  }
  ```

### Object Change Pacing (опционально)

- `new_objects_per_10s` — новые объекты каждые 10 секунд
- `object_entry_exit_rate` — частота появления/исчезновения объектов (в минуту)
- `main_object_switching_rate` — частота смены главного объекта (в минуту)

## ✨ Особенности и улучшения

- **SSIM refinement** — точнее определяются резкие шоты
- **Motion statistics** — медиана, 90-й перцентиль, направление движения
- **Content change smoothing** — скользящее среднее для CLIP embeddings
- **Color deltaE & FFT** — perceptual и high-frequency изменения цвета и света
- **Structural pacing** — медиана и асимметрия для intro/main/climax
- **Audio-visual sync** — синхронизация визуального и аудио темпа (получает аудио данные на вход)
- **Per-person motion pace** — индивидуальный анализ темпа движения для каждого человека
- **Object change pacing** — анализ темпа изменения объектов в сцене
- **CLI интерфейс** — удобная командная строка для обработки видео

## 🎯 Применение

- Оценка динамики и ритма видео
- Визуальный анализ монтажа (YouTube/TikTok/cinematic)
- Генерация фичей для ML-моделей по предсказанию вовлеченности и вирусности видео
- Анализ темпа для оптимизации контента
- Синхронизация аудио и визуального контента
- Анализ поведения людей в видео
- Отслеживание динамики объектов в сцене

## 📝 Примечания

- **Модуль не обрабатывает аудио напрямую** — он получает аудио данные на вход (путь к файлу или предобработанные данные) для синхронизации с визуальным контентом
- Для обработки аудио файлов требуется библиотека `librosa` (опционально)
- Данные о людях и объектах должны быть предоставлены из других модулей (например, `behavior` и `object_detection`)
