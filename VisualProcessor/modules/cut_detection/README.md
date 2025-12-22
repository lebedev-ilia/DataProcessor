# ✂️ Cut Detection Module

## 📌 Описание

Модуль для детекции катов, переходов и статистики по монтажу видео. Предоставляет комплексный анализ видеомонтажа, включая:

- **Hard Cuts** (резкие каты) — детекция мгновенных переходов между кадрами
- **Soft Cuts** (мягкие переходы) — детекция fade-in/out и dissolve
- **Motion-based Cuts** — детекция whip pan и zoom transitions
- **Stylized Transitions** — классификация стилизованных переходов (wipe, slide, glitch и др.)
- **Jump Cuts** — детекция jump cuts (резкие изменения позы при том же фоне)
- **Shot & Scene Analysis** — сегментация на shots и scenes
- **Audio-assisted Detection** — использование аудио для улучшения детекции

## 🚀 Улучшения (v2.0+)

### Версия 2.1 (новые функции)

**Добавлено:**
- ✅ **Speed ramp cuts detection** — детекция ускоренных/замедленных переходов через анализ variance в optical flow
- ✅ **Scene transition types analysis** — анализ типов переходов между сценами (hard cut, fade, dissolve, motion, stylized)
- ✅ **Scene whoosh transition detection** — детекция whoosh звуков между сценами через аудио спектральный анализ
- ✅ **Enhanced edit style classification** — классификация стилей монтажа (fast, cinematic, meme, social, slow, high-action) на основе статистики
- ✅ **Улучшенный CLI** — дополнительные опции для управления производительностью и выводом

## 🚀 Улучшения (v2.0)

### 1. Hard Cuts Detection

**Улучшения:**
- ✅ **Адаптивные пороги** — вместо фиксированных значений используются локальные статистики (медиана + стандартное отклонение)
- ✅ **Deep feature differences** — использование предобученных CNN (ResNet18/50) для извлечения embeddings и детекции через cosine distance
- ✅ **Temporal smoothing** — применение Gaussian фильтра для уменьшения ложных срабатываний

**Параметры:**
- `use_deep_features=True` — использовать deep embeddings
- `use_adaptive_thresholds=True` — адаптивные пороги
- `temporal_smoothing=True` — временное сглаживание

### 2. Soft Cuts (Fade/Dissolve)

**Улучшения:**
- ✅ **Gradient-based detection** — анализ градиентов по всем каналам (HSV + Lab color space)
- ✅ **Optical flow consistency** — проверка плавности optical flow для детекции dissolve
- ✅ **Cumulative distribution** — учет кумулятивных изменений яркости

**Параметры:**
- `use_flow_consistency=True` — использовать optical flow для dissolve detection

### 3. Motion-based Cuts

**Улучшения:**
- ✅ **Direction analysis** — анализ согласованности направления потока (whip pan vs zoom)
- ✅ **Adaptive threshold** — динамический порог на основе percentiles или z-score
- ✅ **Type classification** — автоматическая классификация на whip pan и zoom

**Параметры:**
- `use_direction_analysis=True` — анализ направления потока
- `adaptive_threshold=True` — адаптивный порог

### 4. Stylized Transitions (CLIP Zero-shot)

**Улучшения:**
- ✅ **Temporal aggregation** — усреднение вероятностей по окну ±5-10 кадров
- ✅ **Multi-modal input** — объединение оригинального кадра, difference frame и optical flow visualization
- ✅ **Feature cache** — кэширование features для повторно встречающихся кадров

**Параметры:**
- `use_temporal_aggregation=True` — временная агрегация
- `use_multimodal=True` — мультимодальный ввод

### 5. Jump Cuts

**Улучшения:**
- ✅ **Background embedding** — использование ResNet/CLIP embeddings вместо SSIM для оценки схожести фона
- ✅ **Pose estimation** — комбинирование face + pose landmarks (Mediapipe)
- ✅ **Multi-object support** — поддержка нескольких объектов

**Параметры:**
- `use_background_embedding=True` — использовать deep embeddings для фона
- `use_pose_estimation=True` — использовать pose estimation

### 6. Audio-assisted Detection

**Улучшения:**
- ✅ **Dynamic thresholding** — учет RMS/loudness для определения значимых пиков
- ✅ **Onset clustering** — группировка пиков в кластеры для стабильного сопоставления
- ✅ **Multi-band analysis** — анализ низких и высоких частот отдельно

**Параметры:**
- `use_multiband=True` — мультиполосный анализ
- `use_dynamic_threshold=True` — динамический порог
- `use_clustering=True` — кластеризация onset peaks

### 7. Shot & Scene Aggregation

**Улучшения:**
- ✅ **Semantic clustering** — кластеризация shots по визуальным embeddings (DBSCAN)
- ✅ **Audio + visual fusion** — объединение аудио событий и визуальных признаков
- ✅ **Dynamic scene length** — адаптивный порог на основе типа контента (action vs dialogue)

**Параметры:**
- `use_semantic_clustering=True` — семантическая кластеризация
- `frame_embeddings` — embeddings кадров для кластеризации
- `audio_events` — аудио события для fusion

## ⚡ Установка

### Основные зависимости

```bash
pip install numpy opencv-python scikit-image librosa scipy torch torchvision
```

### Опциональные зависимости

```bash
# Для CLIP zero-shot classification
pip install git+https://github.com/openai/CLIP.git

# Для Mediapipe (face/pose detection)
pip install mediapipe

# Для кластеризации
pip install scikit-learn
```

## 🚀 Использование

### Базовое использование

```python
import cv2
from cut_detection import CutDetectionPipeline

# Загрузка видео
cap = cv2.VideoCapture('video.mp4')
frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frames.append(frame)
cap.release()

# Инициализация pipeline
pipeline = CutDetectionPipeline(
    fps=30,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    clip_zero_shot=True,
    use_deep_features=True,
    use_adaptive_thresholds=True,
    use_semantic_clustering=True
)

# Обработка
result = pipeline.process_video_frames(
    frames_bgr=frames,
    audio_path='audio.wav'  # опционально
)

# Получение features
features = result['features']
detections = result['detections']

print(f"Hard cuts: {features['hard_cuts_count']}")
print(f"Cuts per minute: {features['cuts_per_minute']}")
print(f"Fade in: {features['fade_in_count']}")
print(f"Whip pan transitions: {features['whip_pan_transitions_count']}")
```

### Командная строка

```bash
# Базовое использование
python cut_detection.py --video path/to/video.mp4

# С аудио файлом
python cut_detection.py --video path/to/video.mp4 --audio path/to/audio.wav

# С сохранением результатов в JSON
python cut_detection.py --video path/to/video.mp4 --output results.json

# Опции производительности
python cut_detection.py --video path/to/video.mp4 --no-clip --no-deep-features --device cpu

# Ограничение количества кадров
python cut_detection.py --video path/to/video.mp4 --max-frames 3000
```

**Параметры CLI:**
- `--video` (обязательный) — путь к видеофайлу
- `--audio` (опционально) — путь к аудио файлу для улучшения детекции
- `--output` (опционально) — путь для сохранения результатов в JSON
- `--fps` (опционально) — переопределить FPS (по умолчанию определяется из видео)
- `--device` — устройство: `auto`, `cpu`, или `cuda` (по умолчанию: `auto`)
- `--no-clip` — отключить CLIP-based классификацию переходов
- `--no-deep-features` — отключить deep feature extraction (ускоряет обработку)
- `--max-frames` — максимальное количество кадров для обработки (по умолчанию: 5000)

## 📊 Выходные Features

### Hard Cuts

- `hard_cuts_count` — количество резких катов
- `hard_cuts_per_minute` — катов в минуту
- `hard_cut_strength_mean` — средняя сила катов
- `hard_cut_strength_p25`, `hard_cut_strength_p50`, `hard_cut_strength_p75` — перцентили распределения сил катов

### Soft Cuts

- `fade_in_count` — количество fade-in
- `fade_out_count` — количество fade-out
- `dissolve_count` — количество dissolve
- `avg_fade_duration` — средняя длительность fade

### Motion-based

- `motion_cuts_count` — количество motion-based катов
- `whip_pan_transitions_count` — количество whip pan переходов
- `zoom_transition_count` — количество zoom переходов
- `speed_ramp_cuts_count` — количество speed ramp катов (ускоренные/замедленные переходы)
- `motion_cut_intensity_score` — интенсивность motion катов

### Stylized Transitions

- `transition_hard_cut_count` — количество hard cut переходов
- `transition_fade_count` — количество fade переходов
- `transition_dissolve_count` — количество dissolve переходов
- `transition_whip_pan_count` — количество whip pan переходов
- `transition_zoom_transition_count` — количество zoom переходов
- `transition_wipe_transition_count` — количество wipe переходов
- `transition_slide_transition_count` — количество slide переходов
- `transition_glitch_transition_count` — количество glitch переходов
- `transition_flash_transition_count` — количество flash переходов
- `transition_luma_wipe_transition_count` — количество luma wipe переходов
- `edit_style_*_prob` — вероятности для каждого стиля редактирования (из CLIP zero-shot)

### Edit Style Classification (based on statistics)

Классификация стилей монтажа на основе статистики (из FEATURES.MD):

- `edit_style_fast_prob` — вероятность fast-cut montage стиля
- `edit_style_slow_prob` — вероятность slow-paced editorial стиля
- `edit_style_cinematic_prob` — вероятность cinematic editing стиля
- `edit_style_meme_prob` — вероятность meme-style editing стиля
- `edit_style_social_prob` — вероятность social media style
- `edit_style_high_action_prob` — вероятность high-action-edit (GoPro) стиля

### Jump Cuts

- `jump_cuts_count` — количество jump cuts
- `jump_cut_intensity` — интенсивность jump cuts
- `jump_cut_ratio_per_minute` — соотношение jump cuts в минуту

### Timing & Rhythm

- `cuts_per_minute` — катов в минуту (cuts_per_second удален как дублирующая метрика)
- `median_cut_interval` — медианный интервал между катами
- `min_cut_interval`, `max_cut_interval` — минимальный и максимальный интервалы
- `cut_interval_std` — стандартное отклонение интервалов
- `cut_interval_cv` — коэффициент вариации интервалов (std/mean)
- `cut_interval_entropy` — нормализованная энтропия интервалов [0, 1]
- `cut_rhythm_uniformity_score` — оценка равномерности ритма (1 - CV)

### Shot Statistics

- `avg_shot_length` — средняя длительность shot
- `median_shot_length` — медианная длительность shot
- `shot_length_p10`, `shot_length_p25`, `shot_length_p75`, `shot_length_p90` — перцентили распределения длительностей
- `shot_length_histogram` — нормализованная гистограмма длительностей (8 бинов)
- `short_shots_ratio` — доля коротких shots (<1s)
- `long_shots_ratio` — доля длинных shots (>4s)
- `very_long_shots_count` — количество очень длинных shots (>10s)
- `extremely_short_shots_count` — количество очень коротких shots (<0.25s)

### Scene Statistics

- `scene_count` — количество сцен
- `avg_scene_length_shots` — средняя длительность сцены в shots
- `scene_to_shot_ratio` — соотношение сцен к shots
- `scene_hard_cut_transitions` — количество hard cut переходов между сценами
- `scene_fade_transitions` — количество fade переходов между сценами
- `scene_dissolve_transitions` — количество dissolve переходов между сценами
- `scene_motion_transitions` — количество motion переходов между сценами
- `scene_stylized_transitions` — количество stylized переходов между сценами

### Audio Features

- `audio_cut_alignment_score` — оценка синхронизации катов с аудио
- `audio_spike_cut_ratio` — соотношение аудио пиков к катам
- `scene_whoosh_transition_prob` — вероятность whoosh переходов между сценами (на основе аудио спектрального анализа)

## 📋 Raw Detections

Объект `detections` содержит детальную информацию:

```python
detections = {
    'hard_cut_indices': [10, 45, 120, ...],  # индексы кадров с hard cuts
    'hard_cut_strengths': [2.5, 3.0, 2.8, ...],  # силы катов
    'soft_events': [
        {'type': 'fade_in', 'start': 5, 'end': 12, 'duration_s': 0.23},
        # ...
    ],
    'motion_cut_indices': [30, 80, ...],
    'motion_cut_intensities': [15.2, 18.5, ...],
    'motion_cut_types': ['whip_pan', 'zoom', ...],
    'stylized_counts': {'hard cut': 5, 'fade': 2, ...},
    'jump_cut_indices': [25, 67, ...],
    'jump_cut_scores': [0.8, 0.6, ...],
    'shot_boundaries_frames': [0, 10, 45, 120, ...],
    'scene_boundaries_shot_idx': [(0, 3), (4, 7), ...]
}
```

## 🔧 Настройка параметров

### Адаптация под жанр видео

Для разных жанров могут потребоваться разные пороги:

```python
# Для action видео (быстрые каты)
pipeline = CutDetectionPipeline(
    fps=30,
    use_adaptive_thresholds=True  # автоматическая адаптация
)

# Для документальных (медленные каты)
# Можно использовать фиксированные пороги через параметры функций
```

### Оптимизация производительности

```python
# Отключить deep features для ускорения
pipeline = CutDetectionPipeline(
    use_deep_features=False,
    use_semantic_clustering=False
)

# Использовать CPU вместо GPU
pipeline = CutDetectionPipeline(device='cpu')
```

## ⚡ Производительность

| Конфигурация | Производительность |
|--------------|-------------------|
| Без deep features | ~10-30 FPS |
| С deep features (CPU) | ~2-5 FPS |
| С deep features (GPU) | ~15-30 FPS |
| С CLIP (GPU) | ~5-10 FPS |

## ⚠️ Ограничения

1. **Память** — для длинных видео (>5000 кадров) рекомендуется обработка по частям
2. **CLIP** — требует GPU для приемлемой скорости
3. **Mediapipe** — требуется для jump cut detection
4. **Аудио** — опционально, но улучшает точность детекции

## 🔮 Будущие улучшения

- [ ] Интеграция 3D CNN / I3D для motion-based detection
- [ ] Обученная CNN для fade/dissolve detection
- [ ] Multi-object tracking для jump cuts
- [ ] Real-time processing mode
- [ ] Batch processing для множественных видео
