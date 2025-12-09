# 🌊 Optical Flow Pipeline (RAFT)

## 📌 Описание

Папка содержит продакшен‑пайплайн для расчёта оптического потока на базе torchvision RAFT, сохранения результатов (тензоры + оверлеи) и последующего статистического анализа.

## 📁 Состав

| Файл | Описание |
|------|----------|
| `main.py` | CLI-пайплайн: прогон видео через RAFT, сохранение результатов и (опционально) запуск статистики |
| `core/optical_flow.py` | Обработка видео: инициализация модели, ресайз кадров, расчёт потока, сохранение `.pt` и overlay PNG |
| `core/flow_statistics.py` | Расчёт покадровых/пространственных/временных метрик по сохранённым `.pt` |
| `core/camera_motion.py` | Эвристики движения камеры (shakiness, zoom, pan/truck/pedestal ratios, style stubs) |
| `core/advanced_features.py` | Продвинутые фичи: MEI/MEP, Foreground/Background Motion, Motion Clusters, Smoothness/Jerkiness |
| `core/config.py` | Конфигурации пайплайна и статистики |
| `core/utils.py` | Утилиты: генерация `frame_metadata.csv`, пакетный анализ |
| `FEATURES.MD` | Черновик/спецификация потенциальных фичей |
| `raft_output/` | Пример результатов для тестового видео |

## ⚡ Требования

- Python ≥ 3.10
- PyTorch + torchvision (с поддержкой RAFT весов)
- OpenCV (`opencv-python`)
- Прочее: `numpy`, `pandas`, `scipy`, `tqdm`, `scikit-learn`

### Установка зависимостей

```bash
pip install torch torchvision opencv-python numpy pandas scipy tqdm scikit-learn
```

## 🚀 Быстрый старт

1. Запустите обработку видео:

```bash
python main.py video.mp4 \
  --output raft_output \
  --model small \
  --max_dim 256 \
  --skip 15 \
  --run_stats \
  --run_camera_motion
```

Если видео находится в текущей директории, можно указать только имя файла или не указывать путь (будет использован первый найденный .mp4 файл).

### Аргументы

| Аргумент | Описание |
|----------|----------|
| `video_path` | Путь к видео файлу (опционально, если не указан, будет использован первый .mp4 в текущей директории) |
| `--output` | Директория для результатов (будет создан подкаталог `<video>_<hash>`) |
| `--model` | `small` или `large` (RAFT веса из torchvision) |
| `--max_dim` | Максимальная сторона кадра перед подачей в модель |
| `--skip` | Шаг выборки кадров (обрабатываются пары кадр0 → кадрN) |
| `--no_overlay` | Не сохранять визуализации движения |
| `--run_stats` | Выполнить статистический анализ сохранённых потоков |
| `--run_camera_motion` | Добавить анализ движения камеры (shakiness/zoom/ratios) вместе со статистикой |
| `--no_advanced_features` | Отключить продвинутые фичи (MEI, FG/BG, Clusters, Smoothness) |

## 🔄 Что делает пайплайн

1. **Читает видео**, ресайзит кадры до `max_dim` с сохранением пропорций
2. **Каждые `frame_skip` кадров** считает оптический поток (RAFT)
3. **Сохраняет:**
   - `.pt` тензоры потока (`flow/flow_000000.pt` …)
   - overlay PNG (`overlay/overlay_000000.png`) при включённом `save_overlay`
   - `metadata.json` с параметрами обработки и свойствами видео
4. **Если `--run_stats`:**
   - Покадровые метрики (средняя скорость, проценты движущихся пикселей по порогам, энтропия направлений и др.)
   - Пространственный анализ по сетке (активность регионов, направления, относительные метрики)
   - Временной анализ (тренды, периодичности FFT, переходы, сегментация по k-means)
   - Итоги: `statistical_analysis.json`, `frame_statistics.csv`, `analysis_report.md`
   - С `--run_camera_motion`: блок `camera_motion` в JSON и столбцы `camera_*` в `frame_statistics.csv` (shakiness, affine scale/rotation, entropy и др.)
   - **Продвинутые фичи** (по умолчанию включены, можно отключить через `--no_advanced_features`):
     - **Motion Energy Image (MEI/MEP)**: накопление движения за временное окно
     - **Foreground vs Background Motion**: разделение движения переднего и заднего плана
     - **Motion Clusters**: кластеризация векторов движения по направлению и скорости
     - **Smoothness/Jerkiness**: метрики плавности и резкости движения

## 📁 Структура вывода

```
raft_output/
  <video>_<hash>/
    flow/flow_000000.pt
    overlay/overlay_000000.png
    metadata.json
    frame_metadata.csv          # из core/utils.create_frame_metadata_csv
    statistical_analysis.json   # результаты flow_statistics (если run_stats)
    frame_statistics.csv        # покадровые метрики (+ camera_* при --run_camera_motion)
    analysis_report.md          # краткий человекочитаемый отчёт
```

## 📹 Камерные фичи (camera_motion)

### Покадрово (`frame_statistics.csv`)

Колонки `camera_*`:
- `motion_mean/std/max` — статистики движения
- `motion_energy` — энергия движения
- `motion_entropy` — энтропия движения
- `shake_var/mean/max` — статистики тряски
- `affine_scale/rotation/tx/ty` — аффинные параметры
- `background_ratio` — соотношение фона
- `rotation_speed` — скорость вращения

### Агрегат по ролику (`statistical_analysis.json` → `camera_motion.summary`)

- `shake_mean/std/max` — статистики тряски
- `zoom_in/out_count` — количество зумов
- `zoom_speed_mean` — средняя скорость зума
- `rotation_speed_mean/std` — статистики скорости вращения
- `pan/truck/pedestal/static_ratio` — соотношения типов движения
- `chaos_index` — индекс хаоса
- Стиль-эвристики: `style_handheld/tripod/cinematic/drone/action_cam`
- `motion_energy_sum` — суммарная энергия движения
- `motion_mean_*`, `motion_std_*` — статистики движения
- `n_frames` — количество кадров

### Конфиг порогов

`core/config.py` → `CAMERA_MOTION_CONFIG`:
- `mag_bg_thresh` — порог для фона
- `zoom_eps` — порог для зума
- `sharp_angle_thresh_deg` — порог для резких углов

Включение: `FlowStatisticsConfig.enable_camera_motion`

## 🚀 Продвинутые фичи (advanced_features)

### Motion Energy Image (MEI/MEP)

Накопление движения за временное окно для создания компактного представления активности.

**Фичи:**
- `mei_total_energy` — суммарная энергия движения
- `mei_coverage_ratio` — доля пикселей с движением
- `mei_max_energy` — максимальная энергия
- `mhi_contrast` — контраст Motion History Image
- `mhi_entropy` — энтропия истории движения
- `motion_persistence` — доля пикселей с устойчивым движением

### Foreground vs Background Motion

Разделение движения на передний и задний план для анализа активности объектов.

**Методы разделения:**
- `magnitude_threshold` — по порогу величины движения (по умолчанию)
- `spatial_clustering` — кластеризация по величине и позиции
- `segmentation` — с использованием маски сегментации (требует внешней сегментации)

**Фичи:**
- `foreground_motion_energy` — энергия движения переднего плана
- `background_motion_energy` — энергия движения заднего плана
- `ratio_foreground_background_flow` — соотношение энергий
- `foreground_coverage_ratio` — доля переднего плана

### Motion Clusters

Кластеризация векторов движения для обнаружения различных паттернов движения.

**Фичи:**
- `num_motion_clusters` — количество кластеров движения
- `largest_cluster_coverage` — покрытие крупнейшего кластера
- `cluster_diversity` — разнообразие кластеров
- `cluster_size_distribution` — распределение размеров кластеров

### Smoothness/Jerkiness

Метрики плавности и резкости движения для оценки качества съёмки.

**Фичи:**
- `smoothness_index` — индекс плавности (0-1, выше = плавнее)
- `jerkiness_index` — индекс резкости (ниже = плавнее)
- `flow_temporal_entropy` — энтропия временных изменений
- `movement_stability` — стабильность движения
- `mean_acceleration` — среднее ускорение
- `mean_jerk` — средний рывок (вторая производная)

**Конфигурация:**

В `FlowStatisticsConfig`:
- `enable_advanced_features` — включить/выключить все продвинутые фичи (по умолчанию `True`)
- `enable_mei` — включить MEI/MEP
- `enable_fg_bg` — включить разделение FG/BG
- `enable_clusters` — включить кластеризацию
- `enable_smoothness` — включить метрики плавности
- `fg_bg_method` — метод разделения FG/BG
- `fg_bg_threshold` — порог для magnitude_threshold метода
- `motion_clusters_n` — количество кластеров для кластеризации

## 💡 Мини-FAQ

**Q: Хочу только camera_motion без пространственного анализа?**  
A: Всё равно нужен `--run_stats`, но можно поднять `frame_skip` и `max_dim` уменьшить для скорости.

**Q: Хотите отдельный файл?**  
A: Можно сохранить `camera_motion` из `statistical_analysis.json`; при необходимости вынести в `camera_motion.json` в будущем.

## 🔧 Использование статистики отдельно

Если поток уже посчитан:

```python
from core.flow_statistics import FlowStatisticsAnalyzer
from core.config import FlowStatisticsConfig
from core.utils import analyze_single_video

config = FlowStatisticsConfig()
analyzer = FlowStatisticsAnalyzer(config)
results = analyze_single_video(
    "raft_output/<video_hash>/flow",
    "raft_output/<video_hash>/metadata.json",
    config=config,
    analyzer=analyzer
)
```

## ⚠️ Известные моменты

- Модель RAFT грузится из torchvision: нужен интернет при первом запуске или заранее закэшированные веса
- Для GPU установите правильный билд PyTorch с CUDA; параметр `device="auto"` выберет GPU при наличии
- При `frame_skip` слишком малом значении растёт время обработки и объём файлов

## 🐛 Troubleshooting

| Проблема | Решение |
|----------|---------|
| `FileNotFoundError` | Проверьте путь к видео и права доступа |
| `CUDA out of memory` | Уменьшите `--max_dim` или используйте `--model small` / CPU |
| RAFT веса не скачиваются | Убедитесь, что есть доступ в интернет или положите веса torchvision в локальный кеш |

## 🎯 Применение

- Анализ движения объектов
- Детекция движения камеры
- Оценка стабильности съёмки
- Анализ динамики сцены
- Разделение движения переднего и заднего плана
- Кластеризация паттернов движения
- Оценка качества съёмки (плавность, резкость)
- Компактное представление движения (MEI/MEP)
