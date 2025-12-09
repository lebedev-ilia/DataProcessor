# 📐 Frame Composition Analysis

## 🏆 Overview

Frame Composition Analysis — это модуль для извлечения сложных визуальных метрик композиции из видеороликов. На вход подаётся последовательность кадров, на выходе формируется набор агрегированных признаков, которые можно использовать в ML-моделях (например, трансформерах для регрессии рейтинг-популярности).

### Что анализирует модуль

- **Правило третей** — расположение объектов по сетке 3×3
- **Golden ratio** — правило золотого сечения
- **Balance / Visual weight** — визуальный баланс и вес
- **Leading lines** — направляющие линии
- **Depth & Foreground/Background** — глубина пространства (MiDaS)
- **Symmetry** — горизонтальная, вертикальная, радиальная, диагональная симметрия
- **Negative space** — негативное пространство
- **Framing** — обрамление объектов (5 типов)
- **Clutter / Complexity** — сложность и загромождённость
- **Style classification** — классификация стиля композиции
- **Attention / Saliency** — внимание и салиенси

Модуль оптимизирован под батчевую обработку видео, позволяет легко добавлять новые модули и расширять набор признаков.

## 🔧 Архитектура

```
CompositionPipeline
│
├── ObjectDetector (YOLOv8)
│     └─ Детекция объектов для анализа композиции
│
├── FaceLandmarksExtractor (MediaPipe)
│     └─ Детекция лиц и ключевых точек
│
├── DepthEstimation (MiDaS DPT_Large)
│     └─ Оценка глубины сцены
│
├── AttentionModel (Grad-CAM)
│     └─ Карта внимания для анализа баланса
│
├── Modules
│     ├─ RuleOfThirdsModule
│     ├─ GoldenRatioModule
│     ├─ BalanceModule
│     ├─ LeadingLinesModule
│     ├─ DepthModule
│     ├─ SymmetryModule (расширенный)
│     ├─ NegativeSpaceModule
│     ├─ FramingModule (5 типов)
│     ├─ ClutterModule
│     ├─ StyleClassifierModule
│     └─ SaliencyModule
│
└── Aggregator
      └─ mean / std / min / max / median
```

## 📥 Input

На вход подаётся:

```python
frames: List[np.ndarray]  # RGB images (H, W, 3)
```

Каждый кадр должен иметь формат `(H, W, 3)` в RGB.

## 📤 Output

Модуль возвращает словарь с агрегированными признаками:

```python
{
    "rule_of_thirds_alignment_mean": ...,
    "golden_ratio_alignment_mean": ...,
    "balance_left_right_ratio_mean": ...,
    "leading_lines_strength_mean": ...,
    "depth_mean_mean": ...,
    "horizontal_symmetry_score_mean": ...,
    "negative_space_ratio_mean": ...,
    "framing_strength_mean": ...,
    "scene_complexity_score_mean": ...,
    "style_cinematic_prob_mean": ...,
    "saliency_entropy_mean": ...,
    # ... и другие фичи с суффиксами _mean, _std, _min, _max, _median
}
```

Выходные признаки готовы для подачи в ML-модели.

## 🧩 Modules Description

### 🎯 Rule of Thirds

**Метод:**
- Находит главный объект (детекция объектов / лица)
- Вычисляет расстояние до линий третей и пересечений
- Возвращает score ∈ [0,1]

**Фичи:**
- `main_subject_x_pos`, `main_subject_y_pos` — позиция главного объекта
- `rule_of_thirds_alignment` — выравнивание по правилу третей
- `subject_offcenter_distance`, `subject_offcenter_angle` — смещение от центра
- `secondary_subjects_count`, `secondary_subjects_alignment_score` — вторичные объекты
- `subject_balance_index` — индекс баланса объектов

### 🌀 Golden Ratio

**Метод:**
- Сравнение расположения объекта с золотой спиралью
- Проверка 4 ориентаций золотого сечения
- Корреляция объекта с золотыми точками

**Фичи:**
- `golden_ratio_alignment` — выравнивание по золотому сечению
- `golden_ratio_orientation` — ориентация золотой спирали

### ⚖ Balance & Visual Weight

**Вычисляется через:**
- Средневзвешенные координаты объектов (Grad-CAM attention)
- Яркость, контраст, площадь объекта
- Цветовой вес: saturation / V-channel

**Метрики:**
- `balance_left_right_ratio` — горизонтальный баланс
- `balance_top_bottom_ratio` — вертикальный баланс
- `mass_center_x`, `mass_center_y` — центр масс
- `visual_weight_asymmetry` — асимметрия визуального веса

### ➡ Leading Lines

**Метод:**
- Canny edge detection → определение линий
- Hough transform → выделение доминирующих направлений
- Проверка: линия направлена на главный объект?

**Метрики:**
- `leading_lines_count` — количество направляющих линий
- `leading_lines_strength` — сила направляющих линий
- `leading_lines_direction_mean` — среднее направление
- `leading_lines_to_subject_alignment` — выравнивание линий на объект

### 🔶 Depth & Foreground/Background

**Использует MiDaS DPT_Large:**
- Оценка глубины из одного RGB кадра
- Разделение на foreground/midground/background
- Кинематографические индикаторы глубины

**Метрики:**
- `depth_mean`, `depth_std`, `depth_dynamic_range`, `depth_entropy`
- `foreground_size_ratio`, `midground_presence_ratio`
- `background_clutter_index`, `foreground_depth_distance`
- `num_depth_layers` — количество слоев глубины
- `bokeh_probability`, `shallow_depth_of_field_prob`, `focus_plane_variation`

**Fallback:** Если MiDaS недоступен, используется градиентная оценка глубины.

### 🎭 Symmetry (расширенный)

**Методы:**
- Горизонтальная / вертикальная корреляция картинки с отражённой
- Радиальная симметрия через переход в полярные координаты
- Диагональная симметрия
- Симметрия по квадрантам
- Facial symmetry через landmarks (если лицо обнаружено)
- Object symmetry — симметрия расположения объектов

**Метрики:**
- `horizontal_symmetry_score`, `vertical_symmetry_score`
- `radial_symmetry_score`, `diagonal_symmetry_score`
- `top_bottom_symmetry`, `left_right_symmetry`
- `face_symmetry_score`, `eye_symmetry_score`
- `object_symmetry_score`
- `scene_symmetry_type` — тип симметрии сцены

### 🏞 Negative Space

**Использует сегментацию объектов:**
- `area(background) / area(total)` — соотношение фона
- Распределение пустого пространства по сторонам
- Энтропия фона
- Ratio пустого пространства по сторонам

**Метрики:**
- `negative_space_ratio`
- `negative_space_left/right/top/bottom`
- `empty_background_entropy`
- `object_to_background_ratio`

### 📐 Framing (5 типов)

**Детекция обрамления объектов:**

1. **Rectangular framing** — прямоугольные рамки
2. **Doorway framing** — дверные проемы (вертикальные прямоугольники)
3. **Screen within screen** — экраны внутри кадра (высокий контраст)
4. **Frame-inside-frame** — вложенные рамки
5. **Natural framing** — естественное обрамление (деревья, окна, коридоры)

**Метрики:**
- `framing_present` — наличие обрамления
- `framing_strength` — сила обрамления
- `framing_type` — тип обрамления (может быть несколько)
- `framing_types_count` — количество обнаруженных типов

### 🧩 Clutter / Complexity

**Метрики:**
- Edge density — плотность краёв
- Segmentation entropy — энтропия сегментации
- Object clutter index — индекс загромождённости объектами
- Texture complexity — сложность текстуры фона

**Метрики:**
- `edge_density`
- `region_entropy`
- `object_clutter_index`
- `background_texture_complexity`
- `scene_complexity_score` — общий индекс сложности

### 🎨 Style Classification

**Классификация стиля композиции:**

- Minimalist — минимализм
- Documentary — документальный стиль
- Vlog — влоговый стиль
- Cinematic — кинематографический
- Product-centered — продукт-центрированный
- Interview — интервью
- TikTok — стиль TikTok
- Gaming — игровой стиль
- Artistic — художественный

**Метрики:**
- `style_minimalist_prob`, `style_documentary_prob`, `style_vlog_prob`
- `style_cinematic_prob`, `style_product_centered_prob`
- `style_interview_prob`, `style_tiktok_prob`
- `style_gaming_prob`, `style_artistic_prob`

### 🧲 Attention / Saliency

**Grad-CAM attention map → метрики:**
- `saliency_center_bias_x/y` — смещение к центру
- `saliency_focus_spread` — распространение фокуса
- `saliency_entropy` — энтропия салиенси
- Attention overlap с лицами / объектами

## 📊 Aggregation

Для каждого признака вычисляются:

- **mean** — среднее значение
- **std** — стандартное отклонение
- **min / max** — минимум / максимум
- **median** — медиана

## 🚀 Installation

```bash
pip install numpy opencv-python pillow torch torchvision
pip install ultralytics    # YOLOv8
pip install mediapipe
pip install pytorch-grad-cam
pip install scikit-image
pip install scikit-learn
```

**Опционально (для depth estimation):**
```bash
# MiDaS будет загружен автоматически при первом использовании
# Требует интернет-соединение для загрузки модели
```

## ▶ Usage

### Python API

```python
from balance_composition import analyze_video
import cv2

# Загрузка кадров
cap = cv2.VideoCapture("video.mp4")
frames = []
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frames.append(frame_rgb)
cap.release()

# Анализ
features = analyze_video(frames, use_midas=True)

print(features)
```

### CLI Interface

```bash
# Базовое использование
python balance_composition.py video.mp4

# С указанием выходного файла
python balance_composition.py video.mp4 -o results.json

# Без MiDaS (использовать fallback)
python balance_composition.py video.mp4 --no-midas

# Сэмплирование кадров (каждый 5-й кадр)
python balance_composition.py video.mp4 --fps-sample 5
```

**Параметры CLI:**
- `video_path` — путь к видео файлу (обязательный)
- `-o, --output` — путь к выходному JSON файлу (по умолчанию: `video_name.json`)
- `--no-midas` — отключить MiDaS depth estimation
- `--fps-sample` — сэмплировать каждый N-й кадр (по умолчанию: 1)

## 🔧 Customization

Можно:

- ✅ Заменять любые блоки (например, YOLO → DETR)
- ✅ Отключать ненужные модули
- ✅ Добавлять свои метрики
- ✅ Включать GPU batching
- ✅ Настраивать параметры агрегации

## ☑ Recommended Models

| Задача | Оптимальная модель |
|--------|-------------------|
| Детекция | YOLOv8n-s / YOLOv8m |
| Depth | MiDaS DPT_Large |
| Стиль | ResNet50 (pretrained) |
| Салиенси | Grad-CAM (ResNet50) |
| Лицевые точки | Mediapipe FaceMesh |

## 📎 Notes

- Пайплайн работает кадр-wise, но использует агрегаторы → стабильные фичи
- Все вычисления можно распараллелить
- Модули loosely-coupled: легко расширять
- MiDaS загружается автоматически при первом использовании (требует интернет)
- Если MiDaS недоступен, используется градиентная оценка глубины

## 🎯 Применение

- Оценка композиции кадров
- Анализ визуального качества
- Генерация фичей для ML-моделей
- Оптимизация композиции видео
- Классификация стиля видео

## 📊 Примеры выходных данных

```json
{
  "rule_of_thirds_alignment_mean": 0.72,
  "golden_ratio_alignment_mean": 0.65,
  "balance_left_right_ratio_mean": 1.15,
  "depth_mean_mean": 0.45,
  "horizontal_symmetry_score_mean": 0.68,
  "framing_strength_mean": 0.32,
  "style_cinematic_prob_mean": 0.78,
  ...
}
```

## 🔄 Обновления

**2025-01-XX:**
- ✅ Добавлено правило третей и golden ratio
- ✅ Расширены метрики симметрии (диагональная, по квадрантам, object symmetry)
- ✅ Добавлены 5 типов framing (doorway, screen within screen, frame-inside-frame, natural)
- ✅ Добавлена классификация стиля композиции (9 классов)
- ✅ Интегрирован MiDaS для depth estimation
- ✅ Добавлен CLI интерфейс
- ✅ Улучшена обработка ошибок и fallback механизмы
