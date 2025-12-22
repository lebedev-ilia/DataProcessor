# Описание фичей модуля frames_composition

Модуль для комплексного анализа композиции кадров видео. Извлекает покадровые фичи и агрегирует их по всему видео для оценки качества композиции.

## Основные улучшения (v2.0)

- **Обнаружение объектов и лиц**: Добавлены маски (сегментация), face_pose, eye_gaze, landmarks_visibility_ratio, bbox_area_ratio
- **Composition Anchors**: Объединены Rule of Thirds и Golden Ratio в единую метрику `composition_anchor_distance`
- **Balance**: Используется saliency map (с fallback на brightness+object_mask)
- **Depth Analysis**: Опциональный, компактный, с флагом `depth_reliable`
- **Symmetry**: Упрощена (только horizontal/vertical в fast_mode, diagonal/radial опционально)
- **Negative Space**: Использует сегментацию вместо bbox
- **Visual Complexity**: Упрощена (local variance вместо SLIC)
- **Leading Lines**: Edge thinning + saliency mask для фильтрации
- **Composition Style**: Поддержка learnable classifier и embedding
- **Per-frame вектор**: Компактный вектор ~20 dims для VisualTransformer

## Структура выходных данных

```json
{
    "frame_count": int,
    "video_composition_score": float,
    "numeric_features": {...},
    "qualitative_features": {...},
    "frame_analysis_summary": {...}
}
```

## Frame-level Features (Покадровые фичи)

### 1. Обнаружение объектов и лиц

#### object_data.object_count
Количество обнаруженных объектов в кадре (YOLO11n).

#### object_data.objects
Список объектов с информацией: bbox, center, center_x_norm, center_y_norm, confidence, class, class_id, bbox_area, bbox_area_ratio.

**Улучшения**: 
- Используются маски (сегментация) вместо bbox, если доступно (YOLOv8-seg)
- Добавлены нормализованные координаты и площадь для лучшей интерпретации
- Ограничение количества детекций через `max_detections`

#### face_data.face_count
Количество обнаруженных лиц в кадре (MediaPipe Face Mesh).

#### face_data.faces
Список лиц с информацией: bbox, center, center_x_norm, center_y_norm, landmarks, face_width, face_height, face_area, face_size_ratio, face_pose (yaw, pitch, roll), eye_gaze (x, y), landmarks_visibility_ratio.

**Улучшения**:
- Добавлена оценка позы лица (yaw, pitch, roll) - упрощенная версия на основе landmarks
- Добавлена оценка направления взгляда (eye_gaze) - упрощенная версия
- Добавлен коэффициент видимости landmarks (landmarks_visibility_ratio)
- Определяется главное лицо (main_face) с наибольшей уверенностью

**Применение:** Определение главного субъекта, анализ распределения объектов.

### 2. Composition Anchors (объединенные Rule of Thirds + Golden Ratio + Center)

#### composition_anchors.composition_anchor_distance
Минимальное нормализованное расстояние от главного субъекта до любого эстетического якоря (Rule of Thirds, Golden Ratio, Center). Компактная метрика для трансформера.

#### composition_anchors.closest_anchor_type
Тип ближайшего якоря: 'rule_of_thirds', 'golden_ratio', или 'center'.

#### composition_anchors.alignment_score
Оценка выравнивания (обратная метрика расстояния, 0-1). Для обратной совместимости.

**Примечание**: Rule of Thirds и Golden Ratio объединены в единую метрику для уменьшения избыточности.

### 2.1. Правило третей (Rule of Thirds) - для обратной совместимости

#### rule_of_thirds.alignment_score
Оценка соответствия правилу третей (0-1). Вычисляется как близость главного объекта к точкам пересечения линий третей.

#### rule_of_thirds.main_subject_x, main_subject_y
Нормализованные координаты главного объекта (0-1).

#### rule_of_thirds.distance_to_thirds
Нормализованное расстояние от главного объекта до ближайшей точки пересечения линий третей.

#### rule_of_thirds.balance_score
Баланс распределения объектов по квадрантам (0-1).

#### rule_of_thirds.quadrant_distribution
Распределение объектов по квадрантам: top_left, top_right, bottom_left, bottom_right.

**Применение:** Оценка классической композиции, соответствия профессиональным стандартам.

### 3. Золотое сечение (Golden Ratio) - DEPRECATED

⚠️ **Примечание**: Golden Ratio объединен с Rule of Thirds в `composition_anchors`. Этот раздел оставлен для обратной совместимости.

#### golden_ratio.golden_ratio_score
Оценка соответствия золотому сечению (0-1). Вычисляется через близость главного объекта к золотым точкам (φ ≈ 1.618).

#### golden_ratio.closest_orientation
Ближайшая золотая точка: top_left, top_right, bottom_left, bottom_right.

#### golden_ratio.min_distance_normalized
Минимальное нормализованное расстояние до золотых точек.

**Применение:** Анализ эстетической композиции, более естественного расположения объектов.

### 4. Визуальный баланс (Balance)

#### balance.mass_center_x, mass_center_y
Центр масс весовой карты (0-1, где 0.5 - идеальный центр). 

**Улучшения**: 
- Используется saliency map (если `use_saliency=True`) вместо brightness+object_mask
- Fallback на brightness+object_mask, если saliency недоступен
- Saliency вычисляется через градиенты и локальный контраст (lightweight proxy)
- В продакшн можно заменить на DeepGaze/UNISAL/ViT-attention

#### balance.saliency_center_offset
Расстояние между saliency center of mass и центром кадра (0-1). Новая компактная метрика.

#### balance.center_offset
Смещение центра масс от центра кадра (0-1, где 0 = идеальный центр).

#### balance.quadrant_weights
Вес каждого квадранта: top_left, top_right, bottom_left, bottom_right.

#### balance.left_right_balance
Баланс между левой и правой частями кадра (0-1, где 1 = идеальный баланс).

#### balance.top_bottom_balance
Баланс между верхней и нижней частями кадра (0-1, где 1 = идеальный баланс).

#### balance.overall_balance_score
Общая оценка баланса композиции (0-1).

**Применение:** Оценка визуального веса, предотвращение "перевешивания" кадра.

### 5. Глубина сцены (Depth Analysis)

#### depth.depth_mean
Средняя глубина сцены (0-1, где 0 = ближайший план, 1 = дальний план). Вычисляется через MiDaS.

**Улучшения**:
- Вычисляется только если `use_midas=True` и разрешение >= `min_resolution_for_depth` (по умолчанию 256)
- Добавлен флаг `depth_reliable` для индикации надежности depth analysis
- Компактный набор фичей: depth_mean, depth_std, foreground_ratio, bokeh_potential, depth_p10, depth_p90
- Дополнительные метрики (depth_entropy, depth_edge_density, midground_ratio, background_ratio) доступны только если depth_reliable=True

#### depth.depth_std
Стандартное отклонение глубины. Показывает контрастность глубины.

#### depth.depth_p10, depth_p50, depth_p90
10-й, 50-й и 90-й перцентили глубины.

#### depth.depth_dynamic_range
Динамический диапазон глубины (p90 - p10).

#### depth.foreground_ratio, midground_ratio, background_ratio
Соотношение переднего, среднего и заднего планов. Определяются через перцентили (≤p10 = foreground, ≥p90 = background).

#### depth.depth_edge_density
Плотность границ на карте глубины (0-1). Показывает выраженность переходов между планами.

#### depth.depth_entropy
Энтропия карты глубины. Показывает информационную насыщенность.

#### depth.bokeh_potential
Потенциал для эффекта боке (0-1). Вычисляется через контрастность глубины и выраженность фона.

**Применение:** Анализ кинематографичности, глубины кадра, потенциала для размытия фона.

### 6. Симметрия (Symmetry)

#### symmetry.symmetry_score
Общая оценка симметрии (среднее horizontal/vertical в fast_mode, или всех типов если fast_mode=False, 0-1).

**Улучшения**:
- В `fast_mode=True` вычисляется только horizontal и vertical симметрия
- Diagonal и radial симметрия опциональны (если `fast_mode=False`)
- Это снижает вычислительную сложность для большинства случаев

#### symmetry.dominant_symmetry_type
Доминирующий тип симметрии: horizontal, vertical, diagonal, radial.

#### symmetry.horizontal_symmetry
Оценка горизонтальной симметрии (отражение по вертикальной оси, корреляция Пирсона).

#### symmetry.vertical_symmetry
Оценка вертикальной симметрии (отражение по горизонтальной оси, корреляция Пирсона).

#### symmetry.diagonal_symmetry
Оценка диагональной симметрии (отражение по диагоналям, корреляция Пирсона).

#### symmetry.radial_symmetry
Оценка радиальной симметрии (преобразование в полярные координаты и отражение, корреляция Пирсона).

#### symmetry.symmetry_details
Детализированные оценки всех типов симметрии.

**Применение:** Анализ эстетики, формальности композиции.

### 7. Негативное пространство (Negative Space)

#### negative_space.negative_space_ratio
Доля негативного пространства в кадре (0-1). 

**Улучшения**:
- Использует object_mask из сегментации (если доступно), иначе из bbox
- Это дает более точное negative space для минималистичных/продуктовых кадров

#### negative_space.neg_space_balance_lr
Баланс негативного пространства между левой и правой частями (0-1). Компактная метрика для трансформера.

#### negative_space.object_background_ratio
Доля занятого пространства (1 - negative_space_ratio).

#### negative_space.negative_space_balance
Баланс негативного пространства между левой и правой частями (0-1).

#### negative_space.negative_space_entropy
Энтропия распределения негативного пространства. Показывает равномерность.

#### negative_space.quadrant_distribution
Распределение негативного пространства по квадрантам: top_left, top_right, bottom_left, bottom_right.

**Применение:** Анализ минимализма, профессиональности композиции.

### 8. Визуальная сложность (Complexity)

#### complexity.edge_density
Плотность границ (0-1). Вычисляется через детекцию границ Canny.

#### complexity.texture_entropy
Метрика текстуры (local variance). 

**Улучшения**:
- Заменен SLIC (дорогой) на local variance с downsampling (быстрее и дешевле)
- Вычисляется через локальную дисперсию в grayscale с downsampling
- Сохраняет информативность при меньших вычислительных затратах

#### complexity.color_complexity
Сложность цвета. Вычисляется как стандартное отклонение оттенков (HSV hue).

#### complexity.saturation_level
Уровень насыщенности (0-1). Средняя насыщенность цвета.

#### complexity.overall_complexity
Общая оценка сложности (0-1). Learnable weighted sum edge_density, texture_entropy и color_complexity.

**Примечание**: В текущей реализации используется простая версия с фиксированными весами (0.4, 0.3, 0.3). В продакшн рекомендуется обучить веса на целевой метрике (например, CTR/вовлечённость).

**Применение:** Оценка визуальной насыщенности, предотвращение хаотичности или скучности.

### 9. Ведущие линии (Leading Lines)

#### leading_lines.line_count
Общее количество обнаруженных линий (преобразование Хафа).

#### leading_lines.total_length
Общая длина всех линий.

#### leading_lines.avg_length
Средняя длина линий.

#### leading_lines.horizontal_lines, vertical_lines, diagonal_lines
Количество линий каждого типа (классификация по углу наклона).

#### leading_lines.convergence_score
Оценка сходимости линий (0-1). Показывает, насколько линии сходятся в одной точке.

#### leading_lines.dominant_line_orientation
Доминирующая ориентация линий: 'horizontal' (0°), 'vertical' (90°), 'diagonal', или 'none'. Компактная метрика для трансформера.

**Улучшения**:
- Применяется edge thinning (морфологическая операция) для более точных линий
- Используется saliency mask для фильтрации линий от фона (если `use_saliency=True`)
- Это снижает шум на UGC контенте

#### leading_lines.line_strength
Общая "сила" линий (общая длина / площадь кадра).

**Применение:** Анализ направляющих элементов, профессионализма композиции.

### 10. Стиль композиции (Composition Style)

#### composition_style.style_probabilities
Вероятности различных стилей (нормализованные, сумма = 1):
- **minimalist**: Низкая сложность, высокое негативное пространство, мало объектов
- **cinematic**: Выраженная глубина, четкие границы глубины, центрированная композиция, умеренная симметрия
- **vlog**: Наличие лиц, лицо центрировано по горизонтали, низкая сложность, умеренное количество объектов
- **product_centered**: Крупный главный объект, соответствие правилу третей, высокий потенциал боке

#### composition_style.dominant_style
Доминирующий стиль композиции.

#### composition_style.style_confidence
Уверенность в доминирующем стиле (вероятность доминирующего стиля).

**Применение:** Классификация типа контента, анализ стиля съемки.

### 11. Общая оценка композиции

#### overall_composition_score
Финальная оценка композиции кадра (0-1). 

⚠️ **ВАЖНО**: Не подавать в трансформер! Это агрегированная метрика с фиксированными весами, которая лишит трансформер возможности учиться. Использовать только в агрегатах / для explainability.

Взвешенная сумма оценок (текущая версия, рекомендуется сделать learnable):
- Правило третей: 20%
- Баланс: 15%
- Негативное пространство: 15%
- Глубина: 15%
- Симметрия: 10%
- Ведущие линии: 10%
- Сложность: 10%
- Уверенность в стиле: 5%

**Применение:** Общая оценка качества композиции кадра.

## Video-level Features (Видеоуровневые фичи)

### Агрегированные числовые фичи (numeric_features)

Для каждой покадровой фичи вычисляются 6 статистик:
- `{feature}_mean` - среднее значение
- `{feature}_std` - стандартное отклонение
- `{feature}_min` - минимальное значение
- `{feature}_max` - максимальное значение
- `{feature}_median` - медиана
- `{feature}_range` - размах (max - min)

Пример: `rule_of_thirds.alignment_score` → `rule_of_thirds.alignment_score_mean`, `rule_of_thirds.alignment_score_std`, и т.д.

### Качественные фичи (qualitative_features)

#### dominant_composition_style
Самый частый стиль композиции по всем кадрам.

#### style_distribution
Распределение стилей: количество кадров каждого стиля.

#### dominant_symmetry_type
Самый частый тип симметрии по всем кадрам.

#### symmetry_distribution
Распределение типов симметрии: количество кадров каждого типа.

#### style_consistency
Доля кадров в доминирующем стиле (0-1). Показывает консистентность стиля по видео.

### Сводка анализа (frame_analysis_summary)

#### total_frames_analyzed
Общее количество проанализированных кадров.

#### best_frames
Список 3 кадров с наивысшими оценками композиции: index, score.

#### worst_frames
Список 3 кадров с наинизшими оценками композиции: index, score.

#### style_summary
Статистика по каждому стилю:
- count: количество кадров
- avg_score: средняя оценка композиции
- best_score: наилучшая оценка

#### score_range
Диапазон оценок композиции: min, max, mean.

### Общая оценка видео

#### video_composition_score
Средняя оценка композиции по всем кадрам (0-1).

#### frame_count
Количество обработанных кадров.

## Параметры конфигурации

- `device`: Устройство для обработки ('cuda' или 'cpu')
- `yolo_model_path`: Путь к модели YOLO (по умолчанию: 'yolo11n.pt')
- `yolo_conf_threshold`: Порог уверенности для детекции объектов (по умолчанию: 0.3)
- `max_num_faces`: Максимальное количество лиц для детекции (по умолчанию: 5)
- `min_detection_confidence`: Минимальная уверенность детекции лиц (по умолчанию: 0.5)
- `use_midas`: Использовать ли MiDaS для анализа глубины (по умолчанию: True)
- `num_depth_layers`: Количество слоев глубины (по умолчанию: 3)
- `slic_n_segments`: Количество сегментов для SLIC (по умолчанию: 100)
- `slic_compactness`: Компактность для SLIC (по умолчанию: 10)
- `brightness_weight`: Вес яркости для баланса (по умолчанию: 0.65)
- `object_weight`: Вес объектов для баланса (по умолчанию: 0.35)

## Алгоритм обработки

1. Для каждого кадра извлекаются объекты (YOLO) и лица (MediaPipe)
2. Определяется главный субъект (лицо или самый крупный объект)
3. Вычисляются все покадровые фичи (правило третей, золотое сечение, баланс, глубина, симметрия, негативное пространство, сложность, ведущие линии)
4. Определяется стиль композиции на основе всех фич
5. Вычисляется общая оценка композиции кадра
6. По всем кадрам агрегируются числовые фичи (mean/std/min/max/median/range)
7. Определяются качественные характеристики (доминирующий стиль, симметрия, консистентность)
8. Формируется сводка анализа (лучшие/худшие кадры, статистика по стилям)

## Зависимости

- `ultralytics` (YOLO)
- `mediapipe` (Face Mesh)
- `torch` (MiDaS)
- `opencv-python`
- `numpy`
- `scikit-image` (SLIC)
- `scipy`

## Temporal / Shot-aware признаки (рекомендация для будущих версий)

Для улучшения предсказательной силы рекомендуется добавить:

- **Shot boundary detection**: Детекция границ кадров/сцен
- **Shot-level агрегаты**: Средние значения композиции по каждому шоту
- **shot_change_rate**: Частота смены кадров (кадров в секунду)
- **avg_time_on_subject**: Среднее время на главном субъекте
- **face_presence_ratio**: Доля кадров с лицом
- **temporal_variability**: Стандартное отклонение overall_composition_score по шотам (мера изменчивости стиля)
- **frame_count, frame_sampling_rate**: Временные характеристики важны для популярности

Эти признаки обычно мощны для предсказания вовлечённости и популярности контента.

## Применение

Модуль используется для:
- Оценки качества композиции видео
- Анализа соответствия профессиональным стандартам
- Классификации стиля съемки
- Выявления лучших и худших кадров
- Анализа консистентности стиля по видео
- Извлечения per-frame векторов для VisualTransformer

