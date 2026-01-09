# Описание фичей модуля color_light (оптимизированный набор)

Модуль для комплексного анализа цвета и освещения видео. Извлекает:
- покадровые (frame-level) компактные нормализованные фичи для VisualTransformer;
- сценовые (scene-level) агрегаты;
- видеоуровневые (video-level) агрегаты по сценам.

## Структура выходных данных

```json
{
  "frames": {
    "scene_key": {  // scene_key = "{scene_label}__{scene_id}" from scene_classification
      "frame_idx": {
        "frame_idx": 45,
        "features": { "...": "frame-level фичи" }
      }
    }
  },
  "scenes": {
    "scene_key": { "...": "scene-level агрегаты (+scene_label, +scene_id)" }
  },
  "video_features": { "...": "video-level агрегаты" },
  "sequence_inputs": {
    "frames": [[...]],   // N x D_frame, compact
    "scenes": [[...]],   // опционально
    "global": [...]
  }
}
```

## 1. Frame-level (компактный вектор для VisualTransformer)

Все основные числовые фичи, попадающие в `sequence_inputs["frames"]`, приведены к диапазону **0–1**.

### 1.1. Цвет в HSV

- **hue_mean_norm** — средний hue, нормированный как `hue_mean / 180`.  
- **hue_std_norm** — std(hue) / 180.  
- **hue_entropy** — энтропия распределения hue по 36 бинам.  
- **hue_entropy_weighted** — та же энтропия, но гистограмма взвешена нормированной насыщенностью (saturation / 255).
- **sat_mean_norm** — средняя насыщенность, нормированная как `saturation_mean / 255`.  
- **val_mean_norm** — средняя яркость Value, нормированная как `value_mean / 255`.

Дополнительно в фичах кадра хранятся ненормированные:
- `hue_mean`, `hue_std`, `saturation_mean`, `saturation_std`, `value_mean`, `value_std` — для агрегатов.

### 1.2. Цвет в Lab

- **L_mean** — средняя яркость (L‑канал, 0–255).  
- **L_contrast** — стандартное отклонение L‑канала (контраст).  
- **ab_balance** — баланс тёплых/холодных тонов: разность средних a и b (центрированных относительно 128).  
- **L_mean_norm** — нормированная яркость: `L_mean / 255` (идёт в компактный вектор кадра).

### 1.3. Палитра и доминантные цвета (Lab)

Кластера считаются в пространстве **Lab(a,b)** с помощью KMeans (до 3 кластеров).

- **dominant_lab_a**, **dominant_lab_b** — координаты доминантного кластера в Lab.  
- **dominant_lab_a_norm**, **dominant_lab_b_norm** — нормированные координаты в [0,1]:  
  `(a + 128) / 255`, `(b + 128) / 255` (используются в кадровом векторе).

Дополнительно:

- **colorfulness_index** — индекс цветности по rg/yb.  
- **colorfulness_norm** — нормированный индекс цветности (`colorfulness_index / 100`, обрезанный в [0,1]).  
- **warm_vs_cold_ratio** — отношение количества тёплых пикселей (hue 0–30, 150–180) к холодным.  
- **skin_tone_ratio** — доля пикселей кожи (H: 0–25, S ≥ 20, V ≥ 50).  
- **color_palette_entropy** — энтропия hue по 36 бинам.  
- **color_harmony_complementary_prob**, **color_harmony_analogous_prob** — компактные признаки цветовых гармоний.

Triadic и split‑complementary гармонии убраны, чтобы уменьшить размерность.

### 1.4. Освещение и контраст

Базовые (ненормированные) признаки:

- **brightness_mean**, **brightness_std** — средняя яркость и её std по серому (0–255).  
- **global_contrast** — RMS‑контраст (std по яркости).  
- **local_contrast**, **local_contrast_std** — локальный контраст по окнам.  
- **brightness_entropy**, **contrast_entropy** — энтропия 256/64‑бинных гистограмм.  
- **dynamic_range_db** — динамический диапазон в децибелах.  
- **overexposed_pixels**, **underexposed_pixels** — доля пере/недоэкспонированных пикселей.  
- **highlight_clipping_ratio**, **shadow_clipping_ratio** — доля клиппинга.  
- **lighting_uniformity_index**, **center_brightness**, **corner_brightness**, **vignetting_score** — равномерность освещения и виньетирование.

Нормализованные признаки, идущие в компактный кадровый вектор:

- **global_contrast_norm** — `global_contrast / 255` (обрезано в [0,1]).  
- **local_contrast_mean_norm** — `local_contrast / 255` (обрезано в [0,1]).  
- **overexposed_ratio**, **underexposed_ratio** — алиасы `overexposed_pixels` и `underexposed_pixels` (0–1).  
- **vignetting_score_norm** — то же, что `vignetting_score` (0–1).

### 1.5. Источники света

- **light_source_count_estimate** — оценка количества источников света (целое 0–5).  
- **soft_light_probability**, **hard_light_probability** — вероятности мягкого/жёсткого света по дисперсии лапласиана (0–1).  
- **soft_light_prob** — укороченный алиас `soft_light_probability` (0–1), используемый в компактном векторе.

Угол направления источника света (`light_direction_angle`) из финальных фич исключён.

### 1.6. Итоговый компактный вектор кадра

В `sequence_inputs["frames"]` каждый кадр представлен вектором (примерно 16–18 чисел), собранным из:

- `hue_mean_norm`  
- `hue_std_norm`  
- `hue_entropy_weighted`  
- `sat_mean_norm`  
- `val_mean_norm`  
- `L_mean_norm`  
- `global_contrast_norm`  
- `local_contrast_mean_norm`  
- `colorfulness_norm`  
- `skin_tone_ratio`  
- `overexposed_ratio`  
- `underexposed_ratio`  
- `vignetting_score_norm`  
- `soft_light_prob`  
- `dominant_lab_a_norm`  
- `dominant_lab_b_norm`

Остальные покадровые фичи доступны в `frames[scene][frame_idx]["features"]`, но не идут прямо в VisualTransformer.

## 2. Scene-level (сценовые фичи)

### 2.1. Агрегированные покадровые фичи

Для каждой числовой покадровой фичи автоматически считаются:
- `{feature}_mean` — среднее по кадрам сцены;  
- `{feature}_std` — стандартное отклонение по кадрам сцены.

Метаданные сцены:
- **num_frames** — количество обработанных кадров;  
- **num_frames_norm** — нормированная длина сцены: `num_frames / max_frames_per_scene` (обрезано в [0,1]).

### 2.2. Временные паттерны

- **brightness_change_speed** — среднее \(|Δ value\_mean|\) по кадрам сцены.  
- **scene_flicker_intensity** — std \(|Δ value\_mean|\).  
- **flash_events_count** — число вспышек, где \(|Δ value\_mean|\) превышает `mean + 2 * std`.  
- **flash_events_count_norm** — `flash_events_count / (num_frames - 1)` в [0,1].
- **color_change_speed** — среднее \(|Δ hue\_mean|\) с учётом цикличности hue (используется минимум по 180‑угловой разности).  
- **color_transition_variance** — дисперсия тех же \(\Delta hue\).  
- **color_stability** — \(1 / (1 + mean\_color\_diff)\), где `mean_color_diff` — среднее евклидово расстояние между RGB‑средними соседних кадров.  
- **color_temporal_entropy** — энтропия последовательности `hue_mean` по 18 бинам.  
- **color_pattern_periodicity** — оценка периодичности hue через автокорреляцию и пики.  
- **scene_color_shift_speed** — среднее \(|Δ hue\_mean|\) по кадрам сцены.  
- **scene_contrast** — средний `global_contrast` по кадрам.  
- **dynamic_range** — \(\max(brightness\_mean) - \min(brightness\_mean)\) в сцене.

## 3. Video-level (видеоуровневые фичи)

### 3.1. Агрегаты по сценам

Для каждой числовой сценовой фичи вычисляются:
- `{feature}_mean`  
- `{feature}_std`  
- `{feature}_min`  
- `{feature}_max`

Особенно полезны агрегаты для:
- `hue_entropy`, `colorfulness_norm`, `global_contrast_norm`;  
- `cinematic_lighting_score`, `professional_look_score`;  
- `brightness_change_speed`, `scene_flicker_intensity`, `num_frames_norm`.

### 3.2. Распределения по кадрам

- **color_distribution_entropy** — энтропия распределения `hue_mean`/`hue_mean_norm` по всему видео (36 бинов).  
- **color_distribution_gini** — коэффициент Джини для распределения оттенков по видео.

### 3.3. Стиль цветокоррекции

Компактный набор style‑фич:

- **style_teal_orange_prob**  
- **style_film_prob**  
- **style_desaturated_prob**  
- **style_hyper_saturated_prob**  
- **style_vintage_prob**  
- **style_tiktok_prob**

Все значения нормированы в 0–1 и могут дополнительно сжиматься (например, PCA) при необходимости.

### 3.4. Aesthetic & Cinematic Scores

- **nima_mean**, **nima_std** — оценки эстетики на основе глобального/локального контраста.  
- **laion_mean**, **laion_std** — оценки эстетики на основе цветности.  
- **cinematic_lighting_score** — средний скор по контрасту, локальному контрасту и мягкости света.  
- **professional_look_score** — оценка профессиональности кадра (экспозиция + контраст + цветность).

Все эти скоры находятся в [0,1].

### 3.5. Глобальная динамика

- **global_brightness_change_speed** — среднее \(|Δ brightness\_mean|\) по всему видео.  
- **global_color_change_speed** — среднее \(|Δ hue\_mean|\) (с учётом цикличности).  
- **strobe_transition_frequency** — частота сильных всплесков яркости (стробоскопы).  
- **global_color_periodicity** — периодичность цветовых паттернов по hue.  
- **global_color_shift** — средний глобальный цветовой сдвиг по hue.

## 4. Параметры обработки

- `max_frames_per_scene` — максимум кадров на сцену (по умолчанию 350);  
- `stride` — шаг выборки кадров (по умолчанию 5):  
  при `stride > 1` кадры берутся с этим шагом, иначе — равномерный сэмплинг до `max_frames_per_scene`.

## 5. Алгоритм обработки

1. Для каждой сцены формируется список индексов кадров (по `stride` либо равномерно до `max_frames_per_scene`).  
2. Для каждого выбранного кадра считаются frame-level фичи и формируется компактный вектор для `sequence_inputs["frames"]`.  
3. Для каждой сцены агрегируются покадровые фичи и вычисляются scene-level показатели.  
4. Для всего видео агрегируются сценовые фичи и вычисляются video-level признаки.  
5. Дополнительно формируются последовательности `sequence_inputs["scenes"]` и `sequence_inputs["global"]` (side‑features).

## 6. Зависимости

- `numpy >= 1.21.0`  
- `opencv-python >= 4.5.0`  
- `scipy >= 1.7.0`  
- `scikit-learn >= 1.0.0`

```

