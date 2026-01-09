# `video_pacing` (Visual module, Tier‑0 baseline)

Модуль считает **признаки темпа/монтажа** (shot pacing) и связанные метрики движения/семантических/цветовых изменений **строго на sampled кадрах** от Segmenter.

## Входы

### Основной вход
- **`frames_dir`**: директория Segmenter с `metadata.json` и батчами кадров.
- **`metadata["video_pacing"]["frame_indices"]`**: список индексов в **union-domain** (0..N-1) — кадры, которые обрабатывает модуль.

### Time-axis (обязательно)
- **`metadata["union_timestamps_sec"]`**: список timestamp’ов (сек) для каждого union-кадра. Это **source-of-truth** для времени (монтаж/окна/скорости).

### Зависимости (hard deps, no-fallback)
Модуль не имеет права “молча деградировать”: если зависимости отсутствуют/не покрывают `frame_indices` → **error**.

- **`core_optical_flow`**: `result_store/.../core_optical_flow/flow.npz`
  - используется `motion_norm_per_sec_mean` (кривая движения).
- **`core_clip`**: `result_store/.../core_clip/embeddings.npz`
  - используется `frame_embeddings` (семантические эмбеддинги).

## Выход (артефакт)

Пишется через `BaseModule.save_results()` в:
- `result_store/<platform_id>/<video_id>/<run_id>/video_pacing/video_pacing_features_<ts>_<uid>.npz`

### Ключи NPZ
- **`frame_indices`**: `(N,) int32` — union-domain индексы кадров модуля.
- **`shot_boundary_frame_indices`**: `(S,) int32` — union-domain индексы кадров, являющихся **началом** нового шота (первый элемент обычно равен `frame_indices[0]`).
- **`features`**: `object(dict)` — словарь агрегированных признаков (см. ниже).
- **`meta`**: `object(dict)` — canonical meta от `BaseModule` (run identity keys, schema/producer versions, models_used/model_signature, status/empty_reason и т.д.).

## No-fallback / empty semantics

- **No-fallback**:
  - нет `frame_indices` / нет `union_timestamps_sec` / time-axis не монотонна;
  - нет `core_clip` / нет `core_optical_flow`;
  - `core_*` не покрывают `frame_indices` (несогласованный sampling между компонентами).
- **Empty outputs**: не предусмотрены для baseline; “пусто” считается ошибкой входа (например, `frame_indices` пустой).

## Параметры (config)

Передаются через `config` (CLI: `VisualProcessor/modules/video_pacing/main.py`):
- **`downscale_factor`** (`float`, default `0.25`): downscale для дешёвых визуальных метрик (shot detection / color / lighting).
- **`min_shot_length_seconds`** (`float`, default `0.15`): минимальная длительность шота (в секундах) для merge слишком коротких шотов.
- **`shot_detect_k`** (`float`, default `6.0`): множитель для робастных порогов (MAD) в shot boundary detection.

## Фичи (`features`)

### A) Shot statistics (в секундах)
- **`shots_count`**: число шотов.
- **`shot_duration_mean`**, **`shot_duration_median`**, **`shot_duration_min`**, **`shot_duration_max`**, **`shot_duration_std`**
- **`shot_duration_entropy`**: энтропия распределения длительностей (20 бинов).
- **`shot_duration_mean_normalized`**: `mean / video_length_seconds`.
- **`shot_length_gini`**: Джини по длительностям шотов.
- **`short_shot_fraction`**: доля шотов короче 0.5s.
- **`quick_cut_burst_count`**: число “бурстов” ≥3 cut’ов в окне 1s.
- **`shot_length_histogram_5bins`**: 5‑мерный вектор долей шотов по бинам длительности `[0–0.3, 0.3–0.7, 0.7–1.5, 1.5–3.0, >3.0]`.
- **`tempo_entropy`**: энтропия распределения длительностей по 5 бинам.
- **`cuts_variance`**: дисперсия длительностей шотов (sec²).
- **`cuts_per_10s`**, **`cuts_per_10s_max`**, **`cuts_per_10s_median`**: частота cut’ов (в окнах 10s; значения в 1/sec).
- **`cut_density_map_8bins`**: 8‑мерный вектор плотности cut’ов по 8 равным временным сегментам (в 1/sec).

### B) Pace curve (pattern)
- **`pace_curve_slope`**: тренд по последовательности `log1p(shot_duration_sec)` (линейная регрессия по номеру шота).
- **`pace_curve_slope_normalized`**: `pace_curve_slope * mean(shot_duration_sec)` (масштабирование).
- **`pace_curve_peaks`**, **`pace_curve_peaks_mean_prominence`**, **`pace_curve_peak_positions`**
- **`pace_curve_dominant_period_sec`**, **`pace_curve_power_at_period`**: периодичность по автокорреляции.

### C) Motion (from `core_optical_flow`)
Все значения считаются по кривой `motion_norm_per_sec_mean` (уже нормализована на dt/max(H,W) в core provider).
- **`mean_motion_speed_per_shot`**, **`motion_speed_median`**, **`motion_speed_variance`**, **`motion_speed_90perc`**
- **`share_of_high_motion_frames`**: доля кадров выше 75‑го перцентиля.
- **`share_of_high_motion_shots`**: доля шотов с высокой средней скоростью.
- **`motion_shot_corr`**: корреляция (Пирсон) между длительностью шота и его средней “motion speed”.

### D) Content change rate (from `core_clip`, per-second)
CLIP cosine distance между соседними кадрами, нормализованная на \(dt\) (`union_timestamps_sec`).
- **`frame_embedding_diff_mean`**, **`frame_embedding_diff_std`**
- **`high_change_frames_ratio`**: доля переходов выше 75‑го перцентиля.
- **`scene_embedding_jumps`**: число переходов выше `mean + 2σ`.
- **`semantic_change_burst_count`**: число “бурстов” ≥3 high-change переходов в окне 5s.

### E) Color pacing (per-second)
DeltaE(LAB) между соседними кадрами, нормализованная на \(dt\).
- **`color_change_rate_mean`**, **`color_change_rate_std`**
- **`color_change_bursts`**: пики detrended‑скорости DeltaE.
- **`saturation_change_rate`**, **`brightness_change_rate`**: std от \(\Delta S / dt\), \(\Delta V / dt\) (HSV).

### F) Lighting pacing
- **`luminance_spikes_per_minute`**: количество резких изменений яркости в минуту (по робастному порогу на \(\Delta L / dt\)).

### G) Structural pacing
Медианная длительность шота (sec) в четвертях видео (по последовательности шотов).
- **`intro_speed`**, **`main_speed`**, **`climax_speed`**
- **`pacing_symmetry`**: `(climax - intro) / median_overall`.

## Производительность

- CPU‑heavy: требует чтения sampled кадров и вычисления простых метрик; сложность \(O(N)\) по числу sampled кадров.
- Память: хранит только небольшие массивы переходов/агрегатов, без сохранения пер‑кадровых эмбеддингов (они живут в `core_clip`).


