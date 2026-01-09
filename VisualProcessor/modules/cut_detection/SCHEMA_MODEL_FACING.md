# `cut_detection` — model-facing NPZ schema (v1)

Цель: дать моделям (особенно transformer’ам) **сырые временные сигналы** и события, а не только “финальные” агрегаты/пороговые детекции.
Это позволяет FeatureEncoder’у учиться:
- устойчиво работать на видео любой длины (120 → 72000 кадров),
- делать pooling/attention по важным моментам,
- быть менее зависимым от хрупких эвристик постпроцессинга.

Связанный документ: `docs/models_docs/FEATURE_ENCODER_CONTRACT.md`.

---

## 1) Артефакт и размещение

Путь (per-run):

- `result_store/<platform_id>/<video_id>/<run_id>/cut_detection/cut_detection_model_facing_<ts>_<uid>.npz`

> Этот артефакт **дополнительный** и не заменяет существующий
> `cut_detection_features_<ts>_<uid>.npz` (backward compatible).

CLI policy:
- Default: CLI writes the model-facing artifact (best-effort).
- `--no-write-model-facing-npz` — отключает запись model-facing артеакта.
- `--write-model-facing-npz` — совместимость (явно включает запись, хотя она включена по умолчанию).
- `--require-model-facing-npz` — делает запись обязательной (если запись не удалась → `error`).

---

## 2) Time axis / индексация (критично)

Источник истины времени: `frames_dir/metadata.json.union_timestamps_sec`.

Модуль получает `frame_indices (N,)` в union-domain из `metadata["cut_detection"]["frame_indices"]` и формирует:

- `times_s (N,) = union_timestamps_sec[frame_indices]`
- Для переходов между соседними sampled кадрами определяем “pair index”:
  - pair `i` соответствует переходу `(frame_indices[i-1] → frame_indices[i])` для `i=1..N-1`
  - массивы per-pair имеют длину `N-1` и используют индекс `i=1..N-1`, но храним их в 0-based массиве `[0..N-2]`

Определяем “pair time” (центр перехода):

- `pair_times_s (N-1,) = 0.5 * (times_s[1:] + times_s[:-1])`
- `pair_dt_s (N-1,) = times_s[1:] - times_s[:-1]`

---

## 3) Ключи NPZ (модельные входы)

### 3.1 Identity / alignment (обязательные)

- `frame_indices (N,) int32` — union-domain
- `union_timestamps_sec (N,) float32` — timestamps для `frame_indices` (alias для `times_s`)
- `times_s (N,) float32` — timestamps для `frame_indices`
- `pair_times_s (N-1,) float32`
- `pair_dt_s (N-1,) float32`

### 3.2 Dense curves (per-pair) — минимальный набор (обязательные)

Все значения соответствуют переходам между соседними sampled кадрами.

- `hist_diff_l1 (N-1,) float32`
  - L1 distance между нормированными HSV histogram’ами (cheap content change proxy).
- `ssim_drop (N-1,) float32`
  - `1 - SSIM(grayA, grayB)` после применения `ssim_max_side` policy.
- `flow_mag (N-1,) float32`
  - magnitude proxy:
    - либо Farneback mean magnitude на downscale (`flow_max_side`),
    - либо reuse `core_optical_flow.motion_norm_per_sec_mean` (если aligned), приведённый к “per-pair” (см. §5.2).
- `hard_score (N-1,) float32`
  - “сырая” комбинированная оценка hard cut до морфологии/merge:
    - рекомендуемая семантика: сумма триггеров (hist/ssim/flow/(optional deep)).

Дополнительно (рекомендуется для encoder, особенно если включен cascade режим):
- `ssim_valid_mask (N-1,) bool`
- `flow_valid_mask (N-1,) bool`
- `deep_valid_mask (N-1,) bool`
  - если сигнал не вычислялся на конкретном pair — соответствующее значение в curve **должно быть `NaN`**, а mask=false.

### 3.3 Optional dense curves (per-pair)

Если включены deep features / дополнительные сигналы:

- `deep_cosine_dist (N-1,) float32` (optional)
  - cosine distance между соседними deep embeddings.

Soft cuts (optional, for encoder; shapes noted):
- `soft_hsv_v (N,) float32`
- `soft_lab_l (N,) float32`
- `soft_hist_diff_l1 (N-1,) float32`
- `soft_flow_mag (N-1,) float32`
- `soft_flow_valid_mask (N-1,) bool`

Motion cuts (optional, for encoder; all are per-pair `N-1`):
- `motion_flow_mag (N-1,) float32`
- `motion_dir_consistency (N-1,) float32` (may be `NaN` where not computed)
- `motion_mag_variance (N-1,) float32` (may be `NaN` where not computed)
- `motion_camera_motion_flag (N-1,) bool` (best-effort)
- `motion_dir_valid_mask (N-1,) bool`
- `motion_var_valid_mask (N-1,) bool`
- `motion_cam_valid_mask (N-1,) bool`
- `threshold_hist (N-1,) float32` (optional)
- `threshold_ssim (N-1,) float32` (optional)
- `threshold_flow (N-1,) float32` (optional)
- `threshold_deep (N-1,) float32` (optional)
  - если используются adaptive thresholds, можно сохранить пороги для воспроизводимости.

Если сигнал не вычислялся — ключ **не пишем** (MVP), либо пишем массив `NaN` (если нужна фиксированная форма).

---

## 4) События (events) — модельные “спайки”

Цель: дать encoder’у компактный sparse stream поверх dense curves.

### 4.1 Unified events arrays (рекомендуемый формат)

Сохраняем единый список событий с cap на размер (например 4096; если больше — top-k по strength):

- `event_times_s (E,) float32`
- `event_type_id (E,) int16`
- `event_strength (E,) float32`
- `event_pair_index (E,) int32`
  - индекс в per-pair массивах (`0..N-2`), ближайший к событию.

Опционально:
- `event_contrib_mask (E, C) bool`
  - какие источники внесли вклад (например C=4: hist/ssim/flow/deep).
- `event_start_time_s (E,) float32`
- `event_end_time_s (E,) float32`
  - Для span-событий (fade/dissolve) задаёт границы, для point-событий start=end=event_times_s.

### 4.2 Event taxonomy (v1)

`event_type_id` (int16):
- `1`: hard_cut
- `2`: fade_in
- `3`: fade_out
- `4`: dissolve
- `5`: motion_cut
- `6`: whip_pan
- `7`: zoom
- `8`: speed_ramp
- `9`: jump_cut
- `100+`: reserved (stylized via CLIP / future extensions)

Справочник должен также лежать в `meta.event_type_map` (см. §6), чтобы не зависеть от hard-coded id.

---

## 5) Правила для `flow_mag` и reuse `core_optical_flow`

### 5.1 Если `flow_mag` посчитан внутри `cut_detection`

`flow_mag` — mean magnitude (в пикселях или нормированная, но **одна** семантика на весь run).
В `meta.flow_mag_units` обязательно указать единицы.

### 5.2 Если используем `core_optical_flow`

Если `core_optical_flow.frame_indices` **точно совпадает** с `cut_detection.frame_indices`, то:

- берём `motion_norm_per_sec_mean (N,)`
- переводим в per-pair:
  - `flow_mag_pair[i] = motion_norm_per_sec_mean[i+1]` (для i=0..N-2)
  - (первый элемент провайдера соответствует “первому кадру” и равен 0; поэтому shift оправдан)

В `meta.flow_source` указать `"core_optical_flow"` и `flow_artifact_path`.

---

## 6) Meta (обязательные поля)

NPZ должен содержать `meta` (object dict) по общему контракту `docs/contracts/ARTIFACTS_AND_SCHEMAS.md`:

Обязательное дополнение для model-facing schema:
- `schema_version = "cut_detection_model_facing_npz_v1"`
- `producer = "cut_detection"`
- `status = "ok|empty|error"` (для этого модуля empty обычно недопустим)
- `cut_detection_config`:
  - `ssim_max_side`, `flow_max_side`
  - `prefer_core_optical_flow`, `require_core_optical_flow`
  - `use_deep_features` (bool)
  - `use_adaptive_thresholds` (bool)
  - `temporal_smoothing` (bool) + параметры smoothing (если есть)
- `flow_source`:
  - `"internal_farneback" | "core_optical_flow"`
  - `flow_mag_units` (например `"norm_per_sec_mean"` или `"px_mean"`)
  - `flow_artifact_path` (если core reuse)
- `event_type_map` (dict: int->str) для `event_type_id`

---

## 7) Empty/error semantics

MVP правило:
- `frame_indices` отсутствует/пустой → **error**
- `len(frame_indices) < 2` → **error**
- `union_timestamps_sec` отсутствует/невалиден → **error**

`empty` допускается только если policy явно разрешит “нет данных”, но для baseline рекомендовано fail-fast.


