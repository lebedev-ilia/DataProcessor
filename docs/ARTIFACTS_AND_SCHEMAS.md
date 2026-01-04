# Артефакты, NPZ-схемы и хранилище (полуфинал)

## 1) Источник истины

- **NPZ — source-of-truth** для ML/кэша/повторных прогонов.
- **JSON — presentation layer** (рендер для backend/frontend из NPZ).

Полуфинальное уточнение (Round 1):
- В `result_store` **запрещены** произвольные `*.json` артефакты (кроме `manifest.json`).
  - Legacy JSON допускается только как временный debug в `_tmp_*` и не считается source-of-truth.

## 2) Структура result_store (per-run)

Полуфинальный стандарт:

- `result_store/<platform_id>/<video_id>/<run_id>/<component_name>/...`
- `result_store/<platform_id>/<video_id>/<run_id>/manifest.json`

Зачем:
- параллельные прогоны не конфликтуют
- проще дебаг/аудит

## 3) manifest.json

`manifest.json` — обязательный “source of truth” по конкретному `run_id` (БД — опционально как ускоритель).

Рекомендуемая структура:
- `run`: `platform_id`, `video_id`, `run_id`, `config_hash`, `sampling_policy_version`, `dataprocessor_version`, `created_at`, `updated_at`
- `components[]`: для каждого компонента:
  - `name`
  - `kind` (например: `core`/`module`/`audio`/`text`)
  - `status`: ok/empty/error
  - `artifacts[]`: пути + размеры/хэши
  - `producer_version`, `schema_version`
  - `started_at`, `finished_at`, `duration_ms`
  - `device_used` (если применимо)
  - `error` (если status=error)
  - `notes` (опционально, для дополнительной информации/валидации)
- `render` (опционально): LLM/presentation metadata (см. `LLM_RENDERING.md`)

**Полуфинальное правило observability (Round 2)**: минимальный набор полей per component в manifest включает: `status`, `started_at`, `finished_at`, `duration_ms`, `device_used`, `error`, `notes`, `schema_version`, `producer_version`.

Важно про запись manifest:
- `manifest.json` может апдейтиться несколькими стадиями пайплайна (Audio → Text → Visual).
- Запись должна быть **атомарной** (tmp → replace).
- При параллельном выполнении компонент (внутривидео) manifest апдейтится **из одного потока/процесса** (оркестратор собирает результаты и делает upsert последовательно).

## 4) Обязательная meta-секция в каждом NPZ

Полуфинальный минимум:
- `producer`, `producer_version`, `schema_version`
- `created_at`
- `platform_id`, `video_id`, `run_id`
- `config_hash`, `sampling_policy_version`
- `dataprocessor_version` (обязательно; в baseline может быть `"unknown"`, в проде — версия релиза)
- `status` = ok/empty/error
- `empty_reason` (если empty)

**Полуфинальное правило (Round 2)**: `dataprocessor_version` обязателен в `manifest.json` (секция `run`) и в `meta` каждого NPZ.

Рекомендуемое поле для прод-воспроизводимости:
- `dataprocessor_version` (строка). В baseline v0 может быть `unknown`, в проде обязателен.

### 4.1 Принятые schema_version (baseline v0, текущий набор)

Core providers (VisualProcessor):
- `core_clip`: `core_clip_npz_v1`
- `core_depth_midas`: `core_depth_midas_npz_v1`
- `core_object_detections`: `core_object_detections_npz_v1`
- `core_optical_flow`: `core_optical_flow_npz_v1`
- `core_face_landmarks`: `core_face_landmarks_npz_v1`

Audio:
- `clap_extractor` / `tempo_extractor` / `loudness_extractor`: `audio_npz_v1`

Text:
- `text_processor`: `text_npz_v1`

Примечание (текущее состояние валидатора):
- Runtime валидатор должен быть **строгим по умолчанию** (baseline contract): проверять полный набор meta полей.
  Если нужно временно ослабить проверки для legacy-артефактов, используйте параметр `required_meta_keys` в `validate_npz(...)`.

Рекомендации для воспроизводимости:
- `models_used[]` (если компонент вызывал ML-модели): `model_name`, `model_version`, `weights_digest`, `runtime`, `engine`, `precision`, `device`
- `seed` (если применимо)
- `runtime_env` (опционально: версии CUDA/cuDNN/драйвера)
- `git_commit` (если доступно)

Каноническая спецификация `model_signature/models_used` находится в `docs/MODEL_SYSTEM_RULES.md`.

## 5) Missing/nullable данные (единый стандарт)

В NPZ “None” обычно кодируется так:
- числовые массивы → `NaN`
- булевые маски присутствия → `*_present` / `has_*`
- причина пустоты → `empty_reason` в `meta` (и/или `faces_empty_reason` и т.п.)

Запрещено:
- “заглушки нулями”, если это семантически означает реальное значение.

### 5.1 Каноничный словарь `empty_reason` (Round 2)

Полуфинальный стандартный набор значений `empty_reason`:
- `audio_missing_or_extract_failed`
- `no_faces_in_video`
- `no_text_available`
- `ocr_disabled_by_policy`
- `comments_missing_or_disabled`
- `video_too_short`
- `video_too_long`
- `dependency_missing`

Компоненты должны использовать эти значения там, где применимо; допускаются дополнительные специфичные причины (например, `transcript_missing_for_source_whisper`), но предпочтительно выбирать из стандартного набора.

## 6) Схемы: human + machine

Полуфинал:
- `SCHEMA.md` рядом с модулем (human-friendly)
- машинная схема в `VisualProcessor/schemas/*.json` (единый реестр)

## 7) Валидатор схем

Запуск:
- **runtime**: ловим битые/неполные артефакты сразу в пайплайне
- **CI**: не даём изменениям схемы незаметно ломать совместимость

Минимальные проверки:
- ключи/dtype/shape
- `frame_indices` отсортированы, уникальны
- согласованность `meta` (обязательные поля)

## 8) Audio Tier‑0 (baseline) — per-run NPZ артефакты

На baseline этапе аудио-экстракторы пишут NPZ в тот же `result_store/<platform>/<video>/<run>/...`:

- `clap_extractor/*.npz`
- `tempo_extractor/*.npz`
- `loudness_extractor/*.npz`

Общий формат (гибкий, “tabular-friendly”):
- `feature_names`: object array строк
- `feature_values`: float32 array тех же размеров
- дополнительные ключи по компоненту:
  - `clap_extractor`: `embedding` (float32[D]), `embedding_present` (bool)
  - `tempo_extractor`: `tempo_estimates` (float32[T]), `windowed_times_sec`, `windowed_bpm`, `warnings`
  - `loudness_extractor`: `lufs_present` (bool)
- `meta`: dict (object array) по контракту (producer/created_at/status/…)

Важно:
- у аудио артефактов **может не быть** `frame_indices` (валидатор проверяет их только если ключ присутствует).

## 8.1 Большие массивы и под-артефакты (`.npy`)

Разрешено (полуфинал):
- Для “очень больших” матриц/тензоров (например, эмбеддинги/спектрограммы) допускается хранение как отдельного бинарного файла `.npy`.
- В этом случае:
  - путь к `.npy` должен быть указан либо в NPZ (`payload`/поля), либо в `manifest.json` в `artifacts[]`,
  - `.npy` считается **sub-artifact** конкретного компонента и управляется теми же retention правилами, что и NPZ.
  - source-of-truth для ML остаётся NPZ (NPZ может хранить агрегаты + ссылку на raw-matrix).

## 9) TextProcessor (baseline) — per-run NPZ артефакты

TextProcessor пишет один “агрегированный” NPZ-артефакт в per-run storage:

- `text_processor/*.npz`

Минимальный формат (tabular-friendly):
- `feature_names`: object array строк
- `feature_values`: float32 array
- `payload`: object(dict) (debug / trace)
- `meta`: dict (object array) по общему контракту

TextProcessor может работать в:
- CPU-only режиме (baseline-safe)
- режиме с эмбеддерами (GPU-heavy). Полуфинально принято: **эмбеддеры включаем в Tier‑0** (training schema).


