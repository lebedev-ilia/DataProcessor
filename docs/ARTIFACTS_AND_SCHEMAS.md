# Артефакты, NPZ-схемы и хранилище (полуфинал)

## 1) Источник истины

- **NPZ — source-of-truth** для ML/кэша/повторных прогонов.
- **JSON — presentation layer** (рендер для backend/frontend из NPZ).

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
- `run`: `platform_id`, `video_id`, `run_id`, `config_hash`, `sampling_policy_version`, `created_at`
- `components[]`: для каждого компонента:
  - `name`
  - `status`: ok/empty/error
  - `artifacts[]`: пути + размеры/хэши
  - `producer_version`, `schema_version`
  - timings (опционально)
- `render` (опционально): LLM/presentation metadata (см. `LLM_RENDERING.md`)

## 4) Обязательная meta-секция в каждом NPZ

Полуфинальный минимум:
- `producer`, `producer_version`, `schema_version`
- `created_at`
- `platform_id`, `video_id`, `run_id`
- `config_hash`, `sampling_policy_version`
- `status` = ok/empty/error
- `empty_reason` (если empty)

Рекомендации для воспроизводимости:
- `model_name`, `model_version` (если использовался Triton/ML модель)
- `git_commit` (если доступно)

## 5) Missing/nullable данные (единый стандарт)

В NPZ “None” обычно кодируется так:
- числовые массивы → `NaN`
- булевые маски присутствия → `*_present` / `has_*`
- причина пустоты → `empty_reason` в `meta` (и/или `faces_empty_reason` и т.п.)

Запрещено:
- “заглушки нулями”, если это семантически означает реальное значение.

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


