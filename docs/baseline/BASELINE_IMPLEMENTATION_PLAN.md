# План доведения TrendFlow/DataProcessor до baseline (по новым контрактам)

Этот документ — **единый подробный план работ**, по которому другая нейросеть/разработчик сможет привести проект к первому стабильному baseline (и подготовить фундамент для v2 multimodal transformer), **строго соблюдая** все новые контракты и правила.

Источник правил:
- `docs/contracts/CONTRACTS_OVERVIEW.md`
- `docs/contracts/ARTIFACTS_AND_SCHEMAS.md`
- `docs/contracts/SEGMENTER_CONTRACT.md`
- `docs/contracts/ORCHESTRATION_AND_CACHING.md`
- `docs/baseline/ML_TARGETS_AND_TRAINING.md`
- `docs/contracts/LLM_RENDERING.md`
- `docs/contracts/PRIVACY_AND_RETENTION.md`
- `VisualProcessor/docs/MODULE_STANDARDS.md`

---

## 0) Определение “baseline готов”

Baseline считается готовым, если выполняется весь путь:

1) **Один прогон** DataProcessor по 1 видео (raw video + meta + comments) создаёт:
   - `result_store/<platform_id>/<video_id>/<run_id>/manifest.json`
   - NPZ артефакты Tier‑0 компонентов (Visual/Audio/Temporal) в per-run структуре
2) Из набора прогонов строится training dataset:
   - таблица (parquet/csv) с **агрегатами** (tabular aggregate set)
   - таргеты multi-target + multi-horizon (14/21 обязательно, 7 masked)
3) Обучается baseline модель (CatBoost/LightGBM), есть отчёт метрик:
   - overall + разрез по `video_age_hours_at_snapshot1` buckets
4) Есть “inference script”: загрузка модели + чтение фичей из артефактов → прогноз JSON (presentation layer).

---

## 1) Неподвижные контракты (НЕ нарушать)

Перед любыми изменениями убедиться, что всё новое кодирование данных им соответствует:

- **NPZ — source of truth**, JSON — presentation layer.
- **No-fallback**: нет `frame_indices`/нет dependency → `raise`.
- **Segmenter отвечает за sampling** и пишет `frame_indices` для каждого компонента.
- **frames_dir содержит только union sampled кадры**, а `frame_indices` в metadata — это индексы в union.
- **RGB контракт**: `FrameManager.get()` → RGB (`color_space="RGB"`).
- **Valid empty outputs**: NaN + masks + `empty_reason`.
- **per-run storage** + `manifest.json`.
- **Targets**: deltas + `log1p`, multi-horizon 14/21 (+ 7 optional head).

Если где-то противоречие — сперва обновить документацию/контракт, потом код.

---

## 2) Milestones (коротко)

- **M0**: Стабилизировать “формат входа” для VisualProcessor (frames_dir/metadata) под union sampling.
- **M1**: Стабилизировать storage/manifest/NPZ meta и валидатор схем.
- **M2**: Привести Tier‑0 Visual модули к `BaseModule + NPZ` и новым правилам.
- **M3**: Привести Tier‑0 Audio extractors и их артефакты к тем же правилам (per-run, meta, masks).
- **M4**: Построить датасет: (артефакты → табличные фичи + таргеты).
- **M5**: Обучить baseline (CatBoost/LightGBM) + отчёты + reproducibility.
- **M6**: Inference pipeline: прогноз + JSON рендер (без LLM или с LLM как текст-only).

---

## 3) M0 — Segmenter + frames_dir (union sampling) как единый вход VisualProcessor

### 3.1 Цель
Перестроить подготовку кадров так, чтобы:
- frames_dir был **маленьким** (сотни/тысячи кадров), а не десятки тысяч
- `FrameManager.get(i)` работал на union индексах
- metadata содержал mapping к исходнику

### 3.2 Что именно должно появиться в `frames_dir/metadata.json`
Минимальный контракт (полуфинал):

- `platform_id`, `video_id`, `run_id` (можно не в frames_dir, но удобно иметь)
- `analysis_fps`, `analysis_width`, `analysis_height`
- `color_space="RGB"`
- `total_frames = len(union_frame_indices)`
- `batch_size` (или `chunk_size`) и `batches[]`
- `union_frame_indices_source` **и/или** `union_timestamps_sec`
- per-component секции:
  - `core_clip.frame_indices`
  - `cut_detection.frame_indices`
  - `shot_quality.frame_indices`
  - … (для каждого компонента, который будет запускаться)
  - **все эти indices — в union domain**

### 3.3 Изменения в коде (ориентиры)

- `Segmenter/segmenter.py`
  - Вынести “полное извлечение всех кадров” в legacy/optional режим.
  - Реализовать режим: “compute indices → build union → extract only union frames”.
  - Убедиться, что сохраняемые кадры действительно **RGB** (уже добавлено `color_space="RGB"`).
  - Обновить docstring/README Segmenter.

- `VisualProcessor/utils/frame_manager.py`
  - Должен уметь читать batch формат, который пишет Segmenter.
  - Должен использовать `batch_size` или `chunk_size` (legacy).
  - Должен быть готов к будущему raw-memmap формату.

### 3.4 Acceptance criteria
- На видео 20 минут (72k кадров @60fps) размер frames_dir **не растёт линейно** от 72k (растёт от union N).
- Любой модуль, который получает `frame_indices` из metadata, корректно читает кадры.
- В `metadata.json` явно видно `color_space="RGB"`.

---

## 4) M1 — Артефакты/manifest/schema validator (системный фундамент)

### 4.1 Manifest как обязательный артефакт
Реализовать/добавить генерацию `manifest.json` на каждый run:
- в DataProcessor orchestrator (предпочтительно)
- или в VisualProcessor main, если orchestrator ещё не готов

Manifest должен отражать:
- run identity: `(platform_id, video_id, run_id, config_hash, sampling_policy_version)`
- список компонентов + статус (ok/empty/error) + пути артефактов + versions
- ошибки (если error) и empty_reason (если empty)

### 4.2 NPZ meta contract
Для каждого NPZ обеспечить обязательные meta-поля (см. `docs/contracts/ARTIFACTS_AND_SCHEMAS.md`).

### 4.3 Schema registry и валидатор
Сделать минимальный валидатор:
- проверка наличия meta полей
- проверка `frame_indices` (sorted, unique, dtype int)
- проверка shapes/dtypes ключевых массивов

Где хранить schema:
- human: `modules/<name>/SCHEMA.md` (можно добавить позже)
- machine: `VisualProcessor/schemas/<component>.json`

### 4.4 Acceptance criteria
- Любой артефакт можно верифицировать валидатором.
- Изменение схемы требует bump `schema_version` + обновление docs/schema json.

---

## 5) M2 — Tier‑0 Visual: привести модули к BaseModule + NPZ + контрактам

### 5.1 Tier‑0 Visual компоненты (рекомендуемый старт)
Минимум для baseline + будущего v2:

- Core:
  - `core_clip`
  - `core_face_landmarks` (если нужен)
  - `core_object_detections` (опционально, если даёт сильный сигнал)
  - `core_depth_midas` (опционально, если есть бюджет)
- Modules:
  - `cut_detection` (shots → token=shot)
  - `shot_quality` (агрегаты + per-shot quality)
  - `scene_classification` (semantics)
  - `video_pacing` (pacing curves/energy)
  - `uniqueness` (repetition proxy)
  - `story_structure` (если даёт устойчивые сигналы; иначе Tier‑1)

Пока не включать тяжёлые/нестабильные модули в required, пока не будут стабильны.

### 5.2 Обязательные требования к каждому модулю
- наследуется от `BaseModule`
- `process()`:
  - читает `frame_indices` только из metadata
  - не генерирует семплинг сам
  - `raise`, если нет необходимых core artifacts
- output NPZ:
  - meta + versions + status
  - численные массивы с NaN для missing
  - masks `*_present`

### 5.3 Конкретные задачи по оставшимся модулям (из списка)
Привести к стандарту и закрыть TODO:
- `VisualProcessor/modules/similarity_metrics`
- `VisualProcessor/modules/story_structure`
- `VisualProcessor/modules/text_scoring`
- `VisualProcessor/modules/uniqueness`
- `VisualProcessor/modules/video_pacing`

Для каждого:
1) Найти TODO и убрать их реализацией/рефактором.
2) Перевести на `BaseModule.run()` CLI.
3) Перевести output на NPZ (`save_results()`).
4) Обновить/создать:
   - `README.md` (входы/выходы/зависимости/рекомендации по sampling)
   - `FEATURES_DESCRIPTION.md`
5) Добавить schema entry (json) + (опционально) `SCHEMA.md`.

### 5.4 Acceptance criteria
- Tier‑0 Visual пайплайн отрабатывает на видео без падений при “нормальных пустых случаях” (no faces, no text).
- Любой missing dependency приводит к явному `raise` (и это видно в manifest/status).

---

## 6) M3 — Tier‑0 Audio + Temporal features

### 6.1 Audio Tier‑0 (уже выбран)
- `clap_extractor` (semantic audio)
- `loudness_extractor`
- `tempo_extractor`

Требования:
- per-run storage
- NPZ meta contract + schema
- missing handling (NaN+masks)

### 6.2 Temporal (из мета снапшотов)
Добавить в baseline training table обязательные фичи:
- `video_age_hours_at_snapshot1`
- `duration_sec`, `fps` (analysis fps), language/category
- channel stats (после enrichment `channel_id`)

### 6.3 Acceptance criteria
- Стабильно строится tabular таблица из Visual+Audio+Temporal агрегатов.

---

## 7) M4 — Dataset builder (артефакты → training table + targets)

### 7.1 Enrichment (обязательный шаг)
Так как сейчас есть только `channelTitle`/`authorName`, нужно:
- написать enrichment по `video_id` → `channel_id` (и по возможности comment ids)
- сохранить в датасете

### 7.2 Targets
Считать multi-horizon deltas:
- 14d/21d всегда (полные)
- 7d — optional с mask
И применять `log1p`.

### 7.3 Таблица фичей
Из артефактов собираем:
- обязательный tabular aggregate set (Visual+Audio+Temporal)
- masks как отдельные фичи
- фиксируем версии (dataprocessor/sampling/schema) рядом с dataset

Практический baseline v0 (уже можно запускать):
- `DatasetBuilder/build_training_table.py` — собирает **feature-table** из `result_store/.../manifest.json` + NPZ (пока без таргетов).

### 7.4 Acceptance criteria
- Генерация датасета детерминирована (reproducible).
- Нет leakage: фичи только snapshot1/артефакты; таргеты только future snapshots.

---

## 8) M5 — Обучение baseline (CatBoost/LightGBM)

### 8.1 Минимальный тренировочный пайплайн
- загрузка training table
- split: time + channel-group
- обучение multi-target/multi-horizon (можно multi-head или отдельные модели per target/horizon на старте)
- метрики: MAE/RMSE на log1p + Spearman
- отчёт по age-buckets

### 8.2 Reproducibility
Фиксируем:
- seed
- commit hash (если доступно)
- версии схем/семплинга/конфига

### 8.3 Acceptance criteria
- Есть baseline модель + сохранённый артефакт модели + конфиг.
- Есть отчёт качества overall + buckets.

---

## 9) M6 — Inference (модель → JSON)

### 9.1 Слой данных
- загрузка NPZ артефактов нужных компонентов
- построение feature vector как в training
- прогноз

### 9.2 Presentation layer
- формируем JSON для backend/frontend детерминированно
- (опционально) LLM текст-only поверх render-context (см. `docs/LLM_RENDERING.md`)

### 9.3 Acceptance criteria
- Для одного run можно получить прогноз и итоговый JSON.

---

## 10) Контрольные чекпоинты качества (must-have)

Перед тем как считать baseline “готовым”, проверить:

- **Contract checks**
  - нет fallback семплинга в модулях/корах
  - RGB контракт выполняется (frames_dir metadata + поведение FrameManager)
  - пустые кейсы → NaN+masks+empty_reason
- **Storage checks**
  - per-run структура + manifest
  - артефакты версионированы
- **Dataset checks**
  - targets корректные (delta, log1p)
  - нет leakage
  - корректная обработка missing heads (7d)

---

## 11) Что делать после baseline (коротко, чтобы не потерять v2)

После baseline:
- стандартизировать per-shot sequences (`E_shot`, `A_shot`, pacing curves) как обязательные outputs v2
- реализовать v2 multimodal transformer с token=shot, max_len=256
- baseline/v1 оставлять как sanity-check и fallback


