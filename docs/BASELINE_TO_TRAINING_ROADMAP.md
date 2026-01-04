# План доведения TrendFlow/DataProcessor до обучения baseline (CatBoost/LightGBM)

Этот документ — **единый подробный план**, по которому мы доводим DataProcessor до состояния:
- массовые прогоны видео → валидные артефакты (NPZ + manifest),
- построение датасета (features + targets),
- обучение baseline (CatBoost/LightGBM) + reproducibility,
строго по контрактам `DataProcessor/docs/*`.

## Документы правды (все ссылки внутри `docs/`)

- `DATAPROCESSOR_AUDIT.md` — текущий “чеклист истины” (PASS/FAIL) и root-cause.
- `BASELINE_IMPLEMENTATION_PLAN.md` — milestone’ы M0..M6 и acceptance criteria.
- `REMAINING_BASELINE_TASKS.md` — оставшиеся задачи (особенно M4/M5/M6 и batch processing).
- `ML_TARGETS_AND_TRAINING.md` — таргеты/сплиты/leakage/repro.
- `CONTRACTS_OVERVIEW.md` — короткий список главных правил.
- `ARTIFACTS_AND_SCHEMAS.md` — контракт `manifest.json`, NPZ meta, schema/versioning.
- `ORCHESTRATION_AND_CACHING.md` — required/optional, idempotency/cache TTL.
- `SEGMENTER_CONTRACT.md` — contract для `frames_dir/metadata.json` и sampling.
- `MODEL_SYSTEM_RULES.md` — model_signature/models_used и error_code, OOM policy.
- `ERROR_HANDLING_AND_EDGE_CASES.md` — retry, edge cases, timeouts.
- `PRIVACY_AND_RETENTION.md` — raw/retention/delete request.
- `PRODUCT_CONTRACT.md` — входные ограничения: 5s..20min, max 1080p, preprocessing.
- `DynamicBatching_Q_A.md` — зафиксированные правила DynamicBatching + state-files/state-managers (входит в baseline).
- `component_graph.yaml` — MVP source-of-truth DAG (priority=dependency-ordering) (входит в baseline).
- `BASELINE_PRODUCTION_EXECUTION_PLAN.md` — **порядок реализации (PR‑пакеты)** для production‑baseline → dataset → training.

---

## 0) Определение “готово до baseline training” (Definition of Done)

Считаем, что мы “дошли до обучения baseline”, когда выполняются все пункты:

1) **Сквозной прогон DataProcessor** по одному видео (YouTube URL или локальный файл) создаёт:
   - per-run структуру `result_store/<platform_id>/<video_id>/<run_id>/...`
   - `manifest.json` (полный по полям и семантике статусов)
   - NPZ артефакты Tier‑0 (Visual + Audio + Temporal features в таблице)
   - валидные empty-кейсы (NaN+masks+empty_reason), без “тихих” except→empty

2) **Run identity** одинаков везде:
   - `frames_dir/metadata.json`
   - `manifest.json.run`
   - `meta` каждого NPZ
   Включая `dataprocessor_version` (baseline допускает `"unknown"`).

3) **Dataset builder** строит training table (features) + targets:
   - multi-target (минимум `views`, `likes`)
   - multi-horizon: `14d/21d` required, `7d` optional + masks
   - таргеты = delta, затем `log1p(delta)`
   - temporal feature: `video_age_hours_at_snapshot1` обязательна
   - нет leakage (features только snapshot1, targets только future snapshots)

4) **Training pipeline** обучает baseline (CatBoost/LightGBM), сохраняет:
   - модель + список фич
   - конфиг/seed/версии (reproducibility manifest для обучения)
   - отчёт метрик overall + buckets по `video_age_hours_at_snapshot1`

5) **DynamicBatching + state** (обязательная часть baseline, production-grade):
   - Реализованы state-files + state-managers (по правилам `DynamicBatching_Q_A.md`) на уровнях:
     - Level 1: `state_level1_global.json`
     - Level 2: `run_state.json`
     - Level 3: `state_visual.json`, `state_text.json`, `state_audio.json`, `state_segmenter.json`
     - Level 4 **не используется** (состояние компонентов — секции внутри Level 3).
   - **Durability**: state-manager имеет устойчивый журнал событий (например `state_events.jsonl`) + checkpoint state-file.
   - **Fail-fast**: missing dependency (любая) → `error` и стоп run.
   - **DAG**: есть декларативный граф `component_graph.yaml` (baseline/v1/v2), по которому строится dependency-ordering (“priority”).
   - **Scheduler feedback**: DataProcessor фиксирует и экспортирует события/метрики, нужные для resource checklist (время/память/OOM и т.п.).

6) **Celery/Queue-based execution (обязательная часть baseline)**:
   - Запуск DataProcessor происходит **через очередь** (Celery): backend/батч‑раннер кладёт job, worker исполняет.
   - Есть retry политика (transient vs permanent) и она отражается в state/manifest (`error_code`, retry_count).
   - Есть health endpoints для worker (`/health`, `/health/live`) с проверками: очередь доступна, storage доступен, Triton доступен (если обязателен).

7) **Triton integration (обязательная часть baseline)**:
   - Triton развернут как отдельный сервис и используется для **всех baseline-моделей, которые мы решили вынести в serving** (no-fallback: Triton обязателен).
   - Есть resolved mapping per-run (source-of-truth: профили анализа/БД, см. `MODEL_SYSTEM_RULES.md`/`PRODUCTION_ARCHITECTURE.md`), а в NPZ meta фиксируется `models_used[]/model_signature`.
   - Есть health/ready checks и error taxonomy (`triton_unavailable`, `model_not_found`, `model_load_failed`, `insufficient_gpu_memory`).

8) **External storage + versioning (обязательная часть baseline)**:
   - Артефакты (`result_store`, `manifest.json`, state-files) хранятся во **внешнем хранилище** (MVP: MinIO S3-compatible по `PRODUCTION_ARCHITECTURE.md`).
   - Есть storage adapter (FS ↔ S3/MinIO), чтобы worker’ы были взаимозаменяемы.
   - Версионирование: `dataprocessor_version="unknown"` допустимо для baseline, но schema/producer версии и model_signature обязаны быть.

9) **Оптимизации моделей (обязательная часть baseline)**:
   - Для baseline-моделей определён pipeline оптимизаций (где применимо): ONNX export / TensorRT / quantization.
   - В meta фиксируется `engine/precision/device`, и это влияет на `model_signature` (как в `MODEL_SYSTEM_RULES.md`).

10) **Аудит всех модулей (обязательная часть baseline)**:
   - Пройден аудит модулей всех процессоров (Visual/Audio/Text/Segmenter) на соответствие контрактам (meta/manifest/schema/empty/error/no-fallback/privacy).
   - Для каждого модуля зафиксирована “единица обработки” и требования к batching/resource checklist (см. `DynamicBatching_Q_A.md`).

---

## 1) Неподвижные инварианты (НЕ нарушать)

Основание: `CONTRACTS_OVERVIEW.md`, `ARTIFACTS_AND_SCHEMAS.md`, `SEGMENTER_CONTRACT.md`.

- **NPZ — source-of-truth** (JSON — только presentation layer; исключение: `manifest.json` и будущие state-files).
- **No-fallback**: нет `frame_indices`/dependency → `raise` (а не “пустой результат”).
- **Segmenter = единственный владелец sampling**: `frame_indices` только из `frames_dir/metadata.json`.
- **frames_dir хранит union sampled кадры**, а `frame_indices` работают в union domain.
- **RGB контракт**: кадры в `frames_dir` и `FrameManager.get()` должны быть RGB.
- **Valid empty**: NaN+masks+`empty_reason` (не падение, не массивы нулей).
- **per-run storage**: структура хранения фиксирована.
- **Reproducibility**: фиксируем версии producer/schema и run identity поля.

Если что-то противоречит — сначала обновляем docs/контракт, потом код.

---

## 2) Стратегия работ: сначала закрываем “блокеры обучения”, потом масштабируем

План разбит на этапы. Каждый этап имеет:
- **Цель**
- **Работы**
- **Артефакты/файлы**
- **DoD/Acceptance criteria**
- **Связь с аудитом** (что закрываем из FAIL/GAP)

---

## 3) Этап A — Привести текущие docs-статусы к реальности (1–2 дня)

### Цель
Убрать противоречия вида “✅ M0–M3” в одном документе при наличии FAIL в `DATAPROCESSOR_AUDIT.md`.

### Работы
- Пройтись по `REMAINING_BASELINE_TASKS.md` и:
  - либо привести “✅” в соответствие с audit (поставить ❌/PARTIAL там, где есть FAIL по контрактам),
  - либо явно отделить “функционально работает” от “контрактно соответствует” (например две колонки статуса).

### DoD
- Нет пунктов “готово”, если audit фиксирует FAIL по обязательным контрактам (run identity / meta / manifest).

---

## 4) Этап B — Закрыть критичные FAIL по контрактам (блокер датасета) (3–10 дней)

### B1) Run identity: `dataprocessor_version` + propagation

**Основание**: `CONTRACTS_OVERVIEW.md`, `ARTIFACTS_AND_SCHEMAS.md`, audit разделы 1.1/7/8.

**Что сделать**:
- Root orchestrator (`main.py`) формирует `dataprocessor_version` и прокидывает:
  - в Segmenter → `frames_dir/metadata.json`
  - в Visual/Audio/Text → `RunManifest.run` и NPZ `meta`

**DoD**:
- В одном run `dataprocessor_version` присутствует и одинаков в `metadata.json`, `manifest.json`, NPZ meta (как минимум `"unknown"`).

### B2) Segmenter contract: `analysis_*` + per-component budgets

**Основание**: `SEGMENTER_CONTRACT.md`, `BASELINE_IMPLEMENTATION_PLAN.md` M0, audit 3.x.

**Что сделать**:
- В `frames_dir/metadata.json` должны быть `analysis_fps/analysis_width/analysis_height`.
- Per-component `frame_indices` должны реально отражать бюджеты (а не одинаковые для всех).
- product constraints enforcement (5s..20min, downscale>1080p) — минимум: validation/preprocessing hooks.

**DoD**:
- `analysis_*` есть в metadata.
- Есть доказуемая разница per-component `frame_indices` на одном run.

### B3) Manifest schema completeness: `device_used`, `error_code`, warnings + error semantics

**Основание**: `ARTIFACTS_AND_SCHEMAS.md`, `MODEL_SYSTEM_RULES.md`, audit 7.x.

**Что сделать**:
- Расширить структуру `manifest.json.components[]` по полям:
  - `device_used`
  - `error_code` (обязателен при `status="error"`)
  - `warnings` (опционально, но полезно)
- Привести writers (Visual/Audio/Text) к единому способу заполнения.

**DoD**:
- В `manifest.json` присутствуют поля, и они не пустые там, где применимо.

### B4) NPZ meta completeness: `dataprocessor_version` + `models_used[]/model_signature` + `engine/precision/device`

**Основание**: `ARTIFACTS_AND_SCHEMAS.md`, `MODEL_SYSTEM_RULES.md`, audit 8.x.

**Что сделать**:
- Для core providers и model-использующих компонентов фиксировать `models_used[]` и `model_signature`.
- Для всех NPZ фиксировать `dataprocessor_version`.
- Для model-использующих: `engine/precision/device` обязаны быть.

**DoD**:
- На нескольких NPZ из одного run meta содержит `dataprocessor_version`, а model components содержат `models_used[]/model_signature`.

### B5) Required vs optional (fail-fast policy)

**Основание**: `ORCHESTRATION_AND_CACHING.md` (required по умолчанию), audit 2.2/2.5/required-vs-optional.

**Что сделать**:
- Формализовать “профиль анализа” на уровне orchestrator:
  - какие компоненты enabled
  - какие required vs optional
- По умолчанию: enabled = required, optional только при явной пометке.

**DoD**:
- При падении required компонента run = error.
- При падении optional компонента run продолжает, в manifest = component_status=error + warning.

### B6) Privacy (Text payload)

**Основание**: `PRIVACY_AND_RETENTION.md`, audit 6.2/8.3.

**Что сделать**:
- Запретить сохранение raw текста в NPZ по умолчанию (TextProcessor).
- Разрешить raw только под явной policy/flag, с retention caps.

**DoD**:
- Тест: run TextProcessor без специальных флагов не пишет raw payload.

---

## 5) Этап C — Production foundations для baseline: Celery + Triton + Storage + Versioning + DynamicBatching + Full Module Audit (10–40 дней)

Основание: `PRODUCTION_ARCHITECTURE.md`, `GLOBAL.md`, `MODEL_SYSTEM_RULES.md`, `ORCHESTRATION_AND_CACHING.md`, `DynamicBatching_Q_A.md`, audit 10.1/10.8/10.9.

### C0) Принятые правила (напоминание)
- state-files многоуровневые (Level 1/2/3), **Level 4 не делаем**
- все read/write state только через state-manager уровня
- очередь обновлений state-manager должна быть **устойчивой** (journal + checkpoint)
- missing dependency → `error` и стоп run
- DAG source-of-truth в `component_graph.yaml` (baseline/v1/v2)

### C1) External storage + storage adapter (MinIO/FS)

**Что сделать**:
- Выбрать MVP backend внешнего хранилища: **MinIO (S3-compatible)**.
- Реализовать storage adapter:
  - FS backend (dev)
  - S3/MinIO backend (baseline/prod-like)
- Перевести записи/чтения:
  - `result_store` NPZ + `manifest.json`
  - state-files (`run_state.json`, `state_*.json`) + event journals
на storage adapter.

**DoD**:
- Один и тот же run можно исполнять/читать артефакты независимо от того, где запущен worker (не привязано к локальным путям).

### C2) Celery (queue-based execution) + health endpoints

**Что сделать**:
- Подключить Celery как основной способ запуска DataProcessor (worker читает из очереди).
- Описать payload job (минимум: run_id/video_id/platform_id/config_hash/sampling_policy_version/dataprocessor_version + video_source).
- Реализовать retry policy (transient vs permanent) с отражением в state/manifest (`retry_count`, `error_code`).
- Добавить `/health` и `/health/live` для worker:
  - очередь доступна
  - storage доступен
  - Triton доступен (если обязателен)

**DoD**:
- Можно поставить N job’ов в очередь и увидеть детерминированное исполнение и статусы (через state/manifest/DB).

### C3) Triton integration (model serving) + mapping/versions

**Что сделать**:
- Зафиксировать список baseline-моделей, которые должны идти через Triton.
- Реализовать клиент Triton + контракты ошибок (no-fallback).
- Реализовать resolved mapping per-run:
  - source-of-truth в профилях анализа (в будущем БД) по `MODEL_SYSTEM_RULES.md`
  - сохранить в manifest/state как “resolved mapping” (минимум: component→model_name/model_version/weights_digest/engine/precision/device_policy)
- В NPZ meta фиксировать `models_used[]/model_signature`.

**DoD**:
- GPU-heavy компоненты ходят через Triton и при проблемах с Triton корректно fail-fast с `error_code`.

### C4) Model optimization pipeline (ONNX/TensorRT/quantization)

**Что сделать**:
- Для baseline-моделей определить целевые runtime варианты:
  - `engine=torch` (fallback запрещён, но может быть dev-only)
  - `engine=onnx` / `engine=tensorrt` (prod baseline)
  - `precision=fp16` где безопасно
  - quantization (где применимо) как отдельный артефакт/вариант
- Обеспечить фиксацию `engine/precision/device` в meta и включение в `model_signature`.

**DoD**:
- Для хотя бы 1–2 ключевых GPU-heavy моделей есть оптимизированный вариант и он используется в baseline run.

### C5) Реализация state-manager + DAG runner + DynamicBatching (baseline)

**Что сделать**:
- Ввести state-manager’ы:
  - Level 2: на уровне root DataProcessor (агрегирует прогресс run)
  - Level 3: на уровне каждого Processor (Visual/Text/Audio/Segmenter)
- Ввести устойчивый event journal рядом со state-file (например `state_events.jsonl`), чтобы:
  - восстановить состояние после сбоя,
  - реализовать resume/retry от последнего стабильного checkpoint.
- Определить минимальный runtime schema state (run/processors/components/checkpoints), согласованный с `DynamicBatching_Q_A.md`.

**DoD**:
- Любой компонент/модуль может:
  - отправить update (status/error_code/artifact_paths/updated_at) в state-manager,
  - прочитать текущий status dependency через цепочку state-manager’ов (L3→L2→L3).
- При падении процесса state можно восстановить из `state_events.jsonl` и checkpoint.

### C6) Full module audit + единицы обработки (units) + локальные контракты

**Что сделать**:
- Пройтись по всем модулям/компонентам (Visual/Audio/Text/Segmenter) и для каждого зафиксировать:
  - единицу обработки (unit) и требования к batching,
  - dependency список (hard deps) и отражение в DAG,
  - семантику empty/error + error_code,
  - schema_version bump при изменениях.
- Заполнить `component_graph.yaml` минимум для baseline DAG (Tier‑0 required) и отразить v1/v2 в виде черновых DAG.

**DoD**:
- Есть список “локальных контрактов”/вопросов по модулям (как ты просил) и первые N критичных модулей baseline полностью стандартизированы.

---

## 6) Этап D — Dataset Builder (M4) как главный deliverable до обучения (3–14 дней)

Основание: `REMAINING_BASELINE_TASKS.md` (M4), `ML_TARGETS_AND_TRAINING.md`.

### C1) Feature table: deterministic build из manifest + NPZ

**Что сделать**:
- Убедиться, что `DatasetBuilder/build_training_table.py` читает только per-run структуру (через manifest).
- Добавить temporal features:
  - `video_age_hours_at_snapshot1` (обязательный)
  - `duration_sec`, `analysis_fps`, язык/категория (если есть)

**DoD**:
- На одном наборе run’ов повторный запуск builder’а даёт идентичный output (байтово или по хэшу).

### C2) Targets: multi-horizon deltas + masks

**Что сделать**:
- Реализовать `DatasetBuilder/add_targets.py`:
  - читает snapshots (t0, t+7d, t+14d, t+21d)
  - считает delta
  - делает `log1p`
  - создаёт masks для отсутствующих горизонтов (7d optional)

**DoD**:
- В итоговой таблице есть target columns и masks; нет leakage.

### C3) Enrichment: `channel_id` (для split)

**Что сделать**:
- `DatasetBuilder/enrichment.py`:
  - `video_id → channel_id` (YouTube API или существующий индекс)
  - (опционально) channel stats

**DoD**:
- Для каждого `video_id` в датасете есть `channel_id` (или валидный missing mask + объяснение).

### C4) Full dataset orchestrator

**Что сделать**:
- `DatasetBuilder/build_full_dataset.py`:
  - build features → add targets → enrichment → save parquet/csv + metadata

**DoD**:
- Есть “одна команда”, которая собирает финальный dataset.

---

## 7) Этап E — Training baseline (M5) (3–10 дней)

Основание: `REMAINING_BASELINE_TASKS.md` (M5), `ML_TARGETS_AND_TRAINING.md`.

### D1) Training script

**Что сделать**:
- `Training/train_baseline.py`:
  - загрузка dataset
  - split: time-split + channel-group split
  - обучение per target/horizon (минимум: отдельные модели)
  - метрики: RMSE/MAE на log1p + Spearman
  - отчёт по buckets `video_age_hours_at_snapshot1`

**DoD**:
- Артефакт модели + отчёт метрик сохранены в `Training/artifacts/<train_run_id>/`.

### D2) Reproducibility (training manifest)

**Что сделать**:
- `Training/config.yaml` или `Training/run_manifest.json`:
  - seed
  - commit hash (если есть)
  - версии схем/семплинга/конфига/датапроцессора
  - dataset fingerprint (hash)

**DoD**:
- Повторная тренировка с тем же dataset fingerprint выдаёт сопоставимые метрики (в пределах допустимого).

---

## 8) Этап F — Массовый прогон видео для датасета (batch processing) (2–14 дней)

Основание: `REMAINING_BASELINE_TASKS.md` (Batch Processing), `GLOBAL.md`, `PRODUCTION_ARCHITECTURE.md`.

### E1) Batch orchestrator (offline/MVP)

**Что сделать**:
- `BatchProcessor/process_batch.py` (или привести в порядок текущий runner):
  - читает список видео (CSV/JSON)
  - запускает `main.py` для каждого run
  - пишет `state.jsonl` (успех/ошибка/пусто) + время
  - поддерживает resume (не пересчитывает, если валидный manifest+NPZ уже есть)

**DoD**:
- Можно прогнать 100+ видео без ручного вмешательства.

### E2) Retry policy (transient vs permanent)

**Что сделать**:
- Согласовать с `ERROR_HANDLING_AND_EDGE_CASES.md`:
  - transient: network/timeout/OOM → retry 2–3
  - permanent: corrupted → fail-fast

**DoD**:
- Retry не приводит к silent corruption (всё отражено в manifest/error_code).

---

## 9) Этап G — Inference baseline (опционально до training, но желательно) (2–7 дней)

Основание: `REMAINING_BASELINE_TASKS.md` (M6).

**Что сделать**:
- `Inference/extract_features.py` (как в training)
- `Inference/predict.py`
- `Inference/render_json.py` (presentation layer)

**DoD**:
- Для одного run можно получить прогноз и JSON.

---

## 10) Приоритизация (что делаем первым)

Критические блокеры обучения baseline:
1) **Контракты** (Этап B): run identity, NPZ meta/manifest completeness, required/optional, privacy.
2) **DynamicBatching + state** (Этап C): state-managers/state-files + DAG runner + batching feedback.
3) **M4 dataset builder**: targets + temporal features + enrichment + full orchestrator.
4) **M5 training**: train script + reproducibility.

Масштабирование (batch processing) можно начинать параллельно, но оно должно опираться на валидные артефакты и idempotency.

---

## 11) Риски и “тонкие места”

- `REMAINING_BASELINE_TASKS.md` может быть оптимистичен: сверяемся с `DATAPROCESSOR_AUDIT.md` как источником факта.
- Без `models_used/model_signature` кэш/idempotency будет некорректным.
- Без `channel_id` split будет течь (leakage по каналам).
- Без строгого required/optional будет непредсказуемая полнота датасета.
- Privacy: нельзя допустить, чтобы TextProcessor писал raw в NPZ по умолчанию.
- State/batching: без устойчивого state-manager/event journal будет невозможно безопасно делать resume/retry и массовые прогоны.
- DAG: пока `component_graph.yaml` пустой, “priority=dependency-ordering” не реализуемо — нужно начать с baseline графа Tier‑0.

---

## 12) Блокирующие вопросы (ответы нужны ДО старта реализации)

1) **Snapshots для targets**: где физически лежат snapshot1/2/3/4 метаданные (пути/формат)? (`REMAINING_BASELINE_TASKS.md` упоминает `Interpret/main_ready/` — подтвердить.)

Ответ: лежат в моем HF датасете. Как только дойдем до этого этапа я предоставлю доступ. (заранее скажу что в HF лежит то же самое что лежало в main_ready)

2) **Targets**: baseline точно только `views+likes`? добавляем ли `comments` как target сразу?

Ответ: только `views+likes`

3) **Enrichment**: можно ли использовать YouTube API (квоты/ключи) для `video_id→channel_id`? или есть локальный индекс?

Ответ: можно использовать YouTube API. Ключи есть

4) **Tier‑0 список**: какие компоненты/модули обязаны быть required для baseline training schema (минимальный набор)?

Ответ: Описано в BASELINE_IMPLEMENTATION_PLAN

5) **dataprocessor_version**: baseline значение `"unknown"` достаточно, или хотим фиксировать git commit hash автоматически?

Ответ: unknown достаточно

6) **External storage для state**: какое именно внешнее хранилище считаем MVP (локальный disk на сервере vs MinIO)? (ты сказал “во внешнем хранилище”, нужно выбрать backend.)

Ответ: Выбери сам, либо локальный диск, либо HF

Decision (фиксируем для baseline): **MinIO (S3-compatible)**, т.к. baseline включает Celery/мульти‑worker и storage должен быть единым и совместимым с `PRODUCTION_ARCHITECTURE.md`.

7) **Policy “local-first, upload-on-error/stop”**: подтверждаешь, что при success run мы можем не выгружать state/артефакты во внешнее хранилище, а только при stop/error?

Ответ: Да

Примечание: при переходе на Celery/мульти‑worker “local-only при success” ограничивает resume на другом worker; поэтому в baseline рекомендуем всё же писать manifest/state во внешнее хранилище всегда, а “local-first” оставить как optimisation (кэш) поверх этого.


