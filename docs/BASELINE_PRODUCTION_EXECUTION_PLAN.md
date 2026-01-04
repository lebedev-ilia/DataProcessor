# Execution plan: production-baseline → baseline training (PR-пакеты)

Этот документ — **пошаговый план реализации** (в виде PR‑пакетов) для доведения проекта до “production‑baseline”, который включает:
- контракты артефактов/manifest/meta,
- **DynamicBatching + state-files/state-manager**,
- **Celery (queue-based execution)**,
- **Triton (model serving, no-fallback)**,
- **external storage (MinIO/S3)**,
- **model versioning + model optimizations** (ONNX/TensorRT/quantization где применимо),
- полный audit модулей,
- и только после этого — dataset builder + обучение baseline.

Основание: `BASELINE_TO_TRAINING_ROADMAP.md` (DoD + этапы B/C/D/E), `DATAPROCESSOR_AUDIT.md`, `PRODUCTION_ARCHITECTURE.md`, `MODEL_SYSTEM_RULES.md`, `DynamicBatching_Q_A.md`.

---

## 0) Принципы исполнения

- **Вертикальные срезы**: каждый PR должен давать проверяемый инкремент (команда/скрипт/валидатор/пример run).
- **Контракты раньше оптимизаций**: сперва сделаем корректные meta/manifest/error_code/model_signature и deterministic dataset, затем ускоряем.
- **No-fallback**: для production‑baseline любые критичные зависимости → fail-fast (как согласовано).
- **Фиксация evidence**: после каждого крупного PR обновляем `DATAPROCESSOR_AUDIT.md` (PASS/FAIL) и кладём 1–2 примера run.

---

## 1) PR‑пакеты (порядок работ)

### PR‑0: “Сборка стенда baseline (dev/prod-like)”

**Цель**: зафиксировать минимальный способ запустить полный стек в dev: Redis + Celery worker + MinIO + (опц.) Triton.

**Сделать**:
- Добавить `docker-compose.yml` (или обновить существующий) со службами:
  - `redis` (broker для Celery)
  - `minio` (+ init bucket)
  - `dataprocessor-worker` (запуск Celery worker)
  - `triton` (опционально, но включаем в baseline как отдельный сервис)
- Документировать env vars в `docs/README.md` или `docs/BASELINE_TO_TRAINING_ROADMAP.md`:
  - `S3_ENDPOINT`, `S3_ACCESS_KEY`, `S3_SECRET_KEY`, `S3_BUCKET`, `S3_PREFIX`
  - `CELERY_BROKER_URL`
  - `TRITON_HTTP_URL`

**DoD**:
- Одна команда поднимает стек; worker видит Redis/MinIO; healthcheck‑скрипт/команда это подтверждает.

---

### PR‑1: “Storage adapter + MinIO как default external storage”

**Цель**: все записи/чтения `result_store`, `manifest.json`, state-files идут через единый storage layer (FS/S3).

**Сделать**:
- Ввести модуль `storage/`:
  - интерфейс `Storage` (read/write/list/exists/atomic_write)
  - реализации `FileSystemStorage` и `S3Storage` (MinIO).
- Определить каноничные “keys”:
  - `result_store/<platform>/<video>/<run>/...` (относительные пути внутри bucket/prefix)
  - `state/<platform>/<video>/<run>/...` (state-files + journals)
  - `frames_dir/<platform>/<video>/<run>/...` (если тоже во внешнем хранилище)
- Запретить absolute paths в manifest для external mode (только относительные keys + storage base).

**DoD**:
- Один run можно целиком записать и потом прочитать с другого процесса/машины (через MinIO).

---

### PR‑2: “Run identity & contract core: dataprocessor_version + analysis_* + manifest fields”

**Цель**: закрыть самые жёсткие FAIL из `DATAPROCESSOR_AUDIT.md` (B‑этап roadmap).

**Сделать**:
- Root orchestrator:
  - формирует `dataprocessor_version` (baseline: `"unknown"`) и прокидывает его везде.
  - формирует `analysis_fps/width/height` (или дефолты) и прокидывает в Segmenter.
- Segmenter:
  - пишет `analysis_fps/analysis_width/analysis_height` в `frames_dir/metadata.json`.
  - гарантирует per-component budgets (индексы не одинаковые “всем”).
- Manifest:
  - расширить `components[]` до `device_used`, `error_code`, `warnings`.
  - гарантировать `error_code` при `status="error"`.

**DoD**:
- Новый run (smoke) проходит по audit‑пунктам run identity/Segmenter/meta/manifest как минимум до PARTIAL→PASS.

---

### PR‑3: “NPZ meta: models_used[]/model_signature + engine/precision/device”

**Цель**: сделать reproducibility и корректный idempotency key.

**Сделать**:
- Общая утилита `meta_builder` (используется Visual/Audio/Text):
  - required keys
  - `models_used[]` (если компонент вызывает модель)
  - `model_signature` (функция от models_used + engine/precision/device)
- Привести core providers и model‑использующие модули к заполнению этих полей.

**DoD**:
- На 3–5 NPZ из одного run meta содержит `dataprocessor_version`; model components содержат `models_used[]/model_signature`.

---

### PR‑4: “Required vs optional profiles (fail-fast policy)”

**Цель**: формализовать training schema (Tier‑0 required) и разрешить optional только явно.

**Сделать**:
- Ввести профиль анализа (MVP): YAML/JSON config + `config_hash`.
- В графе/профиле фиксировать `required=true/false` на компонент.
- Оркестратор:
  - при падении required → stop run (error)
  - при падении optional → component error, run может продолжить (если не влияет на training baseline).

**DoD**:
- Есть пример run, где optional падает, но baseline training‑фичи собираются; в manifest видно required/optional семантику.

---

### PR‑5: “DynamicBatching + State managers (уровни 1/2/3)”

**Цель**: реализовать state-files/state-manager (durable journal) и dependency waiting/stop rules.

**Сделать**:
- Реализовать state-managers:
  - Level 2: run_state manager
  - Level 3: per processor managers (`state_visual.json`/…)
- Durability:
  - `state_events.jsonl` (append-only) + checkpoint state-file.
- State schema:
  - `run` + `processors` + `components` + `checkpoints`.
- Жёсткое правило: missing dependency после grace → `error` и stop run (как согласовано).

**DoD**:
- Любой компонент репортит `waiting/running/success/empty/error/skipped`.
- TextProcessor может ждать OCR (по state) и корректно останавливает run при missing dependency.

---

### PR‑6: “DAG runner (component_graph.yaml) + dependency-ordering (‘priority’)”

**Цель**: сделать план исполнения детерминированным по DAG (baseline/v1/v2).

**Сделать**:
- Парсер `docs/component_graph.yaml`.
- Валидация DAG (acyclic, все depends_on существуют).
- Оркестратор строит execution plan:
  - что параллелить
  - что ждать
  - где stop run
- Заполнить baseline DAG минимум для Tier‑0 required из `BASELINE_IMPLEMENTATION_PLAN.md`.

**DoD**:
- На одном run видно, что компоненты запускаются строго по DAG; в state видны ожидания/чекпоинты.

---

### PR‑7: “Celery: production запуск DataProcessor через очередь + health endpoints”

**Цель**: production‑baseline требует queue-based execution.

**Сделать**:
- Celery app + task `process_video_job(payload)`.
- Retry policy (transient vs permanent) с отражением в state/manifest.
- Health endpoints (минимум для worker контейнера):
  - `/health` (readiness): Redis ok, MinIO ok, Triton ok (если required)
  - `/health/live` (liveness): процесс жив.

**DoD**:
- Можно поставить N jobs в Redis, worker обработает, state/manifest обновляются, health endpoints работают.

---

### PR‑8: “Triton integration: клиент + no-fallback + resolved mapping”

**Цель**: baseline включает Triton и версионирование моделей через mapping.

**Сделать**:
- Triton client (HTTP/gRPC) с таймаутами и error taxonomy.
- Resolved mapping per run:
  - на MVP в виде YAML/JSON профиля (source-of-truth потом в БД),
  - записывается в manifest/state,
  - отражается в NPZ meta через `models_used[]/model_signature`.
- Fail-fast: Triton недоступен/модель не найдена → error/stop run.

**DoD**:
- Хотя бы 1 ключевой компонент работает через Triton и корректно пишет meta/manifest.

---

### PR‑9: “Model optimization pipeline (ONNX/TensorRT/quantization)”

**Цель**: baseline требует оптимизации моделей (где применимо).

**Сделать**:
- Build scripts для выбранных baseline‑моделей:
  - export ONNX
  - build TensorRT (если целимся)
  - quantization (если применимо)
- Артефакты оптимизированных моделей имеют версии и `weights_digest`.
- В meta фиксируем `engine/precision/device` и это входит в `model_signature`.

**DoD**:
- Для 1–2 baseline‑моделей есть оптимизированный путь, используемый в run, и он отражён в meta/manifest.

---

### PR‑10: “Full module audit (закрыть audit items)”

**Цель**: baseline включает аудит всех модулей и фиксацию “единицы обработки”.

**Сделать**:
- Пройтись по всем модулям:
  - Visual core + modules
  - Audio extractors
  - TextProcessor
  - Segmenter
- Для каждого: статус в audit + fix plan + единица обработки (unit) + dependency list.
- Заполнить baseline DAG полностью.

**DoD**:
- `DATAPROCESSOR_AUDIT.md` по ключевым пунктам >= PASS/PARTIAL с чёткими root causes и планом закрытия.

---

### PR‑11: “Dataset Builder (M4) + Training (M5)”

**Цель**: после production foundations сделать обучение baseline.

**Сделать**:
- M4: targets (HF dataset snapshots), enrichment (YouTube API), temporal features.
- M5: CatBoost/LightGBM training + reproducibility manifest.

**DoD**:
- Есть обученная baseline модель + отчёт метрик, воспроизводимость.

---

## 2) Критические решения (зафиксировано)

- **Targets**: только `views+likes` (без comments).
- **Snapshots**: лежат в HF dataset (будет доступ на этапе M4).
- **External storage**: MinIO (S3-compatible) как MVP baseline (совместимо с Celery/мульти‑worker).
- **dataprocessor_version**: `"unknown"` достаточно для baseline (позже можно git hash).

---

## 3) Минимальные тесты/проверки на каждый этап

- PR‑1: интеграционный тест storage adapter (FS↔S3): write→read→list→atomic replace.
- PR‑2..3: “contract smoke run” + meta dumps + `artifact_validator.py`.
- PR‑5..6: тест dependency waiting (OCR handshake) + fail-fast stop run.
- PR‑7: enqueue N jobs, verify state/manifest updates, health endpoints.
- PR‑8..9: Triton required + fail-fast + `model_signature` changes when engine/precision changes.
- PR‑11: dataset build determinism + leakage checks + training metrics report.


