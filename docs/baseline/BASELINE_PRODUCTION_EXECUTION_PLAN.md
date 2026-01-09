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

## Progress tracker (ведём прогресс прямо здесь)

Статусы: `TODO` | `IN_PROGRESS` | `DONE` | `BLOCKED(вопрос)`.

### PR‑0 (infra dev/prod-like): Redis + MinIO (+ optional Triton) + bootstrap worker

- [x] **docker-compose.yml добавлен** (redis+minio+minio-init; triton через profile) — `DONE`
- [x] **env.example добавлен** (S3/Redis/Triton env vars; копировать в `.env`) — `DONE`
- [x] **bootstrap worker** (проверка Redis/MinIO) — `DONE`
- [x] **документация**: `docs/PR0_LOCAL_STACK.md` — `DONE`

### PR‑1 (storage adapter): FS + S3/MinIO

- [x] **storage/**: интерфейс + реализации FS/S3 — `DONE`
- [x] **scripts/storage_smoke_test.py** — `DONE`
- [x] **документация**: `docs/PR1_STORAGE_ADAPTER.md` — `DONE`

### PR‑2 (run identity + Segmenter `analysis_*` + manifest fields) — `DONE`

- [x] `main.py`: добавить `--dataprocessor-version` + `--analysis-*` и проброс в Segmenter/Audio/Text/Visual — `DONE`
- [x] `Segmenter/segmenter.py`: писать `analysis_fps/analysis_width/analysis_height` в metadata + принимать `--dataprocessor-version` — `DONE`
- [x] `VisualProcessor/utils/manifest.py`: добавить `device_used/error_code/warnings` — `DONE`
- [x] `AudioProcessor/run_cli.py` + `TextProcessor/run_cli.py`: принимать `--dataprocessor-version`, писать в meta/manifest + error_code — `DONE`

### PR‑2.1 (env/venv hygiene): каноничные venv + doctor + smoke requirements — `DONE`

- [x] Зафиксировать каноничные venv (root orchestrator `.data_venv`, Visual `.vp_venv`, isolated core venv) — `DONE`
- [x] Добавить `requirements/dataprocessor_smoke.txt` (минимум для orchestrator+Segmenter smoke) — `DONE`
- [x] Добавить `scripts/venv_doctor.py` (проверка venv + ffmpeg/ffprobe) — `DONE`
- [x] Документация: `docs/PR2_1_ENVIRONMENTS.md` + апдейт `docs/BASELINE_RUN_CHECKLIST.md` — `DONE`
- [x] Hygiene: не копить runtime‑yaml и игнорить `_runs/` — `DONE`

### PR‑3 (NPZ meta: models_used[]/model_signature + engine/precision/device) — `DONE`

### PR‑4 (Required vs optional profiles (fail-fast policy)) — `DONE`

### PR‑5 (DynamicBatching + State managers (уровни 1/2/3)) — `DONE`

### PR‑6 (DAG runner (component_graph.yaml) + dependency-ordering (‘priority’)) — `DONE`

### PR‑7 (Celery: production запуск DataProcessor через очередь + health endpoints) — `DONE`

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

**Статус PR‑4 (MVP tasks)**:
- [x] Профиль `profiles/*.yaml` с required/optional — `DONE`
- [x] VisualProcessor enforcement по `requirements{component: bool}` — `DONE`
- [x] Root orchestrator (`main.py`) умеет `--profile-path` и enforce required для audio/text — `DONE`
- [x] Док: `docs/PR4_REQUIRED_OPTIONAL_PROFILES.md` — `DONE`
- [x] Smoke‑evidence: optional fail case (искусственный) — `DONE` (`profiles/pr4_optional_fail.yaml`, run_id=`pr4optional`)

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
- Парсер `docs/reference/component_graph.yaml`.
- Валидация DAG (acyclic, все depends_on существуют).
- Оркестратор строит execution plan:
  - что параллелить
  - что ждать
  - где stop run
- Заполнить baseline DAG минимум для Tier‑0 required из `BASELINE_IMPLEMENTATION_PLAN.md`.

**DoD**:
- На одном run видно, что компоненты запускаются строго по DAG; в state видны ожидания/чекпоинты.

**Статус PR‑6 (MVP tasks)**:
- [x] Заполнить `docs/reference/component_graph.yaml` для `stages.baseline` (Tier‑0 минимум) — `DONE`
- [x] Парсер+валидатор DAG (`dag/component_graph.py`) — `DONE`
- [x] VisualProcessor: `execution_order` (sequential execution) — `DONE`
- [x] Root orchestrator: читает DAG (`--dag-path/--dag-stage`) и передает `execution_order` — `DONE`
- [x] Evidence run: `run_id=pr6smoke2` (см. `_runs/result_store/.../manifest.json` и `_runs/state/.../state_visual.json`) — `DONE`

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

**Статус PR‑8 (MVP tasks)**:
- [x] `dp_triton`: минимальный HTTP client + `ready()` + `infer()` (без внешних deps) — `DONE`
- [x] Profile `resolved_model_mapping` → прокидывание в VisualProcessor runtime cfg — `DONE`
- [x] `manifest.json.run.resolved_model_mapping` (per-run reproducibility) — `DONE`
- [x] `core_clip`: `--runtime=triton` + fail-fast если Triton недоступен + корректный `models_used[]/model_signature` — `DONE` *(без e2e проверок)*
- [x] Документация: `docs/PR8_TRITON_INTEGRATION.md` + пример профиля `profiles/pr8_triton_clip.yaml` — `DONE`

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

**Статус PR‑9 (MVP tasks)**:
- [x] `core_depth_midas`: `--engine=torch|onnx` + `--onnx-path` + fail-fast если deps/onnx отсутствуют — `DONE` *(без e2e)*
- [x] Скрипты: `scripts/model_opt/export_midas_onnx.py` + `scripts/model_opt/quantize_onnx_dynamic.py` — `DONE` *(без запуска)*
- [x] Документация: `docs/PR9_MODEL_OPTIMIZATIONS.md` + пример профиля `profiles/pr9_midas_onnx.yaml` — `DONE`

---

### PR‑10: “Full module audit (закрыть audit items)” — **разбиваем на PR‑10.x**

**Цель**: baseline включает аудит всех модулей/процессоров и фиксацию “единицы обработки” для каждого компонента.

Документ‑регламент: `docs/PR10_MODULE_AUDIT_SPLIT.md`.

**PR‑10.0: Audit harness + inventory + регламент**
- Зафиксировать полный инвентарь компонентов (baseline vs non‑baseline).
- Единый чек‑лист аудита (inputs/outputs/meta/models_used/errors/deps/state/manifest).
- DoD: в `DATAPROCESSOR_AUDIT.md` есть структура/шаблоны для закрытия пунктов по всем компонентам.

**PR‑10.1: Segmenter audit closure**
- Контракты metadata/frames/audio, sampling ownership, empty/error policy.

**PR‑10.2: Visual core providers (Tier‑0 baseline) audit closure**
- `core_clip`, `core_face_landmarks`, `core_object_detections`, `core_depth_midas`, `core_optical_flow`.

**PR‑10.3: Visual modules (Tier‑0 baseline) audit closure**
- `cut_detection`, `shot_quality`, `scene_classification`, `video_pacing`, `uniqueness`, `story_structure`.

**PR‑10.4: Audio extractors audit closure**
- `clap_extractor`, `loudness_extractor`, `tempo_extractor`.

**PR‑10.5: TextProcessor audit closure**
- text artifacts + embeddings path (если включен) + deps/empty policy.

**PR‑10.6: Non‑baseline Visual modules**
- Классификация: baseline / experimental / quarantine.
- Минимальный контракт (meta + empty/error) или явное исключение из baseline.

**PR‑10.7: Baseline DAG completion + consistency pass**
- `component_graph.yaml` полностью соответствует реальным зависимостям и профилям.

**DoD (для каждого PR‑10.x)**:
- Соответствующие секции `DATAPROCESSOR_AUDIT.md` имеют >= PASS/PARTIAL с evidence + root causes + fix list.
- `component_graph.yaml` обновлён при выявлении несоответствий.

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


