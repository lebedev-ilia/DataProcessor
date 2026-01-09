## DataProcessor audit (полный) — полуфинал

Цель: “чеклист-истина” для **полного аудита** DataProcessor.  
Используется как единственная опора для достройки проекта: сначала закрываем пункты этого аудита, потом расширяем/рефакторим.

**Документы правды**: только `DataProcessor/docs/*` (другие папки/файлы не являются источником правил).

---

### 0) Как пользоваться этим аудитом

- Для каждого раздела:
  - отмечаем **PASS/FAIL/N/A**
  - прикладываем **evidence** (пути файлов, команды, фрагменты `manifest.json`/`metadata.json`, примеры NPZ meta)
  - фиксируем **gap** (что именно не соответствует контракту) и **fix plan** (минимальные изменения)
- Любые “неясности” конвертируем в отдельный подпункт **Audit Question** (в конце документа), а не решаем “на глаз”.

---

### Audit log (по мере выполнения)

#### Snapshot: структура процессоров (первичный аудит)

- **Segmenter**: `Segmenter/README.md`, `Segmenter/segmenter.py` → **FAIL** по target standard (нет `requirements.txt`, нет `run_cli.py`, нет `config/`, нет `src/`).
- **AudioProcessor**: `AudioProcessor/requirements.txt`, `AudioProcessor/run_cli.py`, `AudioProcessor/config/`, `AudioProcessor/src/` → **PASS** (ближе всего к target standard).
- **TextProcessor**: `TextProcessor/requirements.txt`, `TextProcessor/run_cli.py`, `TextProcessor/config/`, `TextProcessor/src/` → **PASS** (ближе всего к target standard).
- **VisualProcessor**: есть `VisualProcessor/main.py`, `VisualProcessor/config.yaml`, `VisualProcessor/utils/`, но архитектура разнородна (много `modules/*/main.py`, `core/model_process/*/main.py`, нет `requirements.txt`/`run_cli.py` на уровне процессора, нет `src/`) → **FAIL** по target standard (нужен план унификации).

Evidence (пути):
- `Segmenter/README.md`, `Segmenter/segmenter.py`
- `AudioProcessor/requirements.txt`, `AudioProcessor/run_cli.py`, `AudioProcessor/config/settings.py`, `AudioProcessor/src/*`
- `TextProcessor/requirements.txt`, `TextProcessor/run_cli.py`, `TextProcessor/config/config.py`, `TextProcessor/src/*`
- `VisualProcessor/main.py`, `VisualProcessor/config.yaml`, `VisualProcessor/utils/manifest.py`, `VisualProcessor/modules/*/main.py`, `VisualProcessor/core/model_process/*/main.py`

#### Snapshot: run `smoke27` (контракты artifacts/manifest/meta)

Evidence (конкретные файлы):
- `manifest.json`: `/_runs/result_store/youtube/NSumhkOwSg/smoke27/manifest.json`
- `frames_dir metadata`: `/_runs/segmenter_out/NSumhkOwSg/video/metadata.json`
- `audio metadata`: `/_runs/segmenter_out/NSumhkOwSg/audio/metadata.json`
- NPZ meta dumps (через `./.data_venv/bin/python`):
  - `/_runs/result_store/youtube/NSumhkOwSg/smoke27/core_clip/embeddings.npz`
  - `/_runs/result_store/youtube/NSumhkOwSg/smoke27/clap_extractor/2026-01-01_08-29-19-976785_f111b93d.npz`

Первичные результаты (PASS/FAIL):
- **Per-run storage** (`result_store/<platform>/<video>/<run>/...`): **PASS** (run_id = `smoke27`, артефакты разложены по компонентам).
- **`manifest.json` рядом с артефактами**: **PASS** (есть список компонентов, статусы, схемы, пути к артефактам).
- **`dataprocessor_version` в run identity**: **FAIL**
  - отсутствует в `manifest.run`
  - отсутствует в `frames_dir/metadata.json`
  - отсутствует в `meta` NPZ (и у core, и у audio)
- **Segmenter contract**: **PARTIAL/FAIL**
  - `color_space="RGB"`: **PASS**
  - `analysis_fps/analysis_width/analysis_height` (как в `SEGMENTER_CONTRACT.md`) отсутствуют: **FAIL**
  - budgets per component: фактически все компоненты получили одинаковые 120 union-индексов: **FAIL**
- **NPZ meta minimum** (см. `ARTIFACTS_AND_SCHEMAS.md`): **PARTIAL**
  - `producer/producer_version/schema_version/created_at/platform_id/video_id/run_id/config_hash/sampling_policy_version/status` присутствуют: **PASS**
  - `dataprocessor_version` отсутствует: **FAIL**
- **Model signature (`models_used[]`)**: **PASS** (core providers пишут `models_used[]` + `model_signature`; `core_clip` обновлён).

Root-cause pointers (где именно формируется несоответствие):
- `dataprocessor_version` не формируется/не прокидывается:
  - root orchestrator: `main.py` (нет аргумента/поля `dataprocessor_version`, не прокидывает в Segmenter/Audio/Text/Visual).
- `analysis_fps/analysis_width/analysis_height` не фиксируются:
  - Segmenter умеет `analysis_width/analysis_height`, но пишет только `fps` и `height/width` → `Segmenter/segmenter.py` (`process_video_union`).
  - orchestrator не передаёт `--analysis-height 320`/`--analysis-width` в Segmenter.
- `manifest.json` неполный по полям per-component:
  - `VisualProcessor/utils/manifest.py`: структура `ManifestComponent`/`flush()` не содержит `device_used`, `error_code`, `warnings` (в отличие от требований `ARTIFACTS_AND_SCHEMAS.md` и `MODEL_SYSTEM_RULES.md`).

---

### 1) Сквозные инварианты (обязательные для всех процессоров)

Основание: `CONTRACTS_OVERVIEW.md`, `ARTIFACTS_AND_SCHEMAS.md`, `ORCHESTRATION_AND_CACHING.md`, `BASELINE_RUN_CHECKLIST.md`, `MODEL_SYSTEM_RULES.md`.

#### 1.0 Architecture consistency (унификация структуры процессоров)

Основание: `docs/reference/project_questions.md` (раздел 6.1) + цель проекта “привести процессоры к одной структуре”.

Цель аудита здесь — **зафиксировать разнородность** и превратить её в проверяемые критерии + план унификации.

**Target standard (полуфинал, как референс берём Audio/Text)**:

- **Каждый процессор** (`Segmenter`, `VisualProcessor`, `AudioProcessor`, `TextProcessor`) должен иметь предсказуемую структуру:
  - `README.md` (назначение + вход/выход + артефакты + ошибки/empty)
  - `requirements.txt`
  - `run_cli.py` (единая точка входа CLI для baseline/dev)
  - `config/` (конфиги/дефолты)
  - `src/` (python package код), внутри:
    - `core/` (оркестрация/главный процессор)
    - `extractors/` или `components/` (конкретные вычислители)
    - `schemas/` (контракты/модели данных)
    - `utils/`
  - `tests/` (опционально на MVP, но как цель)
- **Orchestrator (root DataProcessor)**:
  - единая точка входа: `main.py` (или `DataProcessor/main.py`, но должна быть 1 каноническая)
  - должен использовать одинаковые паттерны логирования/конфигов/путей для всех подпроцессоров.

**Критерии PASS**:

- Структура процессоров соответствует target standard (или есть явная причина исключения).
- Точки входа/пути не дублируются без необходимости (нет “двух main, трёх cli” для одного и того же).
- Контракты хранения/manifest/NPZ meta не “расползаются” из-за разного layout.

**Evidence**:

- Дерево файлов каждого процессора + список entrypoints (main/run_cli) + ссылка на фактические места записи артефактов.

#### 1.1 Run identity (строго)

- **Критерий**: в рамках одного run **одни и те же**:
  - `platform_id`, `video_id`, `run_id`, `config_hash`, `sampling_policy_version`, `dataprocessor_version`
- **Где должно быть**:
  - `frames_dir/metadata.json` (от Segmenter)
  - `result_store/.../manifest.json` → `run.*`
  - `meta` каждого NPZ
- **Evidence**:
  - один run: приложить 3 файла (`metadata.json`, `manifest.json`, 1–2 NPZ meta-dump) и показать совпадение полей

#### 1.1.1 Канонический реестр полей (минимум полуфинала)

Основание: `ARTIFACTS_AND_SCHEMAS.md`, `MODEL_SYSTEM_RULES.md`, `ORCHESTRATION_AND_CACHING.md`, `PRIVACY_AND_RETENTION.md`.

**A) `manifest.json` → `run` (обязательные поля):**
- `platform_id`: string
- `video_id`: string
- `run_id`: string
- `config_hash`: string
- `sampling_policy_version`: string
- `dataprocessor_version`: string (baseline допускает `"unknown"`, в проде — версия релиза)
- `created_at`: string (UTC ISO)
- `updated_at`: string (UTC ISO)

**B) `manifest.json` → `components[]` (обязательные поля):**
- `name`: string
- `kind`: string (`core`/`module`/`audio`/`text`/…)
- `status`: string (`ok`/`empty`/`error`)
- `started_at`, `finished_at`: string (UTC ISO)
- `duration_ms`: int
- `artifacts[]`: list
  - `path`: string (абсолютный или относительный, но единообразно в проекте)
  - `type`: string (`npz`/`npy`/…)
- `producer_version`: string
- `schema_version`: string
- `device_used`: string|null (если применимо)
- `error`: string|null (обязателен если `status="error"`)
- `error_code`: string|null (обязателен если `status="error"`, см. `MODEL_SYSTEM_RULES.md`)
- `notes`: string|null

**C) NPZ `meta` (обязательные поля):**
- `producer`: string
- `producer_version`: string
- `schema_version`: string
- `created_at`: string (UTC ISO)
- `platform_id`: string
- `video_id`: string
- `run_id`: string
- `config_hash`: string
- `sampling_policy_version`: string
- `dataprocessor_version`: string (**обязателен**, baseline допускает `"unknown"`)
- `status`: string (`ok`/`empty`/`error`)
- `empty_reason`: string|null (обязателен если `status="empty"`, иначе должен быть `null`)

**D) Model signature / `models_used[]` (обязательные, если компонент вызывал модель):**
- `models_used`: list (может быть пустым для компонентов “без моделей”)
  - `model_name`, `model_version`, `weights_digest`
  - `runtime` (`triton`/`inprocess`), `engine` (`torch`/`onnx`/`tensorrt`)
  - `precision` (`fp16`/`fp32`), `device` (`cuda:0`/`cpu`)
- `model_signature`: string (или вычислимый эквивалент), должен входить в idempotency key.

**E) Idempotency key компонента (минимум полуфинала):**
- `(platform_id, video_id, component, config_hash, sampling_policy_version, producer_version, schema_version, model_signature)`

**F) Privacy (raw):**
- По умолчанию **не сохраняем** raw текст/комменты/ocr в NPZ/логах; raw допускается только при явной policy/флаге и в рамках `hard_cap_days` (см. `PRIVACY_AND_RETENTION.md`).

#### 1.1.2 Матрица ответственности (поле → владелец → место в коде → статус `smoke27`)

Цель: по каждому обязательному полю зафиксировать **единственного владельца**, чтобы не было “поля везде/нигде”.

##### Таблица A — `frames_dir/metadata.json` (Segmenter contract)

| Поле | Владелец (кто обязан заполнить) | Где в коде должно выставляться | Статус в `smoke27` |
|---|---|---|---|
| `platform_id` | Root orchestrator (источник) + Segmenter (запись) | `DataProcessor/main.py` → `Segmenter/segmenter.py::Segmenter.run(run_meta)` | **PASS** |
| `video_id` | Root orchestrator + Segmenter | `DataProcessor/main.py`, `Segmenter/segmenter.py::Segmenter.run` | **PASS** |
| `run_id` | Root orchestrator + Segmenter | `DataProcessor/main.py`, `Segmenter/segmenter.py::Segmenter.run` | **PASS** |
| `config_hash` | Root orchestrator + Segmenter | `DataProcessor/main.py`, `Segmenter/segmenter.py::Segmenter.run` | **PASS** |
| `sampling_policy_version` | Root orchestrator + Segmenter | `DataProcessor/main.py`, `Segmenter/segmenter.py::Segmenter.run` | **PASS** |
| `dataprocessor_version` | Root orchestrator (источник) + Segmenter (запись) | `DataProcessor/main.py` (формировать) → `Segmenter/segmenter.py` (записать) | **FAIL** |
| `analysis_fps` | Sampling policy owner (после PR‑10 аудита) + Segmenter (запись) | DEFERRED: проектируем universal sampling policy после аудита компонентов | **DEFERRED** |
| `analysis_width/analysis_height` | Sampling/resolution policy owner (после PR‑10 аудита) + Segmenter (запись) | DEFERRED: проектируем после аудита требований к разрешению | **DEFERRED** |
| `video_path` | Segmenter | `Segmenter/segmenter.py::process_video_union` | **PASS** |
| `source_fps` | Segmenter | `Segmenter/segmenter.py::process_video_union` | **PASS** |
| `fps` (legacy) | Segmenter | `Segmenter/segmenter.py::process_video_union` | **PASS** |
| `height/width` (фактический output) | Segmenter | `Segmenter/segmenter.py::process_video_union` | **PASS** |
| `channels` | Segmenter | `Segmenter/segmenter.py::process_video_union` | **PASS** |
| `color_space` | Segmenter | `Segmenter/segmenter.py::process_video_union` | **PASS** (`RGB`) |
| `chunk_size/batch_size` | Segmenter | `Segmenter/segmenter.py::process_video_union` | **PASS** |
| `cache_size` | Segmenter | `Segmenter/segmenter.py::process_video_union` | **PASS** |
| `batches[]` | Segmenter | `Segmenter/segmenter.py::process_video_union` | **PASS** |
| `total_frames` (union) | Segmenter | `Segmenter/segmenter.py::process_video_union` | **PASS** |
| `union_frame_indices_source` | Segmenter | `Segmenter/segmenter.py::process_video_union` | **PASS** |
| `union_timestamps_sec` | Segmenter | `Segmenter/segmenter.py::process_video_union` | **PASS** |
| `source_total_frames_read` | Segmenter | `Segmenter/segmenter.py::process_video_union` | **PASS** |
| `created_at` | Segmenter | `Segmenter/segmenter.py::process_video_union` | **PASS** |
| `metadata[component].frame_indices` (union domain) | Segmenter | `Segmenter/segmenter.py::Segmenter.run` | **PASS** |
| `metadata[component].source_frame_indices` (debug) | Segmenter | `Segmenter/segmenter.py::Segmenter.run` | **PASS** |
| `metadata[component].num_indices/num_source_indices` | Segmenter | `Segmenter/segmenter.py::Segmenter.run` | **PASS** |

##### Таблица B — `manifest.json.run`

| Поле | Владелец | Где в коде должно выставляться | Статус в `smoke27` |
|---|---|---|---|
| `platform_id/video_id/run_id/config_hash/sampling_policy_version` | Root orchestrator (источник) + writer (Visual/Audio/Text) | `DataProcessor/main.py` + `RunManifest(run_meta)` | **PASS** |
| `dataprocessor_version` | Root orchestrator | `DataProcessor/main.py` (формировать) → пробросить в Visual/Audio/Text → `RunManifest(run_meta)` | **FAIL** |
| `created_at` | Writer stage (обычно Visual) | `VisualProcessor/main.py::_derive_run_context()` / `RunManifest` | **PASS** |
| `updated_at` | RunManifest | `VisualProcessor/utils/manifest.py::flush()` | **PASS** |
| `frames_dir` | Root orchestrator/Visual stage | `DataProcessor/main.py` → `VisualProcessor/main.py` | **PASS** |
| `root_path` | Root orchestrator/Visual stage | `DataProcessor/main.py` → `VisualProcessor/main.py` | **PASS** |

##### Таблица C — `manifest.json.components[]`

| Поле | Владелец | Где в коде должно выставляться | Статус в `smoke27` |
|---|---|---|---|
| `name` | Orchestrator stage | `AudioProcessor/run_cli.py`, `TextProcessor/run_cli.py`, `VisualProcessor/main.py` | **PASS** |
| `kind` | Orchestrator stage | `audio/text/core/module` | **PASS** |
| `status` | Orchestrator stage (из результата + валидатора) | `*_run_cli.py` + `VisualProcessor/main.py` | **PASS** |
| `started_at/finished_at/duration_ms` | Orchestrator stage | `*_run_cli.py` + `VisualProcessor/main.py::_run_component_subprocess` | **PASS** |
| `artifacts[]` (`path/type`) | Orchestrator stage | `*_run_cli.py` + `VisualProcessor/main.py` | **PASS** |
| `producer_version` | Компонент → meta → manifest | Core providers/Audio OK; Modules через BaseModule | **PARTIAL** (core/audio PASS, modules **FAIL**=`unknown`) |
| `schema_version` | Компонент → meta → manifest | writers NPZ meta + `validate_npz` extraction | **PASS** |
| `device_used` | Компонент (источник) + orchestrator (поднять в manifest) | Писать в NPZ meta + переносить в manifest | **FAIL** (в manifest отсутствует) |
| `batch_size` (если применимо) | Компонент | NPZ meta (опц. в manifest) | **FAIL/Gap** |
| `error` | Orchestrator stage | `*_run_cli.py` + `VisualProcessor/main.py` | **PASS** (null) |
| `error_code` | Компонент/Orchestrator stage | нужно расширить `ManifestComponent` | **FAIL** (поля нет) |
| `notes` | Orchestrator stage | валидатор → notes | **PASS** (null) |

##### Таблица D — NPZ `meta` (общий контракт + model signature)

| Поле | Владелец | Где в коде должно выставляться | Статус в `smoke27` |
|---|---|---|---|
| `producer` | Компонент | writers: core providers / `BaseModule.save_results()` / `*_run_cli.py` | **PASS** |
| `producer_version` | Компонент | core: константы; modules: `BaseModule` (версия); audio/text: версии экстракторов/пайплайна | **PARTIAL** (modules FAIL=`unknown`) |
| `schema_version` | Компонент | core: `SCHEMA_VERSION`; modules: default `*_npz_v1`; audio/text: фиксировано | **PASS** |
| `created_at` | Компонент | all writers | **PASS** |
| `platform_id/video_id/run_id/config_hash/sampling_policy_version` | Root orchestrator (источник) + компонент (запись) | propagate + write meta | **PASS** |
| `dataprocessor_version` | Root orchestrator (источник) + компонент (запись) | propagate + write meta | **FAIL** |
| `status` | Компонент | all writers | **PASS** |
| `empty_reason` | Компонент | all writers | **PASS** (null) |
| `device_used` | Компонент | write meta (cpu/cuda:0/…) | **PARTIAL** (есть у `clap_extractor`, нет у проверенных visual NPZ) |
| `engine` | Компонент (если модель) | `torch/onnx/tensorrt` | **FAIL/Gap** |
| `precision` | Компонент (если модель) | `fp16/fp32` | **FAIL/Gap** |
| `models_used[]` | Компонент (если модель) | `model_name/model_version/weights_digest/runtime/engine/precision/device` | **FAIL** |
| `model_signature` | Компонент (если модель) | вычислить/зафиксировать | **FAIL** |
| `seed` (если применимо) | Компонент | фиксировать для воспроизводимости | **N/A/Gap** |
| `runtime_env` (cuda/cudnn/driver) | Orchestrator/component | фиксировать при необходимости | **N/A/Gap** |
| `git_commit` | Orchestrator/component | фиксировать при необходимости | **N/A/Gap** |

#### 1.1.3 Gap index (что блокирует соответствие докам прямо сейчас)

Топ‑приоритет (наиболее “сквозные” пробелы):
- **`dataprocessor_version`**: отсутствует в manifest/NPZ/frames metadata → ломает воспроизводимость и cache index.
- **`analysis_*`**: правила выбора `analysis_fps/analysis_width/analysis_height` требуют design по требованиям компонентов → **DEFERRED** до завершения аудита (см. `SEGMENTER_CONTRACT.md`).
- **`models_used[]`/`model_signature`**: **PARTIAL** — часть компонентов уже пишет (`core_clip`, `core_face_landmarks`, `core_depth_midas`), но нужно закрыть оставшиеся.
- **`device_used` и `error_code` в manifest**: нет полей/нет заполнения → observability не соответствует правилам.
- **`producer_version="unknown"` у Visual modules**: root cause в `BaseModule.save_results()` и структуре модулей.
- **Privacy (Text payload)**: потенциальное сохранение raw текста в NPZ без policy/флага.
- **Segmenter external tools**: Segmenter должен fail-fast если отсутствуют `ffmpeg/ffprobe` (prod requirement).

#### 1.2 Storage per-run (строго)

- **Критерий**: структура:
  - `result_store/<platform_id>/<video_id>/<run_id>/manifest.json`
  - `result_store/<platform_id>/<video_id>/<run_id>/<component_name>/*.npz`
- **Критерий**: `manifest.json` обновляется **атомарно** (tmp → replace) и **мерджится** между стадиями (Audio → Text → Visual).
- **Evidence**: реальный `result_store` пример + содержимое `manifest.json` с компонентами из нескольких стадий.

#### 1.3 NPZ = source-of-truth (строго)

- **Критерий**: в `result_store` не появляются произвольные JSON-артефакты (кроме `manifest.json`).
- **Evidence**: поиск по `result_store` + пример run.

#### 1.4 No-fallback policy (строго)

- **Критерий**: если отсутствует:
  - `metadata[component].frame_indices`, или
  - обязательный dependency artifact  
  → компонент **обязан падать (raise)**, не “генерировать сам”.
- **Evidence**: кодовые места (проверка до запуска) + тест/лог, где при отсутствии dependency run падает корректной ошибкой.

#### 1.5 Valid empty outputs (строго)

- **Критерий**: “пустота данных” (нет лиц/нет аудио/нет текста) — это:
  - `status="empty"` в meta,
  - `empty_reason` (из словаря в `ARTIFACTS_AND_SCHEMAS.md`),
  - массивы чисел — `NaN`,
  - присутствуют `*_present`/validity masks (где применимо).
- **Evidence**: 1–2 NPZ артефакта со статусом empty + meta.

#### 1.6 Model signature / models_used (для компонентов, которые вызывают модели)

- **Критерий**: meta содержит `models_used[]` и это попадает в `model_signature` (см. `MODEL_SYSTEM_RULES.md`).
- **Критерий**: изменение модели/engine/precision/device → новый кэш-ключ.
- **Evidence**: пример NPZ meta, где видно `models_used[]`.

#### 1.7 Retention / GC / privacy (проверка политик)

- **Критерий**: `frames_dir` TTL = 7 дней (см. `ORCHESTRATION_AND_CACHING.md`).
- **Критерий**: hard cap хранения = 60 дней (см. `PRIVACY_AND_RETENTION.md`).
- **Evidence**: конфиг/планировщик удаления (если уже реализован) или issue в плане работ.

---

### 2) Orchestrator / DataProcessor (root pipeline)

Основание: `PRODUCTION_ARCHITECTURE.md`, `ORCHESTRATION_AND_CACHING.md`, `BASELINE_RUN_CHECKLIST.md`, `CONTRACTS_OVERVIEW.md`.

#### 2.1 Execution path (baseline v0) соответствует контрактам

- **Критерий**: путь:
  - Segmenter → (опционально) Audio → (опционально) Text → Visual
- **Критерий**: orchestrator формирует/получает run identity и прокидывает во все подпроцессоры.
- **Evidence**: лог одного run + `manifest.json` содержит компоненты из всех активных стадий.

#### 2.2 Required vs optional (качество / fail-fast)

- **Критерий**: по умолчанию включённые в профиль компоненты = required (fail-fast), optional только если явно помечены.
- **Evidence**: место, где профиль/конфиг определяет required/optional + пример run с частичной ошибкой optional-компонента.

#### 2.3 Cache policy (video/profile)

- **Критерий**: кэш heavy compute учитывает `cache_ttl_days` (дефолт 3 дня, настраивается).
- **Критерий**: компонентный idempotency key использует `model_signature` (см. `ORCHESTRATION_AND_CACHING.md` + `MODEL_SYSTEM_RULES.md`).
- **Evidence**: описание/код выбора reuse + пример cache hit в manifest.

#### 2.4 Наблюдаемость (минимум)

- **Критерий**: per component фиксируются `started_at/finished_at/duration_ms/status/device_used/error/producer_version/schema_version`.
- **Evidence**: `manifest.json` из run.

#### 2.5 Snapshot: root orchestrator code audit (`DataProcessor/main.py`)

Evidence (кодовые точки):
- `DataProcessor/main.py`:
  - формирует `video_id` и `run_id`
  - вычисляет `config_hash` как sha256 от YAML дампа (включая `VisualProcessor/config.yaml` + флаги run_audio/run_text)
  - запускает:
    - Segmenter (union frames_dir)
    - AudioProcessor (опционально)
    - TextProcessor (опционально)
    - VisualProcessor (всегда)
  - прокидывает run identity в Segmenter/Audio/Text/Visual через CLI args

PASS/FAIL (по коду + run `smoke27`):
- **Execution path Segmenter → Audio/Text → Visual**: **PASS**
- **Run identity propagation (platform_id/video_id/run_id/sampling_policy_version/config_hash)**: **PASS**
  - присутствует в `frames_dir/metadata.json` и в `manifest.run` (кроме `dataprocessor_version`).
- **`dataprocessor_version`**: **FAIL**
  - orchestrator не формирует и не прокидывает `dataprocessor_version` ни в Segmenter, ни в manifest/NPZ meta.
- **analysis timeline params (`analysis_fps/analysis_width/analysis_height`)**: **FAIL**
  - orchestrator не передаёт `--analysis-width/--analysis-height` в Segmenter (и не задаёт дефолты из `SEGMENTER_CONTRACT.md`).
- **Required vs optional (fail-fast policy)**: **FAIL/Gap**
  - Audio/Text запускаются с `check=False` (best-effort по умолчанию), что противоречит полуфинальному правилу “все включённые в профиль — required” (см. `ORCHESTRATION_AND_CACHING.md`).
  - Нет явного механизма профиля анализа (required/optional) в коде baseline orchestrator.
- **Product constraints (5s..20min, downscale>1080p)**: **FAIL/Gap**
  - В orchestrator нет pre-validation длительности/разрешения до создания run (как требует `PRODUCT_CONTRACT.md`).
- **Cache/idempotency (reuse)**: **FAIL/Gap**
  - `config_hash` считается, но нет проверки существующих валидных артефактов по idempotency key/TTL (см. `ORCHESTRATION_AND_CACHING.md`).

---

### 3) Segmenter (sampling + frames_dir contract)

Основание: `SEGMENTER_CONTRACT.md`, `BASELINE_RUN_CHECKLIST.md`, `PRODUCT_CONTRACT.md`.

#### 3.1 frames_dir/metadata.json

- **Критерий**: `metadata.json` существует и валиден.
- **Критерий**: `frames_dir` содержит **только union sampled** кадры.
- **Критерий**: `metadata[component].frame_indices` — индексы **в union timeline** (0..N-1).
- **Критерий**: `analysis_fps`, `analysis_width/height`, `color_space="RGB"` присутствуют (правила выбора — **DEFERRED** до завершения аудита компонентов).
- **Evidence**: `metadata.json` + подсчёт количества кадров в папке.

#### 3.2 Budgets per component

- **Критерий**: Segmenter выдаёт отдельные `frame_indices` per component и соблюдает `min/target/max` бюджеты (policy).
- **Параметры** (стартовые ориентиры из дока):
  - `cut_detection`: 400–1500
  - `core_clip`: 200–800
  - `core_depth_midas`: 120–400
  - `core_face_landmarks`: 200–800
  - `shot_quality`: 200–1000
- **Evidence**: `metadata.json` (размеры массивов индексов).

#### 3.3 Ограничения входа (product constraints)

- **Критерий**: min video length = 5 сек, max = 20 мин.
- **Критерий**: downscale >1080p до 1080p до анализа (как описано в `PRODUCT_CONTRACT.md`).
- **Evidence**: место в коде/пайплайне, где эти проверки выполняются + пример ошибки/обработки.

#### 3.4 Snapshot: Segmenter code audit (sampling + frames_dir metadata)

Evidence (кодовые точки):
- `Segmenter/segmenter.py`:
  - `Segmenter.run()` (union-sampled mode): считает per-component `source_frame_indices` по budgets, строит `union_source_indices`, извлекает только union кадры, затем маппит в union-domain `frame_indices` для каждого компонента.
  - `process_video_union()`: сохраняет RGB кадры в `batch_*.npy`, пишет `union_frame_indices_source` + `union_timestamps_sec` (предпочтительно из container timestamp), `height/width/channels`.
  - SKIP semantics: если видео не открывается → Segmenter exit-code `10`, оркестратор завершает run без дальнейших стадий.
  - `_build_default_component_budgets()` + `_build_visual_extractor_configs_from_visual_cfg()`: генерация budgets per component на основе `VisualProcessor/config.yaml`.
- Реальный run evidence:
  - `_runs/segmenter_out/NSumhkOwSg/video/metadata.json` (run_id=`smoke27`)

PASS/FAIL (по коду + по run evidence):
- **Union-only frames_dir**: **PASS**
  - metadata содержит `union_frame_indices_source` и `union_timestamps_sec`, а компоненты получают `frame_indices` в union domain (0..N-1).
- **Run identity keys в metadata**: **PASS**
  - в конце `metadata.json` присутствуют `platform_id`, `video_id`, `run_id`, `sampling_policy_version`, `config_hash` (проброшены через `run_meta`).
- **Budgets per component (разные индексы/разные длины)**: **FAIL (по run evidence) / PARTIAL (по коду)**
  - Код поддерживает разные budgets per component и пишет `source_frame_indices` + `frame_indices`.
  - Но в run `smoke27` все компоненты имеют одинаковые `num_source_indices=120` и одинаковые `frame_indices=0..119` → budgets фактически не проявились.
  - Возможная причина: конфиг, использованный при run, имел одинаковые `target_frames` для множества компонентов, либо drift конфигов (текущий `VisualProcessor/config.yaml` отключает большинство компонентов, что не совпадает с содержимым metadata).
- **analysis timeline fields (`analysis_fps`, `analysis_width`, `analysis_height`)**: **PASS / policy DEFERRED**
  - поля записываются в `frames_dir/metadata.json`, но правила выбора относятся к sampling policy и проектируются после полного аудита компонентов.
- **`dataprocessor_version` в metadata**: **PASS**
  - поле пробрасывается оркестратором и записывается Segmenter в `frames_dir/metadata.json` и `audio/metadata.json`.
- **Product constraints (min/max длительность, pre-downscale >1080p)**: **FAIL/Gap**
  - В Segmenter не видно явной проверки min/max длительности.
  - Downscale >1080p описан как pre-processing (вне Segmenter), но в текущем пайплайне не видно enforce точки.

---

### 4) VisualProcessor (core providers + modules)

Основание: `CONTRACTS_OVERVIEW.md`, `BASELINE_RUN_CHECKLIST.md`, `SEGMENTER_CONTRACT.md`, `ARTIFACTS_AND_SCHEMAS.md`, `MODEL_SYSTEM_RULES.md`.

> В этом аудите мы проверяем VisualProcessor как “чёрный ящик” через его артефакты/контракты и интеграцию с Segmenter/result_store.

#### 4.0 Архитектурная структура VisualProcessor (унификация)

- **Критерий**: наличие 1 канонической точки входа (CLI) и 1 канонического конфигурационного пути.
- **Критерий**: VisualProcessor пишет артефакты строго в per-run `result_store` и не требует “особых путей”.
- **Evidence**: список entrypoints (`VisualProcessor/main.py`, module mains если есть) + правила выбора.

#### 4.1 Контракт по sampling (строго)

- **Критерий**: Visual компоненты используют **только** `metadata[component].frame_indices` и не семплируют сами.
- **Evidence**: по ключевым компонентам (core providers + 2–3 модуля) найти/зафиксировать отсутствие “fallback sampling”.

#### 4.2 Размещение артефактов

- **Критерий**: каждый компонент пишет в `result_store/.../<component>/...`.
- **Критерий**: NPZ meta соответствует минимуму (`ARTIFACTS_AND_SCHEMAS.md`).
- **Evidence**: 2–3 NPZ артефакта (core + module).

#### 4.3 Dependencies: core providers → modules

- **Критерий**: модуль проверяет наличие нужных core artifacts; если нет — fail-fast.
- **Критерий**: если core вернул `status="empty"`, модуль:
  - по зависимым полям — `empty`, по остальным — продолжает (если применимо) (см. решения в `MODEL_SYSTEM_RULES.md`).
- **Evidence**: пример зависимого модуля + пример empty кейса.

#### 4.4 Resource/batching

- **Критерий**: batch_size выбирается до запуска и фиксируется в meta (если компонент батчится).
- **Evidence**: meta поля + конфиг/логика выбора.

#### 4.5 Snapshot: VisualProcessor code audit (core/providers + modules + manifest)

Evidence (кодовые точки):
- `VisualProcessor/main.py`:
  - формирует per-run `run_rs_path = <rs_path>/<platform>/<video_id>/<run_id>`
  - запускает `core providers` и `modules` как subprocess, собирает артефакты, валидирует NPZ, пишет `manifest.json`
  - GPU-gating через `gpu_max_concurrent` + `threading.Semaphore`
- `VisualProcessor/utils/manifest.py`: атомарный merge/upsert `manifest.json` (RunManifest)
- `VisualProcessor/utils/artifact_validator.py`: baseline NPZ meta validation (presence keys)
- `VisualProcessor/modules/base_module.py`: единый каркас для модулей (run identity check, dependency loading, сохранение NPZ)
- Примеры строгого sampling:
  - `VisualProcessor/core/model_process/core_clip/main.py`: `_require_frame_indices()` (no fallback)
  - `VisualProcessor/modules/shot_quality/shot_quality.py`: строгая проверка согласованности `frame_indices` между core providers

PASS/FAIL (первичный, по коду):
- **Per-run result_store layout**: **PASS**
  - `VisualProcessor/main.py` переключает `rs_path` в per-run storage и все subprocess получают `--rs-path <run_rs_path>`
- **Manifest atomic merge**: **PASS**
  - `utils/manifest.py` пишет атомарно (`.tmp` + `os.replace`) и мерджит существующий manifest.
- **Manifest schema completeness**: **FAIL**
  - `ManifestComponent` не поддерживает поля из доков (например `device_used`, `error_code`, расширенные причины empty и т.д.)
  - `run` секция не включает `dataprocessor_version` (см. run `smoke27`)
- **Run identity strictness**: **PARTIAL / риск fallback**
  - `modules/base_module.py` требует в `frames_dir/metadata.json`: `platform_id/video_id/run_id/sampling_policy_version/config_hash` (fail-fast если нет)
  - Но `VisualProcessor/main.py` делает best-effort derivation (`_derive_run_context`) из `frames_dir/metadata.json` и/или config (может замаскировать пропуски upstream, если `frames_dir` metadata неполный/legacy)
- **Sampling contract (no fallback)**: **PARTIAL**
  - `core_clip` и ряд модулей строго требуют `metadata[component].frame_indices` (no fallback)
  - Но `BaseModule.get_frame_indices(..., fallback_to_all=False)` позволяет fallback при `fallback_to_all=True` (в коде предусмотрено), а некоторые legacy куски в модуле эмоций используют fallback источники данных (см. ниже)
- **`producer_version="unknown"` в manifest**: **FAIL (root cause найден)**
  - `BaseModule.save_results()` ставит `producer_version` как `getattr(self, "VERSION", ...)` / `self.producer_version`, иначе `"unknown"`.
  - Многие модули определяют `VERSION` как **module-level константу**, но не как атрибут класса/экземпляра → в meta остаётся `"unknown"`, и `VisualProcessor/main.py` потом берёт `producer_version` из meta для manifest.
  - Пример: `modules/shot_quality/shot_quality.py` имеет `VERSION="2.0"`, но `ShotQualityModule` не выставляет `self.VERSION`.
- **`schema_version` consistency**: **PARTIAL**
  - `BaseModule.save_results()` по умолчанию пишет `schema_version = f"{module_name}_npz_v1"` если не задан `SCHEMA_VERSION` атрибутом.
  - Core providers (например `core_clip`) задают `SCHEMA_VERSION` явно (лучше).
  - Для многих модулей требуется унификация: явные `SCHEMA_VERSION` + единый реестр версий.
- **No-fallback policy (dependencies)**: **PARTIAL**
  - `ShotQualityModule` fail-fast при отсутствии зависимостей и проверяет alignment индексов (хорошо).
  - Но `HighLevelSemanticModule` ловит `Exception` и возвращает `_empty_result()` (скрывает настоящую ошибку; “empty” без явной причины).
  - `EmotionFaceModule`/`VideoEmotionProcessor.frames_with_face_load()` должен использовать `core_face_landmarks.face_present` (face_detection удалён) и не иметь fallback на legacy файлы.
- **Venv isolation**: **PARTIAL / риск**
  - `_build_subprocess_cmd()` выбирает venv для core providers и `.vp_venv` для модулей, но если `python_exec` не найден — fallback на `sys.executable` (может приводить к неявной зависимости от окружения оркестратора).
- **Model signature / `models_used[]`**: **FAIL/Gap**
  - Core providers пишут `model_name` (например `core_clip`), но не фиксируют `models_used[]` в NPZ meta по правилам `MODEL_SYSTEM_RULES.md`.

---

### 5) AudioProcessor

Основание: `ARTIFACTS_AND_SCHEMAS.md`, `ERROR_HANDLING_AND_EDGE_CASES.md`, `MODEL_SYSTEM_RULES.md`, `BASELINE_RUN_CHECKLIST.md`.

#### 5.1 Артефакты и формат

- **Критерий**: пишет per-run NPZ в `result_store` (например: `clap_extractor`, `tempo_extractor`, `loudness_extractor`) согласно `ARTIFACTS_AND_SCHEMAS.md`.
- **Критерий**: meta минимум + корректные `empty_reason` при отсутствии аудио.
- **Evidence**: 1–2 аудио NPZ.

#### 5.2 Ошибки/ретраи

- **Критерий**: transient ошибки можно retry, логические/контрактные — fail-fast (см. `ERROR_HANDLING_AND_EDGE_CASES.md`).
- **Evidence**: обработка/коды ошибок в manifest.

#### 5.3 Model signature (если используются модели)

- **Критерий**: при использовании ML моделей фиксируем `models_used[]` (см. `MODEL_SYSTEM_RULES.md`).
- **Evidence**: NPZ meta.

#### 5.4 Snapshot: AudioProcessor code audit (на основе `run_cli.py` и core)

Evidence (кодовые точки):
- `AudioProcessor/run_cli.py`:
  - формирование per-run пути: `run_rs_path = <rs_base>/<platform>/<video_id>/<run_id>`
  - запись `manifest.json` через `VisualProcessor/utils/manifest.py` (`RunManifest`)
  - запись NPZ через `_atomic_save_npz()` и `_meta()`
- `AudioProcessor/src/core/main_processor.py`: запуск экстракторов, `device_used` в `ExtractorResult`
- `AudioProcessor/src/core/base_extractor.py`: политика выбора устройства (auto/cpu/cuda)
- `AudioProcessor/src/schemas/models.py`: pydantic-схемы (API слой)

PASS/FAIL (первичный):
- **Per-run NPZ artifacts**: **PASS**
  - пишет NPZ для `clap_extractor`/`tempo_extractor`/`loudness_extractor` в per-run директорию
  - meta включает run identity (`platform_id/video_id/run_id/config_hash/sampling_policy_version`) и `status/empty_reason`
- **Manifest merge/atomic**: **PASS**
  - использует общий `RunManifest` (мерджит и атомарно пишет)
- **Запрет JSON артефактов**: **PARTIAL**
  - `MainProcessor` может писать legacy JSON manifest, но `run_cli.py` кладёт debug в `run_rs_path/_tmp_audio/` (допустимо как `_tmp_*`)
  - проверить, что debug JSON действительно не попадает вне `_tmp_audio` (edge-case)
- **`dataprocessor_version`**: **FAIL**
  - `run_cli.py` не пишет `dataprocessor_version` ни в `manifest.run`, ни в NPZ meta (см. evidence run `smoke27`)
- **No-fallback policy (device)**: **FAIL/Gap**
  - `BaseExtractor`: если пользователь явно выбрал `--device cuda`, а CUDA недоступна, происходит переключение на CPU (L76–L79) вместо fail-fast.
  - Это конфликтует с полуфинальными правилами “без fallback” по device (см. `MODEL_SYSTEM_RULES.md` / `MODELS_Q.md`).
- **Schemas usage**: **PARTIAL**
  - pydantic модели присутствуют (`src/schemas/models.py`), но нужно проверить, используются ли они реально в runtime/валидации артефактов или только в API.

---

### 6) TextProcessor

Основание: `ARTIFACTS_AND_SCHEMAS.md`, `MODEL_SYSTEM_RULES.md`, `PRIVACY_AND_RETENTION.md`.

#### 6.1 Артефакт

- **Критерий**: пишет `text_processor/*.npz` per-run с tabular-friendly структурой (см. `ARTIFACTS_AND_SCHEMAS.md`).
- **Evidence**: NPZ + meta.

#### 6.2 Privacy

- **Критерий**: raw текст/комменты/ocr по умолчанию не логируются и не хранятся (кроме разрешённых политик).
- **Evidence**: отсутствие raw в артефактах/логах или флаг/политика хранения.

#### 6.3 Embeddings cache (если есть)

- **Критерий**: если есть кэш embeddings, ключ включает `model_signature` и preprocessing flags (см. `MODEL_SYSTEM_RULES.md`).
- **Evidence**: описание/код ключа кэша.

#### 6.4 Snapshot: TextProcessor code audit (на основе `run_cli.py` и core)

Evidence (кодовые точки):
- `TextProcessor/run_cli.py`:
  - формирование per-run пути: `run_rs_path = <rs_base>/<platform>/<video_id>/<run_id>`
  - запись NPZ через `_atomic_save_npz()` и `_meta()`
  - запись `manifest.json` через `VisualProcessor/utils/manifest.py` (`RunManifest`)
- `TextProcessor/src/core/main_processor.py`: реестр экстракторов, device routing, сбор `features` (results/timings)
- `TextProcessor/src/core/model_registry.py`: singleton registry SentenceTransformer (кэш в памяти)
- `TextProcessor/src/schemas/models.py`: dataclasses `VideoDocument` (входной контракт)

PASS/FAIL (первичный, по коду):
PASS/FAIL (обновлено после фиксов):
- **Per-run NPZ artifact + manifest merge**: **PASS**
  - пишет каноничный артефакт: `result_store/<platform>/<video>/<run>/text_processor/text_features.npz`
  - апдейтит `manifest.json` через общий `RunManifest` (atomic merge).
- **`dataprocessor_version`**: **PASS**
  - прокидывается через CLI (`--dataprocessor-version`) и пишется в `manifest.run` и в NPZ `meta`.
- **Model system (`models_used[]` / `model_signature`)**: **PASS (baseline-level)**
  - `run_cli.py` добавляет `models_used[]` через `dp_models` (ModelManager, local-only).
  - `model_signature` вычисляется через `apply_models_meta` (best-effort).
- **Privacy (raw text)**: **PASS**
  - по умолчанию в NPZ сохраняется только **privacy-safe** `payload_summary` (без raw текста).
  - raw payload допускается только под флагом `--store-raw-payload` и пишется в `_tmp_text/`.
- **No-network policy**: **PASS**
  - убраны runtime downloads (`nltk.download`), SentenceTransformer загружается через ModelManager (local artifacts, offline env).
- **No-fallback policy (device)**: **PASS**
  - запрещён auto fallback `cuda→cpu` внутри embedding ветки; если CUDA/модель недоступны — fail-fast.

Note:
- В `_runs/result_store` сейчас не найдено ни одного `text_processor` артефакта — нужно создать/воспроизвести run с `--run-text`, чтобы подтвердить фактический состав payload и проверить privacy на реальных данных.

---

### 7) Manifest.json (детальный аудит)

Основание: `ARTIFACTS_AND_SCHEMAS.md`, `MODEL_SYSTEM_RULES.md`.

- **Критерий**: `manifest.run` содержит: `platform_id`, `video_id`, `run_id`, `config_hash`, `sampling_policy_version`, `dataprocessor_version`, timestamps.
- **Критерий**: `manifest.components[]` для каждого компонента содержит минимум:
  - `name`, `status`, `started_at`, `finished_at`, `duration_ms`
  - `producer_version`, `schema_version`
  - `device_used` (если применимо)
  - `error/error_code` (если error)
  - `notes/warnings` (если есть)
- **Критерий**: запись manifest атомарна; конкурентные апдейты предотвращены (апдейт из одного процесса/потока).

#### 7.1 Evidence (реальный run `smoke27`)

- `/_runs/result_store/youtube/NSumhkOwSg/smoke27/manifest.json`
- Код записи: `VisualProcessor/utils/manifest.py` (`RunManifest`, атомарный tmp→replace, merge existing)

#### 7.2 PASS/FAIL (по `smoke27` + коду)

- **Пер-run структура и наличие manifest**: **PASS**
  - `manifest.json` лежит рядом с артефактами в `result_store/<platform>/<video>/<run>/manifest.json`.
- **Merge между стадиями (Audio → Visual)**: **PASS**
  - `manifest.components[]` содержит и `kind="audio"` и `kind="core"/"module"` компоненты.
  - `RunManifest` явно загружает существующий manifest и мерджит `run` meta + components.
- **Atomic write**: **PASS**
  - `_atomic_write_json()` пишет во временный файл и делает `os.replace`.
- **`manifest.run.dataprocessor_version`**: **FAIL**
  - В `smoke27` `run` содержит `platform_id/video_id/run_id/config_hash/sampling_policy_version/created_at/updated_at`, но **нет** `dataprocessor_version`.
- **Минимум observability per-component**: **FAIL (частично)**
  - PASS: `name/kind/status/started_at/finished_at/duration_ms/artifacts/producer_version/schema_version/error/notes` присутствуют.
  - FAIL: **нет `device_used`** у компонентов (а он требуется доками для observability, где применимо).
  - FAIL: **нет `error_code`** у компонентов (даже при `status="error"` сейчас было бы нечем кодировать).
- **`model_signature` / `models_used[]` в manifest**: **FAIL/Gap**
  - `MODEL_SYSTEM_RULES.md` требует фиксировать `model_signature` (и/или resolved `models_used[]`) для кэша/воспроизводимости; в manifest этого нет.
  - В core providers иногда есть `model_name` в NPZ, но manifest не содержит “resolved mapping” и не агрегирует это в `run`.
- **`producer_version` для модулей**: **FAIL**
  - В `smoke27` у модулей `producer_version="unknown"` (root cause: `BaseModule.save_results()` читает версию из `self.VERSION/self.producer_version`, а многие модули держат `VERSION` как module-level константу).
- **Конкурентные апдейты (safety)**: **PARTIAL / риск**
  - Текущий baseline orchestrator запускает стадии последовательно, поэтому гонок почти нет.
  - Но `RunManifest.flush()` делает replace без файловой блокировки; если в будущем несколько процессов будут апдейтить один manifest одновременно, возможны race conditions (lost update).

---

### 8) NPZ meta (детальный аудит)

Основание: `ARTIFACTS_AND_SCHEMAS.md`, `MODEL_SYSTEM_RULES.md`, `PRIVACY_AND_RETENTION.md`.

#### 8.1 Required meta keys (baseline contract)

- **Критерий**: в каждом NPZ есть `meta` и он содержит минимум:
  - `producer`, `producer_version`, `schema_version`, `created_at`
  - `platform_id`, `video_id`, `run_id`
  - `config_hash`, `sampling_policy_version`
  - `dataprocessor_version` (**обязателен**)
  - `status`, `empty_reason`
- **Evidence**: meta-dump нескольких NPZ из одного run.

#### 8.2 Model signature / models_used[]

- **Критерий**: если компонент использует ML-модели, в `meta` фиксируется `models_used[]` и (явно или вычислимо) `model_signature`.
- **Критерий**: `model_signature` входит в idempotency key компонента (см. `ORCHESTRATION_AND_CACHING.md`).

#### 8.3 Privacy (raw)

- **Критерий**: по умолчанию raw текст/комменты/ocr не сохраняются в NPZ (кроме debug под флагом/policy).

#### 8.4 Evidence: NPZ meta dump (run `smoke27`)

Проверенные артефакты:
- `core_clip/embeddings.npz`
- `core_depth_midas/depth.npz`
- `cut_detection/*.npz`
- `shot_quality/*.npz`
- `clap_extractor/*.npz`

PASS/FAIL:
- **Meta присутствует и содержит базовые run keys**: **PASS**
  - Везде есть `producer/producer_version/schema_version/created_at/platform_id/video_id/run_id/config_hash/sampling_policy_version/status/empty_reason`.
- **`dataprocessor_version` в meta**: **FAIL**
  - Отсутствует во всех проверенных NPZ (должно быть хотя бы `"unknown"` в baseline).
- **`models_used[]` / `model_signature`**: **PARTIAL**
  - По коду: core providers (`core_clip`, `core_face_landmarks`, `core_depth_midas`, `core_object_detections`) пишут `models_used[]` + `model_signature` через `meta_builder`.
  - Evidence (run-level) нужно обновить отдельным smoke-run, т.к. `smoke27` — старый слепок.
  - По коду: `cut_detection` пишет `models_used[]/model_signature` **только при `use_clip=true`** (CLIP — модель; веса должны быть локальными, runtime downloads запрещены).
- **`device_used`/`engine`/`precision`**: **FAIL/Gap**
  - В `clap_extractor` есть `device_used='cpu'`, но у остальных проверенных NPZ поля отсутствуют.
  - Это мешает воспроизводимости и формированию `model_signature`.
- **Privacy (raw)**: **FAIL/Gap (по коду)**
  - `TextProcessor/run_cli.py` сохраняет `payload` целиком в NPZ, что потенциально включает raw текст/комменты/транскрипт и конфликтует с `PRIVACY_AND_RETENTION.md`.
  - На текущем наборе `_runs` `text_processor` артефактов не найдено, поэтому это пока “risk by design”, а не подтверждение на run.

---

### 9) Audit Questions (если всплывают неоднозначности)

Добавлять сюда только вопросы, которые блокируют аудит/рефактор и не решаются из текущих `docs/*`.

---

### 10) Нереализованный функционал / недостающие интеграции (по документации)

Цель: зафиксировать **что обещано в `docs/*`, но сейчас отсутствует/частично реализовано** в DataProcessor (и смежных процессорах), чтобы это стало backlog'ом реализации.

#### 10.1 DataProcessor-level Dynamic Batching (интеграция обязательна, даже если scheduler “выше”)

Основание: `GLOBAL.md` (Backend/DataProcessor-level batching), `PRODUCTION_ARCHITECTURE.md` (DataProcessor-level batching), `MODEL_SYSTEM_RULES.md` (OOM/batch_size policy), `ORCHESTRATION_AND_CACHING.md` (orchestrator — источник истины по графу).
Детализация/контракт (в разработке): `DynamicBatching_Q_A.md`.

Текущий статус: **FAIL/Gap**
- Сейчас есть только coarse GPU-gating (`gpu_max_concurrent` / semaphore) и фиксированный запуск компонентов.
- Нет ресурсно‑осознанного планирования, нет batch-size адаптации, нет cross-video batching.
- Нет контрактов/интерфейсов на уровне DataProcessor, чтобы “верхний” scheduler мог управлять батчингом компонентов.

Что должно появиться в DataProcessor (ожидаемая интеграция):
- **Компонентный API “batchable”**: компонент принимает batch единиц работы (пачка кадров/клипов/чанков) и возвращает результаты с явным mapping на исходные элементы.
- **Ресурсный профиль компонента**: память/время на unit, поддерживаемые `engine/precision/device`, max/min batch, политика OOM‑retry деградации batch_size.
- **Планировщик на уровне run**: формирование batch’ей для GPU‑heavy компонентов на основе доступных ресурсов (в т.ч. одновременное исполнение разных компонентов).
- **Контракт OOM‑retry**: при OOM допускается уменьшение batch_size и retry, но без “fallback модели”.
- **Наблюдаемость**: `manifest.json`/NPZ meta фиксируют фактический `batch_size`, `engine/precision/device`, `error_code` (включая OOM) и статистику (retries/cache_hit).

Зафиксировано (Round 1, по `DynamicBatching_Q_A.md`):
- **MVP scope**: batchable должны быть **все компоненты, где это имеет смысл** (включая CPU‑алгоритмы).
- **Иерархия batching**: multi-level batching (уровни 1..4). Batch level 1 формирует верхний scheduler; оркестратор/компоненты **не меняют** этот batch, но репортят фактические метрики для автокоррекции чек‑листа.
- **OOM policy**: 3 попытки уменьшения batch_size, последняя — `batch_size=1`; **без деградации данных**; **cuda→cpu запрещено** (no-fallback).
- **Prod buffering**: backend buffer window = **10 секунд**, пользователь при этом видит прогресс “в очереди”.

Примечание (doc-sync):
- Это уточнение по OOM (3 попытки, без деградации данных) может конфликтовать с текущим текстом `ERROR_HANDLING_AND_EDGE_CASES.md` (где описаны другие параметры). Требуется синхронизация документации после утверждения правил.

Зафиксировано дополнительно (Round 2, по `DynamicBatching_Q_A.md`):
- **Cross-video batching**: **разрешён** (смешиваем разные `video_id` в одном batch), но только при строгих constraints (одинаковый `component_name`, `model_signature`, preprocessing/`analysis_*`, resolution bucket). Max videos per batch — динамический по ресурсам, но нужен safety hard cap.
- **Координация зависимостей между процессорами/модулями**: через **внешний state-file прогресса**; модули должны уметь “ждать dependency” (например OCR) не блокируя обработку других видео.

#### 10.2 Required vs optional (fail-fast policy) на уровне профиля анализа

Основание: `ORCHESTRATION_AND_CACHING.md`, `PRODUCTION_ARCHITECTURE.md` (profile_components.required).

Текущий статус: **FAIL/Gap**
- В baseline orchestrator нет формального профиля анализа (required/optional per component).
- Audio/Text запускаются best-effort (`check=False`) вместо “все enabled — required по умолчанию”.

#### 10.3 Idempotency / кэш‑реюз с TTL (до запуска компонентов)

Основание: `ORCHESTRATION_AND_CACHING.md`, `MODEL_SYSTEM_RULES.md`, `PRODUCTION_ARCHITECTURE.md` (`cache_ttl_days`).

Текущий статус: **FAIL/Gap**
- `config_hash` считается, но до запуска не делается систематический поиск валидных артефактов по idempotency key + TTL.
- Нет корректного ключа, т.к. отсутствуют `models_used[]/model_signature`.

#### 10.4 Model system integration: `models_used[]` / `model_signature` / resolved mapping (DB source-of-truth)

Основание: `MODEL_SYSTEM_RULES.md`, `PRODUCTION_ARCHITECTURE.md`.

Текущий статус: **FAIL/Gap**
- В NPZ meta отсутствуют `models_used[]`, `model_signature`, `engine/precision/device`.
- В `manifest.json` отсутствует resolved mapping per-run и компоненты не кодируют модельные версии.
- В baseline DataProcessor нет интеграции с DB‑профилями (`analysis_profiles/profile_components/profile_model_mapping`) как источником mapping.

#### 10.5 Manifest schema completeness (per-component `device_used`, `error_code`, warnings)

Основание: `ARTIFACTS_AND_SCHEMAS.md`, `MODEL_SYSTEM_RULES.md`.

Текущий статус: **FAIL/Gap**
- `RunManifest/ManifestComponent` не поддерживает ряд обязательных полей, и компонентам негде это писать единообразно.

#### 10.6 Retention/GC + delete request (end-to-end)

Основание: `PRIVACY_AND_RETENTION.md`, `ORCHESTRATION_AND_CACHING.md`, `MODEL_SYSTEM_RULES.md`.

Текущий статус: **FAIL/Gap**
- Нет реализованного GC по TTL (frames_dir=7d, hard_cap_days=60) и механизма delete request по `video_id`.
- Нет явного policy‑гейта на сохранение raw (особенно Text/OCR).

#### 10.7 Product constraints enforcement (5s..20min, downscale>1080p, cleanup)

Основание: `PRODUCT_CONTRACT.md`.

Текущий статус: **FAIL/Gap**
- Нет централизованной валидации входа и enforcement лимитов на уровне DataProcessor до запуска стадий.

---

#### 10.8 Queue-based execution + статус прогресса + webhooks + health endpoints (интеграция DataProcessor ↔ backend)

Основание: `PRODUCTION_ARCHITECTURE.md`, `GLOBAL.md`.

Текущий статус: **FAIL/Gap**
- Нет интеграции с job queue (Redis/RabbitMQ/Celery/RQ) как основного пути запуска.
- Нет контракта/реализации обновления статуса прогресса через `/api/runs/{run_id}/status` (через БД) и/или событий (webhook `callback_url`).
- Нет health endpoints для worker: `/health` (readiness) и `/health/live` (liveness) с проверками (queue, MinIO, Triton если нужен).
- Нет минимального набора метрик (Prometheus/Grafana) для `queue_length`, `component_duration`, `errors_total{type=...}` и т.д.

#### 10.9 State/progress file (координация зависимостей + источник прогресса для UI)

Основание: `DynamicBatching_Q_A.md` (Round 2), `PRODUCTION_ARCHITECTURE.md` (progress/status), `GLOBAL.md` (polling/webhooks/manifest).

Текущий статус: **FAIL/Gap**
- Требуется внешний файл состояний (per video/run), на который опираются процессоры/модули для проверки зависимостей (например OCR) и который может использоваться для UI прогресса.
- Сейчас есть только `manifest.json`, но он не формализован как state-machine с checkpoints/ожиданиями и не покрывает нужную семантику “wait for dependency then continue”.
- Нужно утвердить: отдельный `run_state.json` vs расширение `manifest.json`, схему, частоту обновлений, конкурентную запись (lock vs last-write-wins).

Update (Dynamic Batching Q&A):
- **Round 3**: принято решение, что state-file будет **отдельным** (`run_state.json` рядом с `manifest.json`) — см. `DynamicBatching_Q_A.md`.
- В Round 4 требуется финализировать: модель конкурентной записи (single file vs leaf files vs многоуровневые state-files) и семантику priority/зависимостей.

Update (Dynamic Batching Q&A, Round 4):
- Принято: **вариант C — многоуровневые state-files** (Level 1..4) + **state-managers** как единственные писатели state на каждом уровне.
- Принято: **priority = dependency-ordering** (глобальный DAG модулей/процессоров), чтобы строить план параллелизма.
- Принято: если dependency не пришла после ожидания + grace (10s) → **error и стоп run** (fail-fast).
- Принято: OCR handoff Visual→Text — **NPZ артефакт** (не JSON).
- Принято: MVP UI читает прогресс только через **backend API**.

Update (Dynamic Batching Q&A, Round 5):
- **Level 4 отменён**: состояние модулей/компонентов живёт как секции внутри state-file процессора (Level 3), отдельные файлы не делаем.
- **Durability**: очередь обновлений state-manager должна быть устойчивой (рекомендуемый MVP: `state_events.jsonl` + checkpoint state-file).
- **Storage**: state-files должны жить во внешнем хранилище; предлагается “local-first, upload-on-error/stop (finally)”.
- **Fail-fast**: missing dependency → error/stop run относится к **любым** зависимостям (на текущем этапе).
- **DAG (MVP)**: `docs/reference/component_graph.yaml` как source-of-truth; baseline/v1/v2 имеют отдельные DAG.
