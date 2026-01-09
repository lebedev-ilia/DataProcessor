
# Model Manager Q&A (Round 0 → implementation)

Формат работы:
- Я добавляю вопросы и предлагаю дефолт/варианты.
- Ты отвечаешь **прямо под каждым вопросом** (коротко, но однозначно).
- После ответов я реализую полноценный `ModelManager` в коде + документацию.

Основание (канон):
- `docs/models_docs/MODEL_SYSTEM_RULES.md` (model_signature, models_used, no-fallback, no-network)
- `docs/contracts/ORCHESTRATION_AND_CACHING.md` (idempotency key)
- `docs/architecture/PRODUCTION_ARCHITECTURE.md` (profiles, resolved mapping, services)
- `docs/prs/PR8_TRITON_INTEGRATION.md` (resolved_model_mapping, Triton client)
- `docs/prs/PR9_MODEL_OPTIMIZATIONS.md` (engine=onnx, weights_digest)
- `docs/contracts/ARTIFACTS_AND_SCHEMAS.md` (meta contract)

---

## Round 0 — “что такое ModelManager” (границы ответственности)

### Q0.1. ModelManager живёт где?
Варианты:
- A) внутри DataProcessor (root package) и импортируется всеми процессорами (Visual/Audio/Text)
- B) отдельный общий модуль (например `common/model_manager.py`) и используется всеми

**Мой дефолт**: B (единый для всех, но без привязки к Visual-only).

**A**: Согласен, делаем единую большую систему со всеми моделями для всех

### Q0.2. Что входит в обязанности ModelManager (MVP)?
Подтверди/добавь пункты:
- resolve per-run model mapping (component → runtime/engine/precision/device/paths/triton endpoints)
- enforce **no-network** (никаких `from_pretrained("openai/...")`, никаких `load_state_dict_from_url`, никаких `timm(pretrained=True)`)
- enforce **no-fallback** (если выбранная модель недоступна — fail-fast)
- compute `weights_digest` (sha256 локального файла или “unknown” если невозможно/нефайл)
- build canonical `models_used[]` entries + compute `model_signature`
- return “runtime handle” для компонента:
  - triton: параметры клиента/inputs/outputs/dtypes + endpoint
  - onnx: путь к onnx + runtime options
  - torch: (если вообще допускаем) путь к весам/параметры
- optional: preload Tier-0 models (если inprocess), health checks

**A**: согласен

---

## Round 1 — Источник правды: mapping, run manifests, профили

### Q1.1. Откуда приходит resolved mapping в DataProcessor (MVP сейчас)?
Варианты:
- A) только из `--profile-path` YAML (как сейчас)
- B) из БД (backend), а YAML только dev seed
- C) гибрид: если есть mapping в payload — используем его, иначе читаем YAML

**Мой дефолт**: C (чтобы миграция на backend была без ломки API).

**A**: Согласен с дефолтом

### Q1.2. Где мы сохраняем resolved mapping для воспроизводимости?
Варианты:
- A) только `manifest.json.run.resolved_model_mapping`
- B) ещё и в каждом NPZ `meta.resolved_model_mapping` (дублирование)
- C) в `manifest.json` + в NPZ только `models_used[]/model_signature`

**Мой дефолт**: C (NPZ держим компактным; manifest — source-of-truth per run).

**A**: Согласен с дефолтом

### Q1.3. Версионирование внутри mapping
Подтверди, что для каждой модели/компонента в mapping **обязательно** указывать:
- `model_version` (pinned)
- `weights_digest` (sha256 или иной стабильный идентификатор)
- `engine`, `precision`, `runtime`, `device_policy`

**A**: Согласен

---

## Round 2 — Canonical schema для ModelManager (какой “resolved config” он возвращает)

### Q2.1. Единая структура “ResolvedModelSpec” (предложение)
Подтверди/исправь поля (минимум):
- `component_name`
- `runtime`: `triton` | `inprocess`
- `engine`: `onnx` | `torch` | `tensorrt`
- `precision`: `fp16` | `fp32`
- `device`: `cuda` | `cuda:0` | `cpu` (фактическое выбранное, не policy)
- `model_name`, `model_version`, `weights_digest`
- `artifacts`:
  - for onnx: `onnx_path`
  - for torch: `weights_path` (если допустим)
  - optional: `tokenizer_path`, `preprocess_id`
- `triton` (если runtime=triton):
  - `http_url`
  - `model_name`
  - `model_version` (triton version)
  - `inputs[]` (name, dtype, shape)
  - `outputs[]` (name, dtype, shape)
  - optional: `preprocess_preset`

**A**: Согласен

### Q2.2. Нужен ли ModelManager-уровень “model roles”?
Например, `core_clip` имеет 2 роли: `image_encoder`, `text_encoder`.
Варианты:
- A) да, ModelManager должен поддерживать `component_name + role` (1 компонент → несколько моделей)
- B) нет, это остаётся внутри компонента (как сейчас), ModelManager даёт один “пакет” настроек на компонент

**Мой дефолт**: A (иначе core_clip/core_optical_flow/и т.п. будут каждый по-своему решать).

**A**: Согласен с дефолтом

---

## Round 3 — No-network enforcement (как гарантируем, что код не скачает ничего)

### Q3.1. Стратегия запрета сети
Варианты:
- A) только “политика” + code review (мягко)
- B) runtime guard: ModelManager включает “offline mode” (env vars + monkeypatch safeguards) и валит run при попытке скачать
- C) оба: B + статический grep/CI чек

**Мой дефолт**: C.

**A**: Согласен с дефолтом

### Q3.2. Что считать “сетевой попыткой”?
Подтверди список библиотек/вызовов, которые должны быть запрещены в prod runtime:
- `requests.get/httpx.get` к внешним доменам
- `torch.hub.load_state_dict_from_url`
- `transformers.from_pretrained` с remote id без локальных файлов
- `timm.create_model(pretrained=True)`
- `clip.load()` если веса отсутствуют локально

**A**: Да

---

## Round 4 — weights_digest (как считаем, где храним)

### Q4.1. Формат `weights_digest`
Варианты:
- A) голый hex sha256
- B) строка вида `sha256:<hex>`

**Мой дефолт**: B (самоописываемо).

**A**: Согласен с дефолтом

### Q4.2. Как считать digest для Triton?
Варианты:
- A) digest хранится в mapping (source-of-truth) и ModelManager не проверяет
- B) ModelManager может (опционально) проверять digest локального model repo (если доступен на диске)

**Мой дефолт**: A (Triton repo может быть на другом хосте; проверка отдельной процедурой деплоя).

**A**: Согласен с дефолтом

---

## Round 5 — Model signature, models_used[] и связь с кэшем

### Q5.1. Где вычисляем `model_signature`?
Варианты:
- A) ModelManager всегда возвращает `models_used[]` и `model_signature` для компонента
- B) компоненты сами, ModelManager только отдаёт spec

**Мой дефолт**: A (единая канонизация, меньше расхождений).

**A**: Согласен с дефолтом

### Q5.2. Что включаем в model_signature кроме models_used?
Подтверди правило: **engine/precision/device/runtime** включены (как в `docs/models_docs/MODEL_SYSTEM_RULES.md`).

**A**: Да

---

## Round 6 — Device policy (multi-GPU)

### Q6.1. Кто выбирает фактический GPU (`cuda:0/1/...`)?
Варианты:
- A) внешний scheduler (в будущем DynamicBatching) и передаёт `device` в resolved mapping
- B) ModelManager выбирает сам (least-loaded)
- C) компонент выбирает сам

**Мой дефолт**: A (источник правды — верхний scheduler).

**A**: Согласен с дефолтом

### Q6.2. Фиксация device в meta
Подтверди: в `models_used[].device` пишем `cuda`/`cuda:0` (как договоримся) и это влияет на `model_signature`.

**A**: Да

---

## Round 7 — Preprocessing как часть модели

### Q7.1. Preprocessing где живёт?
Варианты:
- A) всегда внутри Triton (ensemble), а ModelManager хранит только `preprocess_preset` id
- B) может быть локально, но тогда preprocessing_version обязателен и входит в `model_signature`

**Мой дефолт**: A (как ты ранее сказал “preprocess в Triton”).

**A**: Согласен с дефолтом, но если вдруг есть CPU модели то логично что пихать их в Triton мы не будем + соответственно и препроцесс тоже

### Q7.2. Нужен ли отдельный `preprocess_signature`?
Если A: то достаточно `preprocess_preset` + `model_version/weights_digest`.

**A**: достаточно `preprocess_preset` + `model_version/weights_digest`

---

## Round 8 — API ModelManager (как будут вызывать компоненты)

### Q8.1. Минимальный API (предложение)
Подтверди/поправь:
- `mm = ModelManager(run_context, resolved_model_mapping, offline=True)`
- `spec = mm.resolve(component="core_clip", role="image")`
- `client = mm.triton_client(spec)` (или `mm.get_runtime(spec)` возвращает callable)
- `models_used = mm.models_used_for(component)` (или по spec)
- `signature = mm.model_signature_for(component)`

**A**: Да, но offline=True убрать так как он и так всегда будет оффлайн, а веса должны быть заготовленны отдельно (Triton не считаеться online так как скорее всего это будет локально запущеный Docker)

### Q8.2. Где хранить run_context?
Подтверди, что run_context включает минимум: `platform_id/video_id/run_id/config_hash/sampling_policy_version/dataprocessor_version`.

**A**: Да

---

## Round 9 — Ошибки и taxonomy

### Q9.1. Набор error_code для ModelManager
Подтверди минимальный набор:
- `model_mapping_missing`
- `model_artifact_missing`
- `weights_digest_mismatch` (если включим проверку)
- `triton_unavailable`
- `triton_model_not_found`
- `triton_infer_failed`
- `onnxruntime_missing`
- `invalid_model_spec`

**A**: Да

### Q9.2. Retry policy
Подтверди: retry только для transient (например `triton_unavailable`/timeout), но не для “model not found”.

**A**: Triton timeout не нужен так как он будет запущен локально и доступен всегда (если недоступен - error)

---

## Round 10 — Где код/артефакты моделей физически лежат

### Q10.1. Local paths vs object storage
Для inprocess/onnx: откуда ModelManager берёт файлы?
- A) только локальный FS на worker (путь в mapping)
- B) может скачивать из MinIO/S3 (НО это сеть) → нужно явно разрешить/запретить

**Мой дефолт**: A (строго локально, как ты просил).

**A**: Локально (потому что некоторые модели - это целые репозитории и это неудобно держать во внешнем store)

### Q10.2. Формат путей в mapping
Варианты:
- A) абсолютные пути
- B) относительные пути от корня репо
- C) относительные от “models_root” (env var)

**Мой дефолт**: C (самый переносимый).

**A**: Согласен с дефолтом

---

## Round 11 — Licenses/compliance (минимум)

### Q11.1. ModelManager должен проверять наличие записи в `docs/MODEL_LICENSES.md`?
Варианты:
- A) да, runtime guard (fail-fast если модели нет в inventory)
- B) только CI check (runtime не трогаем)

**Мой дефолт**: B (runtime не должен парсить docs).

**A**: Согласен с дефолтом

---

## Round 12 — MVP scope “что реализуем первым”

### Q12.1. С чего начинаем в реализации ModelManager?
Предлагаю MVP-реализацию по слоям:
- Layer 1: schema + resolve + `models_used`/`model_signature` + no-network guard
- Layer 2: helpers для Triton specs (включая multi-input/multi-role)
- Layer 3: weights_digest helper (sha256 file)
- Layer 4: optional preload Tier‑0 (inprocess), health hooks

Подтверди, ок ли такой порядок.

**A**: Да


