# TextProcessor

TextProcessor — процессор текстовой модальности. Он извлекает табличные **snapshot‑фичи** из доступного текстового входа (title/description/transcript/comments) и сохраняет результат в **per‑run `result_store`**.

## Контракт входа

- **Единица обработки**: один `VideoDocument` (JSON).
- **Источник**: upstream (backend/ingestion) формирует `VideoDocument` и передаёт путь в CLI флагом `--input-json`.
- **Требования (no‑fallback)**:
  - Если TextProcessor включён в профиль как required, отсутствие входного JSON или ошибка парсинга → run должен падать на уровне orchestrator.
  - Модели/эмбеддеры должны грузиться **только локально** (no‑network policy).

### Формат `VideoDocument`

См. `TextProcessor/src/schemas/models.py`.

Ключевые поля:
- `title: str`
- `description: str`
- `transcripts: Dict[str, str]` (например `{"whisper": "...", "youtube_auto": "..."}`)
- `transcripts_token_ids: Dict[str, List[int]]` (опционально; предпочтительно вместо raw transcript)
- `comments: List[{text: str}]` (опционально)

## Контракт выхода (result_store)

TextProcessor пишет **один NPZ артефакт**:

- `result_store/<platform_id>/<video_id>/<run_id>/text_processor/text_features.npz`

и апдейтит:

- `result_store/<platform_id>/<video_id>/<run_id>/manifest.json`

### NPZ schema

Схема: `schema_version="text_npz_v1"` (см. `docs/contracts/ARTIFACTS_AND_SCHEMAS.md`).

Минимальные ключи:
- `feature_names: object[str]`
- `feature_values: float32[]`
- `payload: object(dict)` — **privacy‑safe summary**, без raw текста по умолчанию
- `meta: object(dict)` — run identity + версии + статус

### Privacy / raw текст

По умолчанию TextProcessor **не сохраняет raw текст** в NPZ.

Для локального дебага есть флаг:
- `--store-raw-payload` → пишет raw payload в `result_store/.../_tmp_text/raw_payload.json` (не source‑of‑truth).

## Запуск (CLI)

Standalone (в локальный `_runs/result_store`):

```bash
python3 TextProcessor/run_cli.py \
  --input-json /path/to/video_document.json \
  --platform-id youtube \
  --video-id <video_id> \
  --run-id <run_id> \
  --sampling-policy-version v1 \
  --dataprocessor-version unknown
```

Embedding‑ветка (тяжелее, может требовать GPU):

```bash
python3 TextProcessor/run_cli.py \
  --input-json /path/to/video_document.json \
  --enable-embeddings
```

## Политика моделей (ModelManager)

TextProcessor использует единый `ModelManager` (`dp_models`) и работает в режиме **no‑network**:

- модели SentenceTransformers должны быть описаны в `dp_models/spec_catalog/text/*.yaml`
- артефакты должны лежать в `DP_MODELS_ROOT` (см. `env.example`)

Текущий локальный embedding‑модельный дефолт: `sentence-transformers/all-MiniLM-L6-v2`.


