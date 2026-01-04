# Оркестрация, DAG, кэширование (полуфинал)

## 1) Где живёт orchestrator

Полуфинал:
- Оркестратор должен быть на уровне **DataProcessor** (единый DAG для Visual/Audio/Text).
- VisualProcessor может иметь внутренний запуск компонентов, но “истинный” граф и решения required/optional — на уровне DataProcessor.

Текущее состояние (baseline v0, референс):
- `DataProcessor/main.py` — реальный orchestrator “1 видео → 1 run”.
- `VisualProcessor/main.py` — внутренний исполнитель core/providers + modules.

## 2) Required vs optional (качество)

Гибридный режим:
- **required компоненты** (training schema) → **fail-fast**
- **optional компоненты** (аналитика/доп. фичи) → **best-effort**

Полуфинальное уточнение (Round 1 + Round 2):
- На этапе разработки/обучения (baseline+v2) мы целимся в **качество и предсказуемость**, поэтому:
  - по умолчанию **все включённые в профиль анализа компоненты считаются required** (fail-fast),
  - “best-effort”/optional допустим только если компонент явно помечен как optional в профиле анализа (см. `PRODUCT_CONTRACT.md`).

**Правило partial failures (Round 2)**:
- Если компонент упал, но он не участвует в prediction schema (не является required для прогноза), то:
  - run считается `status="ok"` (если все required компоненты успешны),
  - component_status="error" для упавшего компонента,
  - в UI показывается предупреждение о частичной ошибке,
  - prediction всё равно выдаётся (если required компоненты ok).

## 3) Idempotency ключ

Компонент должен быть идемпотентным по ключу:
- `(platform_id, video_id, component, config_hash, sampling_policy_version, producer_version, schema_version, model_signature)`

Где `model_signature` — функция от реально использованных моделей/версий/весов + engine/precision/device (см. `docs/MODEL_SYSTEM_RULES.md`).

Если артефакт по ключу уже существует и валиден — пересчёт не делаем.

## 4) Задачи (Celery/очереди)

Рекомендация по этапам:
- старт: “1 видео = 1 job” (внутри job выполняем DAG по компонентам)
- масштабирование: “1 компонент = 1 task” (больше параллелизма, но сложнее кэш)

Параллелизм внутри одного видео (текущий baseline v0):
- Модули VisualProcessor могут выполняться параллельно.
- GPU-задачи ограничиваются `gpu_max_concurrent` (по умолчанию `auto` → 1 на малой VRAM).
- Manifest апдейтится последовательно из оркестратора, чтобы избежать гонок записи.

## 5) Artifact index (кэш по видео)

Нужен быстрый индекс (manifest/таблица) по ключу:
- `(platform_id, video_id, config_hash, sampling_policy_version, dataprocessor_version)`
→ ссылка на `latest_success_run_id` и артефакты.

Примечание:
- `dataprocessor_version` — версия кода пайплайна, а модельная часть кэша обеспечивается через `model_signature` на уровне компонентов (см. `docs/MODEL_SYSTEM_RULES.md`).

Политика кэша “последние 10k видео” трактуется как **10k уникальных video_id** (heavy compute слой).

Полуфинальное уточнение (Round 1):
- Повторный запрос того же видео в проде должен учитывать “возраст анализа”.
- Рекомендуемая политика: если последний успешный run свежее `cache_ttl_days` (дефолт 3 дня, настраивается) — используем кэш, иначе пересчитываем.

## 6) Наблюдаемость

Минимум:
- timings per component
- GPU/CPU mem (если есть)
- status ok/empty/error
- причины empty (empty_reason)

Эти данные пишем в `manifest.json` и/или в БД.

## 7) Frames_dir retention

Полуфинал (Round 1):
- Union-кадры (`frames_dir`) храним **7 дней** для дебага/повторного рендера, затем удаляем.


