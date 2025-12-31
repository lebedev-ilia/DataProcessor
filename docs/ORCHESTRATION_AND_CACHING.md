# Оркестрация, DAG, кэширование (полуфинал)

## 1) Где живёт orchestrator

Полуфинал:
- Оркестратор должен быть на уровне **DataProcessor** (единый DAG для Visual/Audio/Text).
- VisualProcessor может иметь внутренний запуск компонентов, но “истинный” граф и решения required/optional — на уровне DataProcessor.

## 2) Required vs optional (качество)

Гибридный режим:
- **required компоненты** (training schema) → **fail-fast**
- **optional компоненты** (аналитика/доп. фичи) → **best-effort**

## 3) Idempotency ключ

Компонент должен быть идемпотентным по ключу:
- `(platform_id, video_id, run_id, component, config_hash, sampling_policy_version, producer_version, schema_version, model_version*)`

Если артефакт по ключу уже существует и валиден — пересчёт не делаем.

## 4) Задачи (Celery/очереди)

Рекомендация по этапам:
- старт: “1 видео = 1 job” (внутри job выполняем DAG по компонентам)
- масштабирование: “1 компонент = 1 task” (больше параллелизма, но сложнее кэш)

## 5) Artifact index (кэш по видео)

Нужен быстрый индекс (manifest/таблица) по ключу:
- `(platform_id, video_id, config_hash, sampling_policy_version, dataprocessor_version)`
→ ссылка на `latest_success_run_id` и артефакты.

Политика кэша “последние 10k видео” трактуется как **10k уникальных video_id** (heavy compute слой).

## 6) Наблюдаемость

Минимум:
- timings per component
- GPU/CPU mem (если есть)
- status ok/empty/error
- причины empty (empty_reason)

Эти данные пишем в `manifest.json` и/или в БД.


