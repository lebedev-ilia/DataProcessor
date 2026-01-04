# DataProcessor — документация (полуфинальные контракты)

Эта папка — “свод правил” проекта TrendFlow/DataProcessor, собранный из `VisualProcessor/docs/project_questions.md`.

## Что где лежит

- `BASELINE_IMPLEMENTATION_PLAN.md` — единый подробный пошаговый план работ до baseline (по всем контрактам).
- `DATAPROCESSOR_AUDIT.md` — единый чеклист полного аудита DataProcessor (критерии + параметры проверки по всем процессорам).
- `CONTRACTS_OVERVIEW.md` — короткая карта всех ключевых контрактов и терминов.
- `ARTIFACTS_AND_SCHEMAS.md` — NPZ, meta-поля, версии, валидатор, структура `result_store`, `manifest.json`.
- `SEGMENTER_CONTRACT.md` — time-domain семплинг, budgets per component, union кадры, RGB контракт.
- `ORCHESTRATION_AND_CACHING.md` — DAG, required/optional (fail-fast vs best-effort), idempotency, artifact index.
- `PRODUCT_CONTRACT.md` — продуктовые контракты (MVP UX, профили анализа/конфиги, LLM как presentation layer, платформа v1, валидация входных данных).
- `PRODUCTION_ARCHITECTURE.md` — целевая архитектура сервиса (backend/worker/storage/DB/Triton/LLM, коммуникация, батчинг, масштабирование, мониторинг, безопасность).
- `BILLING_AND_PRICING.md` — биллинг и ценообразование (единица биллинга, списание кредитов, прайс-лист, оценка стоимости).
- `ERROR_HANDLING_AND_EDGE_CASES.md` — обработка ошибок, retry политики, edge cases (повреждённые файлы, видео без звука, таймауты).
- `ML_TARGETS_AND_TRAINING.md` — multi-target/multi-horizon, delta targets, split, cold-start stratification, reproducibility.
- `MODEL_SYSTEM_RULES.md` — канонические правила по моделям (model_signature/models_used, Triton mapping, кэш, детерминизм, multi-GPU, prediction fallback, observability).
- `MODEL_LICENSES.md` — инвентарь моделей и лицензий (template).
- `LLM_RENDERING.md` — LLM как presentation layer, воспроизводимость, кэширование, guardrails.
- `PRIVACY_AND_RETENTION.md` — хранение raw текста/комментов, OAuth-верификация владельца канала, retention caps.
- `SITE_DESIGN.md` — дизайн и UX сайта (цветовая палитра, структура страниц, компоненты, адаптивность).
- `BASELINE_RUN_CHECKLIST.md` — короткий чеклист инвариантов перед первым прогоном baseline.

## Связанные документы внутри процессоров

- `VisualProcessor/docs/MODULE_STANDARDS.md` — стандарты реализации модулей (BaseModule, sampling contract, no-fallback, empty outputs).
- `VisualProcessor/docs/project_questions.md` — исходный Q&A (рабочий документ).

## Быстрый ориентир по CLI (baseline v0)

В репозитории есть “орchestrator entrypoint”:
- `DataProcessor/main.py` — запускает Segmenter → (опционально) Audio/Text → Visual.

Под-процессоры:
- `Segmenter/segmenter.py`
- `VisualProcessor/main.py`
- `AudioProcessor/run_cli.py`
- `TextProcessor/run_cli.py` (baseline-safe CPU-only по умолчанию)


