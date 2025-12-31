# DataProcessor — документация (полуфинальные контракты)

Эта папка — “свод правил” проекта TrendFlow/DataProcessor, собранный из `VisualProcessor/docs/project_questions.md`.

## Что где лежит

- `BASELINE_IMPLEMENTATION_PLAN.md` — единый подробный пошаговый план работ до baseline (по всем контрактам).
- `CONTRACTS_OVERVIEW.md` — короткая карта всех ключевых контрактов и терминов.
- `ARTIFACTS_AND_SCHEMAS.md` — NPZ, meta-поля, версии, валидатор, структура `result_store`, `manifest.json`.
- `SEGMENTER_CONTRACT.md` — time-domain семплинг, budgets per component, union кадры, RGB контракт.
- `ORCHESTRATION_AND_CACHING.md` — DAG, required/optional (fail-fast vs best-effort), idempotency, artifact index.
- `ML_TARGETS_AND_TRAINING.md` — multi-target/multi-horizon, delta targets, split, cold-start stratification, reproducibility.
- `LLM_RENDERING.md` — LLM как presentation layer, воспроизводимость, кэширование, guardrails.
- `PRIVACY_AND_RETENTION.md` — хранение raw текста/комментов, OAuth-верификация владельца канала, retention caps.

## Связанные документы внутри процессоров

- `VisualProcessor/docs/MODULE_STANDARDS.md` — стандарты реализации модулей (BaseModule, sampling contract, no-fallback, empty outputs).
- `VisualProcessor/docs/project_questions.md` — исходный Q&A (рабочий документ).


