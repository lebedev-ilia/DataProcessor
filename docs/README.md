# Docs index

- Components index: `docs/COMPONENTS_INDEX.md`
- Baseline models (CPU vs GPU): `docs/models_docs/BASELINE_MODELS.md`
- Baseline GPU branches (fixed-shape) + Triton plan: `docs/models_docs/BASELINE_GPU_BRANCHES.md`
# DataProcessor — документация (полуфинальные контракты)

Эта папка — “свод правил” проекта TrendFlow/DataProcessor, собранный из `docs/reference/project_questions.md`.

## Что где лежит

- **baseline**:
  - `docs/baseline/BASELINE_IMPLEMENTATION_PLAN.md` — единый подробный пошаговый план работ до baseline (по всем контрактам).
  - `docs/baseline/BASELINE_RUN_CHECKLIST.md` — короткий чеклист инвариантов перед первым прогоном baseline.
  - `docs/baseline/BASELINE_TO_TRAINING_ROADMAP.md` — roadmap baseline → training.
  - `docs/baseline/BASELINE_PRODUCTION_EXECUTION_PLAN.md` — prod-run execution plan.
  - `docs/baseline/ML_TARGETS_AND_TRAINING.md` — multi-target/multi-horizon, delta targets, split, reproducibility.
  - `docs/baseline/REMAINING_BASELINE_TASKS.md` — текущий backlog baseline задач.
- **contracts**:
  - `docs/contracts/CONTRACTS_OVERVIEW.md`
  - `docs/contracts/ARTIFACTS_AND_SCHEMAS.md`
  - `docs/contracts/SEGMENTER_CONTRACT.md`
  - `docs/contracts/ORCHESTRATION_AND_CACHING.md`
  - `docs/contracts/ERROR_HANDLING_AND_EDGE_CASES.md`
  - `docs/contracts/PRODUCT_CONTRACT.md`
  - `docs/contracts/LLM_RENDERING.md`
  - `docs/contracts/PRIVACY_AND_RETENTION.md`
  - `docs/contracts/PER_COMPONENT.md`
- **models**:
  - `docs/models_docs/BASELINE_MODELS.md` — baseline models (CPU vs GPU).
  - `docs/models_docs/MODEL_SYSTEM_RULES.md`
  - `docs/models_docs/FEATURE_ENCODER_CONTRACT.md` — контракт encoder’а, который приводит variable-length фичи компонентов к fixed-size для transformer’ов
  - `docs/models_docs/MODEL_MANAGER_PLAN.md`
  - `docs/models_docs/MODEL_INVENTORY.md`
  - `docs/models_docs/MODEL_LICENSES.md`
  - `docs/models_docs/MODELS_Q.md`
  - `docs/models_docs/DynamicBatching_Q_A.md`
- **architecture**:
  - `docs/architecture/PRODUCTION_ARCHITECTURE.md`
  - `docs/architecture/BILLING_AND_PRICING.md`
- **prs**:
  - `docs/prs/PR0_LOCAL_STACK.md`
  - `docs/prs/PR1_STORAGE_ADAPTER.md`
  - `docs/prs/PR2_1_ENVIRONMENTS.md`
  - `docs/prs/PR4_REQUIRED_OPTIONAL_PROFILES.md`
  - `docs/prs/PR5_STATE_FILES_AND_MANAGERS.md`
  - `docs/prs/PR6_DAG_RUNNER.md`
  - `docs/prs/PR7_CELERY_AND_HEALTH.md`
  - `docs/prs/PR8_TRITON_INTEGRATION.md`
  - `docs/prs/PR9_MODEL_OPTIMIZATIONS.md`
  - `docs/prs/PR10_MODULE_AUDIT_SPLIT.md`
- **audits**:
  - `docs/audits/DATAPROCESSOR_AUDIT.md` — единый чеклист полного аудита DataProcessor.
- **reference**:
  - `docs/reference/component_graph.yaml` — декларативный DAG компонентов.
  - `docs/reference/GLOBAL.md` — общий большой “комбайн” (исторически).
  - `docs/reference/project_questions.md` — исходный Q&A (рабочий документ).
- **site**:
  - `docs/site/SITE_DESIGN.md`
  - `docs/site/SITE_SPECIFICATION.md`
  - `docs/site/SITE_Q.md`

## Связанные документы внутри процессоров

- `VisualProcessor/docs/MODULE_STANDARDS.md` — стандарты реализации модулей (BaseModule, sampling contract, no-fallback, empty outputs).
- `docs/reference/project_questions.md` — исходный Q&A (рабочий документ).

## Быстрый ориентир по CLI (baseline v0)

В репозитории есть “орchestrator entrypoint”:
- `DataProcessor/main.py` — запускает Segmenter → (опционально) Audio/Text → Visual.

Под-процессоры:
- `Segmenter/segmenter.py`
- `VisualProcessor/main.py`
- `AudioProcessor/run_cli.py`
- `TextProcessor/run_cli.py` (baseline-safe CPU-only по умолчанию)


