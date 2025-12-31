# DatasetBuilder

Цель: собрать **tabular training table** из per-run артефактов (`result_store/.../manifest.json` + NPZ).

Статус: baseline v0 — собираем **фичи** из NPZ и статусы компонентов. Targets (deltas/horizons) подключаются отдельным шагом.

## Быстрый старт

Сбор фичей:

```bash
python DatasetBuilder/build_training_table.py \
  --rs-base /abs/path/to/VisualProcessor/result_store \
  --out-csv /abs/path/to/training_table.csv
```

## Что попадает в таблицу

- `platform_id`, `video_id`, `run_id`, `config_hash`, `sampling_policy_version`
- `component_status__*` (ok=1, empty=0, error=-1)
- фичи из NPZ:
  - если NPZ содержит `feature_names`/`feature_values` → развернём 1:1
  - иначе возьмём числовые поля и посчитаем агрегаты по массивам (mean/std/min/max/p50/p90)


