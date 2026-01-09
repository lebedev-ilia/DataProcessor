# PR‑6 — DAG runner (`component_graph.yaml`) + dependency-ordering

Цель: сделать порядок исполнения детерминированным и управляемым через декларативный DAG.

## 1) Source-of-truth DAG

Файл: `docs/reference/component_graph.yaml`

В PR‑6 заполнен `stages.baseline.nodes` минимально для Tier‑0 (Visual + Audio) и `segmenter`.

## 2) Парсер/валидатор DAG

Код: `dag/component_graph.py`

Проверки MVP:
- уникальность `component_name`
- все `depends_on_components` существуют в выбранном stage
- граф ацикличен
- deterministic topo order

## 3) Как DAG влияет на runtime

Root `main.py`:
- читает DAG (`--dag-path`, `--dag-stage`)
- строит `execution_order` для VisualProcessor (только из **включенных** компонентов визуального YAML)
- передает `execution_order` в runtime cfg

VisualProcessor:
- если `execution_order` присутствует — выполняет компоненты **последовательно** в этом порядке
- компоненты, которых нет в DAG stage, но они включены в YAML, в MVP выполняются **после** DAG-ordered списка (с warning)
- иначе работает как раньше

## 4) Smoke

```bash
cd "/home/ilya/Рабочий стол/TrendFlowML/DataProcessor"
./VisualProcessor/.vp_venv/bin/python main.py \
  --dag-stage baseline \
  --video-path "./NSumhkOwSg.mp4" \
  --output "./_runs/segmenter_out" \
  --rs-base "./_runs/result_store" \
  --platform-id youtube \
  --video-id NSumhkOwSg \
  --run-id pr6smoke \
  --sampling-policy-version v1 \
  --dataprocessor-version unknown \
  --analysis-fps 30 --analysis-height 320 --analysis-width 568 \
  --visual-cfg-path "./VisualProcessor/config_pr4_optional_fail.yaml"
```

Ожидаемо:
- `manifest.json.components[].started_at` отражает порядок DAG
- в state (`run_state.json` / `state_visual.json`) видно `started_at/finished_at/duration_ms` per component


