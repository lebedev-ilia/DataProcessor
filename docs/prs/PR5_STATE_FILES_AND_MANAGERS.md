# PR‑5 — State-files + state-manager (MVP)

Цель PR‑5: добавить **state-files** (отдельно от `manifest.json`) и **state-manager**, чтобы backend мог получать прогресс детерминированно и без “угадываний” по логам.

## 1) Где лежат state-files

Каноничный layout (см. `storage/paths.py`):

- `state/<platform_id>/<video_id>/<run_id>/run_state.json` (Level‑2, агрегированное состояние run)
- `state/<platform_id>/<video_id>/<run_id>/state_<processor>.json` (Level‑3, владелец = процессор)
- `state/<platform_id>/<video_id>/<run_id>/state_events.jsonl` (append‑only журнал событий)

В MVP пишем state в filesystem под `_runs/state/...` (рядом с `_runs/result_store/...`).

## 2) Статусы

Enum (см. `state/enums.py`):
- `waiting`, `running`, `success`, `empty`, `error`, `skipped`

## 3) Что пишет root orchestrator

Root `main.py` (DataProcessor) в MVP:
- создаёт `run_state.json` и `state_<processor>.json` для: `segmenter`, `audio`, `text`, `visual`
- обновляет их вокруг subprocess‑запусков
- после VisualProcessor — читает `manifest.json` и копирует статусы компонентов в `state_visual.json` (Level‑4 секция `components{}` внутри файла VisualProcessor)

## 4) Smoke run

```bash
cd "/home/ilya/Рабочий стол/TrendFlowML/DataProcessor"
./VisualProcessor/.vp_venv/bin/python main.py \
  --video-path "./NSumhkOwSg.mp4" \
  --rs-base "./_runs/result_store" \
  --output "./_runs/segmenter_out" \
  --platform-id youtube \
  --video-id NSumhkOwSg \
  --run-id pr5smoke \
  --sampling-policy-version v1 \
  --dataprocessor-version unknown \
  --analysis-fps 30 --analysis-height 320 --analysis-width 568 \
  --visual-cfg-path "./VisualProcessor/config_pr2_min.yaml"
```

Проверить:
- `_runs/state/youtube/NSumhkOwSg/pr5smoke/run_state.json`
- `_runs/state/youtube/NSumhkOwSg/pr5smoke/state_events.jsonl`
- `_runs/state/youtube/NSumhkOwSg/pr5smoke/state_visual.json`


