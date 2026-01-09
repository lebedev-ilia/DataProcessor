# PR‑4 — Required vs Optional profiles (MVP)

Цель: формализовать **required/optional** семантику компонентов, чтобы:
- required падение ⇒ **stop run** (fail-fast),
- optional падение ⇒ компонент `error` в manifest, но run может продолжить.

## 1) Profile YAML (MVP)

Профиль живёт в `profiles/*.yaml` и задаёт:
- какие процессоры включены (`audio/text`),
- какие компоненты VisualProcessor required/optional (`visual.requirements`).

Пример: `profiles/pr4_smoke.yaml`.

## 2) Enforcement (MVP)

- **VisualProcessor**: если в конфиге есть `requirements{component: bool}`, то:
  - для `required=true`: `status="error"` ⇒ VisualProcessor завершает процесс с exit code != 0.
  - `status="empty"` не считается ошибкой (валидная пустота).
- **Root orchestrator (`main.py`)**:
  - при `processors.audio.required=true`: non-zero exit от AudioProcessor ⇒ stop run
  - при `processors.text.required=true`: non-zero exit от TextProcessor ⇒ stop run
  - при `required=false`: ошибки фиксируются в manifest, но run продолжается.

## 3) Быстрый smoke

```bash
cd "/home/ilya/Рабочий стол/TrendFlowML/DataProcessor"
./VisualProcessor/.vp_venv/bin/python main.py \
  --profile-path ./profiles/pr4_smoke.yaml \
  --video-path ./NSumhkOwSg.mp4 \
  --output ./_runs/segmenter_out \
  --rs-base ./_runs/result_store \
  --platform-id youtube \
  --video-id NSumhkOwSg \
  --run-id pr4smoke \
  --sampling-policy-version v1 \
  --dataprocessor-version unknown \
  --analysis-fps 30 --analysis-height 320 --analysis-width 568
```


