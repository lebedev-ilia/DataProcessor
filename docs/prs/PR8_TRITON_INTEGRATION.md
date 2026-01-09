## PR‑8: Triton integration (MVP) — client + no-fallback + resolved mapping

### Цель

Добавить **production‑baseline Triton‑интеграцию** без “fallback”:

- `component → model:version` выбирается **per‑run** через **resolved mapping** (MVP: в profile YAML; позже — из БД).
- Если Triton недоступен / модель не найдена / infer падает — **fail-fast** (required компонент → stop run).
- Воспроизводимость: resolved mapping сохраняется в `manifest.json` (`run.resolved_model_mapping`).

> Важно: этот PR можно мерджить **без локального запуска Triton** (как просил пользователь). E2E проверки будут в отдельном шаге/окружении.

---

### Что сделано (код)

- `dp_triton/`:
  - минимальный **HTTP client** (Triton HTTP v2) без внешних зависимостей
  - `ready()` для `/v2/health/ready`
  - `infer()` через JSON payload (MVP, не для perf)
- Root `main.py`:
  - прокидывает `profile.resolved_model_mapping` в `vp_runtime_*.yaml`
- `VisualProcessor/main.py`:
  - мержит `resolved_model_mapping` в per‑component cfg (только scalar keys, которые попадут в CLI)
- `VisualProcessor/core/model_process/core_clip/main.py`:
  - поддерживает `--runtime=inprocess|triton`
  - в `triton` режиме делает `ready()` и затем `infer()`
  - пишет `models_used[]` с `runtime=triton|inprocess`, а также `model_signature`

---

### Resolved mapping (MVP schema)

Профиль анализа (`--profile-path`) может содержать:

```yaml
resolved_model_mapping:
  core_clip:
    runtime: triton
    triton_http_url: "http://triton:8000"
    # image embeddings
    triton_image_model_name: "clip_image"
    triton_image_model_version: "1"
    triton_image_input_name: "INPUT__0"
    triton_image_output_name: "OUTPUT__0"
    triton_image_datatype: "FP32"

    # text embeddings (for shot_quality_prompts)
    triton_text_model_name: "clip_text"
    triton_text_model_version: "1"
    triton_text_input_name: "INPUT__0"
    triton_text_output_name: "OUTPUT__0"
    triton_text_datatype: "INT64"

    # model identity for reproducibility / cache
    model_version: "openai-clip@abcdef"
    weights_digest: "sha256:..."
    engine: "onnx"
    precision: "fp16"
```

Правила:

- `resolved_model_mapping` **per run** сохраняется в `manifest.json` (секция `run`).
- Любая смена `engine/precision/weights_digest/model_version` должна менять `model_signature` (через `models_used[]`).
- На MVP mapping лежит в YAML, но **source-of-truth** в проде будет **БД** (см. `docs/models_docs/MODEL_SYSTEM_RULES.md`).


