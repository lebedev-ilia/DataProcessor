## PR‑9: Model optimization pipeline (ONNX / quantization) — MVP

### Цель

В baseline‑проде должны существовать **оптимизированные артефакты моделей** и способ их использовать:

- экспорт модели в ONNX (как минимум для 1 baseline модели),
- (опционально) квантование,
- переключение компонента на `engine=onnx` (fail-fast если deps/артефакт отсутствуют),
- фиксация `engine/precision/model_version/weights_digest` в `models_used[]` ⇒ влияет на `model_signature`.

---

### Что сделано (MVP)

#### 1) `core_depth_midas` поддерживает `engine=torch|onnx`

Файл: `VisualProcessor/core/model_process/depth_midas/main.py`

- `--engine=torch|onnx`
- `--onnx-path` (обязателен для `--engine=onnx`)
- `--model-version`, `--weights-digest`, `--precision` (для воспроизводимости/кэша)

Поведение:

- Если `--engine=onnx`, но `onnxruntime` не установлен / onnx файл не найден → **fail-fast** (no-fallback).

#### 2) Скрипты оптимизации

- `scripts/model_opt/export_midas_onnx.py` — экспорт MiDaS в ONNX (torch.hub)
- `scripts/model_opt/quantize_onnx_dynamic.py` — опциональное dynamic quantization (onnxruntime)

---

### Как использовать (пример)

Экспорт ONNX (в отдельном окружении с torch):

```bash
python3 scripts/model_opt/export_midas_onnx.py \
  --model-name MiDaS_small \
  --out ./models/optimized/midas/MiDaS_small/model.onnx \
  --h 256 --w 256 --dynamic
```

Опционально: квантование (в окружении с onnxruntime):

```bash
python3 scripts/model_opt/quantize_onnx_dynamic.py \
  --in ./models/optimized/midas/MiDaS_small/model.onnx \
  --out ./models/optimized/midas/MiDaS_small/model.int8.onnx
```

Подключение в профиле (через `resolved_model_mapping`, MVP источник):

```yaml
resolved_model_mapping:
  core_depth_midas:
    engine: onnx
    onnx_path: "./models/optimized/midas/MiDaS_small/model.onnx"
    model_version: "MiDaS_small@unknown"
    weights_digest: "sha256:<fill_from_export>"
    precision: "fp32"
```

---

### Ограничения MVP

- TensorRT pipeline в этом PR не реализован (будет расширение: build engine + triton repo).
- Скрипты не запускаются автоматически в CI (нет гарантированного окружения с torch/onnxruntime).


