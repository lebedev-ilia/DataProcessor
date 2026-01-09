## Baseline GPU ветки (fixed-shape) + Triton план

Цель: все **GPU модели baseline** должны иметь **2–3 фиксированные ветки** (small/medium/large),
экспортированные в **ONNX**, загруженные в **Triton model repository**, и успешно проходящие:
- smoke‑инференс через Triton,
- e2e прогон компонентов через **ModelManager → Triton**.

### Базовый контракт (baseline GPU)

- **Image input contract**: `UINT8 NHWC` (сырой RGB), фиксированный размер ветки.
- **Text input contract**: `INT64` токены (фиксированная длина; batch политика отдельно).
- **No dynamic axes**: формы фиксированы (включая batch=1 для image веток).
- Preprocess (layout/normalize) допускается реализовать как **Triton ensemble** (python backend).

### Важно: два измерения “ветвления” моделей

Для некоторых baseline моделей есть **2 оси разбиения**:
- **(A) input-size ветки**: фиксированные размеры входа (small/medium/large), чтобы покрыть диапазон разрешений видео.
- **(B) сложность/размер архитектуры**: разные варианты одной модели (например, для YOLO: `yolo11n/s/m/l/x`).

Текущая политика (обучение сейчас → прод потом):
- **Сейчас** переносим в Triton только “большие” baseline варианты, которые используются в обучении (например, `yolo11x`).
- **Перед продом** расширяем список: добавляем облегчённые варианты (например, `yolo11n/l`) как дополнительные ветки по сложности.

### Правило выбора ветки (routing)

Маршрутизация делается **до вызова модели** (в оркестраторе/компоненте) по метаданным видео.
Рекомендованная метрика: `max(analysis_width, analysis_height)` (или source‑resolution, если анализ не ресайзится).

Пример (можно подстроить):
- `<= 320` → small
- `<= 448` → medium
- `> 448` → large

Важно: выбранная ветка должна попадать в `resolved_model_mapping` и фиксироваться в `models_used[]`.

---

## 1) Список baseline GPU моделей и веток

### Depth (MiDaS/DPT) — `core_depth_midas`

- **Small**: `midas_256` (256×256)
- **Medium**: `midas_384` (384×384)
- **Large**: `midas_512` (512×512)

Triton модели (ensemble, внешний контракт):
- input: `INPUT__0` (`UINT8`, `[1,S,S,3]`)
- output: `OUTPUT__0` (`FP32`, `[1,S,S]`)

### Optical flow (RAFT) — `core_optical_flow`

- **Small**: `raft_256` (256×256)
- **Medium**: `raft_384` (384×384)
- **Large**: `raft_512` (512×512)

Triton модели (ensemble, внешний контракт):
- inputs: `INPUT0__0`, `INPUT1__0` (`UINT8`, `[1,S,S,3]`)
- output: `OUTPUT__0` (`FP32`, `[1,2,S,S]`)

### Object detections (YOLO11x) — `core_object_detections`

Компонент baseline (Tier‑0), сильно зависит от input size. В README компонента зафиксировано:
- **min shorter side**: 320 px
- **target**: 640 px
- **max useful**: ~1080 px
- **upscale запрещён** (можно letterbox/pad без upscale)

Ветки (fixed-shape, square):
- **Small**: 320
- **Medium**: 640
- **Large**: 960 *(опционально; зависит от VRAM/латентности)*

Практика (local RTX 2060 6GB):
- `yolo11x_960` может падать на инференсе в ORT CUDA/CuDNN (OOM/алгоритм) → на 6GB используем **320/640**.

Дополнительная ось (planned, перед продом):
- варианты по сложности: `yolo11n`, `yolo11s`, `yolo11m`, `yolo11l`, `yolo11x`
- итоговое имя ветки может кодировать обе оси, например: `yolo11x_640`, `yolo11l_640`, `yolo11n_640`

Triton план:
- Triton модели (onnxruntime backend): `yolo11x_320`, `yolo11x_640`, `yolo11x_960`
- Input (ONNX): `images` (`FP32`, `[1,3,S,S]`) (после letterbox+norm)
- Output (ONNX): `output0` (`FP32`, `[1,84,N]`, где N зависит от S)
- Postprocess: decode + NMS делаем в компоненте; ByteTrack остаётся в `core_object_detections` (stateful).

### CLIP image — `core_clip` (image embeddings)

План веток (концептуально):
- **Small**: `clip_image_224` (224×224)
- **Medium**: `clip_image_336` (336×336)
- **Large**: `clip_image_448` (448×448)

Triton модели (ветки) должны принимать:
- input: `INPUT__0` (`UINT8`, `[1,S,S,3]`)
- output: `OUTPUT__0` (`FP32`, `[1,D]`)

Статус: `clip_image_224` — делаем первым; `336/448` зависят от наличия весов/экспортов.

### CLIP text — `core_clip` (prompt embeddings)

Triton модель:
- `clip_text`
- input: `INPUT__0` (`INT64`, `[1,77]`) (fixed-shape, batch=1)
- output: `OUTPUT__0` (`FP32`, `[1,77,512]`) (per-token embeddings; EOT выбирается в клиенте)

Baseline политика (no dynamic axes): prompts считаются **по одному** (29 вызовов на run, сейчас `core_clip` имеет 29 prompt’ов).

---

## 2) Экспорт ONNX → `models/optimized/*`

### Уже готово
- MiDaS/DPT: `models/optimized/midas/*`
- RAFT: `models/optimized/raft/*`
- CLIP: `models/optimized/clip/*` (image@224 + text@77)
- YOLO11x: `models/optimized/yolo11x/*` (320/640/960)
- Places365: `models/optimized/places365/*` (resnet50@224/336/448)

### Нужно сделать
- CLIP image ветки 336/448 (planned)

---

## 3) Triton model repository

Репозиторий моделей:
- `/home/ilya/Рабочий стол/TrendFlowML/DataProcessor/triton/models`

Требуемая структура:
```
<model_name>/
  config.pbtxt
  1/
    <model.onnx> (+ .onnx.data если есть)
```

Для image моделей используем ensemble:
```
<model> (ensemble UINT8 NHWC)
  -> preprocess_<model> (python, UINT8 NHWC -> FP32 NCHW + normalize)
  -> <model>_onnx (onnxruntime)
```

---

## 4) Тесты/бенчмарки после перезапуска Triton

1) Smoke:
- `/v2/health/ready`
- `/v2/repository/index`
- 1–2 инференса на ветку (midas/raft/clip/yolo/places365)

### Чек-лист: какие модели должны быть READY в Triton (local baseline)

Обязательные (baseline GPU):
- `midas_256`, `midas_384`, `midas_512`
- `midas_256_onnx`, `midas_384_onnx`, `midas_512_onnx`
- `preprocess_midas_256`, `preprocess_midas_384`, `preprocess_midas_512`
- `raft_256`, `raft_256_onnx`, `preprocess_raft_256` *(на 6GB это стабильный baseline)*
- `clip_image_224`, `clip_image_224_onnx`, `preprocess_clip_image_224`
- `clip_text`, `clip_text_onnx`
- `places365_resnet50_224`, `places365_resnet50_224_onnx`, `preprocess_places365_224`

Дополнительные (baseline + YOLO11x):
- `yolo11x_320`, `yolo11x_640`, `yolo11x_960` *(на 6GB `yolo11x_960` может быть READY, но inference может OOM)*
- RAFT branches (часто OOM на 6GB, используем при большем VRAM):
  - `raft_384`, `raft_384_onnx`, `preprocess_raft_384`
  - `raft_512`, `raft_512_onnx`, `preprocess_raft_512`
- Places365 branches:
  - `places365_resnet50_336`, `places365_resnet50_336_onnx`, `preprocess_places365_336`
  - `places365_resnet50_448`, `places365_resnet50_448_onnx`, `preprocess_places365_448`

Проверка:

```bash
curl -sS -X POST -H 'Content-Type: application/json' -d '{}' http://localhost:8000/v2/repository/index
```

2) E2E:
- `DataProcessor/main.py` с профилем baseline‑GPU (все компоненты через ModelManager → Triton).

3) Bench:
- `benchmarks/run_bench.py` (Triton HTTP) по baseline GPU матрице (ветка × batch policy).

### Фактические результаты (local, RTX 2060, Triton 24.08, batch=1)

Bench summary (30 repeats) был сохранён в:
- `benchmarks/out/triton-full-local-20260107-225031/summary.json`

P50 latency (ms), batch=1:
- `clip_image_224`: ~43.8ms
- `clip_text`: ~96.1ms
- `midas_256`: ~96.3ms
- `midas_384`: ~267.4ms
- `midas_512`: ~465.7ms
- `raft_256`: ~208.8ms
- `raft_384`: ~417.7ms
- `raft_512`: ~730.2ms

YOLO11x (10 repeats, batch=1):
- `benchmarks/out/triton-baseline-plus-yolo-20260107-232709/summary.json`
- `yolo11x_320` p50: ~438ms
- `yolo11x_640` p50: ~1634ms
- `yolo11x_960`: **OOM/error** на RTX 2060 6GB (см. summary `status=error`)

### E2E (local, через ModelManager → Triton)

Профиль:
- `profiles/pr8_triton_baseline_gpu_local.yaml`

Успешно созданы артефакты:
- `.../core_clip/embeddings.npz`
- `.../core_depth_midas/depth.npz`
- `.../core_optical_flow/flow.npz`

E2E + YOLO (Triton inference + in-process NMS+ByteTrack):
- профиль: `profiles/pr8_triton_baseline_gpu_yolo_local.yaml`
- артефакт: `.../core_object_detections/detections.npz`

---

## Примечания по текущей реализации (local)

- CLIP ONNX экспорт сейчас собран для **OpenAI CLIP ViT‑B/32 @224**:
  - `models/optimized/clip/openai_clip_ViT-B_32_image_224.onnx`
  - `models/optimized/clip/openai_clip_ViT-B_32_text_77.onnx`
- Triton модели добавлены в repo как:
  - `clip_image_224` (ensemble: `UINT8 NHWC` → preprocess → onnx)
  - `clip_text` (ensemble: `INT64 [1,77]` → onnx)
- `core_clip` в Triton-режиме:
  - отправляет `UINT8 NHWC` для `clip_image_224_triton` (ветка)
  - считает text prompts **по одному** (fixed-shape `[1,77]`) для `clip_text_triton`

Текущая практическая деталь:
- Раньше `clip_text_onnx` падал на GPU из‑за `ArgMax` (ORT CUDA EP). Мы убрали `ArgMax` из ONNX экспорта,
  поэтому модель стала совместимой; сейчас `clip_text` остаётся fixed-shape (batch=1) и EOT выбирается в клиенте.

Обновление (fix):
- Мы переэкспортировали `clip_text` **без `ArgMax`**: модель теперь выдаёт `emb_seq` формы `[1,77,512]`,
  а выбор EOT‑токена делается на стороне клиента (см. `core_clip`).

### Быстрый local профиль (ModelManager → Triton)

1) Запусти Triton на repo:
- `/home/ilya/Рабочий стол/TrendFlowML/DataProcessor/triton/models`

2) Экспортни переменную окружения (ModelManager подставляет `${TRITON_HTTP_URL}`):

```bash
export TRITON_HTTP_URL="http://localhost:8000"
```

3) Запусти DataProcessor:

```bash
PY="/home/ilya/Рабочий стол/TrendFlowML/DataProcessor/VisualProcessor/.vp_venv/bin/python"
"$PY" main.py \
  --video-path "/home/ilya/Рабочий стол/TrendFlowML/DataProcessor/NSumhkOwSg.mp4" \
  --profile-path profiles/pr8_triton_baseline_gpu_local.yaml
```


