## Baseline models (CPU vs GPU)

Этот документ фиксирует **baseline набор моделей** и разделяет их на категории:
- **GPU (Triton)** — то, с чем сейчас работаем в первую очередь.
- **CPU / in-process** — локальные артефакты (без сетевых загрузок).

### Scope / решения

- **TextProcessor НЕ входит в baseline** (может быть подключён позже как optional профиль).
- **cut_detection использует CLIP** (zero-shot классификация переходов), но **НЕ грузит локальные веса**.
  - Text embeddings prompts берутся из `core_clip/embeddings.npz`
  - Image embeddings считаются через Triton CLIP image encoder (через `dp_models` spec)
- **Baseline GPU I/O фиксируем**:
  - image models: **UINT8 NHWC** (сырой RGB), preprocess в Triton graph/ensemble
  - text models: **INT64 tokens**

---

### GPU (Triton) — current focus

#### **CLIP**

- **`clip_image_triton`** (role: `clip_image`)
  - Triton model: `clip_image` (version `1`)
  - Used by: `core_clip` (frame embeddings), `cut_detection` (candidate windows)
  - Spec: `dp_models/spec_catalog/vision/clip_image_triton.yaml`

- **CLIP branches (fixed input, recommended baseline)**
  - `clip_image_224_triton` → Triton model `clip_image_224`
  - `clip_image_336_triton` → Triton model `clip_image_336`
  - `clip_image_448_triton` → Triton model `clip_image_448`
  - Specs: `dp_models/spec_catalog/vision/clip_image_*_triton.yaml`

- **`clip_text_triton`** (role: `clip_text`)
  - Triton model: `clip_text` (version `1`)
  - Used by: `core_clip` (prompt embeddings for `shot_quality`, `scene_classification`, `cut_detection`)
  - Spec: `dp_models/spec_catalog/vision/clip_text_triton.yaml`

#### **MiDaS / RAFT branches (fixed input)**

- MiDaS depth:
  - `midas_256_triton` / `midas_384_triton` / `midas_512_triton`
  - Specs: `dp_models/spec_catalog/vision/midas_*_triton.yaml`
- RAFT optical flow:
  - `raft_256_triton` / `raft_384_triton` / `raft_512_triton`
  - Specs: `dp_models/spec_catalog/vision/raft_*_triton.yaml`

#### **Pre‑Triton workflow (before building Triton repo)**

- Pre‑Triton bench + export guide: `docs/models_docs/PRETRITON_BENCH_AND_EXPORT.md`
- Baseline GPU branches (fixed-shape) + Triton plan: `docs/models_docs/BASELINE_GPU_BRANCHES.md`

#### **Benchmark entrypoint**

- Spec matrix: `benchmarks/specs/baseline_gpu.yaml`
- Runner: `python -m benchmarks.run_bench --spec benchmarks/specs/baseline_gpu.yaml --dry-run`

> В профилях `resolved_model_mapping` рекомендуется передавать **model spec** (а не сырые triton_* параметры),
> чтобы фиксировать `models_used[]` через ModelManager.

---

### CPU / in-process (локальные артефакты)

#### **Places365 (scene_classification)**

- **Places365 ResNet50 fixed-shape ветки (baseline)**
  - Runtime: Triton (`onnxruntime_onnx` + preprocess ensemble)
  - Used by: `scene_classification`
  - Specs:
    - `dp_models/spec_catalog/vision/places365_resnet50_224_triton.yaml` → Triton model `places365_resnet50_224`
    - `dp_models/spec_catalog/vision/places365_resnet50_336_triton.yaml` → Triton model `places365_resnet50_336`
    - `dp_models/spec_catalog/vision/places365_resnet50_448_triton.yaml` → Triton model `places365_resnet50_448`

Legacy (CPU / in-process, локальные артефакты):
- `places365_resnet50`, `places365_resnet18` → `dp_models/spec_catalog/vision/places365_*.yaml`

#### **YOLO (object detections)**

- Ultralytics YOLO weights (например `yolo11x.pt`)
  - Runtime: inprocess (torch/ultralytics)
  - Used by: `core_object_detections`
  - Source-of-truth: VisualProcessor config + локальный файл весов (runtime downloads запрещены)

#### **MediaPipe (face landmarks)**

- MediaPipe модели (внутри пакета)
  - Runtime: inprocess
  - Used by: `core_face_landmarks`


