## Component: `core_clip` (Tier‑0 baseline)

### Назначение

`core_clip` вычисляет CLIP эмбеддинги для выборки кадров (union-domain) и сохраняет их в NPZ.
Дополнительно сохраняет **text embeddings** для фиксированных `shot_quality_prompts`, чтобы downstream модуль `shot_quality` мог делать zero‑shot scoring без загрузки CLIP модели.

### Входы

- **Кадры**: через `FrameManager.get(idx)` из `frames_dir` (RGB uint8).
- **Sampling (строго)**: из `frames_dir/metadata.json`:

```json
{
  "core_clip": { "frame_indices": [0, 10, 20] }
}
```

**No-fallback**: если `core_clip.frame_indices` отсутствует или пустой — компонент **падает** (empty недопустим).

### Выходы

Путь: `result_store/<platform_id>/<video_id>/<run_id>/core_clip/embeddings.npz`

Ключи:
- `frame_indices (N,) int32` — union-domain
- `frame_embeddings (N, D) float32`
- `shot_quality_prompts (P,) object`
- `shot_quality_text_embeddings (P, D) float32`
- `meta` (dict, object-array) — canonical meta

### Meta (обязательное)

`meta` содержит:
- run identity: `platform_id`, `video_id`, `run_id`, `config_hash`, `sampling_policy_version`, `dataprocessor_version`
- `producer`, `producer_version`, `schema_version`, `created_at`
- `status="ok"`, `empty_reason=null`
- `models_used[]` + `model_signature`
- `batch_size` (контролируется верхним scheduler/DynamicBatching; auto внутри компонента запрещён)

### Runtime modes

#### `runtime=inprocess`

- модель и препроцессинг берутся из `openai/CLIP` (python package `clip`).

#### `runtime=triton`

В triton режиме **и image, и text эмбеддинги считаются через Triton** (no local inference).

`resolved_model_mapping` должен задать (пример):

```yaml
resolved_model_mapping:
  core_clip:
    runtime: triton
    triton_http_url: "http://triton:8000"

    triton_image_model_name: "clip_image"
    triton_image_model_version: "1"
    triton_image_input_name: "INPUT__0"
    triton_image_output_name: "OUTPUT__0"
    triton_image_datatype: "FP32"

    triton_text_model_name: "clip_text"
    triton_text_model_version: "1"
    triton_text_input_name: "INPUT__0"
    triton_text_output_name: "OUTPUT__0"
    triton_text_datatype: "INT64"

    # 2–3 стандартных варианта под разные input size
    triton_preprocess_preset: "openai_clip_224"  # openai_clip_224 | openai_clip_336 | openai_clip_448

    # batch_size задаётся строго (верхний scheduler; пока вручную в профиле/конфиге)
    batch_size: 16
```

Примечание:
- preprocessing для image embeddings делается локально до `(B,3,S,S) float32` по CLIP mean/std (`S` зависит от preset).
- tokenization для text embeddings делается локально (`clip.tokenize`), далее токены отправляются в Triton.

### Device/runtime semantics (фиксируем)

- `models_used[].runtime`:
  - `inprocess` — локальный inference внутри процесса
  - `triton-gpu` — inference через Triton (в нашем продакшен-контексте предполагается GPU)
- `models_used[].device`:
  - `inprocess`: `cpu|cuda`
  - `triton-gpu`: `cuda`

### Фичи (выход) — группы и оценки

- **`frame_embeddings`**:
  - **алгоритм**: CLIP image encoder на sampled кадрах
  - **оценка реализации**: 9/10
  - **полезность**: 10/10 (базовый универсальный визуальный сигнал, используется многими downstream/головами)
- **`shot_quality_*` (prompts + text embeddings)**:
  - **алгоритм**: CLIP text encoder для фиксированного набора prompts
  - **оценка реализации**: 8/10
  - **полезность**: 7/10 (служебно для `shot_quality`, важно для воспроизводимости/ускорения)

### Sampling requirements (фиксируем требования компонента)

`core_clip` используется downstream несколькими компонентами (например `shot_quality`), поэтому выборка должна быть “универсальной по качеству”:
- **coverage**: обязательно покрывать начало/середину/конец и быть равномерной по времени;
- **cap**: для длинных видео иметь ограничение по числу кадров (чтобы не взрывать стоимость);
- **стабильность**: индексы должны быть отсортированы, уникальны, валидны для union-domain.

Важно:
- Segmenter — единственный владелец sampling.
- **DEFERRED** только синтез глобальной `SamplingPolicy` в Segmenter по всем требованиям компонентов.
  Но сами требования выше считаются обязательной частью контракта `core_clip`.


