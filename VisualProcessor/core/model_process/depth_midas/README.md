## Component: `core_depth_midas` (Tier‑0 baseline)

### Назначение

`core_depth_midas` вычисляет depth maps на primary выборке кадров (union-domain) и сохраняет их в `depth.npz`.

Политика: **Triton-only** (локальные `torch.hub` / `engine=torch` запрещены).

### Входы

- **Кадры**: `FrameManager.get(idx)` из `frames_dir` (RGB uint8).
- **Sampling (строго)**: из `frames_dir/metadata.json`:

```json
{
  "core_depth_midas": { "frame_indices": [0, 10, 20] }
}
```

No-fallback:
- отсутствие/пустота `frame_indices` ⇒ **error**

### Runtime (Triton)

`core_depth_midas` вызывает Triton HTTP v2 и ожидает, что preprocessing живёт на стороне Triton (ensemble/модельный граф).

Контракт:
- **input**: `(B,3,H,W) float32` (минимальная упаковка на клиенте: resize + float32 + NCHW)
- **output**: `(B,1,h,w) float32` или `(B,h,w) float32` (depth logits)
- далее клиент ресайзит depth до `out_height/out_width` (по умолчанию 384×384) и сохраняет в NPZ

Поддерживаем 2–3 пресета размера входа:
- `midas_256`
- `midas_384` (default)
- `midas_512`

### Batch size (scheduler-controlled)

`--batch-size` обязателен и задаётся верхним scheduler/DynamicBatching (пока вручную в конфиге/профиле).
Auto-batching внутри компонента запрещён.

### Выход

Путь: `result_store/<platform_id>/<video_id>/<run_id>/core_depth_midas/depth.npz`

Ключи:
- `frame_indices (N,) int32`
- `depth_maps (N, out_h, out_w) float32`
- `meta` (dict, object-array)

### Empty/error semantics

- Empty недопустим: если отсутствует depth-map хотя бы для одного кадра ⇒ **error**.

### Meta / models_used

- `models_used[].runtime="triton-gpu"`
- `models_used[].engine="onnx"` (served by Triton)
- `models_used[].device="cuda"`
- `weights_digest="unknown"` (baseline)

### Sampling requirements (фиксируем требования компонента)

`core_depth_midas` входит в shared sampling group с `shot_quality` и другими core providers:
`core_clip`, `core_depth_midas`, `core_object_detections`, `core_face_landmarks` должны работать на **одном и том же** primary `frame_indices` (иначе downstream падает из-за mismatch).

### Требования к разрешению (фиксируем требования компонента)

- input frames (analysis timeline): min shorter side **320**, target **640**, max useful **1080**, апскейл запрещён
- output depth map: default **384×384** (допускаются пресеты по бюджету)


