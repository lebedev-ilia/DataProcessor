## Component: `core_object_detections` (Tier‑0 baseline)

### Назначение

`core_object_detections` вычисляет детекции объектов на primary выборке кадров (union-domain) и пишет их в `detections.npz`.
В baseline используется **только YOLO (ultralytics)** + **ByteTrack (required)** для трекинга.

### Входы

- **Кадры**: `FrameManager.get(idx)` из `frames_dir` (RGB uint8).
- **Sampling (строго)**: из `frames_dir/metadata.json`:

```json
{
  "core_object_detections": { "frame_indices": [0, 10, 20] }
}
```

No-fallback:
- отсутствие/пустота `frame_indices` ⇒ **error**
- компонент не выбирает кадры сам

### Выход

Путь: `result_store/<platform_id>/<video_id>/<run_id>/core_object_detections/detections.npz`

Ключи:
- `frame_indices (N,) int32`
- `boxes (N, MAX, 4) float32` (xyxy)
- `scores (N, MAX) float32`
- `class_ids (N, MAX) int32`
- `valid_mask (N, MAX) bool`
- `class_names (M,) str` — `"id:name"` mapping
- `tracks (N, MAX) int32` — `track_id` или `-1`
- `tracks_list (K,) object` — список union-frame_indices для каждого track
- `tracks_list_ids (K,) int32`
- `meta` (dict, object-array)

### Tracking (required)

ByteTrack является **обязательной** частью baseline:
- если трекер не импортируется/падает ⇒ компонент **error** (fail-fast)

Технически `yolox` (ByteTrack) vendored в `ByteTrack/` внутри этого компонента.

### Batch size (scheduler-controlled)

`--batch-size` обязателен и задаётся верхним scheduler/DynamicBatching (пока вручную в конфиге).
Auto-batching внутри компонента запрещён.

### Meta / models_used

- `models_used[].runtime="inprocess"`
- `engine="ultralytics"`
- `weights_digest="unknown"` (baseline)

### Sampling requirements (фиксируем требования компонента)

Этот компонент входит в “shared sampling group” с `shot_quality` и другими core providers:
`core_clip`, `core_depth_midas`, `core_face_landmarks`, `core_object_detections` должны работать на **одном и том же** primary `frame_indices` (иначе downstream падает из-за mismatch).

### Требования к разрешению (фиксируем требования компонента)

- **min shorter side**: 320 px
- **target**: 640 px
- **max useful**: 1080 px
- **апскейл запрещён** (только downscale)


