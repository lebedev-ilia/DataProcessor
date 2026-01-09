## Component: `core_optical_flow` (Tier‑0 baseline)

### Назначение

`core_optical_flow` вычисляет **покадровую кривую движения** (optical flow) на primary выборке кадров и сохраняет её в NPZ для downstream модулей (например, `video_pacing`).

Политика: **Triton-only** (локальный `torch/torchvision` режим запрещён).

### Входы

- **Кадры**: `FrameManager.get(idx)` из `frames_dir` (RGB uint8).
- **Sampling (строго)**: Segmenter обязан положить `metadata["core_optical_flow"]["frame_indices"]` (union-domain).

No-fallback:
- отсутствие/пустота `frame_indices` ⇒ **error**
- `len(frame_indices) < 2` ⇒ **error**

### Runtime (Triton)

Компонент вызывает Triton HTTP v2 и ожидает, что preprocessing (normalization/padding policy) находится в Triton.

Контракт (рекомендованный):
- **input0**: `(B,3,H,W) float32` — предыдущий кадр
- **input1**: `(B,3,H,W) float32` — текущий кадр
- **output**: `(B,2,h,w) float32` — flow (dx, dy)

Поддерживаем 2–3 пресета размера входа:
- `raft_256` (default, быстрее)
- `raft_384`
- `raft_512`

### Batch-size / scheduler

Сейчас компонент считает flow последовательно по парам кадров (внутренний батчинг не реализован).
Дальше это будет оптимизироваться через общий scheduler/DynamicBatching.

### Выход (артефакт)

Путь: `result_store/<platform_id>/<video_id>/<run_id>/core_optical_flow/flow.npz`

Ключи:
- `frame_indices (N,) int32`
- `motion_norm_per_sec_mean (N,) float32`:
  - `0` для первого кадра
  - для остальных: \( \mathrm{mean}(\sqrt{dx^2+dy^2}) / dt / \max(h,w) \)
- `dt_seconds (N,) float32` (`NaN` для первого кадра)
- `meta` (dict, object-array)

### Empty/error semantics

`empty` недопустим. Любая невозможность посчитать кривую движения по контракту ⇒ **error**.

### Meta / models_used

`models_used[]` обязателен:
- `runtime="triton-gpu"`
- `engine="onnx"`
- `device="cuda"`
- `weights_digest="unknown"` (baseline)

### Sampling requirements (shared group)

Компонент должен быть в **shared sampling group** с потребителями, которые требуют выравнивания индексов:
- `video_pacing` (и любой другой модуль, который использует `core_optical_flow`)

Практическое правило: Segmenter должен выдавать одинаковые `frame_indices` для группы (иначе downstream получит mismatch и упадёт).

### Требования к разрешению

Для качества motion-curve достаточно умеренного разрешения, но:
- input frames (analysis timeline): min shorter side **320**, target **640**, max useful **1080**, апскейл запрещён
- внутри модели используются пресеты `raft_256/384/512` (выбор по бюджету)


