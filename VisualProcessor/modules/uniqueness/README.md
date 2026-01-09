# `uniqueness` (Visual module, Tier‑0 baseline)

Baseline‑компонент “уникальности” в MVP: считает **intra‑video** метрики повторяемости/разнообразия по sampled кадрам, используя **только `core_clip` embeddings**.

> Важно: это не “novelty vs reference videos”. Референс‑логика (топ‑видео и т.п.) не является частью Tier‑0 baseline и здесь не реализуется.

## Входы

### Основной вход
- **`frames_dir`**: директория Segmenter с `metadata.json` и батчами кадров.
- **`metadata["uniqueness"]["frame_indices"]`**: индексы кадров в **union-domain** (0..N-1), которые обрабатывает модуль.

### Time-axis (обязательно)
- **`metadata["union_timestamps_sec"]`**: timestamp’ы (сек) для каждого union-кадра — **source-of-truth** времени. Используется, чтобы считать temporal‑метрики **per‑second**, а не в “пер‑кадр” масштабе.

### Зависимости (hard deps, no-fallback)
- **`core_clip`**: `result_store/<platform>/<video>/<run>/core_clip/embeddings.npz`
  - `frame_indices (N,) int32`
  - `frame_embeddings (N, D) float32`

Контракт: `core_clip.frame_indices` обязан **полностью покрывать** `metadata["uniqueness"]["frame_indices"]`. Иначе — error (no-fallback).

## Sampling requirements (Visual)

Компонент строит pairwise similarity \(N\times N\) (сложность \(O(N^2)\)), поэтому Sampling должен быть ограничен.

- **min frames**: 60  
- **target frames**: 120  
- **max frames**: 200  

Если Segmenter выдаст больше `max_frames` — компонент должен **fail-fast** (это ошибка sampling policy).

## Выход (артефакт)

Пишется через `BaseModule.save_results()` в:
- `result_store/<platform_id>/<video_id>/<run_id>/uniqueness/uniqueness_features_<ts>_<uid>.npz`

### Ключи NPZ
- **`frame_indices`**: `(N,) int32` — union-domain кадры модуля.
- **`max_sim_to_other`**: `(N,) float32` — для каждого кадра максимальная cosine similarity к любому *другому* кадру (diag исключена).
- **`cos_dist_next`**: `(N-1,) float32` — cosine distance между соседними кадрами (по времени/порядку sampling).
- **`features`**: `object(dict)` — агрегированные метрики (см. ниже).
- **`meta`**: `object(dict)` — canonical meta (run identity keys, schema/producer versions, models_used/model_signature, status/empty_reason и т.д.).

## Метрики (`features`)

### Repetition / similarity
- **`repeat_threshold`**: порог (cosine similarity), выше которого кадр считается “повтором”.
- **`repetition_ratio`**: доля кадров, у которых `max_sim_to_other >= repeat_threshold`.
- **`pairwise_sim_mean`**: средняя попарная cosine similarity по верхнему треугольнику.
- **`pairwise_sim_p95`**: 95‑й перцентиль попарной similarity.

### Temporal change (per-second)
Считаем cosine distance между соседними кадрами и нормируем на \(dt\) из `union_timestamps_sec`.
- **`temporal_change_mean`**: средняя скорость изменения семантики (per-second).
- **`temporal_change_std`**: std скорости изменения семантики (per-second).

### Diversity proxy
- **`diversity_score`**: `clip(1 - pairwise_sim_mean, 0..1)` (чем меньше средняя similarity, тем выше diversity).
- **`n_frames`**: число sampled кадров \(N\).

## No-fallback / empty semantics

- **No-fallback**:
  - отсутствует `frame_indices`;
  - отсутствует/битый `union_timestamps_sec` или не покрывает `frame_indices`;
  - отсутствует `core_clip/embeddings.npz` или он не покрывает `frame_indices`;
  - `N > max_frames`.
- **Empty outputs**: для baseline не предусмотрены; пустой `frame_indices` → error.

## Параметры (config)

CLI: `VisualProcessor/modules/uniqueness/main.py`
- **`repeat_threshold`** (`float`, default `0.97`)
- **`max_frames`** (`int`, default `200`) — safety‑лимит на \(N\) (дублирует sampling contract).


