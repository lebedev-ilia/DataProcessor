# `shot_quality` (v2)

Модуль оценивает **техническое качество** видео на уровне:
- **кадров** (frame-level признаки по выборке `frame_indices`)
- **шотов** (shot-level агрегаты поверх результатов `cut_detection`)

Модуль рассчитан на **GPU full‑quality** режим и работает **строго** по контракту выборки кадров: индексы задаёт Segmenter/DataProcessor в `metadata.json`.

---

## Зависимости (обязательные)

Модуль требует, чтобы до него были успешно запущены и сохранили артефакты:

- **`core_clip`** → `rs_path/core_clip/embeddings.npz`
  - обязательно должен содержать:
    - `frame_indices (N,) int32`
    - `frame_embeddings (N, D) float32`
    - `shot_quality_prompts (P,) object`
    - `shot_quality_text_embeddings (P, D) float32`
- **`core_depth_midas`** → `rs_path/core_depth_midas/depth.npz`
  - `frame_indices (N,) int32`
  - `depth_maps (N, H, W) float32`
- **`core_object_detections`** → `rs_path/core_object_detections/detections.npz`
  - `frame_indices (N,) int32`
  - `boxes (N, MAX, 4) float32`, `valid_mask (N, MAX) bool`, `class_ids (N, MAX) int32`
- **`core_face_landmarks`** → `rs_path/core_face_landmarks/landmarks.npz`
  - `frame_indices (N,) int32`
  - `face_landmarks (N, FACES, 468, 3) float32`
  - `face_present (N, FACES) bool`
  - `has_any_face bool`, `empty_reason object` (например `"no_faces_in_video"`)
- **`cut_detection`** → `rs_path/cut_detection/<...>.npz` (через `BaseModule.save_results`)
  - используется для шот-сегментации (hard cuts) и shot-level агрегатов.

**Важно**: никаких fallback. Если зависимость отсутствует/ключи отсутствуют/индексы не совпадают — модуль делает `raise`.

---

## Входы

### 1) Кадры
Через `FrameManager.get(frame_idx)` из `frames_dir`. В проекте `FrameManager` возвращает `HxWx3 uint8` кадр (ожидается **RGB**).

### 2) Выборка кадров (Sampling)
`shot_quality` не выбирает кадры сам.

Segmenter обязан записать в `frames_dir/metadata.json`:

```json
{
  "shot_quality": {
    "frame_indices": [0, 5, 10, 15]
  }
}
```

### Рекомендация по “умной” выборке (для Segmenter)
Цель — одинаковая информативность для видео в диапазоне **120 … 36000** кадров.

Рекомендуемая стратегия (описательная, не реализована в модуле):
- **target_N**: 240–1200 кадров (например, 600 как центр)
- **stratified uniform**: равномерно по времени + обязательные кадры начала/середины/конца
- если есть shot boundaries (на уровне Segmenter) — **per-shot sampling**:
  - минимум 1 кадр на шот
  - плюс дополнительные кадры пропорционально длине шота, но с cap

---

## Выход (NPZ)

Модуль сохраняет timestamped артефакт через `BaseModule.save_results()` в директорию:
`rs_path/shot_quality/`

Имя файла имеет вид:
`shot_quality_features_<timestamp>_<uid>.npz`

### Ключи

- **`frame_indices`**: `(N,) int32`
- **`feature_names`**: `(F,) object` — имена признаков в `frame_features`
- **`frame_features`**: `(N, F) float32`
- **`quality_probs`**: `(N, P) float16` — вероятности zero-shot классов качества (через `core_clip`)
- **`shot_ids`**: `(N,) int32` — принадлежность каждого кадра шоту
- **`shot_start_frame`**: `(S,) int32`
- **`shot_end_frame`**: `(S,) int32`
- **`shot_frame_count`**: `(S,) int32` — число sampled кадров в шоте
- **`shot_features_mean/std/min/max`**: `(S, F) float32` — агрегаты по кадрам шота
- **`meta`**: `object` — словарь с версией/маппингами категорий и др.

### “Нет лиц” — это нормально
Если на видео нет лиц, это **не ошибка**:
- `core_face_landmarks` всё равно сохраняет NPZ и выставляет `has_any_face=False`, `empty_reason="no_faces_in_video"`
- `shot_quality` сохраняет `face_*` признаки как `NaN`, а в `meta` пишет `faces_available=False` и причину.

### Human-friendly распаковка

```python
import numpy as np

data = np.load(".../shot_quality_features.npz", allow_pickle=True)

frame_indices = data["frame_indices"]
feature_names = data["feature_names"].tolist()
X = data["frame_features"]  # (N,F)

# frame-level dict (легко смотреть/логировать)
frames = {
    int(frame_indices[i]): {feature_names[j]: float(X[i, j]) for j in range(X.shape[1])}
    for i in range(len(frame_indices))
}

# shot-level
S = int(data["shot_start_frame"].shape[0])
shot_means = data["shot_features_mean"]
shots = [
    {
        "start_frame": int(data["shot_start_frame"][s]),
        "end_frame": int(data["shot_end_frame"][s]),
        "frame_count": int(data["shot_frame_count"][s]),
        "mean": {feature_names[j]: float(shot_means[s, j]) for j in range(shot_means.shape[1])},
    }
    for s in range(S)
]
```

### Как найти “последний” артефакт

```python
from pathlib import Path
import numpy as np

root = Path(".../result_store/shot_quality")
latest = max(root.glob("shot_quality_features_*.npz"), key=lambda p: p.stat().st_mtime)
data = np.load(latest, allow_pickle=True)
```

---

## Фичи (кратко)

Фичи организованы в матрицу `frame_features` и описаны именами в `feature_names`. Сейчас включены группы:
- sharpness / blur
- noise / ISO proxies
- exposure / contrast
- color / cast / fidelity
- compression artifacts
- lens proxies (vignetting, CA, glare, …)
- fog/haziness proxy
- temporal (flicker, rolling shutter)
- depth (mean/std/gradient)
- object detections summary
- face ROI quality (по `core_face_landmarks`)

Классы `quality_probs` соответствуют `core_clip["shot_quality_prompts"]` (P=7).


