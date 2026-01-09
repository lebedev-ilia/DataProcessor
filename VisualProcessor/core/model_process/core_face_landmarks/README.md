## Component: `core_face_landmarks` (Tier‑0 baseline)

### Назначение

`core_face_landmarks` извлекает landmarks лица (MediaPipe FaceMesh) по выборке кадров (union-domain).
Дополнительно (опционально) может извлекать `pose` и `hands`, но **face_mesh обязателен** для baseline, т.к. `shot_quality` зависит от face features.

### Входы

- **Кадры**: `FrameManager.get(idx)` из `frames_dir` (RGB uint8).
- **Sampling (строго)**: из `frames_dir/metadata.json`:

```json
{
  "core_face_landmarks": { "frame_indices": [0, 7, 14] }
}
```

No-fallback: отсутствие/пустота `frame_indices` ⇒ **error**.

### Выход

Путь: `result_store/<platform_id>/<video_id>/<run_id>/core_face_landmarks/landmarks.npz`

Ключи (основные):
- `frame_indices (N,) int32`
- `face_landmarks (N, FACES, 468, 3) float32` (NaN если лицо не найдено)
- `face_present (N, FACES) bool`
- `has_any_face bool`
- `empty_reason object|null` (валидный empty: `"no_faces_in_video"`)

Опциональные (если включены флаги):
- `pose_landmarks`, `pose_present`, `has_any_pose`
- `hands_landmarks`, `hands_present`, `has_any_hands`

Extended empty reasons (не меняют provider-status кроме face):
- `face_empty_reason`, `pose_empty_reason`, `hands_empty_reason`

### Empty semantics

- Если `face_mesh` включён и **лиц нет**: это **валидный empty**:
  - `status="empty"`
  - `empty_reason="no_faces_in_video"`
  - данные `face_landmarks` остаются NaN, `face_present=False`
- Если `pose/hands` включены и не детектируются: это **не error**,
  но записываются `pose_empty_reason="no_pose_detected"` / `hands_empty_reason="no_hands_detected"`.

### Meta / models_used

`models_used[]` содержит одну запись:
- `model_name="mediapipe"`
- `model_version=<mediapipe.__version__>`
- `weights_digest` = sha256 от (mediapipe_version + ключевые параметры конфигурации)
- `runtime="inprocess"`, `engine="mediapipe"`, `precision="fp32"`, `device="cpu"`

### Sampling requirements (фиксируем требования компонента)

Компонент работает **двухэтапно**:

#### Stage 1 — Face detection (лёгкий детектор)

Цель: дешево определить, **где вообще есть лица**, чтобы не гонять FaceMesh по всем кадрам.

Требования к выборке от Segmenter (primary sampling):
- Segmenter обязан передать `core_face_landmarks.frame_indices` (union-domain), покрывающие всё видео.
- Эти `frame_indices` считаются **primary** и определяют форму выходных массивов NPZ.
- Отсутствие/пустота `frame_indices` ⇒ **error** (no-fallback).

Внутренняя политика Stage 1:
- компонент берёт **подвыборку** из primary кадров и запускает лёгкий face detector.
- целевой размер подвыборки (по умолчанию): `target=50` кадров, с ограничениями `min=20`, `max=200`.
- подвыборка строится равномерно по primary списку (stride = ceil(N/target)).

#### Stage 2 — FaceMesh landmarks (дорогая часть)

Цель: строить landmarks **только рядом с детектами лиц**.

Внутренняя политика Stage 2:
- если на Stage 1 обнаружены кадры с лицами, компонент выбирает подмножество primary позиций:
  - берём каждый кадр с детектом лица
  - плюс окно вокруг него \(\pm R\) кадров по позиции в primary списке
  - `R` либо задаётся явно (`--face-mesh-window-radius`), либо выводится из stride Stage 1 (увеличивается на длинных видео).
- FaceMesh запускается **только** на выбранных кадрах, но **артефакт остаётся выровненным по primary `frame_indices`**:
  - для кадров, где FaceMesh не запускался, `face_landmarks` остаются NaN, `face_present=False`.

Итоговое требование к sampling:
- Primary `frame_indices` должны быть достаточно равномерными и покрывать видео (для downstream модулей вроде `shot_quality`).
- При этом стоимость FaceMesh контролируется двухэтапной логикой внутри компонента.

Важно:
- Segmenter — единственный владелец sampling.
- **DEFERRED** только синтез глобальной `SamplingPolicy` в Segmenter по всем требованиям компонентов.

### Требования к разрешению (фиксируем требования компонента)

Рекомендуемые границы качества (будут финализированы после аудита всех компонентов):
- **min shorter side**: 256 px
- **target shorter side**: 320–480 px
- **max useful**: ~720 px
 - **апскейл запрещён** (только downscale)


