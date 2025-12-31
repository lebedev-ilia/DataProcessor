# Segmenter contract: sampling, fps/resolution, frames_dir (полуфинал)

## 1) Роль Segmenter

Segmenter — единственный источник `frame_indices` для каждого компонента (core providers и modules).
Модули/провайдеры **не генерируют семплинг сами**.

## 2) Time-domain, но выход = frame_indices

Полуфинал:
- Segmenter мыслит в **секундах** (time-domain)
- возвращает `frame_indices` (int)

## 3) Budgets per component

Segmenter выдаёт индексы отдельно для каждого компонента и соблюдает budgets `min/target/max` (настройки могут жить в policy).

Стартовые ориентиры (можно менять):
- `cut_detection`: 400–1500
- `core_clip`: 200–800
- `core_depth_midas`: 120–400
- `core_face_landmarks`: 200–800
- `shot_quality`: 200–1000

## 4) Двухпроходность

Допускается Pass1→Pass2:
- Pass1: только дешёвые сигналы (downscale, histogram diff, brightness, cheap motion proxy) и/или лёгкие результаты.
- Pass2: уточнение индексов под дорогие компоненты.

Важно:
- Segmenter **не** генерирует shots/segments как финальный артефакт — это задача `cut_detection`.
- “cut candidates (cheap)” можно держать внутри Segmenter как эвристику без ML.

## 5) frames_dir = только union sampled

Полуфинальный стандарт:
- Segmenter выбирает per-component `frame_indices`.
- Далее строит `union_frame_indices` по всем компонентам.
- **frames_dir хранит только union кадры** (в фиксированном порядке union).
- `frame_indices` в metadata для компонентов — это **индексы в union** (0..N-1), которые валидны для `FrameManager.get()`.

Mapping к исходнику:
- `union_timestamps_sec` и/или `union_frame_indices_source`

## 6) fps/resolution (analysis timeline)

В `frames_dir/metadata.json` фиксируем параметры “analysis timeline”:
- `analysis_fps`
- `analysis_width`, `analysis_height`
- `color_space="RGB"`

Все модули опираются на эти параметры, и это считается частью воспроизводимости.

## 7) Цветовое пространство (RGB)

- Кадры, доступные через `FrameManager.get()`, должны быть **RGB**.
- Если модулю нужен OpenCV BGR — модуль делает conversion локально и явно.


