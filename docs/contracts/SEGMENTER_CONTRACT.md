# Segmenter contract: sampling, fps/resolution, frames_dir (полуфинал)

## 1) Роль Segmenter

Segmenter — единственный источник `frame_indices` для каждого компонента (core providers и modules).
Модули/провайдеры **не генерируют семплинг сами**.

### 1.1) Skip semantics (unreadable video)

- Если видео **не открывается/не декодируется** (например, битый контейнер) — это **SKIP**, а не ERROR:
  - Segmenter завершает работу с exit-code `10`
  - Оркестратор обязан:
    - пометить run как `segmenter=skipped`
    - **не запускать** Audio/Text/Visual для этого run
    - залогировать причину (например `error_code="video_unreadable"`)

## 2) Time-domain, но выход = frame_indices

Полуфинал:
- Segmenter мыслит в **секундах** (time-domain)
- возвращает `frame_indices` (int)

### 2.1) Time-axis = source-of-truth (мультимодальная синхронизация)

Полуфинальный стандарт (фиксируем сейчас):

- **Единая временная ось** — обязательный контракт для синхронизации модальностей.
- Для каждого union-кадра Segmenter записывает:
  - `union_timestamps_sec`: timestamp в секундах (float) — **истина времени**.
- Для аудио Segmenter записывает:
  - `audio/audio.wav` (каноническое имя)
  - `audio/metadata.json` с `duration_sec`, `sample_rate`, `total_samples`.
  - `audio/segments.json` (канонический список временных окон для аудио‑экстракторов)

**Как маппить видео ↔ аудио:**

- `t_frame = union_timestamps_sec[frame_idx]`
- дальше аудио-экстракторы/модули выбирают окно вокруг \(t_frame\) (nearest/overlap/pooling).

Важно:
- Segmenter **не обязан** выравнивать кадры на равномерной сетке “analysis timeline”.
- Любые решения про **sampling policy** (как выбирать кадры, `analysis_fps`, resizing) считаются **DEFERRED** до завершения полного аудита компонентов.

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
- `platform_id`, `video_id`, `run_id`, `sampling_policy_version`, `config_hash` (run identity)

Все модули опираются на эти параметры, и это считается частью воспроизводимости.

**Важно (обновлено)**:
`analysis_fps` и `analysis_width/analysis_height` **НЕ фиксируем как константы** на этом этапе.
Это часть **sampling policy**, которая будет определена после полного аудита компонентов:

- для каждого компонента мы фиксируем:
  1) какие кадры / сколько / в каком порядке нужны для качества,
  2) какое разрешение нужно для качества,
- после этого проектируем универсальную систему выбора кадров (умный sampling) с min/max ограничениями.

На текущем этапе `analysis_*` остаются полями контракта в `frames_dir/metadata.json`,
но правила их выбора считаются **DEFERRED** до завершения аудита.
- Максимальная длительность видео: **20 минут**
- Ограничение входного разрешения: если исходное видео больше 1080p, выполняется downscale до 1080p **до начала анализа** (не в Segmenter, а на этапе pre-processing/downloading).

## 7) Цветовое пространство (RGB)

- Кадры, доступные через `FrameManager.get()`, должны быть **RGB**.
- Если модулю нужен OpenCV BGR — модуль делает conversion локально и явно.

## 8) Привязка директорий к video_id (важно для оркестратора)

- При наличии `video_id` Segmenter должен создавать output folder, следуя **каноническому `video_id`**, а не basename файла.
- Это гарантирует стабильность путей:
  - `frames_dir = <output>/<video_id>/video`

## 8.1) ResultStore per-run (оркестратор)

- Все процессоры (Segmenter → Audio/Text/Visual) должны работать в рамках **одного** `run_rs_path`:
  - `result_store/<platform_id>/<video_id>/<run_id>/`
- `manifest.json` внутри `run_rs_path` — единый агрегированный манифест, в который каждый процессор делает upsert.

## 9) Audio extraction (stable naming + fail-fast)

- Segmenter извлекает аудио в `audio/audio.wav` (стабильное имя, не зависит от названия исходного файла).
- Если `ffmpeg`/`ffprobe` отсутствуют в PATH — Segmenter **падает** (fail-fast).
- Segmenter также пишет `audio/segments.json` — **источник истины** для time‑domain окон аудио‑экстракторов (Tier‑0).

### 9.1) Аудио-метаданные без ffprobe (fallback)

- Если `ffprobe` доступен, Segmenter использует его для `duration_sec`/`sample_rate`.
- Если `ffprobe` вернул невалидные значения для WAV — допускается fallback на локальный парсер WAV (например, через стандартный модуль `wave`).

### 9.2) Роль AudioProcessor (forward plan)

План (будет реализовано в аудите AudioProcessor):
- AudioProcessor **не извлекает** аудио из видео, а берет `audio/audio.wav`, подготовленный Segmenter.
- Аудио-экстракторы должны выдавать **последовательность фичей по time-domain сегментам** + агрегаты по окнам/сегментам.

### 9.3) `audio/segments.json` (contract v1)

Схема:
- `schema_version="audio_segments_v1"`
- `sample_rate`, `total_samples`, `audio_duration_sec`, `video_duration_sec`
- `families`:
  - `primary`: короткие окна вокруг time‑anchors (по умолчанию якоря берутся из `core_clip` sampling, иначе — равномерно по union time-axis). Используется для `clap_extractor` и `loudness_extractor`.
  - `tempo`: длинные sliding windows (`window_sec/stride_sec`) для устойчивого BPM. Используется для `tempo_extractor`.
  - `asr`: длинные sliding windows (`window_sec/stride_sec`) для ASR chunking (Whisper/Triton). Используется для `asr_extractor`.
  - `diarization`: фиксированные окна (`window_sec/stride_sec`) для speaker diarization (speaker embeddings). Используется для `speaker_diarization_extractor`.
  - `emotion`: более длинные перекрывающиеся окна (`window_sec/stride_sec`) для emotion diarization (качество). Используется для `emotion_diarization_extractor`.
  - `source_separation`: длинные окна (`window_sec/stride_sec`) для source separation shares (CPU/feature heavy). Используется для `source_separation_extractor`.

Каждый сегмент содержит:
- `start_sec/end_sec/center_sec`
- `start_sample/end_sample` (индексы в `audio/audio.wav`)

Политика:
- Если Segmenter обнаруживает существенный drift между `audio_duration_sec` и `video_duration_sec` — это **ERROR** (no-fallback), чтобы не ломать мультимодальную синхронизацию.


