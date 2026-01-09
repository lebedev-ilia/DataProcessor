# Контракты TrendFlow / DataProcessor (полуфинал)

Этот документ — сжатое “оглавление контрактов”. Детали см. в остальных файлах папки `docs/`.

## Термины

- **DataProcessor**: верхний продуктовый пайплайн, который обрабатывает 1 видео (video + meta + comments) и сохраняет артефакты.
- **VisualProcessor/AudioProcessor/TextProcessor**: процессоры, которые считают признаки и сохраняют NPZ артефакты.
- **Core providers**: тяжёлые общие провайдеры VisualProcessor (`core_clip`, `core_depth_midas`, `core_object_detections`, `core_face_landmarks`).
- **Module**: модуль VisualProcessor, который использует кадры и/или core providers и пишет NPZ.
- **Segmenter**: отвечает за выборку (семплинг) — выдаёт `frame_indices` отдельно для каждого компонента.
- **Artifact / NPZ**: source-of-truth артефакт с массивами и `meta`.

## Главные правила (если запомнить только 10)

1) **NPZ — source of truth**, JSON — только presentation layer.
2) **No-fallback policy**: если dependency/`frame_indices` отсутствуют — компонент обязан `raise`.
3) **Segmenter отвечает за sampling** и кладёт `frame_indices` для каждого компонента в metadata.
4) **frames_dir хранит только union sampled кадры** (а не все кадры видео).
5) Кадры в `frames_dir` — **RGB** (`color_space="RGB"`).
6) **Empty outputs валидны**: NaN + `*_present` masks + `empty_reason` (не массивы нулей, не падение).
7) **Storage per-run**:
   - `result_store/<platform_id>/<video_id>/<run_id>/<component_name>/...`
   - `manifest.json` рядом с артефактами.
8) **Idempotency**: компонент уникально идентифицируется ключом `(platform_id, video_id, run_id, component, config_hash, sampling_policy_version, dataprocessor_version, versions)`.
   - `config_hash` должен быть **единым для всего run** и прокидываться во все под-процессоры (Segmenter/Visual/Audio/Text).
   - `dataprocessor_version` обязателен в `manifest.run` и в `meta` каждого NPZ (baseline может быть `"unknown"`).
9) **Targets**: multi-target (views+likes) + multi-horizon (14/21 обязательно, 7 с mask), считаем **дельты** и `log1p`.
10) **Reproducibility**: в каждом NPZ фиксируем producer/schema версии, config_hash, sampling_policy_version, dataprocessor_version и model versions.

Дополнения (Round 1, полуфинал):
- **Платформа v1**: только YouTube (`platform_id="youtube"`).
- **Запрет JSON артефактов**: в `result_store` разрешён только `manifest.json` (остальное — NPZ, JSON генерируется только на выдаче во фронт).
- **Retention**:
  - `frames_dir` (union кадры) храним 7 дней (см. `ORCHESTRATION_AND_CACHING.md`).
  - raw OCR/comments: `hard_cap_days=60` (см. `PRIVACY_AND_RETENTION.md`).

## Текущий baseline execution path (референс)

На текущем этапе (baseline v0) минимальный “сквозной” путь выглядит так:

- `DataProcessor/main.py`:
  - вычисляет `platform_id/video_id/run_id/config_hash/sampling_policy_version`
  - запускает `Segmenter/segmenter.py` → создаёт `frames_dir` с `metadata.json` (union sampling)
  - опционально запускает:
    - `AudioProcessor/run_cli.py` → пишет per-run NPZ + апдейтит `manifest.json`
    - `TextProcessor/run_cli.py` → пишет per-run NPZ + апдейтит `manifest.json`
  - запускает `VisualProcessor/main.py` → пишет core/modules NPZ + апдейтит `manifest.json`

## ResultStore: единый per-run storage (обязательное правило)

**Единый результат для одного run** живёт в одной директории:

- `result_store/<platform_id>/<video_id>/<run_id>/`
  - `manifest.json` (единый, агрегируется из всех процессоров)
  - `<component_name>/*.npz` (source-of-truth артефакты)

Политика:
- **Orchestrator (DataProcessor)**:
  - создаёт `run_rs_path` один раз
  - создаёт/обновляет `manifest.json` (run meta)
  - запускает Audio/Text/Visual и передаёт им **явный `run_rs_path`**
- **Processors (AudioProcessor/TextProcessor/VisualProcessor)**:
  - **не должны** придумывать свои отдельные result_store
  - пишут артефакты **только внутри `run_rs_path`**
  - делают `RunManifest(...).upsert_component(...)` (manifest умеет merge существующего файла)

## Параллелизм (baseline v0)

- Baseline-режим: видео обрабатываются последовательно (1 video = 1 job).
- Внутри одного видео допускается параллельный запуск модулей VisualProcessor, но:
  - GPU-тяжёлые задачи ограничиваются лимитом `gpu_max_concurrent` (по умолчанию 1 на малой VRAM).

## Что считается MVP по моделям

## Audio: source-of-truth windows (Segmenter → AudioProcessor)

- Segmenter пишет:
  - `audio/audio.wav`
  - `audio/metadata.json`
  - `audio/segments.json` (contract `audio_segments_v1`)
- AudioProcessor Tier‑0 **не извлекает** аудио из видео и работает только от `frames_dir`:
  - читает `audio/audio.wav` + `audio/segments.json`
  - считает per‑segment sequences + агрегаты

- Для ASR (non‑baseline, но прод‑важно):
  - `asr_extractor` пишет **token IDs** (не raw text) и фиксирует в `models_used[]`:
    - Whisper Triton model (`whisper_*_triton`)
    - shared tokenizer (`shared_tokenizer_v1`)

- Дополнительные non‑baseline аудио‑экстракторы (все работают от `audio/segments.json` и пишут NPZ в `run_rs_path`):
  - `speaker_diarization_extractor` (families.diarization)
  - `emotion_diarization_extractor` (families.emotion)
  - `source_separation_extractor` (families.source_separation)
  - `speech_analysis_extractor` (aggregator: ASR+diarization(+optional pitch))
- Обязательный baseline (CatBoost/LightGBM) — контрольная точка качества данных.
- Prod стартует с **v2 multimodal transformer** (token=shot, `max_len_shots=256`), но baseline/v1 остаются как sanity-check и fallback.


