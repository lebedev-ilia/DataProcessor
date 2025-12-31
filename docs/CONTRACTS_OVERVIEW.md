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
8) **Idempotency**: компонент уникально идентифицируется ключом `(platform_id, video_id, run_id, component, config_hash, sampling_policy_version, versions)`.
9) **Targets**: multi-target (views+likes) + multi-horizon (14/21 обязательно, 7 с mask), считаем **дельты** и `log1p`.
10) **Reproducibility**: в каждом NPZ фиксируем producer/schema версии, config_hash, sampling_policy_version и model versions.

## Что считается MVP по моделям

- Обязательный baseline (CatBoost/LightGBM) — контрольная точка качества данных.
- Prod стартует с **v2 multimodal transformer** (token=shot, `max_len_shots=256`), но baseline/v1 остаются как sanity-check и fallback.


