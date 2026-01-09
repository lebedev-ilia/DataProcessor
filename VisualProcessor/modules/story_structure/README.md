# `story_structure` (Visual module, Tier‑0 baseline)

Модуль оценивает **структуру истории** (hook/climax/energy) по sampled кадрам и core‑провайдерам.

## Входы

### Основной вход
- **`frames_dir`**: директория Segmenter с `metadata.json` и батчами кадров.
- **`metadata["story_structure"]["frame_indices"]`**: индексы кадров в **union-domain** (0..N-1).

### Time-axis (обязательно)
- **`metadata["union_timestamps_sec"]`**: timestamps (sec) для каждого union-кадра — **source-of-truth** времени.
- **No-fallback**: отсутствие/битость `union_timestamps_sec` ⇒ error.

### Зависимости (hard deps, no-fallback)
Компонент требует наличие артефактов и покрытие `frame_indices` (mapping по union-domain индексам):
- **`core_clip`**: `.../core_clip/embeddings.npz` (CLIP embeddings)
- **`core_optical_flow`**: `.../core_optical_flow/flow.npz` (motion curve)
- **`core_face_landmarks`**: `.../core_face_landmarks/landmarks.npz` (face presence)

Важно: `core_face_landmarks` может быть **валидным empty** (`no_faces_in_video`). Это не ошибка для `story_structure`.

## Sampling requirements (Visual)

Компонент линейный по \(N\), но качество hook‑метрик деградирует при слишком редком sampling.

- **min frames**: 60  
- **target frames**: 120  
- **max frames**: 200  

Если Segmenter выдаст `N > max_frames` ⇒ **fail-fast** (это ошибка sampling policy).

## Выход (артефакт)

Пишется через `BaseModule.save_results()` в:
- `result_store/<platform_id>/<video_id>/<run_id>/story_structure/story_structure_features_<ts>_<uid>.npz`

### Ключи NPZ
- **`frame_indices`**: `(N,) int32`
- **`times_s`**: `(N,) float32` — `union_timestamps_sec[frame_indices]`
- **`embedding_sim_next`**: `(N-1,) float32`
- **`embedding_diff_next`**: `(N-1,) float32` — cosine distance
- **`embedding_change_rate_per_sec`**: `(N,) float32` — `diff_next / dt` (с нулём в первом элементе)
- **`motion_norm_per_sec_mean`**: `(N,) float32` — aligned кривая движения из `core_optical_flow`
- **`any_face_present`**: `(N,) bool` — aligned наличие лица (из `core_face_landmarks.face_present`)
- **`story_energy_curve`**: `(N,) float32` — z-score сглаженного комбинированного сигнала (motion + embedding change rate)
- **`story_energy_curve_downsampled_128`**: `(128,) float32`
- **`subtitles_present`**: `bool`
- **`features`**: `object(dict)` — агрегаты (hook/climax/face proxies + placeholders topic features)
- **`meta`**: `object(dict)` — canonical meta (run identity keys + models_used/model_signature + status/empty_reason…)

## Фичи (`features`)

### Hook (первые секунды)
Окно хука: `min(5s, 15% video_length)`. Если sampling даёт меньше 3 точек в этом окне, модуль расширяет окно до первых 3 кадров (чтобы статистики не были вырожденными).

- **`hook_visual_surprise_score/std`**: mean/std `story_energy_curve` на hook.
- **`hook_motion_intensity`**, **`hook_cut_rate`**, **`hook_motion_spikes`**, **`hook_rhythm_score`**: по motion curve на hook.
- **`hook_face_presence`**: доля кадров hook с лицами.

### Climax / peaks
- **`climax_timestamp`**: union-domain кадр кульминации (индекс из `frame_indices`).
- **`climax_time_sec`**
- **`climax_position_normalized`**: позиция в [0..1] по sampled кадрам.
- **`climax_strength`**, **`climax_strength_normalized`** (z)
- **`number_of_peaks`**
- **`time_from_hook_to_climax`**: нормированное время от конца hook до climax.
- **`hook_to_avg_energy_ratio`**

### Character proxies (face presence)
- **`main_character_screen_time`**: доля sampled кадров с лицами.
- **`speaker_switch_rate`**: доля переходов “есть лицо/нет лица” между соседними кадрами.
- **`speaker_switches_per_minute`**

### Subtitles/topic features
Baseline хранит только:
- `has_subtitles` (bool)
- topic‑метрики как `NaN` (будут реализованы в non-baseline через SentenceTransformer).

## Legacy / non-baseline (опционально)

Экспериментальная логика (topic embeddings и т.п.) вынесена в `legacy_story_structure.py` и **должна** грузить модели через `dp_models.ModelManager`.
Локальные веса SentenceTransformer хранятся под `DP_MODELS_ROOT/text/sentence-transformers_all-MiniLM-L6-v2/` и описаны в `dp_models/spec_catalog/text/sentence-transformers_all-MiniLM-L6-v2.yaml`.


