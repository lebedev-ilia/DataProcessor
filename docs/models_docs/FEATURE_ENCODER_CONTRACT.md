# Feature Encoder Contract (v1) — unify variable-length component outputs for Transformers

Цель: после выполнения всех компонентов/процессоров мы получаем набор разнородных артефактов (NPZ) разной длины
(короткое видео ~120 кадров vs длинное ~72000 кадров). Этот слой (“feature encoder”) приводит их к **единому**
представлению фиксированного размера для моделей `baseline`/`v1 transformer`/`v2 transformer`.

Ключевая идея: **компоненты должны отдавать максимально информативные сырые сигналы**, а encoder отвечает за:
- нормализацию (разные fps/resolution/sampling),
- выравнивание по времени,
- сжатие/пулинг до фиксированного бюджета токенов,
- выдачу компактного embedding’а.

---

## 0) Модальности (MVP)

Проблема variable-length существует минимум в 2 модальностях:

- **Visual (video)**: последовательности по времени (кадры/шоты/переходы) от VisualProcessor.
- **Audio**: длинные временные ряды (loudness/tempo/onsets/CLAP и т.д.) от AudioProcessor.

Решение (MVP):
- делаем **отдельный encoder на модальность**: `VideoEncoder` и `AudioEncoder`,
- на выходе каждый выдаёт одинаковый формат (см. §3), чтобы fusion был простым.

Про Text (MVP):
- отдельный `TextEncoder` **не обязателен**, если текстовая модальность представлена в основном как **OCR events по кадрам**.
- OCR в MVP трактуем как **sparse events** и кодируем в `event_tokens` с cap (см. §3.1), либо через простые агрегаты.
- Если позже появятся длинные тексты (ASR/subtitles/comments) — добавим полноценный `TextEncoder`.

---

## 1) Входы encoder’а (source-of-truth)

Encoder читает **только артефакты** (NPZ) и `frames_dir/metadata.json`:

### 1.1 Time-axis (обязательно)

- `frames_dir/metadata.json.union_timestamps_sec` — **source-of-truth** времени.
- Любой компонент, который возвращает последовательности, обязан иметь:
  - `frame_indices (N,)` в union-domain,
  - либо `times_s (N,)` согласованный с `union_timestamps_sec[frame_indices]`.

### 1.2 Канонические типы выходов компонентов (входы encoder’а)

Компонент может отдавать 3 класса сигналов (все допустимы; рекомендуется минимум A):

- **A) Dense time-series** (по sampled позициям)
  - Пример: `core_optical_flow.motion_norm_per_sec_mean[t]`, `cut_detection.hard_score[t]`, `shot_quality.frame_features[t, :]`.

- **B) Sparse events** (список событий во времени)
  - Пример: hard cuts / fades / motion spikes: `[(time_s, type, strength, ...)]`.

- **C) Precomputed embeddings** (кадровые/клипово‑агрегированные)
  - Пример: `core_clip.frame_embeddings[t, D]`.

---

## 2) Минимальный список “базовых” артефактов (Visual, baseline)

Ниже — текущие стабильные контракты из `VisualProcessor` (NPZ):

### 2.1 Core providers (Tier-0)

- `core_clip/embeddings.npz`
  - `frame_indices (N,) int32`
  - `frame_embeddings (N, D) float32`
  - `shot_quality_prompts (P,) object`
  - `shot_quality_text_embeddings (P, D) float32`

- `core_optical_flow/flow.npz`
  - `frame_indices (N,) int32`
  - `motion_norm_per_sec_mean (N,) float32`
  - `dt_seconds (N,) float32`

- `core_depth_midas/depth.npz`
  - `frame_indices (N,) int32`
  - `depth_maps (N, H, W) float32`

- `core_object_detections/detections.npz`
  - `frame_indices (N,) int32`
  - `boxes/scores/class_ids/valid_mask/...`

- `core_face_landmarks/landmarks.npz`
  - `frame_indices (N,) int32`
  - `face_landmarks`, `face_present`, `has_any_face`, `empty_reason`

### 2.2 Modules (пример)

- `cut_detection/cut_detection_features_*.npz`
  - `frame_indices`, `times_s`
  - `features` (dict, scalar aggregates)
  - `detections` (dict, events lists: hard/soft/motion/jump/…)

- `cut_detection/cut_detection_model_facing_*.npz` (recommended)
  - schema: `VisualProcessor/modules/cut_detection/SCHEMA_MODEL_FACING.md`
  - dense per-pair curves + unified events stream (designed for FeatureEncoder input)

- `shot_quality/shot_quality_features_*.npz`
  - `frame_indices`, `frame_features (N,F)`, `quality_probs (N,P)`
  - shot-level агрегаты (по `cut_detection`), см. README модуля

- `video_pacing/video_pacing_features_*.npz`
  - `frame_indices`, `shot_boundary_frame_indices`
  - `features` (dict)

---

## 2.3 Audio (Tier‑0, baseline) — ожидаемые входы для `AudioEncoder`

AudioEncoder работает по тем же принципам:
- на входе **time-series** и/или **events** в секундах,
- на выходе — фиксированный бюджет токенов (§3).

### Source-of-truth time axis (audio)

В аудио нет union-domain кадров, поэтому “ось времени” — это **секунды**:
- любой payload обязан иметь либо:
  - `times_sec (T,) float32` для рядов,
  - либо `segment_centers_sec (N,) float32` для окон/сегментов,
  - либо `events_times_sec (E,) float32` для событий.

### Минимальный Tier‑0 набор (что уже есть в AudioProcessor)

Сейчас для baseline есть 3 required extractor’а (NPZ schema: `audio_npz_v1`):

- `clap_extractor/*.npz`
  - `embedding_sequence (N, D) float32`
  - `segment_centers_sec (N,) float32`
  - `embedding (D,) float32` (global mean)

- `loudness_extractor/*.npz`
  - segment-level скаляры: `rms`, `peak`, `dbfs`, `lufs` (может быть NaN)
  - (если есть) short-term RMS stats / флаги наличия `lufs_present`
  - таймлайны должны быть привязаны к `segments` через `segment_centers_sec` (или отдельный `times_sec`)

- `tempo_extractor/*.npz`
  - агрегаты: `tempo_bpm_mean/median/std`
  - `windowed_bpm.times_sec`, `windowed_bpm.bpm` (sliding windows)

### Рекомендация: привести все аудио выходы к одному виду (для encoder’а)

Чтобы AudioEncoder был простым и стабильным:
- time-series хранить как `(times_sec, values)` для каждой кривой,
- event streams хранить как `(event_times_sec, event_strength, event_type_id)`,
- всегда сохранять `meta` (sampling окна/step, sample_rate, hop_length если применимо).

---

## 3) Гибридный выход encoder’а (MVP contract)

Encoder выдаёт **фиксированный бюджет**, чтобы downstream transformer не зависел от длины видео:

### 3.1 Output tensors

Для каждого видео и для каждой модальности (VideoEncoder/AudioEncoder):

- **`global_embedding (D,) float32`**
  - один вектор на видео (D фиксирован: например 256/512/1024).

- **`summary_tokens (K, D) float32`**
  - фиксированное число токенов K (например 16/32).
  - семантика: “компрессия таймлайна” (multi-scale pooling / attention pooling).

- **`event_tokens (E, D) float32` (optional, capped)**
  - E ≤ `E_max` (например 64), формируются из событий (cut/motion/audio peaks/…).
  - если событий меньше — паддинг; если больше — top‑E по strength/priority.

### 3.2 Обязательная meta-информация

Для воспроизводимости encoder должен сохранить:
- `sampling_policy_version`, `analysis_width/height/fps`, `frame_indices` источника (если применимо)
- список входных артефактов (paths + schema_version)
- версию encoder’а (`encoder_version`, `weights_digest`, `model_signature` если trainable)

---

## 4) Правила выравнивания (alignment) и нормализации

### 4.1 Alignment rule (строго)

- Если компонент выдаёт `frame_indices`, то encoder мапит их в `times_s` через `union_timestamps_sec`.
- Смешивать time-series с разными `frame_indices` можно **только через time join**:
  - интерполяция/ресэмплинг на общую сетку времени (например fixed grid из M точек),
  - или event-based encoding (для sparse событий).

### 4.2 Нормализация (рекомендуемое правило)

Чтобы избежать “зависимости от длины видео”:
- для dense рядов использовать per-video robust normalization: median/MAD или quantile scaling;
- для embeddings — L2‑нормализация (если не сделана);
- для счетчиков — перевод в rate (per-minute/per-second) + лог‑скейл `log1p`.

---

## 5) Как encoder будет обучаться и нужен ли trainable encoder

### 5.1 Нужен ли вообще обучаемый encoder?

Да, **желательно** (особенно для v1/v2 transformer), потому что:
- он учится выделять информативные моменты независимо от длины (attention/pooling),
- он обучается “под задачу” (prediction targets), а не под эвристики компонентов,
- он стабилизирует распределения входов для модели.

Но MVP можно начать с **необучаемого** encoder’а (см. 5.3), чтобы быстро запустить пайплайн.

### 5.2 Основной режим обучения (рекомендация): end-to-end

Encoder обучается **вместе** с основной моделью (transformer):
- loss идёт с конечной цели (baseline/v1/v2),
- градиент проходит в encoder,
- encoder автоматически оптимизирует, какие токены/агрегации важны.

Плюс: можно использовать auxiliary losses (опционально):
- masked modeling по time-series (восстановление пропусков),
- contrastive (clip vs clip, same video vs different),
- consistency losses между resolutions/sampling (invariance к downscale и подсэмплу).

### 5.3 MVP без обучения (чтобы стартануть быстро)

Можно сделать “фиксированный encoder”:
- по каждому компоненту: robust normalize → fixed-grid pooling (mean/max/quantiles over M bins)
- затем linear projection → `D`
- summary tokens: M bins → (M,D) и взять top‑K/агрегаты

Это даст baseline качества и позволит собрать датасет для обучения trainable encoder позже.

---

## 6) Что важно для `cut_detection` под encoder

Рекомендация: добавить “model-facing” выход (не только агрегаты) — минимально:
- dense ряды по позиции (или времени): `hist_diff[t]`, `ssim_drop[t]`, `flow_mag[t]`, `hard_score[t]`
- события: `hard_cuts`, `soft_events`, `motion_cuts` с `time_s` и `strength`
- `meta`: `ssim_max_side`, `flow_max_side`, reuse `core_optical_flow` (да/нет)

Это даст encoder’у возможность самому учиться “что важно” и не зависеть от финальных порогов/постпроцессинга.


