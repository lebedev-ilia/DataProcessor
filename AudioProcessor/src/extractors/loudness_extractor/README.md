## `loudness_extractor` (Audio Tier‑0 baseline, required)

### Назначение

Считает **громкость/динамику**:
- RMS, peak, dBFS,
- опционально LUFS (если установлен `pyloudnorm`),
- статистики по short-term RMS.

### Входы (строго, no‑fallback)

- **`audio/audio.wav`** (Segmenter)
- **`audio/segments.json`** family: `primary` (окна вокруг time‑anchors)

Если `segments` пустой → **error**.

### Выходы (per-run storage)

NPZ пишет AudioProcessor в:
- `result_store/<platform_id>/<video_id>/<run_id>/loudness_extractor/*.npz`

Схема: `audio_npz_v1` (см. `docs/contracts/ARTIFACTS_AND_SCHEMAS.md`).

Полезные поля payload:
- `rms`, `peak`, `dbfs`, `lufs` (может быть NaN)
- `segment_rms_*` агрегаты
- `lufs_present` (bool)

### Модели

ML модели **не используются** (signal processing only). `models_used[]` пустой.


