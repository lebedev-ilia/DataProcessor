## `tempo_extractor` (Audio Tier‑0 baseline, required)

### Назначение

Оценивает **темп (BPM)** и простые ритмические признаки на базе `librosa`.

### Входы (строго, no‑fallback)

- **`audio/audio.wav`** (Segmenter)
- **`audio/segments.json`** family: `tempo` (sliding windows для устойчивого BPM)

Если `segments` пустой → **error**.

### Выходы (per-run storage)

NPZ пишет AudioProcessor в:
- `result_store/<platform_id>/<video_id>/<run_id>/tempo_extractor/*.npz`

Схема: `audio_npz_v1` (см. `docs/contracts/ARTIFACTS_AND_SCHEMAS.md`).

Полезные поля payload:
- `tempo_bpm_*` (mean/median/std)
- `windowed_bpm.times_sec`, `windowed_bpm.bpm`
- `warnings` (например `low_confidence`, `tempo_out_of_range`)

### Модели

ML модели **не используются** (signal processing only). `models_used[]` пустой.


