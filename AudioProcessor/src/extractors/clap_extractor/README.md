## `clap_extractor` (Audio Tier‑0 baseline, required)

### Назначение

Считает **семантический аудио‑эмбеддинг** CLAP по Segmenter‑окнам и отдаёт:
- эмбеддинг по каждому окну,
- агрегат (mean) по всему видео.

### Входы (строго, no‑fallback)

- **`audio/audio.wav`**: готовит Segmenter.
- **`audio/segments.json`**: contract `audio_segments_v1`, family: `primary` (окна вокруг time‑anchors).

Если `segments` пустой → **error**.

### Выходы (per-run storage)

NPZ пишет AudioProcessor в:
- `result_store/<platform_id>/<video_id>/<run_id>/clap_extractor/*.npz`

Схема: `audio_npz_v1` (см. `docs/contracts/ARTIFACTS_AND_SCHEMAS.md`).

Полезные поля payload (внутри NPZ):
- `embedding` (`float32[D]`)
- `embedding_sequence` (`float32[N,D]`)
- `segment_centers_sec` (`float32[N]`)
- `device_used`

### Модель / ModelManager

Модель CLAP грузится **строго локально** через `dp_models`:
- spec: `dp_models/spec_catalog/audio/laion_clap.yaml`
- артефакт: `${DP_MODELS_ROOT}/audio/laion_clap/clap_ckpt.pt`

Сетевые загрузки запрещены.

### Производительность / batching

- Эмбеддинги считаются по сегментам; батчинг внутри экстрактора сейчас минимальный.
- В ресурсный чек-лист для DynamicBatching занесём: latency/VRAM как функцию длины окна и batch.


