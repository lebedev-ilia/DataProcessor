## Текущие метрики производительности (CPU vs GPU)

Источник: логи запуска `run_local_processing.py` от 2025-10-29.

Важно: для `pitch` рекомендуется GPU (torchcrepe). На CPU время выше на порядок.

Сводно по последним прогонам (3 видео):
- CPU общее время: 102.46s
- GPU общее время: 84.30s
- Ускорение: 1.2x
- Экономия времени: 18.16s (17.7%)

Разбивка по видео (total):
- `-69HDT6DZEM.mp4`: CPU 45.77s → GPU 39.03s
- `-JuF2ivdnAg.mp4`: CPU 29.95s → GPU 23.58s
- `-niwQ0xGEGk.mp4`: CPU 26.75s → GPU 21.69s

Пер-экстрактор (из логов, средние ориентировочно):
- video_audio: ~0.17–0.19s (CPU≈GPU)
- mfcc: CPU ~0.02–0.03s; GPU ~0.01–0.11s
- mel: CPU ~0.02–0.04s; GPU ~0.02s
- clap: CPU ~0.21–0.47s; GPU ~0.04–0.09s
- pitch: CPU ~12.5–19.8s; GPU ~7.0–13.7s
- остальные (tempo, loudness, onset, chroma, spectral, vad, quality, rhythmic, voice_quality, hpss, key, band_energy, spectral_entropy): единицы десятых секунды; GPU немного быстрее/сопоставимо

Новые метрики стабильности/качества в экстракторах:
- mel: autocast на GPU; численная стабильность (клип по dB, расчёт центроида/полосы в линейной шкале)
- loudness: безопасный pyloudnorm fallback; frame-wise RMS статистики (+ вектор)
- key: распределение по 24 ключам, top-k, beat-sync опционально
- hpss: shared STFT, реконструкции в `.npy`, `hpss_kwargs`
- band_energy/chroma: векторизация, агрегаты, time-series/downsample/save
- source_separation: принудительный CPU путь внутри UMX при CUDA пайплайне (исправлен device mismatch)

Выводы:
- Ускорение GPU против CPU по текущему набору видео — ~1.2x (общий total), основная выгода — `pitch` и `clap`.
- Благодаря сохранению больших матриц в `.npy` manifest остаётся компактным.
- Логирование и fallbacks повышают устойчивость пайплайна (нет падений при отсутствии зависимостей).

---
История (предыдущие прогоны):
- Ранний прогон: CPU 350.97s; GPU 40.78s (ускорение 8.6x) — с «тяжёлой» конфигурацией pitch.
- После установки tiny-модели в pitch: CPU 54.02s; GPU 33.56s.
- Ускорение: 1.6x
- Экономия времени: 20.46s (37.9%)

Примерные времена по экстракторам (из логов; видео `-69HDT6DZEM.mp4`):
- CPU
  - video_audio: ~0.18s
  - mfcc: ~0.04s
  - mel: ~0.04s
  - clap: ~0.48s
  - pitch: ~18.41s
  - tempo: ~1.71s
  - loudness: ~0.03s
  - onset: ~0.03s
  - chroma: ~0.24s
  - spectral: ~0.23s
  - vad: ~1.59s
  - quality: ~0.03s
- GPU
  - video_audio: ~0.19s
  - mfcc: ~0.07s
  - mel: ~0.02s
  - clap: ~0.05s
  - pitch: ~12.76s
  - tempo: ~0.07s
  - loudness: ~0.04s
  - onset: ~0.04s
  - chroma: ~0.25s
  - spectral: ~0.25s
  - vad: ~1.14s
  - quality: ~0.04s

Примечания:
- Время `pitch` на GPU зависит от длительности и параметров torchcrepe; рекомендуется GPU.
- Время `vad` включает загрузку Silero из кеша; при оффлайн-режиме может варьироваться.
