## Phase 5 — Расширение набора экстракторов и стабилизация вывода

Цели фазы:
- Добавить новые лёгкие экстракторы без тяжёлых зависимостей: tempo, loudness, onset, chroma, spectral, VAD, quality.
- Обеспечить стабильную схему manifest: исключить None/пропуски, добавлять дефолтные значения.
- Уменьшить шум предупреждений от торчаудио и сторонних библиотек.

Основные изменения:
- Реализованы экстракторы:
  - `tempo` (librosa): BPM и статистики.
  - `loudness`: RMS, peak, dBFS, LUFS (при наличии pyloudnorm).
  - `onset` (librosa): онсеты, интервалы и плотность.
  - `chroma` (librosa): 12-полосные хрома + статистики.
  - `spectral` (librosa): centroid, bandwidth, flatness, rolloff, ZCR, contrast + статистики.
  - `vad`: Silero VAD с фолбэком на энерго-бейзлайн.
  - `quality`: dc offset, clipping ratio, crest factor, dynamic range, SNR.
- Зарегистрированы в `MainProcessor` и включены в раннер.
- Нормализация flattening: гарантируются значения по умолчанию, устранены неоднозначности с numpy.
- Подавлены шумные предупреждения torchaudio в `run_local_processing.py`.

Результаты:
- Полный manifest без None/потерянных полей.
- Расширенный отчёт PERFORMANCE_METRICS с новыми экстракторами и обновлёнными суммарными временами.

Следующие шаги (опционально):
- Добавить SpeechBrain-based Диаризацию как опцию.
- Вынести сбор метрик в отдельный модуль и автоматизировать агрегацию.

