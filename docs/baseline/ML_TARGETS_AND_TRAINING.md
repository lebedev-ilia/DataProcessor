# ML targets и обучение (полуфинал)

## 1) Таргеты

MVP решения:
- UI показывает **абсолютный прогноз** (views/likes).
- Модель: **multi-target** (views + likes).
- **multi-horizon**: основа = 14d и 21d (полные), 7d — частичный head с mask.

## 2) Как считать таргеты (дельты)

Так как снапшоты содержат кумулятивные счётчики, таргет считаем как прирост:
- `delta_views_h = viewCount(t_h) - viewCount(t0)`
- `delta_likes_h = likeCount(t_h) - likeCount(t0)`

Нормализация:
- `y = log1p(delta)`

Для отсутствующих горизонтов:
- сохраняем missing mask и выключаем loss по этому head.

## 3) Входы модели и leakage

Полуфинальное правило:
- вход модели = только snapshot1 (никаких future-derived фичей).

## 4) Cold-start и возраст видео

Так как snapshot1 часто снят не “в момент публикации”, вводим обязательную фичу:
- `video_age_hours_at_snapshot1 = snapshot1_time - publishedAt`

И считаем качество по buckets:
- `<24h`, `1–30d`, `>30d`

MVP: **одна универсальная модель** (не отдельная cold-start).

## 5) Splits

Полуфинал:
- time-split по `publishedAt` (доверяем)
- группировка по каналам (channel-group split) после enrichment `channel_id`

## 6) Архитектуры (baseline → v2)

Обязательная контрольная точка:
- baseline (CatBoost/LightGBM) на табличных фичах.

Prod старт:
- v2 multimodal transformer:
  - token = shot
  - `max_len_shots = 256`
  - embeddings (`E_shot`, `A_shot`) + masks
  - meta_token (возраст, длительность, категория, язык, channel stats)

v1 late-fusion — как sanity-check/fallback (даже если v2 основной).

## 7) Reproducibility

Фиксируем в артефактах/manifest:
- `dataprocessor_version`
- `sampling_policy_version`
- `schema_version` всех артефактов
- commit hash (если доступно)
- seed


