# LLM rendering (presentation layer) — полуфинал

## 1) Роль LLM

LLM — это **presentation layer**, не источник истины.

LLM:
- читает компактный render-context, собранный из NPZ (агрегаты + highlights + missing flags)
- генерирует **только текст** (MVP)

LLM НЕ делает:
- вычисление чисел/метрик
- генерацию структурных данных для фронта (карточек/графиков) как source-of-truth

## 2) Guardrails

- Любые числа/факты в тексте должны быть взяты из render-context (NPZ-derived).
- Если данных нет (NaN/missing) → LLM явно пишет “данные недоступны”.

## 3) Воспроизводимость

Каждый LLM-рендер должен иметь версию:
- `llm_provider`
- `llm_model`
- `prompt_version`
- `prompt_hash`
- `locale`

Где хранить:
- в `manifest.json` или отдельном `render.json` рядом с результатом пользователя.

## 4) Кэширование

LLM‑рендер кэшируем отдельно от heavy compute:
- key: `(video_id, run_id, render_profile_id, llm_model, prompt_hash, locale)`

## 5) Язык

Рекомендация (полуфинал):
- язык текста = `locale` пользователя
- если `locale` не задан → RU default


