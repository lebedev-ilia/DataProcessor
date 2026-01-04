## Model system rules (MVP) — полуфинал

Этот документ — **канонические правила** для версионирования, кэширования, воспроизводимости, Triton, observability и policy-решений вокруг ML-моделей в TrendFlow.

Источник решений: `MODELS_Q.md` (Round 1 + Round 2). Если в других документах встречаются противоречия — верным считается **этот** документ.

---

### 1) Термины и “версии”

- **`producer_version`**: версия кода *конкретного компонента* (core provider / module / audio / text). SemVer или git commit.
- **`dataprocessor_version`**: версия *оркестратора/пайплайна* (DataProcessor релиз). **Не равна** версии моделей.
- **`schema_version`**: версия схемы NPZ артефакта (контракт данных).
- **`feature_schema_version`**: версия схемы фичей для моделей прогноза (training/inference).
- **`model_signature`**: идентификатор “какой именно моделью и как” был получен результат (см. раздел 2).

Ключевое правило: **апдейт одной модели не требует bump `dataprocessor_version`**. Версии моделей живут отдельно и входят в `model_signature`.

---

### 2) Model signature (обязательно для кэша и воспроизводимости)

Для каждого NPZ артефакта в `meta` фиксируем список `models_used[]` (может быть пустым, если компонент моделей не вызывал, а только читал upstream артефакты).

Для каждой записи `models_used[]` фиксируем минимум:

- `model_name`: каноническое имя (HF repo id / Triton model name / алиас).
- `model_version`: pinned версия (HF `revision`/commit sha или tag; Triton `version`; наш semver/tag).
- `weights_digest`: sha256/etag/commit (чтобы различать “одинаковый version, разные веса”).
- `runtime`: `triton` | `inprocess`
- `engine`: `torch` | `onnx` | `tensorrt`
- `precision`: `fp32` | `fp16`
- `device`: `cuda:0` / `cuda:1` / `cpu`

`model_signature` компонента = функция от (`models_used` + `engine/precision/device`) и используется:

- в idempotency/cache key,
- в `manifest.json`,
- для дебага/аудита.

**Важно**: смена `engine` (например, TensorRT) считается **другой версией** (другой `model_signature`).

---

### 3) Где живёт mapping `component → model:version` (Triton)

Финально для MVP:

- **Source-of-truth**: профиль анализа в **БД** (пользователь/тариф выбирает профиль, он содержит mapping и параметры).
- `triton_models.yaml` (если появится) допускается **только** как dev/seed-утилита (инициализация дефолтных профилей), но **не** как источник правды в проде.
- Для каждого run мы сохраняем **resolved mapping** в `manifest.json` и/или `meta` NPZ, чтобы воспроизводимость не зависела от текущих настроек/изменений в БД.

---

### 4) Кэширование и idempotency

Политика MVP: **“новая модель = новый кэш” без исключений**. `model_compatibility_token` в MVP не вводим.

Idempotency/cache key компонента должен включать:

- `platform_id`, `video_id`
- `component`
- `config_hash`
- `sampling_policy_version`
- `producer_version`
- `schema_version`
- `model_signature` (через `models_used/engine/precision/device`)

Если ключ совпал и артефакт валиден — пересчёт не делаем.

---

### 5) Детерминизм (reproducibility)

- Допускаем небольшие расхождения из-за FP16/движка/разных GPU.
- В пределах одного окружения (тот же engine/precision/device) детерминизм — **best-effort**.
- В `meta/manifest` фиксируем: seed, engine, precision, device, версии CUDA/cuDNN (если применимо).
- Для сравнения результатов между окружениями используем tolerances; при этом обычно `model_signature` будет отличаться.

---

### 6) Retention / GC (хранение артефактов)

- Hard cap: `hard_cap_days = 60` (см. `docs/PRIVACY_AND_RETENTION.md`).
- GC разрешён:
  - держим минимум: последний успешный `run_id` на ключ `(platform_id, video_id, config_hash, sampling_policy_version, model_signature-set)` + **N=2** предыдущих (для дебага),
  - промежуточные/временные данные (например, `frames_dir`) — короткий TTL (по текущим правилам 7 дней).
- По запросу удаления обязаны удалить артефакты по `video_id` (см. `docs/PRIVACY_AND_RETENTION.md`).

---

### 7) Ошибки загрузки моделей / Triton

- **No-fallback** на альтернативные модели при недоступности выбранной модели: если модель отсутствует/не загружается — **fail-fast**.
- Retry допускается только для transient ошибок (network/timeout/503).
- В `manifest.json` пишем `error_code` (например: `model_load_failed`, `insufficient_gpu_memory`, `triton_unavailable`) и детали.

---

### 8) OOM / batch_size / resource-aware execution

- Оптимальный `batch_size` выбираем **перед запуском** на основе доступной памяти + чеклиста.
- Использованный `batch_size` фиксируем в `meta`.
- При OOM допускается уменьшение batch_size и retry (если компонент поддерживает) — это не “fallback модели”, а деградация параметров.

---

### 9) Precision / engine

- Default GPU precision: **FP16**, если модель валидирована; иначе FP32.
- Precision настраивается per model в конфиге run.
- Engine (torch/onnx/tensorrt) фиксируем и включаем в `model_signature`.

---

### 10) Multi-GPU scheduling (полуфинал)

Принято: **вариант B** — компоненты одного run могут выполняться на разных GPU.

Рекомендуемый MVP-старт:

- простой scheduler: “least-loaded GPU” + ограничения по памяти из чеклиста,
- возможность pinning для отдельных компонентов (например, самые тяжёлые).

---

### 11) Prediction: выбор модели, fallback, confidence

#### 11.1 Выбор модели прогноза

- По умолчанию используем **самую новую stable** (обычно v2).
- Роллауты делаем через A/B buckets (по user_id/video_id), логируем.

#### 11.2 Fallback (degraded-mode)

Гибрид:

- авто-degraded: если primary (v2) упал → пробуем v1 → затем baseline.
- если baseline ок — отдаём результат с `prediction_status="degraded"`.
- в UI добавляем действие “Re-run prediction with another model” (если доступно по тарифу). Это отдельная операция и может стоить доп. кредиты.

#### 11.3 Confidence / intervals

Отдаём “все варианты”, но дефолт:

- `confidence_score` (0..1)
- интервал `p10/p90`
- опционально: `p50`, `p05/p95`, `p025/p975` (если посчитано).

---

### 12) Text embeddings и языки

MVP: предпочтительно одна мультиязычная embedding‑модель (фиксированная размерность).

Пост‑MVP: допускаем “RU‑model + EN‑model”, но это требует:

- либо одинаковой размерности,
- либо явного усложнения схемы фичей и bump `feature_schema_version`.

---

### 13) Observability (ML layer)

Помимо Prometheus/Grafana/логов, предпочитаем отдельный **ML observability слой**:

- таблицы в БД с метриками per run/per component/per model_signature,
- UI/админка для просмотра (latency, error rate, OOM, cache hit, дрейф).

Минимум, который обязан быть в `manifest.json` per component:

- `status`, `started_at`, `finished_at`, `duration_ms`
- `producer_version`, `schema_version`
- `device_used`, `batch_size` (если применимо)
- `model_signature`/`models_used` (если применимо)
- `error`/`error_code`/warnings.

---

### 14) Тестирование

- PR: быстрые тесты (mocks + маленькие фикстуры).
- Nightly/weekly: e2e с реальными моделями на 2–3 маленьких видео.
- Golden tests для критичных компонентов (с float tolerances).
- Performance tests: budgets по latency/memory/throughput; регрессии блокируют релиз.

---

### 15) Лицензии моделей

Обязателен инвентарь `MODEL_LICENSES.md` (model_name, source, license, link, ограничения) + CI-проверка.

---

