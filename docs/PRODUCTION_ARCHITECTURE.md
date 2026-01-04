## Production architecture (MVP) — полуфинал

Цель: описать минимальную архитектуру, которая соответствует контрактам `result_store`/`manifest.json`, масштабируется и не смешивает ответственности.

### 1) Сервисы (MVP)

- **Backend API**: auth, баланс/кредиты, создание run, выдача результатов, админка.
- **Job queue + worker**: постановка задач на обработку, ретраи, лимиты параллелизма.
- **DataProcessor worker**: orchestrator пайплайна (Segmenter → Audio/Text → Visual → render).
- **Triton**: отдельный сервис model serving (GPU scheduling, batching, версии моделей).
- **PostgreSQL**: artifact index (videos/runs/components), кэш LLM-рендера, аудит.
- **Object storage**:
  - dev: локальный диск,
  - prod: **MinIO (S3-compatible)** как дефолтный бесплатный вариант.
- **LLM gateway**: единая точка вызова LLM (ключи, лимиты, логирование prompt_version).

### 2) Границы ответственности

- **DataProcessor не поднимает Triton**: он подключается к нему по endpoint.
- **NPZ артефакты** и `manifest.json` — файловый “source-of-truth” для каждого `run_id`.
- **БД** — ускоритель/индекс/поиск; если БД временно недоступна, по `manifest.json` можно восстановиться.

### 3) Run identity и воспроизводимость

Полуфинальное правило (Round 2, MVP):
- `run_id` генерирует **backend** и передаёт в DataProcessor.
- `config_hash` вычисляется **backend детерминированно** из нормализованного “profile config” (включённые компоненты + их параметры + версии policy) и передаётся в DataProcessor.
- DataProcessor не “придумывает” config_hash сам, а только прокидывает его во все подпроцессоры.
- DataProcessor обязан прокинуть `run_id/config_hash/sampling_policy_version/dataprocessor_version` во все подпроцессоры.

### 4) Кэширование heavy compute

- Heavy compute кэшируется на уровне видео и профиля.
- Политика повторного использования: если последний успешный run для `video_id` свежее порога `cache_ttl_days` (по умолчанию 3 дня, настраивается пользователем/профилем), используем кэш, иначе пересчитываем.

### 5) DB schema MVP (artifact index)

Полуфинальный минимум таблиц (Round 2):
- `users`: пользователи, баланс/кредиты
- `videos`: `video_id`, `platform_id`, метаданные (title/description/etc), `created_at`
- `runs`: `run_id`, `video_id`, `user_id`, `config_hash`, `sampling_policy_version`, `dataprocessor_version`, `status`, `created_at`, `finished_at`
- `run_components`: связь `run_id` ↔ `component_name`, `status`, `schema_version`, `producer_version`, `started_at`, `finished_at`, `duration_ms`, `device_used`, `error`
- `artifacts`: ссылки на NPZ/артефакты (путь, размер, хэш, компонент, run_id)
- `render_cache`: кэш LLM-рендера (ключ: `video_id`, `run_id`, `render_profile_id`, `llm_model`, `prompt_hash`, `locale`)
- `billing_ledger` (опционально): транзакции по кредитам/оплатам
- `analysis_profiles`: профили анализа (наши + пользовательские): `profile_id`, `user_id` (nullable для “наших”), `name`, `description`, `is_public`, `created_at`, `updated_at`
- `profile_components`: включённые компоненты и параметры: `profile_id`, `component_name`, `enabled`, `component_params_json`, `required` (bool)
- `profile_model_mapping`: mapping моделей для Triton/in-process: `profile_id`, `component_name`, `model_name`, `model_version`, `weights_digest` (опционально), `engine`, `precision`, `device_policy`

**Примечание**: БД служит ускорителем/индексом; source-of-truth остаются файлы (`manifest.json`, NPZ). Если БД недоступна, можно восстановиться из файлового хранилища.

### 6) Backend ↔ DataProcessor коммуникация

**Протокол (Round 3, MVP)**:
- **Queue-based (основной)**: backend кладёт задачи в очередь (Redis/RabbitMQ), DataProcessor worker подписывается
- **Преимущества**: масштабируемость, надёжность (persistent queue), декoupling, встроенная поддержка retry
- **Для MVP**: Redis + RQ (Python) или Celery — простые и надёжные варианты
- **Синхронный режим (опционально)**: отдельный HTTP endpoint `/process/sync` для тестов/dev (timeout 5 минут), в проде отключён

**Payload запроса** (обязательные поля):
- `run_id` (UUID, генерирует backend)
- `video_id` (каноничный ID)
- `platform_id` ("youtube")
- `config_hash` (детерминированный хэш профиля)
- `sampling_policy_version`, `dataprocessor_version`
- `video_source`: `{"type": "youtube_url", "url": "..."}` или `{"type": "upload", "path": "...", "meta": {...}}`
- `user_id` (для аудита и биллинга)

**Опциональные поля**:
- `profile_config` (полный JSON профиля) — передаём для первого запуска, потом можно кэшировать по `config_hash`
- `priority` (int, для будущей приоритизации)
- `callback_url` (webhook для уведомлений, опционально)

**Статус прогресса**:
- **Polling (основной)**: backend/UI опрашивает `/api/runs/{run_id}/status` каждые 2-5 секунд
- **Webhooks (опционально)**: DataProcessor может отправлять события в backend через webhook (если указан `callback_url`)
- **Статусы в БД**: worker обновляет статус в БД после каждого этапа (Segmenter → Audio → Text → Visual → Render)
- **Manifest.json**: также обновляется атомарно, можно читать напрямую из MinIO для восстановления

### 7) Batch processing

**Многоуровневый батчинг (Round 3)**:

**Backend-level batching**:
- Backend собирает запросы в буфер (5-10 секунд)
- Отправляет batch в очередь DataProcessor
- Пользователь видит "в очереди" сразу (не ждёт реальной обработки)

**DataProcessor-level batching**:
- **Динамический батчинг per component**: каждый компонент решает, сколько задач обрабатывать параллельно
- **Чек-лист ресурсов**: таблица `component_resource_requirements` в БД:
  - `memory_per_task` (MB)
  - `gpu_memory_per_task` (MB)
  - `cpu_cores_per_task`
- **Алгоритм**: worker читает задачи из очереди и группирует их в batch, учитывая доступные ресурсы
- **Пример**: если доступно 8GB GPU memory, а `face_emotion` требует 2GB per task → batch size = 4

**Приоритизация**: FIFO для MVP (первый пришёл — первый обслужен).

### 8) Масштабирование

**Горизонтальное масштабирование DataProcessor workers**:
- Несколько worker'ов читают из одной очереди (parallel processing)
- 1 worker = 1 видео одновременно (если 1 GPU на worker)
- Если несколько GPU: можно обрабатывать N видео параллельно (N = количество GPU)
- **Auto-scaling**: можно настроить автоматическое добавление worker'ов при росте очереди (Kubernetes HPA или простой скрипт)
- **Мониторинг**: отслеживаем `queue_length` и `worker_count` для принятия решений о масштабировании

**Triton deployment**:
- Отдельный сервер/контейнер для Triton (не на worker'ах)
- Несколько реплик Triton для высокой нагрузки (load balancing)
- Версии моделей фиксируются через **resolved mapping per-run** (source-of-truth: профиль анализа), см. `docs/MODEL_SYSTEM_RULES.md`

**MinIO/S3 доступ**:
- Единый endpoint + credentials (env vars) для MVP
- Все worker'ы используют одни и те же credentials
- MinIO должен быть доступен только внутри приватной сети (не публичный endpoint)

### 9) Deployment strategy

**MVP (простой вариант)**:
- Docker containers для каждого сервиса
- `docker-compose.yml` для локальной разработки
- Для прода: ручной deploy на сервер через `docker-compose up -d` или простой CI-CD (GitHub Actions → SSH deploy)

**Будущее (масштабирование)**:
- Kubernetes для оркестрации (auto-scaling, health checks, rolling updates)
- Helm charts для управления конфигурацией
- CI-CD pipeline (GitHub Actions → build images → push to registry → deploy to K8s)

**Старт**: можно начать с docker-compose, потом мигрировать на K8s когда понадобится масштабирование.

### 10) Мониторинг и observability

**Health check endpoints** (для всех сервисов):
- `/health` (readiness): проверяет, что сервис запущен и готов принимать запросы
  - Backend: БД доступна, очередь доступна
  - Worker: очередь доступна, MinIO доступен, Triton доступен (если требуется)
  - Triton: модели загружены
- `/health/live` (liveness): проверяет, что процесс не завис (просто возвращает 200)

**Метрики (Prometheus + Grafana)**:
- `runs_total` (counter): общее количество запущенных runs
- `runs_success_rate` (gauge): доля успешных runs за последний час
- `component_duration_seconds` (histogram): время обработки per component
- `gpu_utilization_percent` (gauge): использование GPU
- `queue_length` (gauge): длина очереди задач
- `errors_total` (counter by error_type): количество ошибок по типам

**Alerting (критичные алерты для MVP)**:
1. Сервис недоступен: health check failed > 2 минуты → критично
2. Очередь переполнена: `queue_length > 100` → предупреждение
3. Success rate упал: `runs_success_rate < 80%` за последний час → предупреждение
4. GPU OOM часто: `errors_total{error_type="out_of_memory"} > 10%` от всех runs → предупреждение
5. Среднее время обработки выросло: `component_duration_seconds > baseline × 2` → информационный алерт

**Каналы уведомлений**: email/Slack/Telegram (настраивается в Prometheus Alertmanager).

### 11) Безопасность

**Аутентификация backend ↔ DataProcessor**:
- **mTLS** (взаимная TLS-аутентификация) для прода
- Self-signed certificates для dev, Let's Encrypt или внутренний CA для prod
- Certificates в secrets (env vars или Kubernetes secrets)
- **Альтернатива для MVP**: можно начать с API key в заголовках (проще), потом мигрировать на mTLS

**Rate limiting**:
- **Backend API**: лимит per user зависит от подписки (free: 5 анализов/час, premium: 50 анализов/час)
- **Жёсткое ограничение**: максимум 100 анализов/час на пользователя (даже для premium) — защита от злоупотреблений
- **DataProcessor worker**: лимит на уровне очереди (не более N задач в секунду от одного backend)
- **Реализация**: использовать `slowapi` (FastAPI rate limiting) или Redis-based rate limiter

**Secrets management**:
- **MVP (простой вариант)**: env vars в `.env` файле (не в git), передаются через docker-compose или deployment скрипт
- **Следующий этап**: Kubernetes secrets (если используем K8s)
- **Будущее**: внешний secrets manager (HashiCorp Vault, AWS Secrets Manager)
- **Важно**: никогда не коммитить secrets в git, использовать `.gitignore` для `.env` файлов
- **Ротация**: планировать механизм ротации secrets (особенно для LLM API keys)

### 12) Версионирование моделей

Канонические правила версионирования/кэша/воспроизводимости описаны в `docs/MODEL_SYSTEM_RULES.md`.

Кратко (MVP полуфинал):
- `dataprocessor_version` = версия кода пайплайна (не версия моделей).
- Версии моделей фиксируются отдельно через mapping `component → model:version`, который пинится на run и сохраняется в `manifest.json`/NPZ meta как **resolved mapping**.
- Apдейт одной модели **не требует** bump `dataprocessor_version`; он меняет `model_signature` и, соответственно, cache keys.

**Mapping моделей (source-of-truth)**:
- Mapping `component → model:version` хранится в БД в профилях анализа и выбирается при создании run.
- В `manifest.json`/NPZ meta сохраняем **resolved mapping** для воспроизводимости.
- Любые файлы в репозитории для mapping (если появятся) допустимы только как dev/seed утилита, не как источник правды в проде.


