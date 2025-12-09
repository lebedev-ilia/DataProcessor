# AudioProcessor – Документация второй фазы

## Цель фазы
Сделать систему производственным сервисом: предоставить API, очередь задач, масштабирование, мониторинг, контейнеризацию и расширить набор экстракторов при сохранении GPU-эффективности и формата результатов из первой фазы.

## Ключевые результаты фазы (что будет реализовано)
- FastAPI сервис с синхронными и асинхронными эндпоинтами
- Очередь задач (Celery + Redis) для фоновой обработки и масштабирования
- Расширение набора экстракторов (минимум +2 из `DataProcessor/extractors`)
- Гибкое планирование CPU/GPU (пулы, семафоры, ограничения памяти)
- Хранилище результатов: локальный FS + S3 (опционально)
- Мониторинг: Prometheus метрики и структурированное логирование
- Контейнеризация: Docker образы, docker-compose; базовый Helm chart (опционально)
- E2E тесты API и нагрузочные прогоны (CPU/GPU)

## Архитектура второй фазы
```
AudioProcessor/
├── src/
│   ├── api/
│   │   ├── main.py               # Инициализация FastAPI, middlewares, метрики
│   │   └── endpoints.py          # /process, /batch, /status, /extractors
│   ├── workers/
│   │   └── celery_app.py         # Celery конфиг и задания
│   ├── services/
│   │   ├── storage.py            # FS/S3 сохранение артефактов/манифестов
│   │   └── metrics.py            # Prometheus, таймеры/счетчики
│   └── core/                     # Процессоры (sync/async) – reuse из фазы 1
├── config/
│   └── settings.py               # Расширено: брокер, бекенд результатов, S3
├── docker/
│   ├── Dockerfile.api            # Образ API
│   ├── Dockerfile.worker         # Образ worker
│   └── docker-compose.yml        # API + Redis + Workers
├── scripts/
│   ├── run_dev_api.sh            # Локальный запуск API
│   ├── run_worker.sh             # Локальный запуск Celery worker
│   └── load_test.sh              # Нагрузочный тест
└── tests/
    └── api_e2e/                  # Интеграционные тесты API
```

## API спецификация (черновик)
- POST `/process`
  - body: `{ input_uri: str, video_id: str, extractors?: string[], use_gpu?: bool }`
  - resp: `{ task_id: str }` (если async) или `{ manifest: ManifestModel }` (если sync)
- POST `/batch`
  - body: `{ items: ProcessRequest[], concurrency?: int, use_gpu?: bool }`
  - resp: `{ task_id: str }`
- GET `/status/{task_id}`
  - resp: `{ status: pending|processing|completed|failed, progress: number, error?: str }`
- GET `/extractors`
  - resp: `{ available: { name, version, device, description }[] }`

Примечание: формат manifest идентичен фазе 1 (`audio_manifest_v1`).

## Очередь задач (Celery)
- Брокер: Redis
- Роли:
  - API – принимает запросы, кладёт задачи в очередь
  - Worker – выбирает CPU/GPU воркер пулы, выполняет pipeline (извлечение аудио → экстракторы → manifest)
- Нагрузочное масштабирование: количество воркеров и провизия GPU-нод

## Планирование CPU/GPU
- Семафоры: `cpu_sem`, `gpu_sem`, `io_sem`
- Ограничение памяти GPU: эвристика по максимальной длине/батчу; graceful fallback на CPU
- Приоритизация задач: короткие/длинные задания, fair scheduling (по возможности)

## Хранилище результатов
- Локальный FS (по умолчанию) – директории проекта `tests/output_*`
- S3 (опционально) – `manifest_uri` заполняется ссылкой на S3
- Поведение при ошибках: атомарная запись файла, ретраи

## Мониторинг и логирование
- Prometheus метрики: 
  - таймеры по экстракторам, счетчики успехов/ошибок, загрузка GPU/CPU
- Структурные логи (JSON) с кореляционными ID
- Healthcheck: `/healthz` и `/readyz`

## Контейнеризация
- Два образа: API и Worker
- docker-compose: Redis, API, N воркеров
- Базовый Helm chart (опционально): параметры ресурсов, сервисы, HPA

## Расширение экстракторов (минимальный план)
- Добавить РОВНО 2 из каталога `DataProcessor/extractors`:
  - `tempo_extractor` – темпо/ритмика
  - `pitch_extractor` – основная частота + статистики
- Требования: GPU-совместимость (при наличии), единый интерфейс, быстрые агрегаты

## Настройки (дополнение)
- `CELERY_BROKER_URL`, `CELERY_RESULT_BACKEND`
- `USE_GPU`, `MAX_GPU_WORKERS`, `GPU_MEMORY_LIMIT`
- `ENABLE_METRICS`, `PROMETHEUS_PORT`
- `S3_*` (endpoint, bucket, creds)

## Тестирование
- E2E: 
  - одиночная обработка `/process` (CPU и GPU)
  - батч `/batch` на 10–50 роликов, сравнение времени CPU vs GPU
  - проверка структуры manifest и содержания
- Нагрузка:
  - 100 задач, `max_concurrent_videos` = 4/8, метрики throughput/latency

## План работ и приоритеты
1) API слой (эндпоинты, модели, валидация)
2) Celery интеграция (таски, сериализация, статусы)
3) Метрики и логирование
4) Хранилище (S3 фаза – опционально)
5) Экстракторы tempo/pitch
6) Контейнеризация и нагрузочный прогон

## Риски и меры
- Загрузка CLAP/Transformers – подавление шумных логов, таймауты, кеш моделей
- Недостаток VRAM – динамическая подстройка батча и длины окна, fallback CPU
- Большие видео – ограничение длительности, сегментация (фаза 3)

## Выходные артефакты фазы
- Запускаемый API (CPU/GPU)
- Воркеры Celery
- Докер-образы и docker-compose
- Метрики Prometheus и базовые дашборды
- Тестовые отчеты производительности CPU vs GPU
