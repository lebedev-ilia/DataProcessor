## PR-0: Local dev/prod-like stack (Redis + MinIO + bootstrap worker)

Этот документ фиксирует, как поднять **локальный инфраструктурный стенд** для production‑baseline.

### 1) Подготовка env

В корне репозитория:
- скопируй `env.example` → `.env`
- при необходимости поменяй значения (пароли/бакет/префикс)

Важно: `.env` уже в `.gitignore`.

### 2) Запуск

Из корня репозитория:

```bash
docker compose up --build
```

Что поднимется:
- `redis` (broker)
- `minio` (S3-compatible storage)
- `minio-init` (создание bucket)
- `dataprocessor-worker` (bootstrap: проверка Redis/MinIO; это НЕ полный ML worker)

### 3) Triton (опционально в PR-0)

Пока Triton включён как profile, потому что GPU runtime зависит от твоей локальной настройки Docker.

```bash
docker compose --profile triton up --build
```

### 4) Проверка

Ожидаемый результат:
- `dataprocessor-worker` пишет в логи:
  - `redis OK`
  - `minio bucket OK`

Если падает:
- проверь, что `.env` создан и содержит `MINIO_ROOT_USER/MINIO_ROOT_PASSWORD/S3_BUCKET`.


