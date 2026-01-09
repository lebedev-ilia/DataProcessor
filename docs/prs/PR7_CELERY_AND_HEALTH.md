# PR‑7 — Celery (queue execution) + health endpoints (MVP)

Цель: добавить queue-based execution и health endpoints для worker.

## 1) Celery

- Celery app: `dp_queue/celery_app.py`
- Task: `dp_queue/tasks.py` → `dataprocessor.process_video_job(payload)`

Task (MVP) запускает root `main.py` в subprocess (в дальнейшем заменим на чистый python runner).

## 2) Health API

FastAPI app: `health/app.py`

Endpoints:
- `GET /health/live` — liveness
- `GET /health` — readiness (проверяет Redis, MinIO bucket, (опц.) Triton)

## 3) Docker compose

`docker-compose.yml` (PR‑7) добавляет:
- `dataprocessor-celery-worker` — Celery worker
- `dataprocessor-health` — HTTP health service (`:8080`)

Env vars:
- `CELERY_BROKER_URL`
- `CELERY_RESULT_BACKEND`
- `S3_ENDPOINT`, `S3_BUCKET`, `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`
- `TRITON_HTTP_URL` (optional)


