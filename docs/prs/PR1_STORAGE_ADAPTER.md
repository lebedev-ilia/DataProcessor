## PR-1: Storage adapter (FS + S3/MinIO)

Цель PR‑1: дать единую абстракцию чтения/записи артефактов для:
- `result_store/...` (NPZ + `manifest.json`)
- `state/...` (state-files + event journal)
- `frames_dir/...` (если переносим во внешнее хранилище)

В PR‑1 мы **не ломаем** текущий pipeline: просто добавляем слой `storage/` и smoke-test.
Интеграция в writers (manifest/NPZ/state) — начинается в PR‑2/PR‑5 по execution plan.

### 1) Конфигурация через env vars

- `TREND_STORAGE_BACKEND`: `fs` | `s3` (default `s3`)
- FS:
  - `TREND_FS_ROOT` (default `_runs`)
- S3/MinIO:
  - `S3_ENDPOINT` (например `http://minio:9000`)
  - `S3_BUCKET` (например `trendflow`)
  - `S3_PREFIX` (например `trendflowml`)
  - `AWS_ACCESS_KEY_ID`, `AWS_SECRET_ACCESS_KEY`, `AWS_DEFAULT_REGION`

### 2) Canonical key layout (MVP)

Один bucket, разделение через prefix:
- `<S3_PREFIX>/result_store/<platform>/<video>/<run>/...`
- `<S3_PREFIX>/state/<platform>/<video>/<run>/...`
- `<S3_PREFIX>/frames_dir/<platform>/<video>/<run>/...`

Реализовано в `storage/paths.py::KeyLayout`.

### 3) Smoke test

FS:
```bash
TREND_STORAGE_BACKEND=fs TREND_FS_ROOT=/tmp/trendflow python scripts/storage_smoke_test.py
```

MinIO (через docker compose, если `.env` выставлен):
```bash
TREND_STORAGE_BACKEND=s3 python scripts/storage_smoke_test.py
```

Ожидаемо: `OK: <key>`


