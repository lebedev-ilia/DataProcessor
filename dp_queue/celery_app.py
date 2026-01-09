from __future__ import annotations

import os

from celery import Celery


def _env(name: str, default: str | None = None) -> str:
    v = os.getenv(name, default)
    if v is None or v == "":
        raise RuntimeError(f"Missing required env var: {name}")
    return v


celery_app = Celery(
    "dataprocessor",
    broker=_env("CELERY_BROKER_URL", "redis://redis:6379/0"),
    backend=_env("CELERY_RESULT_BACKEND", os.getenv("CELERY_BROKER_URL", "redis://redis:6379/0")),
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
)

celery_app.autodiscover_tasks(["dp_queue"])


