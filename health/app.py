from __future__ import annotations

import os
from urllib.parse import urlparse

import boto3
import redis
import requests
from fastapi import FastAPI


def _env(name: str, default: str | None = None) -> str | None:
    v = os.getenv(name, default)
    if v is None or v == "":
        return None
    return v


app = FastAPI(title="DataProcessor worker health", version="0.1")


@app.get("/health/live")
def live() -> dict:
    return {"status": "ok"}


def _check_redis() -> None:
    broker = _env("CELERY_BROKER_URL", "redis://redis:6379/0")
    u = urlparse(broker)
    host = u.hostname or "redis"
    port = u.port or 6379
    db = int((u.path or "/0").lstrip("/") or "0")
    r = redis.Redis(host=host, port=port, db=db, socket_connect_timeout=2, socket_timeout=2)
    if r.ping() is not True:
        raise RuntimeError("redis ping failed")


def _check_minio() -> None:
    endpoint = _env("S3_ENDPOINT")
    bucket = _env("S3_BUCKET")
    if not endpoint or not bucket:
        raise RuntimeError("S3_ENDPOINT/S3_BUCKET missing")
    s3 = boto3.client("s3", endpoint_url=endpoint)
    buckets = s3.list_buckets().get("Buckets") or []
    names = {b.get("Name") for b in buckets if isinstance(b, dict)}
    if bucket not in names:
        raise RuntimeError(f"s3 bucket not found: {bucket}")


def _check_triton() -> None:
    # optional (only if TRITON_HTTP_URL is set)
    triton = _env("TRITON_HTTP_URL")
    if not triton:
        return
    url = triton.rstrip("/") + "/v2/health/ready"
    r = requests.get(url, timeout=2)
    if r.status_code != 200:
        raise RuntimeError(f"triton not ready: http {r.status_code}")


@app.get("/health")
def health() -> dict:
    checks: dict[str, dict] = {}
    ok = True

    def run(name: str, fn):
        nonlocal ok
        try:
            fn()
            checks[name] = {"ok": True}
        except Exception as e:
            ok = False
            checks[name] = {"ok": False, "error": str(e)}

    run("redis", _check_redis)
    run("minio", _check_minio)
    run("triton", _check_triton)

    return {"ok": ok, "checks": checks}


