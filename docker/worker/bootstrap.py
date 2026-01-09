import os
import sys
import time
from urllib.parse import urlparse

import boto3
import redis


def _env(name: str) -> str:
    v = os.getenv(name)
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v


def check_redis() -> None:
    broker = _env("CELERY_BROKER_URL")
    u = urlparse(broker)
    if u.scheme not in ("redis", "rediss"):
        raise RuntimeError(f"CELERY_BROKER_URL must be redis://... got: {broker}")
    host = u.hostname or "redis"
    port = u.port or 6379
    db = int((u.path or "/0").lstrip("/") or "0")

    r = redis.Redis(host=host, port=port, db=db, socket_connect_timeout=3, socket_timeout=3)
    pong = r.ping()
    if pong is not True:
        raise RuntimeError("Redis PING failed")


def check_minio_bucket() -> None:
    endpoint = _env("S3_ENDPOINT")
    bucket = _env("S3_BUCKET")

    # boto3 uses AWS_* env vars automatically; endpoint_url must be explicit for MinIO.
    s3 = boto3.client("s3", endpoint_url=endpoint)

    # Ensure bucket exists (minio-init should have created it). Here we just validate.
    buckets = s3.list_buckets().get("Buckets") or []
    names = {b.get("Name") for b in buckets if isinstance(b, dict)}
    if bucket not in names:
        raise RuntimeError(f"S3 bucket not found: {bucket}. Found: {sorted(n for n in names if n)}")


def main() -> int:
    print("[bootstrap] starting checks...")
    t0 = time.time()
    check_redis()
    print("[bootstrap] redis OK")
    check_minio_bucket()
    print("[bootstrap] minio bucket OK")
    dt = time.time() - t0
    print(f"[bootstrap] all OK in {dt:.2f}s")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as e:
        print("[bootstrap] FAILED:", repr(e), file=sys.stderr)
        raise


