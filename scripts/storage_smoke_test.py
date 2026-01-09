"""
PR-1 smoke test for storage adapter.

Usage:
  # FS mode
  TREND_STORAGE_BACKEND=fs TREND_FS_ROOT=/tmp/trendflow python scripts/storage_smoke_test.py

  # S3/MinIO mode (env vars are read by boto3 automatically + S3_ENDPOINT/S3_BUCKET/S3_PREFIX)
  TREND_STORAGE_BACKEND=s3 python scripts/storage_smoke_test.py
"""

from __future__ import annotations

import os
import time
import uuid

from storage import FileSystemStorage, S3Storage, load_storage_settings


def main() -> int:
    st = load_storage_settings()

    if st.backend == "fs":
        storage = FileSystemStorage(st.fs_root)
    else:
        storage = S3Storage(endpoint_url=st.s3_endpoint, bucket=st.s3_bucket, region=st.aws_region)

    # Keys include S3_PREFIX explicitly (canonical layout).
    key = f"{st.s3_prefix}/__smoke__/{time.strftime('%Y%m%d-%H%M%S')}_{uuid.uuid4().hex[:8]}.txt".lstrip("/")
    payload = f"ok {time.time()} pid={os.getpid()}\n".encode("utf-8")

    storage.atomic_write_bytes(key, payload, content_type="text/plain")
    got = storage.read_bytes(key)
    assert got == payload, (got, payload)

    assert storage.exists(key) is True
    listed = list(storage.list(f"{st.s3_prefix}/__smoke__".lstrip("/")))
    assert any(o.key == key for o in listed), "written key not found in list()"

    print("OK:", key)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


