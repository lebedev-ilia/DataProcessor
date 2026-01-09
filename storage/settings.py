from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class StorageSettings:
    backend: str  # "fs" | "s3"

    # FS
    fs_root: str

    # S3/MinIO
    s3_endpoint: str
    s3_bucket: str
    s3_prefix: str
    aws_region: str


def _get(name: str, default: str = "") -> str:
    v = os.getenv(name)
    if v is None:
        return default
    return v


def load_storage_settings() -> StorageSettings:
    """
    Loads storage settings from env vars.

    We intentionally support standard boto3 names (AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY)
    but those are handled by boto3 itself; here we only read endpoint/bucket/prefix.
    """
    backend = _get("TREND_STORAGE_BACKEND", "s3").strip().lower()
    if backend not in ("fs", "s3"):
        raise ValueError("TREND_STORAGE_BACKEND must be 'fs' or 's3'")

    return StorageSettings(
        backend=backend,
        fs_root=_get("TREND_FS_ROOT", "_runs"),
        s3_endpoint=_get("S3_ENDPOINT", "http://localhost:9000"),
        s3_bucket=_get("S3_BUCKET", "trendflow"),
        s3_prefix=_get("S3_PREFIX", "trendflowml"),
        aws_region=_get("AWS_DEFAULT_REGION", "us-east-1"),
    )


