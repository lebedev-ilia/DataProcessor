from __future__ import annotations

from typing import Iterable, Optional

# boto3/botocore are required in S3/MinIO mode, but may be absent in some dev envs.
# pyright: ignore[reportMissingImports]
import boto3  # type: ignore
# pyright: ignore[reportMissingImports]
from botocore.exceptions import ClientError  # type: ignore

from .base import NotFoundError, ObjectInfo


class S3Storage:
    """
    S3/MinIO backend.

    Atomicity notes:
    - A single PUT is atomic from the reader perspective (you either get old or new object).
    - There is no true rename; we treat overwrite PUT as "atomic write" for our use cases.
    """

    def __init__(self, *, endpoint_url: str, bucket: str, region: str = "us-east-1") -> None:
        self.endpoint_url = endpoint_url
        self.bucket = bucket
        self.region = region
        self._client = boto3.client("s3", endpoint_url=self.endpoint_url, region_name=self.region)

    def _k(self, key: str) -> str:
        # We treat incoming keys as full object keys inside the bucket,
        # including S3_PREFIX when used (e.g. "trendflowml/result_store/...").
        return str(key).lstrip("/")

    def exists(self, key: str) -> bool:
        k = self._k(key)
        try:
            self._client.head_object(Bucket=self.bucket, Key=k)
            return True
        except ClientError as e:
            code = str(e.response.get("Error", {}).get("Code", ""))
            if code in ("404", "NoSuchKey", "NotFound"):
                return False
            raise

    def list(self, prefix: str) -> Iterable[ObjectInfo]:
        pfx = self._k(prefix).rstrip("/") + "/"
        paginator = self._client.get_paginator("list_objects_v2")
        out = []
        for page in paginator.paginate(Bucket=self.bucket, Prefix=pfx):
            for obj in page.get("Contents") or []:
                k = obj.get("Key")
                if not isinstance(k, str):
                    continue
                out.append(ObjectInfo(key=k, size_bytes=obj.get("Size"), etag=obj.get("ETag")))
        return out

    def read_bytes(self, key: str) -> bytes:
        k = self._k(key)
        try:
            resp = self._client.get_object(Bucket=self.bucket, Key=k)
            return resp["Body"].read()
        except ClientError as e:
            code = str(e.response.get("Error", {}).get("Code", ""))
            if code in ("404", "NoSuchKey", "NotFound"):
                raise NotFoundError(f"S3 key not found: {key}") from e
            raise

    def write_bytes(self, key: str, data: bytes, *, content_type: Optional[str] = None) -> None:
        k = self._k(key)
        extra = {}
        if content_type:
            extra["ContentType"] = content_type
        self._client.put_object(Bucket=self.bucket, Key=k, Body=data, **extra)

    def atomic_write_bytes(self, key: str, data: bytes, *, content_type: Optional[str] = None) -> None:
        # See class docstring: we treat PUT overwrite as atomic enough here.
        self.write_bytes(key, data, content_type=content_type)


