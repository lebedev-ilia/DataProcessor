from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Protocol


class StorageError(RuntimeError):
    pass


class NotFoundError(StorageError):
    pass


@dataclass(frozen=True)
class ObjectInfo:
    key: str
    size_bytes: Optional[int] = None
    etag: Optional[str] = None


class Storage(Protocol):
    """
    Minimal storage interface (FS/S3).

    Keys are *relative* logical paths (e.g. "trendflowml/result_store/youtube/<video>/<run>/...").
    The storage implementation maps them to either filesystem paths or S3 object keys.
    """

    def exists(self, key: str) -> bool: ...

    def list(self, prefix: str) -> Iterable[ObjectInfo]: ...

    def read_bytes(self, key: str) -> bytes: ...

    def write_bytes(self, key: str, data: bytes, *, content_type: Optional[str] = None) -> None: ...

    def atomic_write_bytes(self, key: str, data: bytes, *, content_type: Optional[str] = None) -> None: ...


