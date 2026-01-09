from .base import Storage, StorageError, NotFoundError
from .fs import FileSystemStorage
from .settings import StorageSettings, load_storage_settings

# Optional dependency: boto3 is required only for S3Storage.
try:
    from .s3 import S3Storage  # type: ignore
except Exception:  # pragma: no cover
    S3Storage = None  # type: ignore

__all__ = [
    "Storage",
    "StorageError",
    "NotFoundError",
    "FileSystemStorage",
    "StorageSettings",
    "load_storage_settings",
]

if S3Storage is not None:
    __all__.append("S3Storage")


