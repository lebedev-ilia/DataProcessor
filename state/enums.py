from __future__ import annotations

from enum import Enum


class Status(str, Enum):
    waiting = "waiting"
    running = "running"
    success = "success"
    empty = "empty"
    error = "error"
    skipped = "skipped"


