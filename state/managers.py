from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Any, Dict, Optional

from storage.base import Storage
from storage.fs import FileSystemStorage
from storage.paths import KeyLayout

from .enums import Status


def _utc_iso_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _json_bytes(payload: Dict[str, Any]) -> bytes:
    return (json.dumps(payload, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def _safe_json_load(b: bytes) -> Dict[str, Any]:
    try:
        v = json.loads(b.decode("utf-8"))
        return v if isinstance(v, dict) else {}
    except Exception:
        return {}


@dataclass(frozen=True)
class StatePaths:
    run_state_key: str
    events_key: str
    processor_state_key: str


def build_state_paths(
    layout: KeyLayout,
    *,
    platform_id: str,
    video_id: str,
    run_id: str,
    processor_name: str,
) -> StatePaths:
    base = layout.state_run_prefix(platform_id, video_id, run_id)
    return StatePaths(
        run_state_key=f"{base}/run_state.json",
        events_key=f"{base}/state_events.jsonl",
        processor_state_key=f"{base}/state_{processor_name}.json",
    )


class _JournalWriter:
    def __init__(self, storage: Storage, events_key: str) -> None:
        self.storage = storage
        self.events_key = events_key

    def append_event(self, event: Dict[str, Any]) -> None:
        line = (json.dumps(event, ensure_ascii=False) + "\n").encode("utf-8")

        # Fast path for FileSystemStorage: append to file without read/overwrite.
        if isinstance(self.storage, FileSystemStorage) and hasattr(self.storage, "_abs"):
            abs_path = self.storage._abs(self.events_key)  # type: ignore[attr-defined]
            os.makedirs(os.path.dirname(abs_path), exist_ok=True)
            with open(abs_path, "ab") as f:
                f.write(line)
            return

        # Generic path: read + append + atomic overwrite (OK for MVP/low volume).
        existing = b""
        if self.storage.exists(self.events_key):
            try:
                existing = self.storage.read_bytes(self.events_key)
            except Exception:
                existing = b""
        self.storage.atomic_write_bytes(self.events_key, existing + line, content_type="application/jsonl")


class ProcessorStateManager:
    """
    Level-3 manager: owns a single processor state-file `state_<processor>.json`.
    In MVP, root orchestrator calls it (single-process), but the write semantics
    are compatible with future multi-writer setup (each processor owns its file).
    """

    def __init__(
        self,
        *,
        storage: Storage,
        layout: KeyLayout,
        platform_id: str,
        video_id: str,
        run_id: str,
        processor_name: str,
        run_meta: Dict[str, Any],
    ) -> None:
        self.storage = storage
        self.layout = layout
        self.platform_id = platform_id
        self.video_id = video_id
        self.run_id = run_id
        self.processor_name = processor_name

        self.paths = build_state_paths(
            layout,
            platform_id=platform_id,
            video_id=video_id,
            run_id=run_id,
            processor_name=processor_name,
        )
        self.journal = _JournalWriter(storage, self.paths.events_key)

        self.state: Dict[str, Any] = {
            "schema_version": "state_processor_v1",
            "processor": {
                "name": processor_name,
                "status": Status.waiting.value,
                "started_at": None,
                "finished_at": None,
                "duration_ms": None,
                "error": None,
                "error_code": None,
            },
            "run": dict(run_meta),
            # Level-4 sections (modules/components) live here for this processor
            "components": {},
            "updated_at": _utc_iso_now(),
        }

    def load_if_exists(self) -> None:
        if not self.storage.exists(self.paths.processor_state_key):
            return
        self.state = _safe_json_load(self.storage.read_bytes(self.paths.processor_state_key)) or self.state

    def _flush(self) -> None:
        self.state["updated_at"] = _utc_iso_now()
        self.storage.atomic_write_bytes(self.paths.processor_state_key, _json_bytes(self.state), content_type="application/json")

    def flush(self) -> None:
        """Public flush (used to materialize initial waiting state)."""
        self._flush()

    def set_status(
        self,
        status: Status,
        *,
        error: Optional[str] = None,
        error_code: Optional[str] = None,
        started_at: Optional[str] = None,
        finished_at: Optional[str] = None,
        duration_ms: Optional[int] = None,
    ) -> None:
        p = self.state.setdefault("processor", {})
        p["status"] = status.value
        if started_at is not None:
            p["started_at"] = started_at
        if finished_at is not None:
            p["finished_at"] = finished_at
        if duration_ms is not None:
            p["duration_ms"] = int(duration_ms)
        if error is not None:
            p["error"] = error
        if error_code is not None:
            p["error_code"] = error_code

        self.journal.append_event(
            {
                "ts": _utc_iso_now(),
                "scope": "processor",
                "processor": self.processor_name,
                "status": status.value,
                "error_code": error_code,
            }
        )
        self._flush()

    def upsert_component(
        self,
        *,
        component_name: str,
        status: Status,
        artifacts: Optional[list[dict[str, Any]]] = None,
        error: Optional[str] = None,
        error_code: Optional[str] = None,
        notes: Optional[str] = None,
        device_used: Optional[str] = None,
        started_at: Optional[str] = None,
        finished_at: Optional[str] = None,
        duration_ms: Optional[int] = None,
    ) -> None:
        comps = self.state.setdefault("components", {})
        comps[component_name] = {
            "name": component_name,
            "status": status.value,
            "artifacts": artifacts or [],
            "error": error,
            "error_code": error_code,
            "notes": notes,
            "device_used": device_used,
            "started_at": started_at,
            "finished_at": finished_at,
            "duration_ms": duration_ms,
            "updated_at": _utc_iso_now(),
        }
        self.journal.append_event(
            {
                "ts": _utc_iso_now(),
                "scope": "component",
                "processor": self.processor_name,
                "component": component_name,
                "status": status.value,
                "error_code": error_code,
            }
        )
        self._flush()


class RunStateManager:
    """
    Level-2 manager: owns `run_state.json` (aggregated view).
    In MVP, we just merge processor state files into one run_state snapshot.
    """

    def __init__(
        self,
        *,
        storage: Storage,
        layout: KeyLayout,
        platform_id: str,
        video_id: str,
        run_id: str,
        run_meta: Dict[str, Any],
    ) -> None:
        self.storage = storage
        self.layout = layout
        self.platform_id = platform_id
        self.video_id = video_id
        self.run_id = run_id
        self.run_meta = dict(run_meta)

        self.paths = build_state_paths(
            layout,
            platform_id=platform_id,
            video_id=video_id,
            run_id=run_id,
            processor_name="__run__",
        )
        self.journal = _JournalWriter(storage, self.paths.events_key)

        self.state: Dict[str, Any] = {
            "schema_version": "run_state_v1",
            "run": dict(run_meta),
            "processors": {},
            "updated_at": _utc_iso_now(),
        }

    def load_if_exists(self) -> None:
        if not self.storage.exists(self.paths.run_state_key):
            return
        self.state = _safe_json_load(self.storage.read_bytes(self.paths.run_state_key)) or self.state

    def _flush(self) -> None:
        self.state["updated_at"] = _utc_iso_now()
        self.storage.atomic_write_bytes(self.paths.run_state_key, _json_bytes(self.state), content_type="application/json")

    def init(self, processors: list[str]) -> None:
        self.state = {
            "schema_version": "run_state_v1",
            "run": {**self.run_meta, "created_at": self.run_meta.get("created_at") or _utc_iso_now()},
            "processors": {
                p: {
                    "name": p,
                    "status": Status.waiting.value,
                    "started_at": None,
                    "finished_at": None,
                    "duration_ms": None,
                    "error": None,
                    "error_code": None,
                }
                for p in processors
            },
            "updated_at": _utc_iso_now(),
        }
        self.journal.append_event({"ts": _utc_iso_now(), "scope": "run", "event": "init"})
        self._flush()

    def merge_processor_state(self, processor_name: str, processor_state: Dict[str, Any]) -> None:
        p = (processor_state.get("processor") or {}) if isinstance(processor_state, dict) else {}
        status = p.get("status") or Status.waiting.value
        self.state.setdefault("processors", {})
        self.state["processors"][processor_name] = {
            "name": processor_name,
            "status": status,
            "started_at": p.get("started_at"),
            "finished_at": p.get("finished_at"),
            "duration_ms": p.get("duration_ms"),
            "error": p.get("error"),
            "error_code": p.get("error_code"),
            # Keep a compact snapshot of leaf components (Level-4) for UI.
            "components": processor_state.get("components") if isinstance(processor_state.get("components"), dict) else {},
        }
        self._flush()


