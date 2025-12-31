from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


def _utc_iso_now() -> str:
    # Avoid datetime import churn; this is good enough for manifests.
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def _atomic_write_json(path: str, payload: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


@dataclass
class ManifestComponent:
    name: str
    kind: str  # "core" | "module" | "other"
    status: str  # "ok" | "empty" | "error"
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    duration_ms: Optional[int] = None
    artifacts: List[Dict[str, Any]] = field(default_factory=list)
    error: Optional[str] = None
    notes: Optional[str] = None
    producer_version: Optional[str] = None
    schema_version: Optional[str] = None


class RunManifest:
    def __init__(self, path: str, run_meta: Dict[str, Any]):
        self.path = path
        self.run_meta = dict(run_meta)
        self.components: Dict[str, ManifestComponent] = {}

        # If a manifest already exists, load it so multiple pipeline stages
        # can upsert without overwriting each other (e.g., audio -> visual).
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    existing = json.load(f) or {}
                existing_run = existing.get("run") or {}
                # Merge run metadata (explicit input wins).
                if isinstance(existing_run, dict):
                    # Drop derived field to avoid stale values; flush() will re-add.
                    existing_run.pop("updated_at", None)
                    self.run_meta = {**existing_run, **self.run_meta}

                comps = existing.get("components") or []
                if isinstance(comps, list):
                    for c in comps:
                        if not isinstance(c, dict):
                            continue
                        name = c.get("name")
                        if not isinstance(name, str) or not name:
                            continue
                        self.components[name] = ManifestComponent(
                            name=name,
                            kind=str(c.get("kind") or "other"),
                            status=str(c.get("status") or "error"),
                            started_at=c.get("started_at"),
                            finished_at=c.get("finished_at"),
                            duration_ms=c.get("duration_ms"),
                            artifacts=list(c.get("artifacts") or []),
                            error=c.get("error"),
                            notes=c.get("notes"),
                            producer_version=c.get("producer_version"),
                            schema_version=c.get("schema_version"),
                        )
            except Exception:
                # If existing manifest is corrupted, ignore and allow overwrite.
                self.components = {}

    def upsert_component(self, comp: ManifestComponent) -> None:
        self.components[comp.name] = comp
        self.flush()

    def flush(self) -> None:
        payload: Dict[str, Any] = {
            "run": {
                **self.run_meta,
                "updated_at": _utc_iso_now(),
            },
            "components": [
                {
                    "name": c.name,
                    "kind": c.kind,
                    "status": c.status,
                    "started_at": c.started_at,
                    "finished_at": c.finished_at,
                    "duration_ms": c.duration_ms,
                    "artifacts": c.artifacts,
                    "producer_version": c.producer_version,
                    "schema_version": c.schema_version,
                    "error": c.error,
                    "notes": c.notes,
                }
                for c in sorted(self.components.values(), key=lambda x: (x.kind, x.name))
            ],
        }
        _atomic_write_json(self.path, payload)


