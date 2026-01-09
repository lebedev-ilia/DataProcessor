from __future__ import annotations

from dataclasses import dataclass


def _join(*parts: str) -> str:
    cleaned = []
    for p in parts:
        if p is None:
            continue
        s = str(p).strip("/")
        if not s:
            continue
        cleaned.append(s)
    return "/".join(cleaned)


@dataclass(frozen=True)
class KeyLayout:
    """
    Canonical key layout inside storage backend.

    We keep everything under a single configurable prefix (S3_PREFIX), and then:
    - result_store/...
    - state/...
    - frames_dir/...
    """

    prefix: str

    def result_store_prefix(self) -> str:
        return _join(self.prefix, "result_store")

    def state_prefix(self) -> str:
        return _join(self.prefix, "state")

    def frames_dir_prefix(self) -> str:
        return _join(self.prefix, "frames_dir")

    def result_store_run_prefix(self, platform_id: str, video_id: str, run_id: str) -> str:
        return _join(self.result_store_prefix(), platform_id, video_id, run_id)

    def state_run_prefix(self, platform_id: str, video_id: str, run_id: str) -> str:
        return _join(self.state_prefix(), platform_id, video_id, run_id)

    def frames_dir_run_prefix(self, platform_id: str, video_id: str, run_id: str) -> str:
        return _join(self.frames_dir_prefix(), platform_id, video_id, run_id)


