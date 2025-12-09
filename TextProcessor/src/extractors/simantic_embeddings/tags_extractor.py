from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple

from src.core.base_extractor import BaseExtractor
from src.core.metrics import system_snapshot, process_memory_bytes
from src.core.text_utils import normalize_whitespace
from src.schemas.models import VideoDocument


_HASHTAG_RE = re.compile(r"(?P<hash>#)(?P<tag>[\w\-а-яА-ЯёЁ]+)")


def _extract_hashtags(text: str) -> Tuple[str, List[str]]:
    tags: List[str] = []
    def repl(m: re.Match) -> str:
        tag = m.group("tag").lower()
        tags.append(tag)
        return ""  # remove hashtag token
    cleaned = _HASHTAG_RE.sub(repl, text)
    # collapse excess whitespace
    cleaned = normalize_whitespace(cleaned)
    # keep unique tags preserving order
    seen = set()
    uniq = []
    for t in tags:
        if t not in seen:
            seen.add(t)
            uniq.append(t)
    return cleaned, uniq


class TagsExtractor(BaseExtractor):
    VERSION = "1.0.0"

    def extract(self, doc: VideoDocument) -> Dict[str, Any]:
        import time

        t0 = time.perf_counter()
        sys_before = system_snapshot()
        mem_before = process_memory_bytes()

        title_raw = normalize_whitespace(doc.title or "")
        desc_raw = normalize_whitespace(doc.description or "")

        title_clean, title_tags = _extract_hashtags(title_raw)
        desc_clean, desc_tags = _extract_hashtags(desc_raw)

        # merge and uniquify tags
        merged_tags: List[str] = []
        seen = set()
        for t in title_tags + desc_tags:
            if t not in seen:
                seen.add(t)
                merged_tags.append(t)

        sys_after = system_snapshot()
        mem_after = process_memory_bytes()
        total_s = time.perf_counter() - t0

        return {
            "device": "cpu",
            "version": self.VERSION,
            "system": {
                "pre_init": sys_before,
                "post_init": sys_before,
                "post_process": sys_after,
                "peaks": {
                    "ram_peak_mb": int(max(mem_before, mem_after) / 1024 / 1024),
                    "gpu_peak_mb": 0,
                },
            },
            "timings_s": {"total": round(total_s, 3)},
            "result": {
                "cleaned_texts": {
                    "title": title_clean,
                    "description": desc_clean,
                },
                "hashtags": merged_tags,
            },
            "error": None,
        }


