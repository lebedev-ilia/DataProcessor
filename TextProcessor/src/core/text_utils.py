from __future__ import annotations

import re
from typing import Dict, Optional


_WHITESPACE_RE = re.compile(r"\s+", re.UNICODE)


def normalize_whitespace(text: Optional[str]) -> str:
    """
    Нормализует пробелы в строке: тримит и схлопывает повторы.
    Возвращает пустую строку, если вход None.
    """
    if not text:
        return ""
    return _WHITESPACE_RE.sub(" ", text).strip()


def choose_best_transcript(transcripts: Optional[Dict[str, str]]) -> str:
    """
    Простейшая эвристика выбора транскрипта: предпочесть 'whisper', затем 'youtube_auto', иначе любой первый.
    Возвращает нормализованный текст или пустую строку.
    """
    if not transcripts:
        return ""
    if "whisper" in transcripts and transcripts["whisper"]:
        return normalize_whitespace(transcripts["whisper"])  # type: ignore[index]
    if "youtube_auto" in transcripts and transcripts["youtube_auto"]:
        return normalize_whitespace(transcripts["youtube_auto"])  # type: ignore[index]
    # fallback: first non-empty
    for v in transcripts.values():
        if v:
            return normalize_whitespace(v)
    return ""


