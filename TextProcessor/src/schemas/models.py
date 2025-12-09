from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class Comment:
    text: str


@dataclass
class VideoDocument:
    title: str
    description: str
    transcripts: Dict[str, str] = field(default_factory=dict)
    video_description_by_neuro: Optional[str] = None
    trend_words: Optional[str] = None
    comments: List[Comment] = field(default_factory=list)
    speakers: Optional[Dict[str, Dict[str, Any]]] = None


def video_document_from_dict(data: Dict) -> VideoDocument:
    comments_raw = data.get("comments") or []
    comments: List[Comment] = []
    for c in comments_raw:
        if isinstance(c, dict) and "text" in c:
            comments.append(Comment(text=str(c.get("text", ""))))
        else:
            comments.append(Comment(text=str(c)))

    return VideoDocument(
        title=str(data.get("title", "")),
        description=str(data.get("description", "")),
        transcripts=dict(data.get("transcripts") or {}),
        video_description_by_neuro=data.get("video_description_by_neuro"),
        trend_words=data.get("trend_words"),
        comments=comments,
        speakers=data.get("speakers"),
    )


