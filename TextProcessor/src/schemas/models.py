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
    # Optional: tokenized transcripts (shared tokenizer under dp_models).
    # Example:
    #   {"whisper": [101, 2023, ...], "youtube_auto": [...]}.
    transcripts_token_ids: Dict[str, List[int]] = field(default_factory=dict)
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

    doc = VideoDocument(
        title=str(data.get("title", "")),
        description=str(data.get("description", "")),
        transcripts=dict(data.get("transcripts") or {}),
        transcripts_token_ids=dict(data.get("transcripts_token_ids") or {}),
        video_description_by_neuro=data.get("video_description_by_neuro"),
        trend_words=data.get("trend_words"),
        comments=comments,
        speakers=data.get("speakers"),
    )

    # If raw transcripts are missing but token IDs are provided, decode to text using shared tokenizer.
    # This keeps artifacts free of raw transcript text while allowing TextProcessor to operate.
    try:
        if (not doc.transcripts) and isinstance(doc.transcripts_token_ids, dict) and doc.transcripts_token_ids:
            token_ids = doc.transcripts_token_ids.get("whisper") or None
            if isinstance(token_ids, list) and token_ids:
                from dp_models import get_global_model_manager  # type: ignore

                mm = get_global_model_manager()
                tok_spec = mm.get_spec(model_name="shared_tokenizer_v1")
                _, _, _, _, _wd, artifacts = mm.resolve(tok_spec)
                tok_path = list(artifacts.values())[0] if artifacts else None
                if not tok_path:
                    raise RuntimeError("shared_tokenizer_v1 artifacts are empty")
                from tokenizers import Tokenizer  # type: ignore

                tok = Tokenizer.from_file(tok_path)
                text = tok.decode([int(x) for x in token_ids], skip_special_tokens=True)
                if isinstance(text, str) and text.strip():
                    doc.transcripts["whisper"] = text.strip()
    except Exception:
        # Fail-fast is enforced at pipeline level when transcript is required; schema remains flexible.
        pass

    return doc


