from __future__ import annotations

import re
from typing import Any, Dict, List, Tuple, Optional
import unicodedata

import numpy as np
try:
    import emoji as _emoji  # type: ignore
except Exception as e:
    print(f"Error importing emoji: {e}")
    _emoji = None  # type: ignore

from src.core.base_extractor import BaseExtractor
from src.core.metrics import system_snapshot, process_memory_bytes
from src.schemas.models import VideoDocument
from src.core.text_utils import normalize_whitespace


_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_MENTION_RE = re.compile(r"(?<!\w)@([A-Za-z0-9_\.]+)")
_TIMESTAMP_RE = re.compile(r"\b(?:\d{1,2}:){1,2}\d{2}\b")  # 01:23 or 1:02:03
_QUESTION_PREFIX_RE = re.compile(r"^(кто|что|где|когда|почему|зачем|как|who|what|where|when|why|how)\b", re.IGNORECASE)
_NUMBER_RE = re.compile(r"\d+")
_DATE_TIME_RE = re.compile(r"\b(\d{1,2}[./-]\d{1,2}([./-]\d{2,4})?|\d{1,2}:\d{2})\b")
# Python's re doesn't support \p{..}. Use Unicode categories instead.
def _is_punct_symbol(ch: str) -> bool:
    try:
        cat = unicodedata.category(ch)
        # P* → punctuation, S* → symbols
        return len(cat) > 0 and cat[0] in ("P", "S")
    except Exception:
        return False

# Optional dependencies
try:
    from langdetect import detect_langs as _detect_langs  # type: ignore
except Exception as e:
    print(f"Error importing detect_langs: {e}")
    _detect_langs = None  # type: ignore
try:
    import spacy as _spacy  # type: ignore
except Exception as e:
    print(f"Error importing spacy: {e}")
    _spacy = None  # type: ignore


def _tokenize(text: str) -> List[str]:
    text = normalize_whitespace(text or "")
    if not text:
        return []
    return re.findall(r"\w+", text, flags=re.UNICODE)


def _sentences(text: str) -> List[str]:
    text = normalize_whitespace(text or "")
    if not text:
        return []
    parts = re.split(r"[.!?]+\s+", text)
    return [p for p in parts if p]


class LexicalStatsExtractor(BaseExtractor):
    VERSION = "1.0.0"

    def extract(self, doc: VideoDocument) -> Dict[str, Any]:
        import time

        t0 = time.perf_counter()
        sys_before = system_snapshot()
        mem_before = process_memory_bytes()

        title = str(getattr(doc, "title", "") or "")
        description = str(getattr(doc, "description", "") or "")

        transcripts = getattr(doc, "transcripts", {}) or {}
        transcript = " ".join([str(transcripts.get(k, "")) for k in ("whisper", "youtube_auto") if transcripts.get(k)])

        title_tokens = _tokenize(title)
        desc_tokens = _tokenize(description)
        trans_tokens = _tokenize(transcript)

        # Title features
        title_len_words = len(title_tokens)
        title_len_chars = len(title)
        title_avg_word_len = float(np.mean([len(t) for t in title_tokens])) if title_tokens else 0.0
        title_exclamation_count = title.count("!")
        title_question_count = title.count("?")
        def _is_emoji(ch: str) -> bool:
            try:
                return (_emoji is not None) and (ch in _emoji.EMOJI_DATA)
            except Exception as e:
                print(f"Error checking emoji: {e}")
                return False
        emoji_count_title = sum(1 for c in title if _is_emoji(c))
        title_type_token_ratio = float(len(set(map(str.lower, title_tokens))) / max(1, len(title_tokens)))
        title_punct_ratio = float(sum(1 for c in title if _is_punct_symbol(c)) / max(1, len(title)))
        title_capital_words_ratio = float(sum(1 for t in title_tokens if t.isupper()) / max(1, len(title_tokens)))
        title_question_prefix_flag = bool(_QUESTION_PREFIX_RE.search(title.strip()))
        title_number_presence = bool(_NUMBER_RE.search(title))
        title_time_mention_flag = bool(_DATE_TIME_RE.search(title))

        # Description features
        description_len_words = len(desc_tokens)
        description_num_urls = len(_URL_RE.findall(description))
        description_num_mentions = len(_MENTION_RE.findall(description))
        description_has_timestamps_flag = bool(_TIMESTAMP_RE.search(description))
        emoji_count_description = sum(1 for c in description if _is_emoji(c))
        # emoji diversity (по всем полям)
        all_text = (title or "") + "\n" + (description or "") + "\n" + (transcript or "")
        all_emojis = [c for c in all_text if _is_emoji(c)]
        emoji_diversity = float(len(set(all_emojis)) / max(1, len(all_emojis))) if all_emojis else 0.0

        # Transcript features
        transcript_len_words = len(trans_tokens)
        sents = _sentences(transcript)
        transcript_avg_sentence_len = float(np.mean([len(_tokenize(s)) for s in sents])) if sents else 0.0
        # доля вопросительных предложений
        if sents:
            _q = sum(1 for s in sents if "?" in s)
            question_ratio_transcript = float(_q / max(1, len(sents)))
        else:
            question_ratio_transcript = 0.0
        lexical_diversity_transcript = float(len(set(map(str.lower, trans_tokens))) / max(1, len(trans_tokens)))
        # rare_word_ratio_transcript proxy: words longer than 12 chars
        rare_word_ratio_transcript = float(sum(1 for t in trans_tokens if len(t) > 12) / max(1, len(trans_tokens)))
        # stopword_ratio_transcript proxy: simple list for ru/en
        stopwords = set([
            "и","в","во","не","что","он","на","я","с","со","как","а","то","все","она","так","его","но","да","ты","к","у",
            "the","a","an","and","or","but","if","in","on","with","for","to","of","is","are","was","were","be","been","it",
        ])
        stopword_ratio_transcript = float(sum(1 for t in map(str.lower, trans_tokens) if t in stopwords) / max(1, len(trans_tokens)))
        # 33) title_stopword_ratio
        title_stopword_ratio = float(sum(1 for t in map(str.lower, title_tokens) if t in stopwords) / max(1, len(title_tokens)))

        # 48) Readability proxy for transcript (simple): avg_sentence_len / avg_word_len
        avg_word_len_transcript = float(np.mean([len(t) for t in trans_tokens])) if trans_tokens else 0.0
        readability_score_transcript = float(transcript_avg_sentence_len / max(1e-6, avg_word_len_transcript)) if transcript_avg_sentence_len > 0 else 0.0

        # 49) title_clickbait_score (rule-based): keywords + punct signals
        clickbait_words = {
            # RU
            "шок","срочно","невероятно","топ","лучшие","секрет","удивит","скандал","честно","разоблачение",
            # EN
            "shocking","urgent","incredible","top","best","secret","you won't believe","scandal","honest","exposed",
        }
        tl = title.lower()
        cb_hits = 0
        for w in clickbait_words:
            if w in tl:
                cb_hits += 1
        cb_signal = cb_hits + (1 if title_exclamation_count > 0 else 0) + (1 if title_question_prefix_flag else 0)
        title_clickbait_score = float(min(1.0, cb_signal / 3.0))

        # 57–58) Language detection best-effort (title+description)
        text_language = "unknown"
        language_confidence = 0.0
        if _detect_langs is not None:
            try:
                lang_candidates = _detect_langs((title + " \n " + description).strip()[:5000])
                if lang_candidates:
                    text_language = getattr(lang_candidates[0], "lang", "unknown")
                    language_confidence = float(getattr(lang_candidates[0], "prob", 0.0))
            except Exception as e:
                print(f"Error detecting language: {e}")
                pass

        # 59) orthographic_error_rate proxy: share of tokens not matching simple alpha pattern or too few vowels
        def _is_wellformed(tok: str) -> bool:
            t = tok.lower()
            if not t:
                return False
            # Allow only letters and marks (Unicode categories starting with 'L' or 'M')
            for ch in t:
                cat = unicodedata.category(ch)
                if not (cat and cat[0] in ("L", "M")):
                    return False
            vowels = "аеёиоуыэюяaeiouy"
            return any(ch in vowels for ch in t)
        ortho_bad = sum(1 for t in trans_tokens if not _is_wellformed(t))
        orthographic_error_rate = float(ortho_bad / max(1, len(trans_tokens)))

        # 60) avg_token_frequency_percentile proxy: inverse normalized word length (shorter words → higher percentile)
        if trans_tokens:
            lens = np.array([len(t) for t in trans_tokens], dtype=np.float32)
            freq_proxy = 1.0 - np.clip(lens / 20.0, 0.0, 1.0)
            avg_token_frequency_percentile = float(freq_proxy.mean())
        else:
            avg_token_frequency_percentile = 0.0

        # 46) pos_distribution_transcript (optional via spaCy, if available)
        pos_distribution_transcript: Dict[str, float] = {}
        named_entity_density = 0.0
        if _spacy is not None and transcript_len_words > 0:
            nlp = None
            # try pick a lightweight model; if none available, skip
            for model in ("ru_core_news_sm", "en_core_web_sm"):
                try:
                    nlp = _spacy.load(model)
                    break
                except Exception as e:
                    nlp = None
            if nlp is not None:
                try:
                    doc_sp = nlp(transcript[:5000])
                    counts: Dict[str, int] = {}
                    for t in doc_sp:
                        pos = t.pos_ or "X"
                        counts[pos] = counts.get(pos, 0) + 1
                    total = sum(counts.values())
                    if total > 0:
                        pos_distribution_transcript = {k: v / total for k, v in counts.items()}
                    # named entity density by token share
                    ent_tokens = sum(len(ent) for ent in doc_sp.ents)
                    if len(doc_sp) > 0:
                        named_entity_density = float(ent_tokens / len(doc_sp))
                except Exception as e:
                    print(f"Error processing transcript: {e}")
                    pos_distribution_transcript = {}

        # punctuation entropy (title + description combined)
        def _entropy_from_counts(counts: Dict[str, int]) -> float:
            total = sum(counts.values())
            if total <= 0:
                return 0.0
            probs = np.array([c / total for c in counts.values()], dtype=np.float32)
            return float(-np.sum(probs * np.log(probs + 1e-9)))

        puncts_text = title + " " + description
        punct_counts: Dict[str, int] = {}
        for ch in puncts_text:
            if _is_punct_symbol(ch):
                punct_counts[ch] = punct_counts.get(ch, 0) + 1
        punctuation_entropy = _entropy_from_counts(punct_counts)

        special_character_ratio = float(sum(1 for ch in (title + description) if not ch.isalnum() and not ch.isspace()) / max(1, len(title + description)))
        upper_lower_ratio_title = float(sum(1 for c in title if c.isupper()) / max(1, sum(1 for c in title if c.islower()) or 1))

        results: Dict[str, Any] = {
            "title_len_words": int(title_len_words),
            "title_len_chars": int(title_len_chars),
            "title_avg_word_len": float(title_avg_word_len),
            "title_exclamation_count": int(title_exclamation_count),
            "title_question_count": int(title_question_count),
            "emoji_count_title": int(emoji_count_title),
            "title_stopword_ratio": float(title_stopword_ratio),
            "title_type_token_ratio": float(title_type_token_ratio),
            "title_punctuation_ratio": float(title_punct_ratio),
            "title_capital_words_ratio": float(title_capital_words_ratio),
            "title_question_prefix_flag": bool(title_question_prefix_flag),
            "title_number_presence": bool(title_number_presence),
            "title_time_mention_flag": bool(title_time_mention_flag),
            "title_clickbait_score": float(title_clickbait_score),
            "description_len_words": int(description_len_words),
            "description_num_urls": int(description_num_urls),
            "description_num_mentions": int(description_num_mentions),
            "description_has_timestamps_flag": bool(description_has_timestamps_flag),
            "emoji_count_description": int(emoji_count_description),
            "transcript_len_words": int(transcript_len_words),
            "transcript_avg_sentence_len": float(transcript_avg_sentence_len),
            "lexical_diversity_transcript": float(lexical_diversity_transcript),
            "rare_word_ratio_transcript": float(rare_word_ratio_transcript),
            "stopword_ratio_transcript": float(stopword_ratio_transcript),
            "question_ratio_transcript": float(question_ratio_transcript),
            "readability_score_transcript": float(readability_score_transcript),
            "pos_distribution_transcript": pos_distribution_transcript,
            "text_language": text_language,
            "language_confidence": float(language_confidence),
            "punctuation_entropy": float(punctuation_entropy),
            "special_character_ratio": float(special_character_ratio),
            "upper_lower_ratio_title": float(upper_lower_ratio_title),
            "emoji_diversity": float(emoji_diversity),
            "named_entity_density": float(named_entity_density),
            "orthographic_error_rate": float(orthographic_error_rate),
            "avg_token_frequency_percentile": float(avg_token_frequency_percentile),
        }

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
            "result": {"lexical_stats": results},
            "error": None,
        }


