from __future__ import annotations

from typing import Any, Dict, List, Optional
import unicodedata
import numpy as np

from src.core.base_extractor import BaseExtractor
from src.core.metrics import system_snapshot, process_memory_bytes
from src.schemas.models import VideoDocument
from src.core.text_utils import normalize_whitespace


def _tokenize(text: str) -> List[str]:
    t = normalize_whitespace(text or "")
    if not t:
        return []
    return [tok for tok in t.split() if tok]


class ASRTextProxyExtractor(BaseExtractor):
    VERSION = "1.0.0"

    def __init__(self) -> None:
        pass

    def _detect_lang(self, text: str) -> tuple[str, float]:
        try:
            from langdetect import detect_langs  # type: ignore
        except Exception as e:
            print(f"Error importing langdetect: {e}")
            return "unknown", 0.0
        try:
            cands = detect_langs(text[:5000])
            if cands:
                return getattr(cands[0], "lang", "unknown"), float(getattr(cands[0], "prob", 0.0))
        except Exception as e:
            print(f"Error detecting language: {e}")
        return "unknown", 0.0

    def extract(self, doc: VideoDocument) -> Dict[str, Any]:
        import time

        t0 = time.perf_counter()
        sys_before = system_snapshot()
        mem_before = process_memory_bytes()

        # Expect transcripts with confidence under doc.transcripts_meta if available; otherwise fallback
        transcript_meta = getattr(doc, "transcripts_meta", None)
        items: List[Dict[str, Any]] = []
        total_audio_duration: Optional[float] = None
        language_declared: Optional[str] = None

        # Flexible intake from document extras if present
        try:
            if isinstance(transcript_meta, dict):
                items = list(transcript_meta.get("with_confidence", []) or [])
                total_audio_duration = transcript_meta.get("total_audio_duration")
                language_declared = transcript_meta.get("language_declared")
        except Exception as e:
            print(f"Error parsing transcript meta: {e}")
            items = []

        # Also try to parse from transcripts text if items missing (best-effort: no confidences)
        if not items:
            # fabricate single item from combined transcript
            tr = getattr(doc, "transcripts", {}) or {}
            text = " ".join([str(tr.get(k, "")) for k in ("whisper", "youtube_auto") if tr.get(k)])
            if text.strip():
                items = [{"text": text, "confidence": 0.8, "start": 0.0, "end": 0.0}]

        texts = [normalize_whitespace(str(s.get("text", ""))) for s in items]
        full_text = " ".join(texts).strip()
        confidences = [float(s.get("confidence", 0.0)) for s in items if s.get("confidence") is not None]
        starts = [float(s.get("start", 0.0)) for s in items]
        ends = [float(s.get("end", 0.0)) for s in items]

        # Durations
        seg_durations = [max(0.0, e - s) for s, e in zip(starts, ends)] if starts and ends else []
        # Word-based estimate (160 wpm heuristic)
        words_count = len(_tokenize(full_text))
        est_duration_min = max(0.5, (words_count or 1) / 160.0)
        est_duration_sec = est_duration_min * 60.0

        if total_audio_duration is not None and float(total_audio_duration) > 1.0:
            duration_sec = float(total_audio_duration)
        elif seg_durations and sum(seg_durations) > 1.0:
            duration_sec = float(sum(seg_durations))
        else:
            # Fallback to robust word-based estimate
            duration_sec = est_duration_sec
        duration_min = max(1e-6, duration_sec / 60.0)

        # 1) ASR confidence
        if confidences:
            asr_conf_mean = float(np.mean(confidences))
            asr_conf_std = float(np.std(confidences))
            # chunked means in blocks of ~10 items
            block = max(1, int(round(len(confidences) / max(1, len(confidences) // 10 + 1))))
            chunk_means = [float(np.mean(confidences[i:i+block])) for i in range(0, len(confidences), block)]
            asr_conf_chunked_min = float(np.min(chunk_means)) if chunk_means else asr_conf_mean
        else:
            asr_conf_mean = 0.0
            asr_conf_std = 0.0
            asr_conf_chunked_min = 0.0

        # Tokens
        tokens = _tokenize(full_text)
        total_words = len(tokens)
        total_chars = len(full_text)

        # 2) Errors and rarity proxies
        # rare word: length > 12 or contains digits/symbols heavily
        def _is_rare(tok: str) -> bool:
            if len(tok) > 12:
                return True
            has_digit = any(ch.isdigit() for ch in tok)
            sym_ratio = sum(1 for ch in tok if not (unicodedata.category(ch)[0] in ("L", "M") or ch.isdigit())) / max(1, len(tok))
            return has_digit or sym_ratio > 0.4

        rare_word_ratio = float(sum(1 for t in tokens if _is_rare(t)) / max(1, total_words))
        low_conf_rate = float(sum(1 for c in confidences if c < 0.5) / max(1, len(confidences))) if confidences else 0.0
        asr_error_proxy = float(min(1.0, 0.5 * rare_word_ratio + 0.5 * low_conf_rate))

        # oov proxy: tokens with many non-letter marks
        def _is_oov(tok: str) -> bool:
            letters = sum(1 for ch in tok if unicodedata.category(ch)[0] in ("L", "M"))
            return letters < max(1, len(tok) // 2)

        oov_rate_asr_tokens = float(sum(1 for t in tokens if _is_oov(t)) / max(1, total_words))

        # 3) Speech rhythm
        speech_rate_wpm = float(total_words / duration_min)
        speech_character_density = float(total_chars / max(1e-6, duration_sec))
        # pauses approximation: comma/semicolon/colon per sentence; sentences by .?!
        n_sent = max(1, full_text.count(".") + full_text.count("?") + full_text.count("!"))
        pauses = full_text.count(",") + full_text.count(";") + full_text.count(":")
        pause_density_proxy = float(pauses / n_sent)
        filler_lexicon = {"ээ", "мм", "ну", "типа", "короче", "значит", "э", "эээ", "mmm", "uh", "um"}
        filler_word_ratio = float(sum(1 for w in tokens if w.lower() in filler_lexicon) / max(1, total_words))

        # 4) Structure and emotions
        sentence_intonation_proxy = float((full_text.count("!") + full_text.count("?")) / n_sent)

        # named entities coverage (optional spaCy): percent of entities considered covered if confidences present
        named_entities_covered_by_asr = None
        try:
            import spacy  # type: ignore
            nlp = None
            for model in ("ru_core_news_sm", "en_core_web_sm"):
                try:
                    nlp = spacy.load(model)
                    break
                except Exception as e:
                    print(f"Error loading spaCy model {model}: {e}")
                    nlp = None
            if nlp is not None and full_text:
                doc_sp = nlp(full_text[:5000])
                total_entities = len(doc_sp.ents)
                if total_entities > 0:
                    # If we have confidences at all, assume entities are covered by ASR text (no alignment available)
                    named_entities_covered_by_asr = 1.0 if confidences else 0.0
                else:
                    named_entities_covered_by_asr = 0.0
        except Exception as e:
            print(f"Error processing named entities: {e}")
            named_entities_covered_by_asr = None

        # language mismatch
        detected_lang, _det_p = self._detect_lang(full_text) if full_text else ("unknown", 0.0)
        asr_language_mismatch_flag = None
        if language_declared:
            try:
                asr_language_mismatch_flag = bool(detected_lang and language_declared and detected_lang != language_declared)
            except Exception as e:
                print(f"Error checking language mismatch: {e}")
                asr_language_mismatch_flag = None

        # acoustic noise proxy: proportion of low confidence runs
        if confidences:
            try:
                acoustic_noise_proxy = float(sum(1 for c in confidences if c < 0.5) / len(confidences))
            except Exception as e:
                print(f"Error calculating acoustic noise proxy: {e}")
                acoustic_noise_proxy = 0.0
        else:
            acoustic_noise_proxy = 0.0

        # speaker count estimate textual: heuristics by dashes and colon cues (avoid generic words)
        dash_cues = full_text.count("— ") + full_text.count("- ")
        colon_cues = full_text.count(": ")
        speaker_cues = dash_cues + colon_cues
        speaker_count_estimate_textual = int(min(10, max(1, round(speaker_cues ** 0.5)))) if full_text else 0

        features: Dict[str, Any] = {
            "asr_confidence_mean": asr_conf_mean,
            "asr_confidence_std": asr_conf_std,
            "asr_confidence_chunked_min": asr_conf_chunked_min,
            "rare_word_ratio": float(rare_word_ratio),
            "low_conf_rate": float(low_conf_rate),
            "asr_error_proxy": float(asr_error_proxy),
            "oov_rate_asr_tokens": float(oov_rate_asr_tokens),
            "speech_rate_wpm": float(speech_rate_wpm),
            "speech_character_density": float(speech_character_density),
            "pause_density_proxy": float(pause_density_proxy),
            "filler_word_ratio": float(filler_word_ratio),
            "sentence_intonation_proxy": float(sentence_intonation_proxy),
            "named_entities_covered_by_asr": named_entities_covered_by_asr,
            "asr_language_mismatch_flag": asr_language_mismatch_flag,
            "acoustic_noise_proxy": float(acoustic_noise_proxy),
            "speaker_count_estimate_textual": int(speaker_count_estimate_textual),
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
            "result": {"asr_text_proxy": features},
            "error": None,
        }


