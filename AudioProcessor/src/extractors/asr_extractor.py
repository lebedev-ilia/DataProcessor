"""
ASR extractor (Whisper) — Triton-backed, no-network, token-IDs output (no raw text).
"""
import time
import logging
import os
import numpy as np
from typing import Dict, Any, Optional, List

from src.core.base_extractor import BaseExtractor, ExtractorResult
from src.core.audio_utils import AudioUtils
from dp_triton import TritonHttpClient, TritonError  # type: ignore

logger = logging.getLogger(__name__)


class ASRExtractor(BaseExtractor):
    """
    Whisper ASR via Triton. Output is token IDs from a shared tokenizer (dp_models),
    so TextProcessor can decode without storing raw transcript text in artifacts.
    """
    
    name = "asr_extractor"
    version = "2.0.0"
    description = "Whisper ASR via Triton (token IDs, no raw text)"
    category = "speech"
    dependencies = ["numpy", "dp_triton", "dp_models"]
    estimated_duration = 8.0
    
    gpu_required = False
    gpu_preferred = True
    gpu_memory_required = 0.0  # client-side only; model runs on Triton
    
    def __init__(
        self, 
        device: str = "auto",
        model_size: str = "small",
        sample_rate: int = 16000
    ):
        """
        Инициализация ASR экстрактора.
        
        Args:
            device: Устройство для обработки
            model_size: Whisper size: small|medium|large (Triton model selection via ModelManager)
            sample_rate: Частота дискретизации
        """
        super().__init__(device=device)
        
        self.model_size = str(model_size or "small").strip().lower()
        if self.model_size not in ("small", "medium", "large"):
            raise ValueError(f"ASR | unsupported model_size={self.model_size}. Expected: small|medium|large")
        self.sample_rate = int(sample_rate)
        
        self.audio_utils = AudioUtils(device=device, sample_rate=sample_rate)

        # Resolve models via ModelManager (no-network).
        try:
            from dp_models import get_global_model_manager  # type: ignore
            from dp_models.errors import ModelManagerError  # type: ignore

            self._mm = get_global_model_manager()
        except Exception as e:
            raise RuntimeError(f"ASR | ModelManager is required but failed to init: {e}") from e

        # Shared tokenizer must exist locally (B: shared tokenizer contract).
        try:
            tok_spec = self._mm.get_spec(model_name="shared_tokenizer_v1")
            _d, _p, _rt, _eng, tok_digest, tok_artifacts = self._mm.resolve(tok_spec)
            self.tokenizer_model_name = str(tok_spec.model_name)
            self.tokenizer_weights_digest = str(tok_digest)
            self.tokenizer_artifact_path = list(tok_artifacts.values())[0] if tok_artifacts else None
            if not self.tokenizer_artifact_path:
                raise RuntimeError("ASR | shared_tokenizer_v1 has empty artifacts")
        except Exception as e:
            raise RuntimeError(f"ASR | shared tokenizer is missing/invalid: {e}") from e

        # Whisper Triton spec selection by size.
        whisper_spec_name = f"whisper_{self.model_size}_triton"
        try:
            self.whisper_spec = self._mm.get_spec(model_name=whisper_spec_name)
            dev, prec, rt, eng, wd, _art = self._mm.resolve(self.whisper_spec)
            if str(rt) != "triton":
                raise RuntimeError(f"ASR | expected runtime=triton in spec {whisper_spec_name}, got {rt}")
            self.whisper_model_name = str(self.whisper_spec.model_name)
            self.whisper_weights_digest = str(wd)
            rp = self.whisper_spec.runtime_params or {}
            self.triton_http_url = self._expand_env(str(rp.get("triton_http_url") or os.environ.get("TRITON_HTTP_URL") or ""))
            self.triton_model_name = str(rp.get("triton_model_name") or "")
            self.triton_model_version = rp.get("triton_model_version")
            self.triton_input_name = str(rp.get("triton_input_name") or "AUDIO__0")
            self.triton_input_datatype = str(rp.get("triton_input_datatype") or "FP32")
            self.triton_output_token_ids_name = str(rp.get("triton_output_token_ids_name") or "TOKEN_IDS__0")
            self.triton_output_token_ids_datatype = str(rp.get("triton_output_token_ids_datatype") or "INT32")
            self.triton_output_lang_id_name = str(rp.get("triton_output_lang_id_name") or "LANG_ID__0")
            self.triton_output_lang_id_datatype = str(rp.get("triton_output_lang_id_datatype") or "INT32")
            if not self.triton_http_url or not self.triton_model_name:
                raise RuntimeError("ASR | Triton runtime_params missing triton_http_url/triton_model_name")
        except Exception as e:
            raise RuntimeError(f"ASR | failed to resolve whisper triton spec via ModelManager: {e}") from e

        self._client = TritonHttpClient(base_url=self.triton_http_url, timeout_sec=10.0)

    def _expand_env(self, s: str) -> str:
        """
        Expand simple ${VAR} placeholders (used in model specs).
        """
        out = str(s)
        if "${" not in out:
            return out
        import re

        def repl(m):
            k = m.group(1)
            return os.environ.get(k, "")

        return re.sub(r"\$\{([^}]+)\}", repl, out)

    def _infer_segment_token_ids(self, audio_1d: np.ndarray) -> tuple[np.ndarray, int]:
        if not self._client.ready():
            raise TritonError(f"Triton is not ready at {self.triton_http_url}", error_code="triton_unavailable")
        x = np.asarray(audio_1d, dtype=np.float32).reshape(1, -1)
        outs = self._client.infer_multi(
            model_name=self.triton_model_name,
            model_version=(str(self.triton_model_version) if self.triton_model_version else None),
            input_name=self.triton_input_name,
            input_tensor=x,
            input_datatype=self.triton_input_datatype,
            outputs=[
                (self.triton_output_token_ids_name, self.triton_output_token_ids_datatype),
                (self.triton_output_lang_id_name, self.triton_output_lang_id_datatype),
            ],
        )
        tok = outs[self.triton_output_token_ids_name].output
        lang = outs[self.triton_output_lang_id_name].output
        tok = np.asarray(tok, dtype=np.int32)
        # allow [1, L] or [L]
        if tok.ndim == 2 and tok.shape[0] == 1:
            tok = tok[0]
        tok = tok.reshape(-1)
        lang_id = int(np.asarray(lang).reshape(-1)[0]) if np.asarray(lang).size else -1
        return tok, lang_id

    def run_segments(self, input_uri: str, tmp_path: str, segments: List[Dict[str, Any]]) -> ExtractorResult:
        """
        Run ASR on Segmenter-provided long windows (families.asr) and return token ids per segment.
        No raw transcript is stored.
        """
        start_time = time.time()
        try:
            if not self._validate_input(input_uri):
                return self._create_result(
                    success=False,
                    error="Некорректный входной файл",
                    processing_time=time.time() - start_time,
                )
            if not isinstance(segments, list) or not segments:
                raise ValueError("segments is empty (no-fallback)")

            token_ids_by_segment: list[np.ndarray] = []
            lang_id_by_segment: list[int] = []
            seg_st: list[float] = []
            seg_en: list[float] = []
            seg_center: list[float] = []

            for seg in segments:
                ss = int(seg.get("start_sample"))
                es = int(seg.get("end_sample"))
                st = float(seg.get("start_sec"))
                en = float(seg.get("end_sec"))
                c = float(seg.get("center_sec"))

                wav_t, sr = self.audio_utils.load_audio_segment(input_uri, start_sample=ss, end_sample=es, target_sr=self.sample_rate)
                wav_np = self.audio_utils.to_numpy(wav_t)
                if wav_np.ndim == 2:
                    wav_np = wav_np[0]
                wav_np = np.asarray(wav_np, dtype=np.float32).reshape(-1)
                if int(sr) != int(self.sample_rate):
                    # load_audio_segment should resample; keep strictness anyway
                    raise RuntimeError(f"ASR | segment SR mismatch: got {sr} expected {self.sample_rate}")

                tok, lang_id = self._infer_segment_token_ids(wav_np)
                token_ids_by_segment.append(tok.astype(np.int32))
                lang_id_by_segment.append(int(lang_id))
                seg_st.append(float(st))
                seg_en.append(float(en))
                seg_center.append(float(c))

            payload: Dict[str, Any] = {
                "token_ids_by_segment": [t.tolist() for t in token_ids_by_segment],
                "lang_id_by_segment": lang_id_by_segment,
                "segment_start_sec": seg_st,
                "segment_end_sec": seg_en,
                "segment_center_sec": seg_center,
                "segments_count": int(len(token_ids_by_segment)),
                "sample_rate": int(self.sample_rate),
                "whisper_model_name": self.whisper_model_name,
                "tokenizer_model_name": self.tokenizer_model_name,
                "device_used": "cuda",  # Triton-backed assumption in this project
            }
            return self._create_result(True, payload=payload, processing_time=time.time() - start_time)
        except TritonError as e:
            return self._create_result(False, error=str(e), processing_time=time.time() - start_time)
        except Exception as e:
            return self._create_result(False, error=str(e), processing_time=time.time() - start_time)
    
    def run(self, input_uri: str, tmp_path: str) -> ExtractorResult:
        """
        Извлечение транскрипции речи.
        
        Args:
            input_uri: Путь к аудио файлу
            tmp_path: Временная директория
            
        Returns:
            ExtractorResult with token IDs (requires segments mode in production).
        """
        return self._create_result(
            success=False,
            error="ASRExtractor | run() is not supported in production. Use run_segments() with Segmenter-provided families.asr windows.",
            processing_time=0.0,
        )
    
    def _validate_input(self, input_uri: str) -> bool:
        """Валидация входного файла."""
        if not super()._validate_input(input_uri):
            return False
        
        # Проверяем, что это аудио файл
        audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.mp4', '.avi', '.mov'}
        if not any(input_uri.lower().endswith(ext) for ext in audio_extensions):
            self.logger.error(f"Файл не является поддерживаемым аудио/видео форматом: {input_uri}")
            return False
        
        return True
    
    def get_model_info(self) -> Dict[str, Any]:
        """Получение информации о модели Whisper."""
        return {
            "model_size": self.model_size,
            "language": self.language,
            "task": self.task,
            "sample_rate": self.sample_rate,
            "device": self.device,
            "fp16_enabled": self.device == "cuda"
        }