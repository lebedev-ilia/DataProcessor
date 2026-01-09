"""
Speaker diarization (segment-level) via Triton speaker embedding model + clustering.
"""
import time
import logging
import numpy as np
import os
from typing import Dict, Any, Optional, List
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics.pairwise import cosine_distances

from src.core.base_extractor import BaseExtractor, ExtractorResult
from src.core.audio_utils import AudioUtils
from dp_triton import TritonHttpClient, TritonError  # type: ignore

logger = logging.getLogger(__name__)


class SpeakerDiarizationExtractor(BaseExtractor):
    """
    Computes speaker embeddings on fixed audio windows (from Segmenter families.diarization),
    clusters them into speaker IDs, and stores mean per-speaker embeddings in the NPZ (no temp .npy).
    """
    
    name = "speaker_diarization_extractor"
    version = "2.0.0"
    description = "Speaker diarization via Triton embeddings + clustering"
    category = "speech"
    dependencies = ["numpy", "scikit-learn", "dp_triton", "dp_models"]
    estimated_duration = 6.0
    
    gpu_required = False
    gpu_preferred = True
    gpu_memory_required = 0.0  # Triton-backed model
    
    def __init__(
        self, 
        device: str = "auto",
        model_size: str = "small",
        min_speakers: int = 1,
        max_speakers: int = 6,
        sample_rate: int = 16000,
        clustering_method: str = "agglomerative"
    ):
        """
        Инициализация диаризационного экстрактора.
        
        Args:
            device: Устройство для обработки
            model_size: small|large (Triton model selection via ModelManager)
            min_speakers: Минимальное количество спикеров
            max_speakers: Максимальное количество спикеров
            sample_rate: Частота дискретизации
            clustering_method: Метод кластеризации ('agglomerative', 'kmeans')
        """
        super().__init__(device=device)
        
        self.model_size = str(model_size or "small").strip().lower()
        if self.model_size not in ("small", "large"):
            raise ValueError(f"speaker_diarization | unsupported model_size={self.model_size}. Expected: small|large")
        self.min_speakers = int(min_speakers)
        self.max_speakers = int(max_speakers)
        if self.min_speakers <= 0 or self.max_speakers <= 0 or self.max_speakers < self.min_speakers:
            raise ValueError("speaker_diarization | invalid min/max speakers bounds")
        self.sample_rate = int(sample_rate)
        self.clustering_method = str(clustering_method or "agglomerative")
        
        self.audio_utils = AudioUtils(device=device, sample_rate=sample_rate)

        # ModelManager: resolve Triton runtime params (no-network).
        try:
            from dp_models import get_global_model_manager  # type: ignore

            self._mm = get_global_model_manager()
        except Exception as e:
            raise RuntimeError(f"speaker_diarization | ModelManager is required but failed to init: {e}") from e

        spec_name = f"speaker_diarization_{self.model_size}_triton"
        try:
            self.model_spec = self._mm.get_spec(model_name=spec_name)
            _dev, _prec, rt, _eng, wd, _arts = self._mm.resolve(self.model_spec)
            if str(rt) != "triton":
                raise RuntimeError(f"speaker_diarization | expected runtime=triton in spec {spec_name}, got {rt}")
            self.model_name = str(self.model_spec.model_name)
            self.weights_digest = str(wd)
            rp = self.model_spec.runtime_params or {}
            self.triton_http_url = self._expand_env(str(rp.get("triton_http_url") or os.environ.get("TRITON_HTTP_URL") or ""))
            self.triton_model_name = str(rp.get("triton_model_name") or "")
            self.triton_model_version = rp.get("triton_model_version")
            self.triton_input_name = str(rp.get("triton_input_name") or "AUDIO__0")
            self.triton_input_datatype = str(rp.get("triton_input_datatype") or "FP32")
            self.triton_output_name = str(rp.get("triton_output_embeddings_name") or "EMB__0")
            self.triton_output_datatype = str(rp.get("triton_output_embeddings_datatype") or "FP32")
            if not self.triton_http_url or not self.triton_model_name:
                raise RuntimeError("speaker_diarization | Triton runtime_params missing triton_http_url/triton_model_name")
        except Exception as e:
            raise RuntimeError(f"speaker_diarization | failed to resolve model spec via ModelManager: {e}") from e

        self._client = TritonHttpClient(base_url=self.triton_http_url, timeout_sec=10.0)

    def _expand_env(self, s: str) -> str:
        if "${" not in str(s):
            return str(s)
        import re

        def repl(m):
            return os.environ.get(m.group(1), "")

        return re.sub(r"\$\{([^}]+)\}", repl, str(s))

    def _rms_and_peak(self, x: np.ndarray) -> tuple[float, float]:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return 0.0, 0.0
        rms = float(np.sqrt(float(np.mean(x * x)) + 1e-12))
        peak = float(np.max(np.abs(x)) + 1e-12)
        return rms, peak

    def run_segments(self, input_uri: str, tmp_path: str, segments: List[Dict[str, Any]]) -> ExtractorResult:
        """
        Segmenter-driven diarization: compute embeddings on provided windows and cluster.

        Empty semantics:
        - if audio is essentially silent -> status=empty + empty_reason=audio_silent
        Error semantics:
        - duration < 5 sec -> error (policy)
        - missing model/spec/triton -> error
        """
        start_time = time.time()
        try:
            if not self._validate_input(input_uri):
                return self._create_result(False, error="Некорректный входной файл", processing_time=time.time() - start_time)
            if not isinstance(segments, list) or not segments:
                raise ValueError("segments is empty (no-fallback)")

            # Fail-fast for short audio (<5 sec): use metadata fields in segments.
            dur_sec = float(max((float(s.get("end_sec", 0.0)) for s in segments), default=0.0))
            if dur_sec < 5.0:
                raise RuntimeError(f"speaker_diarization | audio too short for diarization (<5s): duration_sec={dur_sec:.3f}")

            # Load and pad segments to fixed length for batching.
            # Use max segment length to avoid truncation; pad with zeros.
            lens = []
            waves = []
            starts = []
            ends = []
            centers = []
            for seg in segments:
                ss = int(seg.get("start_sample"))
                es = int(seg.get("end_sample"))
                st = float(seg.get("start_sec"))
                en = float(seg.get("end_sec"))
                c = float(seg.get("center_sec"))
                wav_t, sr = self.audio_utils.load_audio_segment(input_uri, start_sample=ss, end_sample=es, target_sr=self.sample_rate)
                wav = self.audio_utils.to_numpy(wav_t)
                wav = wav[0] if wav.ndim == 2 else wav.reshape(-1)
                wav = np.asarray(wav, dtype=np.float32).reshape(-1)
                if int(sr) != int(self.sample_rate):
                    raise RuntimeError(f"speaker_diarization | segment SR mismatch: got {sr} expected {self.sample_rate}")
                lens.append(int(wav.shape[0]))
                waves.append(wav)
                starts.append(st)
                ends.append(en)
                centers.append(c)

            max_len = int(max(lens) if lens else 0)
            if max_len <= 0:
                raise RuntimeError("speaker_diarization | no audio samples in segments")

            # Silence detection on concatenated windows (cheap): if peak too low -> empty.
            concat = np.concatenate([w for w in waves if w.size], axis=0) if waves else np.zeros((0,), dtype=np.float32)
            rms, peak = self._rms_and_peak(concat)
            # Threshold chosen for float audio in [-1..1]. If audio extraction is broken, load_audio_segment should error earlier.
            if peak < 1e-3 and rms < 1e-4:
                payload = {
                    "status": "empty",
                    "empty_reason": "audio_silent",
                    "speaker_segments": [],
                    "speaker_count": 0,
                    "speaker_embeddings_mean": [],
                    "speaker_ids": [],
                    "segment_start_sec": starts,
                    "segment_end_sec": ends,
                    "segment_center_sec": centers,
                    "segments_count": int(len(segments)),
                    "sample_rate": int(self.sample_rate),
                    "rms": float(rms),
                    "peak": float(peak),
                    "model_name": self.model_name,
                }
                return self._create_result(True, payload=payload, processing_time=time.time() - start_time)

            padded = np.zeros((len(waves), max_len), dtype=np.float32)
            for i, w in enumerate(waves):
                n = int(w.shape[0])
                padded[i, :n] = w

            if not self._client.ready():
                raise TritonError(f"Triton is not ready at {self.triton_http_url}", error_code="triton_unavailable")

            emb = self._client.infer(
                model_name=self.triton_model_name,
                model_version=(str(self.triton_model_version) if self.triton_model_version else None),
                input_name=self.triton_input_name,
                input_tensor=padded,
                output_name=self.triton_output_name,
                datatype=self.triton_input_datatype,
            ).output
            emb = np.asarray(emb, dtype=np.float32)
            if emb.ndim != 2 or emb.shape[0] != padded.shape[0]:
                raise RuntimeError(f"speaker_diarization | unexpected embeddings shape from Triton: {emb.shape}")

            # Cluster embeddings -> segment-level speaker labels
            labels = self._cluster_speakers_arr(emb)
            processed = self._process_from_labels(
                labels=labels,
                embeddings=emb,
                segment_starts=starts,
                segment_ends=ends,
            )

            payload = {
                **processed,
                "segment_start_sec": starts,
                "segment_end_sec": ends,
                "segment_center_sec": centers,
                "segments_count": int(len(segments)),
                "sample_rate": int(self.sample_rate),
                "device_used": "cuda",
                "rms": float(rms),
                "peak": float(peak),
                "model_name": self.model_name,
            }
            return self._create_result(True, payload=payload, processing_time=time.time() - start_time)
        except TritonError as e:
            return self._create_result(False, error=str(e), processing_time=time.time() - start_time)
        except Exception as e:
            return self._create_result(False, error=str(e), processing_time=time.time() - start_time)
    
    def run(self, input_uri: str, tmp_path: str) -> ExtractorResult:
        """
        Извлечение диаризации спикеров.
        
        Args:
            input_uri: Путь к аудио файлу
            tmp_path: Временная директория
            
        Returns:
            ExtractorResult с диаризацией спикеров
        """
        return self._create_result(
            success=False,
            error="speaker_diarization_extractor | run() is not supported in production. Use run_segments() with Segmenter families.diarization windows.",
            processing_time=0.0,
        )
    
    def _estimate_speaker_count(self, emb: np.ndarray) -> int:
        """
        Heuristic speaker count estimation, clamped to [min_speakers, max_speakers].
        """
        n = int(emb.shape[0])
        if n < 2:
            return self.min_speakers
        d = cosine_distances(emb)
        mean_distance = float(np.mean(d[np.triu_indices_from(d, k=1)]))
        # Simple heuristic bands; tuned to be conservative.
        if mean_distance > 0.32:
            k = min(self.max_speakers, max(self.min_speakers, 3))
        elif mean_distance > 0.24:
            k = min(self.max_speakers, max(self.min_speakers, 2))
        else:
            k = self.min_speakers
        return int(max(self.min_speakers, min(self.max_speakers, k)))

    def _cluster_speakers_arr(self, emb: np.ndarray) -> List[int]:
        if emb.size == 0:
            return []
        k = self._estimate_speaker_count(emb)
        if k <= 1:
            return [0] * int(emb.shape[0])
        clustering = AgglomerativeClustering(n_clusters=int(k))
        labels = clustering.fit_predict(emb)
        return [int(x) for x in labels.tolist()]

    def _process_from_labels(
        self,
        *,
        labels: List[int],
        embeddings: np.ndarray,
        segment_starts: List[float],
        segment_ends: List[float],
    ) -> Dict[str, Any]:
        speaker_segments = []
        for i, sid in enumerate(labels):
            speaker_segments.append(
                {
                    "start": float(segment_starts[i]),
                    "end": float(segment_ends[i]),
                    "duration": float(max(0.0, float(segment_ends[i]) - float(segment_starts[i]))),
                    "speaker_id": int(sid),
                    "segment_index": int(i),
                }
            )

        uniq = sorted(set(int(x) for x in labels)) if labels else []
        speaker_count = int(len(uniq))

        # Mean embedding per speaker
        speaker_ids = []
        speaker_emb = []
        per_stats = {}
        for sid in uniq:
            idx = [i for i, x in enumerate(labels) if int(x) == int(sid)]
            if not idx:
                continue
            m = np.mean(embeddings[np.asarray(idx, dtype=np.int32)], axis=0)
            speaker_ids.append(int(sid))
            speaker_emb.append(np.asarray(m, dtype=np.float32))
            per_stats[int(sid)] = {
                "segments_count": int(len(idx)),
                "total_duration": float(sum(float(segment_ends[i]) - float(segment_starts[i]) for i in idx)),
            }

        speaker_embeddings_mean = np.stack(speaker_emb, axis=0).astype(np.float32) if speaker_emb else np.zeros((0, 0), dtype=np.float32)
        duration = float(max(segment_ends) if segment_ends else 0.0)

        return {
            "speaker_segments": speaker_segments,
            "speaker_count": speaker_count,
            "duration": duration,
            "speaker_ids": speaker_ids,
            "speaker_embeddings_mean": speaker_embeddings_mean,
            "speaker_stats": per_stats,
        }
    
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
    
    def get_encoder_info(self) -> Dict[str, Any]:
        """Получение информации об энкодере."""
        return {
            "model_size": self.model_size,
            "min_speakers": self.min_speakers,
            "max_speakers": self.max_speakers,
            "sample_rate": self.sample_rate,
            "clustering_method": self.clustering_method,
            "device": self.device
        }
