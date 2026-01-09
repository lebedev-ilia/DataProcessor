"""
Source separation extractor (Triton-backed) + Segmenter windows.

We intentionally do NOT output stems (too large). We output:
- per-window energy shares for [vocals, drums, bass, other]
- aggregated mean shares (and basic dispersion)

Policy:
- NO fallback (model missing / triton unavailable => ERROR)
- uses Segmenter `audio/segments.json` family: `source_separation`
- `<5s` audio => ERROR
- truly silent audio => EMPTY (status="empty", empty_reason="audio_silent")

Triton interface (spec-defined):
- input: log-mel features, shape [B, n_mels, T] float32
- output: per-source energies, shape [B, 4] float32 (order: vocals, drums, bass, other)
"""

from __future__ import annotations

import os
import time
import logging
from typing import Any, Dict, List, Optional

import numpy as np

from src.core.base_extractor import BaseExtractor, ExtractorResult
from src.core.audio_utils import AudioUtils
from dp_triton import TritonHttpClient, TritonError  # type: ignore

logger = logging.getLogger(__name__)


class SourceSeparationExtractor(BaseExtractor):
    name = "source_separation_extractor"
    version = "2.0.0"
    description = "Source separation shares via Triton (log-mel input)"
    category = "source_separation"
    dependencies = ["numpy", "dp_triton", "dp_models", "torchaudio", "torch"]
    estimated_duration = 12.0

    gpu_required = False
    gpu_preferred = True
    gpu_memory_required = 0.0  # Triton-backed

    def __init__(
        self,
        device: str = "auto",
        model_size: str = "small",
        batch_size: int = 8,
    ):
        super().__init__(device=device)
        self.model_size = str(model_size or "small").strip().lower()
        if self.model_size not in ("small", "medium", "large"):
            raise ValueError(f"source_separation | unsupported model_size={self.model_size}. Expected: small|medium|large")
        self.batch_size = max(1, int(batch_size))

        # Resolve model via ModelManager.
        try:
            from dp_models import get_global_model_manager  # type: ignore

            self._mm = get_global_model_manager()
        except Exception as e:
            raise RuntimeError(f"source_separation | ModelManager is required but failed to init: {e}") from e

        spec_name = f"source_separation_{self.model_size}_triton"
        try:
            self.model_spec = self._mm.get_spec(model_name=spec_name)
            _dev, _prec, rt, _eng, wd, _arts = self._mm.resolve(self.model_spec)
            if str(rt) != "triton":
                raise RuntimeError(f"source_separation | expected runtime=triton in spec {spec_name}, got {rt}")
            self.model_name = str(self.model_spec.model_name)
            self.weights_digest = str(wd)
            rp = self.model_spec.runtime_params or {}
            self.triton_http_url = self._expand_env(str(rp.get("triton_http_url") or os.environ.get("TRITON_HTTP_URL") or ""))
            self.triton_model_name = str(rp.get("triton_model_name") or "")
            self.triton_model_version = rp.get("triton_model_version")
            self.triton_input_name = str(rp.get("triton_input_name") or "MEL__0")
            self.triton_input_datatype = str(rp.get("triton_input_datatype") or "FP32")
            self.triton_output_name = str(rp.get("triton_output_energy_name") or "ENERGY__0")
            self.triton_output_datatype = str(rp.get("triton_output_energy_datatype") or "FP32")
            # preprocess params
            self.sample_rate = int(rp.get("sample_rate") or 44100)
            self.n_fft = int(rp.get("n_fft") or 2048)
            self.hop_length = int(rp.get("hop_length") or 512)
            self.n_mels = int(rp.get("n_mels") or 64)
            if not self.triton_http_url or not self.triton_model_name:
                raise RuntimeError("source_separation | Triton runtime_params missing triton_http_url/triton_model_name")
        except Exception as e:
            raise RuntimeError(f"source_separation | failed to resolve model spec via ModelManager: {e}") from e

        self._client = TritonHttpClient(base_url=self.triton_http_url, timeout_sec=15.0)
        self.audio_utils = AudioUtils(device=device, sample_rate=self.sample_rate)

        # Build mel transform lazily (import torch/torchaudio here).
        try:
            import torch
            import torchaudio

            self._torch = torch
            self._mel = torchaudio.transforms.MelSpectrogram(
                sample_rate=self.sample_rate,
                n_fft=self.n_fft,
                hop_length=self.hop_length,
                n_mels=self.n_mels,
                power=2.0,
            )
            self._amptodb = torchaudio.transforms.AmplitudeToDB(stype="power")
        except Exception as e:
            raise RuntimeError(f"source_separation | torchaudio/torch is required for mel preprocessing: {e}") from e

        self._source_names = ["vocals", "drums", "bass", "other"]

    def _expand_env(self, s: str) -> str:
        if "${" not in str(s):
            return str(s)
        import re

        def repl(m):
            return os.environ.get(m.group(1), "")

        return re.sub(r"\$\{([^}]+)\}", repl, str(s))

    @staticmethod
    def _rms_and_peak(x: np.ndarray) -> tuple[float, float]:
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        if x.size == 0:
            return 0.0, 0.0
        rms = float(np.sqrt(float(np.mean(x * x)) + 1e-12))
        peak = float(np.max(np.abs(x)) + 1e-12)
        return rms, peak

    def _mel_log(self, wav_1d: np.ndarray) -> np.ndarray:
        """
        wav_1d: float32 [-1..1], shape [T]
        returns log-mel [n_mels, frames] float32
        """
        t = self._torch.from_numpy(np.asarray(wav_1d, dtype=np.float32).reshape(1, -1))  # [1, T]
        # Always on CPU for stable preprocessing.
        t = t.cpu()
        with self._torch.no_grad():
            mel = self._mel(t)  # [1, n_mels, frames]
            mel_db = self._amptodb(mel)  # [1, n_mels, frames]
        out = mel_db.squeeze(0).contiguous().numpy().astype(np.float32)
        return out

    def run_segments(self, input_uri: str, tmp_path: str, segments: List[Dict[str, Any]]) -> ExtractorResult:
        start_time = time.time()
        try:
            if not self._validate_input(input_uri):
                return self._create_result(False, error="Некорректный входной файл", processing_time=time.time() - start_time)
            if not isinstance(segments, list) or not segments:
                raise ValueError("segments is empty (no-fallback)")

            dur_sec = float(max((float(s.get("end_sec", 0.0)) for s in segments), default=0.0))
            if dur_sec < 5.0:
                raise RuntimeError(f"source_separation | audio too short (<5s): duration_sec={dur_sec:.3f}")

            # Load audio windows and compute mel.
            mels: list[np.ndarray] = []
            starts: list[float] = []
            ends: list[float] = []
            centers: list[float] = []
            peaks: list[float] = []
            rmss: list[float] = []

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
                    raise RuntimeError(f"source_separation | segment SR mismatch: got {sr} expected {self.sample_rate}")

                rms, peak = self._rms_and_peak(wav)
                rmss.append(float(rms))
                peaks.append(float(peak))
                mels.append(self._mel_log(wav))
                starts.append(st)
                ends.append(en)
                centers.append(c)

            # Global silence decision: only if ALL windows are silent.
            if (max(peaks) if peaks else 0.0) < 1e-3 and (max(rmss) if rmss else 0.0) < 1e-4:
                payload: Dict[str, Any] = {
                    "status": "empty",
                    "empty_reason": "audio_silent",
                    "segments_count": int(len(segments)),
                    "sample_rate": int(self.sample_rate),
                    "model_name": self.model_name,
                    "device_used": "cuda",
                }
                return self._create_result(True, payload=payload, processing_time=time.time() - start_time)

            if not self._client.ready():
                raise TritonError(f"Triton is not ready at {self.triton_http_url}", error_code="triton_unavailable")

            # Pad mel time dimension for batching
            t_max = int(max(m.shape[1] for m in mels)) if mels else 0
            if t_max <= 0:
                raise RuntimeError("source_separation | empty mel features")
            batch_in = np.zeros((len(mels), self.n_mels, t_max), dtype=np.float32)
            for i, m in enumerate(mels):
                batch_in[i, :, : m.shape[1]] = m

            energies = []
            for start in range(0, batch_in.shape[0], self.batch_size):
                b = batch_in[start : start + self.batch_size]
                res = self._client.infer(
                    model_name=self.triton_model_name,
                    model_version=(str(self.triton_model_version) if self.triton_model_version else None),
                    input_name=self.triton_input_name,
                    input_tensor=b,
                    output_name=self.triton_output_name,
                    datatype=self.triton_input_datatype,
                ).output
                e = np.asarray(res, dtype=np.float32)
                if e.ndim != 2 or e.shape[1] != 4:
                    raise RuntimeError(f"source_separation | unexpected energy output shape: {e.shape} (expected [B,4])")
                energies.append(e)
            energy = np.concatenate(energies, axis=0) if energies else np.zeros((0, 4), dtype=np.float32)
            if energy.shape[0] != len(segments):
                raise RuntimeError(f"source_separation | energy count mismatch: {energy.shape[0]} vs {len(segments)}")

            total = np.sum(energy, axis=1, keepdims=True) + 1e-9
            shares = energy / total  # [N,4]

            # Aggregate (mean + std) over windows
            share_mean = np.mean(shares, axis=0).astype(np.float32) if shares.size else np.zeros((4,), dtype=np.float32)
            share_std = np.std(shares, axis=0).astype(np.float32) if shares.size else np.zeros((4,), dtype=np.float32)

            payload = {
                "share_sequence": shares.astype(np.float32),
                "energy_sequence": energy.astype(np.float32),
                "share_mean": share_mean,
                "share_std": share_std,
                "segments_count": int(len(segments)),
                "segment_start_sec": starts,
                "segment_end_sec": ends,
                "segment_center_sec": centers,
                "sample_rate": int(self.sample_rate),
                "device_used": "cuda",
                "model_name": self.model_name,
                "source_order": self._source_names,
            }
            return self._create_result(True, payload=payload, processing_time=time.time() - start_time)
        except TritonError as e:
            return self._create_result(False, error=str(e), processing_time=time.time() - start_time)
        except Exception as e:
            return self._create_result(False, error=str(e), processing_time=time.time() - start_time)

    def run(self, input_uri: str, tmp_path: str) -> ExtractorResult:
        return self._create_result(
            success=False,
            error="source_separation_extractor | run() is not supported in production. Use run_segments() with Segmenter families.source_separation windows.",
            processing_time=0.0,
        )


