from __future__ import annotations

import glob
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn

from src.core.base_extractor import BaseExtractor
from src.core.metrics import system_snapshot, process_memory_bytes
from src.core.path_utils import default_artifacts_dir


class ChunkAutoencoder(nn.Module):
    def __init__(self, input_dim: int = 384, bottleneck_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Linear(512, bottleneck_dim),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck_dim, 512),
            nn.ReLU(),
            nn.Linear(512, input_dim),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z = self.encoder(x)
        x_hat = self.decoder(z)
        return z, x_hat


def _latest(art_dir: Path, pattern: str) -> Optional[Path]:
    files = glob.glob(str(art_dir / pattern))
    if not files:
        return None
    files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return Path(files[0])


def _l2n(v: np.ndarray) -> np.ndarray:
    n = float(np.linalg.norm(v))
    if n > 0:
        return v / n
    return v


class LongformEmbeddingSummaryExtractor(BaseExtractor):
    """
    Сжимает chunk embeddings длинного текста через автоэнкодер (инференс-режим, без обучения).
    Собирает все доступные чанковые эмбеддинги (whisper/youtube), пропускает через encoder и усредняет bottleneck-представления.
    """

    VERSION = "1.0.0"

    def __init__(
        self,
        artifacts_dir: str | None = None,
        input_dim: int = 384,
        bottleneck_dim: int = 256,
        device: str = "cpu",
    ) -> None:
        self.artifacts_dir = Path(artifacts_dir).expanduser().resolve() if artifacts_dir else default_artifacts_dir()
        self.device = device
        self.bottleneck_dim = bottleneck_dim
        self.model = ChunkAutoencoder(input_dim=input_dim, bottleneck_dim=bottleneck_dim).to(self.device)
        self.model.eval()

    def _load_all_chunk_embeddings(self) -> List[np.ndarray]:
        out: List[np.ndarray] = []
        for src in ["whisper", "youtube_auto"]:
            p = _latest(self.artifacts_dir, f"transcript_{src}_embedding_*.npy")
            if p is None:
                continue
            try:
                arr = np.load(p)
                arr = np.asarray(arr, dtype=np.float32)
                if arr.ndim == 1:
                    arr = arr.reshape(1, -1)
                out.append(arr)
            except Exception:
                continue
        return out

    def extract(self, doc: Any) -> Dict[str, Any]:
        import time

        t0 = time.perf_counter()
        sys_before = system_snapshot()
        mem_before = process_memory_bytes()

        chunks_list = self._load_all_chunk_embeddings()
        if not chunks_list:
            sys_after = system_snapshot()
            mem_after = process_memory_bytes()
            total_s = time.perf_counter() - t0
            return {
                "device": self.device,
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
                "result": {"longform_summary": {"error": "no_chunk_embeddings_found"}},
                "error": None,
            }

        all_chunks = np.vstack(chunks_list)
        x = torch.tensor(all_chunks, dtype=torch.float32, device=self.device)

        with torch.no_grad():
            z, _ = self.model(x)
        z_np = z.detach().cpu().numpy()

        summary = _l2n(np.mean(z_np, axis=0))

        # save artifact
        out_path = self.artifacts_dir / "longform_summary_embedding.npy"
        tmp = out_path.with_suffix(".tmp.npy")
        np.save(tmp, summary.astype(np.float32))
        tmp.replace(out_path)

        sys_after = system_snapshot()
        mem_after = process_memory_bytes()
        total_s = time.perf_counter() - t0

        return {
            "device": self.device,
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
                "longform_summary": {
                    "path": str(out_path.resolve()),
                    "bottleneck_dim": int(self.bottleneck_dim),
                    "n_chunks": int(all_chunks.shape[0]),
                }
            },
            "error": None,
        }



