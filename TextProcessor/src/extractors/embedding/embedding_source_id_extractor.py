import glob
import os
import uuid
import hashlib
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from src.core.base_extractor import BaseExtractor
from src.core.metrics import system_snapshot, process_memory_bytes


class EmbeddingSourceIdExtractor(BaseExtractor):
    VERSION = "1.0.0"

    def __init__(
        self,
        vector_store_uri: str = "faiss://semantic_titles_v1",
        model_version: str = "unknown",
        artifacts_dir: str = "/home/ilya/Рабочий стол/DataProcessor/TextProcessor/.artifacts",
    ) -> None:
        self.vector_store_uri = vector_store_uri
        self.model_version = model_version
        self.artifacts_dir = Path(artifacts_dir)

    @staticmethod
    def _generate_stable_id(file_path: str) -> str:
        if not os.path.exists(file_path):
            return "missing_embedding"
        try:
            with open(file_path, "rb") as f:
                content = f.read()
            sha = hashlib.sha1(content).hexdigest()
            return f"{sha[:12]}-{uuid.uuid5(uuid.NAMESPACE_DNS, file_path)}"
        except Exception:
            return str(uuid.uuid4())

    def _find_primary_embedding(self) -> Optional[str]:
        # Priority: title -> transcript aggregated -> description
        patterns = [
            "title_embedding_*.npy",
            "transcript_*_mean_embedding_*.npy",
            "description_embedding_*.npy",
        ]
        for pattern in patterns:
            files = glob.glob(str(self.artifacts_dir / pattern))
            if files:
                files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
                return files[0]
        return None

    @staticmethod
    def _try_read_model_meta(npy_path: str) -> Optional[str]:
        try:
            meta_path = Path(npy_path).with_suffix(".meta.json")
            if meta_path.exists():
                import json as _json
                data = _json.loads(meta_path.read_text())
                mv = data.get("model")
                if isinstance(mv, str) and mv:
                    return mv
        except Exception:
            pass
        # special handling for transcript chunks: meta stored in cache by hash
        try:
            name = Path(npy_path).name
            if name.startswith("transcript_") and "_embedding_" in name:
                h = name.split("_embedding_")[-1].split(".")[0]
                cache_dir = Path(str(self.artifacts_dir).replace(".artifacts", ".cache/transcript_embed"))
                meta_path2 = cache_dir / f"{h}.json"
                if meta_path2.exists():
                    import json as _json
                    data2 = _json.loads(meta_path2.read_text())
                    mv2 = data2.get("model")
                    if isinstance(mv2, str) and mv2:
                        return mv2
        except Exception:
            pass
        return None

    def extract(self, doc: Any) -> Dict[str, Any]:
        import time

        t0 = time.perf_counter()
        sys_before = system_snapshot()
        mem_before = process_memory_bytes()

        emb_path = self._find_primary_embedding()
        if not emb_path:
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
                "result": {"embedding_source_id": {"error": "no_embedding_found"}},
                "error": None,
            }

        vector_id = self._generate_stable_id(emb_path)
        # determine model_version from meta files of the selected embedding
        model_version = self._try_read_model_meta(emb_path) or self.model_version

        result = {
            "embedding_source_id": {
                "vector_id": vector_id,
                "vector_store_uri": self.vector_store_uri,
                "model_version": model_version,
                "created_at": datetime.utcnow().isoformat() + "Z",
                "embedding_path": emb_path,
            }
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
            "result": result,
            "error": None,
        }


