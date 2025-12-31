from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import warnings
warnings.filterwarnings("ignore", category=FutureWarning, module=r"torch\.cuda")
warnings.filterwarnings("ignore", message="The pynvml package is deprecated", category=FutureWarning)

import numpy as np


def _validate_artifact(entry: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Validate artifact metadata: file exists, shape matches, dtype matches.
    Supports vectors only (1D) for now.
    """
    try:
        path = entry.get("path")
        shape = entry.get("shape")
        dtype = entry.get("dtype")
        if not path or not shape or not dtype:
            return False, "missing metadata"
        p = Path(path)
        if not p.exists():
            return False, "file not found"
        arr = np.load(p)
        if list(arr.shape) != list(shape):
            return False, f"shape mismatch: {arr.shape} != {shape}"
        if str(arr.dtype) != str(dtype):
            return False, f"dtype mismatch: {arr.dtype} != {dtype}"
        return True, "ok"
    except Exception as e:
        return False, f"exception: {e}"


def _flatten_metrics(extractor_result: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    out["device"] = extractor_result.get("device")
    out["version"] = extractor_result.get("version")
    # per-extractor timings
    timings_by = extractor_result.get("timings_by_extractor") or {}
    if isinstance(timings_by, dict):
        for name, t in timings_by.items():
            if not isinstance(t, dict):
                continue
            for k, v in t.items():
                out[f"timing_{name}_{k}_s"] = v
    # compute peak across systems_by_extractor if available
    peak_mb_candidates: List[int] = []
    peak_gpu_mb_candidates: List[int] = []
    systems_by = extractor_result.get("systems_by_extractor") or {}
    if isinstance(systems_by, dict):
        for _name, sysinfo in systems_by.items():
            if not isinstance(sysinfo, dict):
                continue
            pk = (sysinfo.get("peaks") or {})
            val_mb = pk.get("ram_peak_mb")
            if isinstance(val_mb, (int, float)):
                peak_mb_candidates.append(int(val_mb))
            else:
                val_b = pk.get("ram_peak_bytes")
                if isinstance(val_b, (int, float)):
                    peak_mb_candidates.append(int(round(val_b / 1_000_000)))
            gpu_mb = pk.get("gpu_peak_mb")
            if isinstance(gpu_mb, (int, float)):
                peak_gpu_mb_candidates.append(int(gpu_mb))
    if not peak_mb_candidates:
        sys = extractor_result.get("system") or {}
        peaks = (sys.get("peaks") or {})
        val_mb = peaks.get("ram_peak_mb")
        if isinstance(val_mb, (int, float)):
            peak_mb_candidates.append(int(val_mb))
        else:
            val_b = peaks.get("ram_peak_bytes")
            if isinstance(val_b, (int, float)):
                peak_mb_candidates.append(int(round(val_b / 1_000_000)))
    if peak_mb_candidates:
        out["ram_peak_mb"] = max(peak_mb_candidates)
    if peak_gpu_mb_candidates:
        out["gpu_peak_mb"] = max(peak_gpu_mb_candidates)
    out["error"] = extractor_result.get("error")
    return out


def run_once(input_path: str, devices_config_json: dict[str, str] | None = None) -> Dict[str, Any]:
    from src.core.main_processor import MainProcessor, load_document_from_json
    doc = load_document_from_json(input_path)
    processor = MainProcessor(devices_config=devices_config_json)

    t0 = time.perf_counter()
    features = processor.run(doc)
    total_ms = (time.perf_counter() - t0) * 1000.0

    # (no validations in output per user request)

    flattened = _flatten_metrics(features if isinstance(features, dict) else {})

    return {
        "total_s": round(total_ms / 1000.0, 3),
        "features": features,
        "flattened": flattened,
    }


def main() -> None:

    repeat = 1
    input = os.path.abspath(os.path.join(os.path.dirname(__file__), "example_input.json"))
    devices_config = {
        # 1) Clean texts / extract hashtags and lexical/asr stats
        "cpu": [
            "LexicalStatsExtractor",
            "TagsExtractor",
            "ASRTextProxyExtractor",
        ],
        # 2) All embedders on GPU in strict order
        "gpu": [
            "TitleEmbedder",
            "DescriptionEmbedder",
            "TranscriptChunkEmbedder",
            "CommentsEmbedder",
            "HashtagEmbedder",
            "SpeakerTurnEmbeddingsAggregatorExtractor",
            "QAEmbeddingPairsExtractor",
            "EmbeddingPairTopKExtractor",
            "SemanticTopicExtractor",
        ],
        # 3) Aggregators and pairwise metrics (depend on produced artifacts)
        "cpu2": [
            "TranscriptAggregatorExtractor",
            "CommentsAggregationExtractor",
            "EmbeddingStatsExtractor",
            "EmbeddingShiftIndicatorExtractor",
            # "LongformEmbeddingSummaryExtractor", недоступен пока необучен энкодер на 100к видео
            # "SemanticClusterExtractor", недоступен пока нет корпуса минимум из 1000 видео
            # "TopKSimilarCorpusTitlesExtractor", недоступен пока нет корпуса минимум из 1000 видео
            "TitleEmbeddingClusterEntropyExtractor",
            "TitleToHashtagCosineExtractor",
            "CosineMetricsExtractor",
            "EmbeddingSourceIdExtractor",
        ],
    }

    runs: List[Dict[str, Any]] = []
    for _ in range(max(1, repeat)):
        runs.append(run_once(input, devices_config))

    total_times = [r.get("total_s", 0.0) for r in runs]
    total_seconds = [round((t or 0.0) / 1000.0, 3) for t in total_times]
    summary = {
        "num_runs": len(runs),
        "total_s": sum(total_times),
        "errors": [r.get("flattened", {}).get("error") for r in runs if r.get("flattened", {}).get("error")],
    }

    report = {"runs": runs, "summary": summary}

    json.dump(report, open("report.json", "w"), ensure_ascii=False, indent=4)


if __name__ == "__main__":
    main()


