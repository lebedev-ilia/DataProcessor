#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from huggingface_hub import HfApi  # type: ignore


@dataclass
class IndexMeta:
    owner: str
    datasets: List[str]
    total_files_seen: int
    total_mp4: int


def main() -> int:
    p = argparse.ArgumentParser(description="Build mapping video_id -> HF dataset repo for videos1..videos11")
    p.add_argument("--owner", type=str, required=True)
    p.add_argument("--datasets", type=str, required=True, help="Comma-separated dataset repo names (without owner)")
    p.add_argument("--out", type=str, required=True)
    args = p.parse_args()

    owner = args.owner.strip()
    datasets = [x.strip() for x in args.datasets.split(",") if x.strip()]
    api = HfApi()

    mapping: Dict[str, str] = {}
    total_files = 0
    total_mp4 = 0

    for ds in datasets:
        repo_id = f"{owner}/{ds}"
        files = api.list_repo_files(repo_id=repo_id, repo_type="dataset")
        total_files += len(files)
        for f in files:
            if not f.lower().endswith(".mp4"):
                continue
            total_mp4 += 1
            vid = Path(f).name
            if vid.lower().endswith(".mp4"):
                vid = vid[:-4]
            if vid and vid not in mapping:
                mapping[vid] = repo_id

    out_path = Path(args.out).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "meta": IndexMeta(owner=owner, datasets=datasets, total_files_seen=total_files, total_mp4=total_mp4).__dict__,
        "video_id_to_repo": mapping,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[ok] wrote {out_path} (video_ids={len(mapping)}, mp4_files={total_mp4})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


