#!/usr/bin/env python3
"""
One-time bootstrap: download/copy all required baseline model artifacts into DP_MODELS_ROOT.

Goal:
- allow network ONCE during setup
- populate DP_MODELS_ROOT with pinned caches + canonical artifacts
- then run everything offline via dp_models.offline.enforce_offline_env()

What we populate:
- torch_cache/ (torch.hub repos + checkpoints, torchvision weights)
- hf_cache/ (huggingface hub snapshots: openai CLIP, etc.)
- visual/places365/ (categories + checkpoint copied from torch cache if present)
- visual/clip/ (openai clip weights .pt copied from ~/.cache/clip if present; optional download)
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path
from typing import Optional


def _mkdir(p: str) -> None:
    Path(p).mkdir(parents=True, exist_ok=True)


def _copy_if_exists(src: str, dst: str) -> bool:
    if not os.path.exists(src):
        return False
    _mkdir(os.path.dirname(dst))
    shutil.copy2(src, dst)
    return True


def _env_home() -> str:
    return str(Path.home())


def _default_models_root(repo_root: str) -> str:
    return os.path.join(repo_root, "dp_models", "bundled_models")


def _repo_root() -> str:
    # scripts/model_opt/bootstrap_models_root.py lives at <repo>/scripts/model_opt/...
    return os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))


# Ensure repo root is importable (dp_models, dp_triton, etc.)
_RR = _repo_root()
if _RR not in sys.path:
    sys.path.insert(0, _RR)


def main() -> None:
    ap = argparse.ArgumentParser("Bootstrap DP_MODELS_ROOT (one-time download/copy)")
    ap.add_argument("--models-root", type=str, default=None, help="Destination DP_MODELS_ROOT (default: dp_models/bundled_models)")
    ap.add_argument("--allow-network", action="store_true", help="Allow network downloads (recommended for bootstrap)")
    ap.add_argument("--from-cache-only", action="store_true", help="Do not download; only copy from existing local caches")
    ap.add_argument(
        "--preset",
        type=str,
        default=None,
        choices=["baseline_tier0", "text_models"],
        help="Convenience preset (enables a bundle of flags).",
    )
    ap.add_argument("--bootstrap-midas", action="store_true", help="Warm torch.hub MiDaS repo + checkpoints into TORCH_HOME")
    ap.add_argument("--bootstrap-raft", action="store_true", help="Warm torchvision RAFT weights into TORCH_HOME")
    ap.add_argument("--bootstrap-hf-clip", action="store_true", help="Download HF openai/clip-vit-base-patch32 into HF_HOME cache")
    ap.add_argument("--bootstrap-openai-clip", action="store_true", help="Ensure OpenAI CLIP ViT-B/32 weights exist in DP_MODELS_ROOT/clip_cache")
    ap.add_argument("--bootstrap-places365", action="store_true", help="Copy Places365 checkpoint from torch cache to DP_MODELS_ROOT/visual/places365")
    ap.add_argument("--bootstrap-text", action="store_true", help="Download text models into DP_MODELS_ROOT/text/ per dp_models specs")
    ap.add_argument("--bootstrap-clap", action="store_true", help="Download LAION CLAP checkpoint + required tokenizer cache into DP_MODELS_ROOT (for clap_extractor)")
    args = ap.parse_args()

    repo_root = _repo_root()
    models_root = os.path.abspath(args.models_root or _default_models_root(repo_root))
    _mkdir(models_root)

    # Pin caches into models_root (allow network in bootstrap if requested).
    from dp_models.offline import pin_cache_env  # type: ignore

    pin_cache_env(models_root, offline=False)

    allow_net = bool(args.allow_network) and not bool(args.from_cache_only)

    # Presets
    if args.preset == "baseline_tier0":
        args.bootstrap_midas = True
        args.bootstrap_raft = True
        args.bootstrap_openai_clip = True
        args.bootstrap_places365 = True
        args.bootstrap_clap = True
    if args.preset == "text_models":
        args.bootstrap_text = True

    # --- MiDaS (torch.hub) ---
    if args.bootstrap_midas:
        try:
            import torch  # type: ignore
        except Exception as e:
            raise RuntimeError(f"torch is required for MiDaS bootstrap: {e}") from e
        if not allow_net:
            # torch.hub will still use local repo if present; we just verify import path.
            pass
        # This will use cache if present; may download repo/weights if missing and allow_net=True.
        # We intentionally do not run inference; just load to populate caches.
        try:
            _ = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", pretrained=True, trust_repo=True, verbose=False)
        except Exception as e:
            if allow_net:
                raise
            raise RuntimeError(f"MiDaS bootstrap failed (cache-only). Enable --allow-network or prefill cache. Error: {e}") from e

    # --- RAFT (torchvision) ---
    if args.bootstrap_raft:
        try:
            import torchvision.models.optical_flow as models  # type: ignore
        except Exception as e:
            raise RuntimeError(f"torchvision is required for RAFT bootstrap: {e}") from e
        try:
            # Will download weights if missing (unless already cached).
            _ = models.raft_small(weights=models.Raft_Small_Weights.DEFAULT, progress=False)
        except Exception as e:
            if allow_net:
                raise
            raise RuntimeError(f"RAFT bootstrap failed (cache-only). Enable --allow-network or prefill cache. Error: {e}") from e

    # --- HF CLIP (Transformers) ---
    if args.bootstrap_hf_clip:
        try:
            from huggingface_hub import snapshot_download  # type: ignore
        except Exception as e:
            raise RuntimeError(f"huggingface_hub is required: {e}") from e
        if not allow_net:
            raise RuntimeError("--bootstrap-hf-clip requires network (or implement local-only HF cache copy)")
        # This will download into HF_HOME (pinned to models_root/hf_cache).
        snapshot_download(repo_id="openai/clip-vit-base-patch32", local_dir=None)

    # --- OpenAI CLIP (clip package) ---
    if args.bootstrap_openai_clip:
        clip_cache = os.path.join(models_root, "clip_cache")
        _mkdir(clip_cache)
        # Prefer copying from existing ~/.cache/clip first
        home = _env_home()
        copied = False
        copied |= _copy_if_exists(os.path.join(home, ".cache", "clip", "ViT-B-32.pt"), os.path.join(clip_cache, "ViT-B-32.pt"))
        copied |= _copy_if_exists(os.path.join(home, ".cache", "clip", "ViT-L-14.pt"), os.path.join(clip_cache, "ViT-L-14.pt"))
        if not copied:
            if not allow_net:
                raise RuntimeError("OpenAI CLIP weights not found in ~/.cache/clip and network is disabled")
            try:
                import clip  # type: ignore
                import torch  # type: ignore
            except Exception as e:
                raise RuntimeError(f"openai clip package is required to download weights: {e}") from e
            # Force download into clip_cache
            _m, _p = clip.load("ViT-B/32", device="cpu", download_root=clip_cache)
            del _m, _p
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    # --- Places365 ---
    if args.bootstrap_places365:
        # Our spec expects visual/places365/resnet50_places365.pth.tar
        src = os.path.join(models_root, "torch_cache", "hub", "checkpoints", "resnet50_places365.pth.tar")
        # If it's not in pinned cache yet, try user's default cache.
        if not os.path.exists(src):
            src = os.path.join(_env_home(), ".cache", "torch", "hub", "checkpoints", "resnet50_places365.pth.tar")
        dst = os.path.join(models_root, "visual", "places365", "resnet50_places365.pth.tar")
        if not _copy_if_exists(src, dst):
            raise RuntimeError(f"Places365 checkpoint not found at {src}. Run --bootstrap-places365 after caches are populated.")

        # categories file is already bundled in repo; keep it.
        repo_cats = os.path.join(repo_root, "dp_models", "bundled_models", "visual", "places365", "categories_places365.txt")
        dst_cats = os.path.join(models_root, "visual", "places365", "categories_places365.txt")
        if os.path.exists(repo_cats) and not os.path.exists(dst_cats):
            _copy_if_exists(repo_cats, dst_cats)

    # --- LAION CLAP (Audio baseline) ---
    if args.bootstrap_clap:
        if not allow_net:
            raise RuntimeError("--bootstrap-clap requires --allow-network (or provide artifacts manually)")
        # 1) Download CLAP checkpoint into DP_MODELS_ROOT/audio/laion_clap/clap_ckpt.pt
        clap_dir = os.path.join(models_root, "audio", "laion_clap")
        _mkdir(clap_dir)
        dst_ckpt = os.path.join(clap_dir, "clap_ckpt.pt")
        if not os.path.isfile(dst_ckpt):
            import urllib.request

            # laion_clap default (non-fusion) is 630k-audioset-best.pt (model_id=1)
            url = "https://huggingface.co/lukewys/laion_clap/resolve/main/630k-audioset-best.pt"
            tmp = dst_ckpt + ".tmp"
            print(f"[bootstrap] Downloading CLAP ckpt: {url}")
            urllib.request.urlretrieve(url, tmp)
            os.replace(tmp, dst_ckpt)
            print(f"[bootstrap] Saved CLAP ckpt: {dst_ckpt}")
        else:
            print(f"[bootstrap] CLAP ckpt already exists: {dst_ckpt}")

        # 2) Pre-warm HF tokenizer caches for laion_clap.
        # IMPORTANT: laion_clap imports tokenizers at import-time (training/data.py):
        #   - BertTokenizer.from_pretrained("bert-base-uncased")
        #   - RobertaTokenizer.from_pretrained("roberta-base")
        #   - BartTokenizer.from_pretrained("facebook/bart-base")
        # We only need tokenizer assets (not model weights), so we download a minimal subset.
        try:
            from huggingface_hub import snapshot_download  # type: ignore
        except Exception as e:
            raise RuntimeError(f"huggingface_hub is required for --bootstrap-clap: {e}") from e
        # Tokenizer-only patterns
        bert_patterns = [
            "vocab.txt",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "config.json",
        ]
        roberta_patterns = [
            "vocab.json",
            "merges.txt",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "config.json",
        ]
        bart_patterns = [
            "vocab.json",
            "merges.txt",
            "tokenizer.json",
            "tokenizer_config.json",
            "special_tokens_map.json",
            "config.json",
        ]

        print("[bootstrap] Downloading HF cache (tokenizers only): roberta-base")
        snapshot_download(repo_id="roberta-base", local_dir=None, repo_type="model", allow_patterns=roberta_patterns)
        print("[bootstrap] Downloading HF cache (tokenizers only): bert-base-uncased")
        snapshot_download(repo_id="bert-base-uncased", local_dir=None, repo_type="model", allow_patterns=bert_patterns)
        print("[bootstrap] Downloading HF cache (tokenizers only): facebook/bart-base")
        snapshot_download(repo_id="facebook/bart-base", local_dir=None, repo_type="model", allow_patterns=bart_patterns)

        # Best-effort cleanup of interrupted blobs (they can confuse later audits).
        try:
            hub_dir = os.path.join(models_root, "hf_cache", "hub")
            for m in ("models--roberta-base", "models--bert-base-uncased", "models--facebook--bart-base"):
                blobs_dir = os.path.join(hub_dir, m, "blobs")
                if os.path.isdir(blobs_dir):
                    for fn in os.listdir(blobs_dir):
                        if fn.endswith(".incomplete"):
                            try:
                                os.remove(os.path.join(blobs_dir, fn))
                            except Exception:
                                pass
        except Exception:
            pass

    # --- Text models (dp_models specs) ---
    if args.bootstrap_text:
        if not allow_net:
            raise RuntimeError("--bootstrap-text requires --allow-network (or implement cache-only HF copy)")
        try:
            from huggingface_hub import snapshot_download  # type: ignore
        except Exception as e:
            raise RuntimeError(f"huggingface_hub is required for --bootstrap-text: {e}") from e

        # 1) sentence-transformers/all-MiniLM-L6-v2
        dst_dir = os.path.join(models_root, "text", "sentence-transformers_all-MiniLM-L6-v2")
        _mkdir(dst_dir)
        snapshot_download(repo_id="sentence-transformers/all-MiniLM-L6-v2", local_dir=dst_dir, repo_type="model", local_dir_use_symlinks=False)

        # 2) intfloat/multilingual-e5-large
        dst_dir = os.path.join(models_root, "text", "intfloat_multilingual-e5-large")
        _mkdir(dst_dir)
        snapshot_download(repo_id="intfloat/multilingual-e5-large", local_dir=dst_dir, repo_type="model", local_dir_use_symlinks=False)

        # 3) shared tokenizer (best-effort): copy tokenizer.json from bert-base-uncased snapshot if present
        tok_dir = os.path.join(models_root, "text", "shared_tokenizer_v1")
        _mkdir(tok_dir)
        bert_dir = os.path.join(models_root, "text", "_bert-base-uncased_snapshot")
        _mkdir(bert_dir)
        snapshot_download(repo_id="bert-base-uncased", local_dir=bert_dir, repo_type="model", local_dir_use_symlinks=False)
        tok_json = None
        for cand in ("tokenizer.json", "tokenizer.model"):
            c = os.path.join(bert_dir, cand)
            if os.path.isfile(c):
                tok_json = c
                break
        if tok_json and tok_json.endswith("tokenizer.json"):
            _copy_if_exists(tok_json, os.path.join(tok_dir, "tokenizer.json"))
        else:
            # Leave a clear marker to avoid silent failures later.
            with open(os.path.join(tok_dir, "MISSING_TOKENIZER_JSON.txt"), "w", encoding="utf-8") as f:
                f.write("tokenizer.json was not found in bert-base-uncased snapshot. Please provide tokenizer.json for shared_tokenizer_v1.\n")

    print(f"[bootstrap] DP_MODELS_ROOT prepared at: {models_root}")
    print(f"[bootstrap] TORCH_HOME={os.environ.get('TORCH_HOME')}")
    print(f"[bootstrap] HF_HOME={os.environ.get('HF_HOME')}")


if __name__ == "__main__":
    main()


