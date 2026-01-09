import argparse
import os
import sys
from datetime import datetime
from typing import List, Tuple, Optional

import numpy as np      # type: ignore
import torch            # type: ignore
from PIL import Image   # type: ignore

_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if _path not in sys.path:
    sys.path.append(_path)
_repo_root = os.path.dirname(_path)
if _repo_root not in sys.path:
    sys.path.append(_repo_root)

from utils.frame_manager import FrameManager
from utils.logger import get_logger
from utils.resource_probe import pick_device
from utils.utilites import load_metadata
from utils.meta_builder import apply_models_meta, model_used


NAME = "core_clip"
VERSION = "2.0"
SCHEMA_VERSION = "core_clip_npz_v1"
LOGGER = get_logger(NAME)

SHOT_QUALITY_PROMPTS: List[str] = [
    "cinematic shot, high-quality professional footage",
    "professional low-light cinematic footage",
    "good smartphone video quality",
    "poor smartphone video quality, grainy, noisy",
    "webcam low resolution footage",
    "screen recording of display",
    "cctv surveillance camera footage low quality",
]

# Prompt sets used by downstream modules (scene_classification).
# These are kept here so downstream modules can be strictly "core_clip-only" and offline.
SCENE_AESTHETIC_PROMPTS: List[str] = [
    "aesthetic beautiful scene",
    "professional photography",
    "ugly unappealing scene",
    "amateur photography",
]

SCENE_LUXURY_PROMPTS: List[str] = [
    "luxury expensive high-end scene",
    "premium elegant sophisticated",
    "cheap low-quality scene",
    "budget affordable scene",
]

SCENE_ATMOSPHERE_PROMPTS: List[str] = [
    "cozy warm comfortable scene",
    "scary frightening dark scene",
    "epic grand majestic scene",
    "neutral ordinary scene",
]

# Prompt set used by `modules/cut_detection` (stylized transition classification).
# IMPORTANT: these prompts are embedded via CLIP text encoder here (core provider),
# so downstream modules do NOT load CLIP weights (single source-of-truth + no-network).
CUT_DETECTION_TRANSITION_PROMPTS: List[str] = [
    "hard cut",
    "fade",
    "dissolve",
    "whip pan",
    "zoom transition",
    "wipe transition",
    "slide transition",
    "glitch transition",
    "flash transition",
    "luma wipe transition",
]

_CLIP_IMG_SIZE = 224
_CLIP_MEAN = np.asarray([0.48145466, 0.4578275, 0.40821073], dtype=np.float32)
_CLIP_STD = np.asarray([0.26862954, 0.26130258, 0.27577711], dtype=np.float32)


def _load_triton_spec_via_model_manager(model_spec_name: str) -> dict:
    """
    Resolve Triton model spec via dp_models.ModelManager (no-network, reproducible).
    Returns dict with keys:
      - client: TritonHttpClient
      - rp: runtime_params
      - models_used_entry: dict (model_used)
    """
    from dp_models import get_global_model_manager  # type: ignore

    mm = get_global_model_manager()
    rm = mm.get(model_name=str(model_spec_name))
    rp = rm.spec.runtime_params or {}
    handle = rm.handle or {}
    client = None
    if isinstance(handle, dict):
        client = handle.get("client")
    if client is None:
        raise RuntimeError(f"{NAME} | ModelManager returned empty Triton client handle for: {model_spec_name}")
    if not isinstance(rp, dict) or not rp:
        raise RuntimeError(f"{NAME} | ModelManager returned empty runtime_params for: {model_spec_name}")
    return {"client": client, "rp": rp, "models_used_entry": rm.models_used_entry}

def _clip_preprocess_batch(frames_rgb_uint8: List[np.ndarray], *, image_size: int) -> np.ndarray:
    """
    Preprocess RGB uint8 frames to CLIP float32 tensor: (B,3,224,224).
    This intentionally avoids loading model weights (needed for Triton runtime).
    """
    s = int(image_size)
    if s <= 0:
        raise ValueError(f"{NAME} | invalid image_size={image_size}")
    out: List[np.ndarray] = []
    for fr in frames_rgb_uint8:
        img = Image.fromarray(fr)
        img = img.resize((s, s), resample=Image.BICUBIC)
        arr = np.asarray(img, dtype=np.float32) / 255.0  # (H,W,3) RGB
        arr = (arr - _CLIP_MEAN) / (_CLIP_STD + 1e-12)
        arr = np.transpose(arr, (2, 0, 1))  # (3,H,W)
        out.append(arr.astype(np.float32))
    return np.stack(out, axis=0) if out else np.zeros((0, 3, s, s), dtype=np.float32)


def _clip_resize_batch_uint8(frames_rgb_uint8: List[np.ndarray], *, image_size: int) -> np.ndarray:
    """
    Resize RGB uint8 frames to fixed square size and keep UINT8 NHWC.
    Used for baseline GPU contract where Triton ensemble owns normalize/layout conversion.
    """
    s = int(image_size)
    if s <= 0:
        raise ValueError(f"{NAME} | invalid image_size={image_size}")
    out: List[np.ndarray] = []
    for fr in frames_rgb_uint8:
        img = Image.fromarray(fr)
        img = img.resize((s, s), resample=Image.BICUBIC)
        out.append(np.asarray(img, dtype=np.uint8))
    return np.stack(out, axis=0) if out else np.zeros((0, s, s, 3), dtype=np.uint8)

def _triton_infer_embeddings(
    *,
    client,
    model_name: str,
    model_version: Optional[str],
    input_name: str,
    input_tensor: np.ndarray,
    output_name: str,
    datatype: str,
) -> np.ndarray:
    res = client.infer(
        model_name=model_name,
        model_version=model_version,
        input_name=input_name,
        input_tensor=input_tensor,
        output_name=output_name,
        datatype=datatype,
    )
    out = np.asarray(res.output, dtype=np.float32)
    # L2 normalize (standard CLIP practice)
    norms = np.linalg.norm(out, axis=-1, keepdims=True) + 1e-9
    return out / norms


def _require_frame_indices(meta: dict, name: str) -> List[int]:
    """
    Strict contract: frame sampling is owned by Segmenter/DataProcessor.
    Providers MUST use metadata[name].frame_indices and MUST NOT fallback.
    """
    block = meta.get(name)
    if not isinstance(block, dict) or "frame_indices" not in block:
        raise RuntimeError(
            f"{name} | metadata missing '{name}.frame_indices'. "
            "Segmenter must provide per-provider frame_indices. No fallback is allowed."
        )
    frame_indices = block.get("frame_indices")
    if not isinstance(frame_indices, list) or not frame_indices:
        raise RuntimeError(f"{name} | metadata '{name}.frame_indices' is empty/invalid.")
    return [int(x) for x in frame_indices]


def compute_text_embeddings(
    model,
    device: str,
    prompts: List[str],
) -> np.ndarray:
    import clip  # type: ignore

    text = clip.tokenize(prompts).to(device)
    with torch.no_grad():
        emb = model.encode_text(text)
        emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-9)
    return emb.detach().cpu().numpy().astype(np.float32)


def init_clip(model_name: str, preferred_device: str = "auto") -> Tuple[torch.nn.Module, callable, str]:
    import clip # type: ignore

    device = pick_device(preferred_device)
    # Optional: pin OpenAI CLIP weights cache root (offline-friendly).
    dl_root = os.environ.get("DP_CLIP_WEIGHTS_DIR")
    if isinstance(dl_root, str) and dl_root.strip():
        model, preprocess = clip.load(model_name, device=device, download_root=str(dl_root).strip())
    else:
        model, preprocess = clip.load(model_name, device=device)

    model.eval()

    LOGGER.info(
        f"{NAME} | CLIP initialized | model: {model_name} | device: {device}"
    )

    return model, preprocess, device


def compute_clip_embeddings(
    frame_manager: FrameManager,
    frame_indices: List[int],
    model_name: str,
    batch_size: int,
) -> np.ndarray:

    if not frame_indices:
        LOGGER.warning(f"{NAME} | No frame indices provided")
        return np.zeros((0, 0), dtype=np.float32)

    model, preprocess, device = init_clip(model_name)

    n_frames = len(frame_indices)

    embeddings_out = None
    embed_dim = None

    try:
        with torch.no_grad():
            for start in range(0, n_frames, batch_size):
                batch_ids = frame_indices[start : start + batch_size]

                images = []
                for idx in batch_ids:
                    frame = frame_manager.get(idx)
                    img = Image.fromarray(frame)
                    images.append(preprocess(img))

                batch_tensor = torch.stack(images).to(device)
                
                emb = model.encode_image(batch_tensor)

                # L2 normalization (standard CLIP practice)
                emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-9)

                emb_np = emb.cpu().numpy().astype(np.float32)

                if embeddings_out is None:
                    embed_dim = emb_np.shape[1]
                    embeddings_out = np.zeros((n_frames, embed_dim), dtype=np.float32)

                embeddings_out[start : start + len(batch_ids)] = emb_np

                if start % (batch_size * 10) == 0:
                    LOGGER.info(
                        f"{NAME} | processed {start + len(batch_ids)}/{n_frames}"
                    )
    finally:
        del model
        torch.cuda.empty_cache()

    return embeddings_out


def main():
    parser = argparse.ArgumentParser(description="Production CLIP per-frame embedding extractor")
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--rs-path", required=True)
    parser.add_argument("--model-name", default="ViT-B/32")
    parser.add_argument("--model-version", default="unknown")
    parser.add_argument("--weights-digest", default="unknown")
    parser.add_argument("--engine", default="torch")
    parser.add_argument("--precision", default="fp32")
    parser.add_argument("--runtime", default="inprocess", choices=["inprocess", "triton"])
    # Triton (HTTP v2) options (used when --runtime=triton)
    parser.add_argument("--triton-http-url", default=None)
    # Prefer ModelManager specs for Triton (recommended; overrides explicit triton_* args when provided).
    parser.add_argument("--triton-image-model-spec", default=None, help="dp_models spec name (e.g., clip_image_triton)")
    parser.add_argument("--triton-text-model-spec", default=None, help="dp_models spec name (e.g., clip_text_triton)")
    # image embeddings
    parser.add_argument("--triton-image-model-name", default=None)
    parser.add_argument("--triton-image-model-version", default=None)
    parser.add_argument("--triton-image-input-name", default="INPUT__0")
    parser.add_argument("--triton-image-output-name", default="OUTPUT__0")
    parser.add_argument("--triton-image-datatype", default="FP32")
    # text embeddings (required by shot_quality_* contract)
    parser.add_argument("--triton-text-model-name", default=None)
    parser.add_argument("--triton-text-model-version", default=None)
    parser.add_argument("--triton-text-input-name", default="INPUT__0")
    parser.add_argument("--triton-text-output-name", default="OUTPUT__0")
    parser.add_argument("--triton-text-datatype", default="INT64")
    parser.add_argument(
        "--triton-preprocess-preset",
        type=str,
        default="openai_clip_224",
        choices=["openai_clip_224", "openai_clip_336", "openai_clip_448"],
        help="Image preprocess preset used for Triton runtime (controls resize).",
    )
    parser.add_argument("--batch-size", type=int, required=True, help="Batch size (must be provided by scheduler/orchestrator)")
    args = parser.parse_args()

    meta_path = os.path.join(args.frames_dir, "metadata.json")
    meta = load_metadata(meta_path, NAME)

    total_frames = int(meta["total_frames"])

    frame_indices = _require_frame_indices(meta, NAME)
    if len(frame_indices) <= 0:
        # Contract: empty is not allowed for core_clip in baseline.
        raise RuntimeError(f"{NAME} | empty frame_indices is not allowed (no-fallback)")
    LOGGER.info(f"{NAME} | sampled frames: {len(frame_indices)} / total={total_frames}")

    frame_manager = FrameManager(
        frames_dir=args.frames_dir,
        chunk_size=meta.get("chunk_size", 32),
        cache_size=meta.get("cache_size", 2),
    )

    runtime = str(args.runtime or "inprocess").strip().lower()
    if runtime not in ("inprocess", "triton"):
        raise RuntimeError(f"{NAME} | invalid runtime: {runtime}")

    # In triton runtime we MUST compute both image and text embeddings via Triton (no local model inference).
    model = None
    preprocess = None
    device = "cpu"
    if runtime == "inprocess":
        model, preprocess, device = init_clip(args.model_name, preferred_device="auto")
    batch_size = int(args.batch_size)
    if batch_size <= 0:
        raise RuntimeError(f"{NAME} | --batch-size must be > 0 (scheduler-controlled); got {batch_size}")
    try:
        client = None
        if runtime == "triton":
            img_mm = None
            txt_mm = None
            # Recommended path: resolve Triton params via ModelManager specs.
            if args.triton_image_model_spec and args.triton_text_model_spec:
                img_mm = _load_triton_spec_via_model_manager(str(args.triton_image_model_spec))
                txt_mm = _load_triton_spec_via_model_manager(str(args.triton_text_model_spec))
                client = img_mm["client"]
            else:
                # Legacy path: explicit Triton args (kept for backward compatibility).
                if not args.triton_http_url:
                    raise RuntimeError(f"{NAME} | runtime=triton requires --triton-http-url (no-fallback)")

                from dp_triton import TritonHttpClient, TritonError  # local import (repo code)

                client = TritonHttpClient(base_url=str(args.triton_http_url), timeout_sec=5.0)
                if not client.ready():
                    raise TritonError(
                        f"{NAME} | Triton is not ready at {args.triton_http_url}",
                        error_code="triton_unavailable",
                    )

                if not args.triton_image_model_name:
                    raise RuntimeError(f"{NAME} | runtime=triton requires --triton-image-model-name")
                if not args.triton_text_model_name:
                    raise RuntimeError(f"{NAME} | runtime=triton requires --triton-text-model-name")

        # --- image embeddings ---
        n_frames = len(frame_indices)
        embeddings_out = None
        embed_dim = None
        with torch.no_grad():
            for start in range(0, n_frames, batch_size):
                # Fixed-shape Triton branches are exported with batch=1 (no dynamic axes).
                # If model input is UINT8 (baseline branch), enforce batch_size=1 on client.
                effective_bs = batch_size
                triton_image_dt = None
                if runtime == "triton":
                    # We'll resolve datatype below; here we only need a hint to force bs=1.
                    if "img_mm" in locals() and img_mm is not None:
                        triton_image_dt = str((img_mm["rp"] or {}).get("triton_input_datatype") or "")
                    else:
                        triton_image_dt = str(args.triton_image_datatype or "")
                    if str(triton_image_dt).strip().upper() == "UINT8":
                        effective_bs = 1

                batch_ids = frame_indices[start : start + effective_bs]
                images: List[np.ndarray] = []
                for idx in batch_ids:
                    frame = frame_manager.get(idx)
                    images.append(frame)

                if runtime == "triton":
                    assert client is not None
                    preset = str(args.triton_preprocess_preset or "openai_clip_224").strip().lower()
                    if preset == "openai_clip_224":
                        image_size = 224
                    elif preset == "openai_clip_336":
                        image_size = 336
                    elif preset == "openai_clip_448":
                        image_size = 448
                    else:
                        raise RuntimeError(f"{NAME} | unknown triton_preprocess_preset: {preset}")

                    if "img_mm" in locals() and img_mm is not None:
                        rp = img_mm["rp"]
                        triton_model_name = str(rp.get("triton_model_name"))
                        triton_model_version = str(rp.get("triton_model_version") or "") or None
                        triton_input_name = str(rp.get("triton_input_name"))
                        triton_output_name = str(rp.get("triton_output_name"))
                        triton_datatype = str(rp.get("triton_input_datatype") or "FP32")
                    else:
                        triton_model_name = str(args.triton_image_model_name)
                        triton_model_version = str(args.triton_image_model_version) if args.triton_image_model_version else None
                        triton_input_name = str(args.triton_image_input_name)
                        triton_output_name = str(args.triton_image_output_name)
                        triton_datatype = str(args.triton_image_datatype)
                    dt = str(triton_datatype or "FP32").strip().upper()
                    if dt == "UINT8":
                        inp = _clip_resize_batch_uint8(images, image_size=image_size)  # (B,S,S,3) uint8
                    else:
                        inp = _clip_preprocess_batch(images, image_size=image_size)  # (B,3,S,S) float32
                    try:
                        emb_np = _triton_infer_embeddings(
                            client=client,
                            model_name=triton_model_name,
                            model_version=triton_model_version,
                            input_name=triton_input_name,
                            input_tensor=inp,
                            output_name=triton_output_name,
                            datatype=triton_datatype,
                        )
                    except Exception as e:
                        raise RuntimeError(f"{NAME} | Triton infer failed: {e}") from e
                else:
                    assert model is not None and preprocess is not None
                    imgs = [preprocess(Image.fromarray(fr)) for fr in images]
                    batch_tensor = torch.stack(imgs).to(device)
                    emb = model.encode_image(batch_tensor)
                    emb = emb / (emb.norm(dim=-1, keepdim=True) + 1e-9)
                    emb_np = emb.detach().cpu().numpy().astype(np.float32)

                if embeddings_out is None:
                    embed_dim = int(emb_np.shape[1])
                    embeddings_out = np.zeros((n_frames, embed_dim), dtype=np.float32)
                embeddings_out[start : start + len(batch_ids)] = emb_np
                if start % (batch_size * 10) == 0:
                    LOGGER.info(f"{NAME} | processed {start + len(batch_ids)}/{n_frames}")

        embeddings = embeddings_out if embeddings_out is not None else np.zeros((0, 0), dtype=np.float32)

        # --- text embeddings for downstream modules ---
        # Compute all prompt embeddings in one batch, then split.
        all_prompts: List[str] = []
        all_prompts.extend(SHOT_QUALITY_PROMPTS)
        all_prompts.extend(SCENE_AESTHETIC_PROMPTS)
        all_prompts.extend(SCENE_LUXURY_PROMPTS)
        all_prompts.extend(SCENE_ATMOSPHERE_PROMPTS)
        all_prompts.extend(CUT_DETECTION_TRANSITION_PROMPTS)

        n_shot = len(SHOT_QUALITY_PROMPTS)
        n_aes = len(SCENE_AESTHETIC_PROMPTS)
        n_lux = len(SCENE_LUXURY_PROMPTS)
        n_atm = len(SCENE_ATMOSPHERE_PROMPTS)
        n_cut = len(CUT_DETECTION_TRANSITION_PROMPTS)
        sl_shot = slice(0, n_shot)
        sl_aes = slice(n_shot, n_shot + n_aes)
        sl_lux = slice(n_shot + n_aes, n_shot + n_aes + n_lux)
        sl_atm = slice(n_shot + n_aes + n_lux, n_shot + n_aes + n_lux + n_atm)
        sl_cut = slice(n_shot + n_aes + n_lux + n_atm, n_shot + n_aes + n_lux + n_atm + n_cut)

        if runtime == "triton":
            assert client is not None
            import clip  # type: ignore

            toks = clip.tokenize(all_prompts)  # (P,77) int64
            toks_np = toks.detach().cpu().numpy().astype(np.int64)
            if "txt_mm" in locals() and txt_mm is not None:
                rp = txt_mm["rp"]
                triton_model_name = str(rp.get("triton_model_name"))
                triton_model_version = str(rp.get("triton_model_version") or "") or None
                triton_input_name = str(rp.get("triton_input_name"))
                triton_output_name = str(rp.get("triton_output_name"))
                triton_datatype = str(rp.get("triton_input_datatype") or "INT64")
            else:
                triton_model_name = str(args.triton_text_model_name)
                triton_model_version = str(args.triton_text_model_version) if args.triton_text_model_version else None
                triton_input_name = str(args.triton_text_input_name)
                triton_output_name = str(args.triton_text_output_name)
                triton_datatype = str(args.triton_text_datatype)
            # Fixed-shape baseline: clip_text Triton model expects shape [1,77] (no dynamic axes).
            # New ONNX export avoids ArgMax inside the model and returns per-token embeddings:
            #   output shape: (1,77,512)
            # We select EOT embedding OUTSIDE the model using argmax(tokens).
            outs: List[np.ndarray] = []
            for i in range(int(toks_np.shape[0])):
                ti = toks_np[i : i + 1, :]
                try:
                    seq = _triton_infer_embeddings(
                        client=client,
                        model_name=triton_model_name,
                        model_version=triton_model_version,
                        input_name=triton_input_name,
                        input_tensor=ti,
                        output_name=triton_output_name,
                        datatype=triton_datatype,
                    )
                except Exception as e:
                    raise RuntimeError(f"{NAME} | Triton text infer failed (i={i}): {e}") from e

                arr = np.asarray(seq, dtype=np.float32)
                if arr.ndim == 2 and arr.shape[-1] == 512:
                    # Backward-compatible: old clip_text returned (1,512)
                    emb = arr[0]
                else:
                    if arr.ndim != 3 or arr.shape[0] != 1 or arr.shape[1] != 77:
                        raise RuntimeError(f"{NAME} | clip_text output has invalid shape: {arr.shape}")
                    eot_pos = int(np.argmax(ti[0]))
                    eot_pos = max(0, min(eot_pos, 76))
                    emb = arr[0, eot_pos, :]  # (512,)
                outs.append(emb.reshape(1, -1))
            all_text_embeddings = np.concatenate(outs, axis=0) if outs else np.zeros((0, 0), dtype=np.float32)
        else:
            assert model is not None
            all_text_embeddings = compute_text_embeddings(
                model=model,
                device=device,
                prompts=all_prompts,
            )

        shot_quality_text_embeddings = np.asarray(all_text_embeddings[sl_shot], dtype=np.float32)
        scene_aesthetic_text_embeddings = np.asarray(all_text_embeddings[sl_aes], dtype=np.float32)
        scene_luxury_text_embeddings = np.asarray(all_text_embeddings[sl_lux], dtype=np.float32)
        scene_atmosphere_text_embeddings = np.asarray(all_text_embeddings[sl_atm], dtype=np.float32)
        cut_detection_transition_text_embeddings = np.asarray(all_text_embeddings[sl_cut], dtype=np.float32)

    finally:
        if model is not None:
            del model
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    LOGGER.info(
        f"{NAME} | embeddings computed | shape: {embeddings.shape}"
    )

    frame_manager.close()

    out_dir = os.path.join(args.rs_path, NAME)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "embeddings.npz")

    created_at = datetime.utcnow().isoformat()
    runtime_meta = "triton-gpu" if runtime == "triton" else "inprocess"
    # Device semantics:
    # - inprocess: actual compute device ("cuda" or "cpu")
    # - triton: we assume GPU-backed triton by default in this project ("cuda")
    device_meta = "cuda" if runtime == "triton" else str(device)
    meta_out = {
        "producer": NAME,
        "producer_version": VERSION,
        "schema_version": SCHEMA_VERSION,
        "created_at": created_at,
        "status": "ok",
        "empty_reason": None,
        "model_name": args.model_name,
        "total_frames": int(total_frames),
        "batch_size": int(batch_size),
        "runtime": runtime_meta,
        "device": device_meta,
    }
    required_run_keys = ["platform_id", "video_id", "run_id", "sampling_policy_version", "config_hash"]
    missing = [k for k in required_run_keys if not meta.get(k)]
    if missing:
        raise RuntimeError(f"{NAME} | frames metadata missing required run identity keys: {missing}")
    for k in required_run_keys:
            meta_out[k] = meta.get(k)
    # Optional but recommended for reproducibility
    if meta.get("dataprocessor_version") is not None:
        meta_out["dataprocessor_version"] = meta.get("dataprocessor_version")

    # PR-3: model system baseline (core_clip may use both image+text encoders on Triton).
    models_used_list = []
    if runtime == "triton" and "img_mm" in locals() and "txt_mm" in locals() and img_mm is not None and txt_mm is not None:
        models_used_list.append(img_mm["models_used_entry"])
        models_used_list.append(txt_mm["models_used_entry"])
    else:
        models_used_list.append(
            model_used(
                model_name=str(args.model_name),
                model_version=str(args.model_version or "unknown"),
                weights_digest=str(args.weights_digest or "unknown"),
                runtime=runtime_meta,
                engine=str(args.engine or "unknown"),
                precision=str(args.precision or "unknown"),
                device=device_meta,
            )
        )
    meta_out = apply_models_meta(meta_out, models_used=models_used_list)

    np.savez_compressed(
        out_path,
        # legacy fields (kept)
        version=VERSION,
        created_at=created_at,
        model_name=args.model_name,
        total_frames=total_frames,
        frame_indices=np.array(frame_indices, dtype=np.int32),
        frame_embeddings=embeddings,
        # downstream contract for shot_quality
        shot_quality_prompts=np.array(SHOT_QUALITY_PROMPTS, dtype=object),
        shot_quality_text_embeddings=shot_quality_text_embeddings,
        # downstream contract for scene_classification semantics (zero-shot prompts)
        scene_aesthetic_prompts=np.array(SCENE_AESTHETIC_PROMPTS, dtype=object),
        scene_aesthetic_text_embeddings=scene_aesthetic_text_embeddings,
        scene_luxury_prompts=np.array(SCENE_LUXURY_PROMPTS, dtype=object),
        scene_luxury_text_embeddings=scene_luxury_text_embeddings,
        scene_atmosphere_prompts=np.array(SCENE_ATMOSPHERE_PROMPTS, dtype=object),
        scene_atmosphere_text_embeddings=scene_atmosphere_text_embeddings,
        # downstream contract for cut_detection (stylized transitions, CLIP zero-shot)
        cut_detection_transition_prompts=np.array(CUT_DETECTION_TRANSITION_PROMPTS, dtype=object),
        cut_detection_transition_text_embeddings=cut_detection_transition_text_embeddings,
        # canonical meta (required by artifact_validator)
        meta=np.asarray(meta_out, dtype=object),
    )

    LOGGER.info(f"{NAME} | Saved result: {out_path}")


if __name__ == "__main__":
    main()