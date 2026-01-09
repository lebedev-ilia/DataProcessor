#!/usr/bin/env python3
"""
Export OpenAI CLIP (image encoder + text encoder) to ONNX for Triton.

Policy (baseline GPU):
- fixed shapes (no dynamic axes)
- offline-friendly: use DP_MODELS_ROOT clip_cache and forbid network when --offline

Outputs:
- image encoder: float32 [1,3,S,S] -> float32 [1,D] (L2-normalized)
- text encoder (NO ArgMax): int64 [1,77] -> float32 [1,77,D] (per-token, L2-normalized)
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
import math

import torch  # type: ignore

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, os.pardir))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


class ClipImageEncoder(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.model.encode_image(x)
        y = y / (y.norm(dim=-1, keepdim=True) + 1e-9)
        return y


class ClipTextEncoder(torch.nn.Module):
    def __init__(self, model: torch.nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Export-friendly text encoder WITHOUT ArgMax.

        OpenAI CLIP encode_text() selects the EOT token via:
          x = x[torch.arange(x.shape[0]), tokens.argmax(dim=-1)] @ text_projection
        That introduces ArgMax which can be missing in some ORT CUDA EP builds.

        Here we produce per-token projected embeddings:
          y = ln_final(transformer(token_embedding + pos)) @ text_projection   => (B,77,D)
        Then caller (client) selects the EOT position using argmax(tokens) OUTSIDE the model.
        """
        # Ensure expected shape [B,77]
        x = self.model.token_embedding(tokens).type(self.model.dtype)  # (B,77,C)
        x = x + self.model.positional_embedding.type(self.model.dtype)
        x = x.permute(1, 0, 2)  # (77,B,C)
        x = self.model.transformer(x)
        x = x.permute(1, 0, 2)  # (B,77,C)
        x = self.model.ln_final(x)
        # Project to embedding space (D=512 for ViT-B/32).
        x = x @ self.model.text_projection
        x = x.to(torch.float32)
        # L2-normalize per token vector.
        x = x / (x.norm(dim=-1, keepdim=True) + 1e-9)
        return x


def _resize_vit_positional_embedding_for_size(model: torch.nn.Module, image_size: int) -> None:
    """
    Patch OpenAI CLIP ViT positional embeddings to match a new input resolution.

    OpenAI CLIP ViT models are trained with a fixed 2D positional embedding grid for the visual transformer.
    For fixed-shape Triton branches we export separate ONNX models per input size, so we can interpolate the
    positional grid once and then export a fully static graph.
    """
    visual = getattr(model, "visual", None)
    if visual is None:
        raise RuntimeError("CLIP model has no .visual; cannot resize positional embeddings")

    pos = getattr(visual, "positional_embedding", None)
    conv1 = getattr(visual, "conv1", None)
    if pos is None or conv1 is None:
        raise RuntimeError("CLIP visual has no positional_embedding/conv1; expected ViT visual encoder")

    # Patch size inferred from patchify conv.
    patch = int(getattr(conv1, "kernel_size")[0])
    if patch <= 0:
        raise RuntimeError(f"Invalid patch size from visual.conv1: {patch}")

    # Triton branches use square input. Conv stride = patch, no padding.
    new_grid = int(image_size // patch)
    if new_grid <= 0:
        raise RuntimeError(f"Invalid new_grid for image_size={image_size}, patch={patch}")

    pos_t = pos.detach().to(torch.float32)  # (1+old_grid^2, width)
    if pos_t.ndim != 2 or pos_t.shape[0] < 2:
        raise RuntimeError(f"Unexpected positional_embedding shape: {tuple(pos_t.shape)}")

    old_tokens = int(pos_t.shape[0])
    old_grid_f = math.sqrt(float(old_tokens - 1))
    old_grid = int(old_grid_f)
    if old_grid * old_grid != (old_tokens - 1):
        raise RuntimeError(
            f"positional_embedding tokens do not form a square grid: tokens={old_tokens}, grid~={old_grid_f}"
        )

    if old_grid == new_grid:
        # Still update input_resolution for correctness.
        if hasattr(visual, "input_resolution"):
            visual.input_resolution = int(image_size)
        return

    cls = pos_t[:1, :]  # (1,width)
    grid = pos_t[1:, :].reshape(old_grid, old_grid, -1).permute(2, 0, 1).unsqueeze(0)  # (1,width,gh,gw)

    grid_resized = torch.nn.functional.interpolate(
        grid,
        size=(new_grid, new_grid),
        mode="bicubic",
        align_corners=False,
    )
    grid_resized = grid_resized.squeeze(0).permute(1, 2, 0).reshape(new_grid * new_grid, -1)  # (new^2,width)
    pos_new = torch.cat([cls, grid_resized], dim=0).to(pos.dtype)

    visual.positional_embedding = torch.nn.Parameter(pos_new)
    if hasattr(visual, "input_resolution"):
        visual.input_resolution = int(image_size)


def main() -> None:
    ap = argparse.ArgumentParser("Export OpenAI CLIP to ONNX (fixed shapes)")
    ap.add_argument("--model", default="ViT-B/32", choices=["ViT-B/32", "ViT-L/14"])
    ap.add_argument("--image-size", type=int, default=224, help="Square input size for image encoder export")
    ap.add_argument("--out-dir", required=True, help="Output directory (files will be placed inside)")
    ap.add_argument("--opset", type=int, default=18)
    ap.add_argument("--models-root", type=str, default=None, help="DP_MODELS_ROOT (pins TORCH_HOME/HF_HOME + DP_CLIP_WEIGHTS_DIR)")
    ap.add_argument("--offline", action="store_true", help="Strict no-network mode (requires cached weights)")
    ap.add_argument("--skip-text", action="store_true", help="Skip exporting text encoder (useful when exporting multiple image sizes)")
    args = ap.parse_args()

    out_dir = os.path.abspath(str(args.out_dir))
    os.makedirs(out_dir, exist_ok=True)

    if args.models_root:
        from dp_models.offline import pin_cache_env  # type: ignore

        pin_cache_env(str(args.models_root), offline=bool(args.offline))

    # Strict offline: block socket.connect.
    net_guard_ctx = None
    if args.offline:
        from dp_models.offline import network_guard  # type: ignore

        net_guard_ctx = network_guard(enabled=True)

    import clip  # type: ignore

    download_root = os.environ.get("DP_CLIP_WEIGHTS_DIR") or None

    if net_guard_ctx is not None:
        with net_guard_ctx:
            model, _ = clip.load(str(args.model), device="cpu", jit=False, download_root=download_root)
    else:
        model, _ = clip.load(str(args.model), device="cpu", jit=False, download_root=download_root)

    model.eval()

    # --- image encoder ---
    image_size = int(args.image_size)
    if image_size <= 0:
        raise SystemExit(f"--image-size must be > 0; got {image_size}")

    # Patch positional embeddings for the requested fixed image size (static branch).
    _resize_vit_positional_embedding_for_size(model, image_size=image_size)

    img_path = os.path.join(
        out_dir, f"openai_clip_{str(args.model).replace('/', '_')}_image_{image_size}.onnx"
    )
    img = ClipImageEncoder(model)
    dummy_img = torch.randn(1, 3, image_size, image_size, dtype=torch.float32)
    torch.onnx.export(
        img,
        dummy_img,
        img_path,
        export_params=True,
        opset_version=int(args.opset),
        do_constant_folding=True,
        input_names=["input"],
        output_names=["emb"],
        dynamic_axes=None,
    )

    # --- text encoder ---
    txt_path = os.path.join(out_dir, f"openai_clip_{str(args.model).replace('/', '_')}_text_77.onnx")
    if not bool(args.skip_text):
        txt = ClipTextEncoder(model)
        dummy_tok = torch.zeros((1, 77), dtype=torch.int64)
        torch.onnx.export(
            txt,
            dummy_tok,
            txt_path,
            export_params=True,
            opset_version=int(args.opset),
            do_constant_folding=True,
            input_names=["tokens"],
            output_names=["emb_seq"],
            dynamic_axes=None,
        )

    print(f"Exported: {img_path}")
    print(f"sha256: {sha256_file(img_path)}")
    if not bool(args.skip_text):
        print(f"Exported: {txt_path}")
        print(f"sha256: {sha256_file(txt_path)}")


if __name__ == "__main__":
    main()


