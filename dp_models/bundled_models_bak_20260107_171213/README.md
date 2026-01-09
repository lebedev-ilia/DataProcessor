# dp_models/bundled_models

This directory contains **repo-bundled offline model artifacts** intended for **local development**.

## Usage

Set:

- `DP_MODELS_ROOT=/abs/path/to/TrendFlowML/DataProcessor/dp_models/bundled_models`

Model specs in `dp_models/spec_catalog/` reference artifacts **relative to** `DP_MODELS_ROOT`, for example:

- `visual/places365/categories_places365.txt`
- `visual/clip/openai_clip-vit-base-patch32/pytorch_model.bin`

Additional pinned caches (created by bootstrap scripts):
- `torch_cache/` (TORCH_HOME): torch.hub repos + torchvision checkpoints
- `hf_cache/` (HF_HOME): HuggingFace hub cache
- `clip_cache/` (DP_CLIP_WEIGHTS_DIR): OpenAI CLIP `.pt` weights (e.g. `ViT-B-32.pt`)

## Production note

In production, `DP_MODELS_ROOT` should typically point to a mounted read-only volume or another managed local path on the worker.


