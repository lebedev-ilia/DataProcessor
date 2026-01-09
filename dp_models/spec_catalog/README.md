# dp_models spec catalog

This directory contains **declarative model specs** used by `dp_models.ModelManager`.

Rules:
- **No-network**: specs must point to local artifacts under `${DP_MODELS_ROOT}`.
- **No-fallback**: if local artifacts are missing, ModelManager raises `weights_missing`.
- Add new models by dropping a YAML/JSON spec file here (and adding a provider if needed).

Bundled layout (expected under `${DP_MODELS_ROOT}`):
- `visual/places365/categories_places365.txt`
- `visual/places365/resnet50_places365.pth.tar` (and other checkpoints)

Supported in-process engines (today):
- `sentence-transformers`: local SentenceTransformer directory (HF id downloads are forbidden)
- `torchscript` / `torch-jit` / `jit`: local TorchScript file (`torch.jit.load`)
- `torch` / `torch-state-dict`: local checkpoint + python factory (`runtime_params.factory`) + `load_state_dict`


