# `scene_classification`

Scene segmentation + classification on **Places365** with CLIP-based semantics computed strictly from `core_clip`.

## Inputs

- **Primary**: frames from `frames_dir` via `FrameManager` (union domain indices).
- **Time-axis**: `metadata.json` must include `union_timestamps_sec` (source-of-truth).
- **Dependencies** (fail-fast):
  - `core_clip` must exist in `rs_path/core_clip/embeddings.npz` and provide:
    - `frame_indices`, `frame_embeddings`
    - `scene_aesthetic_text_embeddings`, `scene_luxury_text_embeddings`, `scene_atmosphere_text_embeddings`

## Models (ModelManager)

Places365 model is loaded **only via** `dp_models` (no URLs, no `pretrained=True`).

Required env:
- `DP_MODELS_ROOT=/abs/path/to/local/models`
  - For local dev in this repo, you can point it to `DataProcessor/models`.

Supported `model_arch`:
- **Places365 ResNet**: `resnet18`, `resnet50` → specs `places365_resnet18`, `places365_resnet50`
- **timm backbones** (when `use_timm=true`): `efficientnet_*`, `convnext_*`, `vit_*`, `regnetx_*`, `resnet50`, `resnet101`
  - resolved as specs `places365_timm_<arch>`

## Outputs (NPZ)

Saved to: `rs_path/scene_classification/scene_classification_features_*.npz`

Canonical keys:
- **`scenes`**: dict mapping `scene_id -> scene_dict` where `scene_id` is `s0000`, `s0001`, ...
  - `scene_dict` includes:
    - `scene_label` (Places365 label)
    - `indices` (list of union frame indices in this scene)
    - `start_frame`, `end_frame`, `length_frames`, `length_seconds`
    - Places365 aggregates: `mean_score`, `class_entropy_mean`, `top1_prob_mean`, `top1_vs_top2_gap_mean`, `fraction_high_confidence_frames`
    - Ontology aggregates: `mean_indoor`, `mean_outdoor`, `mean_nature`, `mean_urban`
    - CLIP semantics (from `core_clip`): `mean_aesthetic_score`, `aesthetic_std`, `aesthetic_frac_high`, `mean_luxury_score`,
      `mean_cozy`, `mean_scary`, `mean_epic`, `mean_neutral`, `atmosphere_entropy`
    - Stability: `scene_change_score`, `label_stability`
    - `dominant_places_topk_ids`, `dominant_places_topk_probs`
- Flat arrays (`scene_ids`, `scene_label`, `start_frame`, …) duplicated for NPZ-friendly tabular access.
- `meta` includes `models_used[]` and `model_signature` (Places365 + upstream `core_clip`).

## Semantics / scene count rules

- **One scene is valid** (e.g. if all processed frames have the same dominant Places label).
- `frame_indices` **must be >= 2**, иначе `error` (no-fallback).
- Scenes are formed by consecutive frames with the same predicted label, then filtered by `min_scene_seconds` (or `min_scene_length / fps`).

## Downstream usage

- `color_light` consumes `scene_classification.scenes` and treats each `scene_id` as a unique segment (label collisions are handled).


