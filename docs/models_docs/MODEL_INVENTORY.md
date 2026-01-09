# Model Inventory (current codebase)

This document is a **repo audit snapshot** of where models are loaded/used today, and what needs to be standardized by the unified **ModelManager**.

Key policies (see `docs/MODEL_SYSTEM_RULES.md`, `MODEL_MANAGER_Q.md`):
- **No-network**: no runtime downloads for weights/categories/tokenizers/NLP data.
- **No-fallback**: if a required model/artifact is missing → fail-fast (unless explicitly documented as optional).
- **Reproducibility**: every artifact must record `models_used[]`, `model_signature`, `weights_digest` (when applicable).

---

## VisualProcessor

### Core providers (VisualProcessor/core/model_process)

#### `core_clip` (`VisualProcessor/core/model_process/core_clip/main.py`)
- **Role**: image embeddings (and optionally text embeddings).
- **Runtime**:
  - `inprocess`: uses `clip` (OpenAI CLIP python package) via `clip.load(model_name, device=...)`.
  - `triton`: uses `dp_triton` HTTP client and a client-side preprocess preset (`openai_clip_224|336|448`).
- **Local artifacts**:
  - `inprocess`: **requires local CLIP weights** (should be enforced; currently `clip.load()` can trigger download if cache is missing).
  - `triton`: model lives in Triton repo; client needs endpoint URL + model name/version.
- **Meta**:
  - accepts `--model-version`, `--weights-digest`, `--engine`, `--precision`, `--runtime`.
  - writes `models_used[]` via `apply_models_meta(...)`.

#### `core_depth_midas` (`VisualProcessor/core/model_process/depth_midas/main.py`)
- **Role**: depth estimation.
- **Runtime**: **Triton-only** (client is pure numpy/cv2; preprocessing assumed in Triton).
- **Engine**: expects `onnx` behind Triton.
- **Local artifacts**: Triton deployment provides weights; client passes `--model-version`, `--weights-digest`.

#### `core_optical_flow` (`VisualProcessor/core/model_process/core_optical_flow/main.py`)
- **Role**: optical flow / motion curve.
- **Runtime**: **Triton-only** (2-input model).
- **Engine**: expects `onnx` behind Triton.
- **Local artifacts**: Triton deployment provides weights; client passes `--model-version`, `--weights-digest`.

#### `core_object_detections` (`VisualProcessor/core/model_process/object_detections/main.py`)
- **Role**: object detection + tracking.
- **Model**:
  - **Ultralytics YOLO** loaded via `YOLO(model_path)`.
  - ByteTrack is vendored (`ByteTrack/yolox/...`) and used for tracking.
- **Local artifacts**:
  - YOLO weight file path (`model_path`) must exist locally.
  - ByteTrack is repo-local.
- **Network risk**: low (Ultralytics can download if given a model name instead of a local path; ModelManager must enforce `local_path` only).

#### `core_face_landmarks` (`VisualProcessor/core/model_process/core_face_landmarks/main.py`)
- **Role**: pose/hands/face landmarks.
- **Model**: **MediaPipe** solutions (`face_detection`, `pose`, `hands`, `face_mesh`).
- **Local artifacts**: MediaPipe bundles models in the package (no explicit weights path).
- **Network risk**: low (assuming installed wheels include models).

---

### Visual modules (VisualProcessor/modules)

#### `cut_detection` (`VisualProcessor/modules/cut_detection/cut_detection.py`)
- **Models used**:
  - Optional deep features: `torchvision.models.resnet18/resnet50(pretrained=True)` (**network risk** if weights missing).
  - Optional CLIP (`clip` package): requires local CLIP weights (no-network enforced in recent changes).
- **Status**:
  - Deep features are **disabled by default** (`use_deep_features=false`).
  - CLIP path/weights root is configurable (`clip_download_root` / env) and should be enforced as local-only.

#### `shot_quality` (`VisualProcessor/modules/shot_quality/shot_quality.py`)
- **Models used**: relies on **core providers**: `core_clip`, `core_depth_midas`, `core_object_detections`, `core_face_landmarks` (per DAG).
- **Runtime**: does not load torch models directly (numpy-only post-processing + core artifacts).

#### `scene_classification` (`VisualProcessor/modules/scene_classification/scene_classification.py`)
- **Models used**:
  - Places365 classifier via **ModelManager** (`dp_models`):
    - Legacy in-process (torch): local artifacts under `DP_MODELS_ROOT` (no-network, fail-fast).
    - Triton (baseline fixed-shape): `places365_resnet50_{224,336,448}_triton` (HTTP client + Triton repo models).
  - CLIP semantics (aesthetic/luxury/atmosphere) computed **only from `core_clip`** embeddings and `core_clip`-exported prompt text-embeddings.
- **Network risk**: **low** (no URL fetch; fails fast if local artifacts are missing).

#### `action_recognition` (`VisualProcessor/modules/action_recognition/action_recognition_slowfast.py`)
- **Model**: `torchvision.models.video.slowfast_r50(pretrained=True)`.
- **Network risk**: **high** if torchvision weights not cached locally.
- **ModelManager target**: load weights from `MODELS_ROOT` and fail-fast if missing; never rely on `pretrained=True`.

#### `high_level_semantic` (`VisualProcessor/modules/high_level_semantic/hl_semantic.py`)
- **Model**: OpenAI CLIP via `clip.load(model_name, ...)`.
- **Network risk**: **high** if CLIP weights not present locally.
- **ModelManager target**: route through `core_clip` (preferred) or enforce local-only CLIP weights.

#### `emotion_face` (`VisualProcessor/modules/emotion_face/...`)
- **Model**: EmoNet loaded from local path like `.../models/emonet/pretrained/emonet_8.pth`.
- **Local artifacts**: repo-local weight file path.
- **Other deps**: face selection now comes from `core_face_landmarks.face_present` (two-stage core provider). The legacy `face_detection` module was removed.

#### `story_structure`
- **Baseline module**: uses only `core_clip` (no local model loading).
- **Legacy pipeline**: uses `SentenceTransformer(..., cache_folder=...)` and may download if not pre-cached.

---

## AudioProcessor

### `asr_extractor` (`AudioProcessor/src/extractors/asr_extractor.py`)
- **Model**: Whisper **via Triton** (client uses `dp_triton`).
- **Network risk**: **low** (no runtime downloads; model served by deployment).
- **ModelManager**: specs:
  - `dp_models/spec_catalog/audio/whisper_small_triton.yaml`
  - `dp_models/spec_catalog/audio/whisper_medium_triton.yaml`
  - `dp_models/spec_catalog/audio/whisper_large_triton.yaml`
- **Tokenizer**: shared tokenizer under `dp_models/spec_catalog/text/shared_tokenizer_v1.yaml` (used to interpret token IDs).

### `clap_extractor` (`AudioProcessor/src/extractors/clap_extractor.py`)
- **Model**: LAION CLAP via `laion_clap.CLAP_Module(...).load_ckpt()`.
- **Network risk**: **high** unless CKPT is present locally (package may download).
- **ModelManager**: enforced via spec `dp_models/spec_catalog/audio/laion_clap.yaml` (local-only, fail-fast).

### `speaker_diarization_extractor` (`AudioProcessor/src/extractors/speaker_diarization_extractor.py`)
- **Model**: speaker embedding model **via Triton** (client uses `dp_triton`) + clustering (CPU).
- **Network risk**: **low** (no runtime downloads; model served by deployment).
- **ModelManager**:
  - `dp_models/spec_catalog/audio/speaker_diarization_small_triton.yaml`
  - `dp_models/spec_catalog/audio/speaker_diarization_large_triton.yaml`

### `emotion_diarization_extractor` (`AudioProcessor/src/extractors/emotion_diarization_extractor.py`)
- **Model**: emotion diarization model **via Triton** (client uses `dp_triton`).
- **Network risk**: **low** (no runtime downloads; model served by deployment).
- **ModelManager**:
  - `dp_models/spec_catalog/audio/emotion_diarization_small_triton.yaml`
  - `dp_models/spec_catalog/audio/emotion_diarization_large_triton.yaml`

### `source_separation_extractor` (`AudioProcessor/src/extractors/source_separation_extractor.py`)
- **Model**: source separation model **via Triton** (client uses `dp_triton`) on log‑mel inputs.
- **Network risk**: **low** (no runtime downloads; model served by deployment).
- **ModelManager**:
  - `dp_models/spec_catalog/audio/source_separation_small_triton.yaml`
  - `dp_models/spec_catalog/audio/source_separation_medium_triton.yaml`
  - `dp_models/spec_catalog/audio/source_separation_large_triton.yaml`

### `hpss_extractor` (`AudioProcessor/src/extractors/hpss_extractor.py`)
- **Model**: none (signal processing via `librosa.decompose.hpss`).

### `speech_analysis_extractor` (`AudioProcessor/src/extractors/speech_analysis_extractor.py`)
- **Model**: none directly (aggregator). Calls:
  - `asr_extractor` (Whisper via Triton)
  - `speaker_diarization_extractor` (Triton embeddings + clustering)
  - optional `pitch_extractor` (signal processing)
- **Network risk**: **low** (inherits no-network policy from sub-extractors; no downloads).

---

## TextProcessor

### Embeddings (SentenceTransformers)
- **Registry**: `TextProcessor/src/core/model_registry.py` caches `SentenceTransformer(model_name, device=device)`.
- **Models observed**:
  - `intfloat/multilingual-e5-large` (used broadly in `simantic_embeddings/*`, `semantic_topic_extractor.py`).
  - `all-MiniLM-L6-v2` appears in Visual `story_structure` legacy pipeline.
- **Network risk**: **high** unless HF cache is pre-populated; SentenceTransformers will fetch from HuggingFace by default.
- **ModelManager target**:
  - enforce `local_path` under `MODELS_ROOT` or offline HF cache pinned by digest.
  - set offline env flags and fail-fast if missing.

### Optional NLP models (spaCy)
- **Sites**:
  - `TextProcessor/src/extractors/asr_text_proxy_audio_features/asr_text_proxy_extractor.py` uses `spacy.load("ru_core_news_sm"|"en_core_web_sm")`.
  - `TextProcessor/src/extractors/lexico_static_features/lexical_stats_extractor.py` also uses spaCy optionally.
- **Network risk**: **medium/high** (spaCy models are separate packages that must be installed; not downloaded at runtime, but missing packages cause fallbacks).
- **ModelManager target**: treat spaCy pipelines as install-time artifacts; record version in `models_used[]` when enabled.

### BERTopic stack
- `SemanticTopicExtractor` uses `BERTopic` + optional `UMAP`/`HDBSCAN`.
- Embedding model is still `SentenceTransformer` (covered above).

---

## Shared / infra

### `dp_triton` (`dp_triton/http_client.py`)
- Not a model itself, but defines how Triton runtime is accessed and thus must be integrated into ModelManager runtime abstraction.

### `health/app.py`
- Uses `requests.get(...)` for health checks; not a model, but relevant to “no-network” policy for production environments (health endpoints should not depend on external URLs).


