# ModelManager plan (unified, extensible)

This is the implementation plan for a **single, unified ModelManager** for the whole DataProcessor (Visual/Audio/Text), aligned with `MODEL_MANAGER_Q.md` and `docs/MODEL_SYSTEM_RULES.md`.

Goals:
- **No-network, no-fallback** enforcement for all model artifacts.
- Consistent `models_used[]`, `weights_digest`, `model_signature` in outputs/manifests.
- One registry-style system where **adding a new model is easy** (declarative spec + small provider adapter).

---

## 1) Proposed architecture

### 1.1 Core abstractions

- **`ModelSpec`** (declarative, YAML/JSON):
  - `model_name`, `model_version`
  - `role` (e.g. `image_embedding`, `asr`, `audio_embedding`, `object_detection`, `scene_classification`)
  - `runtime`: `inprocess` | `triton`
  - `engine`: `torch` | `onnx` | `tensorrt`
  - `precision`: `fp32` | `fp16`
  - `device_policy`: `cpu` | `cuda` | `auto` | `cuda:0`
  - `local_artifacts`: list of required paths under `${DP_MODELS_ROOT}` (files/dirs)
  - `weights_digest`: required for `inprocess`; provided externally for `triton`
  - `preprocess_preset` (optional) + `preprocess_signature` policy
  - `input_schema`, `output_schema` (for validation and future feature extraction alignment)
  - `resource_requirements` (VRAM/RAM estimate, batch constraints)

- **`ResolvedModel`** (runtime instance):
  - `spec` + resolved device/precision/runtime
  - loaded model handle (torch module / client wrapper / callable)
  - `models_used_entry` (for `meta.models_used[]`)
  - `model_signature` (canonical string)

- **`ModelProvider`** (plugin contract):
  - `supports(spec) -> bool`
  - `validate_local_artifacts(spec, models_root) -> None` (fail-fast)
  - `load(spec, ctx) -> ResolvedModel`
  - `infer(resolved, inputs, ctx) -> outputs` (optional; some modules call provider directly)

- **`ModelManager`**:
  - loads `ModelSpec` catalog (YAML files) + run/profile config overrides
  - resolves to `ResolvedModelMapping` (for manifest + reproducibility)
  - exposes `get(role|name, ctx)` returning `ResolvedModel`
  - global, thread-safe **LRU** caching per `(model_name, device, precision, runtime, preprocess_preset)`

### 1.2 Where it lives

Create a top-level package, e.g.:
- `dp_models/`
  - `manager.py` (`ModelManager`)
  - `specs/` (YAML catalog)
  - `providers/` (provider plugins)
  - `errors.py` (standard error codes)
  - `digests.py` (SHA256 utilities for weights)
  - `signatures.py` (canonical `model_signature_for(spec)`)

### 1.3 Enforcing “no-network”

At runtime, ModelManager sets/validates:
- `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`
- `SENTENCE_TRANSFORMERS_HOME=${DP_MODELS_ROOT}/hf_cache`
- `TORCH_HOME=${DP_MODELS_ROOT}/torch_cache`
- disable `torch.hub` usage entirely (or wrap and fail-fast)

Additionally:
- provider implementations **must not** call `requests.get`, `from_pretrained(...)` without `local_files_only=True`, `clip.load(...)` without a validated local download root, etc.
- unit tests should monkeypatch `socket` to ensure no outbound connections are made.

---

## 2) Migration plan (phased)

### Phase A — Foundations (no functional change, just infrastructure)
- Implement `ModelSpec`, `ResolvedModel`, `ModelProvider`, `ModelManager`.
- Implement canonical `model_signature_for(spec)` + `models_used_entry(spec)`.
- Implement `weights_digest`:
  - inprocess: `sha256(file)` (or directory manifest hash).
  - triton: supplied by deployment, validated present.
- Add `docs/MODEL_INVENTORY.md` (this file already exists) as the baseline map.

### Phase B — TextProcessor first (lowest coupling)
Target: `TextProcessor/src/core/model_registry.py` → ModelManager.
- Replace `get_model()` with `ModelManager.get(role="text_embedding", ...)`.
- Add `ModelSpec` for `intfloat/multilingual-e5-large` as a **local artifact** (either cloned snapshot or cached repo under `DP_MODELS_ROOT`).
- Ensure all text embedder extractors record `models_used[]` and `model_signature` into their manifests/artifacts.

### Phase C — AudioProcessor models (Whisper/CLAP/SpeechBrain/Resemblyzer/Open-Unmix)
Replace implicit downloads:
- Whisper: `whisper.load_model("small")` → load from local weights path.
- LAION CLAP: `load_ckpt()` → explicit local ckpt.
- SpeechBrain: `from_hparams(...)` must point to a fully local bundle and must not fetch missing files.
- Resemblyzer: ensure the encoder weights exist locally and are referenced explicitly.
- Open-Unmix: treat pretrained weights as explicit artifacts.

Outcome: **all audio models become deterministic and offline**.

### Phase D — Visual “problem modules” (explicit network and pretrained=True)
Focus modules that currently violate no-network/no-fallback:
- `scene_classification`:
  - remove `requests.get(CATEGORY_URL)`; require local `categories_places365.txt`.
  - replace Places365 model URLs with local checkpoint path.
  - disallow HF CLIP in-module; use `core_clip` only.
  - disallow `timm(pretrained=True)` unless weights are local and pinned.
- `action_recognition`:
  - replace `slowfast_r50(pretrained=True)` with local weights.
- `high_level_semantic`:
  - route to `core_clip` or enforce local CLIP weights through ModelManager.
- `face_detection` (InsightFace):
  - require local model pack and pin version/digest.

### Phase E — Visual core providers
Core providers already accept `model_version/weights_digest/runtime`.
Wire them to ModelManager so the orchestrator doesn’t manually pass these flags:
- `core_clip`, `core_depth_midas`, `core_optical_flow`, `core_object_detections`, `core_face_landmarks`.

---

## 3) How to add a new model (the “easy extensibility” contract)

To add a model, you do **two things**:

1) Add a new spec file, e.g. `dp_models/specs/<role>/<model_name>.yaml`
- include local artifacts under `${DP_MODELS_ROOT}`
- set runtime/engine/precision defaults

2) Implement (or reuse) a provider:
- if it’s a new framework/runtime, add `dp_models/providers/<framework>.py`
- register it in the provider registry

That’s it. No component code should hardcode URLs or framework-specific loading logic.

---

## 4) Standard error codes (model-related)

ModelManager raises structured errors with `error_code`:
- `model_not_found`
- `weights_missing`
- `weights_digest_mismatch`
- `triton_unavailable`
- `insufficient_gpu_memory`
- `unsupported_runtime`
- `unsupported_engine`
- `network_forbidden`

---

## 5) Tests & CI enforcement

Add tests that:
- validate `ModelSpec` schema and canonicalization of `model_signature`
- ensure `weights_digest` is stable and correct
- ensure **no-network**:
  - monkeypatch `socket.socket.connect` to throw and run a “smoke load” of all Tier-0 models
- ensure **no-fallback**:
  - delete a required local artifact and assert a hard error

---

## 6) Immediate next steps (PR-sized)

1) Add `dp_models/` skeleton with `ModelManager` + spec loader + signature/digest utilities.
2) Wire `TextProcessor/src/core/model_registry.py` to use ModelManager (first concrete integration).
3) Add a small “model health” endpoint hook for `health/app.py` (optional): report Tier-0 readiness without contacting the internet.


