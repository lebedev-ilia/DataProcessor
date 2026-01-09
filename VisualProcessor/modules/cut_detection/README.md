# `cut_detection` (Visual module, Tier‑0 baseline)

## Purpose

Detects **hard cuts** and **soft transitions** (fade/dissolve + motion transitions) on a sampled frame sequence and produces:
- a **shot boundary** timeline used by downstream modules (notably `shot_quality`)
- a rich set of **editing / pacing** features.

This module is part of the **baseline** stage and is **required**.

## Inputs

- **Primary input (frames)**: `frames_dir/metadata.json` + RGB frames via `FrameManager`.
  - **Sampling**: uses only `metadata["cut_detection"]["frame_indices"]` provided by Segmenter (no internal resampling).
  - **Time axis**: uses `union_timestamps_sec` from `frames_dir/metadata.json` as the **source‑of‑truth** timeline.
- **Optional input (audio)**: `audio/audio.wav` produced by Segmenter (auto‑resolved).

## Dependencies (inputs from other components)

Used for **jump‑cut** detection quality:
- **Required (baseline)**:
  - `core_face_landmarks` (`rs_path/core_face_landmarks/landmarks.npz`)
  - `core_object_detections` (`rs_path/core_object_detections/detections.npz`)

Optional (performance / shared motion curve):
- `core_optical_flow` (`rs_path/core_optical_flow/flow.npz`) — if enabled and aligned sampling, can be reused to avoid duplicate optical flow computation inside cut_detection.

The DAG stage must ensure these cores run **before** `cut_detection`.

## Outputs (artifact contract)

Artifact path (timestamped, via `BaseModule.save_results`):
- `result_store/<run>/<video_id>/cut_detection/cut_detection_features_<ts>_<uid>.npz`

Additional (recommended) model-facing artifact (schema v1):
- `result_store/<platform_id>/<video_id>/<run_id>/cut_detection/cut_detection_model_facing_<ts>_<uid>.npz`
  - schema: `VisualProcessor/modules/cut_detection/SCHEMA_MODEL_FACING.md`

Writing policy (MVP):
- Default: CLI writes the model-facing NPZ **best-effort** (non-fatal).
- Disable (rare): CLI `--no-write-model-facing-npz`
- Make it **required** (fail-fast on write problems): CLI `--require-model-facing-npz`

Hard cuts performance knobs:
- Default mode computes SSIM + optical flow for **all** frame pairs (higher quality / stable thresholds).
- Optional speed mode (OFF by default): histogram-gated cascade for hard cuts:
  - CLI: `--hard-cuts-cascade --hard-cuts-cascade-keep-top-p 0.25 --hard-cuts-cascade-hist-margin 0.0`
  - Tradeoff: may miss rare cuts where histogram change is extremely low; use only for `fast` experiments.
  - Model-facing NPZ note: in cascade mode `ssim_drop/flow_mag/deep_cosine_dist` may contain `NaN` for pairs where the signal was not computed; use `*_valid_mask` keys.

Hard cuts presets (CLI convenience):
- `--hard-cuts-preset quality|default|fast`
  - `quality`: ssim=640, flow=384, cascade=off
  - `default`: ssim=512, flow=320, cascade=off
  - `fast`: ssim=384, flow=256, cascade=on (keep_top_p=0.25, margin=0.0)
  - explicit flags `--ssim-max-side/--flow-max-side/--hard-cuts-cascade*` override preset values.

Soft/motion optimizations:
- If `--prefer-core-optical-flow` is enabled and aligned, soft_cuts and motion_based_cuts reuse the core motion curve for `*_flow_mag` where possible.
- Motion detection uses a cascade when core flow is available: it computes expensive Farneback direction/variance only for motion spike candidates (best-effort).

NPZ keys (high level):
- `meta`: dict (required baseline keys + `models_used[]` / `model_signature` when model‑based)
- `frame_indices`: `int32 [N]` (union‑domain indices used by this module)
- `times_s`: `float32 [N]` = `union_timestamps_sec[frame_indices]`
- `features`: dict of scalar features (counts/ratios/statistics)
- `detections`: dict with intermediate events:
  - `hard_cuts`: list[int] (positions in sampled sequence)
  - `soft_events`: list[dict] with `{type, start, end, duration_s}`
  - `motion_cuts`: list[int]
  - `jump_cuts`: list[int] (subset of hard cuts)

### Model-facing output (recommended for Transformers)

For `baseline/v1/v2` models (especially transformers), **events-only** output is not enough.
We recommend exposing (and saving) **raw per-step signals** so a downstream FeatureEncoder can learn
robust pooling/attention over time:

- **Dense curves (per frame_pair / per sampled step)**:
  - `hist_diff[t]` — cheap content change proxy
  - `ssim_drop[t]` — structural similarity drop
  - `flow_mag[t]` — motion magnitude (preferably from `core_optical_flow`)
  - `hard_score[t]` — combined hard-cut score before postprocessing
- **Sparse events**:
  - hard/soft/motion events with `{time_s, type, strength, contributors}`

Rationale:
- Keeps the pipeline reproducible while letting the model learn thresholds and interactions.
- Avoids overfitting to brittle postprocessing heuristics.
- Works for both short and very long videos via fixed-budget encoding (see `docs/models_docs/FEATURE_ENCODER_CONTRACT.md`).

### Models metadata

- If `use_clip=true`, module records `models_used[]` with:
  - `model_name="openai_clip_vit_b32"`, `runtime="inprocess"`, `engine="clip"`, `device`, and best‑effort `weights_digest` (sha256 of local weight file).

## Sampling / units‑of‑processing requirements (Visual)

- **Coverage goal**: uniform coverage over the entire video to reliably detect cuts and compute pacing statistics.
- **Min/target/max** (start values, will be refined after full audit):
  - `min_frames`: **400**
  - `target_frames`: **800**
  - `max_frames`: **1500**
- **Time axis**: `union_timestamps_sec` is required and must be monotonic.
- **Max sampling gap (quality gate)**:
  - if `max(diff(times_s)) > 6.0s` → **error** (sampling too sparse for reliable cut detection).
- **Resolution**:
  - This module is robust to moderate downscaling; it does not require per‑component high‑res.
  - It operates on RGB frames from Segmenter; upstream should avoid upscaling (no‑upscale policy).

## Empty / error semantics (no‑fallback)

- `frame_indices` missing/empty → **error**
- `union_timestamps_sec` missing/invalid/non‑monotonic → **error**
- `core_face_landmarks` or `core_object_detections` artifact missing (baseline) → **error**
- `len(frame_indices) < 2` → **error**

Valid “empty” is generally **not expected** for this baseline module (it must produce a timeline).

## Performance notes

- Pure CPU heuristics (hist/SSIM/Farneback) + optional audio features.
- CLIP (if enabled) is the heaviest part; runtime downloads are forbidden (weights must exist locally).

### Performance knobs (quality-preserving)

This module has explicit, deterministic knobs to keep quality while reducing CPU cost on high-res inputs:

- **`flow_max_side`** (default **320**): caps resolution used for Farneback optical flow magnitude.
- **`ssim_max_side`** (default **512**): caps resolution used for SSIM (grayscale) drop metric.

Rules:
- If Segmenter already outputs frames with max-side <= these values, behavior is unchanged.
- If frames are large (e.g., 720p+), SSIM/flow are computed on downscaled grayscale images (aspect ratio preserved).
- This is not an adaptive heuristic based on content; it is a deterministic performance policy.

### Optional reuse of `core_optical_flow`

If `core_optical_flow` ran before this module and produced an aligned motion curve (`flow.npz`), you can enable reuse:
- CLI: `--prefer-core-optical-flow` (best-effort)
- CLI: `--require-core-optical-flow` (fail-fast if missing/mismatch)

This avoids duplicate CPU Farneback computation and is consistent with the project plan to share heavy signals via core providers.

---

## Quality & optimization notes (implementation guidance)

### 1) Prefer shared motion from `core_optical_flow`

If `core_optical_flow/flow.npz` is available **and** aligned with `cut_detection.frame_indices`,
reuse it to avoid redundant optical-flow work and to keep motion semantics consistent across modules.
CPU Farneback should remain only as a fallback.

### 2) Cascade / early-exit (cheap → expensive)

To preserve quality but reduce cost on “easy” segments:
- compute `hist_diff` first (cheap)
- only compute `ssim_drop` / `flow_mag` for candidates (or at a lower rate) when `hist_diff` suggests a possible transition

This reduces average runtime without changing deterministic behavior (policy can be parameterized explicitly).

### 3) Stability / determinism

- All performance knobs (`ssim_max_side`, `flow_max_side`, reuse flags) must be **explicit parameters** (no hidden heuristics).
- Always use `union_timestamps_sec` as the source-of-truth timeline.
- If sampling is too sparse (`max gap > 6s`), fail-fast: cut statistics become unreliable.

### 4) Quality risks to watch

- Over-aggressive temporal smoothing can suppress isolated true cuts. Prefer “keep strong spikes” logic.
- `soft_cuts` and `motion_based_cuts` are inherently noisier; treat them as **probabilistic cues** for the model,
  not as ground-truth boundaries.


