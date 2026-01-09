## Pre‑Triton bench + ONNX export (baseline GPU models)

Цель: **до Triton** определить оптимальные фиксированные размеры входов, замерить latency/memory, затем **экспортировать ONNX ветки** для Triton.

### Контракты (важно)

- **Image input contract (baseline GPU)**: **`UINT8 NHWC`** (сырой RGB).
- Preprocess (resize/normalize/layout) — будет перенесён в **Triton ensemble/graph**.
- **Никаких dynamic axes**: только фиксированные ветки (например 256/384/512).
- **Offline policy**: никакие веса не скачиваются в рантайме; всё лежит в `DP_MODELS_ROOT`.

---

## 1) DP_MODELS_ROOT (offline bundle)

Установи переменную окружения:

- `DP_MODELS_ROOT=/abs/path/to/DataProcessor/dp_models/bundled_models`

Внутри должны быть:

- `torch_cache/` (**TORCH_HOME**): torch.hub repos + checkpoints (MiDaS/RAFT/torchvision).
- `hf_cache/` (**HF_HOME**): HuggingFace cache (в т.ч. tokenizer caches для laion_clap).
- `clip_cache/` (**DP_CLIP_WEIGHTS_DIR**): OpenAI CLIP weights (`ViT-B-32.pt`, …).
- `visual/…`: явные артефакты по `dp_models/spec_catalog/vision/*.yaml`
- `audio/…`: явные артефакты по `dp_models/spec_catalog/audio/*.yaml`

### Минимум для текущего pre‑triton

- MiDaS: `dp_models/bundled_models/torch_cache/hub/checkpoints/midas_v21_small_256.pt` и `dpt_hybrid_384.pt`
- RAFT: `dp_models/bundled_models/torch_cache/hub/checkpoints/raft_small_*.pth` (+ optional `raft_large_*.pth`)

### Минимум для baseline‑audio CLAP (offline)

- CLAP checkpoint: `dp_models/bundled_models/audio/laion_clap/clap_ckpt.pt`
- HF caches (tokenizers only):
  - `roberta-base`
  - `bert-base-uncased`
  - `facebook/bart-base`

---

## 2) Pre‑Triton bench (torch/torchvision, offline)

Spec matrix:
- `benchmarks/specs/pretriton_gpu.yaml`

Run (через venv VisualProcessor):

```bash
cd "/home/ilya/Рабочий стол/TrendFlowML/DataProcessor"
PY="/home/ilya/Рабочий стол/TrendFlowML/DataProcessor/VisualProcessor/.vp_venv/bin/python"
MR="/home/ilya/Рабочий стол/TrendFlowML/DataProcessor/dp_models/bundled_models"

"$PY" -m benchmarks.run_pretriton_bench \
  --spec benchmarks/specs/pretriton_gpu.yaml \
  --device cuda --dtype fp16 \
  --models-root "$MR" \
  --offline
```

Outputs:
- `benchmarks/out/<run_id>/results.jsonl`
- `benchmarks/out/<run_id>/summary.json`

Notes:
- MiDaS может работать в fp16.
- RAFT: в pre‑triton бенче fp16 может быть нестабилен на `grid_sample`, поэтому бенч принудительно использует fp32 compute для RAFT при `--dtype fp16` (в `results.jsonl` пишется `dtype_used`).

Важно про `benchmarks.run_pretriton_bench`:
- модель загружается **1 раз на вариант** (latency = preprocess+forward, без cost загрузки весов);
- `summary.json` пишется **инкрементально** (если прогон упадёт/прервётся — останется частичный summary).

Если у тебя есть старый прогон только с `results.jsonl` (без `summary.json`), можно собрать summary пост‑фактум:

```bash
PY="/home/ilya/Рабочий стол/TrendFlowML/DataProcessor/VisualProcessor/.vp_venv/bin/python"
"$PY" -m benchmarks.summarize_results \
  --in benchmarks/out/<run_id>/results.jsonl \
  --bench pretriton
```

### Пример результатов (p50, ms)

Прогон: `benchmarks/out/pretriton-full3-20260107-175801/summary.json` (cuda, requested fp16; RAFT считает fp32).

| variant | batch=1 p50 | batch=8 p50 |
|---|---:|---:|
| `midas_small_256` | 14.9 | 21.4 |
| `midas_hybrid_384` | 31.5 | 160.2 |
| `midas_hybrid_512` | 45.6 | 288.6 |
| `raft_small_256` | 33.8 | 66.9 |
| `raft_small_384` | 34.9 | 145.9 |
| `raft_small_512` | 43.3 | 260.2 |

### ONNX I/O (полезно для следующего шага: Triton repo)

- MiDaS/DPT:
  - input: `input` (FLOAT, `[1,3,H,W]`)
  - output: `depth` (FLOAT, `[1,H,W]`)
- RAFT:
  - inputs: `input0`, `input1` (FLOAT, `[1,3,H,W]`)
  - output: `flow` (FLOAT, `[1,2,H,W]`)

---

## 3) Export ONNX (fixed‑size branches, offline)

### MiDaS

```bash
cd "/home/ilya/Рабочий стол/TrendFlowML/DataProcessor"
PY="/home/ilya/Рабочий стол/TrendFlowML/DataProcessor/VisualProcessor/.vp_venv/bin/python"
MR="/home/ilya/Рабочий стол/TrendFlowML/DataProcessor/dp_models/bundled_models"

"$PY" scripts/model_opt/export_midas_onnx.py --model-name MiDaS_small --h 256 --w 256 \
  --out models/optimized/midas/midas_small_256.onnx --models-root "$MR" --offline

"$PY" scripts/model_opt/export_midas_onnx.py --model-name DPT_Hybrid --h 384 --w 384 \
  --out models/optimized/midas/dpt_hybrid_384.onnx --models-root "$MR" --offline

"$PY" scripts/model_opt/export_midas_onnx.py --model-name DPT_Hybrid --h 512 --w 512 \
  --out models/optimized/midas/dpt_hybrid_512.onnx --models-root "$MR" --offline
```

### RAFT

```bash
cd "/home/ilya/Рабочий стол/TrendFlowML/DataProcessor"
PY="/home/ilya/Рабочий стол/TrendFlowML/DataProcessor/VisualProcessor/.vp_venv/bin/python"
MR="/home/ilya/Рабочий стол/TrendFlowML/DataProcessor/dp_models/bundled_models"

"$PY" scripts/model_opt/export_raft_onnx.py --model raft_small --h 256 --w 256 \
  --out models/optimized/raft/raft_small_256.onnx --models-root "$MR" --offline

"$PY" scripts/model_opt/export_raft_onnx.py --model raft_small --h 384 --w 384 \
  --out models/optimized/raft/raft_small_384.onnx --models-root "$MR" --offline

"$PY" scripts/model_opt/export_raft_onnx.py --model raft_small --h 512 --w 512 \
  --out models/optimized/raft/raft_small_512.onnx --models-root "$MR" --offline

# optional
"$PY" scripts/model_opt/export_raft_onnx.py --model raft_large --h 384 --w 384 \
  --out models/optimized/raft/raft_large_384.onnx --models-root "$MR" --offline
```

Export outputs are currently stored as:
- `models/optimized/midas/*.onnx` + `*.onnx.data`
- `models/optimized/raft/*.onnx` + `*.onnx.data`

---

## 4) Next step (Triton)

После выбора веток по `summary.json`:

- собрать Triton model repository:
  - `midas_256`, `midas_384`, `midas_512` (ONNX)
  - `raft_256`, `raft_384`, `raft_512` (ONNX, two inputs)
- определить I/O tensors (names, dtypes) и собрать preprocess в ensemble:
  - client always sends `UINT8 NHWC`
  - ensemble делает resize/layout/normalize и вызывает ONNX backend


