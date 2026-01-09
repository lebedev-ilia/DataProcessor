## Benchmark harness (baseline GPU / Triton)

Цель: прогонять **ветки моделей** (фиксированные формы, без dynamic axes) и собирать:
- latency (warmup + steady-state, p50/p90/p99 по итогам)
- RSS (CPU)
- (опционально) GPU VRAM через NVML (если доступно)

Контракт baseline GPU I/O:
- image models: **UINT8 NHWC** (сырой RGB кадр), preprocess внутри Triton graph/ensemble
- text models: **INT64 tokens**

### Запуск

1) Укажи Triton URL (или экспортируй переменную окружения):

```bash
export TRITON_HTTP_URL="http://triton:8000"
```

2) Запусти бенч:

```bash
python -m benchmarks.run_bench --spec benchmarks/specs/baseline_gpu.yaml
```

### Важно для 6GB GPU (local)

Если прогонять **все модели подряд** в одном Triton процессе, ORT CUDA может начать возвращать OOM
для последующих моделей (особенно MiDaS/RAFT/YOLO960) из-за удержания/фрагментации VRAM.

Рекомендация: гонять “baseline_gpu_local” **по группам** и делать restart Triton между группами:
- `benchmarks/specs/baseline_gpu_local_group_clip_places.yaml`
- `benchmarks/specs/baseline_gpu_local_group_midas_raft.yaml`
- `benchmarks/specs/baseline_gpu_local_group_yolo.yaml`

Опции:
- `--out-dir benchmarks/out/<name>` (по умолчанию timestamp)
- `--warmup 5` / `--repeats 30`
- `--filter clip_image_` (прогон только части моделей)
- `--dry-run` (только план прогонов)

### Выход

Пишется:
- `results.jsonl` — одна строка на один прогон
- `summary.json` — агрегаты по (model_variant × batch)


