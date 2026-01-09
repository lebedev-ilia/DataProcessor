## Baseline component/model results (data)

Этот документ — **таблица с фактическими измерениями** по baseline (данные).
Протокол и правила измерений описаны в:
- `docs/baseline/BASELINE_COMPONENT_MODEL_CHECKLIST.md`

Рекомендуемый способ обновлять таблицы из артефактов прогонов (JSON), чтобы избежать дублей и ручных ошибок:
- `scripts/baseline/render_checklist_results_md.py`

### Как читать VRAM

Мы фиксируем **только** VRAM delta по процессу `tritonserver`:
- `vram_triton_delta_run_mb`: **приближение “памяти прогона”** (peak-before)

Важно:
- Большой baseline (4–5GB) — нормально: Triton держит модели загруженными + ORT CUDA memory pools.
- Для честных сравнения/дельт на 6GB — **перезапускаем Triton** между тяжёлыми моделями/группами.

---

### Results (Visual) — micro (per-frame)

Колонки:
- `component_or_module`: что считаем (core/module)
- `model_branch`: fixed-shape ветка модели
- `source_resolution`: исходный WxH (видео-кадр)
- `selected_branch`: выбранная ветка по routing (S)
- `latency_ms_mean_stable` / `spikes`
- `rss_peak_mb`
- `vram_triton_delta_run_mb`
- `status` / `notes`
- `out_dir`: где лежит полный JSON

| component_or_module | model_branch | source_resolution | selected_branch | latency_ms_mean_stable | spikes | rss_peak_mb | vram_triton_delta_run_mb | status | notes | out_dir |
|---|---|---:|---:|---:|:---:|---:|---:|---|---|---|
| core_clip | clip_image_448 | 1280×720 (16:9, S=720) | 448 | 166.837 | false | 55.754 | 10 | ok | docker/titrion restart before run | `/tmp/checklist-core-clip-1280x720-docker-restart-20260108-051224` |
| scene_classification | places365_resnet50_448 | 1280×720 (16:9, S=720) | 448 | 158.558 | true | 53.172 | 688 | ok | spikes observed; restart recommended on 6GB | `/tmp/checklist-places365-1280x720-20260108-051431` |


