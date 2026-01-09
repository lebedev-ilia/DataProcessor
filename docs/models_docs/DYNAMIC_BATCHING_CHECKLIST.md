## Dynamic Batching — resource checklist (MVP)

Цель: это **источник правды** для scheduler’а и оркестраторов про:
- **единицу обработки** (unit) для каждого компонента,
- **стоимость** (latency / memory) на unit в фиксированных bucket’ах входа,
- **ограничения батчинга** (cross-video, max batch, Triton policy),
- **зависимости** (hard deps, optional deps),
- **какие параметры влияют на cost** (analysis_* / ветка / knobs).

Связанные документы:
- `docs/models_docs/DynamicBatching_Q_A.md` — решения/контракты (OOM/backoff/headroom/observability, Triton=Level4)
- `docs/models_docs/FEATURE_ENCODER_CONTRACT.md` — как приводим variable-length outputs компонентов к fixed-size представлению для v1/v2 transformer
- `docs/baseline/BASELINE_COMPONENT_MODEL_CHECKLIST.md` — протокол измерений и сетка входов
- `docs/models_docs/MODEL_SYSTEM_RULES.md` — `model_signature`, idempotency, no-fallback

---

### 0) Правила (MVP) — что scheduler может/не может делать

- **No-fallback на CPU** (cuda→cpu запрещено).
- **OOM policy**: 3 попытки, backoff=2s, последняя `batch_size=1`; если на `1` OOM → `error` и стоп.
- **GPU budget** (принято): pinned weights + activation peak + headroom:
  - `free_vram_mb = gpu_total_mb - gpu_used_mb`
  - `headroom_mb = max(1024, round(free_vram_mb * 0.25))`
  - `effective_budget_mb = max(0, free_vram_mb - headroom_mb)`
  - `batch_size = clamp(floor(effective_budget_mb / gpu_memory_per_task_mb), 1, max_batch_size_component)`
- **Cross-video batching разрешён**, если совпадают:
  - `component_name`
  - `model_signature` (если есть модели)
  - `runtime` (`triton`/`inprocess`) и endpoint/device policy
  - preprocessing параметры, влияющие на модель (минимум: `analysis_fps/analysis_width/analysis_height`, resize policy)
  - resolution bucket
- **Hard cap** на cross-video micro-batch: **8 видео**.
- **Triton** = Level 4: DataProcessor формирует micro-batches для RPC, Triton может дальше агрегировать.

---

### 1) Схема строки чек-листа (как заполнять)

Каждая запись = (component, unit, input_bucket, runtime/model_signature, knobs) → cost/constraints.

Поля (минимум):
- **component**: каноническое имя
- **owner**: `visual|audio|text|segmenter|models|fetcher`
- **unit**: что именно “батчим” (frame/segment/clip/video)
- **unit_cost_model**: как масштабируется по входу (linear in frames, per-segment, etc.)
- **input_bucket**: (для visual) `source WxH + fps + short_side_bucket`; (для audio) `duration_sec_bucket`
- **runtime**: `inprocess|triton`
- **model_signature**: если применимо (иначе `N/A`)
- **gpu_memory_per_task_mb**: пик на unit (для планировщика batch_size)
- **cpu_rss_peak_mb**: пик RSS на unit (если релевантно)
- **latency_ms_per_unit**: stable mean per unit
- **max_batch_size_component**: hard cap компонента (если применимо)
- **cross_video_batching**: `yes/no` + constraints
- **dependencies**: hard/optional
- **notes**: важные детали (например “reuse core provider artifact”, “requires aligned sampling”)
- **provenance**: откуда цифры (json path + git commit + date)

---

### 2) Первые заполненные данные (seed)

> Важно: это **seed**. Дальше мы расширяем сетку bucket’ов (S=224/320/448/640 и т.д.) и добавляем остальные компоненты.

#### 2.1 Visual / module: `cut_detection`

Единица обработки для планировщика:
- **unit**: `video_frames` (последовательный анализ внутри одного видео)
- **unit_cost_model**: примерно линейный по числу кадров: `T ≈ n_frames * latency_ms_per_frame_pair` (пары соседних кадров)
- Cross-video batching: **да** (параллельный запуск по видео), но внутри видео батчинг “по кадрам” не применяется.

Параметры, влияющие на cost:
- `source_resolution` (WxH)
- `fps`
- knobs: `ssim_max_side`, `flow_max_side`
- optional optimization: reuse `core_optical_flow` (если sampling aligned)

Seed measurements (synthetic, 1280x720@30fps, n_frames=64):

| component | unit | input_bucket | knobs | latency_ms_per_unit | gpu_memory_per_task_mb | runtime | cross_video_batching | dependencies | provenance |
|---|---|---|---|---:|---:|---|---|---|---|
| cut_detection.detect_hard_cuts(cpu_no_deep) | frame_pair | 1280x720@30fps | ssim_max_side=512, flow_max_side=320 | 48.33 | N/A (not measured) | inprocess | yes (per-video parallelism) | optional: core_optical_flow | `/tmp/checklist-components-cut_detection-split-optimized-1280x720-64f-20260109-0246/checklist_components_micro_results.json` @ `7c2cfe26` |

Дополнение: CPU memory (seed)
- На том же synthetic прогоне (1280x720@30fps, n_frames=64) после добавления fallback RSS через `resource.getrusage`:
  - `cpu_rss_peak_mb` ≈ **353.4 MB**
  - JSON: `/tmp/checklist-components-cut_detection-rss-check/checklist_components_micro_results.json`

Sub-step breakdown (same run; per frame_pair):

| substep | latency_ms_per_unit | comment |
|---|---:|---|
| load_frames_only | 0.0074 | IO/cache overhead (negligible) |
| feature_histogram_diff_only | 3.92 | cheap visual diff |
| feature_ssim_only | 30.31 | heavy; controlled by `ssim_max_side` |
| feature_farneback_flowmag_only | 14.80 | heavy; controlled by `flow_max_side` (or reuse `core_optical_flow`) |
| postprocess_only(cached_features) | 0.0076 | negligible |

Notes:
- `cut_detection` поддерживает reuse motion curve от `core_optical_flow/flow.npz`:
  - `--prefer-core-optical-flow` (best-effort)
  - `--require-core-optical-flow` (fail-fast)
- Если `core_optical_flow.frame_indices` не совпадает с `cut_detection.frame_indices`, reuse запрещён (иначе нарушим контракт времени/оси).
- `gpu_memory_per_task_mb` для `cut_detection` пока `N/A` (у нас в micro-bench VRAM смешан с GUI процессами; для корректной цифры надо измерять RSS/VRAM в “чистом” окружении или через отдельный probe процесса).

#### 2.2 Policy: выбор preset качества для `cut_detection` (MVP, v1)

Цель: дать scheduler’у **детерминированное правило**, какой preset применять по умолчанию,
а `fast` оставлять как режим “сжать время ценой качества”.

Рекомендуемое правило (v1):
- **S ≤ 320** → `quality` (влияние на время небольшое, качество максимальное)
- **320 < S ≤ 720** → `default`
- **S > 720** → `default`, но scheduler может переключить на `fast` при перегрузе (throughput emergency)

Где S = short side bucket (как в baseline сетке).

##### Matrix sweep (12 resolutions × 2 aspects × 3 quality presets)

Прогон завершён: **72/72**. Агрегированный файл:
- `docs/baseline/out/cut_detection_matrix_final_20260109-065203/hard/matrix_results.json`
Машиночитаемая таблица стоимости (для scheduler’а):
- `docs/models_docs/resource_costs/cut_detection_costs_v1.json`
  - soft cuts: `docs/models_docs/resource_costs/cut_detection_soft_costs_v1.json`
  - motion cuts: `docs/models_docs/resource_costs/cut_detection_motion_costs_v1.json`

Профили качества (knobs):
- `quality`: `ssim_max_side=640`, `flow_max_side=384`
- `default`: `ssim_max_side=512`, `flow_max_side=320`
- `fast`: `ssim_max_side=384`, `flow_max_side=256`, `hard_cuts_cascade=on` (hist-gated; см. `run_cut_detection_matrix.py` preset syntax)

Сводка по latency (stable mean, **ms / frame_pair**) по 12 short-side bucket’ам:

| aspect | preset | min | p50 | p95 | max |
|---|---|---:|---:|---:|---:|
| 16:9 | fast | 3.90 | 7.68 | 15.39 | 16.18 |
| 16:9 | default | 16.80 | 47.35 | 56.81 | 57.60 |
| 16:9 | quality | 16.98 | 53.69 | 61.84 | 62.08 |
| 9:16 | fast | 4.08 | 7.58 | 15.22 | 16.17 |
| 9:16 | default | 17.25 | 36.43 | 45.71 | 46.54 |
| 9:16 | quality | 17.52 | 51.91 | 60.78 | 60.97 |

Примечание:
- Это synthetic benchmark (шумовые кадры), поэтому абсолютные значения “детектора” по качеству не оцениваются, но **профиль производительности** и зависимость от resolution/knobs — репрезентативны.

##### Quality impact (real videos, analysis downscale)

Мы отдельно проверили, как presets расходятся **на реальных видео** при сниженной analysis timeline
(`analysis_width=360`, `analysis_height=640`). Сравнение делали по hard cuts по времени (tolerance по dt).

Артефакты отчётов:
- `NSumhkOwSg.mp4` (all frames): `docs/baseline/out/cut_detection_quality_eval/quality_report_NSumhkOwSg_allframes_downscale360x640_ref_default.json`
  - ref=`default`: `quality` vs `default` **F1=1.000**, `fast` vs `default` **F1=1.000**
- `-3Mbinqzig4.mp4` + `-3s8SdV4bsU.mp4` (all frames): `docs/baseline/out/cut_detection_quality_eval/quality_report_2videos_allframes_downscale360x640_ref_default.json`
  - `-3Mbinqzig4`: `quality` vs `default` **F1=0.929** (fp=2), `fast` vs `default` **F1=0.889** (fp=2, fn=1)
  - `-3s8SdV4bsU`: `quality` vs `default` **F1=0.948** (fp=3, fn=2), `fast` vs `default` **F1=0.926** (fp=3, fn=4)

Вывод (MVP):
- `quality` почти всегда близок к `default`, обычно “добавляет” немного extra cuts.
- `fast` чаще начинает терять часть cuts относительно `default` (FN), поэтому его лучше включать только как emergency/throughput режим.

Рекомендации (без изменения policy):
- **Для продакшена лучше считать `default` “опорным”**: на реальных видео он стабильнее, а `quality/fast` можно оценивать как отклонение от него.
- **`quality` безопаснее, чем `fast`**: чаще даёт FP (лишние cuts), но реже теряет (FN). Это обычно предпочтительнее, если downstream может пережить “лишние” границы.
- **`fast` включать только при явной перегрузке**: когда важнее throughput, и допустимы FN/сдвиги cut’ов.
- **Quality eval всегда проводить на downscale-analysis**, а не на full-res: это быстрее и ближе к реальному baseline routing (всё равно большинство core-моделей работают на фиксированных ветках).
- **Если F1 падает**: первым делом поднимать `flow_max_side`, затем `ssim_max_side` (flow чаще “ломает” recall при сильном downscale).

---

### 3) Cut detection — остальные логические части (следующие после hard_cuts)

#### 3.1 Soft cuts (`detect_soft_cuts`)

- **component**: `cut_detection.detect_soft_cuts(cpu)`
- **unit**: `frame_pair`
- **parameters impacting cost**:
  - `flow_max_side` (downscale для Farneback)
- **cost table (12 short-side bucket’ов × 2 aspect × 3 preset)**:
  - matrix: `docs/baseline/out/cut_detection_matrix_final_20260109-065203/soft/matrix_results.json`
  - machine-readable: `docs/models_docs/resource_costs/cut_detection_soft_costs_v1.json`
 - **quality impact (subset, downscale)**:
  - report: `docs/baseline/out/cut_detection_quality_eval/quality_report_soft_2videos_subset180_downscale360x640_fps15_ref_default_20260109.json`

Сводка по latency (stable mean, **ms / frame_pair**) по 12 short-side bucket’ам:

| aspect | preset | min | p50 | p95 | max |
|---|---|---:|---:|---:|---:|
| 16:9 | fast | 10.38 | 14.75 | 29.48 | 33.52 |
| 16:9 | default | 13.51 | 21.02 | 33.78 | 36.51 |
| 16:9 | quality | 12.69 | 28.03 | 39.68 | 41.79 |
| 9:16 | fast | 10.18 | 14.81 | 29.61 | 33.61 |
| 9:16 | default | 13.14 | 20.92 | 33.46 | 35.61 |
| 9:16 | quality | 12.93 | 27.67 | 40.53 | 42.64 |

#### 3.2 Motion-based cuts (`detect_motion_based_cuts`)

- **component**: `cut_detection.detect_motion_based_cuts(cpu)`
- **unit**: `frame_pair`
- **parameters impacting cost**:
  - `flow_max_side` (override low-res flow)
- **cost table (12 short-side bucket’ов × 2 aspect × 3 preset)**:
  - matrix: `docs/baseline/out/cut_detection_matrix_final_20260109-065203/motion/matrix_results.json`
  - machine-readable: `docs/models_docs/resource_costs/cut_detection_motion_costs_v1.json`
 - **quality impact (subset, downscale)**:
  - report: `docs/baseline/out/cut_detection_quality_eval/quality_report_motion_2videos_subset180_downscale360x640_fps15_ref_default_20260109.json`

Сводка по latency (stable mean, **ms / frame_pair**) по 12 short-side bucket’ам:

| aspect | preset | min | p50 | p95 | max |
|---|---|---:|---:|---:|---:|
| 16:9 | fast | 25.79 | 53.81 | 155.73 | 183.21 |
| 16:9 | default | 35.27 | 64.46 | 165.50 | 194.06 |
| 16:9 | quality | 48.20 | 75.50 | 178.29 | 206.39 |
| 9:16 | fast | 25.56 | 54.69 | 156.70 | 185.31 |
| 9:16 | default | 34.64 | 64.68 | 206.89 | 283.04 |
| 9:16 | quality | 47.33 | 76.67 | 178.51 | 205.46 |


