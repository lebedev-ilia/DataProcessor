## Baseline component/model checklist (measurements)

Цель: зафиксировать **канонический чек‑лист** baseline и собрать измерения:
- **Latency** (per-frame, 10 повторов; если есть выбросы — считаем mean по “стабильным” и помечаем, что есть spikes)
- **GPU VRAM** (per-process `tritonserver` + delta на время прогона)
- **CPU RAM peak** (RSS peak)

Принципиальное решение про “разрешение входа”:
- **Вход чек‑листа = исходный размер кадра (source WxH)**.
- Компонент внутри делает **routing** и **resize/crop/letterbox** под фиксированную ветку модели.
- В отчёте фиксируем **оба**:
  - `source_resolution` (WxH)
  - `selected_branch` (например `S=224/336/448/256/384/512/320/640`)

---

### Где лежат данные

- **Данные (таблица результатов)**: `docs/baseline/BASELINE_COMPONENT_MODEL_RESULTS.md`
- **Сырые артефакты прогонов**: `docs/baseline/out/...` (рекомендуемый путь) или явный `--out-dir`
- **Рендер таблицы из JSON (рекомендуется вместо ручного append)**:
  - `scripts/baseline/render_checklist_results_md.py`

---

### 0) Протокол измерений (обязательно)

- **Режим**: micro (per-frame)
- **Повторы** (repeats): по умолчанию 10 (на один run)
- **Warmup**: по умолчанию 1 (до измерений)
- **Runs**: по умолчанию 1 (повторяем блок warmup+repeats для устойчивости)
- Рекомендуемые пресеты:
  - `--profile fast` (дефолт): warmup=1, repeats=10, runs=1
  - `--profile stable`: warmup=3, repeats=30, runs=3 *(рекомендуется для финальных чисел / сравнения моделей)*
  - `--profile debug`: как fast, но пишет time-series по ресурсам (для расследования spikes)
- **Измеряем**:
  - `latency_ms_mean_stable` (+ `spikes: true/false`)
  - распределение latency: `p50/p95/p99`, `min/max`, `std`, `cv`
  - `spike_fraction` + `spikes_rule` (детектор spikes фиксирован и воспроизводим)
  - `cpu_rss_peak_mb` (peak на время измерения, RSS процесса раннера)
  - **GPU VRAM (Triton-aware)**:
    - `vram_triton_before_*_mb` / `vram_triton_peak_mb` / `vram_triton_after_*_mb`
    - `vram_triton_delta_run_mb`: **peak-before delta** (пик за run минус “before” на старте run)
    - `vram_triton_drift_mb`: “уплывание” памяти за весь блок (помогает понять, когда нужен restart Triton)
  - `status`: ok/error (+ error text)
  - `selected_model_spec` / `triton_model_name` / `model_signature` (через `models_used[]`, где применимо)
- **Ограничения 6GB**:
  - Если требуется рестарт Triton между группами/компонентами — это **явно фиксируется** в чек‑листе.

Важно про интерпретацию VRAM:
- Большая “base” величина (например 4–5GB) — это **нормально**: Triton держит загруженные модели + ORT CUDA memory pools/arenas и кеши.
- Поэтому для “памяти прогона” используем именно **delta** (`peak-before`), но в результаты записываем **только** `vram_triton_delta_run_mb`.
- Для честных сравнений на 6GB предпочтительно запускать на “чистом” Triton после рестарта.

Технически, VRAM измеряем так:
- Предпочтительно через **NVML** (если доступен `pynvml`), иначе fallback на `nvidia-smi`.
- Трекаем **max-over-run** (polling) и считаем delta относительно “before”.

---

### 1) Сетка входов (Visual)

Решение: baseline входы для Visual делятся на 2 формата:
- **16:9 (landscape)**
- **Shorts/TikTok (portrait 9:16)**

Сетка задаётся по **short side S** (в пикселях), а (W,H) вычисляется так:
- **16:9**: \(H = S\), \(W = round(S * 16 / 9)\)
- **9:16**: \(W = S\), \(H = round(S * 16 / 9)\)

Кандидатный список (черновик, можно поправить):
- `S`: 128, 160, 224, 256, 280, 320, 384, 448, 512, 640, 720, 768, 896, 960, 1080

---

### 2) Состав baseline (to be filled)

#### VisualProcessor — core providers (все core)

- `core_clip` (Triton branches: 224/336/448)
- `core_depth_midas` (Triton branches: 256/384/512)
- `core_optical_flow` (RAFT branches: 256/(384?)/(512?))
- `core_object_detections` (YOLO branches: 320/640/(960?))
- `core_face_landmarks` (inprocess)

#### VisualProcessor — modules (6 шт, включая scene_classification)

User-provided list (currently 7 items; confirm if нужно ровно 6):
- `cut_detection`
- `optical_flow`
- `scene_classification`
- `shot_quality`
- `story_structure`
- `uniqueness`
- `video_pacing`

#### AudioProcessor — components (3 шт)

User-provided list:
- `clap_extractor`
- `loudness_extractor`
- `tempo_extractor`

Input grid (Audio):
- единица обработки: **segment** из Segmenter
- ось входа: **duration_sec**
- диапазон: **0.1 .. 20.0 сек**
- количество точек: **25**
- распределение: **равномерно** (по умолчанию линейно; если нужно log‑scale — явно фиксируем)

---

### 2.1) Routing → selected_branch (фиксируем в отчёте)

Правило routing для чек‑листа (по source resolution):
- Берём \(D = max(W, H)\).
- Пороговая логика (как в доках, примерный стандарт):
  - \(D \\le 320\) → **small**
  - \(D \\le 448\) → **medium**
  - \(D > 448\) → **large**

Соответствие веток (baseline fixed-shape):
- **CLIP image**: small=224, medium=336, large=448
- **Places365**: small=224, medium=336, large=448
- **MiDaS**: small=256, medium=384, large=512
- **RAFT**: small=256, medium=384, large=512 *(на 6GB large/medium могут быть нестабильны; фиксируем статус отдельно)*
- **YOLO11x**: small=320, medium=640, large=960 *(на 6GB large часто OOM)*

---

### 3) Таблица измерений (шаблон)

Рекомендуемый формат строки:
- `component`
- `source_resolution` (WxH) / для audio: `duration_sec`
- `selected_branch` (fixed S или N/A)
- `latency_ms_mean_stable` (+ `spikes`)
- `gpu_vram_peak_mb`
- `vram_triton_delta_run_mb`
- `cpu_rss_peak_mb`
- `status` / `error`
- `models_used[]` (ссылкой на `model_signature`)
- `notes` (например “требует restart Triton перед прогоном”)

---

### 4) Результаты (to be filled)

#### Visual core

#### Visual modules

#### Audio components


