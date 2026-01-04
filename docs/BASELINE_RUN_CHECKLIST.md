# Baseline run checklist (перед первым прогоном)

Этот чеклист нужен, чтобы “с самого начала” убедиться, что правила/контракты соблюдены и первый прогон не превратится в набор случайных артефактов.

## 1) Run identity (единая идентичность прогона)

Перед запуском убедиться, что orchestrator формирует и прокидывает:
- `platform_id`
- `video_id`
- `run_id`
- `sampling_policy_version`
- `config_hash` (**одинаковый для Segmenter/Visual/Audio/Text в рамках одного run**)

Где это должно жить:
- `frames_dir/metadata.json` (от Segmenter)
- `result_store/.../manifest.json` → `run.*`
- `meta` внутри каждого NPZ-артефакта

## 2) Segmenter / frames_dir contract

В `frames_dir` должен быть `metadata.json`, и он должен соответствовать `docs/SEGMENTER_CONTRACT.md`:

- **Union-only frames**:
  - `total_frames == len(union_*)`
  - `frames_dir` содержит **только union кадры**, а не все кадры видео
- **Union-domain индексы**:
  - `metadata[component].frame_indices` — это индексы **в union timeline** (0..N-1)
- **RGB contract**:
  - `color_space == "RGB"`
  - `FrameManager.get()` возвращает RGB; если кому-то нужен BGR — конвертация локально в компоненте
- **Batch storage**:
  - `batch_size` (или legacy `chunk_size`) и `batches[]`

## 3) No-fallback policy

Для core/providers и modules:
- если нет `metadata[component].frame_indices` → компонент обязан `raise`
- если нет обязательного dependency artifact → компонент обязан `raise`

Исключения:
- “пустота” данных — допустима только как **valid empty** (см. следующий пункт)

## 4) Empty outputs (валидная пустота)

Если данных “нет” (например, нет аудио, нет лиц, нет текста):
- компонент пишет NPZ со `status="empty"` и `empty_reason`
- численные массивы содержат `NaN`
- есть булевые маски присутствия (`*_present`)

## 5) result_store / manifest

Структура per-run:
- `result_store/<platform_id>/<video_id>/<run_id>/manifest.json`
- `result_store/<platform_id>/<video_id>/<run_id>/<component>/...*.npz`

Manifest:
- должен **мерджиться** между стадиями (Audio → Text → Visual)
- запись должна быть атомарной

## 6) NPZ meta минимальный контракт

В каждом NPZ `meta` должны быть поля из `docs/ARTIFACTS_AND_SCHEMAS.md`:
- `producer`, `producer_version`, `schema_version`
- `created_at`
- `platform_id`, `video_id`, `run_id`
- `config_hash`, `sampling_policy_version`
- `status`, `empty_reason`

Примечание:
- Runtime валидатор может быть мягче, но baseline-данные должны быть полными.

## 7) Параллелизм (baseline-safe)

Baseline safe defaults:
- 1 видео за раз (внешний orchestrator)
- внутри одного видео:
  - можно параллелить модули CPU
  - GPU-задачи ограничивать `gpu_max_concurrent` (обычно 1 на 6GB)

## 9) Venv split for conflicting deps (example: MediaPipe)

Некоторые core провайдеры могут требовать отдельную виртуальную среду из-за конфликтов зависимостей.
Пример: `core_face_landmarks` (MediaPipe).

Правило:
- В `VisualProcessor/config.yaml` для такого core можно указать `venv_path`.
- Оркестратор VisualProcessor использует этот venv **только для выбора интерпретатора** и **не прокидывает** `venv_path` как CLI аргумент.

Рекомендация:
- держать `requirements.txt` рядом с core для воспроизводимого создания venv.

## 10) MediaPipe GPU note (чтобы не путаться)

Текущая реализация `core_face_landmarks` использует legacy API `mediapipe.solutions.*`.
В этом режиме (на типичных Linux desktop установках) inference обычно выполняется **на CPU** (TFLite + XNNPACK),
и это нормально/ожидаемо. Наличие EGL/GL логов само по себе не означает GPU inference.

Если потребуется GPU inference:
- нужно делать отдельный core на базе **MediaPipe Tasks API** (новая архитектура) и/или отдельной сборки,
- и отдельно оговаривать формат выходного NPZ (или сохранять тот же контракт).
## 8) Что считается “готово к первому прогону”

Минимум:
- Segmenter пишет корректный `frames_dir/metadata.json` (union + RGB + per-component indices)
- VisualProcessor пишет per-run `manifest.json` и хотя бы 1 NPZ artifact
- (опционально) AudioProcessor пишет per-run NPZ и апдейтит общий manifest


