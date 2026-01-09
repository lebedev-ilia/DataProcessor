## PR‑10: Full module audit — разбиение на части + единый чек‑лист

### Зачем дробим PR‑10

PR‑10 по определению включает аудит **всех модулей/процессоров** (Visual core+modules, Audio, Text, Segmenter) и закрытие пунктов из `DATAPROCESSOR_AUDIT.md`. Это слишком большой объём для “одного PR” — высокий риск пропусков и деградации качества.

Поэтому PR‑10 делим на **серии PR‑10.x**, где каждый PR закрывает конкретный слой системы и обновляет аудит/контракты.

---

### Инвентарь (что обязаны покрыть)

#### A) Segmenter (processor)
- `segmenter` (кадры/аудио/metadata контракт)

#### B) Visual core providers (Tier‑0 baseline)
Из `docs/reference/component_graph.yaml`:
- `core_clip`
- `core_face_landmarks`
- `core_object_detections`
- `core_depth_midas`
- `core_optical_flow`

#### C) Visual modules (Tier‑0 baseline)
Из `docs/reference/component_graph.yaml`:
- `cut_detection`
- `shot_quality`
- `scene_classification`
- `video_pacing`
- `uniqueness`
- `story_structure`

#### D) Audio (Tier‑0 baseline, опционально по профилю)
Из `docs/reference/component_graph.yaml`:
- `clap_extractor`
- `loudness_extractor`
- `tempo_extractor`

Дополнительно (non‑baseline, но **prod‑важно**; аудировано в PR‑10.4):
- `asr_extractor`
- `speaker_diarization_extractor`
- `emotion_diarization_extractor`
- `source_separation_extractor`
- `speech_analysis_extractor` (aggregator)

#### E) TextProcessor (в baseline может быть выключен профилем)
- `text_processor` (и embedding‑ветка, если включена)

#### F) “Non‑baseline” Visual modules (есть в репо, но не в baseline DAG)
Нужно **явно классифицировать**: включаем в baseline / оставляем как experimental / удаляем.
Примеры (по дереву `VisualProcessor/modules/`): `action_recognition`, `behavioral`, `color_light`, `detalize_face_modules`, `emotion_face`, `frames_composition`, `high_level_semantic`, `micro_emotion`, `optical_flow`, и др. (модуль `face_detection` удалён; face presence теперь даёт `core_face_landmarks`).

---

### Единый чек‑лист аудита для *каждого* компонента (core/module/extractor/processor)

#### 1) Контракт входов
- **Входные артефакты/файлы**: какие именно нужны (paths/keys), кто владелец.
- **Sampling**: использует ли строго `metadata[component].frame_indices` (no-fallback).
- **Config**: какие параметры читаются из cfg/profile; какие скаляры должны быть CLI‑флагами.

#### 2) Контракт выходов
- **Артефакты**: 1) имя файла, 2) schema_version, 3) ключевые массивы/поля.
- **NPZ meta**:
  - run identity: `platform_id`, `video_id`, `run_id`, `config_hash`, `sampling_policy_version`
  - `producer`, `producer_version`, `schema_version`
  - `models_used[]` + `model_signature` (если модель используется)
  - `status` + `empty_reason` (если empty допустим)

#### 3) Ошибки и empty
- Явные условия **empty** (валидный пустой результат) vs **error**.
- `error_code` taxonomy (минимум): `missing_dependency`, `model_load_failed`, `triton_unavailable`, `non_zero_exit`, `artifact_validation_failed`.

#### 4) Dependencies (DAG)
- Hard deps / soft deps / checkpoints (если есть ожидания).
- `docs/reference/component_graph.yaml` должен отражать реальность (и наоборот).

#### 5) State/Manifest
- Компонент корректно отражается в `manifest.json` и `state_*.json`:
  - `status`, `started_at`, `finished_at`, `duration_ms`
  - `schema_version`, `producer_version`
  - `device_used` (best-effort), `batch_size` (если применимо)

#### 6) Воспроизводимость и кэш
- Что входит в idempotency key (как минимум: `config_hash` + `sampling_policy_version` + `producer_version` + `schema_version` + `model_signature`).
- Если есть оптимизированные модели (PR‑9): `engine/precision/weights_digest` обязаны попадать в meta.

#### 7) Env / deps / portability
- Где живёт venv (общая/изолированная) и почему.
- Фиксация зависимостей: `requirements.txt` (где нужно), отсутствие “скрытых” импорта‑в‑рантайме.

---

### Разбиение PR‑10 на под‑PR’ы (как работаем)

- **PR‑10.0**: Audit harness + inventory + шаблоны/регламент
- **PR‑10.1**: Segmenter audit closure
- **PR‑10.2**: Visual core providers (Tier‑0) audit closure
- **PR‑10.3**: Visual modules (Tier‑0) audit closure
- **PR‑10.4**: Audio extractors audit closure
- **PR‑10.5**: TextProcessor audit closure
- **PR‑10.6**: Non‑baseline Visual modules: classification + minimal contract or quarantine
- **PR‑10.7**: Baseline DAG completion + consistency pass (graph ↔ actual usage ↔ profiles)

Каждый PR‑10.x имеет DoD:
- обновлены соответствующие секции в `docs/audits/DATAPROCESSOR_AUDIT.md` (PASS/PARTIAL + evidence + root cause + fix list),
- обновлён `docs/reference/component_graph.yaml` (если выявлены несоответствия),
- добавлены/обновлены профили (если нужно),
- минимальные “dry” проверки: валидация схем/meta (без e2e на тяжёлых моделях).


