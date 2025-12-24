## План рефакторинга под core‑провайдеры моделей

Этот документ описывает поэтапный план перехода VisualProcessor к архитектуре с единым слоем `core`‑моделей и модульным слоем аналитики, а также трекинг прогресса.

---

### 1. Цели и общая архитектура

Очень важно: У каждого семейства моделей в core должна быть своя виртуальная среда и запускаються они соответственно через subprocess в main.py 

**Цель:** все тяжёлые модели (Mediapipe, CLIP, MiDaS, RAFT, EmoNet, OpenFace, YOLO, VideoMAE, OCR и т.п.) должны запускаться **один раз на видео** в слое `core`, сохранять **сырые/универсальные фичи**, а все модульные пайплайны (`video_pacing`, `story_structure`, `behavioral`, `detalize_face_modules`, `text_scoring`, `uniqueness`, …) должны работать **только с уже сохранёнными данными + кадрами через `FrameManager`**, не инициализируя модели заново.

- **Слой 1 — core‑провайдеры** (`VisualProcessor/core/model_process/*`):  
  запускаются первыми, читают `frames_dir` + `metadata.json` от `Segmenter`, считают свои фичи и пишут их в `results_store/core_*`.
- **Слой 2 — модульные анализаторы** (`VisualProcessor/modules/*`):  
  последовательно читают результаты из `results_store` + при необходимости кадры через `FrameManager`, но **не создают модели**.

Прогресс:
- [x] Базовая структура `core/` создана.
- [x] Полный список семейств моделей и потребляющих модулей зафиксирован в этом плане.
- [x] Пайплайн `VisualProcessor/main.py` разделён на фазы core → modules.

---

### 2. Инвентаризация моделей и владельцев (что и где используется)

Задача: формально описать, **какие модели** есть и **какие модули** их используют напрямую (для понимания, какие core‑провайдеры нужны).

- **CLIP / high‑level визуальные эмбеддинги**
  - Используют: `high_level_semantic`, `story_structure`, `video_pacing`, `cut_detection`, `shot_quality` (CLIP часть), возможно другие.
  - Тип данных: per‑frame/per‑shot CLIP‑эмбеддинги, mean embedding, event/energy curves.
- **SentenceTransformer (текстовые эмбеддинги)**
  - Использует: `story_structure` (topics), потенциально `text_scoring`/`uniqueness`.
  - Тип данных: эмбеддинги субтитров / OCR‑фраз, topic‑кластеры.
- **Mediapipe (pose / hands / face mesh)**
  - Используют: `behavioral`, `detalize_face_modules`, частично `frames_composition`.
  - Тип данных: landmarks лица/позы/рук, bbox, базовые атрибуты по кадрам.
- **InsightFace (FaceAnalysis)**
  - Использует: `face_detection`.
  - Тип данных: bbox лиц, ID/quality фичи.
- **MiDaS (depth estimation)**
  - Используют: `frames_composition`, `shot_quality`.
- **RAFT / Farneback (optical flow)**
  - Используют: `optical_flow`, `video_pacing`, `story_structure`, `cut_detection`, `text_scoring`.
- **YOLO / OWL‑ViT / OWLv2 (object detection)**
  - Используют: `object_detection`, `frames_composition`.
- **EmoNet (face emotions)**
  - Использует: `emotion_face`.
- **OpenFace (через Docker)**
  - Использует: `micro_emotion`.
- **VideoMAE / action recognition модели**
  - Использует: `action_recognition`.
- **DnCNN / CBDNet / др. модели качества**
  - Использует: `shot_quality`.
- **OCR (EasyOCR / Tesseract)**
  - Использует: `text_scoring`.
- **Аудио‑эмбеддинги / energy**
  - Частично: `video_pacing`, внешние аудио‑модули.
  - **Примечание**: Аудио‑эмбеддинги будут приходить извне (не реализуется как core‑провайдер внутри VisualProcessor).

Прогресс:
- [x] Список семейств моделей собран.
- [x] Для каждого core провайдера описаны точные форматы выходов в `FEATURES_DESCRIPTION_core_*.md` файлах.

---

### 3. Целевая схема пайплайна (уровень процессов)

1. **`Segmenter`**  
   - Вход: исходный `video_path`.  
   - Выход:
     - `frames_dir` с кадрами и батчами,
     - `metadata.json` с:
       - `total_frames`, `fps`, `height`, `width`, `channels`,
       - `chunk_size`, `batches`,
       - `module_frame_indices` (какие кадры использовать каждому модулю).

2. **`VisualProcessor/main.py`**  
   Работает в две фазы:

   - **Фаза 1 — core‑провайдеры моделей**:
     - Читает `config.yaml` → список `core_providers`.
     - Последовательно запускает:
       - `core/model_process/clip_embeddings`,
       - `core/model_process/optical_flow`,
       - `core/model_process/face_landmarks`,
       - `core/model_process/depth_midas`,
       - `core/model_process/object_detections`,
       - `core/model_process/openface`,
       - `core/model_process/audio_embeddings` (когда появится).
     - Каждый провайдер:
       - Берёт кадры через `FrameManager` + `metadata.json`,
       - Считает свои сырые фичи,
       - Пишет в `results_store/core_<name>/...`.

   - **Фаза 2 — модульные анализаторы**:
     - Читает `config.yaml` → список `modules`.
     - Последовательно запускает:
       - `video_pacing`, `story_structure`, `behavioral`, `detalize_face_modules`,
         `text_scoring`, `similarity_metrics`, `uniqueness`, и др.
     - Модули:
       - Читают необходимые им `core_*` результаты и (опционально) результаты других модулей,
       - При необходимости получают кадры через `FrameManager`,
       - Не инициализируют модели, а только агрегируют/пост‑обрабатывают.

Прогресс:
- [x] `VisualProcessor/main.py` разделён на фазы core → modules.
- [x] `config.yaml` обновлён, чтобы задавать `core_providers` и `modules`.

---

### 4. Дизайн и структура core‑провайдеров

Цель: для каждого семейства моделей сделать отдельный провайдер в `core/model_process`, с чётким форматом выхода.

#### 4.1. Core CLIP / high-level embeddings

- Путь: `core/model_process/clip_embeddings/`.
- Вход: `--frames-dir`, `--rs-path`, `--model-name` (например, `ViT-B/32`), `--frame-indices` (опционально).
- Логика:
  - Загружает CLIP один раз.
  - Проходит по кадрам (по `frame_indices` из `metadata` или всем).
  - Считает:
    - `frame_embeddings`: массив `[N, D]` (можно downsample),
    - `mean_embedding`, `weighted_mean_embedding`,
    - при необходимости `semantic_energy_curve`.
- Выход (например NPZ/JSON): `results_store/core_clip/{video_id}.npz`.

#### 4.2. Core optical flow

- Путь: `core/model_process/optical_flow/`.
- Вход: `--frames-dir`, `--rs-path`, `--model-type` (`raft_large` / `raft_small` / `farneback`).
- Логика:
  - Считает оптический поток между последовательными кадрами.
  - Сохраняет:
    - `flow_mag_curve`: `[N-1]` средняя величина по кадру,
    - `flow_dir_curve`: `[N-1]` (усреднённый угол),
    - опциональные downsampled/агрегированные кривые.

#### 4.3. Core face landmarks (Mediapipe)

- Путь: `core/model_process/face_landmarks/`.
- Вход: `--frames-dir`, `--rs-path`, флаги `--use-pose`, `--use-hands`, `--use-face-mesh`.
- Логика:
  - Инициализирует Mediapipe (pose, hands, face_mesh) один раз.
  - Для каждого `frame_idx`:
    - Сохраняет:
      - список лиц с bbox, ключевыми landmark‑точками,
      - позу тела/рук, при необходимости gaze/ключевые углы.
- Выход: `results_store/core_face_landmarks/{video_id}.json`.

#### 4.4. Core depth (MiDaS), objects (YOLO/OWL), audio, OpenFace

- Аналогично: для MiDaS, YOLO/OWLViT, аудио‑эмбеддингов, OpenFace:
  - Выделить отдельные провайдеры в `core/model_process`.
  - Чётко описать формат:
    - depth: per‑frame карты или их summary,
    - objects: per‑frame bbox + классы + доверие,
    - audio: energy curve + embedding,
    - openface: per‑frame AU/pose/gaze + метаданные.

Прогресс:
- [x] Реализован core CLIP‑провайдер (MVP, per-frame embeddings в `core_clip/embeddings.npz`).
- [x] Реализован core optical_flow‑провайдер (RAFT + статистика в `optical_flow/statistical_analysis.json`).
- [x] Реализован core face_landmarks‑провайдер (Mediapipe pose/hands/face_mesh в `core_face_landmarks/landmarks.json`).
- [x] Реализован core depth_midas‑провайдер (статистика глубины в `core_depth_midas/depth.json`).
- [x] Реализован core object_detections‑провайдер (YOLO/OWL-ViT детекции в `core_object_detections/detections.json`).
- [ ] Реализован core openface провайдер (через Docker, для micro_emotion).
- [x] Для каждого провайдера есть мини‑`FEATURES_DESCRIPTION_core_*.md` с форматом данных.
- **Примечание**: `audio_embeddings` будет приходить извне, не реализуется как core‑провайдер.

---

### 5. Обновление модулей: переход на core‑данные

Для каждого модуля необходимо:

1. Определить, **какие модели он сейчас инициализирует сам**.
2. Определить, **какие core‑провайдеры** могут закрыть эти потребности.
3. Ввести слой `get_or_load_from_core(...)`:
   - Сначала попытаться прочитать данные из `results_store/core_*`,
   - При отсутствии — либо:
     - временно посчитать локально (и опционально сохранить в core‑формат),
     - либо (после полного перехода) падать с понятной ошибкой.
4. После стабилизации core‑пайплайна — удалить локальную инициализацию моделей.

#### 5.1. behavioral и detalize_face_modules (Mediapipe)

- Сейчас:
  - Оба модуля независимо инициализируют Mediapipe (pose/hands/face_mesh) и гоняют кадры, вытаскивая landmarks.
- План:
  - В обоих модулях вынести доступ к landmarks в helper:
    ```python
    def get_face_landmarks(frame_indices, rs_path):
        core_path = f"{rs_path}/core_face_landmarks"
        if exists(core_path):
            return load_core_face_landmarks(...)
        else:
            # временный fallback: локальный расчёт (текущая логика)
    ```
  - Переписать бизнес‑логику так, чтобы она работала с абстрактной структурой `landmarks_data`, не зная, откуда она взялась.
  - После того как `core_face_landmarks` будет всегда запускаться до этих модулей — удалить fallback.

Прогресс:
- [x] В `behavioral` добавлен helper для чтения core_face_landmarks и чтение core‑данных в `process_video`.
- [x] В `detalize_face_modules` добавлен helper для чтения core_face_landmarks (метод `_load_core_face_landmarks` и использование в `extract`).
- [x] Локальная инициализация Mediapipe в этих модулях удалена (используется только core_face_landmarks, без fallback).

#### 5.2. video_pacing и story_structure (CLIP + optical flow)

- Сейчас:
  - Инициализируют CLIP и считают optical flow (частично дублируя `high_level_semantic` и `optical_flow`).
- План:
  - Ввести helpers:
    ```python
    def get_clip_embeddings_or_core(rs_path): ...
    def get_optical_flow_or_core(rs_path): ...
    ```
  - Внутри этих функций:
    - приоритет: читать из `core_clip` и `core_optical_flow`,
    - временный fallback: считать локально и (по возможности) сохранить.
  - Логику определения шотов, energy‑кривых и кульминаций переписать на использование уже готовых кривых/эмбеддингов.
  - После стабилизации core‑слоя — удалить локальные `clip.load` и Farneback/RAFT‑инициализацию.

Прогресс:
- [x] video_pacing использует core_clip и core_optical_flow (локальная инициализация CLIP и Farneback удалена).
- [x] story_structure использует core_optical_flow и core_clip (локальная инициализация CLIP и Farneback удалена).

#### 5.3. text_scoring (OCR + motion/face/audio)

- Сейчас:
  - Сам вызывает OCR (EasyOCR/Tesseract), сам использует motion/face/audio сигналы.
- План:
  - Разделить:
    - OCR (можно временно оставить локально, либо позже вынести в `core_ocr`),
    - чтение motion/face/audio из core‑провайдеров.
  - Обновить `extract_features`, чтобы:
    - motion брался из `core_optical_flow`,
    - face‑сигналы — из `core_face_landmarks` или `face_detection`,
    - аудио‑energy — из `core_audio_embeddings` (когда появится).

Прогресс:
- [x] text_scoring читает motion/face/audio сигналы из core‑провайдеров (fallback удалён, используется только core).
- [ ] (опционально) OCR вынесен в отдельный core_ocr‑провайдер (OCR остаётся локальным, так как не является core провайдером).

#### 5.4. shot_quality, frames_composition, object_detection, scene_classification, action_recognition, emotion_face, micro_emotion

- Для каждого:
  - Проверить, какие модели можно вынести в core (MiDaS, YOLO, EmoNet, VideoMAE, OpenFace и др.).
  - Определить, нужны ли эти модели в нескольких местах, или модуль является единственным потребителем (в последнем случае можно оставить модель внутри, но зафиксировать формат выхода на будущее).

Прогресс:
- [x] Для каждого модуля составлен список core‑зависимостей (см. ниже).
- [x] `frames_composition`: внедрены чтение из `core_depth_midas` и `core_object_detections` (локальная инициализация YOLO и MiDaS удалена).
- [x] `shot_quality`: обновлён для использования `core_clip` (CLIP эмбеддинги изображений теперь читаются из core провайдера). MiDaS не используется в текущей реализации, поэтому не требует core_depth_midas.
- [x] `object_detection`: обновлён для использования `core_object_detections` при выборе YOLO (через аргументы `use_queries=False`). При выборе OWL-ViT (`use_queries=True`) модуль работает самостоятельно.
- [x] `scene_classification`: обновлён для опционального использования `core_clip` для семантических фичей (aesthetic, luxury, atmosphere).
- [x] `action_recognition`: единственный потребитель SlowFast/VideoMAE, оставлен как есть.
- [x] `action_recognition`: единственный потребитель SlowFast/VideoMAE, оставлен как есть.
- [x] `emotion_face`: единственный потребитель EmoNet, оставлен как есть.
- [x] `micro_emotion`: единственный потребитель OpenFace, оставлен как есть (core openface провайдер может быть реализован в будущем).

---

### 6. Обновление конфигурации и документации

Задачи:

1. **`config.yaml` VisualProcessor**
   - Добавить секции:
     ```yaml
     core_providers:
       - core_clip
       - core_optical_flow
       - core_face_landmarks
       # ...

     modules:
       - video_pacing
       - story_structure
       - behavioral
       - detalize_face_modules
       - text_scoring
       - similarity_metrics
       - uniqueness
       # ...
     ```
2. **Документация:**
   - В `docs/models_desc.md` и `docs/features_desc.md` добавить раздел про `core`‑слой и указать, какие модули используют какие провайдеры.
   - Добавить отдельный `FEATURES_DESCRIPTION_core_*.md` для каждого провайдера (форматы данных).

Прогресс:
- [x] `config.yaml` обновлён с разделами core_providers и modules (добавлены комментарии).
- [x] В `models_desc.md` добавлено описание core‑слоя.
- [x] Для каждого core‑провайдера описан формат фич (FEATURES_DESCRIPTION_core_*.md созданы).

---

### 7. Стратегия миграции и контроль качества

1. **MVP‑этап:**
   - Реализовать core‑провайдеры как тонкие обёртки над текущими реализациями внутри модулей (код можно переиспользовать).
   - В модулях добавить `get_or_load_from_core` без удаления старых путей.
   - Разделить `VisualProcessor/main.py` на две фазы, но допустить, что если core‑файла нет — модуль всё ещё может посчитать локально.

2. **Стабилизация:**
   - В интеграционных тестах/ручных запусках убедиться, что:
     - core‑слой отрабатывает корректно,
     - все модули успешно читают core‑данные и выдают те же (или лучше) фичи.

3. **Очистка:**
   - Постепенно удалять локальную инициализацию моделей из модулей, когда core‑слой гарантированно запускается.
   - Чистить `requirements.txt` модулей от неиспользуемых тяжёлых зависимостей.

4. **Мониторинг и версионирование:**
   - Для каждого core‑выхода добавлять `version`, `model_name`, `created_at`.
   - В потребителях валидировать совместимость версий и логировать предупреждения при несовпадении.
   - **Примечание**: Валидация версий в потребителях может быть добавлена в будущем при необходимости.

Прогресс:
- [x] Версионирование добавлено во все core провайдеры:
  - `core_clip`: добавлен `version` в embeddings.npz (уже были `model_name`, `created_at`)
  - `core_face_landmarks`: добавлен `model_name` = "Mediapipe" (уже были `version`, `created_at`)
  - `core_depth_midas`: есть `version`, `model_name`, `created_at`
  - `core_object_detections`: есть `version`, `created_at`, `model`, `model_family`
  - `optical_flow`: добавлены `model_type` и `model_name` в analysis_info (уже были `version`, `timestamp`/`created_at`)
- [x] MVP‑core провайдеры (optical_flow, core_clip, core_face_landmarks, depth_midas, object_detections) реализованы и интегрированы.
- [x] Все модули (`video_pacing`, `story_structure`, `behavioral`, `detalize_face_modules`, `text_scoring`, `frames_composition`) работают **только с core‑данными**, без fallback.
- [x] Локальная инициализация моделей в модулях удалена:
  - `behavioral`: удалён Mediapipe (pose/hands/face_mesh)
  - `detalize_face_modules`: удалён Mediapipe face_mesh
  - `video_pacing`: удалён CLIP и Farneback optical flow
  - `story_structure`: удалён CLIP, Farneback optical flow, Mediapipe face_mesh
  - `text_scoring`: удалён fallback для motion/face данных
  - `frames_composition`: удалены YOLO, MiDaS, MediaPipe face_mesh, torch, torchvision
  - `shot_quality`: CLIP эмбеддинги изображений теперь читаются из `core_clip` (локальная инициализация CLIP для изображений удалена)
  - `scene_classification`: опционально использует `core_clip` для семантических фичей (aesthetic, luxury, atmosphere) - если `rs_path` указан, использует core данные вместо локальной CLIP модели
  - `object_detection`: при выборе YOLO (`use_queries=False`) использует `core_object_detections` (локальная YOLO модель удалена). При выборе OWL-ViT (`use_queries=True`) работает самостоятельно.
- [x] Зависимости очищены из `requirements.txt` модулей:
  - `video_pacing`: удалён CLIP, удалён импорт torch
  - `story_structure`: удалён CLIP, удалён импорт torch и mediapipe
  - `behavioral`, `detalize_face_modules`, `frames_composition`: не имели requirements.txt (зависимости были в коде, теперь удалены)


