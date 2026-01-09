# Оставшиеся задачи для масштабной обработки baseline

## Статус Milestones

### ✅ Завершено
- **M0**: Segmenter + frames_dir (union sampling) - ✅
- **M1**: Артефакты/manifest/schema validator - ✅
  - `artifact_validator.py` - проверка контрактов
  - `quality_validator.py` - проверка качества фичей
- **M2**: Tier-0 Visual модули - ✅
  - Все модули интегрированы: `cut_detection`, `shot_quality`, `video_pacing`, `story_structure`, `similarity_metrics`, `text_scoring`, `uniqueness`, `scene_classification`
- **M3**: Tier-0 Audio extractors - ✅
  - `clap_extractor`, `tempo_extractor`, `loudness_extractor` работают

### ⏳ Осталось

## 1) M4 — Dataset Builder (приоритет: ВЫСОКИЙ)

### 1.1 Текущее состояние
- ✅ Есть `DatasetBuilder/build_training_table.py` - собирает feature table из manifest.json + NPZ
- ❌ Нет targets (multi-horizon deltas)
- ❌ Нет enrichment (channel_id)
- ❌ Нет temporal features (video_age_hours_at_snapshot1, duration_sec, fps)

### 1.2 Что нужно сделать

#### A) Добавить Targets (multi-horizon deltas)
**Файл**: `DatasetBuilder/add_targets.py` (новый)

Задачи:
1. Читать snapshot метаданные (snapshot1, snapshot2, snapshot3)
2. Вычислять deltas:
   - `delta_7d = snapshot2 - snapshot1` (optional, с mask)
   - `delta_14d = snapshot3 - snapshot1` (обязательный)
   - `delta_21d = snapshot4 - snapshot1` (обязательный, если есть snapshot4)
3. Применять `log1p` к deltas
4. Создавать masks для missing targets (7d может отсутствовать)

**Входные данные**:
- Нужны snapshot метаданные (views, likes, comments, etc.) на разные моменты времени
- Источник: `Interpret/main_ready/` или отдельный файл с snapshots

**Выход**:
- Добавить колонки `target_*_7d`, `target_*_14d`, `target_*_21d` в training table
- Добавить `mask_*_7d` для optional targets

#### B) Добавить Enrichment (channel_id)
**Файл**: `DatasetBuilder/enrichment.py` (новый)

Задачи:
1. Обогатить `video_id` → `channel_id` через YouTube API или существующий индекс
2. Добавить channel stats (если доступны):
   - `channel_subscriber_count`
   - `channel_total_videos`
   - `channel_created_at`
3. Сохранить в training table

**Входные данные**:
- `video_id` из manifest.json
- YouTube API или локальный индекс channel_id

#### C) Добавить Temporal Features
**Файл**: Модифицировать `DatasetBuilder/build_training_table.py`

Задачи:
1. Извлекать из manifest.json или video metadata:
   - `video_age_hours_at_snapshot1` (критично для split)
   - `duration_sec` (из audio или video metadata)
   - `fps` (analysis_fps из frames_dir/metadata.json)
   - `language` (если есть в metadata)
   - `category` (если есть в metadata)
2. Добавить в training table

#### D) Интеграция
**Файл**: `DatasetBuilder/build_full_dataset.py` (новый orchestrator)

Задачи:
1. Вызвать `build_training_table.py` → feature table
2. Вызвать `add_targets.py` → добавить targets
3. Вызвать `enrichment.py` → добавить channel_id и stats
4. Объединить всё в финальный dataset (parquet/csv)
5. Сохранить metadata dataset (версии, config_hash, etc.)

### 1.3 Acceptance criteria
- ✅ Генерация датасета детерминирована (reproducible)
- ✅ Нет leakage: фичи только snapshot1/артефакты; таргеты только future snapshots
- ✅ Все required targets присутствуют (14d, 21d)
- ✅ Optional targets имеют masks (7d)

---

## 2) M5 — Обучение Baseline (CatBoost/LightGBM)

### 2.1 Что нужно сделать

#### A) Training Pipeline
**Файл**: `Training/train_baseline.py` (новый)

Задачи:
1. Загрузка training table (parquet/csv)
2. Split:
   - Time-based: `video_age_hours_at_snapshot1 < threshold` → train, иначе → val/test
   - Channel-group: не допускать leakage между train/val/test (один channel только в одном split)
3. Feature engineering:
   - Обработка missing values (NaN → 0 или median)
   - Feature selection (опционально, для baseline можно все)
4. Обучение:
   - Multi-target: `views`, `likes`, `comments` (или что есть в targets)
   - Multi-horizon: отдельные модели для 7d/14d/21d или multi-head
   - CatBoost или LightGBM
5. Метрики:
   - MAE/RMSE на `log1p` scale
   - Spearman correlation
   - Отчёт по `video_age_hours_at_snapshot1` buckets (0-24h, 24-48h, 48-72h, etc.)

#### B) Reproducibility
**Файл**: `Training/config.yaml` (новый)

Задачи:
1. Фиксировать:
   - `seed` (random seed)
   - `commit_hash` (если доступно git)
   - `dataprocessor_version`
   - `sampling_policy_version`
   - `schema_version` для каждого компонента
   - `config_hash` из runs
2. Сохранять в `Training/artifacts/<run_id>/config.yaml`

#### C) Model Artifacts
**Файл**: `Training/save_model.py` (новый)

Задачи:
1. Сохранять:
   - Модель (CatBoost/LightGBM pickle)
   - Feature names (для inference)
   - Config (для reproducibility)
   - Metrics report (JSON)
2. Структура: `Training/artifacts/<run_id>/`

### 2.2 Acceptance criteria
- ✅ Есть baseline модель + сохранённый артефакт модели + конфиг
- ✅ Есть отчёт качества overall + buckets
- ✅ Reproducibility: можно переобучить с теми же результатами

---

## 3) M6 — Inference Pipeline

### 3.1 Что нужно сделать

#### A) Feature Extraction для Inference
**Файл**: `Inference/extract_features.py` (новый)

Задачи:
1. Читать NPZ артефакты из `result_store/<platform_id>/<video_id>/<run_id>/`
2. Построить feature vector как в training (используя те же агрегаты)
3. Обработать missing values (как в training)
4. Вернуть feature vector для модели

#### B) Prediction
**Файл**: `Inference/predict.py` (новый)

Задачи:
1. Загрузить модель из `Training/artifacts/<run_id>/`
2. Загрузить feature vector через `extract_features.py`
3. Сделать прогноз (multi-target, multi-horizon)
4. Применить inverse `log1p` для получения реальных значений
5. Вернуть predictions dict

#### C) Presentation Layer (JSON)
**Файл**: `Inference/render_json.py` (новый)

Задачи:
1. Формировать JSON для backend/frontend детерминированно
2. Структура:
   ```json
   {
     "video_id": "...",
     "run_id": "...",
     "predictions": {
       "views": {
         "7d": {"value": 1234, "confidence": 0.8},
         "14d": {"value": 5678, "confidence": 0.85},
         "21d": {"value": 9012, "confidence": 0.82}
       },
       "likes": {...},
       "comments": {...}
     },
     "features_used": {...},
     "model_info": {...}
   }
   ```
3. (Опционально) LLM текст-only поверх render-context (см. `docs/LLM_RENDERING.md`)

### 3.2 Acceptance criteria
- ✅ Для одного run можно получить прогноз и итоговый JSON
- ✅ Feature extraction использует те же агрегаты что и training
- ✅ JSON детерминирован (reproducible)

---

## 4) Масштабная обработка (Batch Processing)

### 4.1 Текущее состояние
- ✅ Есть `main.py` - orchestrator для одного видео
- ✅ Есть `BatchRunner/run_batch.py` - но нужно проверить интеграцию

### 4.2 Что нужно сделать

#### A) Batch Orchestrator
**Файл**: `BatchProcessor/process_batch.py` (новый или модифицировать `BatchRunner/run_batch.py`)

Задачи:
1. Читать список видео (CSV/JSON) с метаданными:
   - `video_path` или `video_id` (для скачивания)
   - `platform_id`
   - `video_id`
   - Опционально: `channel_id`, `snapshot1_meta`, etc.
2. Для каждого видео:
   - Вызвать `main.py` с правильными параметрами
   - Обработать ошибки (retry логика)
   - Сохранить статус в `BatchProcessor/state.jsonl`
3. Параллелизм:
   - По умолчанию: последовательно (GPU ограничен)
   - Опционально: параллельно несколько видео (если GPU позволяет)
4. Мониторинг:
   - Progress bar
   - Логирование в файл
   - Статистика: успешно/ошибки/пустые

#### B) Error Handling & Retry
**Файл**: Модифицировать `main.py` или создать wrapper

Задачи:
1. Retry логика:
   - Transient errors (OOM, network) → retry 2-3 раза
   - Permanent errors (corrupted video) → skip
2. Graceful degradation:
   - Если компонент упал → пометить run как error
3. Recovery:
   - Если run частично завершён → resume (проверить manifest.json)

#### C) Мониторинг и Логирование
**Файл**: `BatchProcessor/monitor.py` (новый)

Задачи:
1. Progress tracking:
   - `BatchProcessor/progress.json` - текущий статус
   - `BatchProcessor/state.jsonl` - история всех runs
2. Метрики:
   - Время обработки на видео
   - Успешность компонентов
   - GPU/CPU использование
3. Алерты:
   - Если >10% ошибок → предупреждение
   - Если GPU OOM → предупреждение

#### D) Оптимизация производительности
**Файл**: Модифицировать существующие компоненты

Задачи:
1. Кэширование:
   - Проверять `config_hash` → если run уже существует → skip
   - Idempotency по `(platform_id, video_id, run_id, config_hash)`
2. Batch processing внутри компонентов:
   - Уже есть `auto_batch_size()` для core providers
   - Убедиться что все компоненты используют батчинг эффективно
3. Параллелизм:
   - Intra-video: уже есть (modules параллельно)
   - Inter-video: добавить опцию для параллельной обработки нескольких видео

### 4.3 Acceptance criteria
- ✅ Можно обработать 100+ видео без ручного вмешательства
- ✅ Ошибки обрабатываются gracefully (retry, skip, resume)
- ✅ Есть мониторинг прогресса и статистика

---

## 5) Приоритеты для запуска масштабной обработки

### Критично (блокирует масштабную обработку):
1. **M4.1.A** - Targets (multi-horizon deltas) - БЕЗ ЭТОГО НЕЛЬЗЯ ОБУЧАТЬ
2. **M4.1.C** - Temporal features (video_age_hours_at_snapshot1) - НУЖНО ДЛЯ SPLIT
3. **4.2.A** - Batch orchestrator - НУЖНО ДЛЯ МАСШТАБНОЙ ОБРАБОТКИ

### Важно (нужно для полного baseline):
4. **M4.1.B** - Enrichment (channel_id) - УЛУЧШАЕТ КАЧЕСТВО
5. **M5** - Обучение baseline - ЦЕЛЬ BASELINE
6. **M6** - Inference - НУЖНО ДЛЯ ИСПОЛЬЗОВАНИЯ

### Опционально (можно отложить):
7. **4.2.B** - Error handling & retry - УЛУЧШАЕТ НАДЁЖНОСТЬ
8. **4.2.C** - Мониторинг - УЛУЧШАЕТ ОПЫТ
9. **4.2.D** - Оптимизация - УЛУЧШАЕТ ПРОИЗВОДИТЕЛЬНОСТЬ

---

## 6) Минимальный план для первого масштабного запуска

### Шаг 1: Dataset Builder (M4)
1. Добавить temporal features в `build_training_table.py`
2. Создать `add_targets.py` для multi-horizon deltas
3. Протестировать на smoke27 + несколько других runs

### Шаг 2: Batch Processing (4.2.A)
1. Создать `BatchProcessor/process_batch.py` который:
   - Читает список видео
   - Вызывает `main.py` для каждого
   - Сохраняет статус
2. Протестировать на 10-20 видео

### Шаг 3: Обучение (M5)
1. Создать `Training/train_baseline.py`
2. Обучить на собранном dataset
3. Оценить метрики

### Шаг 4: Inference (M6)
1. Создать `Inference/predict.py`
2. Протестировать на новых видео

---

## 7) Полезные команды для проверки

```bash
# 1. Проверить что все runs валидны
python VisualProcessor/utils/artifact_validator.py <run_dir>
python VisualProcessor/utils/quality_validator.py <run_dir>

# 2. Собрать feature table (без targets)
python DatasetBuilder/build_training_table.py \
  --rs-base _runs/result_store \
  --out-csv dataset/features.csv

# 3. Обработать batch видео (после создания BatchProcessor)
python BatchProcessor/process_batch.py \
  --video-list videos.csv \
  --rs-base _runs/result_store \
  --config VisualProcessor/config.yaml
```

