# Описание фичей модуля action_recognition

## Общее описание

Модуль action_recognition использует модель **SlowFast** (Meta AI) для распознавания действий в видео. SlowFast — это dual-pathway CNN архитектура, которая специально разработана для анализа движения и темпа в видео. Особенностью SlowFast является встроенная обработка motion через fast pathway, что делает внешнее вычисление motion magnitude избыточным.

Модуль извлекает два типа фичей:
1. **Sequence Features** — для VisualTransformer (временные последовательности embeddings)
2. **Aggregate Features** — для MLP/Tabular Head (агрегированные статистики)

## Используемые модели

Модуль использует SlowFast R50 из библиотеки torchvision:
- **slowfast_r50** — предобученная модель SlowFast ResNet-50
- Модель имеет два пути: slow pathway (для семантики) и fast pathway (для движения)
- Извлекаются embeddings размерности 2048d, которые проецируются в 256d для использования в VisualTransformer

## Структура выходных данных

Результаты модуля организованы в следующую структуру:
- `total_frames` — общее количество кадров, переданных в модуль
- `num_tracks` — количество обработанных треков (людей)
- `processing_params` — параметры обработки (clip_len, batch_size)
- `results` — результаты для каждого трека (ключ — ID трека)

---

## 1. total_frames

Общее количество кадров видео, переданных в модуль для обработки.

## 2. num_tracks

Количество обработанных треков (людей) в видео.

**Алгоритм**: Вычисляется как количество элементов в словаре `results`.

## 3. processing_params

Параметры обработки видео.

### 3.1. clip_len

Длина клипа в кадрах, используемая для обработки. По умолчанию 32 кадра (стандарт для SlowFast). SlowFast использует два пути: slow pathway (каждый 16-й кадр) и fast pathway (каждый 2-й кадр).

### 3.2. batch_size

Размер батча для inference модели. Влияет на скорость обработки и потребление памяти.

---

## 4. results

Результаты распознавания действий для каждого трека. Ключ — ID трека (строка), значение — словарь с фичами.

### 4.1. sequence_features

Временные последовательности фичей для VisualTransformer. Содержит:

#### 4.1.1. embedding_normed_256d

Массив L2-нормализованных embeddings для каждого клипа. Формат: `[num_clips, 256]`.

**Алгоритм**: 
```python
# Извлечение features из SlowFast (2048d)
raw_embeddings = slowfast_model.forward_features(clips)  # [num_clips, 2048]
# Проекция в 256d
embeddings_raw_256d = linear_projection(raw_embeddings)  # [num_clips, 256]
# L2 нормализация для VisualTransformer
norms = ||embeddings_raw_256d||_2
embedding_normed_256d = embeddings_raw_256d / norms  # [num_clips, 256]
```

**Использование**: Подается в VisualTransformer для обучения временных паттернов. Содержит информацию о ритме, жестах и темпе действий. **Важно**: Используются нормализованные embeddings (||e|| = 1), чтобы избежать влияния масштаба.

#### 4.1.2. temporal_diff_normalized

Массив нормализованных временных различий между соседними клипами, основанный на косинусной дистанции. Формат: `[num_clips]`.

**Алгоритм**: 
```python
temporal_diff_normalized[0] = 0.0
for i in range(1, num_clips):
    # Косинусная дистанция: 1.0 - cosine_similarity(e_t, e_{t-1})
    temporal_diff_normalized[i] = 1.0 - cosine_similarity(
        embedding_normed[i],
        embedding_normed[i - 1],
    )
```

**Диапазон**: \[0.0, 2.0]

**Использование**: Показывает изменение действий между клипами. Косинусная форма:
- устойчива к масштабу,
- интерпретируема (0 = одинаково, 1 ≈ ортогонально, 2 = противоположно),
- лучше работает в attention-механизмах.

---

### 4.2. CORE DYNAMICS

#### 4.2.1. mean_embedding_norm_raw

Средняя норма **raw** embeddings (до L2 нормализации) по всем клипам. Показывает общую "энергию" действий.

**Алгоритм**: 
```python
# Используются raw embeddings (до нормализации)
embedding_norms_raw = [||e_raw||_2 for e_raw in raw_embeddings]
mean_embedding_norm_raw = mean(embedding_norms_raw)
```

**Диапазон**: 0.0 — inf (обычно 0.5 — 2.0)

**Важно**: Используются raw embeddings, а не нормализованные, иначе норма всегда ≈ 1.

#### 4.2.2. std_embedding_norm_raw

Стандартное отклонение норм **raw** embeddings. Показывает вариативность "энергии" действий.

**Алгоритм**: 
```python
embedding_norms_raw = [||e_raw||_2 for e_raw in raw_embeddings]
std_embedding_norm_raw = std(embedding_norms_raw)
```

**Диапазон**: 0.0 — inf

**Важно**: Используются raw embeddings для корректного вычисления вариативности.

#### 4.2.3. temporal_variance

Временная вариация embeddings. Показывает, насколько embeddings отклоняются от среднего значения. Использует нормализованные embeddings для устойчивости.

**Алгоритм**: 
```python
# Используются нормализованные embeddings
mean_embedding_normed = mean(normed_embeddings, axis=0)
temporal_variance = mean([||e - mean_embedding_normed||_2 for e in normed_embeddings])
```

**Диапазон**: 0.0 — inf (обычно 0.0 — 2.0 для нормализованных embeddings)

**Интерпретация**: Высокие значения указывают на разнообразие действий, низкие — на стабильность.

#### 4.2.4. max_temporal_jump

Максимальный скачок между соседними клипами. Показывает максимальное изменение действий (hook moment). Использует нормализованные embeddings.

**Алгоритм**: 
```python
# Используются нормализованные embeddings
temporal_jumps = [||normed_embeddings[i] - normed_embeddings[i-1]||_2 
                 for i in range(1, len(normed_embeddings))]
max_temporal_jump = max(temporal_jumps)
```

**Диапазон**: 0.0 — 2.0 (для нормализованных embeddings, максимум = 2.0)

**Интерпретация**: Высокие значения указывают на резкие изменения действий (hook moments).

---

### 4.3. TEMPORAL STRUCTURE

#### 4.3.1. stability

Стабильность действий (0.0-1.0), вычисляемая через кластеризацию embeddings (k-means) с фиксацией пространства через PCA.

**Алгоритм (для нормальных треков, num_clips ≥ 3)**: 
```python
# Фиксация пространства через PCA перед кластеризацией
n_pca_components = min(32, embedding_dim, num_clips - 1)
pca = PCA(n_components=n_pca_components)
embeddings_for_cluster = pca.fit_transform(normed_embeddings)

# Выбор k: min(5, num_clips // 2)
k = min(5, max(1, num_clips // 2))
kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
labels = kmeans.fit_predict(embeddings_for_cluster)

# Вычисление стабильности
stability = longest_run_fraction(labels)
```

**Диапазон**: 0.0 — 1.0

**Важно**: PCA фиксация пространства делает кластеризацию устойчивой к масштабу и шуму.

#### 4.3.2. switch_rate_per_sec

Частота смены действий в секунду, вычисленная через кластеризацию embeddings (используются те же labels, что и для stability).

**Алгоритм (для нормальных треков, num_clips ≥ 3)**: 
```python
# Используются те же labels из k-means кластеризации (см. stability)
transitions = sum(labels[1:] != labels[:-1])
total_time_sec = num_clips * clip_len / fps
switch_rate_per_sec = transitions / total_time_sec
```

**Диапазон**: 0.0 — inf

#### 4.3.3. early_late_embedding_shift

Сдвиг embeddings между первой и второй половиной последовательности. Вычисляется как `1 - cosine_similarity` для лучшей интерпретации в MLP.

**Алгоритм**: 
```python
mid = len(normed_embeddings) // 2
early_embedding = mean(normed_embeddings[:mid], axis=0)
late_embedding = mean(normed_embeddings[mid:], axis=0)
cosine_sim = cosine_similarity(early_embedding, late_embedding)
early_late_embedding_shift = 1.0 - cosine_sim  # 0 = одинаково, 1 = сильно изменилось
```

**Диапазон**: 0.0 — 2.0 (обычно 0.0 — 1.0)

**Интерпретация**: 
- 0.0 = действия не изменились
- 1.0 = действия сильно изменились

---

### 4.4. MOTION-AWARE FEATURES (упрощенные)

**Важно**: SlowFast уже имеет встроенную обработку motion через fast pathway. Поэтому мы используем только легковесный сигнал на основе temporal differences.

#### 4.4.1. motion_entropy

Энтропия распределения временных различий между клипами. Показывает структурированность движения.

**Алгоритм**: 
```python
# Используем нормализованные embeddings для вычисления temporal differences
temporal_diffs = [||normed_embeddings[i] - normed_embeddings[i-1]||_2 
                 for i in range(1, len(normed_embeddings))]
temporal_diffs_norm = temporal_diffs / (max(temporal_diffs) + 1e-6)
temporal_diffs_probs = temporal_diffs_norm / (sum(temporal_diffs_norm) + 1e-6)
motion_entropy_raw = -sum(temporal_diffs_probs * log(temporal_diffs_probs + 1e-12))

# Нормализация в [0, 1] по максимальной энтропии log(T)
T = len(temporal_diffs)
motion_entropy = motion_entropy_raw / (log(T + 1e-6) + 1e-12)
```

**Диапазон**: \[0.0, 1.0]

**Примечание**: Убраны optical flow и motion_weighted_embedding_energy, так как SlowFast уже обрабатывает motion внутренне.

---

### 4.5. DIVERSITY METRICS

#### 4.5.1. num_unique_actions

Количество уникальных кластеров действий (через k-means кластеризацию embeddings).

**Алгоритм**: 
```python
labels = kmeans.fit_predict(embeddings)
num_unique_actions = len(unique(labels))
```

**Диапазон**: 1 — num_clips

#### 4.5.2. dominant_action_ratio

Доля клипов в доминирующем кластере. Показывает стабильность действия.

**Алгоритм**: 
```python
unique, counts = unique(labels, return_counts=True)
dominant_action_ratio = max(counts) / len(labels)
```

**Диапазон**: 0.0 — 1.0

#### 4.5.3. embedding_entropy

Энтропия eigenvalues ковариационной матрицы embeddings. Показывает глобальное разнообразие в пространстве embeddings.

**Алгоритм**: 
```python
# Используем нормализованные embeddings
cov_matrix = cov(normed_embeddings.T)

# Численная стабилизация: добавляем маленький сдвиг по диагонали
cov_matrix += 1e-5 * I

# Для симметричных матриц используем eigh
eigenvalues = eigh(cov_matrix).eigenvalues
eigenvalues = abs(eigenvalues[eigenvalues > 1e-6])  # Фильтруем малые значения
eigenvalues_norm = eigenvalues / (sum(eigenvalues) + 1e-6)
embedding_entropy = -sum(eigenvalues_norm * log(eigenvalues_norm + 1e-12))
```

**Диапазон**: 0.0 — log(num_components)

**Преимущества**: Быстрее и устойчивее, чем PCA-based подход. Показывает глобальное разнообразие.

---

### 4.6. MULTI-PERSON CONTEXT

Доступны только если обрабатывается несколько треков (≥2).

#### 4.7.1. is_multi_person

Булево значение, указывающее на наличие нескольких людей в сцене.

**Алгоритм**: `True` если `num_tracks >= 2`, иначе `False`.

#### 4.7.2. num_persons

Количество людей (треков) в сцене, ограничено максимумом 5.

**Алгоритм**: 
```python
num_persons = min(num_tracks, 5)
```

**Диапазон**: 1 — 5

#### 4.6.3. action_synchronization

Синхронизация действий между людьми (0.0-1.0), где 1.0 — полная синхронизация. Вычисляется через cosine similarity между mean embeddings треков.

**Алгоритм**: 
```python
# Получаем mean embeddings для каждого трека
track_embeddings = [mean(track['sequence_features']['embedding_256d'], axis=0) for track in tracks]
# Вычисляем pairwise cosine similarities
similarities = [cosine_similarity(track_embeddings[i], track_embeddings[j]) 
                for i in range(len(tracks)) for j in range(i+1, len(tracks))]
action_synchronization = mean(similarities)
```

**Диапазон**: -1.0 — 1.0 (обычно 0.0 — 1.0)

---

## Технические детали

### Motion-aware агрегация

Модуль использует optical flow (алгоритм Farneback) для вычисления motion magnitude каждого кадра. При агрегации результатов клипы с большим движением получают больший вес, что позволяет более точно определять действия в динамических сценах.

### Создание клипов

Клипы создаются с использованием скользящего окна:
- Длина клипа: `clip_len` (по умолчанию 16 кадров)
- Шаг (stride): `clip_len // 2` (по умолчанию)
- Для коротких треков клипы дополняются последним кадром до нужной длины

### Обработка треков

Модуль обрабатывает несколько треков (людей) параллельно:
- Каждый трек обрабатывается независимо
- Результаты агрегируются по трекам
- При наличии нескольких треков автоматически выполняется анализ групповых действий

### Sequence Features для VisualTransformer

Sequence features извлекаются для каждого клипа и подаются в VisualTransformer:
- `embedding_normed_256d`: [num_clips, 256] — L2-нормализованные embeddings (для VisualTransformer)
- `temporal_diff_normalized`: [num_clips] — нормализованные временные различия

Эти фичи позволяют трансформеру самостоятельно выучить:
- Стабильность действий (через embedding similarity)
- Переключения между действиями (через temporal_diff)
- "Hook moments" (ключевые моменты через embedding peaks)

**Преимущества SlowFast embeddings:**
- Содержат информацию о ритме и темпе (fast pathway)
- Содержат информацию о семантике (slow pathway)
- Более устойчивы к fps/stride изменениям
- Лучше работают с короткими клипами

**Важно**: Используются нормализованные embeddings (||e|| = 1), чтобы избежать влияния масштаба.

### Aggregate Features для MLP/Tabular Head

Aggregate features — это статистики, агрегированные по всей последовательности:
- **Core Dynamics** (4 фичи): `mean_embedding_norm_raw`, `std_embedding_norm_raw`, `temporal_variance`, `max_temporal_jump`
- **Temporal Structure** (3 фичи): `stability`, `switch_rate_per_sec`, `early_late_embedding_shift` (1 - cosine)
- **Motion-aware** (1 фича): `motion_entropy` (только entropy от temporal_diff)
- **Diversity** (3 фичи): `num_unique_actions`, `dominant_action_ratio`, `embedding_entropy` (cov eigenvalues)
- **Multi-person Context** (3 фичи): `is_multi_person`, `num_persons`, `action_synchronization`

**Итого**: ~14 фичей для каждого трека.

**Убрано**:
- ❌ `motion_weighted_embedding_energy` (избыточно, коррелирует с temporal_variance)
- ❌ Optical Flow (SlowFast уже обрабатывает motion)
- ❌ Fine-grained presence features (keyword-based подход не работает с SlowFast)

### Параметры конфигурации

**clip_len** (по умолчанию 32):
- Стандартная длина клипа для SlowFast
- SlowFast использует два пути: slow pathway (каждый 16-й кадр) и fast pathway (каждый 2-й кадр)
- Влияет на временное разрешение анализа

**batch_size** (по умолчанию 4):
- Размер батча для inference
- Меньше чем для VideoMAE, так как SlowFast более требователен к памяти
- Влияет на скорость обработки и потребление памяти GPU

**stride** (по умолчанию `clip_len // 2`):
- Шаг скользящего окна
- Меньший stride = больше клипов = более детальный анализ, но медленнее

**embedding_dim** (по умолчанию 256):
- Целевая размерность embeddings после проекции из 2048d
- Используется для VisualTransformer

### Производительность

- Рекомендуется использовать GPU для ускорения inference
- Оптимальный batch_size зависит от доступной памяти GPU (обычно 4-8)
- Для длинных треков можно увеличить stride для уменьшения количества клипов

### Требования к входным данным

Модуль ожидает:
- `FrameManager` с методом `get(idx)` для получения кадров
- Кадры в формате RGB (H, W, 3) или grayscale (H, W)
- Атрибут `fps` в FrameManager (опционально, по умолчанию 30.0) для правильного вычисления временных метрик
- Словарь `{track_id: [список индексов кадров]}` для обработки треков
