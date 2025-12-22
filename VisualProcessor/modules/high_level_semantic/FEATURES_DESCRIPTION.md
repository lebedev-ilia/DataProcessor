# Описание фичей модуля high_level_semantic

Модуль извлекает высокоуровневые семантические фичи из видео, используя CLIP embeddings для сцен с trainable projection и мультимодальный анализ. Вычисляет embeddings сцен и видео, темы, события, эмоциональное выравнивание, нарративные фичи и жанровую классификацию.

## Общие принципы реализации

### CLIP → нормализация → проекция
Для каждой сцены получается CLIP embedding (512/768/1024), L2-нормализуется и проходит через trainable linear projection (например, 512→64 или 128). Это даёт компактный, task-specific embedding, который идёт в VisualTransformer или в downstream модель.

### Метаданные сцен
Хранятся метаданные: `scene_id`, `start_ts`, `end_ts`, `representative_frame_idx`, `scene_duration`, `scene_source_confidence`.

### Мультимодальная агрегация — learnable
Не жёстко 45/35/20; вместо этого — либо обучаемые веса, либо attention fusion (multi-head fusion между clip, audio, text, emotion, ocr).

### Reliability flags
`emotion_reliable`, `audio_reliable`, `ocr_reliable`, `clip_confidence` — downstream модель должна уметь учитывать отсутствие/ненадёжность модальности.

### Fast / Full режим
- **fast**: CLIP + audio coarse features + subtitles
- **full**: fine emotion, speaker diarization, multimodal attention, topic modeling, events detection

## 1. Scene-level Semantic Embeddings

### scene_embeddings
Массив проектированных embeddings для каждой сцены в видео. Получены через CLIP модель → L2-нормализация → trainable linear projection. Размерность проекции: 32-128 (по умолчанию 64). Каждая сцена представлена одним репрезентативным кадром (обычно средний кадр сцены).

### scene_metadata
Список словарей с метаданными для каждой сцены:
- `scene_id`: уникальный идентификатор сцены
- `start_ts`: время начала сцены в секундах
- `end_ts`: время окончания сцены в секундах
- `representative_frame_idx`: индекс репрезентативного кадра
- `scene_duration`: длительность сцены в секундах
- `scene_source_confidence`: уверенность в определении границ сцены (0..1)

### scene_sim_adjacent_mean
Средняя косинусная схожесть между соседними сценами (на проектированных embeddings). Показывает плавность семантических переходов между сценами.

### scene_sim_adjacent_std
Стандартное отклонение схожести между соседними сценами. Показывает вариативность семантических переходов.

## 2. Video-level Embeddings

### mean_embedding
Среднее арифметическое всех проектированных scene embeddings. Представляет общую семантику видео.

### weighted_mean_embedding
Взвешенное среднее scene embeddings с learnable attention weights (не жёстко 45/35/20). Веса вычисляются через multimodal attention fusion или обучаемые параметры. Представляет семантику видео с учетом мультимодальных сигналов.

### max_embedding
Максимальные значения по каждой размерности из всех scene embeddings. Выделяет наиболее выраженные семантические признаки. Редко добавляет сигнал, можно убрать или держать как опцию.

### var_embedding
Дисперсия по каждой размерности scene embeddings. Показывает разнообразие семантического контента.

### video_embedding_norm_mean
Норма среднего embedding. Показывает общую силу семантического сигнала.

### video_embedding_norm_weighted
Норма взвешенного среднего embedding. Показывает силу семантического сигнала с учетом мультимодальных весов.

### video_embedding_var_mean
Среднее значение дисперсии embeddings (scalar, не full vector). Показывает среднее разнообразие семантических признаков.

## 3. Topic / Concept Detection

### topic_probabilities
Словарь вероятностей для каждой темы. Если предоставлены topic_vectors, вычисляется схожесть сцен с каждой темой и агрегируется через softmax с температурой. Если предоставлен video_topic_embedding, вычисляется схожесть с ним.

### per_scene_dominant_topic
Список названий доминантной темы для каждой сцены. Определяется как тема с максимальной схожестью для каждой сцены.

### topic_diversity_score
Нормализованная энтропия распределения тем (0..1). Показывает разнообразие тем в видео (выше = больше разнообразие).

### topic_transition_rate
Частота смены тем между сценами. Вычисляется как отношение количества переходов между темами к общему количеству переходов между сценами.

## 4. Events / Key Moments Detection

### number_of_events
Количество обнаруженных ключевых событий в видео. События детектируются как пики в комбинированном мультимодальном сигнале с learnable весами.

### event_rate_per_minute
Количество событий в минуту. Показывает плотность ключевых моментов.

### event_strength_max
Максимальная сила обнаруженного события. Показывает интенсивность самого сильного события.

### event_types
Список типов событий для каждого обнаруженного события. Типы определяются доминантным каналом в момент пика: "face", "audio", "scene_jump", "text", "pose", "ocr".

### event_timestamps
Список временных меток событий в секундах. Показывает моменты появления ключевых событий.

### event_context_embeddings
Embeddings контекста вокруг каждого события (±1s). Вычисляется как среднее scene embeddings в окрестности события.

Алгоритм детекции:
1. Собрать нормализованные кривые: face_emotion_intensity, audio_energy, scene_jump_score, ocr_activity, pose_activity
2. Сгладить (gaussian/median, window ~1–3s), комбинировать с learnable весами
3. Детектировать пики: peak где value > mean + k*std и расстояние между пиками > min_distance (например 1–2s)

## 5. Emotion Alignment

### emotion_correlation
Корреляция Пирсона между кривой эмоций лиц и кривой эмоций текста (если доступна). Показывает согласованность эмоций в визуальном и текстовом каналах.

### emotion_lag_seconds
Задержка между пиками эмоций лиц и текста в секундах (в допустимом лаге ±5s). Положительное значение означает, что текстовые эмоции следуют за визуальными.

### emotion_alignment_score
Общая оценка выравнивания эмоций между каналами. Вычисляется через нормализованную кросс-корреляцию (максимум в допустимом лаге).

### emotion_alignment_reliability
Доля совпадающих пиков между кривыми эмоций (в пределах ±0.5s после учёта лага). Показывает надёжность выравнивания.

Методы:
- Кросс-корреляция для поиска оптимального лага
- Опционально: DTW для нелинейного выравнивания (если доступна библиотека dtaidistance)

## 6. Emotion Features

### avg_emotion_valence
Средняя валентность эмоций по всем кадрам. Показывает общую позитивность/негативность эмоций в видео.

### emotion_variance
Дисперсия эмоций. Показывает вариативность эмоциональных состояний.

### peak_emotion_intensity
Максимальная интенсивность эмоций. Показывает пиковую эмоциональную точку в видео.

### face_presence_ratio
Доля сцен/кадров с обнаруженным лицом (0..1). Показывает, насколько часто присутствуют лица в видео.

### avg_face_valence
Средняя валентность лиц (если доступна отдельная кривая валентности).

### avg_face_arousal
Средняя активация/возбуждение лиц (если доступна отдельная кривая активации).

Важно: нормализовать эмоции по модели/дате, хранить emotion_model_version (если доступно).

## 7. Narrative / Story Features

### narrative_embedding
Embedding нарратива, полученный через multimodal attention fusion визуальных scene embeddings и текстовых caption embeddings (если доступны). Используется проекция (128→64) для компактности. Представляет семантику повествования.

### story_flow_score
Оценка плавности повествования. Вычисляется как средняя косинусная схожесть между соседними сценами (в проектированном space). Выше значение = более плавное повествование.

### narrative_complexity_score
Оценка сложности повествования. Вычисляется как std(similarity_adjacent) + topic_transition_rate + topic_diversity (комбинируется). Выше значение = более сложное повествование.

### cross_modal_novelty_score
Оценка новизны контента между сценами. Вычисляется как mean(1 - cosine(similar_adjacent_scenes)) или measure of KL divergence of topic distributions between adjacent scenes. Удобен как measure of freshness.

## 8. Multimodal Features

### multimodal_attention_score
Корреляция между нормализованными активациями модальностей (скаляр 0..1). Хорош для detect синхронных моментов. Вычисляется через корреляцию между face/audio/text/ocr кривыми.

### multimodal_attention_overall
Общая оценка мультимодального внимания на уровне всего видео (скаляр).

## 9. Genre / Style Classification

### genre_probabilities
Словарь вероятностей для каждого жанра/стиля. Вычисляется через zero-shot классификацию CLIP с использованием текстовых промптов и temperature scaling для калибровки. Показывает вероятностное распределение по жанрам.

### per_scene_top_class
Список названий топ-класса для каждой сцены. Определяется как класс с максимальной схожестью для каждой сцены.

### dominant_genre
Название доминантного жанра (класс с максимальной вероятностью).

### genre_confidence
Уверенность в доминантном жанре (максимальная вероятность).

Zero-shot CLIP полезна, но калибруйте prompts. Confidence calibration важна — используется temperature scaling.

## 10. Per-Scene Vectors для VisualTransformer

### per_scene_vectors
Компактный per-scene вектор (размер ≈ 64) для VisualTransformer. Компоненты:
- `proj_clip_emb` — projected CLIP embedding (dim 32–48), L2-norm
- `scene_duration_norm` (1) — нормализованная длительность сцены
- `scene_position_norm` (1) — нормализованная позиция сцены в видео (0..1)
- `audio_energy_norm` (1) — средняя аудио энергия за сцену
- `face_presence_flag` (1) — флаг наличия лица
- `avg_face_valence` (1) — средняя валентность лица
- `avg_face_arousal` (1) — средняя активация лица
- `text_activity_flag` (1) — флаг активности текста
- `text_sentiment` (1) — сентимент текста
- `topic_onehot_or_topk_embedding` (4) — embedding темы (top-1 topic id → learnable embedding)
- `scene_novelty_score` (1) — оценка новизны сцены
- `multimodal_attention_score` (1) — оценка мультимодального внимания для сцены
- `scene_visual_confidence` (1) — уверенность в визуальном качестве сцены

Итого: ~48–64 числа. Все числовые — z-normalize по train set. Для position в трансформере дополнительно используются обычные позиционные эмбеддинги.

## 11. Reliability Flags

### emotion_reliable
Флаг надёжности данных эмоций (bool). True если face_emotion_curve доступна и содержит значимый сигнал.

### audio_reliable
Флаг надёжности аудио данных (bool). True если audio_energy_curve доступна и содержит значимый сигнал.

### ocr_reliable
Флаг надёжности OCR данных (bool). True если ocr_activity_curve доступна и содержит значимый сигнал.

### clip_confidence
Уверенность в CLIP embeddings (float, 0..1). Вычисляется на основе норм embeddings (должны быть близки к 1.0 для нормализованных).

## 12. Видео-уровневые агрегаты (не в трансформер)

### scene_count
Количество сцен в видео.

### avg_scene_duration
Средняя длительность сцены в секундах.

### shot_count
Количество кадров/шотов (если доступно из cut detection).

## Методы вычисления

1. **Scene Extraction**: Сцены определяются через cut detection + ограничение min_scene_duration (например 0.5–1s) и cap на max_scenes (например 200) — чтобы не взрываться на длинных видео.

2. **CLIP Embeddings**: Используется CLIP модель (по умолчанию ViT-L/14 или ViT-B/32) для получения визуальных embeddings сцен. Embeddings L2-нормализуются и проецируются через trainable linear projection.

3. **Multimodal Integration**: Модуль интегрирует данные из других модулей (emotion_face, audio processor, cut_detection, OCR) для мультимодального анализа с learnable attention fusion.

4. **Event Detection**: События детектируются через анализ пиков в комбинированном мультимодальном сигнале с learnable весами и улучшенным алгоритмом поиска пиков (scipy.signal.find_peaks).

5. **Topic Detection**: Темы определяются через cosine similarity с topic vectors и softmax с температурой для калибровки.

6. **Emotion Alignment**: Выравнивание эмоций через кросс-корреляцию с поиском оптимального лага, опционально DTW для нелинейного выравнивания.

## Зависимости

- **clip** или **open_clip**: Библиотеки для CLIP модели
- **torch**: PyTorch для работы с моделями и trainable layers
- **numpy**: Для численных вычислений
- **scipy**: Для фильтрации сигналов и поиска пиков
- **sklearn**: Для вычисления схожести
- **dtaidistance** (опционально): Для DTW выравнивания эмоций

## Использование

Модуль может использовать данные из других модулей через параметры:
- `scene_frames`: список репрезентативных кадров сцен
- `scene_embeddings`: предвычисленные CLIP embeddings (опционально)
- `scene_metadata`: метаданные сцен (опционально)
- `face_emotion_curve`: кривая эмоций лиц (per-frame)
- `audio_energy_curve`: кривая аудио энергии (per-frame)
- `text_features`: словарь с текстовыми фичами из TextProcessor
- `mode`: "fast" или "full" (по умолчанию "full")
- `class_prompts`: список промптов для zero-shot классификации

### Режимы работы

- **fast**: CLIP + audio coarse features + subtitles (быстрая обработка)
- **full**: fine emotion, speaker diarization, multimodal attention, topic modeling, events detection (полная обработка)
