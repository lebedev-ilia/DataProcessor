## Семантические фичи: обзор по экстракторам

Ниже приведены фичи, формируемые текущим пайплайном, со ссылкой на артефакты, форматом значений, кратким описанием алгоритма и ориентировочными затратами времени (по полю `timings_by_extractor` в report.json).

### TagsExtractor
- **Что делает**: извлекает хэштеги из `title` и `description`; возвращает очищенные тексты без `#тегов` и список тегов.
- **Поля**:
  - `results_by_extractor.TagsExtractor.cleaned_texts.title` — строка (очищенный заголовок).
  - `results_by_extractor.TagsExtractor.cleaned_texts.description` — строка (очищенное описание).
  - `results_by_extractor.TagsExtractor.hashtags` — массив строк (нормализованные теги без `#`).
- **Алгоритм**: регулярное выражение для `#tag`, нормализация пробелов; объединение тегов из title/description (уникализация, порядок сохранён).
- **Время**: ~0.0–0.01 s.

### TitleEmbedder
- **Что делает**: эмбеддинг очищенного `title` (L2-нормализованный вектор).
- **Поля**:
  - `results_by_extractor.TitleEmbedder.title_embedding.path` — путь к `.npy` (вектор `float32`, dim зависит от модели, например 768/1024).
  - `results_by_extractor.TitleEmbedder.title_embedding.shape` — форма (D,).
  - `results_by_extractor.TitleEmbedder.title_embedding_norm` — L2-норма до нормализации.
- **Алгоритм**: SentenceTransformers (модель из ModelRegistry), `encode` без нормализации → вычисление L2 → итоговый вектор нормализуется.
- **Время**: `timings_by_extractor.TitleEmbedder.encode` ~0.001–0.4 s; `total` ~0.01–0.4 s; `init` — время загрузки модели при первом создании.

### DescriptionEmbedder
- **Что делает**: эмбеддинг `description` (поддержка длинного текста, chunk-and-aggregate).
- **Поля**:
  - `results_by_extractor.DescriptionEmbedder.description_embedding.path` — путь к `.npy` (вектор `float32`, D).
  - `results_by_extractor.DescriptionEmbedder.description_embedding.shape` — форма (D,).
  - `results_by_extractor.DescriptionEmbedder.description_embedding_norm` — L2-норма до нормализации.
- **Алгоритм**: разбиение на чанки по токенам/словам, `encode(normalize_embeddings=False)` → attention-weighted pooling по длине → L2-нормализация.
- **Время**: `timings_by_extractor.DescriptionEmbedder.total` ~0.05–0.4 s; `init` — время загрузки модели при первом создании.

### HashtagEmbedder
- **Что делает**: эмбеддинг списка `hashtags` (полученных ранее), усреднение по тегам с L2-нормализацией результата.
- **Поля**:
  - `results_by_extractor.HashtagEmbedder.hashtag_embedding.path` — путь к `.npy` (вектор `float32`, D).
  - `results_by_extractor.HashtagEmbedder.hashtag_embedding.count` — число тегов.
  - `results_by_extractor.HashtagEmbedder.hashtag_embedding.l2_norm` — L2-норма усреднённого вектора (после нормализации ≈1.0).
- **Алгоритм**: батчевый `encode` тегов → L2-нормализация поштучно → среднее по осям → финальная L2-нормализация.
- **Время**: `timings_by_extractor.HashtagEmbedder.total` ~0.01–0.2 s.

### TranscriptChunkEmbedder
- **Что делает**: строит эмбеддинги чанков транскрипта для источников `whisper` и/или `youtube_auto` отдельно.
- **Поля**:
  - `results_by_extractor.TranscriptChunkEmbedder.transcript_chunks_by_source.whisper.embeddings_path` — путь к `.npy` (матрица `float32` формы (N, D)).
  - Аналогично для `youtube_auto`.
  - `...meta_path` — JSON с метаданными (`chunks`, `n_chunks`, `embedding_dim`, `model`, `device`).
- **Алгоритм**: разбиение на предложения с overlap → батчевый `encode(normalize_embeddings=False)` → L2-нормализация каждого чанка → сохранение.
- **Время**: `timings_by_extractor.TranscriptChunkEmbedder.total` ~0.1–0.8 s (зависит от длины текста и батча).

### TranscriptAggregatorExtractor
- **Что делает**: агрегирует чанковые эмбеддинги транскриптов по источникам и совместно.
- **Поля**:
  - `results_by_extractor.TranscriptAggregatorExtractor.transcript_aggregates.whisper.aggregate_mean.path` — путь к агрегированному вектору.
  - `...aggregate_maxpool.path` — путь к max-pooled вектору.
  - Для `youtube_auto` и `combined` аналогично; присутствуют также `count`, `std`.
- **Алгоритм**: 
  - weighted mean: экспоненциальное позиционное затухание × (опционально) confidences (`whisper_confidence`), L2-нормализация.
  - maxpool: поэлементный максимум по чанкам, L2-нормализация.
- **Время**: `timings_by_extractor.TranscriptAggregatorExtractor.load` ~0.0–0.02 s; `aggregate` ~0.0–0.02 s; `total` ~0.01–0.05 s.

### CommentsEmbedder
- **Что делает**: эмбеддинг массива комментариев.
- **Поля**:
  - `results_by_extractor.CommentsEmbedder.comments_embeddings.path` — путь к `.npy` (матрица `float32` формы (N, D)).
  - `...shape`, `...count` — размерность и количество.
- **Алгоритм**: батчевый `encode(normalize_embeddings=False)` → L2-нормализация каждой строки → сохранение.
- **Время**: `timings_by_extractor.CommentsEmbedder.total` ~0.02–0.3 s (зависит от N).

### CommentsAggregationExtractor
- **Что делает**: агрегирует эмбеддинги комментариев по стратегиям weighted mean и median.
- **Поля**:
  - `results_by_extractor.CommentsAggregationExtractor.comments_aggregates.weighted_mean.path` — путь к агрегированному вектору.
  - `...median.path` — путь к медианному вектору; поля `count`, `std` присутствуют для обеих стратегий.
- **Алгоритм**:
  - weighted mean: веса = likes × authority × recency (если доступны); нормировка весов; среднее и L2 нормализация.
  - median: поэлементная медиана и L2 нормализация.
- **Время**: `timings_by_extractor.CommentsAggregationExtractor.mean|median|total` обычно < 0.05 s.

### TitleToHashtagCosineExtractor
- **Что делает**: косинусная близость между `title_embedding` и агрегированным `hashtag_embedding`.
- **Поля**:
  - `results_by_extractor.TitleToHashtagCosineExtractor.title_to_hashtag_cosine` — число в диапазоне [-1, 1].
- **Алгоритм**: загрузка L2‑нормированных эмбеддингов заголовка и усреднённых эмбеддингов хэштегов → косинус.
- **Время**: ~0.0–0.01 s.

### CosineMetricsExtractor
- **Что делает**: базовые косинусы между агрегированными источниками.
- **Поля**:
  - `title_description_cosine`, `title_transcript_cosine`, `description_transcript_cosine`,
  - `transcript_comments_cosine_mean`, `transcript_comments_cosine_median`.
- **Алгоритм**: загрузка путей агрегатов из `.artifacts` → косинусы пар (с защитой от пустых векторов).
- **Время**: ~0.0–0.02 s.

### EmbeddingPairTopKExtractor
- **Что делает**: top‑K пары «title ↔ transcript chunks», плюс косинус `title ↔ description`, опционально rerank CrossEncoder.
- **Поля**:
  - `results_by_extractor.EmbeddingPairTopKExtractor.embedding_pair_topk_scores.title_transcript_topk_cosine` — список K чисел.
  - `...title_transcript_topk_cross` — список K вероятностей (если CrossEncoder включен).
  - `...title_description_cosine` — число.
- **Алгоритм**: FAISS/NP косинусная матрица → top‑K индексы → при наличии текстов чанков — CrossEncoder c численно устойчивым softmax.
- **Время**: ~0.01–0.2 s (без CrossEncoder); ~0.05–0.5 s (с CrossEncoder), зависит от K.

### TitleEmbeddingClusterEntropyExtractor
- **Что делает**: измеряет «размытость» заголовка по распределению близостей к ближайшим кластерам.
- **Поля**:
  - `title_embedding_cluster_entropy.entropy` — энтропия softmax по top‑K центроидам.
  - `distinct_clusters_topk`, `top_k`, `temperature`, `clusters_path`.
- **Алгоритм**: L2‑нормализация центроидов и title; косинусы → top‑K → softmax(temperature) → энтропия.
- **Время**: ~0.0–0.02 s.

### EmbeddingShiftIndicatorExtractor
- **Что делает**: индикатор смыслового сдвига в транскрипте между началом и концом.
- **Поля**:
  - `embedding_shift_indicator.cosine_begin_end` — косинус между средними по окнам.
  - `shift_flag` — boolean (ниже порога), `n_chunks`, `n_window_chunks`.
- **Алгоритм**: средние по первым и последним N чанкам агрегированных `combined/whisper/youtube_auto` → косинус, порог.
- **Время**: ~0.0–0.01 s.

### SpeakerTurnEmbeddingsAggregatorExtractor
- **Что делает**: эмбеддинги по спикерам; агрегирует mean/max.
- **Поля**:
  - `results_by_extractor.SpeakerTurnEmbeddingsAggregatorExtractor.speaker_embeddings.<name>.mean.path` — `.npy` вектор.
  - `...max.path`, `...count_turns`.
- **Алгоритм**: encode блоков речи на общей модели → L2‑нормализация → mean/max → сохранение.
- **Время**: ~0.01–0.2 s (зависит от количества реплик).

### QAEmbeddingPairsExtractor
- **Что делает**: извлекает вопросительные фразы из `title/description/transcript/comments` и считает их эмбеддинги.
- **Поля**:
  - `qa_embeddings.path` (N×D), `meta_path` (источники, тексты, счётчики), `num_questions`, `embedding_dim`.
- **Алгоритм**: регэксп для вопросов → батчевый encode → L2‑нормализация поштучно → сохранение.
- **Время**: ~0.01–0.3 s.

### EmbeddingStatsExtractor
- **Что делает**: статистики по чанкам транскрипта.
- **Поля**:
  - `embedding_variance_across_chunks.l2_variance`, `topk_variances`.
  - `embedding_topic_mix_entropy.topic_entropy` (если доступны `topic_probs`).
- **Алгоритм**: дисперсия по размерностям и её L2; энтропия усреднённого распределения тем.
- **Время**: ~0.0–0.02 s.

### EmbeddingSourceIdExtractor
- **Что делает**: формирует «паспорт» векторного объекта для векторного стора (Faiss/Qdrant/Milvus/Weaviate).
- **Поля**:
  - `embedding_source_id.vector_id` — стабильный ID (sha1 содержимого + uuid5 по пути).
  - `vector_store_uri`, `model_version`, `created_at`, `embedding_path`.
- **Алгоритм**: выбирает основной эмбеддинг (title → transcript aggregate → description), читает `*.meta.json` или кэш‑мету транскрипта, чтобы определить `model_version`.
- **Время**: ~0.0–0.01 s.

## Общие примечания
- Все большие вектора/матрицы сохраняются как артефакты `.npy`, в отчёте возвращаются только метаданные и пути.
- Временные метрики по каждому экстрактору доступны в `features.timings_by_extractor.<ExtractorName>`: `init` (инициализация), `encode/aggregate/total` (фактическая работа), а также суммарное `runs[].total_s`.
- Модели переиспользуются через ModelRegistry (singleton), `encode` выполняется в `torch.no_grad()`.
- Все GPU‑экстракторы теперь возвращают поле `model_version` в своём результате, а также пишут `*.meta.json` рядом с артефактами (`{"model": <model_name>}`); это используется `EmbeddingSourceIdExtractor`.

