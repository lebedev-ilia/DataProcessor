## Общие инженерные установки (повторно применяются ко многим фичам)

- Батчинг: 64–512 последовательностей на GPU в зависимости от длины и памяти.
- Mixed precision: fp16 для ускорения, затем L2-normalize в fp32 для стабильности.
- Кэш эмбеддингов: SHA256(content + model_version) → хранить в vector-store/kv.
- Токенизация: SentencePiece/Byte-Pair; для RU/EN — multilingual tokenizer.
- Дименсиональность: ориентируйтесь на 768/1024/1536 в зависимости от модели; унифицируйте для downstream (PCA/linear projection при необходимости).
- ANN: Faiss (HNSW/IVF+PQ) или Milvus; при ранжировании — hybrid sparse+dense.
- Cross-encoder rerank: использовать только для top-k (k=50–200) из ANN/ BM25.

### 1. title_embedding

**Лучший алгоритм (рекомендация)**

- Модель: fine-tuned SBERT-style bi-encoder (например: sentence-transformers fine-tuned on in-domain pairs), или коммерческий плотный embedding (text-embedding-3 / E5) при наличии доступа.
- Предобработка: нормализация unicode, strip HTML, replace URLs with <URL>, lowercase опционно (если модель чувствительна).
- Pooling: pooled sentence embedding (mean of token vectors) или CLS depending on model; затем L2-normalize.
- Инженерно: батчировать все заголовки; кешировать по content_hash. Размер вектора 768–1536.

**Альтернатива**

- Universal Sentence Encoder / off-the-shelf SBERT без дообучения. Быстрее стартовать, но хуже domain fit.

### 2. title_embedding_norm (L2-norm of title_embedding)

**Лучший алгоритм**

- Вычислить L2-норму вектора, затем сохранить как отдельную фичу. Нормализация векторов должна выполняться отдельно (нормализованный вектор хранится для косинуса; norm хранится для других сигналов).

**Альтернатива**

- Max-abs norm или спектральная нормировка (полезно если векторы сопоставляются из разных моделей).

### 3. description_embedding

**Лучший алгоритм**

- Модель: long-context SBERT / encoder, либо chunk-and-aggregate: разбить описание на куски ≤512 токенов, embedded each chunk with the same bi-encoder, затем использовать attention-weighted pooling (веса по tf-idf/position/ASR_confidence если применимо).
- Pooling: weighted mean + L2 normalize.

**Альтернатива**

- TF-IDF + SVD (Latent Semantic Analysis) — дешёвая, работает на больших объёмах, но теряет синтаксическую/контекстную информацию.

### 4. description_embedding_norm

**Лучший алгоритм**

- Аналогично title_embedding_norm: вычислить L2-норму description_embedding после pooling.

**Альтернатива**

- Использовать cosine magnitude vs corpus mean (z-score) как нормализованный показатель.

### 5. transcript_chunk_embeddings[] (list of chunk embeddings)

**Лучший алгоритм**

- Chunking: разбить транскрипт по смысловым границам (предложения) либо по фиксированным окнам (~128–512 токенов, overlap 10–20%).
- Модель: sentence-transformers (fine-tuned for long text) или dedicated long-encoder (Longformer/LED) для chunk→embedding.
- Инженерно: сохранить список embedding'ов (chunk timestamp map) в vector store; при генерации downstream — доступ к конкретному чанку по таймкоду.

**Альтернатива**

- Use transformer summarizer to compress chunk into short summary then embed summaries — экономит память, но теряет детализацию.

### 6. transcript_embedding_mean

**Лучший алгоритм**

- Взять chunk embeddings и сделать weighted mean. Весing: w = ASR_confidence * position_decay (например, небольшая отдача ранним чанкам). Затем L2-normalize.
- Размер/точность: сохранять также count_chunks и std_of_chunks.

**Альтернатива**

- Simple mean across chunks (проще, но менее чувствителен к качеству частей).

### 7. transcript_embedding_maxpool

**Лучший алгоритм**

- Поэлементный max-pool по всем chunk embeddings → затем L2-normalize. Хорошо подчёркивает сильные локальные сигналы (ключевые фразы).

**Альтернатива**

- Concatenate mean+max ( mayor improvement but vector doubling; можно PCA afterwards).

### 8. comments_embeddings_agg_mean

**Лучший алгоритм**

- Получить эмбеддинги каждого комментария (тот же bi-encoder), затем attention-weighted mean, где веса рассчитываются на основе: likes_on_comment, commenter_authority_score, recency_weight (если snapshot). Если likes не доступны, использовать кластерный вес (размер кластера комментариев).
- L2 norm итогового вектора.

**Альтернатива**

- Простое mean embedding (быстро и робастно).

### 9. comments_embeddings_agg_median

**Лучший алгоритм**

- Component-wise median across comment embeddings — устойчив к выбросам (spam/одиночные токсичные комменты).

**Альтернатива**

- Trimmed mean (удалить top/bottom 5% на каждой компоненте).

### 10. hashtag_embedding

**Лучший алгоритм**

- Сформировать строку из нормализованных хэштегов (remove #, normalize variants), embed через SBERT, L2 normalize. Если карточина большая — average of per-hashtag embeddings weighted by frequency.

**Альтернатива**

- Hashing trick (categorical → dense via embedding table) — эффективно при огромном количестве уникальных тегов, но теряет семантику.

### 11. title_description_cosine

**Лучший алгоритм**

- Вычислить cosine(title_embedding_normalized, description_embedding_normalized). При значениях около 1 → высокая семантическая согласованность.
- Для спорных случаев (cosine в промежутке 0.35–0.65) — использовать cross-encoder (BERT/DeBERTa) для rerank и более точного score.

**Альтернатива**

- Jaccard token overlap / normalized token intersection (простая, но чувствительна к стоп-словам).

### 12. title_transcript_cosine

**Лучший алгоритм**

- Cosine между title_embedding и transcript_embedding_mean. При низких значениях — сигнал clickbait. Для high-value use: cross-encoder on (title, transcript_excerpt) for top-k chunks to compute min/max similarity → take max over chunks for fine localization.

**Альтернатива**

- Keyword presence (title words in first N seconds of transcript) — дешёвый, но грубый.

### 13. description_transcript_cosine

**Лучший алгоритм**

- Cosine между description_embedding и transcript_embedding_mean. Для длинных описаний — chunk ↔ chunk aggregation and cross-encoder rerank for top matches.

**Альтернатива**

- Overlap of extracted keyphrases (KeyBERT) between description and transcript.

### 14. transcript_comments_cosine_mean

**Лучший алгоритм**

- Для каждого comment_embedding считать cosine к ближайшему transcript_chunk_embedding; агрегировать mean/median/topK. Это покажет насколько комментарии тематически связаны с содержимым.
- Оптимизация: ANN для transcript_chunks per video to accelerate nearest neighbor.

**Альтернатива**

- Compute global cosine between transcript_embedding_mean and comments_embeddings_agg_mean (less granular).

### 15. embedding_pair_topk_scores (top-k cross-encoder scores for pairs)

**Лучший алгоритм**

- Pipeline: retrieve top-K candidates by dense ANN (cosine) or hybrid BM25+dense → rerank with cross-encoder (BERT/DeBERTa large) to get high-precision pairwise scores (title↔transcript, title↔description). Use temperature-scaled logits for interpretability.

**Альтернатива**

- Use just dense cosine scores (faster, less precise).

### 16. semantic_cluster_id

**Лучший алгоритм**

- Предварительно собрать эмбеддинги корпуса → reduce dim (PCA to 128) → HDBSCAN (density clustering) to discover clusters; при incoming video — find nearest cluster centroid via Faiss (ANN) and return cluster_id. HDBSCAN даёт noise/outlier label.

**Альтернатива**

- K-means on PCA (faster but forced fixed k, хуже при неравномерных плотностях).

### 17. semantic_cluster_distance

**Лучший алгоритм**

- Хранить centroid в том же векторном пространстве; distance = cosine(title_or_transcript_embedding, centroid). Для outlier detection использовать threshold derived from cluster dispersion (e.g., 95th percentile distance).

**Альтернатива**

- Mahalanobis distance using cluster covariance (сложнее, но лучше для неравномерных распределений).

### 18. embedding_variance_across_chunks

**Лучший алгоритм**

- Для каждого векторного измерения считать variance across chunk_embeddings; сохранить L2 norm of variance vector (scalar) + top-k dim variances. Высокая variance → мультитематичность.

**Альтернатива**

- Mean pairwise cosine dispersion: 1 − mean(cosine between all pairs of chunk embeddings).

### 19. embedding_topic_mix_entropy

**Лучший алгоритм**

- Прямо получить topic probabilities (напра. BERTopic) для каждого chunk → усреднить → compute Shannon entropy of the averaged topic distribution. Высокая энтропия = multi-topic.

**Альтернатива**

- Entropy on LDA topic vector (но BERTopic обычно даёт лучшее качество для short/medium texts).

### 20. embedding_language_aware

**Лучший алгоритм**

- Использовать multilingual SBERT или language-specific encoder (e.g., RuBERT-based embedding for Russian content). Route text to language-specific encoder to improve representation.

**Альтернатива**

- One multilingual model for all languages (easier infra, чуть хуже per-language performance).

### 21. title_to_hashtag_cosine

**Лучший алгоритм**

- Embedding title и pooled hashtags, затем cosine. Нормализация хэштегов перед embedding (stemming, variant collapse).

**Альтернатива**

- Binary overlap of lemmatized tokens (быстро, но грубее).

### 22. topk_similar_corpus_titles

**Лучший алгоритм**

- ANN (Faiss HNSW) on title embeddings → retrieve top-k IDs + scores; при необходимости rerank top-k with cross-encoder for precision. Хранить also temporal metadata for similarity filters.

**Альтернатива**

- BM25 retrieval on title text (для лексически похожих, но семантически слабых matches).

### 23. longform_embedding_summary

**Лучший алгоритм**

- Compress chunk embeddings via a learned autoencoder (reduce vector to smaller dimension, e.g., 1536→256) или run PCA on chunk embeddings and keep first N components; это экономит место и сохраняет signal.

**Альтернатива**

- Mean pooling (простой, но менее компактный/информативный).

### 24. speaker_turn_embeddings_agg

**Лучший алгоритм**

- Если доступна диаризация: group transcript by speaker turns, embed each speaker block with same encoder, then store per-speaker aggregated embeddings (mean/max). Use speaker metadata to weight (host vs guest).

**Альтернатива**

- Heuristic speaker estimation from transcript (e.g., name cues) then do same embedding; менее точный без dedicated diarization.

### 25. qa_embedding_pairs (question-like phrases)

**Лучший алгоритм**

- Extract question sentences (interrogative detection using classifier or simple heuristics) → embed each question separately → produce set of QA embeddings. Это полезно для FAQ matching.

**Альтернатива**

- Extract all sentences containing wh-words (когда/почему/как) via regex + embed (быстрее, но шумнее).

### 26. embedding_shift_indicator

**Лучший алгоритм**

- Compute embeddings for beginning window (first N chunks) and ending window (last N chunks); compute cosine(begin, end). If cosine < threshold (tuned on validation), flag shift. Use multiple scales (30s, 2min windows).

**Альтернатива**

- KL divergence on topic distributions from BERTopic (требует topic model).

### 27. title_embedding_cluster_entropy

**Лучший алгоритм**

- Soft cluster membership: compute distances/scores to K nearest cluster centroids → apply softmax → entropy of that distribution. High entropy = ambiguous title.

**Альтернатива**

- Count of distinct clusters within top-K nearest (discrete measure).

### 28. embedding_source_id (vector store ref)

**Лучший алгоритм**

- Store embeddings in vector DB and return a stable URI/ref ID (vector_id) + model_version. Use this ID for lazy retrieval in downstream models. Ensure consistency between feature store and vector store (transactional ref).

**Альтернатива**

- Store embeddings in parquet blobs with pointer string; дешевле, но медленнее для ANN.

## Короткие рекомендации по hyperparams и балансам

- Chunk size: 128–512 токенов; overlap 10–20% для сохранения контекста.
- Pooling: mean + max concat часто даёт лучший рабочий компромисс; если память критична — mean + attentive pooling (weighting by tf-idf or ASR_confidence).
- Dimensionality reduction: при большом объёме корпусов — PCA→128 или product quantization (PQ) для Faiss/IVF+PQ.
- Rerank: cross-encoder только для top-k (k=50–200) — экономически оправданно.
- Evaluation: проверяйте качество эмбеддингов на downstream задачах (retrieval MAP, clustering purity, correlation with human labels).