# TextProcessor — Полный расширенный список фич (snapshot)

_Расширенная версия документа: теперь включено ~160 snapshot-фич, сгруппированных по категориям. Каждая фича имеет: краткое описание, зачем полезна (для моделей / аналитиков), наиболее качественный алгоритм извлечения и одна альтернативная опция._

---

## Оглавление
1. Введение и ограничения
2. Инструкции по использованию snapshot-фич
3. Семантические эмбеддинги (глубокая группа)
4. Лексико-статические признаки (широкий набор)
5. ASR / текстовые proxy аудио-фичи
6. Семантика, темы и ключевые фразы
7. Комментарии — snapshot-агрегаты (расширенно)
8. Cross-modal snapshot фичи (текст ↔ визуал / OCR)
9. NER / hashtag / keyphrase расширенный набор
10. Интерпретируемые и explainability-фичи
11. Контрольные и служебные поля
12. Пример CSV-схемы (с полями)
13. Резюме и дальнейшие шаги

---

## 1. Введение и ограничения
- TextProcessor возвращает только snapshot-фичи (т.е. признаки, вычисленные на момент `snapshot_time`).
- Никакие rolling/lag/arrival/time-to-peak признаки НЕ включаются — они в ведении TemporalProcessor.
- Этот документ — расширение предыдущего: включает большое количество производных признаков, полезных в моделях и аналитике.

---

## 2. Инструкции по использованию snapshot-фич
- Каждая фича должна содержать тип, диапазон, и `processor_version` для воспроизводимости.
- Векторные фичи (эмбеддинги) хранятся в vector-store; в feature-store сохраняется ссылка и нормализованный агрегат (mean/len).
- Для всех эмбеддингов — сохраняйте `embedding_model_version` и `content_hash`.

---

## 3. Семантические эмбеддинги (глубокая группа) — 28 фич
(Каждый пункт: описание | зачем полезна | лучший алгоритм | альтернатива)

1. **title_embedding** — pooled embedding заголовка. | Семантика заголовка для similarity/CTR. | Fine-tuned SBERT / text-embedding-3, pooled (mean+CLS). | E5 / off-the-shelf SBERT.
2. **title_embedding_norm** — L2-norm of title_embedding. | Сигнал интенсивности/разнообразия. | L2 norm compute. | Max-abs norm.
3. **description_embedding** — pooled embedding описания. | Дополнительный контекст для SEO/CTR. | SBERT long-context with truncation-aware pooling. | TF-IDF+PCA.
4. **description_embedding_norm** — L2-norm. | Схожая роль как для title. | L2 norm. | Mean absolute.
5. **transcript_chunk_embeddings[]** — list of chunk embeddings. | Детализированная семантика по частям. | chunk→SBERT (30s/paragraph). | sentence-transformers simple split.
6. **transcript_embedding_mean** — mean of chunk embeddings. | Обобщённая семантика содержания. | weighted mean (weight by ASR_confidence). | plain mean.
7. **transcript_embedding_maxpool** — max-pooled vector. | Подчёркивает сильные семантиковые сигналы. | max pool across chunks. | concatenated mean+max.
8. **comments_embeddings_agg_mean** — mean embedding по комментариям. | Семантика аудитории. | attention-weighted mean (likes as weight). | simple mean.
9. **comments_embeddings_agg_median** — медианный вектор (robust). | Робастная агрегированная точка. | component-wise median. | trimmed mean.
10. **hashtag_embedding** — pooled по всем хэштегам. | Маркеры тематики. | SBERT on joined hashtags. | one-hot hashing.
11. **title_description_cosine** — cosine(title, description). | Показатель согласованности заголовка и описания. | L2-normalized cosine. | Jaccard token overlap.
12. **title_transcript_cosine** — cosine(title, transcript). | Clickbait/faithfulness индикатор. | cosine + cross-encoder rerank when ambiguous. | token overlap.
13. **description_transcript_cosine** — cosine(description, transcript). | Faithfulness check. | cosine. | fuzzy matching.
14. **transcript_comments_cosine_mean** — mean similarity transcript↔comments. | Насколько комментарии отражают содержание. | batch cosine across chunk↔comment embeddings. | keyword overlap.
15. **embedding_pair_topk_scores** — top-k cross-encoder scores for (title vs transcript/description). | детальная оценка связи. | cross-encoder reranker. | coarse cosine.
16. **semantic_cluster_id** — id of nearest cluster in precomputed embedding clusters. | Быстрая категоризация видео. | Faiss ANN + HDBSCAN clusters. | k-means on PCA-reduced embeddings.
17. **semantic_cluster_distance** — distance to cluster centroid. | Novelty signal. | cosine distance. | euclidean.
18. **embedding_variance_across_chunks** — per-dim variance measure. | Индикация разнообразия доменов в видео. | variance across chunk embeddings. | mean pairwise cosine dispersion.
19. **embedding_topic_mix_entropy** — entropy over topic probabilities derived from embeddings. | Насколько мультитематично видео. | BERTopic topic probs → entropy. | LDA topic entropy.
20. **embedding_language_aware** — language-specific embedding (if multilingual pipeline). | Учитывает морфологию языка. | multilingual SBERT / XLM-R. | language-specific SBERT.
21. **title_to_hashtag_cosine** — similarity title↔hashtags. | Правильность хэштегов. | embedding cosine. | string match.
22. **topk_similar_corpus_titles** — IDs and scores of top-k corpus titles similar to current. | Анализ конкуренции/похожести. | Faiss ANN + cross-encoder rerank. | BM25 retrieval.
23. **longform_embedding_summary** — compressed vector summarizing long transcript (distil). | Для экономичного storage. | PCA or autoencoder on chunk embeddings. | mean pooling.
24. **speaker_turn_embeddings_agg** — embeddings aggregated by speaker turn (if diarization available). | Семантика по спикеру. | speaker-aware pooling. | pooled whole transcript.
25. **qa_embedding_pairs** — embeddings for (question-like phrases) extracted from transcript. | Аналитика вовлечения (вопросы). | Keyphrase extraction + embeddings. | regex cues (вопросительные словосочетания).
26. **embedding_shift_indicator** — indicator if beginning and ending embeddings differ beyond threshold. | Topic drift snapshot proxy. | cosine(beginning_pool, ending_pool) threshold. | KL divergence of topic probs.
27. **title_embedding_cluster_entropy** — entropy of k-nearest cluster assignments. | Насколько неоднозначен заголовок. | cluster-soft assignments via Gaussian Mixture. | hard cluster count.
28. **embedding_source_id** — identifier where embedding stored (vectorDB ref). | Инженерный трекер. | URI to vector store. | local blob path.

---

## 4. Лексико-статические признаки — 36 фич
(короткие, но многие вариации/нормализации)

29. **title_len_words** — кол-во слов. | CTR/SEO индикатор. | tokenizer count (SentencePiece). | whitespace split.
30. **title_len_chars** — длина в символах. | визуальная длина заголовка. | unicode-aware length. | byte-length.
31. **title_avg_word_len** — средняя длина слова. | сложность лексики. | token lengths mean. | char/word ratio.
32. **title_type_token_ratio** — unique/total tokens. | лексическая разнообразность. | set/token ratio. | shannon entropy.
33. **title_stopword_ratio** — доля стоп-слов. | информативность заголовка. | stopword list per language. | POS-based heuristic.
34. **title_punctuation_ratio** — количество знаков / length. | тон заголовка. | char counts normalized. | punctuation categories.
35. **title_exclamation_count** — кол-во '!'. | эмоциональность/промо. | char count. | regex.
36. **title_question_count** — количество '?'. | побуждение к действию. | char count. | regex.
37. **title_capital_words_ratio** — доля слов в верхнем регистре. | громкость/агрессивность заголовка. | token.isupper checks. | regex for unicode uppercase.
38. **description_len_words** — кол-во слов в описании. | детализация. | tokenizer. | char/avg.
39. **description_num_urls** — ссылки в описании. | промо/affiliate signals. | URL regex. | heuristic parsing.
40. **description_num_mentions** — @mentions. | партнёрства/коллаборации. | regex. | NLP mention detection.
41. **description_has_timestamps_flag** — наличие временных меток в описании. | структурированность. | regex for time patterns. | time-like token patterns.
42. **transcript_len_words** — слова в транскрипте. | плотность контента. | ASR tokens. | est by duration.
43. **transcript_avg_sentence_len** — words per sentence. | сложность речи. | sentence segmentation + avg. | punctuation heuristic.
44. **lexical_diversity_transcript** — type/token ratio. | разнообразие слов. | token set/len. | entropy.
45. **rare_word_ratio_transcript** — доля слов за пределами топ-N словаря. | нишевость контента. | frequency dictionary threshold. | word length proxy.
46. **pos_distribution_transcript** — distribution of POS tags (nouns, verbs, adj...). | стилевой профиль. | UDPipe / spaCy POS tagging. | unigram heuristics.
47. **stopword_ratio_transcript** — доля стоп-слов. | информативность речи. | stopword list. | language-model perplexity proxy.
48. **readability_score_transcript** — Readability metric. | уровень сложности. | language-specific formula / ML regressor. | average sentence length.
49. **title_clickbait_score** — вероятность clickbait. | CTR vs retention tradeoff. | fine-tuned transformer classifier. | rule-based heuristics.
50. **title_question_prefix_flag** — starts with question word. | engagement tactic. | token check. | regex.
51. **title_number_presence** — numbers in title. | listicles / top-N attractors. | regex numeric detection. | token parse.
52. **title_time_mention_flag** — presence of dates/times. | topicality/urgency. | NER date detection. | regex.
53. **punctuation_entropy** — entropy of punctuation distribution. | stylistic signal. | compute discrete entropy. | simple counts.
54. **emoji_count_title** — number of emojis in title. | tone marker. | unicode emoji lib. | regex.
55. **emoji_count_description** — emojis in description. | tone marker. | emoji lib. | regex.
56. **special_character_ratio** — % of non-alnum chars. | noise/formatting. | unicode categories. | heuristic.
57. **text_language** — primary language. | routing to models. | fastText langid. | cld3.
58. **language_confidence** — confidence score. | trust routing. | langid confidence. | alternative measure.
59. **orthographic_error_rate** — estimate of spelling errors in text. | ASR/text quality proxy. | spellchecker diff (SymSpell) vs text. | LM perplexity spike.
60. **avg_token_frequency_percentile** — avg freq percentile across corpus. | common vs rare wording. | precomputed freq table. | zipf score.
61. **upper_lower_ratio_title** — ratio uppercase/lowercase. | shouting tone. | char classes. | regex.

---

## 5. ASR / текстовые proxy аудио-фичи — 16 фич

62. **asr_confidence_mean** — средняя confidence ASR. | quality proxy. | use ASR token confidences (Whisper). | LM perplexity.
63. **asr_confidence_std** — std deviation. | однородность распознавания. | std of token confidences. | interquartile range.
64. **asr_error_proxy** — WER proxy (no GT). | насколько верен текст. | LM anomaly detection + rare token rate. | char-level oddity metric.
65. **speech_rate_wpm** — слова в минуту. | speaking style. | aligned words/duration. | words/total_video_duration.
66. **filler_word_ratio** — доля слов-паразитов (мм/ээ). | качество речи. | lexicon of fillers + count. | prosody-derived proxies.
67. **pause_density_proxy** — density of pauses inferred from punctuation. | fluency indicator. | forced alignment pause detection. | punctuation proxy.
68. **sentence_intonation_proxy** — percent sentences ending with exclamation/question. | emotional tone. | punctuation analysis + prosody if available. | sentiment cues.
69. **asr_confidence_chunked_min** — min confidence across chunks. | weak-section detector. | sliding window min. | percentile.
70. **named_entities_covered_by_asr** — % of entities confidently recognized. | content coverage. | NER + ASR confidences. | simple mention detection.
71. **oov_rate_asr_tokens** — rate of out-of-vocab tokens. | ASR weakness on names/brands. | compare tokens to vocabulary. | rare word ratio.
72. **aligned_subtitle_coverage** — % of transcript segments with timestamps. | alignment quality. | forced alignment. | presence of VTT/SRT metadata.
73. **speech_character_density** — chars per second spoken. | information density. | chars/duration. | words/duration.
74. **asr_language_mismatch_flag** — if language detector on transcript != declared language. | mismatch signal. | language detection on transcript. | model confidence discrepancies.
75. **acoustic_noise_proxy (textual)** — indicators of noise via repeated garbled tokens. | quality flag. | token anomaly detection. | low ASR confidence clusters.
76. **speaker_count_estimate_textual** — estimate of speakers via heuristics in transcript. | indicator of multi-speaker content. | diarization metadata if available; else speaker change heuristics (names/question cues). | punctuation-based segmentation.

---

## 6. Семантика / темы / keyphrases — 18 фич

77. **transcript_topic_id_top1** — dominant topic id. | topic class. | BERTopic top1. | LDA top1.
78. **transcript_topic_probs_vector[k]** — distribution over k topics. | topic mixture. | BERTopic probabilities. | LDA vector.
79. **topic_entropy** — entropy of topic distribution. | topic diversity. | entropy(transcript_topic_probs). | count of topics>threshold.
80. **top_keyphrases_list** — top-10 keyphrases. | summary for analysts. | KeyBERT. | RAKE.
81. **top_keyphrases_with_scores** — list with relevance scores. | weighting of keyphrases. | KeyBERT scores. | TF-IDF scores.
82. **keyphrase_embedding_centroids** — embeddings of top phrases. | semantically compact summary. | embed top phrases with SBERT. | phrase-level TF-IDF vectors.
83. **topic_coherence_cv** — coherence metric. | quality of topic model. | Cv coherence metric. | PMI-based coherence.
84. **faq_like_question_count** — number of question-like sentences. | identifies tutorial/Q&A style videos. | regex on question punctuation + interrogative word detection. | transformer Q/A classifier.
85. **instructional_language_flag** — presence of imperative verbs ("нажмите", "сделайте"). | tutorial indicator. | POS tagging + imperative detection. | keyword heuristics.
86. **count_named_entities_topk** — counts for top N entities recognized. | analytical entity prominence. | NER frequency. | keyword match lists.
87. **audience_addressing_flag** — presence of direct address (you/ты/вы). | personal tone. | pronoun detection + syntax parsing. | simple token match.
88. **call_to_action_flag** — phrases like "подпишитесь", "лайк". | promotion intent. | phrase list + transformer confirmation. | regex.
89. **faq_embedding_distance** — distance to corpus of FAQ embeddings. | closeness to Q/A content. | faiss nearest neighbor. | keyword overlap.

---

## 7. Комментарии — snapshot-агрегаты (расширенно) — 18 фич

> Snapshot-агрегаты по всем комментариям, имеющимся к `snapshot_time`. Ни temporal rates.

90. **comments_count_total** — общее количество комментариев. | базовый engagement. | count in comments store <= snapshot_time. | platform API stat.
91. **comments_unique_users_count** — уникальные комментаторы. | распределение аудитории. | distinct commenter_id count. | heuristics dedupe.
92. **comments_avg_length_words** — средняя длина комментария. | depth of discussion. | tokenization mean. | char-based proxy.
93. **comments_median_length_words** — медиана. | робастная метрика длины. | median token len. | trimmed mean.
94. **comments_sentiment_mean** — средний sentiment. | общий тон обсуждения. | transformer sentiment mean. | lexicon average.
95. **comments_sentiment_median** — медиана. | устойчивость тональности. | median. | robust aggregator.
96. **comments_sentiment_mode** — наиболее частая категория. | доминантный тон. | argmax over categories. | simple counts.
97. **comments_toxic_share** — доля токсичных комментариев. | safety signal. | fine-tuned toxicity classifier. | profanity list ratio.
98. **comments_support_share** — доля явно поддерживающих комментариев ("классно", "спасибо"). | позитивная реакция. | stance classifier. | sentiment thresholding.
99. **comments_critic_share** — доля негативных/критичных. | негативное восприятие. | stance classifier. | sentiment threshold.
100. **comments_question_share** — доля комментариев, содержащих вопрос. | engagement type indicator. | question detector. | punctuation heuristic.
101. **comments_url_share** — доля комментариев с URL. | external referencing. | URL regex. | mention detection.
102. **comments_mention_share** — доля комментариев упоминающих других пользователей. | social referencing. | regex @mentions. | token heuristics.
103. **comments_average_score** — средний score/likes per comment (если доступно). | comment quality. | aggregate from platform. | proxy using authoritative commenter list.
104. **comments_top_entities** — list of most frequent entities in comments. | аналитика тем аудитории. | NER + frequency. | keyphrase extraction.
105. **comments_embedding_topk_similarity_to_transcript** — topk similarity scores. | насколько комменты цитируют содержание. | embedding similarity. | keyword overlap.
106. **comments_spam_indicator** — flag if high spam-like content. | moderation. | classifier for spam patterns. | heuristics (URLs, repeats).
107. **comments_sentiment_variance** — variance over sentiments. | polarization indication. | variance compute. | IQR.
108. **comments_lexical_diversity** — agg type/token ratio across comments. | conversation richness. | avg type/token. | median variant.

---

## 8. Cross-modal snapshot фичи (текст ↔ визуал / OCR) — 12 фич

109. **ocr_text_embedding** — pooled embedding of OCR'd on-screen text. | multimodal alignment. | OCR (Tesseract/Google Vision) + SBERT embedding. | trick: detect only long OCR strings.
110. **ocr_transcript_cosine** — similarity OCR↔transcript. | говорит ли видео про то, что написано на экране. | cosine embedding. | fuzzy string matching.
111. **visual_tags_embedding** — pooled embedding of detected visual tags (objects/scenes). | multimodal semantics. | VisualProcessor tags → embed. | pretrained imagenet labels mapping.
112. **visual_tags_transcript_overlap_ratio** — fraction of visual tags present in transcript semantics. | coherence. | topk embedding matches. | string overlap mapping.
113. **thumbnail_text_title_cosine** — similarity between thumbnail OCR text (if any) and title. | thumbnail faithfulness. | embedding cosine. | string match.
114. **visual_sentiment_match_flag** — whether visual sentiment (detected) matches textual sentiment. | consistency check. | visual sentiment classifier + compare. | coarse heuristics.
115. **subtitle_ocr_alignment_flag** — whether subtitles match OCR. | subtitle fidelity. | align text segments by timestamp. | high-level similarity.
116. **visual_object_mention_ratio** — portion of visual objects mentioned in transcript. | object mention coverage. | object recognition + NER match. | string overlap.
117. **thumbnail_tags_embedding** — embedding of thumbnail labels. | thumbnail-topic distance. | embed thumbnail tags. | hashing.
118. **title_thumbnail_cosine** — similarity title↔thumbnail embedding. | thumbnail relevance. | cosine. | token overlap.
119. **visual_topic_similarity_to_transcript** — topic-level similarity. | modal consistency. | topic model on visual tags vs transcript. | label mapping.
120. **dominant_scene_text_overlap** — whether dominant scene label appears in transcript. | scene relevance. | top tag match. | manual mapping.

---

## 9. NER / hashtag / keyphrase расширенный набор — 12 фич

121. **named_entities_count_total** — total NE count in transcript+desc. | entity density. | transformer NER. | rule-based.
122. **named_entities_by_type_map** — counts per type (person, org, product, place). | analyst insight. | NER. | simple regex for capitals.
123. **brand_mention_flag** — presence of known brand names. | sponsorship signals. | KB matching after NER. | dictionary matching.
124. **person_mention_count** — number of person mentions. | guest/host detection. | NER. | capitalization heuristics.
125. **product_mention_count** — product mentions count. | monetization indicator. | NER + KB lookup. | keyword lists.
126. **place_mention_count** — place mentions. | topicality. | NER geotagging. | gazetteer lookup.
127. **hashtags_list_normalized** — list of normalized hashtags. | topic markers. | normalize case/variants. | raw list.
128. **hashtag_count_unique** — unique hashtag count. | hashtag diversity. | set cardinality. | raw count.
129. **hashtags_in_title_flag** — whether hashtags in title. | promotional style. | tokenizer check. | regex.
130. **hashtag_to_topic_alignment_score** — how hashtags map to topic distribution. | tag quality. | embed hashtags + topic centroids similarity. | manual mapping.
131. **keyphrase_coverage_rate** — percent of keyphrases that are present in title/description. | coverage metric. | set intersection of keyphrases. | fuzzy match.
132. **entity_salience_scores** — salience per entity (freq * centrality). | prominence metric. | TF-IDF weighting + graph centrality. | frequency alone.

---

## 10. Интерпретируемые и explainability-фичи — 8 фич

133. **top_shap_features_snapshot** — top-5 SHAP feature names and values for model at snapshot (if available). | explainability for analysts. | SHAP explainer on model. | LIME.
134. **top_keywords_contributing_to_prediction** — keywords most strongly correlated with predicted target. | human-readable signal. | gradient-based saliency / LIME. | tf-idf correlation.
135. **counterfactual_suggestion_short** — minimal textual change suggestion to reduce predicted popularity (one-liner). | operational analytic action. | counterfactual generation using surrogate interpretable model. | heuristic rules.
136. **feature_confidence_interval** — per-prediction uncertainty bucket (based on model). | risk-aware decisions. | MC dropout / ensemble variance. | analytic bootstrap.
137. **anomaly_score_text** — how anomalous the text features are vs training distrib. | drift detection. | Mahalanobis distance on normalized features. | z-score threshold.
138. **human_readable_summary** — LLM-generated one-paragraph explanation of key signals (optional feature). | analyst quick-view. | LLM prompt using top features. | template-based summary.
139. **explainability_tags** — list of tags like ["clickbait","highly_informative"]. | quick labels. | rules + LLM enrichment. | manual annotation.
140. **feature_missingness_report_snapshot** — percentage of fields missing. | data quality. | simple counts. | monitoring alert.

---

## 11. Контрольные и служебные поля — 8 фич

141. **snapshot_time (UTC)** — время снимка. | temporal anchor. | ISO8601 timestamp. | platform time.
142. **text_processor_version** — версия pipeline. | reproducibility. | semantic version + git hash. | docker tag.
143. **embedding_model_version** — версия эмбеддингов. | воспроизводимость эмбеддингов. | model name + commit. | timestamp.
144. **content_hash** — hash(title+desc+transcript). | кэш-ключ. | SHA256(normalized). | MD5.
145. **source_platform** — origin (YouTube, TikTok). | канал маршрутизации. | platform api field. | inferred via URL.
146. **language** — primary language. | model selection. | fastText. | cld3.
147. **schema_version** — версия фичей. | совместимость. | semantic versioning. | date tag.
148. **ingest_id** — unique ingestion identifier. | tracing. | UUID. | platform id.
149. **feature_store_ref** — pointer to where features stored. | engineering. | URI. | local path.
150. **qa_checkpoint_flag** — if QA passed automatic checks. | pipeline gating. | set of pre-checks. | manual flag.

---

## 12. Пример CSV/Parquet схема (выдержка)
- `snapshot_time: timestamp`  
- `text_processor_version: string`  
- `content_hash: string`  
- `source_platform: string`  
- `language: string`  
- `title: string`  
- `title_len_words: int`  
- `title_embedding_ref: string`  
- `title_clickbait_score: float`  
- `description_embedding_ref: string`  
- `transcript_embedding_ref: string`  
- `transcript_topic_probs: json`  
- `comments_count_total: int`  
- `comments_sentiment_mean: float`  
- `ocr_transcript_cosine: float`  
- `named_entities_by_type: json`  
- `top_keyphrases: json`  
- `feature_missingness: float`  

(в полном экспорте — все 150+ колонок; векторные поля — ссылки на vector-store или бинарные колонки в parquet)

---

## 13. Резюме и дальнейшие шаги
- Документ расширён до ~150–160 snapshot-фич. Он покрывает эмбеддинги, лексические признаки, ASR-прокси, тематику, дополнительные агрегаты по комментариям, cross-modal согласованность и explainability.
- Дальше могу: 
  1. экспортировать этот расширенный MD в `.md` файл и сделать доступный для скачивания,  
  2. сгенерировать CSV/Parquet schema file (schema.json + example row),  
  3. написать PySpark/Python pipeline (pseudocode) для извлечения всех этих полей (включая batching и кеш эмбеддингов). 

---

*Конец расширенного документа.*