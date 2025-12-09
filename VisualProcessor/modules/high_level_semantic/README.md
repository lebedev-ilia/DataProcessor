🧠 High-Level Video Semantics Module
Semantic Understanding & Multimodal Feature Extraction

Этот модуль отвечает за извлечение высокоуровневой семантики видео, включая:

смысловые эмбеддинги сцен и видео

мультимодальные связи между видео, аудио и (опционально) текстом

события, эмоции, сюжетные пики

жанры, стили, типы контента

концепты, темы, и narrative-структуру

🤝 Важно:
Модуль НЕ анализирует текст напрямую.
Все текстовые фичи приходят из TextProcessor / OCRProcessor / ASRProcessor.

📦 Функциональные блоки
1. 🎬 Scene-Level Semantic Embeddings

Для каждой сцены (shot) вычисляется:

scene_embedding — CLIP-подобный визуальный embedding (1024–4096 dims)

scene_semantic_strength — насколько сцена "информативна"

scene_transition_score — семантические скачки при смене сцен

Используется для:

сегментации видео по смыслам

анализа тем, разнообразия контента

построения семантической карты видео

2. 🧩 Video-Level Semantic Embeddings

Объединение всех сценовых эмбеддингов:

video_mean_embedding

video_weighted_embedding (вес — динамичность, внимание, эмоции)

video_semantic_variance

embedding_contrast (насколько разнообразен смысл)

Используется в:

рекомендациях

поиске похожего контента

качественной оценке уникальности

3. 🎭 High-Level Topic / Concept Detection

Используются:

мультимодальные модели (CLIP zero-shot, VideoCLIP)

caption embeddings (передаётся текстовая часть из другого модуля)

Результат:

topic_probabilities

topic_diversity_score

topic_transition_rate

main_concept_embedding

secondary_concepts

4. ⚡ Event Detection & Key Moments

Модуль определяет ключевые события, такие как:

переходы

внезапные действия

элементы юмора

tutorial-шаги

реакции

музыкальные пики

визуальные "удерживающие" моменты

Фичи:

n_events

event_importance_score

event_timeline

event_type

5. 😊 Emotion / Sentiment (Video-Level)

Модуль принимает эмоции:

лиц (FaceProcessor)

аудио (AudioProcessor → speech prosody)

текстовые эмоции (приходят из TextProcessor)

Фичи:

avg_emotion_valence

emotion_variance

peak_emotion_intensity

emotion_arc_start/mid/end

emotion_timeline

6. 📖 Narrative & Story Structure

Через CaptionEmbeddings и Visual-Semantic Flow:

narrative_embedding

story_flow_score

narrative_complexity_score

hook_quality_score (первые ~3–5 сек)

climax_strength_score

resolution_presence_flag

7. 🔀 Multimodal Semantic Analysis

Модуль объединяет:

визуальные эмбеддинги

аудио-эмбеддинги

текстовые эмбеддинги из других процессоров

Фичи:

multimodal_attention_score

multimodal_event_alignment

cross_modal_novelty

modal_consistency_score (визуал ↔ звук ↔ текст)

8. 🏷 High-Level Classification

Мультимодальная классификация:

genre_probabilities

style_probability_distribution

content_type_probabilities

video_archetype (обучающее, развлекательное, реакция, обзор и пр.)

📊 Итоговые фичи модуля
🎬 Scene-level

scene_embedding

scene_transition_score

scene_semantic_strength

🧩 Video-level

video_mean_embedding

video_weighted_embedding

video_semantic_variance

embedding_contrast

🏷 Topics & Concepts

topic_probabilities

topic_diversity_score

topic_transition_rate

⚡ Events

n_events

event_importance_score

event_timeline

😊 Emotion

emotion_arc_start/mid/end

avg_emotion_valence

peak_emotion_intensity

📖 Narrative

narrative_embedding

story_flow_score

climax_strength

hook_quality_score

🔀 Multimodal

multimodal_attention_score

multimodal_event_alignment

cross_modal_novelty

modal_consistency_score

🏷 Labels

genre_probabilities

style_probability

content_type_probabilities

🔌 Взаимодействие с другими модулями
Источник	Что подаёт
TextProcessor	текстовые embedding’и, темы, эмоции, timeline
AudioProcessor	аудио-эмбеддинги, энергия, музыкальные пики, prosody
FaceProcessor	эмоции лица, интенсивность, temporal curve
VideoDynamicsProcessor	движение, динамика, action peaks
🛠 Архитектура пайплайна
Video → SceneSplitter 
      → VisualCLIPEncoder → scene_embeddings
      → SceneAggregator → video_embeddings
      → EventDetector → events
      → EmotionMerger (видео+аудио+текст)
      → NarrativeAnalyzer
      → TopicClassifier (multimodal)
      → MultimodalFusion
      → HighLevelClassifier (genre/style/type)
      → Final Feature Package

🧪 Output формата

Модуль возвращает:

{
    "scene_embeddings": [...],
    "video_embedding": [...],
    "topics": {...},
    "events": {...},
    "emotions": {...},
    "narrative": {...},
    "multimodal": {...},
    "classifications": {...},
}

🚀 Цели модуля

дать полное смысловое представление видео

оценить динамику, storyline, удержание
определить жанр, стиль, тематику

сформировать embedding, пригодный для:
    рекомендаций
    кластеризации
    анализа качества видео
    ML-моделей (регрессия popularity_score)