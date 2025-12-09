"""
# 📊 8. Метрики сравнений

Модуль для вычисления метрик схожести между видео.
Сравнивает текущее видео с референсными видео по различным аспектам:
- Семантическая схожесть
- Тематическое пересечение
- Визуальный стиль и композиция
- Текст и OCR
- Аудио и речь
- Эмоции и поведение
- Временной ритм
- Высокоуровневые сравнительные оценки
- Групповые метрики для батчей видео
"""

from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import jaccard_score
from scipy.stats import wasserstein_distance
import warnings
warnings.filterwarnings('ignore')


class SimilarityMetrics:
    """
    Класс для вычисления метрик схожести между видео.
    
    Сравнивает текущее видео с набором референсных видео по различным аспектам:
    - Семантические embeddings
    - Тематические концепты
    - Визуальный стиль (цвет, свет, композиция)
    - Текст и OCR
    - Аудио характеристики
    - Эмоции и поведение
    - Временной ритм и pacing
    """
    
    def __init__(
        self,
        top_n: int = 10,
        similarity_weights: Optional[Dict[str, float]] = None
    ):
        """
        Args:
            top_n: Количество топ видео для усреднения метрик
            similarity_weights: Веса для различных категорий метрик при вычислении overall_similarity_score
        """
        self.top_n = top_n
        
        # Веса по умолчанию для overall_similarity_score
        self.similarity_weights = similarity_weights or {
            'semantic': 0.25,
            'topics': 0.15,
            'visual': 0.15,
            'text': 0.10,
            'audio': 0.15,
            'emotion': 0.10,
            'temporal': 0.10
        }
    
    # ==================== A. Semantic Similarity ====================
    
    def compute_semantic_similarity(
        self,
        video_embedding: np.ndarray,
        reference_embeddings: List[np.ndarray]
    ) -> Dict[str, float]:
        """
        Вычисляет семантическую схожесть на основе embeddings.
        
        Args:
            video_embedding: Embedding текущего видео (1D array)
            reference_embeddings: Список embeddings референсных видео
            
        Returns:
            Словарь с метриками семантической схожести
        """
        if len(reference_embeddings) == 0:
            return {
                'semantic_similarity_mean': 0.0,
                'semantic_similarity_max': 0.0,
                'semantic_similarity_min': 0.0,
                'semantic_novelty_score': 1.0
            }
        
        # Нормализуем embeddings
        video_emb_norm = video_embedding / (np.linalg.norm(video_embedding) + 1e-10)
        
        similarities = []
        for ref_emb in reference_embeddings:
            ref_emb_norm = ref_emb / (np.linalg.norm(ref_emb) + 1e-10)
            # Cosine similarity
            sim = np.dot(video_emb_norm, ref_emb_norm)
            similarities.append(float(sim))
        
        similarities = np.array(similarities)
        
        # Сортируем и берем топ-N
        top_similarities = np.sort(similarities)[-self.top_n:] if len(similarities) > self.top_n else similarities
        
        return {
            'semantic_similarity_mean': float(np.mean(top_similarities)),
            'semantic_similarity_max': float(np.max(similarities)),
            'semantic_similarity_min': float(np.min(similarities)),
            'semantic_novelty_score': float(1.0 - np.max(similarities))
        }
    
    # ==================== B. Topic / Concept Overlap ====================
    
    def compute_topic_overlap(
        self,
        video_topics: Union[List[str], np.ndarray, Dict[str, float]],
        reference_topics_list: List[Union[List[str], np.ndarray, Dict[str, float]]]
    ) -> Dict[str, float]:
        """
        Вычисляет тематическое пересечение между видео.
        
        Args:
            video_topics: Темы текущего видео (список строк, массив вероятностей или словарь)
            reference_topics_list: Список тем референсных видео
            
        Returns:
            Словарь с метриками тематического пересечения
        """
        if len(reference_topics_list) == 0:
            return {
                'topic_overlap_score': 0.0,
                'topic_diversity_comparison': 0.0,
                'key_concept_match_ratio': 0.0
            }
        
        # Преобразуем в множества ключевых слов, если нужно
        def to_set(topics):
            if isinstance(topics, dict):
                # Берем топ-10 тем по вероятности
                sorted_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)
                return set([t[0] for t in sorted_topics[:10]])
            elif isinstance(topics, (list, np.ndarray)):
                if len(topics) > 0 and isinstance(topics[0], str):
                    return set(topics[:20])  # Берем первые 20
                else:
                    # Это вероятности, берем индексы с вероятностью > 0.1
                    return set(np.where(np.array(topics) > 0.1)[0].astype(str))
            return set()
        
        video_topics_set = to_set(video_topics)
        
        overlap_scores = []
        diversity_diffs = []
        concept_matches = []
        
        for ref_topics in reference_topics_list:
            ref_topics_set = to_set(ref_topics)
            
            # Jaccard similarity
            if len(video_topics_set) == 0 and len(ref_topics_set) == 0:
                jaccard = 1.0
            elif len(video_topics_set) == 0 or len(ref_topics_set) == 0:
                jaccard = 0.0
            else:
                intersection = len(video_topics_set & ref_topics_set)
                union = len(video_topics_set | ref_topics_set)
                jaccard = intersection / union if union > 0 else 0.0
            
            overlap_scores.append(jaccard)
            
            # Разница в разнообразии тем
            diversity_diff = abs(len(video_topics_set) - len(ref_topics_set)) / max(len(video_topics_set), len(ref_topics_set), 1)
            diversity_diffs.append(diversity_diff)
            
            # Доля совпадающих ключевых концептов
            if len(video_topics_set) > 0:
                match_ratio = len(video_topics_set & ref_topics_set) / len(video_topics_set)
            else:
                match_ratio = 0.0
            concept_matches.append(match_ratio)
        
        return {
            'topic_overlap_score': float(np.mean(overlap_scores)),
            'topic_diversity_comparison': float(np.mean(diversity_diffs)),
            'key_concept_match_ratio': float(np.mean(concept_matches))
        }
    
    # ==================== C. Style & Composition Similarity ====================
    
    def compute_style_similarity(
        self,
        video_visual_features: Dict[str, Any],
        reference_visual_features_list: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Вычисляет схожесть визуального стиля и композиции.
        
        Args:
            video_visual_features: Визуальные фичи текущего видео (цвет, свет, типы кадров, монтаж, движение)
            reference_visual_features_list: Список визуальных фичей референсных видео
            
        Returns:
            Словарь с метриками визуальной схожести
        """
        if len(reference_visual_features_list) == 0:
            return {
                'color_histogram_similarity': 0.0,
                'lighting_pattern_similarity': 0.0,
                'shot_type_distribution_similarity': 0.0,
                'cut_rate_similarity': 0.0,
                'motion_pattern_similarity': 0.0
            }
        
        color_sims = []
        lighting_sims = []
        shot_type_sims = []
        cut_rate_sims = []
        motion_sims = []
        
        for ref_features in reference_visual_features_list:
            # Color histogram similarity
            if 'color_histogram' in video_visual_features and 'color_histogram' in ref_features:
                hist1 = np.array(video_visual_features['color_histogram']).flatten()
                hist2 = np.array(ref_features['color_histogram']).flatten()
                if len(hist1) == len(hist2):
                    # Cosine similarity для гистограмм
                    hist1_norm = hist1 / (np.linalg.norm(hist1) + 1e-10)
                    hist2_norm = hist2 / (np.linalg.norm(hist2) + 1e-10)
                    color_sim = np.dot(hist1_norm, hist2_norm)
                    color_sims.append(color_sim)
            
            # Lighting pattern similarity
            if 'lighting_features' in video_visual_features and 'lighting_features' in ref_features:
                light1 = np.array(video_visual_features['lighting_features'])
                light2 = np.array(ref_features['lighting_features'])
                if light1.shape == light2.shape:
                    light_sim = 1.0 - cosine(light1.flatten(), light2.flatten())
                    lighting_sims.append(max(0.0, light_sim))
            
            # Shot type distribution similarity
            if 'shot_type_distribution' in video_visual_features and 'shot_type_distribution' in ref_features:
                dist1 = np.array(video_visual_features['shot_type_distribution'])
                dist2 = np.array(ref_features['shot_type_distribution'])
                if len(dist1) == len(dist2):
                    # Нормализуем распределения
                    dist1_norm = dist1 / (dist1.sum() + 1e-10)
                    dist2_norm = dist2 / (dist2.sum() + 1e-10)
                    # Earth Mover's Distance или cosine similarity
                    shot_sim = 1.0 - wasserstein_distance(dist1_norm, dist2_norm) / (np.max(dist1_norm) + np.max(dist2_norm) + 1e-10)
                    shot_type_sims.append(max(0.0, min(1.0, shot_sim)))
            
            # Cut rate similarity
            if 'cut_rate' in video_visual_features and 'cut_rate' in ref_features:
                cut1 = video_visual_features['cut_rate']
                cut2 = ref_features['cut_rate']
                # Нормализуем разницу
                max_cut = max(abs(cut1), abs(cut2), 1.0)
                cut_sim = 1.0 - abs(cut1 - cut2) / max_cut
                cut_rate_sims.append(max(0.0, cut_sim))
            
            # Motion pattern similarity
            if 'motion_pattern' in video_visual_features and 'motion_pattern' in ref_features:
                motion1 = np.array(video_visual_features['motion_pattern'])
                motion2 = np.array(ref_features['motion_pattern'])
                if len(motion1) == len(motion2):
                    # Корреляция между паттернами движения
                    try:
                        corr, _ = pearsonr(motion1, motion2)
                        motion_sims.append(max(0.0, corr) if not np.isnan(corr) else 0.0)
                    except:
                        motion_sims.append(0.0)
        
        return {
            'color_histogram_similarity': float(np.mean(color_sims)) if color_sims else 0.0,
            'lighting_pattern_similarity': float(np.mean(lighting_sims)) if lighting_sims else 0.0,
            'shot_type_distribution_similarity': float(np.mean(shot_type_sims)) if shot_type_sims else 0.0,
            'cut_rate_similarity': float(np.mean(cut_rate_sims)) if cut_rate_sims else 0.0,
            'motion_pattern_similarity': float(np.mean(motion_sims)) if motion_sims else 0.0
        }
    
    # ==================== D. Text & OCR Similarity ====================
    
    def compute_text_similarity(
        self,
        video_text_features: Dict[str, Any],
        reference_text_features_list: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Вычисляет схожесть текста и OCR.
        
        Args:
            video_text_features: Текстовые фичи текущего видео (OCR embeddings, layout, timing)
            reference_text_features_list: Список текстовых фичей референсных видео
            
        Returns:
            Словарь с метриками текстовой схожести
        """
        if len(reference_text_features_list) == 0:
            return {
                'ocr_text_semantic_similarity': 0.0,
                'text_layout_similarity': 0.0,
                'text_timing_similarity': 0.0
            }
        
        ocr_sims = []
        layout_sims = []
        timing_sims = []
        
        for ref_features in reference_text_features_list:
            # OCR text semantic similarity
            if 'ocr_embedding' in video_text_features and 'ocr_embedding' in ref_features:
                emb1 = np.array(video_text_features['ocr_embedding'])
                emb2 = np.array(ref_features['ocr_embedding'])
                if emb1.shape == emb2.shape:
                    emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-10)
                    emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-10)
                    ocr_sim = np.dot(emb1_norm, emb2_norm)
                    ocr_sims.append(ocr_sim)
            
            # Text layout similarity (позиции, длина, font size)
            if 'text_layout' in video_text_features and 'text_layout' in ref_features:
                layout1 = np.array(video_text_features['text_layout'])
                layout2 = np.array(ref_features['text_layout'])
                if len(layout1) == len(layout2):
                    layout_sim = 1.0 - cosine(layout1.flatten(), layout2.flatten())
                    layout_sims.append(max(0.0, layout_sim))
            
            # Text timing similarity
            if 'text_timing' in video_text_features and 'text_timing' in ref_features:
                timing1 = np.array(video_text_features['text_timing'])
                timing2 = np.array(ref_features['text_timing'])
                if len(timing1) == len(timing2):
                    try:
                        corr, _ = pearsonr(timing1, timing2)
                        timing_sims.append(max(0.0, corr) if not np.isnan(corr) else 0.0)
                    except:
                        timing_sims.append(0.0)
        
        return {
            'ocr_text_semantic_similarity': float(np.mean(ocr_sims)) if ocr_sims else 0.0,
            'text_layout_similarity': float(np.mean(layout_sims)) if layout_sims else 0.0,
            'text_timing_similarity': float(np.mean(timing_sims)) if timing_sims else 0.0
        }
    
    # ==================== E. Audio / Speech Similarity ====================
    
    def compute_audio_similarity(
        self,
        video_audio_features: Dict[str, Any],
        reference_audio_features_list: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Вычисляет схожесть аудио характеристик.
        
        Args:
            video_audio_features: Аудио фичи текущего видео (embeddings, tempo, energy)
            reference_audio_features_list: Список аудио фичей референсных видео
            
        Returns:
            Словарь с метриками аудио схожести
        """
        if len(reference_audio_features_list) == 0:
            return {
                'audio_embedding_similarity': 0.0,
                'speech_content_similarity': 0.0,
                'music_tempo_similarity': 0.0,
                'audio_energy_pattern_similarity': 0.0
            }
        
        audio_emb_sims = []
        speech_sims = []
        tempo_sims = []
        energy_sims = []
        
        for ref_features in reference_audio_features_list:
            # Audio embedding similarity
            if 'audio_embedding' in video_audio_features and 'audio_embedding' in ref_features:
                emb1 = np.array(video_audio_features['audio_embedding'])
                emb2 = np.array(ref_features['audio_embedding'])
                if emb1.shape == emb2.shape:
                    emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-10)
                    emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-10)
                    audio_sim = np.dot(emb1_norm, emb2_norm)
                    audio_emb_sims.append(audio_sim)
            
            # Speech content similarity (по ASR)
            if 'speech_embedding' in video_audio_features and 'speech_embedding' in ref_features:
                speech1 = np.array(video_audio_features['speech_embedding'])
                speech2 = np.array(ref_features['speech_embedding'])
                if speech1.shape == speech2.shape:
                    speech1_norm = speech1 / (np.linalg.norm(speech1) + 1e-10)
                    speech2_norm = speech2 / (np.linalg.norm(speech2) + 1e-10)
                    speech_sim = np.dot(speech1_norm, speech2_norm)
                    speech_sims.append(speech_sim)
            
            # Music tempo similarity
            if 'tempo' in video_audio_features and 'tempo' in ref_features:
                tempo1 = video_audio_features['tempo']
                tempo2 = ref_features['tempo']
                max_tempo = max(abs(tempo1), abs(tempo2), 1.0)
                tempo_sim = 1.0 - abs(tempo1 - tempo2) / max_tempo
                tempo_sims.append(max(0.0, tempo_sim))
            
            # Audio energy pattern similarity
            if 'energy_pattern' in video_audio_features and 'energy_pattern' in ref_features:
                energy1 = np.array(video_audio_features['energy_pattern'])
                energy2 = np.array(ref_features['energy_pattern'])
                if len(energy1) == len(energy2):
                    try:
                        corr, _ = pearsonr(energy1, energy2)
                        energy_sims.append(max(0.0, corr) if not np.isnan(corr) else 0.0)
                    except:
                        energy_sims.append(0.0)
        
        return {
            'audio_embedding_similarity': float(np.mean(audio_emb_sims)) if audio_emb_sims else 0.0,
            'speech_content_similarity': float(np.mean(speech_sims)) if speech_sims else 0.0,
            'music_tempo_similarity': float(np.mean(tempo_sims)) if tempo_sims else 0.0,
            'audio_energy_pattern_similarity': float(np.mean(energy_sims)) if energy_sims else 0.0
        }
    
    # ==================== F. Emotion & Behavior Similarity ====================
    
    def compute_emotion_behavior_similarity(
        self,
        video_emotion_features: Dict[str, Any],
        reference_emotion_features_list: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Вычисляет схожесть эмоций и поведения.
        
        Args:
            video_emotion_features: Фичи эмоций/поведения текущего видео (emotion curve, pose, behavior)
            reference_emotion_features_list: Список фичей эмоций/поведения референсных видео
            
        Returns:
            Словарь с метриками схожести эмоций и поведения
        """
        if len(reference_emotion_features_list) == 0:
            return {
                'emotion_curve_similarity': 0.0,
                'pose_motion_similarity': 0.0,
                'behavior_pattern_similarity': 0.0
            }
        
        emotion_sims = []
        pose_sims = []
        behavior_sims = []
        
        for ref_features in reference_emotion_features_list:
            # Emotion curve similarity
            if 'emotion_curve' in video_emotion_features and 'emotion_curve' in ref_features:
                curve1 = np.array(video_emotion_features['emotion_curve'])
                curve2 = np.array(ref_features['emotion_curve'])
                if len(curve1) == len(curve2):
                    try:
                        corr, _ = pearsonr(curve1, curve2)
                        emotion_sims.append(max(0.0, corr) if not np.isnan(corr) else 0.0)
                    except:
                        emotion_sims.append(0.0)
            
            # Pose motion similarity
            if 'pose_motion' in video_emotion_features and 'pose_motion' in ref_features:
                pose1 = np.array(video_emotion_features['pose_motion'])
                pose2 = np.array(ref_features['pose_motion'])
                if pose1.shape == pose2.shape:
                    pose1_norm = pose1 / (np.linalg.norm(pose1) + 1e-10)
                    pose2_norm = pose2 / (np.linalg.norm(pose2) + 1e-10)
                    pose_sim = np.dot(pose1_norm.flatten(), pose2_norm.flatten())
                    pose_sims.append(pose_sim)
            
            # Behavior pattern similarity
            if 'behavior_pattern' in video_emotion_features and 'behavior_pattern' in ref_features:
                behavior1 = np.array(video_emotion_features['behavior_pattern'])
                behavior2 = np.array(ref_features['behavior_pattern'])
                if len(behavior1) == len(behavior2):
                    try:
                        corr, _ = pearsonr(behavior1, behavior2)
                        behavior_sims.append(max(0.0, corr) if not np.isnan(corr) else 0.0)
                    except:
                        behavior_sims.append(0.0)
        
        return {
            'emotion_curve_similarity': float(np.mean(emotion_sims)) if emotion_sims else 0.0,
            'pose_motion_similarity': float(np.mean(pose_sims)) if pose_sims else 0.0,
            'behavior_pattern_similarity': float(np.mean(behavior_sims)) if behavior_sims else 0.0
        }
    
    # ==================== G. Temporal / Pacing Similarity ====================
    
    def compute_temporal_similarity(
        self,
        video_pacing_features: Dict[str, Any],
        reference_pacing_features_list: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Вычисляет схожесть временного ритма и pacing.
        
        Args:
            video_pacing_features: Фичи pacing текущего видео (pacing curve, shot duration, scene length)
            reference_pacing_features_list: Список фичей pacing референсных видео
            
        Returns:
            Словарь с метриками схожести временного ритма
        """
        if len(reference_pacing_features_list) == 0:
            return {
                'pacing_curve_similarity': 0.0,
                'shot_duration_distribution_similarity': 0.0,
                'scene_length_similarity': 0.0,
                'temporal_pattern_novelty': 1.0
            }
        
        pacing_sims = []
        shot_duration_sims = []
        scene_length_sims = []
        
        for ref_features in reference_pacing_features_list:
            # Pacing curve similarity
            if 'pacing_curve' in video_pacing_features and 'pacing_curve' in ref_features:
                curve1 = np.array(video_pacing_features['pacing_curve'])
                curve2 = np.array(ref_features['pacing_curve'])
                if len(curve1) == len(curve2):
                    try:
                        corr, _ = pearsonr(curve1, curve2)
                        pacing_sims.append(max(0.0, corr) if not np.isnan(corr) else 0.0)
                    except:
                        pacing_sims.append(0.0)
            
            # Shot duration distribution similarity
            if 'shot_duration_distribution' in video_pacing_features and 'shot_duration_distribution' in ref_features:
                dist1 = np.array(video_pacing_features['shot_duration_distribution'])
                dist2 = np.array(ref_features['shot_duration_distribution'])
                if len(dist1) == len(dist2):
                    dist1_norm = dist1 / (dist1.sum() + 1e-10)
                    dist2_norm = dist2 / (dist2.sum() + 1e-10)
                    # Wasserstein distance
                    wd = wasserstein_distance(dist1_norm, dist2_norm)
                    max_wd = np.max(dist1_norm) + np.max(dist2_norm)
                    shot_sim = 1.0 - wd / (max_wd + 1e-10)
                    shot_duration_sims.append(max(0.0, min(1.0, shot_sim)))
            
            # Scene length similarity
            if 'scene_lengths' in video_pacing_features and 'scene_lengths' in ref_features:
                len1 = np.array(video_pacing_features['scene_lengths'])
                len2 = np.array(ref_features['scene_lengths'])
                if len(len1) > 0 and len(len2) > 0:
                    # Сравниваем средние и std
                    mean1, std1 = np.mean(len1), np.std(len1)
                    mean2, std2 = np.mean(len2), np.std(len2)
                    max_mean = max(abs(mean1), abs(mean2), 1.0)
                    max_std = max(abs(std1), abs(std2), 1.0)
                    mean_sim = 1.0 - abs(mean1 - mean2) / max_mean
                    std_sim = 1.0 - abs(std1 - std2) / max_std
                    scene_sim = (mean_sim + std_sim) / 2.0
                    scene_length_sims.append(max(0.0, scene_sim))
        
        # Temporal pattern novelty = 1 - mean similarity
        mean_pacing_sim = np.mean(pacing_sims) if pacing_sims else 0.0
        
        return {
            'pacing_curve_similarity': float(np.mean(pacing_sims)) if pacing_sims else 0.0,
            'shot_duration_distribution_similarity': float(np.mean(shot_duration_sims)) if shot_duration_sims else 0.0,
            'scene_length_similarity': float(np.mean(scene_length_sims)) if scene_length_sims else 0.0,
            'temporal_pattern_novelty': float(1.0 - mean_pacing_sim)
        }
    
    # ==================== H. High-level Comparative Scores ====================
    
    def compute_high_level_scores(
        self,
        all_similarity_metrics: Dict[str, float],
        reference_videos_metadata: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, float]:
        """
        Вычисляет высокоуровневые сравнительные оценки.
        
        Args:
            all_similarity_metrics: Словарь со всеми метриками схожести из категорий A-G
            reference_videos_metadata: Метаданные референсных видео (для trend_alignment и viral_pattern)
            
        Returns:
            Словарь с высокоуровневыми оценками
        """
        # Overall similarity score = weighted sum
        weights = self.similarity_weights
        
        semantic_score = all_similarity_metrics.get('semantic_similarity_mean', 0.0)
        topics_score = all_similarity_metrics.get('topic_overlap_score', 0.0)
        visual_score = np.mean([
            all_similarity_metrics.get('color_histogram_similarity', 0.0),
            all_similarity_metrics.get('lighting_pattern_similarity', 0.0),
            all_similarity_metrics.get('shot_type_distribution_similarity', 0.0)
        ])
        text_score = all_similarity_metrics.get('ocr_text_semantic_similarity', 0.0)
        audio_score = all_similarity_metrics.get('audio_embedding_similarity', 0.0)
        emotion_score = all_similarity_metrics.get('emotion_curve_similarity', 0.0)
        temporal_score = all_similarity_metrics.get('pacing_curve_similarity', 0.0)
        
        overall_similarity = (
            weights['semantic'] * semantic_score +
            weights['topics'] * topics_score +
            weights['visual'] * visual_score +
            weights['text'] * text_score +
            weights['audio'] * audio_score +
            weights['emotion'] * emotion_score +
            weights['temporal'] * temporal_score
        )
        
        uniqueness_score = 1.0 - overall_similarity
        
        # Trend alignment score (насколько похоже на топ видео в нише)
        # Если есть метаданные с популярностью, используем их
        trend_alignment = overall_similarity  # По умолчанию = overall_similarity
        
        if reference_videos_metadata:
            # Можно взвесить по популярности референсных видео
            popular_videos_similarity = overall_similarity  # Упрощенная версия
            trend_alignment = popular_videos_similarity
        
        # Viral pattern score (схожесть с успешными видео)
        viral_pattern = overall_similarity  # Упрощенная версия
        
        if reference_videos_metadata:
            # Можно фильтровать только вирусные видео и считать схожесть с ними
            viral_videos = [v for v in reference_videos_metadata if v.get('is_viral', False)]
            if len(viral_videos) > 0:
                viral_pattern = overall_similarity  # Упрощенная версия
        
        return {
            'overall_similarity_score': float(overall_similarity),
            'uniqueness_score': float(uniqueness_score),
            'trend_alignment_score': float(trend_alignment),
            'viral_pattern_score': float(viral_pattern)
        }
    
    # ==================== I. Group / Batch Metrics ====================
    
    def compute_batch_metrics(
        self,
        video_embeddings: List[np.ndarray],
        video_features_list: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Вычисляет групповые метрики для батча видео.
        
        Args:
            video_embeddings: Список embeddings всех видео в батче
            video_features_list: Список фичей всех видео в батче
            
        Returns:
            Словарь с групповыми метриками
        """
        if len(video_embeddings) < 2:
            return {
                'cluster_similarity_mean': 0.0,
                'inter_video_variance_topics': 0.0,
                'inter_video_variance_emotions': 0.0,
                'inter_video_variance_editing': 0.0,
                'inter_video_variance_audio': 0.0
            }
        
        # Cluster similarity metrics (средняя схожесть между всеми парами)
        pairwise_similarities = []
        for i in range(len(video_embeddings)):
            for j in range(i + 1, len(video_embeddings)):
                emb1 = video_embeddings[i] / (np.linalg.norm(video_embeddings[i]) + 1e-10)
                emb2 = video_embeddings[j] / (np.linalg.norm(video_embeddings[j]) + 1e-10)
                sim = np.dot(emb1, emb2)
                pairwise_similarities.append(sim)
        
        cluster_similarity = np.mean(pairwise_similarities) if pairwise_similarities else 0.0
        
        # Inter-video variance по различным аспектам
        topics_variance = 0.0
        emotions_variance = 0.0
        editing_variance = 0.0
        audio_variance = 0.0
        
        if video_features_list:
            # Topics variance
            topic_vectors = []
            for features in video_features_list:
                if 'topic_embedding' in features:
                    topic_vectors.append(features['topic_embedding'])
            if topic_vectors:
                topics_variance = float(np.var([np.linalg.norm(t) for t in topic_vectors]))
            
            # Emotions variance
            emotion_means = []
            for features in video_features_list:
                if 'emotion_mean' in features:
                    emotion_means.append(features['emotion_mean'])
            if emotion_means:
                emotions_variance = float(np.var(emotion_means))
            
            # Editing variance (cut rate)
            cut_rates = []
            for features in video_features_list:
                if 'cut_rate' in features:
                    cut_rates.append(features['cut_rate'])
            if cut_rates:
                editing_variance = float(np.var(cut_rates))
            
            # Audio variance (tempo)
            tempos = []
            for features in video_features_list:
                if 'tempo' in features:
                    tempos.append(features['tempo'])
            if tempos:
                audio_variance = float(np.var(tempos))
        
        return {
            'cluster_similarity_mean': float(cluster_similarity),
            'inter_video_variance_topics': float(topics_variance),
            'inter_video_variance_emotions': float(emotions_variance),
            'inter_video_variance_editing': float(editing_variance),
            'inter_video_variance_audio': float(audio_variance)
        }
    
    # ==================== Main Method ====================
    
    def extract_all(
        self,
        video_embedding: np.ndarray,
        reference_embeddings: List[np.ndarray],
        video_topics: Optional[Union[List[str], np.ndarray, Dict[str, float]]] = None,
        reference_topics_list: Optional[List[Union[List[str], np.ndarray, Dict[str, float]]]] = None,
        video_visual_features: Optional[Dict[str, Any]] = None,
        reference_visual_features_list: Optional[List[Dict[str, Any]]] = None,
        video_text_features: Optional[Dict[str, Any]] = None,
        reference_text_features_list: Optional[List[Dict[str, Any]]] = None,
        video_audio_features: Optional[Dict[str, Any]] = None,
        reference_audio_features_list: Optional[List[Dict[str, Any]]] = None,
        video_emotion_features: Optional[Dict[str, Any]] = None,
        reference_emotion_features_list: Optional[List[Dict[str, Any]]] = None,
        video_pacing_features: Optional[Dict[str, Any]] = None,
        reference_pacing_features_list: Optional[List[Dict[str, Any]]] = None,
        reference_videos_metadata: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, Any]:
        """
        Главный метод для вычисления всех метрик схожести.
        
        Args:
            video_embedding: Embedding текущего видео
            reference_embeddings: Список embeddings референсных видео
            video_topics: Темы текущего видео
            reference_topics_list: Список тем референсных видео
            video_visual_features: Визуальные фичи текущего видео
            reference_visual_features_list: Список визуальных фичей референсных видео
            video_text_features: Текстовые фичи текущего видео
            reference_text_features_list: Список текстовых фичей референсных видео
            video_audio_features: Аудио фичи текущего видео
            reference_audio_features_list: Список аудио фичей референсных видео
            video_emotion_features: Фичи эмоций/поведения текущего видео
            reference_emotion_features_list: Список фичей эмоций/поведения референсных видео
            video_pacing_features: Фичи pacing текущего видео
            reference_pacing_features_list: Список фичей pacing референсных видео
            reference_videos_metadata: Метаданные референсных видео (опционально)
            
        Returns:
            Словарь со всеми метриками схожести
        """
        features = {}
        
        # A. Semantic similarity
        semantic_metrics = self.compute_semantic_similarity(video_embedding, reference_embeddings)
        features.update(semantic_metrics)
        
        # B. Topic overlap
        if video_topics is not None and reference_topics_list is not None:
            topic_metrics = self.compute_topic_overlap(video_topics, reference_topics_list)
            features.update(topic_metrics)
        else:
            features.update({
                'topic_overlap_score': 0.0,
                'topic_diversity_comparison': 0.0,
                'key_concept_match_ratio': 0.0
            })
        
        # C. Style & Composition
        if video_visual_features is not None and reference_visual_features_list is not None:
            style_metrics = self.compute_style_similarity(video_visual_features, reference_visual_features_list)
            features.update(style_metrics)
        else:
            features.update({
                'color_histogram_similarity': 0.0,
                'lighting_pattern_similarity': 0.0,
                'shot_type_distribution_similarity': 0.0,
                'cut_rate_similarity': 0.0,
                'motion_pattern_similarity': 0.0
            })
        
        # D. Text & OCR
        if video_text_features is not None and reference_text_features_list is not None:
            text_metrics = self.compute_text_similarity(video_text_features, reference_text_features_list)
            features.update(text_metrics)
        else:
            features.update({
                'ocr_text_semantic_similarity': 0.0,
                'text_layout_similarity': 0.0,
                'text_timing_similarity': 0.0
            })
        
        # E. Audio / Speech
        if video_audio_features is not None and reference_audio_features_list is not None:
            audio_metrics = self.compute_audio_similarity(video_audio_features, reference_audio_features_list)
            features.update(audio_metrics)
        else:
            features.update({
                'audio_embedding_similarity': 0.0,
                'speech_content_similarity': 0.0,
                'music_tempo_similarity': 0.0,
                'audio_energy_pattern_similarity': 0.0
            })
        
        # F. Emotion & Behavior
        if video_emotion_features is not None and reference_emotion_features_list is not None:
            emotion_metrics = self.compute_emotion_behavior_similarity(
                video_emotion_features, reference_emotion_features_list
            )
            features.update(emotion_metrics)
        else:
            features.update({
                'emotion_curve_similarity': 0.0,
                'pose_motion_similarity': 0.0,
                'behavior_pattern_similarity': 0.0
            })
        
        # G. Temporal / Pacing
        if video_pacing_features is not None and reference_pacing_features_list is not None:
            temporal_metrics = self.compute_temporal_similarity(video_pacing_features, reference_pacing_features_list)
            features.update(temporal_metrics)
        else:
            features.update({
                'pacing_curve_similarity': 0.0,
                'shot_duration_distribution_similarity': 0.0,
                'scene_length_similarity': 0.0,
                'temporal_pattern_novelty': 1.0
            })
        
        # H. High-level scores
        high_level = self.compute_high_level_scores(features, reference_videos_metadata)
        features.update(high_level)
        
        return {
            'features': features,
            'all_metrics': features  # Для обратной совместимости
        }


# ==================== Example Usage ====================

if __name__ == "__main__":
    # Пример использования
    import numpy as np
    
    # Создаем экземпляр
    similarity = SimilarityMetrics(top_n=10)
    
    # Примерные данные текущего видео
    video_embedding = np.random.randn(512)
    video_topics = ["cooking", "tutorial", "food"]
    
    # Референсные видео
    reference_embeddings = [np.random.randn(512) for _ in range(5)]
    reference_topics_list = [
        ["cooking", "recipe"],
        ["gaming", "tutorial"],
        ["cooking", "food", "tutorial"],
        ["travel", "vlog"],
        ["cooking", "diy"]
    ]
    
    # Вычисляем метрики
    result = similarity.extract_all(
        video_embedding=video_embedding,
        reference_embeddings=reference_embeddings,
        video_topics=video_topics,
        reference_topics_list=reference_topics_list
    )
    
    print("Similarity metrics:")
    for key, value in result['features'].items():
        print(f"  {key}: {value:.4f}")
