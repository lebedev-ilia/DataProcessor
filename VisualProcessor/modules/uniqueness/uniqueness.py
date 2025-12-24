"""
# ⭐ 9. Степень уникальности видео      🟥🟥🟥

Модуль для вычисления степени уникальности видео по различным аспектам.
Сравнивает текущее видео с референсными/топовыми видео и вычисляет novelty scores.

🔹 A. Semantic / Content Novelty
🔹 B. Visual / Style Novelty
🔹 C. Editing & Pacing Novelty
🔹 D. Audio Novelty
🔹 E. Text / OCR Novelty
🔹 F. Behavioral & Motion Novelty
🔹 G. Multimodal Novelty
🔹 H. Temporal / Trend Novelty
"""

from typing import Dict, List, Optional, Any, Union, Tuple
import numpy as np
from scipy.spatial.distance import cosine
from scipy.stats import pearsonr, entropy, wasserstein_distance
from sklearn.metrics.pairwise import cosine_similarity
import warnings

warnings.filterwarnings("ignore")


class UniquenessModule:
    """
    Модуль для вычисления степени уникальности видео.
    
    Вычисляет novelty scores по различным аспектам:
    - Семантическая новизна (content novelty)
    - Визуальная новизна (visual/style novelty)
    - Новизна монтажа и ритма (editing/pacing novelty)
    - Аудио новизна
    - Текстовая новизна (OCR/text novelty)
    - Поведенческая новизна (behavioral/motion novelty)
    - Мультимодальная новизна
    - Временная/трендовая новизна
    """
    
    def __init__(
        self,
        top_n: int = 100,
        novelty_weights: Optional[Dict[str, float]] = None
    ):
        """
        Args:
            top_n: Количество топ видео для сравнения (по умолчанию 100)
            novelty_weights: Веса для различных категорий при вычислении overall_novelty_index
        """
        self.top_n = top_n
        
        # Веса по умолчанию для overall_novelty_index
        self.novelty_weights = novelty_weights or {
            'semantic': 0.20,
            'visual': 0.15,
            'editing': 0.15,
            'audio': 0.10,
            'text': 0.10,
            'behavioral': 0.10,
            'multimodal': 0.15,
            'temporal': 0.05
        }
    
    # ==================== A. Semantic / Content Novelty ====================
    
    def compute_semantic_novelty(
        self,
        video_embedding: np.ndarray,
        reference_embeddings: List[np.ndarray],
        video_topics: Optional[Union[List[str], np.ndarray, Dict[str, float]]] = None,
        reference_topics_list: Optional[List[Union[List[str], np.ndarray, Dict[str, float]]]] = None
    ) -> Dict[str, float]:
        """
        Вычисляет семантическую новизну контента.
        
        Args:
            video_embedding: Embedding текущего видео (1D array)
            reference_embeddings: Список embeddings референсных видео
            video_topics: Темы текущего видео
            reference_topics_list: Список тем референсных видео
            
        Returns:
            Словарь с метриками семантической новизны
        """
        if len(reference_embeddings) == 0:
            return {
                "semantic_novelty_score": 1.0,
                "semantic_novelty_max": 1.0,
                "semantic_novelty_topk_mean": 1.0,
                "semantic_novelty_topk_median": 1.0,
                "topic_novelty_score": 1.0,
                "concept_diversity_score": 0.0,
                "concept_diversity_entropy": 0.0,
                "concept_diversity_unique_norm": 0.0,
            }

        # Нормализуем embeddings
        video_emb_norm = video_embedding / (np.linalg.norm(video_embedding) + 1e-10)

        similarities: List[float] = []
        for ref_emb in reference_embeddings:
            ref_emb_norm = ref_emb / (np.linalg.norm(ref_emb) + 1e-10)
            sim = float(np.dot(video_emb_norm, ref_emb_norm))
            similarities.append(sim)

        similarities_arr = np.asarray(similarities, dtype=np.float32)
        if similarities_arr.size == 0:
            max_similarity = 0.0
            topk_mean_sim = 0.0
            topk_median_sim = 0.0
        else:
            similarities_sorted = np.sort(similarities_arr)[::-1]
            max_similarity = float(similarities_sorted[0])
            k = min(5, similarities_sorted.size)
            topk = similarities_sorted[:k]
            topk_mean_sim = float(topk.mean())
            topk_median_sim = float(np.median(topk))

        # Semantic novelty family: 1 - similarity
        semantic_novelty_max = 1.0 - max_similarity
        semantic_novelty_topk_mean = 1.0 - topk_mean_sim
        semantic_novelty_topk_median = 1.0 - topk_median_sim

        # Topic novelty: доля новых концептов
        topic_novelty = 1.0
        if video_topics is not None and reference_topics_list is not None:
            topic_novelty = self._compute_topic_novelty(video_topics, reference_topics_list)

        # Concept diversity: энтропия + нормализованный unique count
        diversity_entropy = 0.0
        diversity_unique_norm = 0.0
        if video_topics is not None:
            if isinstance(video_topics, dict):
                probs = np.array(list(video_topics.values()), dtype=np.float32)
                if probs.size > 0:
                    probs = probs / (probs.sum() + 1e-10)
                    diversity_entropy = float(
                        entropy(probs) / (np.log(len(probs) + 1e-10))
                    )
                    diversity_unique_norm = float(
                        len([p for p in probs if p > 0.01]) / np.log(len(probs) + 1.0)
                    )
            elif isinstance(video_topics, (list, np.ndarray)) and len(video_topics) > 0:
                if isinstance(video_topics[0], str):
                    unique_count = len(set(video_topics))
                    total_count = len(video_topics)
                    diversity_unique_norm = float(
                        unique_count / np.log(total_count + 1.0)
                    ) if total_count > 0 else 0.0
                else:
                    arr = np.asarray(video_topics, dtype=np.float32)
                    probs = arr / (arr.sum() + 1e-10)
                    diversity_entropy = float(
                        entropy(probs) / (np.log(len(probs) + 1e-10))
                    )

        # Для обратной совместимости concept_diversity_score = entropy-версия
        concept_diversity_score = diversity_entropy

        return {
            "semantic_novelty_score": float(semantic_novelty_max),
            "semantic_novelty_max": float(semantic_novelty_max),
            "semantic_novelty_topk_mean": float(semantic_novelty_topk_mean),
            "semantic_novelty_topk_median": float(semantic_novelty_topk_median),
            "topic_novelty_score": float(topic_novelty),
            "concept_diversity_score": float(concept_diversity_score),
            "concept_diversity_entropy": float(diversity_entropy),
            "concept_diversity_unique_norm": float(diversity_unique_norm),
        }
    
    def _compute_topic_novelty(
        self,
        video_topics: Union[List[str], np.ndarray, Dict[str, float]],
        reference_topics_list: List[Union[List[str], np.ndarray, Dict[str, float]]]
    ) -> float:
        """Вычисляет долю новых концептов, которых нет в популярных видео."""
        def to_set(topics):
            if isinstance(topics, dict):
                sorted_topics = sorted(topics.items(), key=lambda x: x[1], reverse=True)
                return set([t[0] for t in sorted_topics[:10]])
            elif isinstance(topics, (list, np.ndarray)):
                if len(topics) > 0 and isinstance(topics[0], str):
                    return set(topics[:20])
                else:
                    return set(np.where(np.array(topics) > 0.1)[0].astype(str))
            return set()
        
        video_topics_set = to_set(video_topics)
        if len(video_topics_set) == 0:
            return 0.0
        
        # Собираем все темы из референсных видео
        all_reference_topics = set()
        for ref_topics in reference_topics_list:
            all_reference_topics.update(to_set(ref_topics))
        
        # Новые концепты = те, которых нет в референсных
        new_concepts = video_topics_set - all_reference_topics
        novelty_ratio = len(new_concepts) / len(video_topics_set) if len(video_topics_set) > 0 else 0.0
        
        return float(novelty_ratio)
    
    def _compute_concept_diversity(
        self,
        video_topics: Union[List[str], np.ndarray, Dict[str, float]]
    ) -> float:
        """Вычисляет разнообразие концептов."""
        if isinstance(video_topics, dict):
            probs = np.array(list(video_topics.values()))
            probs = probs / (probs.sum() + 1e-10)
            diversity = entropy(probs) / np.log(len(probs) + 1e-10) if len(probs) > 0 else 0.0
            return float(diversity)
        elif isinstance(video_topics, (list, np.ndarray)):
            unique_count = len(set(video_topics)) if isinstance(video_topics[0], str) else len(video_topics)
            total_count = len(video_topics)
            return float(unique_count / total_count) if total_count > 0 else 0.0
        return 0.0
    
    # ==================== B. Visual / Style Novelty ====================
    
    def compute_visual_novelty(
        self,
        video_visual_features: Dict[str, Any],
        reference_visual_features_list: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Вычисляет визуальную новизну стиля.
        
        Args:
            video_visual_features: Визуальные фичи текущего видео (цвет, свет, композиция, типы кадров)
            reference_visual_features_list: Список визуальных фичей референсных видео
            
        Returns:
            Словарь с метриками визуальной новизны
        """
        if len(reference_visual_features_list) == 0:
            return {
                'color_palette_novelty': 1.0,
                'lighting_style_novelty': 1.0,
                'shot_type_novelty': 1.0,
                'camera_motion_novelty': 1.0
            }
        
        color_novelties = []
        lighting_novelties = []
        shot_type_novelties = []
        camera_motion_novelties = []
        
        for ref_features in reference_visual_features_list:
            # Color palette novelty
            if 'color_histogram' in video_visual_features and 'color_histogram' in ref_features:
                hist1 = np.array(video_visual_features['color_histogram']).flatten()
                hist2 = np.array(ref_features['color_histogram']).flatten()
                if len(hist1) == len(hist2):
                    hist1_norm = hist1 / (np.linalg.norm(hist1) + 1e-10)
                    hist2_norm = hist2 / (np.linalg.norm(hist2) + 1e-10)
                    sim = np.dot(hist1_norm, hist2_norm)
                    color_novelties.append(1.0 - sim)
            
            # Lighting style novelty
            if 'lighting_features' in video_visual_features and 'lighting_features' in ref_features:
                light1 = np.array(video_visual_features['lighting_features'])
                light2 = np.array(ref_features['lighting_features'])
                if light1.shape == light2.shape:
                    sim = 1.0 - cosine(light1.flatten(), light2.flatten())
                    lighting_novelties.append(max(0.0, 1.0 - sim))
            
            # Shot type novelty
            if 'shot_type_distribution' in video_visual_features and 'shot_type_distribution' in ref_features:
                dist1 = np.array(video_visual_features['shot_type_distribution'])
                dist2 = np.array(ref_features['shot_type_distribution'])
                if len(dist1) == len(dist2):
                    dist1_norm = dist1 / (dist1.sum() + 1e-10)
                    dist2_norm = dist2 / (dist2.sum() + 1e-10)
                    wd = wasserstein_distance(dist1_norm, dist2_norm)
                    max_wd = np.max(dist1_norm) + np.max(dist2_norm)
                    sim = 1.0 - wd / (max_wd + 1e-10)
                    shot_type_novelties.append(max(0.0, min(1.0, 1.0 - sim)))
            
            # Camera motion novelty
            if 'camera_motion_features' in video_visual_features and 'camera_motion_features' in ref_features:
                motion1 = np.array(video_visual_features['camera_motion_features'])
                motion2 = np.array(ref_features['camera_motion_features'])
                if len(motion1) == len(motion2):
                    try:
                        corr, _ = pearsonr(motion1, motion2)
                        motion_novelty = 1.0 - max(0.0, corr) if not np.isnan(corr) else 1.0
                        camera_motion_novelties.append(motion_novelty)
                    except:
                        camera_motion_novelties.append(1.0)
        
        return {
            'color_palette_novelty': float(np.mean(color_novelties)) if color_novelties else 1.0,
            'lighting_style_novelty': float(np.mean(lighting_novelties)) if lighting_novelties else 1.0,
            'shot_type_novelty': float(np.mean(shot_type_novelties)) if shot_type_novelties else 1.0,
            'camera_motion_novelty': float(np.mean(camera_motion_novelties)) if camera_motion_novelties else 1.0
        }
    
    # ==================== C. Editing & Pacing Novelty ====================
    
    def compute_editing_novelty(
        self,
        video_pacing_features: Dict[str, Any],
        reference_pacing_features_list: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Вычисляет новизну монтажа и ритма.
        
        Args:
            video_pacing_features: Фичи pacing текущего видео (cut rate, shot duration, scene length, pacing curve)
            reference_pacing_features_list: Список фичей pacing референсных видео
            
        Returns:
            Словарь с метриками новизны монтажа
        """
        if len(reference_pacing_features_list) == 0:
            return {
                'cut_rate_novelty': 1.0,
                'shot_duration_distribution_novelty': 1.0,
                'scene_length_novelty': 1.0,
                'pacing_pattern_novelty': 1.0
            }
        
        cut_rate_novelties = []
        shot_duration_novelties = []
        scene_length_novelties = []
        pacing_pattern_novelties = []
        
        for ref_features in reference_pacing_features_list:
            # Cut rate novelty
            if 'cut_rate' in video_pacing_features and 'cut_rate' in ref_features:
                cut1 = video_pacing_features['cut_rate']
                cut2 = ref_features['cut_rate']
                max_cut = max(abs(cut1), abs(cut2), 1.0)
                diff = abs(cut1 - cut2) / max_cut
                cut_rate_novelties.append(min(1.0, diff))
            
            # Shot duration distribution novelty
            if 'shot_duration_distribution' in video_pacing_features and 'shot_duration_distribution' in ref_features:
                dist1 = np.array(video_pacing_features['shot_duration_distribution'])
                dist2 = np.array(ref_features['shot_duration_distribution'])
                if len(dist1) == len(dist2):
                    dist1_norm = dist1 / (dist1.sum() + 1e-10)
                    dist2_norm = dist2 / (dist2.sum() + 1e-10)
                    wd = wasserstein_distance(dist1_norm, dist2_norm)
                    max_wd = np.max(dist1_norm) + np.max(dist2_norm)
                    sim = 1.0 - wd / (max_wd + 1e-10)
                    shot_duration_novelties.append(max(0.0, min(1.0, 1.0 - sim)))
            
            # Scene length novelty
            if 'scene_lengths' in video_pacing_features and 'scene_lengths' in ref_features:
                len1 = np.array(video_pacing_features['scene_lengths'])
                len2 = np.array(ref_features['scene_lengths'])
                if len(len1) > 0 and len(len2) > 0:
                    mean1, std1 = np.mean(len1), np.std(len1)
                    mean2, std2 = np.mean(len2), np.std(len2)
                    max_mean = max(abs(mean1), abs(mean2), 1.0)
                    max_std = max(abs(std1), abs(std2), 1.0)
                    mean_diff = abs(mean1 - mean2) / max_mean
                    std_diff = abs(std1 - std2) / max_std
                    scene_novelty = (mean_diff + std_diff) / 2.0
                    scene_length_novelties.append(min(1.0, scene_novelty))
            
            # Pacing pattern novelty
            if 'pacing_curve' in video_pacing_features and 'pacing_curve' in ref_features:
                curve1 = np.array(video_pacing_features['pacing_curve'])
                curve2 = np.array(ref_features['pacing_curve'])
                if len(curve1) == len(curve2):
                    try:
                        corr, _ = pearsonr(curve1, curve2)
                        pacing_novelty = 1.0 - max(0.0, corr) if not np.isnan(corr) else 1.0
                        pacing_pattern_novelties.append(pacing_novelty)
                    except:
                        pacing_pattern_novelties.append(1.0)
        
        return {
            'cut_rate_novelty': float(np.mean(cut_rate_novelties)) if cut_rate_novelties else 1.0,
            'shot_duration_distribution_novelty': float(np.mean(shot_duration_novelties)) if shot_duration_novelties else 1.0,
            'scene_length_novelty': float(np.mean(scene_length_novelties)) if scene_length_novelties else 1.0,
            'pacing_pattern_novelty': float(np.mean(pacing_pattern_novelties)) if pacing_pattern_novelties else 1.0
        }
    
    # ==================== D. Audio Novelty ====================
    
    def compute_audio_novelty(
        self,
        video_audio_features: Dict[str, Any],
        reference_audio_features_list: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Вычисляет аудио новизну.
        
        Args:
            video_audio_features: Аудио фичи текущего видео (embeddings, tempo, energy, voice style)
            reference_audio_features_list: Список аудио фичей референсных видео
            
        Returns:
            Словарь с метриками аудио новизны
        """
        if len(reference_audio_features_list) == 0:
            return {
                'music_track_novelty': 1.0,
                'voice_style_novelty': 1.0,
                'sound_effects_novelty': 1.0,
                'audio_energy_pattern_novelty': 1.0
        }
        
        music_novelties = []
        voice_novelties = []
        sound_effects_novelties = []
        energy_pattern_novelties = []
        
        for ref_features in reference_audio_features_list:
            # Music track novelty (BPM, style)
            if 'audio_embedding' in video_audio_features and 'audio_embedding' in ref_features:
                emb1 = np.array(video_audio_features['audio_embedding'])
                emb2 = np.array(ref_features['audio_embedding'])
                if emb1.shape == emb2.shape:
                    emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-10)
                    emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-10)
                    sim = np.dot(emb1_norm, emb2_norm)
                    music_novelties.append(1.0 - sim)
            
            # Voice style novelty
            if 'voice_embedding' in video_audio_features and 'voice_embedding' in ref_features:
                voice1 = np.array(video_audio_features['voice_embedding'])
                voice2 = np.array(ref_features['voice_embedding'])
                if voice1.shape == voice2.shape:
                    voice1_norm = voice1 / (np.linalg.norm(voice1) + 1e-10)
                    voice2_norm = voice2 / (np.linalg.norm(voice2) + 1e-10)
                    sim = np.dot(voice1_norm, voice2_norm)
                    voice_novelties.append(1.0 - sim)
            elif 'tempo' in video_audio_features and 'tempo' in ref_features:
                # Fallback: use tempo difference
                tempo1 = video_audio_features['tempo']
                tempo2 = ref_features['tempo']
                max_tempo = max(abs(tempo1), abs(tempo2), 1.0)
                tempo_diff = abs(tempo1 - tempo2) / max_tempo
                voice_novelties.append(min(1.0, tempo_diff))
            
            # Sound effects novelty
            if 'sound_effects_features' in video_audio_features and 'sound_effects_features' in ref_features:
                sfx1 = np.array(video_audio_features['sound_effects_features'])
                sfx2 = np.array(ref_features['sound_effects_features'])
                if sfx1.shape == sfx2.shape:
                    sfx1_norm = sfx1 / (np.linalg.norm(sfx1) + 1e-10)
                    sfx2_norm = sfx2 / (np.linalg.norm(sfx2) + 1e-10)
                    sim = np.dot(sfx1_norm, sfx2_norm)
                    sound_effects_novelties.append(1.0 - sim)
            
            # Audio energy pattern novelty
            if 'energy_pattern' in video_audio_features and 'energy_pattern' in ref_features:
                energy1 = np.array(video_audio_features['energy_pattern'])
                energy2 = np.array(ref_features['energy_pattern'])
                if len(energy1) == len(energy2):
                    try:
                        corr, _ = pearsonr(energy1, energy2)
                        energy_novelty = 1.0 - max(0.0, corr) if not np.isnan(corr) else 1.0
                        energy_pattern_novelties.append(energy_novelty)
                    except:
                        energy_pattern_novelties.append(1.0)
        
        return {
            'music_track_novelty': float(np.mean(music_novelties)) if music_novelties else 1.0,
            'voice_style_novelty': float(np.mean(voice_novelties)) if voice_novelties else 1.0,
            'sound_effects_novelty': float(np.mean(sound_effects_novelties)) if sound_effects_novelties else 1.0,
            'audio_energy_pattern_novelty': float(np.mean(energy_pattern_novelties)) if energy_pattern_novelties else 1.0
        }
    
    # ==================== E. Text / OCR Novelty ====================
    
    def compute_text_novelty(
        self,
        video_text_features: Dict[str, Any],
        reference_text_features_list: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Вычисляет текстовую новизну (OCR, заголовки, мемы).
        
        Args:
            video_text_features: Текстовые фичи текущего видео (OCR embeddings, layout, style)
            reference_text_features_list: Список текстовых фичей референсных видео
            
        Returns:
            Словарь с метриками текстовой новизны
        """
        if len(reference_text_features_list) == 0:
            return {
                'ocr_text_novelty': 1.0,
                'text_layout_novelty': 1.0,
                'text_style_novelty': 1.0
            }
        
        ocr_novelties = []
        layout_novelties = []
        style_novelties = []
        
        for ref_features in reference_text_features_list:
            # OCR text novelty (новые ключевые слова/фразы)
            if 'ocr_embedding' in video_text_features and 'ocr_embedding' in ref_features:
                emb1 = np.array(video_text_features['ocr_embedding'])
                emb2 = np.array(ref_features['ocr_embedding'])
                if emb1.shape == emb2.shape:
                    emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-10)
                    emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-10)
                    sim = np.dot(emb1_norm, emb2_norm)
                    ocr_novelties.append(1.0 - sim)
            
            # Text layout novelty (необычное расположение текста)
            if 'text_layout' in video_text_features and 'text_layout' in ref_features:
                layout1 = np.array(video_text_features['text_layout'])
                layout2 = np.array(ref_features['text_layout'])
                if len(layout1) == len(layout2):
                    layout_sim = 1.0 - cosine(layout1.flatten(), layout2.flatten())
                    layout_novelties.append(max(0.0, 1.0 - layout_sim))
            
            # Text style novelty (уникальные шрифты, цвета, эффекты)
            if 'text_style_features' in video_text_features and 'text_style_features' in ref_features:
                style1 = np.array(video_text_features['text_style_features'])
                style2 = np.array(ref_features['text_style_features'])
                if style1.shape == style2.shape:
                    style1_norm = style1 / (np.linalg.norm(style1) + 1e-10)
                    style2_norm = style2 / (np.linalg.norm(style2) + 1e-10)
                    sim = np.dot(style1_norm, style2_norm)
                    style_novelties.append(1.0 - sim)
        
        return {
            'ocr_text_novelty': float(np.mean(ocr_novelties)) if ocr_novelties else 1.0,
            'text_layout_novelty': float(np.mean(layout_novelties)) if layout_novelties else 1.0,
            'text_style_novelty': float(np.mean(style_novelties)) if style_novelties else 1.0
        }
    
    # ==================== F. Behavioral & Motion Novelty ====================
    
    def compute_behavioral_novelty(
        self,
        video_behavior_features: Dict[str, Any],
        reference_behavior_features_list: List[Dict[str, Any]]
    ) -> Dict[str, float]:
        """
        Вычисляет поведенческую новизну (движения людей, объектов).
        
        Args:
            video_behavior_features: Фичи поведения текущего видео (pose motion, object interaction, actions)
            reference_behavior_features_list: Список фичей поведения референсных видео
            
        Returns:
            Словарь с метриками поведенческой новизны
        """
        if len(reference_behavior_features_list) == 0:
            return {
                'pose_motion_novelty': 1.0,
                'object_interaction_novelty': 1.0,
                'action_sequence_novelty': 1.0
            }
        
        pose_novelties = []
        interaction_novelties = []
        action_novelties = []
        
        for ref_features in reference_behavior_features_list:
            # Pose motion novelty (необычные движения/редкие паттерны)
            if 'pose_motion' in video_behavior_features and 'pose_motion' in ref_features:
                pose1 = np.array(video_behavior_features['pose_motion'])
                pose2 = np.array(ref_features['pose_motion'])
                if pose1.shape == pose2.shape:
                    pose1_norm = pose1 / (np.linalg.norm(pose1) + 1e-10)
                    pose2_norm = pose2 / (np.linalg.norm(pose2) + 1e-10)
                    sim = np.dot(pose1_norm.flatten(), pose2_norm.flatten())
                    pose_novelties.append(1.0 - sim)
            
            # Object interaction novelty
            if 'object_interaction' in video_behavior_features and 'object_interaction' in ref_features:
                int1 = np.array(video_behavior_features['object_interaction'])
                int2 = np.array(ref_features['object_interaction'])
                if int1.shape == int2.shape:
                    int1_norm = int1 / (np.linalg.norm(int1) + 1e-10)
                    int2_norm = int2 / (np.linalg.norm(int2) + 1e-10)
                    sim = np.dot(int1_norm.flatten(), int2_norm.flatten())
                    interaction_novelties.append(1.0 - sim)
            
            # Action sequence novelty
            if 'action_sequence' in video_behavior_features and 'action_sequence' in ref_features:
                act1 = np.array(video_behavior_features['action_sequence'])
                act2 = np.array(ref_features['action_sequence'])
                if len(act1) == len(act2):
                    try:
                        corr, _ = pearsonr(act1, act2)
                        action_novelty = 1.0 - max(0.0, corr) if not np.isnan(corr) else 1.0
                        action_novelties.append(action_novelty)
                    except:
                        action_novelties.append(1.0)
        
        return {
            'pose_motion_novelty': float(np.mean(pose_novelties)) if pose_novelties else 1.0,
            'object_interaction_novelty': float(np.mean(interaction_novelties)) if interaction_novelties else 1.0,
            'action_sequence_novelty': float(np.mean(action_novelties)) if action_novelties else 1.0
        }
    
    # ==================== G. Multimodal Novelty ====================
    
    def compute_multimodal_novelty(
        self,
        all_novelty_metrics: Dict[str, float],
        video_events: Optional[List[Dict[str, Any]]] = None,
        reference_events_list: Optional[List[List[Dict[str, Any]]]] = None
    ) -> Dict[str, float]:
        """
        Вычисляет мультимодальную новизну.
        
        Args:
            all_novelty_metrics: Все метрики новизны из категорий A-F
            video_events: События текущего видео (опционально)
            reference_events_list: Список событий референсных видео (опционально)
            
        Returns:
            Словарь с метриками мультимодальной новизны
        """
        # Multimodal novelty score = средневзвешенная новизна по всем модальностям
        weights = self.novelty_weights
        
        semantic_novelty = all_novelty_metrics.get('semantic_novelty_score', 0.0)
        visual_novelty = np.mean([
            all_novelty_metrics.get('color_palette_novelty', 0.0),
            all_novelty_metrics.get('lighting_style_novelty', 0.0),
            all_novelty_metrics.get('shot_type_novelty', 0.0)
        ])
        editing_novelty = np.mean([
            all_novelty_metrics.get('cut_rate_novelty', 0.0),
            all_novelty_metrics.get('pacing_pattern_novelty', 0.0)
        ])
        audio_novelty = all_novelty_metrics.get('music_track_novelty', 0.0)
        text_novelty = all_novelty_metrics.get('ocr_text_novelty', 0.0)
        behavioral_novelty = all_novelty_metrics.get('pose_motion_novelty', 0.0)
        
        multimodal_novelty = (
            weights['semantic'] * semantic_novelty +
            weights['visual'] * visual_novelty +
            weights['editing'] * editing_novelty +
            weights['audio'] * audio_novelty +
            weights['text'] * text_novelty +
            weights['behavioral'] * behavioral_novelty
        )
        
        # Novel event alignment score (новые события, которых нет в топ видео)
        novel_event_score = 1.0
        if video_events is not None and reference_events_list is not None:
            novel_event_score = self._compute_novel_event_score(video_events, reference_events_list)
        
        # Overall novelty index
        overall_novelty = (
            weights['semantic'] * semantic_novelty +
            weights['visual'] * visual_novelty +
            weights['editing'] * editing_novelty +
            weights['audio'] * audio_novelty +
            weights['text'] * text_novelty +
            weights['behavioral'] * behavioral_novelty +
            weights['multimodal'] * multimodal_novelty +
            weights['temporal'] * all_novelty_metrics.get('trend_alignment_score', 0.0)
        )
        
        return {
            'multimodal_novelty_score': float(multimodal_novelty),
            'novel_event_alignment_score': float(novel_event_score),
            'overall_novelty_index': float(overall_novelty)
        }
    
    def _compute_novel_event_score(
        self,
        video_events: List[Dict[str, Any]],
        reference_events_list: List[List[Dict[str, Any]]]
    ) -> float:
        """Вычисляет долю новых событий, которых нет в референсных видео."""
        if len(video_events) == 0:
            return 0.0
        
        # Собираем все типы событий из референсных видео
        all_reference_event_types = set()
        for ref_events in reference_events_list:
            for event in ref_events:
                event_type = event.get('type', event.get('event_type', ''))
                if event_type:
                    all_reference_event_types.add(event_type)
        
        # Новые события = те, которых нет в референсных
        video_event_types = set()
        for event in video_events:
            event_type = event.get('type', event.get('event_type', ''))
            if event_type:
                video_event_types.add(event_type)
        
        new_event_types = video_event_types - all_reference_event_types
        novelty_ratio = len(new_event_types) / len(video_event_types) if len(video_event_types) > 0 else 0.0
        
        return float(novelty_ratio)
    
    # ==================== H. Temporal / Trend Novelty ====================
    
    def compute_temporal_novelty(
        self,
        video_metadata: Optional[Dict[str, Any]] = None,
        reference_videos_metadata: Optional[List[Dict[str, Any]]] = None,
        similarity_scores: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """
        Вычисляет временную/трендовую новизну.
        
        Args:
            video_metadata: Метаданные текущего видео (дата создания, категория и т.д.)
            reference_videos_metadata: Метаданные референсных видео
            similarity_scores: Предвычисленные similarity scores (опционально)
            
        Returns:
            Словарь с метриками временной новизны
        """
        # Trend alignment score = насколько видео соответствует текущему тренду (низкий → уникально)
        trend_alignment = 0.5  # По умолчанию
        if similarity_scores is not None:
            # Если есть similarity scores, trend alignment = средняя схожесть с топ видео
            trend_alignment = similarity_scores.get('overall_similarity_score', 0.5)
        elif reference_videos_metadata is not None and len(reference_videos_metadata) > 0:
            # Упрощенная версия: если видео похоже на популярные → высокий trend alignment
            trend_alignment = 0.5  # Можно улучшить, используя метаданные о популярности
        
        # Historical similarity score = сравнение с прошлым контентом
        historical_similarity = 0.5  # По умолчанию
        if video_metadata is not None and reference_videos_metadata is not None:
            # Можно использовать дату создания для сравнения с историческим контентом
            # Упрощенная версия
            historical_similarity = 0.5
        
        # Early adopter score = насколько видео новаторское в своей нише
        early_adopter = 1.0 - trend_alignment  # Упрощенная версия
        
        return {
            'trend_alignment_score': float(trend_alignment),
            'historical_similarity_score': float(historical_similarity),
            'early_adopter_score': float(early_adopter)
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
        video_pacing_features: Optional[Dict[str, Any]] = None,
        reference_pacing_features_list: Optional[List[Dict[str, Any]]] = None,
        video_audio_features: Optional[Dict[str, Any]] = None,
        reference_audio_features_list: Optional[List[Dict[str, Any]]] = None,
        video_text_features: Optional[Dict[str, Any]] = None,
        reference_text_features_list: Optional[List[Dict[str, Any]]] = None,
        video_behavior_features: Optional[Dict[str, Any]] = None,
        reference_behavior_features_list: Optional[List[Dict[str, Any]]] = None,
        video_events: Optional[List[Dict[str, Any]]] = None,
        reference_events_list: Optional[List[List[Dict[str, Any]]]] = None,
        video_metadata: Optional[Dict[str, Any]] = None,
        reference_videos_metadata: Optional[List[Dict[str, Any]]] = None,
        similarity_scores: Optional[Dict[str, float]] = None
    ) -> Dict[str, Any]:
        """
        Главный метод для вычисления всех метрик уникальности.
        
        Args:
            video_embedding: Embedding текущего видео
            reference_embeddings: Список embeddings референсных видео
            video_topics: Темы текущего видео
            reference_topics_list: Список тем референсных видео
            video_visual_features: Визуальные фичи текущего видео
            reference_visual_features_list: Список визуальных фичей референсных видео
            video_pacing_features: Фичи pacing текущего видео
            reference_pacing_features_list: Список фичей pacing референсных видео
            video_audio_features: Аудио фичи текущего видео
            reference_audio_features_list: Список аудио фичей референсных видео
            video_text_features: Текстовые фичи текущего видео
            reference_text_features_list: Список текстовых фичей референсных видео
            video_behavior_features: Фичи поведения текущего видео
            reference_behavior_features_list: Список фичей поведения референсных видео
            video_events: События текущего видео
            reference_events_list: Список событий референсных видео
            video_metadata: Метаданные текущего видео
            reference_videos_metadata: Метаданные референсных видео
            similarity_scores: Предвычисленные similarity scores (опционально)
            
        Returns:
            Словарь со всеми метриками уникальности
        """
        features = {}
        
        # A. Semantic / Content Novelty
        semantic_novelty = self.compute_semantic_novelty(
            video_embedding, reference_embeddings,
            video_topics, reference_topics_list
        )
        features.update(semantic_novelty)
        
        # B. Visual / Style Novelty
        if video_visual_features is not None and reference_visual_features_list is not None:
            visual_novelty = self.compute_visual_novelty(
                video_visual_features, reference_visual_features_list
            )
            features.update(visual_novelty)
        else:
            features.update({
                'color_palette_novelty': 0.0,
                'lighting_style_novelty': 0.0,
                'shot_type_novelty': 0.0,
                'camera_motion_novelty': 0.0
            })
        
        # C. Editing & Pacing Novelty
        if video_pacing_features is not None and reference_pacing_features_list is not None:
            editing_novelty = self.compute_editing_novelty(
                video_pacing_features, reference_pacing_features_list
            )
            features.update(editing_novelty)
        else:
            features.update({
                'cut_rate_novelty': 0.0,
                'shot_duration_distribution_novelty': 0.0,
                'scene_length_novelty': 0.0,
                'pacing_pattern_novelty': 0.0
            })
        
        # D. Audio Novelty
        if video_audio_features is not None and reference_audio_features_list is not None:
            audio_novelty = self.compute_audio_novelty(
                video_audio_features, reference_audio_features_list
            )
            features.update(audio_novelty)
        else:
            features.update({
                'music_track_novelty': 0.0,
                'voice_style_novelty': 0.0,
                'sound_effects_novelty': 0.0,
                'audio_energy_pattern_novelty': 0.0
            })
        
        # E. Text / OCR Novelty
        if video_text_features is not None and reference_text_features_list is not None:
            text_novelty = self.compute_text_novelty(
                video_text_features, reference_text_features_list
            )
            features.update(text_novelty)
        else:
            features.update({
                'ocr_text_novelty': 0.0,
                'text_layout_novelty': 0.0,
                'text_style_novelty': 0.0
            })
        
        # F. Behavioral & Motion Novelty
        if video_behavior_features is not None and reference_behavior_features_list is not None:
            behavioral_novelty = self.compute_behavioral_novelty(
                video_behavior_features, reference_behavior_features_list
            )
            features.update(behavioral_novelty)
        else:
            features.update({
                'pose_motion_novelty': 0.0,
                'object_interaction_novelty': 0.0,
                'action_sequence_novelty': 0.0
            })
        
        # G. Multimodal Novelty
        multimodal_novelty = self.compute_multimodal_novelty(
            features, video_events, reference_events_list
        )
        features.update(multimodal_novelty)
        
        # H. Temporal / Trend Novelty
        temporal_novelty = self.compute_temporal_novelty(
            video_metadata, reference_videos_metadata, similarity_scores
        )
        features.update(temporal_novelty)
        
        return {
            'features': features,
            'all_metrics': features  # Для обратной совместимости
        }


# ==================== Example Usage ====================

if __name__ == "__main__":
    # Пример использования
    import numpy as np
    
    # Создаем экземпляр
    uniqueness = UniquenessModule(top_n=100)
    
    # Примерные данные текущего видео
    video_embedding = np.random.randn(512)
    video_topics = ["cooking", "tutorial", "food", "unique_recipe"]
    
    # Референсные видео
    reference_embeddings = [np.random.randn(512) for _ in range(10)]
    reference_topics_list = [
        ["cooking", "recipe"],
        ["gaming", "tutorial"],
        ["cooking", "food", "tutorial"],
        ["travel", "vlog"],
        ["cooking", "diy"]
    ]
    
    # Вычисляем метрики уникальности
    result = uniqueness.extract_all(
        video_embedding=video_embedding,
        reference_embeddings=reference_embeddings,
        video_topics=video_topics,
        reference_topics_list=reference_topics_list
    )
    
    print("Uniqueness metrics:")
    for key, value in result['features'].items():
        print(f"  {key}: {value:.4f}")
