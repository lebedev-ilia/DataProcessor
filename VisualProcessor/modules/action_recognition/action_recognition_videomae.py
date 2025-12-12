# action_recognition_videomae_motion.py

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from collections import defaultdict
import torch
import cv2
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
try:
    from scipy import stats
except ImportError:
    stats = None


def entropy_of_prob(p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))


def longest_run_fraction(labels: List[int]) -> float:
    if len(labels) == 0:
        return 0.0
    max_run = cur = 1
    for a, b in zip(labels, labels[1:]):
        cur = cur + 1 if a == b else 1
        max_run = max(max_run, cur)
    return max_run / max(1, len(labels))


class VideoMAEActionRecognizer:
    def __init__(
        self,
        frame_manager,
        model_name: str,
        clip_len: int = 16,
        stride: Optional[int] = None,
        batch_size: int = 8,
        device: Optional[str] = None,
    ):
        self.fm = frame_manager
        self.clip_len = clip_len
        self.stride = stride or max(1, clip_len // 2)
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        self.feature_extractor = VideoMAEImageProcessor.from_pretrained(model_name)
        self.model = VideoMAEForVideoClassification.from_pretrained(model_name).to(self.device)
        self.model.eval()

        self.id2label = getattr(self.model.config, "id2label", None)

    # ----------------------------
    # Load frames and compute motion magnitude
    # ----------------------------
    def _load_frames_with_motion(self, indices: List[int]):
        frames = []
        motion_mags = []

        prev_gray = None
        for idx in indices:
            im = self.fm.get(idx)
            if im.ndim == 2:
                im = np.stack([im] * 3, axis=-1)
            if im.shape[-1] == 4:
                im = im[..., :3]
            frames.append(im.astype(np.uint8))

            # FrameManager обычно возвращает кадры в RGB формате
            # Если кадр уже grayscale (ndim == 2), используем его напрямую
            if im.ndim == 2:
                gray = im
            elif im.shape[-1] == 3:
                # Предполагаем RGB формат (стандарт для большинства библиотек)
                # Если кадры в BGR, можно изменить на COLOR_BGR2GRAY
                gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
            else:
                # Для других форматов берем первый канал
                gray = im[..., 0]
            if prev_gray is not None:
                flow = cv2.calcOpticalFlowFarneback(prev_gray, gray,
                                                    None,
                                                    0.5, 3, 15, 3, 5, 1.2, 0)
                mag = np.mean(np.sqrt(flow[...,0]**2 + flow[...,1]**2))
                motion_mags.append(mag)
            else:
                motion_mags.append(0.0)
            prev_gray = gray

        return frames, motion_mags

    # ----------------------------
    # Make clips (sliding window)
    # ----------------------------
    def _make_clips(self, frames: List[np.ndarray], motion_mags: List[float]):
        n = len(frames)
        if n < self.clip_len:
            frames = frames + [frames[-1]]*(self.clip_len - n)
            motion_mags = motion_mags + [motion_mags[-1]]*(self.clip_len - len(motion_mags))
            n = len(frames)

        clips = []
        clips_motion = []

        for s in range(0, n - self.clip_len + 1, self.stride):
            clips.append(frames[s:s+self.clip_len])
            clips_motion.append(np.mean(motion_mags[s:s+self.clip_len]))

        if not clips:
            clips.append(frames[-self.clip_len:])
            clips_motion.append(np.mean(motion_mags[-self.clip_len:]))

        return clips, clips_motion

    # ----------------------------
    # Batched inference
    # ----------------------------
    def _infer(self, clips: List[List[np.ndarray]]) -> List[np.ndarray]:
        results = []
        B = self.batch_size

        for i in range(0, len(clips), B):
            batch = clips[i:i+B]

            inputs = self.feature_extractor(batch, return_tensors="pt")
            inputs = {k: v.to(self.device) for k,v in inputs.items()}

            with torch.no_grad():
                logits = self.model(**inputs).logits
                probs = torch.softmax(logits, dim=-1).cpu().numpy()

            results.extend(probs)

        return results

    # ----------------------------
    # Aggregation с учетом motion
    # ----------------------------
    def _aggregate(self, probs_seq: List[np.ndarray], motion_weights: Optional[List[float]]=None, clip_len: Optional[int] = None, fps: Optional[float] = None) -> dict:
        # Используем параметры из self, если не переданы
        clip_len = clip_len or self.clip_len
        fps = fps or getattr(self.fm, 'fps', 30.0)
        
        arr = np.stack(probs_seq, axis=0)  # [num_clips, num_classes]
        if motion_weights is not None:
            motion_weights = np.array(motion_weights)
            motion_weights = motion_weights / (motion_weights.sum() + 1e-6)
            weighted_mean_prob = (arr * motion_weights[:, None]).sum(axis=0)
        else:
            weighted_mean_prob = arr.mean(axis=0)

        # EMA по вероятностям (временное сглаживание)
        alpha = 0.3
        ema_prob = np.zeros_like(weighted_mean_prob)
        for p in arr:
            ema_prob = alpha * p + (1-alpha) * ema_prob

        # labels
        labels = np.argmax(arr, axis=1)
        transitions = int(np.sum(labels[1:] != labels[:-1]))
        dominant = int(weighted_mean_prob.argmax())
        dominant_conf = float(weighted_mean_prob.max())

        # Top-k
        topk = 5
        topk_idx = np.argsort(weighted_mean_prob)[-topk:][::-1]

        # Motion-aware weighted features
        mean_entropy = float(np.array([-np.sum(p*np.log(p+1e-12)) for p in arr]).mean())
        stability = longest_run_fraction(labels)
        total_time_sec = len(labels)*clip_len/fps
        switch_rate = transitions / max(1e-6, total_time_sec)

        # Mode + confidence - исправлено: берем вероятность mode_label из weighted_mean_prob
        unique, counts = np.unique(labels, return_counts=True)
        mode_label = int(unique[np.argmax(counts)])
        mode_conf = float(weighted_mean_prob[mode_label])  # ✅ Исправлено: берем из weighted_mean_prob

        features = {
            "num_clips": len(arr),
            "mean_entropy": mean_entropy,
            "dominant_action": dominant,
            "dominant_confidence": dominant_conf,
            "topk_labels": topk_idx.tolist(),
            "topk_vals": weighted_mean_prob[topk_idx].tolist(),
            "stability": stability,
            "switch_rate_per_sec": switch_rate,
            "mode_action": mode_label,
            "mode_confidence": mode_conf,
            "ema_confidence": float(ema_prob.max()),  # ✅ Добавлено: используем вычисленный ema_prob
        }

        if self.id2label is not None and dominant in self.id2label:
            features["dominant_action_label"] = self.id2label[dominant]

        # Добавляем новые фичи
        # Fine-grained actions
        fine_grained = self._analyze_fine_grained_actions(probs_seq, self.id2label)
        features.update(fine_grained)
        
        # Action complexity
        complexity = self._compute_action_complexity(probs_seq, motion_weights)
        features.update(complexity)
        
        # Action planning & intent
        intent = self._predict_action_intent(probs_seq, clip_len, fps)
        features.update(intent)
        
        # Scene activity
        scene_activity = self._analyze_scene_activity(probs_seq, motion_weights, clip_len, fps)
        features.update(scene_activity)

        return features

    # ----------------------------
    # Fine-grained actions analysis
    # ----------------------------
    def _analyze_fine_grained_actions(self, probs_seq: List[np.ndarray], id2label: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Анализ детальных (fine-grained) действий на основе вероятностей.
        Группирует действия по категориям и определяет тонкие паттерны.
        """
        arr = np.stack(probs_seq, axis=0)  # [num_clips, num_classes]
        
        # Определяем категории fine-grained действий
        fine_grained_categories = {
            'face_touch': ['face', 'touch', 'scratch', 'rub'],
            'hair_touch': ['hair', 'comb', 'brush'],
            'pointing': ['point', 'finger', 'gesture'],
            'waving': ['wave', 'hand'],
            'nodding': ['nod', 'head'],
            'shrugging': ['shrug', 'shoulder'],
            'clapping': ['clap', 'applause'],
            'lip_sync': ['lip', 'mouth', 'speak', 'talk'],
            'scrolling': ['phone', 'scroll', 'swipe', 'device'],
            'reacting': ['react', 'reaction', 'watch', 'screen']
        }
        
        # Анализ вероятностей по категориям
        category_scores = {}
        if id2label:
            for category, keywords in fine_grained_categories.items():
                category_probs = []
                for idx, label in id2label.items():
                    label_lower = label.lower()
                    if any(kw in label_lower for kw in keywords):
                        category_probs.append(arr[:, int(idx)])
                
                if category_probs:
                    category_probs = np.stack(category_probs, axis=1)  # [num_clips, num_matching_classes]
                    category_scores[category] = {
                        'mean_prob': float(np.mean(category_probs)),
                        'max_prob': float(np.max(category_probs)),
                        'temporal_mean': float(np.mean(np.max(category_probs, axis=1))),
                        'presence_ratio': float(np.mean(np.max(category_probs, axis=1) > 0.1))
                    }
        
        # Детекция конкретных fine-grained действий
        fine_grained_detections = []
        for i, probs in enumerate(arr):
            top3_idx = np.argsort(probs)[-3:][::-1]
            for idx in top3_idx:
                if id2label and str(idx) in id2label:
                    label = id2label[str(idx)].lower()
                    prob = float(probs[idx])
                    if prob > 0.15:  # Порог для детекции
                        # Проверяем, является ли это fine-grained действием
                        for category, keywords in fine_grained_categories.items():
                            if any(kw in label for kw in keywords):
                                fine_grained_detections.append({
                                    'clip_idx': i,
                                    'action': id2label[str(idx)],
                                    'probability': prob,
                                    'category': category
                                })
                                break
        
        return {
            'category_scores': category_scores,
            'fine_grained_detections': fine_grained_detections[:20],  # Ограничиваем количество
            'num_fine_grained_actions': len(set(d['category'] for d in fine_grained_detections))
        }

    # ----------------------------
    # Action complexity score
    # ----------------------------
    def _compute_action_complexity(self, probs_seq: List[np.ndarray], motion_weights: Optional[List[float]] = None) -> Dict[str, float]:
        """
        Вычисляет сложность действия на основе:
        - Энтропии распределения вероятностей
        - Разнообразия действий
        - Координации (изменение действий во времени)
        - Точности (уверенность в действиях)
        """
        arr = np.stack(probs_seq, axis=0)  # [num_clips, num_classes]
        
        # 1. Энтропия (выше энтропия = выше сложность)
        entropies = np.array([entropy_of_prob(p) for p in arr])
        mean_entropy = float(np.mean(entropies))
        max_entropy = float(np.max(entropies))
        entropy_complexity = mean_entropy / max(1e-6, np.log(arr.shape[1]))  # Нормализуем
        
        # 2. Разнообразие действий (количество уникальных действий)
        labels = np.argmax(arr, axis=1)
        num_unique_actions = len(np.unique(labels))
        diversity_score = num_unique_actions / max(1, len(labels))
        
        # 3. Координация (изменение действий во времени)
        # Высокая координация = плавные переходы, низкая = хаотичные
        transitions = np.sum(labels[1:] != labels[:-1])
        transition_rate = transitions / max(1, len(labels) - 1)
        coordination_level = 1.0 - min(1.0, transition_rate)  # Инвертируем
        
        # 4. Точность (уверенность в действиях)
        confidences = np.max(arr, axis=1)
        mean_confidence = float(np.mean(confidences))
        precision_score = mean_confidence
        
        # 5. Motion-weighted complexity (если есть motion weights)
        if motion_weights is not None:
            motion_weights = np.array(motion_weights)
            motion_weights = motion_weights / (motion_weights.sum() + 1e-6)
            weighted_entropy = float(np.sum(entropies * motion_weights))
            motion_complexity = weighted_entropy / max(1e-6, np.log(arr.shape[1]))
        else:
            motion_complexity = entropy_complexity
        
        # Итоговый complexity score (комбинация всех факторов)
        complexity_score = (
            0.3 * entropy_complexity +
            0.2 * diversity_score +
            0.2 * coordination_level +
            0.15 * (1.0 - precision_score) +  # Низкая уверенность = выше сложность
            0.15 * motion_complexity
        )
        
        return {
            'complexity_score': float(complexity_score),
            'entropy_complexity': float(entropy_complexity),
            'diversity_score': float(diversity_score),
            'coordination_level': float(coordination_level),
            'action_precision': float(precision_score),
            'motion_complexity': float(motion_complexity),
            'num_unique_actions': int(num_unique_actions)
        }

    # ----------------------------
    # Action planning & intent
    # ----------------------------
    def _predict_action_intent(self, probs_seq: List[np.ndarray], clip_len: int, fps: float) -> Dict[str, Any]:
        """
        Предсказывает будущие действия и намерения на основе временной последовательности.
        """
        if len(probs_seq) < 2:
            return {
                'predicted_action': None,
                'prediction_confidence': 0.0,
                'action_preparation_time': 0.0,
                'intent_score': 0.0
            }
        
        arr = np.stack(probs_seq, axis=0)  # [num_clips, num_classes]
        labels = np.argmax(arr, axis=1)
        
        # 1. Предсказание следующего действия (простая экстраполяция тренда)
        # Используем последние 3 клипа для предсказания
        if len(labels) >= 3:
            recent_labels = labels[-3:]
            # Простое предсказание: наиболее вероятное следующее действие
            # на основе последних вероятностей
            recent_probs = arr[-3:].mean(axis=0)
            predicted_action = int(np.argmax(recent_probs))
            prediction_confidence = float(recent_probs[predicted_action])
        else:
            predicted_action = int(labels[-1])
            prediction_confidence = float(arr[-1, predicted_action])
        
        # 2. Время подготовки к действию (action preparation time)
        # Анализируем, как долго вероятность действия нарастает перед его детекцией
        preparation_times = []
        for action_id in np.unique(labels):
            action_probs = arr[:, action_id]
            # Находим моменты, когда вероятность начинает расти
            threshold = 0.1
            rising_indices = []
            for i in range(1, len(action_probs)):
                if action_probs[i] > threshold and action_probs[i] > action_probs[i-1] * 1.1:
                    rising_indices.append(i)
            
            if rising_indices:
                # Время от начала роста до пика
                peak_idx = int(np.argmax(action_probs))
                if peak_idx > rising_indices[0]:
                    prep_time = (peak_idx - rising_indices[0]) * clip_len / fps
                    preparation_times.append(prep_time)
        
        action_preparation_time = float(np.mean(preparation_times)) if preparation_times else 0.0
        
        # 3. Intent score (оценка намерения)
        # Высокий intent = стабильное нарастание вероятности действия
        intent_scores = []
        for action_id in np.unique(labels):
            action_probs = arr[:, action_id]
            if len(action_probs) > 2:
                # Проверяем тренд роста
                trend = np.polyfit(range(len(action_probs)), action_probs, 1)[0]
                if trend > 0:  # Растущий тренд
                    intent_scores.append(float(trend * np.max(action_probs)))
        
        intent_score = float(np.mean(intent_scores)) if intent_scores else 0.0
        
        result = {
            'predicted_action': int(predicted_action),
            'prediction_confidence': float(prediction_confidence),
            'action_preparation_time': action_preparation_time,
            'intent_score': intent_score
        }
        
        if self.id2label is not None and str(predicted_action) in self.id2label:
            result['predicted_action_label'] = self.id2label[str(predicted_action)]
        
        return result

    # ----------------------------
    # Scene activity type
    # ----------------------------
    def _analyze_scene_activity(self, probs_seq: List[np.ndarray], motion_weights: Optional[List[float]] = None, 
                                clip_len: int = 16, fps: float = 30.0) -> Dict[str, Any]:
        """
        Анализирует общий уровень активности сцены.
        """
        arr = np.stack(probs_seq, axis=0)  # [num_clips, num_classes]
        
        # 1. Action entropy (разнообразие действий)
        action_entropies = [entropy_of_prob(p) for p in arr]
        mean_action_entropy = float(np.mean(action_entropies))
        
        # 2. Action change rate
        labels = np.argmax(arr, axis=1)
        transitions = np.sum(labels[1:] != labels[:-1])
        total_time_sec = len(labels) * clip_len / fps
        action_change_rate = transitions / max(1e-6, total_time_sec)
        
        # 3. Dominant action stability
        unique, counts = np.unique(labels, return_counts=True)
        dominant_action = int(unique[np.argmax(counts)])
        dominant_ratio = float(np.max(counts) / len(labels))
        
        # 4. Number of unique actions
        num_unique_actions = len(unique)
        
        # 5. Motion-weighted activity (если есть motion weights)
        if motion_weights is not None:
            motion_weights = np.array(motion_weights)
            motion_weights = motion_weights / (motion_weights.sum() + 1e-6)
            weighted_entropy = float(np.sum(action_entropies * motion_weights))
        else:
            weighted_entropy = mean_action_entropy
        
        # 6. Классификация типа активности
        # High action intensity: высокая энтропия, высокая частота смены
        # Low action intensity: низкая энтропия, низкая частота смены
        # Chaotic motion: высокая энтропия, высокая частота смены, низкая стабильность
        # Static: низкая энтропия, низкая частота смены, высокая стабильность
        
        activity_scores = {
            'high_action_intensity': (
                0.4 * min(1.0, mean_action_entropy / 3.0) +
                0.3 * min(1.0, action_change_rate / 2.0) +
                0.3 * (1.0 - dominant_ratio)
            ),
            'low_action_intensity': (
                0.4 * max(0.0, 1.0 - mean_action_entropy / 3.0) +
                0.3 * max(0.0, 1.0 - action_change_rate / 2.0) +
                0.3 * dominant_ratio
            ),
            'chaotic_motion': (
                0.3 * min(1.0, mean_action_entropy / 3.0) +
                0.4 * min(1.0, action_change_rate / 2.0) +
                0.3 * (1.0 - dominant_ratio)
            ),
            'static': (
                0.4 * max(0.0, 1.0 - mean_action_entropy / 3.0) +
                0.3 * max(0.0, 1.0 - action_change_rate / 2.0) +
                0.3 * dominant_ratio
            )
        }
        
        scene_activity_type = max(activity_scores, key=activity_scores.get)
        scene_activity_score = activity_scores[scene_activity_type]
        
        return {
            'scene_activity_type': scene_activity_type,
            'scene_activity_score': float(scene_activity_score),
            'activity_scores': {k: float(v) for k, v in activity_scores.items()},
            'action_entropy': mean_action_entropy,
            'action_change_rate': float(action_change_rate),
            'num_unique_actions': int(num_unique_actions),
            'dominant_action_ratio': float(dominant_ratio),
            'weighted_activity_entropy': float(weighted_entropy)
        }

    # ----------------------------
    # Multi-person actions analysis
    # ----------------------------
    def _analyze_multi_person_actions(self, results_per_track: Dict[int, Dict[str, Any]], 
                                      id2label: Optional[Dict] = None) -> Dict[str, Any]:
        """
        Анализирует групповые действия (multi-person actions) на основе действий нескольких людей.
        """
        if len(results_per_track) < 2:
            return {
                'is_multi_person': False,
                'multi_person_actions': [],
                'group_activity_type': None
            }
        
        # Определяем категории групповых действий
        multi_person_categories = {
            'group_walking': ['walk', 'move', 'group'],
            'fighting': ['fight', 'punch', 'kick', 'combat'],
            'hugging': ['hug', 'embrace'],
            'handshakes': ['handshake', 'shake'],
            'arguing': ['argue', 'debate', 'discuss'],
            'teaching': ['teach', 'instruct', 'explain'],
            'collaborating': ['collaborate', 'work', 'together'],
            'crowds_running': ['run', 'crowd', 'group'],
            'dancing_together': ['dance', 'dancing']
        }
        
        # Собираем все действия от всех треков
        all_actions = []
        for track_id, track_results in results_per_track.items():
            dominant_action = track_results.get('dominant_action')
            if dominant_action is not None and id2label:
                if str(dominant_action) in id2label:
                    all_actions.append({
                        'track_id': track_id,
                        'action': id2label[str(dominant_action)],
                        'confidence': track_results.get('dominant_confidence', 0.0)
                    })
        
        # Анализируем групповые действия
        detected_group_actions = []
        action_labels = [a['action'].lower() for a in all_actions]
        
        for category, keywords in multi_person_categories.items():
            matches = sum(1 for label in action_labels if any(kw in label for kw in keywords))
            if matches >= 2:  # Минимум 2 человека с похожими действиями
                detected_group_actions.append({
                    'category': category,
                    'num_participants': matches,
                    'confidence': min(1.0, matches / len(all_actions))
                })
        
        # Определяем тип групповой активности
        if detected_group_actions:
            group_activity_type = max(detected_group_actions, key=lambda x: x['confidence'])['category']
        else:
            # Анализируем синхронность действий
            if len(set(action_labels)) == 1:
                group_activity_type = 'synchronized_activity'
            elif len(set(action_labels)) <= len(all_actions) * 0.5:
                group_activity_type = 'coordinated_activity'
            else:
                group_activity_type = 'independent_activity'
        
        return {
            'is_multi_person': True,
            'num_persons': len(results_per_track),
            'multi_person_actions': detected_group_actions,
            'group_activity_type': group_activity_type,
            'individual_actions': all_actions,
            'action_synchronization': float(1.0 - len(set(action_labels)) / max(1, len(action_labels)))
        }

    # ----------------------------
    # Main API
    # ----------------------------
    def process(self, frame_indices_per_person: Dict[int, List[int]]) -> Dict[int, Dict[str, Any]]:
        all_clips = []
        all_motion = []
        meta = []

        for track_id, indices in frame_indices_per_person.items():
            if len(indices) == 0:
                continue
            frames, motion_mags = self._load_frames_with_motion(indices)
            clips, clips_motion = self._make_clips(frames, motion_mags)

            all_clips.extend(clips)
            all_motion.extend(clips_motion)
            meta.extend([track_id]*len(clips))

        if not all_clips:
            return {}

        probs_all = self._infer(all_clips)

        # Group by track
        per_track_probs = defaultdict(list)
        per_track_motion = defaultdict(list)
        for tid, prob, motion in zip(meta, probs_all, all_motion):
            per_track_probs[tid].append(prob)
            per_track_motion[tid].append(motion)

        results = {}
        for tid in per_track_probs:
            results[tid] = self._aggregate(per_track_probs[tid], motion_weights=per_track_motion[tid])

        # Добавляем multi-person actions анализ, если есть несколько треков
        if len(results) >= 2:
            multi_person = self._analyze_multi_person_actions(results, self.id2label)
            # Добавляем в каждый трек информацию о групповых действиях
            for tid in results:
                results[tid]['multi_person_context'] = multi_person

        return results
