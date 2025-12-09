import os
import math
import psutil
import torch
import sys
import torch.nn.functional as F
from torchvision import transforms
import cv2
import json
import shutil
from pathlib import Path
from typing import List, Tuple, Optional
from numpy.lib.format import open_memmap
import gc
import numpy as np

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FRAME_TMP_PREFIX = "tmp_frames"
CHUNK_SIZE = 64
SCAN_STRIDE = 20

def log(*a, **kw):
    print(*a, file=sys.stderr, **kw)

def segmentation(timeline, fps=30, max_gap_seconds=0.5, max_segment_length_sec=3.0):
    """
    Преобразует список отдельных кадров с лицами в сегменты.
    Разбивает слишком длинные сегменты.
    """
    if not timeline:
        return []
    
    max_gap_frames = int(max_gap_seconds * fps)
    max_segment_frames = int(max_segment_length_sec * fps)
    
    segments = []
    sorted_timeline = sorted(timeline)
    current_start = sorted_timeline[0]
    current_end = sorted_timeline[0]
    
    for i in range(1, len(sorted_timeline)):
        current_frame = sorted_timeline[i]
        prev_frame = sorted_timeline[i-1]
        
        # Проверяем два условия:
        # 1. Разрыв во времени (новая сцена)
        # 2. Сегмент стал слишком длинным
        gap_too_big = (current_frame - prev_frame) > max_gap_frames
        segment_too_long = (current_frame - current_start) > max_segment_frames
        
        if gap_too_big or segment_too_long:
            # Сохраняем текущий сегмент
            segments.append((current_start, current_end))
            # Начинаем новый сегмент
            current_start = current_frame
        
        current_end = current_frame
    
    # Добавляем последний сегмент
    segments.append((current_start, current_end))
    
    return segments

def select_from_segments(segments, total_frames, fps=30, 
                        max_samples_per_segment=10,
                        short_threshold_sec=1.0,
                        medium_threshold_sec=5.0):
    """
    Выбирает кадры из сегментов с разной стратегией в зависимости от длины.
    """
    selected_indices = []
    
    for start, end in segments:
        segment_length_frames = end - start + 1
        segment_length_sec = segment_length_frames / fps
        
        # АДАПТИВНОЕ количество выборок в зависимости от длины сегмента
        if segment_length_sec <= short_threshold_sec:
            # Короткий сегмент (< 1 сек) - берем все
            adaptive_samples = min(segment_length_frames, 5)
            samples = list(range(start, end + 1))
            if len(samples) > adaptive_samples:
                # Берем равномерно
                step = len(samples) // adaptive_samples
                samples = samples[::step][:adaptive_samples]
            
        elif segment_length_sec <= medium_threshold_sec:
            # Средний сегмент (1-5 сек)
            adaptive_samples = min(segment_length_frames, max_samples_per_segment)
            step = max(1, segment_length_frames // adaptive_samples)
            samples = list(range(start, end + 1, step))
            if samples[-1] != end:
                samples.append(end)
                
        else:
            # Длинный сегмент (> 5 сек) - берем пропорционально длине
            # Например, 1 кадр в секунду, но не более max_samples_per_segment * 3
            frames_per_second = max(1, int(fps / 2))  # Половина FPS
            adaptive_samples = min(
                segment_length_frames,
                max_samples_per_segment * 3,
                int(segment_length_sec * frames_per_second)
            )
            
            # Ключевые точки
            key_points = [
                start,
                start + segment_length_frames // 4,
                start + segment_length_frames // 2,
                start + 3 * segment_length_frames // 4,
                end
            ]
            
            samples = set(key_points)
            remaining_slots = adaptive_samples - len(key_points)
            
            if remaining_slots > 0:
                step = segment_length_frames // (remaining_slots + 1)
                for i in range(1, remaining_slots + 1):
                    samples.add(start + i * step)
            
            samples = sorted(samples)
        
        selected_indices.extend(samples)
    
    return sorted(list(set(selected_indices)))

def uniform_time_coverage(total_frames, target_samples=50):
    """
    Равномерная выборка кадров по всему видео.
    
    Args:
        total_frames: int - общее количество кадров
        target_samples: int - сколько кадров выбрать
        
    Returns:
        list[int] - равномерно распределенные индексы
    """
    if total_frames <= target_samples:
        # Если видео короче целевого количества - берем все
        return list(range(total_frames))
    
    # Равномерная выборка с шагом
    step = max(1, total_frames // target_samples)
    indices = list(range(0, total_frames, step))
    
    # Обрезаем до нужного количества
    indices = indices[:target_samples]
    
    # Гарантируем наличие первого и последнего кадра
    if indices[0] != 0:
        indices[0] = 0
    if indices[-1] != total_frames - 1:
        indices.append(total_frames - 1)
    
    return sorted(list(set(indices)))

def build_emotion_curve(emo_results):
    """
    Строит кривые валентности и активации из результатов EmoNet.
    
    Args:
        emo_results: list[dict] - результаты predict_emonet_batch
        
    Returns:
        dict со всеми кривыми и метриками
    """
    valence = []
    arousal = []
    dominant_emotions = []
    emotion_vectors = []
    
    for result in emo_results:
        # Валентность и активация
        valence.append(result.get('valence', 0.0))
        arousal.append(result.get('arousal', 0.0))
        
        # Доминантная эмоция
        emotions = result.get('emotions', {})
        if emotions:
            dominant = max(emotions.items(), key=lambda x: x[1])
            dominant_emotions.append({
                'emotion': dominant[0],
                'confidence': dominant[1]
            })
        else:
            dominant_emotions.append({'emotion': 'Neutral', 'confidence': 1.0})
        
        # Полный вектор эмоций для анализа
        emotion_vector = [emotions.get(e, 0.0) for e in 
                         ['Neutral', 'Happy', 'Sad', 'Surprise', 
                          'Fear', 'Disgust', 'Anger', 'Contempt']]
        emotion_vectors.append(emotion_vector)
    
    # Вычисляем производные (скорость изменения)
    valence_diff = np.diff(valence) if len(valence) > 1 else [0]
    arousal_diff = np.diff(arousal) if len(arousal) > 1 else [0]
    
    # Вычисляем интенсивность (длина вектора в пространстве V-A)
    intensity = [np.sqrt(v**2 + a**2) for v, a in zip(valence, arousal)]
    
    return {
        'valence': valence,
        'arousal': arousal,
        'intensity': intensity,
        'dominant_emotions': dominant_emotions,
        'emotion_vectors': emotion_vectors,
        'valence_diff': list(valence_diff),
        'arousal_diff': list(arousal_diff),
        'combined_diff': [abs(v) + abs(a) for v, a in zip(valence_diff, arousal_diff)]
    }

def detect_keyframes(emotion_curve, EMOTION_CLASSES, threshold=0.3, smooth_window=5):
    """
    Находит ключевые кадры с предварительным сглаживанием.
    """
    # Сначала сглаживаем кривые для удаления шума
    valence_smooth = np.convolve(
        emotion_curve['valence'], 
        np.ones(smooth_window)/smooth_window, 
        mode='same'
    )
    arousal_smooth = np.convolve(
        emotion_curve['arousal'], 
        np.ones(smooth_window)/smooth_window, 
        mode='same'
    )
    
    # Вычисляем изменения на сглаженных кривых
    valence_diff = np.abs(np.diff(valence_smooth))
    arousal_diff = np.abs(np.diff(arousal_smooth))
    combined_diff = (valence_diff + arousal_diff) / 2
    
    # Ищем пики изменений
    keyframes = {}
    
    # Более чувствительный алгоритм для плавных изменений
    for i in range(1, len(combined_diff) - 1):
        # Проверяем локальные максимумы
        is_local_max = (
            combined_diff[i] > combined_diff[i-1] and 
            combined_diff[i] > combined_diff[i+1]
        )
        
        # Адаптивный порог: 30% от максимального изменения
        adaptive_threshold = max(threshold, np.max(combined_diff) * 0.3)
        
        if is_local_max and combined_diff[i] > adaptive_threshold:
            keyframes[i] = {
                'type': 'transition',
                'score': float(combined_diff[i]),
                'valence_change': float(valence_diff[i]),
                'arousal_change': float(arousal_diff[i])
            }
    
    # Также ищем абсолютные максимумы каждой эмоции
    emotion_vectors = np.array(emotion_curve['emotion_vectors'])
    for emotion_idx in range(emotion_vectors.shape[1]):
        emotion_curve_smooth = np.convolve(
            emotion_vectors[:, emotion_idx],
            np.ones(3)/3,
            mode='same'
        )
        
        # Ищем локальные максимумы для каждой эмоции
        for i in range(1, len(emotion_curve_smooth) - 1):
            if (emotion_curve_smooth[i] > emotion_curve_smooth[i-1] and 
                emotion_curve_smooth[i] > emotion_curve_smooth[i+1] and
                emotion_curve_smooth[i] > 0.5):  # Порог уверенности
                
                if i not in keyframes or keyframes[i]['score'] < emotion_curve_smooth[i]:
                    keyframes[i] = {
                        'type': 'emotion_peak',
                        'score': float(emotion_curve_smooth[i]),
                        'emotion_idx': emotion_idx,
                        'emotion_name': list(EMOTION_CLASSES.values())[emotion_idx]
                    }
    
    return dict(sorted(keyframes.items(), key=lambda x: x[1]['score'], reverse=True))

def compress_sequence(selected_indices, emo_results, keyframes_indices, target_length):
    """
    Сжимает длинную последовательность до target_length.
    """
    n_original = len(selected_indices)
    
    # Гарантированно включаем keyframes (самые важные)
    keyframe_idxs = list(keyframes_indices.keys())
    keyframe_idxs = sorted(keyframe_idxs[:target_length // 2])  # Половина слотов для ключевых
    
    selected_idxs_set = set(keyframe_idxs)
    selected_emotions = [emo_results[i] for i in keyframe_idxs]
    
    # Равномерно выбираем остальные из неключевых кадров
    remaining_slots = target_length - len(keyframe_idxs)
    non_keyframe_idxs = [i for i in range(n_original) if i not in selected_idxs_set]
    
    if remaining_slots > 0 and non_keyframe_idxs:
        step = max(1, len(non_keyframe_idxs) // remaining_slots)
        for i in range(0, len(non_keyframe_idxs), step):
            if len(selected_idxs_set) < target_length:
                idx = non_keyframe_idxs[i]
                selected_idxs_set.add(idx)
                selected_emotions.append(emo_results[idx])
    
    # Преобразуем обратно в глобальные индексы
    final_indices = [selected_indices[i] for i in sorted(selected_idxs_set)]
    
    return final_indices[:target_length], selected_emotions[:target_length]

def slightly_modify_emotion(emotion, noise_scale=0.05):
    """
    Создает слегка модифицированную версию эмоции.
    Используется для дублирования ключевых кадров с вариациями.
    
    Args:
        emotion: dict - исходные эмоции
        noise_scale: float - масштаб шума (0-1)
    
    Returns:
        dict - модифицированные эмоции
    """
    modified = emotion.copy()
    
    # Добавляем небольшой шум к валентности и активации
    if 'valence' in modified:
        modified['valence'] += np.random.uniform(-noise_scale, noise_scale)
        modified['valence'] = np.clip(modified['valence'], -1, 1)
    
    if 'arousal' in modified:
        modified['arousal'] += np.random.uniform(-noise_scale, noise_scale)
        modified['arousal'] = np.clip(modified['arousal'], -1, 1)
    
    # Добавляем шум к вероятностям эмоций
    if 'emotions' in modified:
        emotions = modified['emotions'].copy()
        
        # Выбираем случайное изменение для каждой эмоции
        for key in emotions:
            change = np.random.uniform(-noise_scale/2, noise_scale/2)
            emotions[key] = np.clip(emotions[key] + change, 0, 1)
        
        # Нормализуем чтобы сумма была 1
        total = sum(emotions.values())
        if total > 0:
            for key in emotions:
                emotions[key] /= total
        
        modified['emotions'] = emotions
    
    return modified

def interpolate_emotions(emotions, n_points=10, method='linear'):
    """
    Создает интерполированные эмоциональные состояния между существующими.
    
    Args:
        emotions: list[dict] - исходные эмоции
        n_points: int - сколько точек интерполяции создать
        method: str - метод интерполяции ('linear' или 'cubic')
    
    Returns:
        list[dict] - интерполированные эмоции
    """
    from scipy import interpolate

    if len(emotions) < 2 or n_points <= 0:
        return []
    
    # Подготавливаем данные для интерполяции
    valence = [e.get('valence', 0) for e in emotions]
    arousal = [e.get('arousal', 0) for e in emotions]
    
    # Матрица вероятностей эмоций
    emotion_keys = ['Neutral', 'Happy', 'Sad', 'Surprise', 
                   'Fear', 'Disgust', 'Anger', 'Contempt']
    emotion_probs = []
    
    for e in emotions:
        probs = [e.get('emotions', {}).get(key, 0) for key in emotion_keys]
        emotion_probs.append(probs)
    
    # Создаем интерполированные точки
    original_indices = np.linspace(0, len(emotions)-1, len(emotions))
    new_indices = np.linspace(0, len(emotions)-1, len(emotions) + n_points)
    
    # Интерполируем валентность и активацию
    if method == 'cubic' and len(emotions) >= 4:
        valence_interp = interpolate.interp1d(original_indices, valence, kind='cubic')
        arousal_interp = interpolate.interp1d(original_indices, arousal, kind='cubic')
    else:
        # Линейная интерполяция
        valence_interp = interpolate.interp1d(original_indices, valence, kind='linear')
        arousal_interp = interpolate.interp1d(original_indices, arousal, kind='linear')
    
    # Интерполируем вероятности эмоций
    emotion_interps = []
    for i in range(len(emotion_keys)):
        probs = [p[i] for p in emotion_probs]
        if method == 'cubic' and len(emotions) >= 4:
            interp = interpolate.interp1d(original_indices, probs, kind='cubic')
        else:
            interp = interpolate.interp1d(original_indices, probs, kind='linear')
        emotion_interps.append(interp)
    
    # Создаем интерполированные эмоции
    interpolated = []
    for idx in new_indices:
        # Пропускаем исходные точки
        if idx in original_indices:
            continue
        
        # Получаем интерполированные значения
        try:
            v = float(valence_interp(idx))
            a = float(arousal_interp(idx))
            
            # Интерполируем вероятности эмоций
            emotion_probs_interp = {}
            for i, key in enumerate(emotion_keys):
                prob = float(emotion_interps[i](idx))
                emotion_probs_interp[key] = max(0, prob)
            
            # Нормализуем вероятности
            total = sum(emotion_probs_interp.values())
            if total > 0:
                for key in emotion_probs_interp:
                    emotion_probs_interp[key] /= total
            
            interpolated.append({
                'valence': v,
                'arousal': a,
                'emotions': emotion_probs_interp,
                'is_interpolated': True
            })
        except:
            continue
    
    return interpolated[:n_points]

def expand_sequence(selected_indices, emo_results, keyframes_indices, target_length):
    """
    Расширяет короткую последовательность до target_length.
    """
    n_original = len(selected_indices)
    
    # 1. Стратегическое дублирование keyframes
    expanded_indices = list(selected_indices)
    expanded_emotions = list(emo_results)
    
    keyframe_idxs = list(keyframes_indices.keys())
    keyframe_idxs = keyframe_idxs[:min(len(keyframe_idxs), target_length // 3)]
    
    for idx in keyframe_idxs:
        # Дублируем каждый ключевой кадр 1-2 раза с небольшими вариациями
        for dup in range(1, 3):
            if len(expanded_indices) >= target_length:
                break
            
            # Добавляем "слегка измененную" версию эмоций
            modified_emotion = slightly_modify_emotion(emo_results[idx])
            expanded_emotions.append(modified_emotion)
            expanded_indices.append(selected_indices[idx])  # Тот же индекс
    
    # 2. Временная интерполяция между соседними кадрами
    if len(expanded_indices) < target_length:
        n_needed = target_length - len(expanded_indices)
        interpolated = interpolate_emotions(emo_results, n_points=n_needed)
        expanded_emotions.extend(interpolated)
        # Для интерполированных кадров используем специальные индексы
        expanded_indices.extend([-1] * len(interpolated))
    
    return expanded_indices[:target_length], expanded_emotions[:target_length]

def temporal_smoothing(emotions, window=3):
    """
    Применяет скользящее среднее для сглаживания эмоциональных кривых.
    
    Args:
        emotions: list[dict] - список словарей с эмоциями
        window: int - размер окна сглаживания (нечетное)
    
    Returns:
        list[dict] - сглаженные эмоции
    """
    if window < 1 or len(emotions) <= window:
        return emotions
    
    n = len(emotions)
    half_window = window // 2
    smoothed = []
    
    for i in range(n):
        # Определяем границы окна
        start = max(0, i - half_window)
        end = min(n, i + half_window + 1)
        
        window_emotions = emotions[start:end]
        
        # Сглаживаем валентность и активацию
        valence_sum = sum(e.get('valence', 0) for e in window_emotions)
        arousal_sum = sum(e.get('arousal', 0) for e in window_emotions)
        
        smoothed_valence = valence_sum / len(window_emotions)
        smoothed_arousal = arousal_sum / len(window_emotions)
        
        # Для эмоций используем среднее вероятностей
        emotion_keys = ['Neutral', 'Happy', 'Sad', 'Surprise', 
                       'Fear', 'Disgust', 'Anger', 'Contempt']
        
        smoothed_emotion_probs = {}
        for key in emotion_keys:
            prob_sum = sum(e.get('emotions', {}).get(key, 0) for e in window_emotions)
            smoothed_emotion_probs[key] = prob_sum / len(window_emotions)
        
        # Нормализуем вероятности эмоций (чтобы сумма была = 1)
        total = sum(smoothed_emotion_probs.values())
        if total > 0:
            for key in smoothed_emotion_probs:
                smoothed_emotion_probs[key] /= total
        
        # Создаем сглаженный результат
        smoothed.append({
            'valence': smoothed_valence,
            'arousal': smoothed_arousal,
            'emotions': smoothed_emotion_probs
        })
    
    return smoothed

def validate_sequence_quality(emotions, min_length=20, min_diversity_threshold=0.2, is_static_face=False, neutral_percentage=0.0, logger=None):
    """
    Проверяет качество эмоциональной последовательности.
    Для монотонных видео с нейтральными эмоциями снижаем требования.
    """
    if len(emotions) < min_length:
        return {
            'is_valid': False,
            'reason': f'Sequence too short: {len(emotions)} < {min_length}',
            'metrics': {},
            'overall_score': 0,
            'is_monotonic': False
        }
    
    # Извлекаем данные
    valence = [e.get('valence', 0) for e in emotions]
    arousal = [e.get('arousal', 0) for e in emotions]
    
    # 1. Эмоциональное разнообразие
    emotion_counts = {}
    for e in emotions:
        emotions_dict = e.get('emotions', {})
        if emotions_dict:
            dominant = max(emotions_dict.items(), key=lambda x: x[1])[0]
            emotion_counts[dominant] = emotion_counts.get(dominant, 0) + 1
    
    # Коэффициент Шеннона
    total = sum(emotion_counts.values())
    entropy = 0
    for count in emotion_counts.values():
        p = count / total
        if p > 0:
            entropy -= p * math.log2(p)
    
    max_entropy = math.log2(len(emotion_counts)) if emotion_counts else 0
    diversity_score = entropy / max_entropy if max_entropy > 0 else 0
    
    # 2. Наличие переходов
    valence_changes = np.abs(np.diff(valence))
    arousal_changes = np.abs(np.diff(arousal))
    
    significant_transitions = sum(1 for v in valence_changes if v > 0.3)
    significant_transitions += sum(1 for a in arousal_changes if a > 0.3)
    
    transition_score = min(1.0, significant_transitions / 5)
    
    # 3. Отсутствие монотонности
    similarity_threshold = 0.1
    max_monotonic_streak = 0
    current_streak = 1
    
    for i in range(1, len(emotions)):
        v_diff = abs(valence[i] - valence[i-1])
        a_diff = abs(arousal[i] - arousal[i-1])
        
        if v_diff < similarity_threshold and a_diff < similarity_threshold:
            current_streak += 1
            max_monotonic_streak = max(max_monotonic_streak, current_streak)
        else:
            current_streak = 1
    
    monotonicity_score = 1.0 - min(1.0, max_monotonic_streak / len(emotions))
    
    # 4. Дисперсия
    valence_var = np.var(valence) if len(valence) > 1 else 0
    arousal_var = np.var(arousal) if len(arousal) > 1 else 0
    variance_score = min(1.0, (valence_var + arousal_var) / 0.5)
    
    # 5. Итоговый скоринг
    weights = {
        'diversity': 0.45,
        'transitions': 0.45,
        'monotonicity': 0.35,
        'variance': 0.35
    }
    
    overall_score = (
        diversity_score * weights['diversity'] +
        transition_score * weights['transitions'] +
        monotonicity_score * weights['monotonicity'] +
        variance_score * weights['variance']
    )
    
    # АДАПТИВНЫЕ ПОРОГИ для монотонных видео
    is_monotonic_video = (neutral_percentage > 0.7 or 
                         (diversity_score < 0.1 and significant_transitions < 2))
    
    if is_monotonic_video:
        # Для монотонных видео сильно снижаем требования
        quality_threshold = 0.2
        diversity_threshold = 0.05
        log_message = "Монотонное видео: снижаю требования к качеству"
    elif is_static_face:
        # Для статичных лиц снижаем требования
        quality_threshold = 0.3
        diversity_threshold = max(0.05, min_diversity_threshold * 0.5)
        log_message = "Статичное лицо: снижаю требования"
    else:
        # Стандартные требования
        quality_threshold = 0.4
        diversity_threshold = min_diversity_threshold
        log_message = "Стандартные требования"
    
    is_valid = overall_score >= quality_threshold and diversity_score >= diversity_threshold

    logger.log(f"[VALIDATION QUALITY] overall_score: {overall_score} |>=| quality_threshold: {quality_threshold} | diversity_score: {diversity_score} |>=| diversity_threshold: {diversity_threshold}")
    
    return {
        'is_valid': bool(is_valid),
        'is_monotonic': bool(is_monotonic_video),
        'overall_score': float(round(overall_score, 3)),
        'log_message': str(log_message),
        'metrics': {
            'diversity_score': float(round(diversity_score, 3)),
            'transition_score': float(round(transition_score, 3)),
            'monotonicity_score': float(round(monotonicity_score, 3)),
            'variance_score': float(round(variance_score, 3)),
            'different_emotions': int(len(emotion_counts)),
            'significant_transitions': int(significant_transitions),
            'max_monotonic_streak': int(max_monotonic_streak),
            'sequence_length': int(len(emotions)),
            'neutral_percentage': float(neutral_percentage)
        }
    }

def save_for_user(data, video_path, output_dir='user_results'):
    """
    Сохраняет детализированные результаты для пользователя.
    """
    import json
    import os
    from datetime import datetime
    
    os.makedirs(output_dir, exist_ok=True)
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output_dir, f"{video_name}_{timestamp}_analysis.json")
    
    # Функция для преобразования несериализуемых типов
    def make_serializable(obj):
        if isinstance(obj, bool):
            return bool(obj)  # Явно преобразуем в bool
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)  # numpy types -> float
        elif isinstance(obj, np.ndarray):
            return obj.tolist()  # numpy array -> list
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(item) for item in obj]
        elif hasattr(obj, '__dict__'):
            return make_serializable(obj.__dict__)
        else:
            return obj
    
    # Подготавливаем данные для сериализации
    serializable_data = make_serializable({
        'metadata': {
            'video_path': str(video_path),  # Преобразуем Path в строку
            'video_name': str(video_name),
            'analysis_timestamp': str(timestamp),
            'processing_version': '1.0'
        },
        'summary': {
            'total_frames_analyzed': len(data.get('original_emotions', [])),
            'keyframes_count': len(data.get('keyframes', [])),
            'dominant_emotion': None,
            'is_static_face': data.get('processing_stats', {}).get('faces_found', 0) > 0.8 * data.get('processing_stats', {}).get('total_frames', 1)
        },
        'keyframes': data.get('keyframes', []),
        'emotion_profile': {
            'dominant_emotion': data.get('processing_stats', {}).get('dominant_emotion', 'Unknown'),
            'neutral_percentage': data.get('processing_stats', {}).get('neutral_percentage', 0),
            'valence_avg': data.get('processing_stats', {}).get('valence_avg', 0),
            'arousal_avg': data.get('processing_stats', {}).get('arousal_avg', 0)
        },
        'quality_metrics': data.get('quality_metrics', {}),
        'processing_stats': data.get('processing_stats', {}),
        'is_monotonic': data.get('quality_metrics', {}).get('is_monotonic', False),
        'is_valid': data.get('quality_metrics', {}).get('is_valid', False)
    })
    
    # Сохраняем в файл
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(serializable_data, f, indent=2, ensure_ascii=False)
    
    print(f"[INFO] User analysis saved to: {output_file}")
    return output_file

def save_for_model(data, video_path, output_dir='model_data'):
    """
    Сохраняет нормализованные данные для обучения модели.
    """
    import numpy as np
    import json
    import os
    from datetime import datetime
    
    os.makedirs(output_dir, exist_ok=True)
    
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    # Функция для преобразования данных
    def prepare_for_json(obj):
        if isinstance(obj, bool):
            return bool(obj)
        elif isinstance(obj, (int, float)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: prepare_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [prepare_for_json(item) for item in obj]
        else:
            return obj
    
    # Сохраняем эмоции в numpy формате
    emotions_array = []
    for emo in data.get('emotions', []):
        vector = [emo.get('valence', 0), emo.get('arousal', 0)]
        
        emotion_order = ['Neutral', 'Happy', 'Sad', 'Surprise', 
                        'Fear', 'Disgust', 'Anger', 'Contempt']
        
        emotions_dict = emo.get('emotions', {})
        for emotion in emotion_order:
            vector.append(emotions_dict.get(emotion, 0))
        
        emotions_array.append(vector)
    
    # Сохраняем как numpy файл
    np_array = np.array(emotions_array, dtype=np.float32)
    npy_file = os.path.join(output_dir, f"{video_name}_{timestamp}_emotions.npy")
    np.save(npy_file, np_array)
    
    # Сохраняем метаданные
    metadata = prepare_for_json({
        'video_name': str(video_name),
        'original_path': str(video_path),
        'processing_timestamp': str(timestamp),
        'sequence_length': int(len(np_array)),
        'feature_dim': int(np_array.shape[1]),
        'frame_indices': [int(idx) for idx in data.get('indices', [])],
        'video_metadata': data.get('video_metadata', {}),
        'normalized': True,
        'quality_score': float(data.get('quality_score', 0)),
        'processing_attempt': int(data.get('processing_attempt', 0)),
        'data_format': {
            'columns': ['valence', 'arousal'] + 
                      ['Neutral', 'Happy', 'Sad', 'Surprise', 
                       'Fear', 'Disgust', 'Anger', 'Contempt'],
            'dtype': 'float32',
            'shape': [int(dim) for dim in np_array.shape]
        }
    })
    
    meta_file = os.path.join(output_dir, f"{video_name}_{timestamp}_meta.json")
    with open(meta_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    print(f"[INFO] Model data saved: {npy_file}")
    print(f"[INFO] Metadata saved: {meta_file}")
    
    return {
        'npy_file': str(npy_file),
        'meta_file': str(meta_file),
        'array_shape': [int(dim) for dim in np_array.shape]
    }

def get_video_type(timeline, total_frames, segments):
    if len(timeline) > total_frames * 0.8:
        return "STATIC_FACE"
    elif len(segments) == 1:
        return "CONTINUOUS_FACE"
    else:
        return "DYNAMIC_FACES"

def adaptive_params(current_params, retry_count, diversity_score, transition_count, video_type, segments_count, faces_found, neutral_percentage, log):
    if retry_count == 1:
        if neutral_percentage > 0.8:  # Если >80% нейтральных эмоций
            log("[Retry] Видео явно монотонное, сразу снижаю требования")
            current_params['min_diversity'] = 0.05
            current_params['quality_threshold'] = 0.15
            return current_params  # Сразу возвращаем

        if video_type == "STATIC_FACE" or (segments_count == 1 and faces_found > 100):
            # Статичное лицо или один длинный сегмент
            current_params['samples_per_segment'] = 100
            current_params['segment_max_gap'] = 0.2
            current_params['keyframe_threshold'] = 0.15
            current_params['min_diversity'] = 0.1
            log("[Retry] Стратегия для STATIC_FACE: Увеличиваю выборку, снижаю требования")
        
        elif diversity_score < 0.2:
            current_params['keyframe_threshold'] = 0.1
            current_params['samples_per_segment'] = 80
            current_params['segment_max_gap'] = 0.3
            log("[Retry] Стратегия для LOW_DIVERSITY: Максимальная детализация")
        
        elif transition_count < 2:
            current_params['quality_threshold'] = 0.25
            current_params['min_diversity'] = 0.1
            current_params['samples_per_segment'] = 60
            log("[Retry] Стратегия для FEW_TRANSITIONS: Снижаю требования, увеличиваю выборку")
    
    return current_params

def analyze_emotion_profile(emo_results):
    """Анализирует, какие эмоции преобладают."""
    emotion_totals = {}
    valence_avg = 0
    arousal_avg = 0
    
    for result in emo_results:
        valence_avg += result.get('valence', 0)
        arousal_avg += result.get('arousal', 0)
        
        emotions = result.get('emotions', {})
        if emotions:
            dominant = max(emotions.items(), key=lambda x: x[1])[0]
            emotion_totals[dominant] = emotion_totals.get(dominant, 0) + 1
    
    valence_avg /= len(emo_results)
    arousal_avg /= len(emo_results)
    
    # Определяем доминантную эмоцию
    dominant_emotion = None
    if emotion_totals:
        dominant_emotion = max(emotion_totals.items(), key=lambda x: x[1])[0]
    
    return {
        'dominant_emotion': dominant_emotion,
        'emotion_distribution': emotion_totals,
        'valence_avg': valence_avg,
        'arousal_avg': arousal_avg,
        'is_neutral_dominant': dominant_emotion == 'Neutral',
        'neutral_percentage': emotion_totals.get('Neutral', 0) / len(emo_results) if len(emo_results) > 0 else 0
    }

def sample_for_static_face(segments, total_frames, fps, target_samples=100):
    """
    Специальная выборка для статичных лиц.
    Берет больше кадров в начале, середине и конце.
    """
    selected = []
    
    for start, end in segments:
        length = end - start + 1
        duration = length / fps
        
        # Для очень длинных сегментов используем стратегию:
        # - Чаще в начале (первые 3 секунды)
        # - Реже в середине
        # - Чаще в конце (последние 3 секунды)
        
        # Начало (первые 3 секунды)
        start_frames = min(int(3 * fps), length // 3)
        for i in range(0, start_frames, max(1, start_frames // 10)):
            selected.append(start + i)
        
        # Середина (выборка)
        middle_start = start + start_frames
        middle_end = end - min(int(3 * fps), length // 3)
        middle_length = middle_end - middle_start + 1
        
        if middle_length > 0:
            step = max(1, middle_length // 20)
            for i in range(middle_start, middle_end + 1, step):
                selected.append(i)
        
        # Конец (последние 3 секунды)
        end_frames = min(int(3 * fps), length // 3)
        for i in range(max(0, length - end_frames), length, max(1, end_frames // 10)):
            selected.append(start + i)
    
    # Ограничиваем и убираем дубликаты
    selected = sorted(list(set(selected)))
    
    if len(selected) > target_samples:
        step = len(selected) // target_samples
        selected = selected[::step][:target_samples]
    
    return selected

def analyze_emotion_changes(emo_results, window=5):
    """Анализирует характер изменений эмоций"""
    valence = [e.get('valence', 0) for e in emo_results]
    arousal = [e.get('arousal', 0) for e in emo_results]
    
    # Вычисляем изменения
    valence_changes = np.abs(np.diff(valence))
    arousal_changes = np.abs(np.diff(arousal))
    
    # Характер изменений: резкие скачки vs плавные изменения
    sharp_transitions = sum(1 for v in valence_changes if v > 0.3)
    sharp_transitions += sum(1 for a in arousal_changes if a > 0.3)
    
    # Плавные изменения (мелкие, но частые)
    smooth_changes = sum(1 for v in valence_changes if 0.05 < v <= 0.15)
    smooth_changes += sum(1 for a in arousal_changes if 0.05 < a <= 0.15)
    
    # Общая активность изменений
    total_change_magnitude = np.sum(valence_changes) + np.sum(arousal_changes)
    avg_change_magnitude = total_change_magnitude / len(valence_changes) if len(valence_changes) > 0 else 0
    
    return {
        'sharp_transitions': int(sharp_transitions),
        'smooth_changes': int(smooth_changes),
        'total_change_magnitude': float(total_change_magnitude),
        'avg_change_magnitude': float(avg_change_magnitude),
        'change_type': 'sharp' if sharp_transitions > smooth_changes else 'smooth'
    }

def print_memory_usage(label="", log=None):
    """Выводит использование памяти текущим процессом"""
    process = psutil.Process(os.getpid())
    
    # В байтах
    memory_info = process.memory_info()
    
    # В мегабайтах
    rss_mb = memory_info.rss / 1024 / 1024  # Resident Set Size (физическая память)
    vms_mb = memory_info.vms / 1024 / 1024  # Virtual Memory Size
    
    # Процент использования от общей памяти
    memory_percent = process.memory_percent()
    
    log(f"[MEMORY {label}] RSS: {rss_mb:.1f} MB | VMS: {vms_mb:.1f} MB | {memory_percent:.1f}%")

def get_available_memory_mb():
    """Возвращает доступную память в MB"""
    return psutil.virtual_memory().available / 1024 / 1024

def calculate_max_frames_by_memory(image_shape, available_memory_mb, safety_factor=0.5):
    """
    Рассчитывает максимальное количество кадров, которое можно обработать
    с учетом доступной памяти.
    
    Args:
        image_shape: (H, W, C) - размер одного кадра
        available_memory_mb: доступная память в MB
        safety_factor: коэффициент безопасности (0.5 = использовать только 50% памяти)
    
    Returns:
        int: максимальное количество кадров
    """
    if image_shape is None:
        return 300  # значение по умолчанию
    
    H, W, C = image_shape
    bytes_per_frame = H * W * C  # uint8 = 1 байт на канал
    
    # Учитываем дополнительную память для:
    # 1. Кадры в RAM (оригиналы)
    # 2. Кадры после конвертации в RGB
    # 3. Тензоры для модели
    # 4. Результаты
    memory_per_frame_mb = (bytes_per_frame * 3) / 1024 / 1024  # ×3 для запаса
    
    max_frames = int((available_memory_mb * safety_factor) / memory_per_frame_mb)
    
    # Минимальные и максимальные ограничения
    max_frames = max(50, min(max_frames, 1000))
    
    return max_frames

def cleanup_memory():
    """Полная очистка памяти между попытками"""
    gc.collect()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Принудительный сбор мусора
    for i in range(3):
        gc.collect()

def compute_steps(total_frames, MAX_SCANS=2000, MIN_SCANS=50, BASE=500, SCALE_FACTOR=200):
    import math
    try:
        # FIX 1: Правильно вычисляем target_scans
        target_scans = MIN_SCANS + SCALE_FACTOR * math.log2(total_frames / BASE + 1)
        target_scans = int(min(MAX_SCANS, max(MIN_SCANS, target_scans)))
        target_scans = min(target_scans, total_frames)  # Не больше общего числа кадров
    except Exception as e:
        log(f"Ошибка расчета target_scans: {e}")
        target_scans = min(MAX_SCANS, total_frames // 10)
    
    try:
        # FIX 2: scan_stride должен быть целым числом
        scan_stride = max(1, total_frames // target_scans)
        scan_stride = int(scan_stride)
    except Exception as e:
        log(f"Ошибка расчета scan_stride: {e}")
        scan_stride = max(1, total_frames // 100)
    
    return scan_stride, target_scans 

def frame_writer(video_path: str, out_dir: str, batch_size: int = 512, logger=None):
    os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    
    # Получаем FPS из видео
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30  # значение по умолчанию

    p_meta = os.path.join(out_dir, "metadata.json")

    if os.path.exists(p_meta):
        with open(p_meta, "r") as f:
            meta = json.load(f)
        cap.release()
        return meta
    else:
        meta = {
            "total_frames": 0,
            "batch_size": batch_size,
            "fps": fps,
            "batches": []
        }

    H = W = C = None
    batch_id = 0
    buf_count = 0
    arr = None
    current_path = None  # Добавлено: храним path для частичного батча

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if H is None:
            H, W, C = frame.shape
            # Добавлено: сохраняем размеры в meta
            meta["height"] = H
            meta["width"] = W
            meta["channels"] = C

        # фиксируем возможные битые размеры
        if frame.shape != (H, W, C):
            frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)

        # создаём mmap под батч
        if buf_count == 0:
            fname = f"batch_{batch_id:05d}.npy"
            current_path = os.path.join(out_dir, fname)

            arr = open_memmap(
                filename=current_path,
                mode="w+",
                dtype=np.uint8,
                shape=(batch_size, H, W, C)
            )

        arr[buf_count] = frame
        buf_count += 1
        meta["total_frames"] += 1

        # батч заполнен → записываем мету, закрываем
        if buf_count == batch_size:
            arr.flush()
            del arr  
            arr = None  # Очистка

            logger.log(f"Записан Батч №{batch_id+1} | start: {batch_id * batch_size} | end: {batch_id * batch_size + buf_count - 1}")

            meta["batches"].append({
                "batch_index": batch_id,
                "path": fname,
                "start_frame": batch_id * batch_size,
                "end_frame": batch_id * batch_size + buf_count - 1
            })

            buf_count = 0
            batch_id += 1

    # последний неполный батч
    if buf_count > 0:
        arr.flush()
        # Добавлено: обрезаем файл до реального размера (для экономии диска и точности)
        frame_bytes = H * W * C
        actual_bytes = buf_count * frame_bytes
        os.truncate(current_path, actual_bytes)
        del arr  
        arr = None  # Очистка

        fname = f"batch_{batch_id:05d}.npy"  # Уже определён
        meta["batches"].append({
            "batch_index": batch_id,
            "path": fname,
            "start_frame": batch_id * batch_size,
            "end_frame": batch_id * batch_size + buf_count - 1
        })

    with open(os.path.join(out_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)

    cap.release()

    return meta

class FrameManager:

    def __init__(self, frames_dir: str, chunk_size: int = CHUNK_SIZE, cache_size: int = 2):
        self.frames_dir = Path(frames_dir)
        meta_path = self.frames_dir / "metadata.json"
        if not meta_path.exists():
            raise FileNotFoundError(f"metadata.json not found in {self.frames_dir}")
        meta = json.load(open(meta_path, "r", encoding="utf-8"))
        self.total_frames = int(meta["total_frames"])
        self.batch_size = int(meta.get("batch_size", chunk_size))
        self.batches = meta["batches"]
        # Добавлено: читаем размеры из meta
        self.height = meta["height"]
        self.width = meta["width"]
        self.channels = meta["channels"]
        self.fps = meta.get("fps", 30)
        # create mapping of batch_index -> batch_info
        self.batch_by_idx = {b["batch_index"]: b for b in self.batches}
        self.cache = {}  # pid -> memmap ndarray
        self.cache_order = []  # LRU list
        self.cache_size = int(cache_size)

    def _load_batch_pid(self, pid: int):
        if pid in self.cache:
            # move to MRU
            try:
                self.cache_order.remove(pid)
            except ValueError:
                pass
            self.cache_order.append(pid)
            return self.cache[pid]
        batch = self.batch_by_idx.get(pid)
        if batch is None:
            raise IndexError(f"batch {pid} not found")
        p = self.frames_dir / batch["path"]
        # Добавлено: вычисляем реальный num_frames для батча (для частичных)
        num_frames = batch["end_frame"] - batch["start_frame"] + 1
        # Изменено: используем np.memmap вместо np.load (для raw файлов без заголовка)
        arr = np.memmap(
            str(p),
            dtype=np.uint8,
            mode="r",
            shape=(num_frames, self.height, self.width, self.channels)
        )
        self.cache[pid] = arr
        self.cache_order.append(pid)
        # evict
        while len(self.cache_order) > self.cache_size:
            ev = self.cache_order.pop(0)
            self.cache.pop(ev, None)
        return arr

    def get(self, idx: int) -> np.ndarray:
        if idx < 0 or idx >= self.total_frames:
            raise IndexError("Frame index out of bounds")
        pid = idx // self.batch_size
        local = idx - pid * self.batch_size
        arr = self._load_batch_pid(pid)
        if local >= arr.shape[0]:
            # should not happen but guard
            raise IndexError(f"Local index {local} >= chunk size {arr.shape[0]} (global {idx})")
        return np.asarray(arr[local])

    # Добавлено: метод для очистки кэша (вызовем после видео)
    def close(self):
        self.cache.clear()
        self.cache_order.clear()

def safe_det_score(face) -> float:
    return float(getattr(face, "det_score", getattr(face, "score", 0.0) or 0.0))

def detect_face(frame_bgr: np.ndarray, face_app, thr: float = 0.5) -> bool:
    """
    frame_bgr: OpenCV BGR ndarray
    Returns True if any face >= thr
    """
    try:
        faces = face_app.get(frame_bgr)
        if not faces:
            return False
        best = max(safe_det_score(f) for f in faces)
        return best >= thr
    except Exception:
        return False

def scan_for_faces(fm: FrameManager, face_app, scan_stride: int = SCAN_STRIDE, detect_thr: float = 0.5) -> List[int]:
    """
    Scan video for face presence. Returns sorted list of frame indices where face detected.
    scan_stride can be >1 to speed up scanning.
    """
    total = fm.total_frames
    timeline = []
    cn = 0
    # iterate with stride, but we will later use windows around hits
    for i in range(0, total, scan_stride):
        try:
            frame = fm.get(i)
            cn += 1
        except IndexError:
            continue
        # detect expects BGR
        if detect_face(frame, face_app, thr=detect_thr):
            timeline.append(i)
    # Optionally refine: try to fill small gaps by checking neighbors? we'll keep it simple
    return timeline, cn

EMOTION_CLASSES = {
    0: "Neutral", 1: "Happy", 2: "Sad", 3: "Surprise",
    4: "Fear", 5: "Disgust", 6: "Anger", 7: "Contempt"
}

preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def choose_batch_size_by_vram():
    if not torch.cuda.is_available():
        return 1
    props = torch.cuda.get_device_properties(0)
    total_gb = props.total_memory / (1024 ** 3)
    # heuristics
    if total_gb >= 24:
        return 64
    if total_gb >= 12:
        return 32
    if total_gb >= 8:
        return 16
    if total_gb >= 6:
        return 8
    if total_gb >= 4:
        return 4
    return 1

def predict_emonet_batch(frames: List[np.ndarray], model, batch_size: Optional[int] = None, use_amp: bool = True):
    """
    frames: list of RGB ndarrays (H,W,3)
    returns list of dicts {valence, arousal, emotions: {label:prob,...}}
    """
    if batch_size is None:
        batch_size = choose_batch_size_by_vram()
    results = []
    model_device = DEVICE

    # move model already loaded to DEVICE
    for i in range(0, len(frames), batch_size):
        chunk = frames[i:i + batch_size]
        tensors = [preprocess(f) for f in chunk]
        batch_tensor = torch.stack(tensors).to(model_device)

        # inference
        try:
            if use_amp and torch.cuda.is_available():
                with torch.amp.autocast("cuda"):
                    out = model(batch_tensor)
            else:
                out = model(batch_tensor)
        except RuntimeError as e:
            # OOM guard: reduce batch and retry single-batch fallback
            torch.cuda.empty_cache()
            if batch_size > 1:
                # recursively try with smaller batch
                return predict_emonet_batch(frames, model, batch_size=max(1, batch_size // 2), use_amp=use_amp)
            else:
                raise e

        vals = out["valence"].detach().cpu().numpy()
        arous = out["arousal"].detach().cpu().numpy()
        logits = out["expression"].detach()
        probs = F.softmax(logits, dim=1).cpu().numpy()

        for j in range(len(chunk)):
            results.append({
                "valence": float(vals[j]),
                "arousal": float(arous[j]),
                "emotions": {EMOTION_CLASSES[k]: float(probs[j][k]) for k in range(len(EMOTION_CLASSES))}
            })
        # free
        del batch_tensor, out, logits
        torch.cuda.empty_cache()
    return results

def create_tmp(video_path):
    base = Path(video_path).stem
    tmp_dir = f"{FRAME_TMP_PREFIX}_{base}"
    # if os.path.exists(tmp_dir):
    #     shutil.rmtree(tmp_dir)
    os.makedirs(tmp_dir, exist_ok=True)

    return tmp_dir

def process_frames_in_batches(fm, indices, model, log, batch_size_load=50, batch_size_process=16):
    """
    Обрабатывает кадры батчами для экономии памяти.
    """
    emo_results = []
    total_batches = (len(indices) - 1) // batch_size_load + 1
    
    # Сначала получаем все эмоции батчами
    for batch_idx in range(0, len(indices), batch_size_load):
        batch_start = batch_idx
        batch_end = min(batch_idx + batch_size_load, len(indices))
        batch_indices = indices[batch_start:batch_end]
        
        current_batch = batch_idx // batch_size_load + 1
        
        log(f"[process_video] Батч {current_batch}/{total_batches}: "
            f"кадры {batch_start}-{batch_end-1}")
        
        # Загружаем текущий батч кадров
        frames_batch = []
        for idx in batch_indices:
            frame_bgr = fm.get(idx)
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            frames_batch.append(frame_rgb)
        
        # Анализируем эмоции в батче
        batch_results = predict_emonet_batch(frames_batch, model, batch_size=batch_size_process)
        emo_results.extend(batch_results)
        
        # НЕМЕДЛЕННАЯ ОЧИСТКА ПАМЯТИ
        del frames_batch
        del batch_results
        torch.cuda.empty_cache()
        gc.collect()
    
    return emo_results
