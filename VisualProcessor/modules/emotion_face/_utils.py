import os
import gc
import cv2
import math
import psutil
import torch
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from pathlib import Path
from typing import List, Optional

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
FRAME_TMP_PREFIX = "tmp_frames"

from utils.logger import get_logger
logger = get_logger("VideoEmotionProcessor")

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

def detect_keyframes(emotion_curve, EMOTION_CLASSES, threshold=0.3, smooth_window=5, 
                    prominence=0.1, min_distance=8):
    """
    Находит ключевые кадры с предварительным сглаживанием и использованием scipy.signal.find_peaks.
    
    Args:
        emotion_curve: dict with 'valence', 'arousal', 'intensity', etc.
        EMOTION_CLASSES: dict mapping emotion indices to names
        threshold: minimum change threshold for transitions
        smooth_window: window size for Gaussian smoothing
        prominence: minimum prominence for peak detection (normalized scale, default 0.1)
        min_distance: minimum distance between peaks in frames (default 8-12 frames)
    """
    from scipy import signal
    from scipy.ndimage import gaussian_filter1d
    
    # Gaussian smoothing instead of simple moving average (sigma = 1-3 frames)
    sigma = min(3.0, smooth_window / 3.0)
    valence_smooth = gaussian_filter1d(
        np.array(emotion_curve['valence']), 
        sigma=sigma, 
        mode='nearest'
    )
    arousal_smooth = gaussian_filter1d(
        np.array(emotion_curve['arousal']), 
        sigma=sigma, 
        mode='nearest'
    )
    
    # Compute intensity = sqrt(valence² + arousal²)
    intensity = np.sqrt(valence_smooth**2 + arousal_smooth**2)
    
    # Normalize intensity to [0, 1] for prominence calculation
    intensity_min = np.min(intensity)
    intensity_max = np.max(intensity)
    if intensity_max > intensity_min:
        intensity_norm = (intensity - intensity_min) / (intensity_max - intensity_min)
    else:
        intensity_norm = intensity
    
    # Detect peaks in intensity using scipy.signal.find_peaks
    peaks, peak_properties = signal.find_peaks(
        intensity_norm,
        prominence=prominence,
        distance=max(1, min_distance)
    )
    
    keyframes = {}
    
    # Add emotion peaks
    for peak_idx in peaks:
        if peak_idx < len(intensity):
            keyframes[peak_idx] = {
                'type': 'emotion_peak',
                'score': float(intensity_norm[peak_idx]),
                'intensity': float(intensity[peak_idx]),
                'valence': float(valence_smooth[peak_idx]),
                'arousal': float(arousal_smooth[peak_idx])
            }
    
    # Detect transitions: significant changes in valence/arousal
    valence_diff = np.abs(np.diff(valence_smooth))
    arousal_diff = np.abs(np.diff(arousal_smooth))
    combined_diff = np.sqrt(valence_diff**2 + arousal_diff**2)
    
    # Normalize combined_diff for prominence
    if len(combined_diff) > 0:
        diff_min = np.min(combined_diff)
        diff_max = np.max(combined_diff)
        if diff_max > diff_min:
            combined_diff_norm = (combined_diff - diff_min) / (diff_max - diff_min)
        else:
            combined_diff_norm = combined_diff
        
        # Find peaks in change signal (transitions)
        transition_peaks, _ = signal.find_peaks(
            combined_diff_norm,
            prominence=prominence,
            distance=max(1, min_distance)
        )
        
        # Add transitions (shift by 1 since diff reduces length by 1)
        for trans_idx in transition_peaks:
            frame_idx = trans_idx + 1  # diff shifts indices
            if frame_idx < len(valence_smooth):
                # Only add if not already a peak, or if transition score is higher
                if frame_idx not in keyframes or combined_diff_norm[trans_idx] > keyframes[frame_idx].get('score', 0):
                    keyframes[frame_idx] = {
                        'type': 'transition',
                        'score': float(combined_diff_norm[trans_idx]),
                        'valence_change': float(valence_diff[trans_idx]),
                        'arousal_change': float(arousal_diff[trans_idx]),
                        'valence': float(valence_smooth[frame_idx]),
                        'arousal': float(arousal_smooth[frame_idx])
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

    logger.info(f"[VALIDATION QUALITY] overall_score: {overall_score} |>=| quality_threshold: {quality_threshold} | diversity_score: {diversity_score} |>=| diversity_threshold: {diversity_threshold}")
    
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

def save_for_user(data, output, output_dir='user_results'):
    """
    Сохраняет детализированные результаты для пользователя.
    """
    import json
    import os
    from datetime import datetime
    
    os.makedirs(f"{output}/{output_dir}", exist_ok=True)
    
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(output, output_dir, f"{timestamp}_analysis.json")
    
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
            'output': str(output),  # Преобразуем Path в строку
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
    
    logger.info(f"[INFO] User analysis saved to: {output_file}")
    return output_file

def save_for_model(data, output, output_dir='model_data'):
    """
    Сохраняет нормализованные данные для обучения модели.
    """
    import numpy as np
    import json
    import os
    from datetime import datetime
    
    os.makedirs(f"{output}/{output_dir}", exist_ok=True)
    
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
    npy_file = os.path.join(output, output_dir, f"{timestamp}_emotions.npy")
    np.save(npy_file, np_array)
    
    # Сохраняем метаданные
    metadata = prepare_for_json({
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
    
    meta_file = os.path.join(output, output_dir, f"{timestamp}_meta.json")
    with open(meta_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
    logger.info(f"[INFO] Model data saved: {npy_file}")
    logger.info(f"[INFO] Metadata saved: {meta_file}")
    
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

def analyze_emotion_profile(emo_results, use_weighted_means=True):
    """
    Анализирует, какие эмоции преобладают.
    
    Args:
        emo_results: список результатов эмоций
        use_weighted_means: использовать weighted means по confidence (рекомендуется True)
    """
    emotion_totals = {}
    valence_sum = 0
    arousal_sum = 0
    confidence_sum = 0
    
    for result in emo_results:
        # Get confidence (emotion_confidence or face_confidence, fallback to 1.0)
        conf = result.get('emotion_confidence', result.get('face_confidence', 1.0))
        if not use_weighted_means:
            conf = 1.0  # Unweighted
        
        valence_sum += result.get('valence', 0) * conf
        arousal_sum += result.get('arousal', 0) * conf
        confidence_sum += conf
        
        emotions = result.get('emotions', {})
        if emotions:
            dominant = max(emotions.items(), key=lambda x: x[1])[0]
            # Weight by confidence
            emotion_totals[dominant] = emotion_totals.get(dominant, 0) + conf
    
    if confidence_sum > 0:
        valence_avg = valence_sum / confidence_sum
        arousal_avg = arousal_sum / confidence_sum
    else:
        valence_avg = 0
        arousal_avg = 0
    
    # Определяем доминантную эмоцию
    dominant_emotion = None
    if emotion_totals:
        dominant_emotion = max(emotion_totals.items(), key=lambda x: x[1])[0]
    
    total_weighted_frames = sum(emotion_totals.values())
    neutral_percentage = emotion_totals.get('Neutral', 0) / total_weighted_frames if total_weighted_frames > 0 else 0
    
    return {
        'dominant_emotion': dominant_emotion,
        'emotion_distribution': emotion_totals,
        'valence_avg': valence_avg,
        'arousal_avg': arousal_avg,
        'valence_std': float(np.std([r.get('valence', 0) for r in emo_results])),
        'arousal_std': float(np.std([r.get('arousal', 0) for r in emo_results])),
        'is_neutral_dominant': dominant_emotion == 'Neutral',
        'neutral_percentage': neutral_percentage
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
        print(f"_utils | compute_steps | Ошибка расчета target_scans: {e}")
        target_scans = min(MAX_SCANS, total_frames // 10)
    
    try:
        # FIX 2: scan_stride должен быть целым числом
        scan_stride = max(1, total_frames // target_scans)
        scan_stride = int(scan_stride)
    except Exception as e:
        print(f"_utils | compute_steps | Ошибка расчета scan_stride: {e}")
        scan_stride = max(1, total_frames // 100)
    
    return scan_stride, target_scans 


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

def predict_emonet_batch(frames: List[np.ndarray], model, batch_size: Optional[int] = None, use_amp: bool = True, 
                         temperature: float = 1.0, face_confidence: Optional[List[float]] = None):
    """
    frames: list of RGB ndarrays (H,W,3)
    returns list of dicts {valence, arousal, emotions: {label:prob,...}, emotion_confidence, is_valid}
    
    Args:
        temperature: Temperature scaling factor for calibration (default 1.0, can be tuned on validation)
        face_confidence: Optional list of face detection confidence scores (0-1) for each frame
    """
    if batch_size is None:
        batch_size = choose_batch_size_by_vram()
    results = []
    model_device = DEVICE
    
    if face_confidence is None:
        face_confidence = [1.0] * len(frames)

    # move model already loaded to DEVICE
    for i in range(0, len(frames), batch_size):
        chunk = frames[i:i + batch_size]
        chunk_confidence = face_confidence[i:i + batch_size]
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
                return predict_emonet_batch(frames, model, batch_size=max(1, batch_size // 2), 
                                          use_amp=use_amp, temperature=temperature, face_confidence=face_confidence)
            else:
                raise e

        vals = out["valence"].detach().cpu().numpy()
        arous = out["arousal"].detach().cpu().numpy()
        logits = out["expression"].detach()
        
        # Temperature scaling for calibration
        if temperature != 1.0:
            logits = logits / temperature
        
        probs = F.softmax(logits, dim=1).cpu().numpy()

        for j in range(len(chunk)):
            # Compute emotion confidence: max softmax probability * face detection confidence
            max_prob = float(np.max(probs[j]))
            detection_conf = chunk_confidence[j] if j < len(chunk_confidence) else 1.0
            emotion_confidence = max_prob * detection_conf
            
            # Mark as invalid if face confidence is too low
            is_valid = detection_conf >= 0.3  # Threshold for valid face detection
            
            results.append({
                "valence": float(vals[j]),
                "arousal": float(arous[j]),
                "emotions": {EMOTION_CLASSES[k]: float(probs[j][k]) for k in range(len(EMOTION_CLASSES))},
                "emotion_confidence": float(emotion_confidence),
                "face_confidence": float(detection_conf),
                "is_valid": bool(is_valid)
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
