"""
Модуль для расширенных фичей анализа эмоций:
- Микроэмоции (micro-expressions)
- Физиологические сигналы (стресс, уверенность, нервозность)
- Асимметрия лица для оценки искренности
- Индивидуальность выражения эмоций
"""
import numpy as np
from typing import List, Dict, Any, Tuple, Optional
import math


def detect_micro_expressions(
    emotions: List[Dict[str, Any]],
    fps: float = 30.0,
    min_duration_sec: float = 0.03,
    max_duration_sec: float = 0.5,
    change_threshold: Optional[float] = None,
    min_frames: int = 2
) -> Dict[str, Any]:
    """
    Детектирует микроэмоции - резкие изменения эмоций длительностью 0.03-0.5 секунды.
    
    Args:
        emotions: Список словарей с эмоциями (valence, arousal, emotions)
        fps: Частота кадров
        min_duration_sec: Минимальная длительность микроэмоции (сек)
        max_duration_sec: Максимальная длительность микроэмоции (сек)
        change_threshold: Порог изменения для детекции (0-1). Если None, используется adaptive threshold (85th percentile)
        min_frames: Минимальное количество кадров для микроэмоции (рекомендуется >= 2)
    
    Returns:
        Словарь с информацией о микроэмоциях
    """
    if len(emotions) < 2:
        return {
            "microexpressions_count": 0,
            "microexpressions": [],
            "microexpression_rate": 0.0,
            "avg_duration": 0.0
        }
    
    # Вычисляем изменения между кадрами
    valence_changes = []
    arousal_changes = []
    emotion_changes = []
    
    for i in range(1, len(emotions)):
        prev = emotions[i-1]
        curr = emotions[i]
        
        # Изменение валентности и активации
        v_diff = abs(curr.get('valence', 0) - prev.get('valence', 0))
        a_diff = abs(curr.get('arousal', 0) - prev.get('arousal', 0))
        
        valence_changes.append(v_diff)
        arousal_changes.append(a_diff)
        
        # Изменение доминантной эмоции
        prev_emotions = prev.get('emotions', {})
        curr_emotions = curr.get('emotions', {})
        
        if prev_emotions and curr_emotions:
            prev_dominant = max(prev_emotions.items(), key=lambda x: x[1])[0]
            curr_dominant = max(curr_emotions.items(), key=lambda x: x[1])[0]
            
            # Изменение вероятности доминантной эмоции
            prev_prob = prev_emotions.get(prev_dominant, 0)
            curr_prob = curr_emotions.get(curr_dominant, 0)
            
            emotion_change = abs(curr_prob - prev_prob)
            if prev_dominant != curr_dominant:
                emotion_change += 0.3  # Дополнительный вес при смене эмоции
            
            emotion_changes.append(emotion_change)
        else:
            emotion_changes.append(0.0)
    
    # Комбинированное изменение: sqrt(Δv² + Δa²) + emotion_change
    combined_changes = [
        np.sqrt(v**2 + a**2) + e * 0.5
        for v, a, e in zip(valence_changes, arousal_changes, emotion_changes)
    ]
    
    # Adaptive threshold: use 85th percentile if threshold not provided
    if change_threshold is None:
        if len(combined_changes) > 0:
            change_threshold = float(np.percentile(combined_changes, 85))
        else:
            change_threshold = 0.4
    
    # Детекция резких изменений (микроэмоций)
    # min_frames: require at least 2 frames (at 30fps, 0.03s ≈ 1 frame, which is too short)
    min_duration_frames = max(min_frames, int(min_duration_sec * fps))
    max_duration_frames = min(int(max_duration_sec * fps), 15)  # Cap at 15 frames
    
    microexpressions = []
    in_microexpression = False
    micro_start = None
    
    for i, change in enumerate(combined_changes):
        if change >= change_threshold and not in_microexpression:
            # Начало микроэмоции
            in_microexpression = True
            micro_start = i
        elif in_microexpression:
            # Проверяем, закончилась ли микроэмоция
            duration_frames = i - micro_start + 1
            
            if change < change_threshold * 0.5 or duration_frames > max_duration_frames:
                # Конец микроэмоции
                if min_duration_frames <= duration_frames <= max_duration_frames:
                    # Валидная микроэмоция
                    duration_sec = duration_frames / fps
                    
                    # Определяем тип микроэмоции
                    start_idx = micro_start
                    end_idx = i
                    
                    start_emotion = emotions[start_idx]
                    end_emotion = emotions[min(end_idx, len(emotions)-1)]
                    
                    start_dominant = max(
                        start_emotion.get('emotions', {}).items(), 
                        key=lambda x: x[1]
                    )[0] if start_emotion.get('emotions') else 'Neutral'
                    
                    end_dominant = max(
                        end_emotion.get('emotions', {}).items(), 
                        key=lambda x: x[1]
                    )[0] if end_emotion.get('emotions') else 'Neutral'
                    
                    microexpressions.append({
                        "start_frame": int(micro_start),
                        "end_frame": int(i),
                        "duration_sec": float(duration_sec),
                        "intensity": float(combined_changes[micro_start]),
                        "type": f"{start_dominant}_to_{end_dominant}",
                        "valence_change": float(
                            end_emotion.get('valence', 0) - start_emotion.get('valence', 0)
                        ),
                        "arousal_change": float(
                            end_emotion.get('arousal', 0) - start_emotion.get('arousal', 0)
                        )
                    })
                
                in_microexpression = False
                micro_start = None
    
    # Обрабатываем незавершенную микроэмоцию
    if in_microexpression and micro_start is not None:
        duration_frames = len(combined_changes) - micro_start + 1  # +1 to include start frame
        if min_duration_frames <= duration_frames <= max_duration_frames:
            duration_sec = duration_frames / fps
            start_emotion = emotions[micro_start]
            end_emotion = emotions[-1]
            
            start_dominant = max(
                start_emotion.get('emotions', {}).items(), 
                key=lambda x: x[1]
            )[0] if start_emotion.get('emotions') else 'Neutral'
            
            end_dominant = max(
                end_emotion.get('emotions', {}).items(), 
                key=lambda x: x[1]
            )[0] if end_emotion.get('emotions') else 'Neutral'
            
            microexpressions.append({
                "start_frame": int(micro_start),
                "end_frame": int(len(emotions) - 1),
                "duration_sec": float(duration_sec),
                "intensity": float(combined_changes[micro_start]),
                "type": f"{start_dominant}_to_{end_dominant}",
                "valence_change": float(
                    end_emotion.get('valence', 0) - start_emotion.get('valence', 0)
                ),
                "arousal_change": float(
                    end_emotion.get('arousal', 0) - start_emotion.get('arousal', 0)
                )
            })
    
    # Вычисляем метрики
    total_duration_sec = len(emotions) / fps if fps > 0 else 1.0
    microexpression_rate = len(microexpressions) / total_duration_sec if total_duration_sec > 0 else 0.0
    
    avg_duration = (
        np.mean([m["duration_sec"] for m in microexpressions]) 
        if microexpressions else 0.0
    )
    
    return {
        "microexpressions_count": len(microexpressions),
        "microexpressions": microexpressions,
        "microexpression_rate": float(microexpression_rate),
        "avg_duration": float(avg_duration),
        "total_duration_sec": float(total_duration_sec)
    }


def compute_physiological_signals(
    emotions: List[Dict[str, Any]],
    microexpressions: Optional[Dict[str, Any]] = None,
    fps: float = 30.0
) -> Dict[str, Any]:
    """
    Вычисляет физиологические сигналы: стресс, уверенность, нервозность.
    
    ⚠️ NOTE: These are heuristic-based scores. For production use, consider training
    a learned meta-model (X -> label) with labeled data or weak supervision.
    Current implementation uses rule-based heuristics as initial estimates.
    
    Args:
        emotions: Список словарей с эмоциями
        microexpressions: Результат detect_micro_expressions (опционально)
        fps: Частота кадров
    
    Returns:
        Словарь с физиологическими индексами (heuristic-based, not validated)
    """
    if not emotions:
        return {
            "stress_level_score": 0.0,
            "confidence_face_score": 0.0,
            "tension_face_index": 0.0,
            "nervousness_score": 0.0
        }
    
    # Извлекаем временные ряды
    valence = np.array([e.get('valence', 0) for e in emotions])
    arousal = np.array([e.get('arousal', 0) for e in emotions])
    
    # Получаем вероятности эмоций
    emotion_keys = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt']
    emotion_probs = {}
    for key in emotion_keys:
        emotion_probs[key] = np.array([
            e.get('emotions', {}).get(key, 0) for e in emotions
        ])
    
    # 1. СТРЕСС (stress_level_score)
    # Высокий стресс: высокий arousal + отрицательная валентность + страх/гнев
    stress_indicators = []
    
    # Высокий arousal с отрицательной валентностью
    high_arousal_negative = np.mean(
        (arousal > 0.3) & (valence < 0)
    )
    stress_indicators.append(high_arousal_negative)
    
    # Высокая вероятность страха или гнева
    fear_anger_prob = np.mean(
        emotion_probs.get('Fear', np.zeros(len(emotions))) + 
        emotion_probs.get('Anger', np.zeros(len(emotions)))
    )
    stress_indicators.append(fear_anger_prob)
    
    # Высокая вариативность (нестабильность)
    valence_var = np.var(valence)
    arousal_var = np.var(arousal)
    variability = (valence_var + arousal_var) / 2.0
    stress_indicators.append(min(1.0, variability * 2.0))
    
    # Частота микроэмоций (если доступно)
    if microexpressions:
        micro_rate = microexpressions.get('microexpression_rate', 0.0)
        stress_indicators.append(min(1.0, micro_rate / 2.0))  # Нормализуем
    
    stress_level_score = float(np.mean(stress_indicators))
    
    # 2. УВЕРЕННОСТЬ (confidence_face_score)
    # Высокая уверенность: положительная валентность + умеренный arousal + счастье/нейтральность
    confidence_indicators = []
    
    # Положительная валентность
    positive_valence_ratio = np.mean(valence > 0.2)
    confidence_indicators.append(positive_valence_ratio)
    
    # Умеренный arousal (не слишком высокий, не слишком низкий)
    moderate_arousal_ratio = np.mean(
        (arousal > -0.2) & (arousal < 0.5)
    )
    confidence_indicators.append(moderate_arousal_ratio)
    
    # Высокая вероятность счастья или нейтральности
    happy_neutral_prob = np.mean(
        emotion_probs.get('Happy', np.zeros(len(emotions))) + 
        emotion_probs.get('Neutral', np.zeros(len(emotions)))
    )
    confidence_indicators.append(happy_neutral_prob)
    
    # Низкая вариативность (стабильность)
    stability = 1.0 - min(1.0, variability)
    confidence_indicators.append(stability)
    
    confidence_face_score = float(np.mean(confidence_indicators))
    
    # 3. НАПРЯЖЕНИЕ (tension_face_index)
    # Высокое напряжение: высокий arousal + низкая вариативность + отрицательные эмоции
    tension_indicators = []
    
    # Высокий arousal
    high_arousal_ratio = np.mean(arousal > 0.4)
    tension_indicators.append(high_arousal_ratio)
    
    # Низкая вариативность (зажатость)
    low_variability = 1.0 - min(1.0, variability * 2.0)
    tension_indicators.append(low_variability)
    
    # Отрицательные эмоции
    negative_emotions_prob = np.mean(
        emotion_probs.get('Sad', np.zeros(len(emotions))) + 
        emotion_probs.get('Anger', np.zeros(len(emotions))) +
        emotion_probs.get('Fear', np.zeros(len(emotions)))
    )
    tension_indicators.append(negative_emotions_prob)
    
    tension_face_index = float(np.mean(tension_indicators))
    
    # 4. НЕРВОЗНОСТЬ (nervousness_score)
    # Нервозность: высокая вариативность + высокий arousal + страх/удивление
    nervousness_indicators = []
    
    # Высокая вариативность
    nervousness_indicators.append(min(1.0, variability * 3.0))
    
    # Высокий arousal
    nervousness_indicators.append(high_arousal_ratio)
    
    # Страх или удивление
    fear_surprise_prob = np.mean(
        emotion_probs.get('Fear', np.zeros(len(emotions))) + 
        emotion_probs.get('Surprise', np.zeros(len(emotions)))
    )
    nervousness_indicators.append(fear_surprise_prob)
    
    # Частота микроэмоций
    if microexpressions:
        micro_rate = microexpressions.get('microexpression_rate', 0.0)
        nervousness_indicators.append(min(1.0, micro_rate / 1.5))
    
    nervousness_score = float(np.mean(nervousness_indicators))
    
    return {
        "stress_level_score": float(np.clip(stress_level_score, 0.0, 1.0)),
        "confidence_face_score": float(np.clip(confidence_face_score, 0.0, 1.0)),
        "tension_face_index": float(np.clip(tension_face_index, 0.0, 1.0)),
        "nervousness_score": float(np.clip(nervousness_score, 0.0, 1.0)),
        "indicators_breakdown": {
            "stress_indicators": [float(x) for x in stress_indicators],
            "confidence_indicators": [float(x) for x in confidence_indicators],
            "tension_indicators": [float(x) for x in tension_indicators],
            "nervousness_indicators": [float(x) for x in nervousness_indicators]
        }
    }


def compute_face_asymmetry(
    landmarks: Optional[np.ndarray] = None,
    face_data: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Вычисляет асимметрию лица для оценки искренности и естественности.
    
    Args:
        landmarks: Массив landmarks лица (N, 2) или (N, 3) для 3D
        face_data: Альтернативный источник данных (bbox, landmarks из InsightFace)
    
    Returns:
        Словарь с метриками асимметрии
    """
    if landmarks is None and face_data is None:
        return {
            "asymmetry_score": 0.0,
            "eyebrow_asymmetry": 0.0,
            "mouth_asymmetry": 0.0,
            "eye_asymmetry": 0.0,
            "overall_symmetry": 1.0,
            "sincerity_score": 0.5
        }
    
    # Если передан face_data, пытаемся извлечь landmarks
    if landmarks is None and face_data:
        # Пытаемся получить landmarks из face_data
        if hasattr(face_data, 'landmark_2d_106'):
            landmarks = face_data.landmark_2d_106
        elif hasattr(face_data, 'landmark_3d_68'):
            landmarks = face_data.landmark_3d_68[:, :2]  # Берем только 2D проекцию
        elif isinstance(face_data, dict) and 'landmarks' in face_data:
            landmarks = np.array(face_data['landmarks'])
    
    if landmarks is None:
        return {
            "asymmetry_score": 0.0,
            "eyebrow_asymmetry": 0.0,
            "mouth_asymmetry": 0.0,
            "eye_asymmetry": 0.0,
            "overall_symmetry": 1.0,
            "sincerity_score": 0.5
        }
    
    landmarks = np.array(landmarks)
    
    # Если 3D landmarks, берем только 2D проекцию
    if landmarks.shape[1] == 3:
        landmarks = landmarks[:, :2]
    
    # Находим центр лица (средняя точка между ключевыми точками)
    # Для InsightFace 106 landmarks:
    # - Нос: ~30-35
    # - Левый глаз: ~36-41
    # - Правый глаз: ~42-47
    # - Левая бровь: ~17-21
    # - Правая бровь: ~22-26
    # - Рот: ~48-67
    
    # Упрощенная версия: используем центр масс всех точек
    face_center = np.mean(landmarks, axis=0)
    
    # Разделяем на левую и правую части
    left_mask = landmarks[:, 0] < face_center[0]
    right_mask = landmarks[:, 0] >= face_center[0]
    
    left_points = landmarks[left_mask]
    right_points = landmarks[right_mask]
    
    if len(left_points) == 0 or len(right_points) == 0:
        return {
            "asymmetry_score": 0.0,
            "eyebrow_asymmetry": 0.0,
            "mouth_asymmetry": 0.0,
            "eye_asymmetry": 0.0,
            "overall_symmetry": 1.0,
            "sincerity_score": 0.5
        }
    
    # Общая асимметрия: сравниваем распределение точек
    left_center = np.mean(left_points, axis=0)
    right_center = np.mean(right_points, axis=0)
    
    # Отражаем правую часть относительно центра
    right_reflected = right_center.copy()
    right_reflected[0] = 2 * face_center[0] - right_center[0]
    
    # Расстояние между центрами (нормализованное)
    asymmetry_distance = np.linalg.norm(left_center - right_reflected)
    max_distance = np.max(np.linalg.norm(landmarks - face_center, axis=1))
    asymmetry_score = min(1.0, asymmetry_distance / max_distance if max_distance > 0 else 0.0)
    
    # Асимметрия бровей (упрощенная версия)
    # Для InsightFace: брови примерно в индексах 17-26
    if landmarks.shape[0] >= 68:
        # Стандартные 68 landmarks
        left_eyebrow_indices = list(range(17, 22))
        right_eyebrow_indices = list(range(22, 27))
    else:
        # Адаптивная версия: берем верхние точки слева и справа
        left_eyebrow_indices = np.where(
            (landmarks[:, 0] < face_center[0]) & 
            (landmarks[:, 1] < face_center[1])
        )[0][:5] if np.any((landmarks[:, 0] < face_center[0]) & (landmarks[:, 1] < face_center[1])) else []
        right_eyebrow_indices = np.where(
            (landmarks[:, 0] >= face_center[0]) & 
            (landmarks[:, 1] < face_center[1])
        )[0][:5] if np.any((landmarks[:, 0] >= face_center[0]) & (landmarks[:, 1] < face_center[1])) else []
    
    eyebrow_asymmetry = 0.0
    if len(left_eyebrow_indices) > 0 and len(right_eyebrow_indices) > 0:
        left_eyebrow_y = np.mean(landmarks[left_eyebrow_indices, 1])
        right_eyebrow_y = np.mean(landmarks[right_eyebrow_indices, 1])
        eyebrow_diff = abs(left_eyebrow_y - right_eyebrow_y)
        max_y_diff = np.max(landmarks[:, 1]) - np.min(landmarks[:, 1])
        eyebrow_asymmetry = min(1.0, eyebrow_diff / max_y_diff if max_y_diff > 0 else 0.0)
    
    # Асимметрия рта
    if landmarks.shape[0] >= 68:
        mouth_indices = list(range(48, 68))
    else:
        # Адаптивная версия: нижние точки
        mouth_indices = np.where(landmarks[:, 1] > face_center[1])[0][-10:] if np.any(landmarks[:, 1] > face_center[1]) else []
    
    mouth_asymmetry = 0.0
    if len(mouth_indices) > 0:
        mouth_points = landmarks[mouth_indices]
        mouth_left = mouth_points[mouth_points[:, 0] < face_center[0]]
        mouth_right = mouth_points[mouth_points[:, 0] >= face_center[0]]
        
        if len(mouth_left) > 0 and len(mouth_right) > 0:
            mouth_left_center = np.mean(mouth_left, axis=0)
            mouth_right_center = np.mean(mouth_right, axis=0)
            mouth_reflected = mouth_right_center.copy()
            mouth_reflected[0] = 2 * face_center[0] - mouth_right_center[0]
            mouth_diff = np.linalg.norm(mouth_left_center - mouth_reflected)
            max_mouth_diff = np.max(np.linalg.norm(mouth_points - face_center, axis=1))
            mouth_asymmetry = min(1.0, mouth_diff / max_mouth_diff if max_mouth_diff > 0 else 0.0)
    
    # Асимметрия глаз (аналогично)
    eye_asymmetry = 0.0
    if landmarks.shape[0] >= 68:
        left_eye_indices = list(range(36, 42))
        right_eye_indices = list(range(42, 48))
        
        if len(left_eye_indices) > 0 and len(right_eye_indices) > 0:
            left_eye_y = np.mean(landmarks[left_eye_indices, 1])
            right_eye_y = np.mean(landmarks[right_eye_indices, 1])
            eye_diff = abs(left_eye_y - right_eye_y)
            max_y_diff = np.max(landmarks[:, 1]) - np.min(landmarks[:, 1])
            eye_asymmetry = min(1.0, eye_diff / max_y_diff if max_y_diff > 0 else 0.0)
    
    # Общая симметрия (обратная асимметрии)
    overall_symmetry = 1.0 - asymmetry_score
    
    # Оценка искренности
    # ⚠️ WARNING: sincerity_score is a research/audit-only metric.
    # It should NOT be used in production models without clinical validation and legal review.
    # This is based on pseudoscientific assumptions about facial asymmetry and emotion.
    # Use only for research purposes or as an initial heuristic, not as a final metric.
    ideal_symmetry_range = (0.7, 0.95)  # Идеальный диапазон
    if overall_symmetry < ideal_symmetry_range[0]:
        sincerity_score = overall_symmetry / ideal_symmetry_range[0] * 0.5
    elif overall_symmetry > ideal_symmetry_range[1]:
        sincerity_score = 0.5 + (1.0 - overall_symmetry) / (1.0 - ideal_symmetry_range[1]) * 0.5
    else:
        sincerity_score = 0.5 + (overall_symmetry - ideal_symmetry_range[0]) / (
            ideal_symmetry_range[1] - ideal_symmetry_range[0]
        ) * 0.5
    
    sincerity_score = np.clip(sincerity_score, 0.0, 1.0)
    
    return {
        "asymmetry_score": float(asymmetry_score),
        "eyebrow_asymmetry": float(eyebrow_asymmetry),
        "mouth_asymmetry": float(mouth_asymmetry),
        "eye_asymmetry": float(eye_asymmetry),
        "overall_symmetry": float(overall_symmetry),
        "sincerity_score": float(sincerity_score),  # ⚠️ AUDIT-ONLY / RESEARCH: Not validated for production use
        "_sincerity_warning": "This metric is research-only and should not be used in production without clinical validation"
    }


def compute_emotional_individuality(
    emotions: List[Dict[str, Any]],
    fps: float = 30.0
) -> Dict[str, Any]:
    """
    Анализирует индивидуальность выражения эмоций:
    - Насколько автор "эмоциональный"
    - Стилевые паттерны выражения эмоций
    - Интенсивность выражения
    
    Args:
        emotions: Список словарей с эмоциями
        fps: Частота кадров
    
    Returns:
        Словарь с метриками индивидуальности
    """
    if not emotions:
        return {
            "emotional_intensity_baseline": 0.0,
            "expressivity_index": 0.0,
            "emotional_style_vector": {},
            "emotional_range": 0.0,
            "dominant_style": "neutral"
        }
    
    # Извлекаем данные
    valence = np.array([e.get('valence', 0) for e in emotions])
    arousal = np.array([e.get('arousal', 0) for e in emotions])
    
    # Интенсивность эмоций (длина вектора в пространстве V-A)
    intensity = np.sqrt(valence**2 + arousal**2)
    
    # 1. Базовый уровень интенсивности (emotional_intensity_baseline)
    emotional_intensity_baseline = float(np.mean(intensity))
    
    # 2. Индекс выразительности (expressivity_index)
    # Учитывает: среднюю интенсивность, вариативность, диапазон эмоций
    intensity_mean = np.mean(intensity)
    intensity_std = np.std(intensity)
    intensity_range = np.max(intensity) - np.min(intensity)
    
    # Нормализуем компоненты
    intensity_mean_norm = min(1.0, intensity_mean / 1.414)  # Максимум sqrt(2) для V-A
    intensity_std_norm = min(1.0, intensity_std / 0.5)
    intensity_range_norm = min(1.0, intensity_range / 1.414)
    
    expressivity_index = float(
        (intensity_mean_norm * 0.4 + intensity_std_norm * 0.3 + intensity_range_norm * 0.3)
    )
    
    # 3. Стилевой вектор (emotional_style_vector)
    # Анализируем паттерны: как человек выражает разные эмоции
    
    emotion_keys = ['Neutral', 'Happy', 'Sad', 'Surprise', 'Fear', 'Disgust', 'Anger', 'Contempt']
    emotion_probs = {}
    for key in emotion_keys:
        probs = [e.get('emotions', {}).get(key, 0) for e in emotions]
        emotion_probs[key] = np.array(probs)
    
    # Средние вероятности каждой эмоции
    emotion_means = {key: float(np.mean(probs)) for key, probs in emotion_probs.items()}
    
    # Вариативность каждой эмоции
    emotion_stds = {key: float(np.std(probs)) for key, probs in emotion_probs.items()}
    
    # Максимальная вероятность каждой эмоции
    emotion_maxs = {key: float(np.max(probs)) for key, probs in emotion_probs.items()}
    
    emotional_style_vector = {
        "emotion_means": emotion_means,
        "emotion_stds": emotion_stds,
        "emotion_maxs": emotion_maxs,
        "valence_mean": float(np.mean(valence)),
        "valence_std": float(np.std(valence)),
        "arousal_mean": float(np.mean(arousal)),
        "arousal_std": float(np.std(arousal))
    }
    
    # 4. Эмоциональный диапазон (emotional_range)
    # Насколько широк диапазон выражаемых эмоций
    active_emotions = sum(1 for mean in emotion_means.values() if mean > 0.1)
    max_emotion_prob = max(emotion_means.values())
    emotion_entropy = -sum(
        p * math.log2(p) if p > 0 else 0 
        for p in emotion_means.values()
    )
    max_entropy = math.log2(len(emotion_keys))
    normalized_entropy = emotion_entropy / max_entropy if max_entropy > 0 else 0
    
    emotional_range = float(
        (active_emotions / len(emotion_keys)) * 0.5 + 
        normalized_entropy * 0.5
    )
    
    # 5. Доминантный стиль
    # Определяем, какой стиль выражения преобладает
    if emotional_intensity_baseline < 0.3:
        dominant_style = "reserved"
    elif emotional_intensity_baseline < 0.6:
        dominant_style = "moderate"
    elif emotional_intensity_baseline < 0.8:
        dominant_style = "expressive"
    else:
        dominant_style = "highly_expressive"
    
    # Дополнительная характеристика по вариативности
    if intensity_std < 0.1:
        dominant_style += "_stable"
    elif intensity_std > 0.3:
        dominant_style += "_variable"
    
    return {
        "emotional_intensity_baseline": float(emotional_intensity_baseline),
        "expressivity_index": float(expressivity_index),
        "emotional_style_vector": emotional_style_vector,
        "emotional_range": float(emotional_range),
        "dominant_style": dominant_style,
        "intensity_stats": {
            "mean": float(intensity_mean),
            "std": float(intensity_std),
            "min": float(np.min(intensity)),
            "max": float(np.max(intensity)),
            "range": float(intensity_range)
        },
        "active_emotions_count": int(active_emotions),
        "emotion_entropy": float(emotion_entropy),
        "normalized_entropy": float(normalized_entropy)
    }

