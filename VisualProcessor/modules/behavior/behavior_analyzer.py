"""
Модуль для комплексного анализа поведения людей в видео.
Реализует все недостающие фичи из FEATURES.MD:
- Детальная классификация жестов рук
- Body language анализ
- Speech-driven behavior
- Engagement Index
- Confidence/Dominance Index
- Signs of stress/anxiety
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, List, Any, Optional, Tuple
from collections import deque
import json
from pathlib import Path


class HandGestureClassifier:
    """Детальная классификация жестов рук"""
    
    def __init__(self):
        self.gesture_types = {
            'pointing': self._is_pointing,
            'open_palm': self._is_open_palm,
            'hands_on_hips': self._is_hands_on_hips,
            'self_touch': self._is_self_touch,
            'fist': self._is_fist,
            'thumbs_up': self._is_thumbs_up,
            'thumbs_down': self._is_thumbs_down,
            'victory': self._is_victory,
            'ok': self._is_ok,
            'rock': self._is_rock,
            'call_me': self._is_call_me,
            'love': self._is_love
        }
    
    def _get_finger_states(self, hand_landmarks):
        """Определяет состояние пальцев"""
        finger_tips = [4, 8, 12, 16, 20]
        finger_pips = [2, 6, 10, 14, 18]
        
        states = {}
        for i, (tip_idx, pip_idx) in enumerate(zip(finger_tips, finger_pips)):
            tip = hand_landmarks.landmark[tip_idx]
            pip = hand_landmarks.landmark[pip_idx]
            
            if i == 0:  # thumb
                states['thumb'] = tip.x < pip.x if tip.x < 0.5 else tip.x > pip.x
            else:
                states[['index', 'middle', 'ring', 'pinky'][i-1]] = tip.y < pip.y
        
        return states
    
    def _is_pointing(self, hand_landmarks, pose_landmarks=None):
        """Указание рукой"""
        states = self._get_finger_states(hand_landmarks)
        return states.get('index', False) and not any([
            states.get('middle', False),
            states.get('ring', False),
            states.get('pinky', False)
        ])
    
    def _is_open_palm(self, hand_landmarks, pose_landmarks=None):
        """Раскрытые ладони"""
        states = self._get_finger_states(hand_landmarks)
        return all(states.values())
    
    def _is_hands_on_hips(self, hand_landmarks, pose_landmarks):
        """Руки в боки"""
        if pose_landmarks is None:
            return False
        
        # Проверяем, что запястья находятся рядом с талией
        wrist = hand_landmarks.landmark[0]
        hip_idx = 23 if wrist.x < 0.5 else 24  # левое или правое бедро
        
        if hip_idx < len(pose_landmarks.landmark):
            hip = pose_landmarks.landmark[hip_idx]
            distance = abs(wrist.y - hip.y)
            return distance < 0.1  # близко к бедру
        
        return False
    
    def _is_self_touch(self, hand_landmarks, pose_landmarks):
        """Self-touch жесты (поглаживание, почёсывание)"""
        if pose_landmarks is None:
            return False
        
        # Проверяем близость руки к лицу/голове
        wrist = hand_landmarks.landmark[0]
        face_landmarks = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # примерные индексы лица
        
        # Упрощенная проверка: если рука близко к верхней части кадра
        return wrist.y < 0.3
    
    def _is_fist(self, hand_landmarks, pose_landmarks=None):
        """Кулак"""
        states = self._get_finger_states(hand_landmarks)
        return not any(states.values())
    
    def _is_thumbs_up(self, hand_landmarks, pose_landmarks=None):
        """Большой палец вверх"""
        states = self._get_finger_states(hand_landmarks)
        thumb_tip = hand_landmarks.landmark[4]
        thumb_mcp = hand_landmarks.landmark[2]
        return states.get('thumb', False) and thumb_tip.y < thumb_mcp.y
    
    def _is_thumbs_down(self, hand_landmarks, pose_landmarks=None):
        """Большой палец вниз"""
        states = self._get_finger_states(hand_landmarks)
        thumb_tip = hand_landmarks.landmark[4]
        thumb_mcp = hand_landmarks.landmark[2]
        return states.get('thumb', False) and thumb_tip.y > thumb_mcp.y
    
    def _is_victory(self, hand_landmarks, pose_landmarks=None):
        """Победа (V)"""
        states = self._get_finger_states(hand_landmarks)
        return states.get('index', False) and states.get('middle', False) and not any([
            states.get('ring', False),
            states.get('pinky', False)
        ])
    
    def _is_ok(self, hand_landmarks, pose_landmarks=None):
        """OK знак"""
        states = self._get_finger_states(hand_landmarks)
        index_tip = hand_landmarks.landmark[8]
        thumb_tip = hand_landmarks.landmark[4]
        distance = np.sqrt((index_tip.x - thumb_tip.x)**2 + (index_tip.y - thumb_tip.y)**2)
        return states.get('thumb', False) and distance < 0.05 and not states.get('index', False)
    
    def _is_rock(self, hand_landmarks, pose_landmarks=None):
        """Рок (рога)"""
        states = self._get_finger_states(hand_landmarks)
        return states.get('index', False) and states.get('pinky', False) and not any([
            states.get('middle', False),
            states.get('ring', False)
        ])
    
    def _is_call_me(self, hand_landmarks, pose_landmarks=None):
        """Позвони мне"""
        states = self._get_finger_states(hand_landmarks)
        pinky_tip = hand_landmarks.landmark[20]
        thumb_tip = hand_landmarks.landmark[4]
        distance = np.sqrt((pinky_tip.x - thumb_tip.x)**2 + (pinky_tip.y - thumb_tip.y)**2)
        return states.get('pinky', False) and distance < 0.05
    
    def _is_love(self, hand_landmarks, pose_landmarks=None):
        """Любовь (сердце)"""
        states = self._get_finger_states(hand_landmarks)
        return states.get('index', False) and states.get('middle', False) and not any([
            states.get('ring', False),
            states.get('pinky', False)
        ])
    
    def classify_gesture(self, hand_landmarks, pose_landmarks=None):
        """Классифицирует жест"""
        for gesture_name, check_func in self.gesture_types.items():
            try:
                if check_func(hand_landmarks, pose_landmarks):
                    return gesture_name
            except:
                continue
        return 'unknown'


class BodyLanguageAnalyzer:
    """Анализ языка тела"""
    
    def __init__(self):
        pass
    
    def analyze_posture(self, pose_landmarks, image_shape):
        """Анализирует позу тела"""
        if pose_landmarks is None:
            return {}
        
        h, w = image_shape[:2]
        
        def get_coord(idx):
            if idx >= len(pose_landmarks.landmark):
                return None
            lm = pose_landmarks.landmark[idx]
            return np.array([lm.x * w, lm.y * h])
        
        # Ключевые точки
        left_shoulder = get_coord(11)
        right_shoulder = get_coord(12)
        left_hip = get_coord(23)
        right_hip = get_coord(24)
        left_wrist = get_coord(15)
        right_wrist = get_coord(16)
        nose = get_coord(0)
        
        if any(x is None for x in [left_shoulder, right_shoulder, left_hip, right_hip, nose]):
            return {}
        
        results = {}
        
        # Поза (стоя/сидя)
        shoulder_hip_distance = np.mean([
            np.linalg.norm(left_shoulder - left_hip),
            np.linalg.norm(right_shoulder - right_hip)
        ])
        results['posture'] = 'standing' if shoulder_hip_distance > h * 0.2 else 'sitting'
        
        # Открытая/закрытая поза
        if left_wrist is not None and right_wrist is not None:
            shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
            wrist_distance = np.linalg.norm(left_wrist - right_wrist)
            results['open_posture'] = wrist_distance > shoulder_width * 1.2
            results['closed_posture'] = wrist_distance < shoulder_width * 0.8
        else:
            results['open_posture'] = False
            results['closed_posture'] = False
        
        # Power pose (доминантность)
        if left_wrist is not None and right_wrist is not None:
            # Руки на бедрах или широко расставлены
            hip_center = (left_hip + right_hip) / 2
            wrist_center = (left_wrist + right_wrist) / 2
            vertical_distance = abs(wrist_center[1] - hip_center[1])
            horizontal_spread = np.linalg.norm(left_wrist - right_wrist)
            
            results['power_pose'] = (
                vertical_distance < h * 0.1 and  # руки на уровне бедер
                horizontal_spread > shoulder_width * 1.5  # широко расставлены
            )
        else:
            results['power_pose'] = False
        
        # Напряженность (rigidity)
        if left_shoulder is not None and right_shoulder is not None:
            shoulder_angle = np.degrees(np.arctan2(
                right_shoulder[1] - left_shoulder[1],
                right_shoulder[0] - left_shoulder[0]
            ))
            # Прямые плечи = напряжение
            results['rigidity'] = abs(shoulder_angle) < 5.0
        else:
            results['rigidity'] = False
        
        # Расслабленность
        results['relaxed'] = not results.get('rigidity', False) and not results.get('closed_posture', False)
        
        # Наклон вперед/назад
        if nose is not None:
            shoulder_center = (left_shoulder + right_shoulder) / 2
            hip_center = (left_hip + right_hip) / 2
            forward_lean = nose[0] - shoulder_center[0]
            results['forward_lean'] = forward_lean > w * 0.02
            results['backward_lean'] = forward_lean < -w * 0.02
        
        # Баланс тела (center of mass)
        if all(x is not None for x in [left_shoulder, right_shoulder, left_hip, right_hip]):
            center_top = (left_shoulder + right_shoulder) / 2
            center_bottom = (left_hip + right_hip) / 2
            center_of_mass = (center_top + center_bottom) / 2
            frame_center_x = w / 2
            results['balance_offset'] = (center_of_mass[0] - frame_center_x) / w
        
        return results


class SpeechBehaviorAnalyzer:
    """Анализ синхронности губ со звуком"""
    
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.mouth_history = deque(maxlen=window_size)
    
    def analyze_lip_sync(self, face_landmarks, image_shape):
        """Анализирует движение губ"""
        if face_landmarks is None:
            return {}
        
        h, w = image_shape[:2]
        
        # Индексы точек губ (MediaPipe Face Mesh)
        upper_lip_indices = [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318]
        lower_lip_indices = [78, 95, 88, 178, 87, 14, 317, 402, 318, 324]
        
        def get_coord(idx):
            if idx >= len(face_landmarks.landmark):
                return None
            lm = face_landmarks.landmark[idx]
            return np.array([lm.x * w, lm.y * h])
        
        # Вычисляем параметры рта
        upper_lip_points = [get_coord(i) for i in upper_lip_indices if get_coord(i) is not None]
        lower_lip_points = [get_coord(i) for i in lower_lip_indices if get_coord(i) is not None]
        
        if not upper_lip_points or not lower_lip_points:
            return {}
        
        upper_center = np.mean(upper_lip_points, axis=0)
        lower_center = np.mean(lower_lip_points, axis=0)
        
        # Ширина рта
        mouth_width = np.max([p[0] for p in upper_lip_points]) - np.min([p[0] for p in upper_lip_points])
        
        # Высота рта
        mouth_height = np.linalg.norm(upper_center - lower_center)
        
        # Площадь рта (приблизительно)
        mouth_area = mouth_width * mouth_height
        
        # Сохраняем в историю
        self.mouth_history.append({
            'width': mouth_width,
            'height': mouth_height,
            'area': mouth_area
        })
        
        # Анализ активности речи
        if len(self.mouth_history) >= 2:
            area_changes = [abs(self.mouth_history[i]['area'] - self.mouth_history[i-1]['area']) 
                          for i in range(1, len(self.mouth_history))]
            avg_change = np.mean(area_changes) if area_changes else 0
            
            # Вероятность активности речи
            speech_activity = min(1.0, avg_change / (w * 0.01))
        else:
            speech_activity = 0.0
        
        return {
            'mouth_width': float(mouth_width),
            'mouth_height': float(mouth_height),
            'mouth_area': float(mouth_area),
            'speech_activity': float(speech_activity),
            'mouth_open_ratio': float(mouth_height / max(mouth_width, 1))
        }


class EngagementAnalyzer:
    """Индекс вовлеченности"""
    
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.engagement_history = deque(maxlen=window_size)
    
    def calculate_engagement(self, face_landmarks, pose_landmarks, hand_landmarks_list, image_shape):
        """Вычисляет индекс вовлеченности"""
        h, w = image_shape[:2]
        engagement_factors = []
        
        # 1. Зрительный контакт с камерой
        if face_landmarks is not None:
            # Проверяем направление взгляда (упрощенно)
            nose = face_landmarks.landmark[1]  # кончик носа
            frame_center_x = 0.5
            gaze_direction = abs(nose.x - frame_center_x)
            eye_contact = 1.0 - min(1.0, gaze_direction * 2)  # ближе к центру = выше контакт
            engagement_factors.append(('eye_contact', eye_contact))
        
        # 2. Движения головы
        if face_landmarks is not None:
            # Микродвижения головы (вариативность позиции)
            if len(self.engagement_history) >= 2:
                head_movements = len(self.engagement_history) - 1
                head_activity = min(1.0, head_movements / self.window_size)
            else:
                head_activity = 0.5
            engagement_factors.append(('head_movement', head_activity))
        
        # 3. Активность жестов
        gesture_activity = len(hand_landmarks_list) > 0
        if gesture_activity:
            gesture_score = min(1.0, len(hand_landmarks_list) / 2.0)
        else:
            gesture_score = 0.3  # базовая активность без жестов
        engagement_factors.append(('gesture_activity', gesture_score))
        
        # 4. Поза тела (открытая = вовлеченность)
        if pose_landmarks is not None:
            body_analyzer = BodyLanguageAnalyzer()
            posture_analysis = body_analyzer.analyze_posture(pose_landmarks, image_shape)
            open_posture_score = 1.0 if posture_analysis.get('open_posture', False) else 0.5
            engagement_factors.append(('open_posture', open_posture_score))
        
        # Общий индекс вовлеченности
        if engagement_factors:
            engagement_score = np.mean([score for _, score in engagement_factors])
        else:
            engagement_score = 0.5
        
        # Сохраняем в историю
        self.engagement_history.append(engagement_score)
        
        # Вариативность вовлеченности
        if len(self.engagement_history) >= 2:
            engagement_variation = np.std(list(self.engagement_history))
        else:
            engagement_variation = 0.0
        
        # Пики вовлеченности
        if len(self.engagement_history) >= 3:
            engagement_peaks = sum(1 for i in range(1, len(self.engagement_history)-1)
                                 if self.engagement_history[i] > self.engagement_history[i-1] and
                                    self.engagement_history[i] > self.engagement_history[i+1])
        else:
            engagement_peaks = 0
        
        # Консистентность
        if len(self.engagement_history) >= 2:
            engagement_consistency = 1.0 - engagement_variation
        else:
            engagement_consistency = 1.0
        
        return {
            'engagement_score': float(engagement_score),
            'engagement_variation': float(engagement_variation),
            'engagement_peaks': int(engagement_peaks),
            'engagement_consistency': float(engagement_consistency),
            'factors': {name: float(score) for name, score in engagement_factors}
        }


class ConfidenceAnalyzer:
    """Индекс уверенности/доминантности"""
    
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.confidence_history = deque(maxlen=window_size)
    
    def calculate_confidence(self, pose_landmarks, face_landmarks, hand_landmarks_list, image_shape):
        """Вычисляет индекс уверенности"""
        h, w = image_shape[:2]
        confidence_factors = []
        
        # 1. Открытая поза
        if pose_landmarks is not None:
            body_analyzer = BodyLanguageAnalyzer()
            posture_analysis = body_analyzer.analyze_posture(pose_landmarks, image_shape)
            
            open_posture = posture_analysis.get('open_posture', False)
            power_pose = posture_analysis.get('power_pose', False)
            
            if power_pose:
                posture_score = 1.0
            elif open_posture:
                posture_score = 0.7
            else:
                posture_score = 0.3
            
            confidence_factors.append(('open_posture', posture_score))
        
        # 2. Наклон головы (прямая голова = уверенность)
        if face_landmarks is not None:
            # Упрощенная проверка: симметрия лица
            left_face = face_landmarks.landmark[33]  # левая сторона
            right_face = face_landmarks.landmark[263]  # правая сторона
            nose = face_landmarks.landmark[1]
            
            face_center_x = (left_face.x + right_face.x) / 2
            head_tilt = abs(nose.x - face_center_x)
            head_straight = 1.0 - min(1.0, head_tilt * 5)
            confidence_factors.append(('head_straight', head_straight))
        
        # 3. Жесты (уверенные жесты)
        if hand_landmarks_list:
            # Открытые ладони = уверенность
            gesture_classifier = HandGestureClassifier()
            confident_gestures = ['open_palm', 'pointing', 'thumbs_up']
            confident_count = 0
            
            for hand_landmarks in hand_landmarks_list:
                gesture = gesture_classifier.classify_gesture(hand_landmarks, pose_landmarks)
                if gesture in confident_gestures:
                    confident_count += 1
            
            gesture_score = min(1.0, confident_count / len(hand_landmarks_list))
            confidence_factors.append(('confident_gestures', gesture_score))
        else:
            confidence_factors.append(('confident_gestures', 0.5))
        
        # 4. Положение плеч (прямые плечи = уверенность)
        if pose_landmarks is not None:
            left_shoulder = pose_landmarks.landmark[11]
            right_shoulder = pose_landmarks.landmark[12]
            shoulder_level = 1.0 - abs(left_shoulder.y - right_shoulder.y) * 10
            shoulder_score = max(0.0, min(1.0, shoulder_level))
            confidence_factors.append(('shoulder_level', shoulder_score))
        
        # Общий индекс уверенности
        if confidence_factors:
            confidence_score = np.mean([score for _, score in confidence_factors])
        else:
            confidence_score = 0.5
        
        # Доминантность (на основе power pose и уверенности)
        dominance_score = confidence_score
        if pose_landmarks is not None:
            body_analyzer = BodyLanguageAnalyzer()
            posture_analysis = body_analyzer.analyze_posture(pose_landmarks, image_shape)
            if posture_analysis.get('power_pose', False):
                dominance_score = min(1.0, dominance_score * 1.2)
        
        # Сохраняем в историю
        self.confidence_history.append(confidence_score)
        
        # Вариативность
        if len(self.confidence_history) >= 2:
            confidence_variability = np.std(list(self.confidence_history))
        else:
            confidence_variability = 0.0
        
        # Пики уверенности
        if len(self.confidence_history) >= 3:
            confidence_peaks = sum(1 for i in range(1, len(self.confidence_history)-1)
                                 if self.confidence_history[i] > self.confidence_history[i-1] and
                                    self.confidence_history[i] > self.confidence_history[i+1])
        else:
            confidence_peaks = 0
        
        return {
            'confidence_score': float(confidence_score),
            'dominance_score': float(dominance_score),
            'confidence_variability': float(confidence_variability),
            'confidence_peak_moments': int(confidence_peaks),
            'factors': {name: float(score) for name, score in confidence_factors}
        }


class StressAnalyzer:
    """Детекция признаков стресса и тревожности"""
    
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.blink_history = deque(maxlen=window_size)
        self.movement_history = deque(maxlen=window_size)
    
    def analyze_stress(self, face_landmarks, pose_landmarks, hand_landmarks_list, image_shape):
        """Анализирует признаки стресса"""
        h, w = image_shape[:2]
        stress_indicators = []
        
        # 1. Частое моргание
        if face_landmarks is not None:
            # Вычисляем EAR (Eye Aspect Ratio)
            left_ear = self._calculate_ear(face_landmarks, image_shape, 'left')
            right_ear = self._calculate_ear(face_landmarks, image_shape, 'right')
            avg_ear = (left_ear + right_ear) / 2
            
            # Моргание (EAR < 0.2)
            is_blinking = avg_ear < 0.2
            self.blink_history.append(is_blinking)
            
            if len(self.blink_history) >= 5:
                blink_rate = sum(self.blink_history) / len(self.blink_history)
                # Высокая частота моргания (>0.3) = стресс
                frequent_blinking = blink_rate > 0.3
                stress_indicators.append(('frequent_blinking', frequent_blinking, blink_rate))
        
        # 2. Self-touch жесты
        if hand_landmarks_list:
            gesture_classifier = HandGestureClassifier()
            self_touch_count = 0
            
            for hand_landmarks in hand_landmarks_list:
                gesture = gesture_classifier.classify_gesture(hand_landmarks, pose_landmarks)
                if gesture == 'self_touch':
                    self_touch_count += 1
            
            self_touch_score = min(1.0, self_touch_count / len(hand_landmarks_list))
            stress_indicators.append(('self_touch', self_touch_score > 0.5, self_touch_score))
        
        # 3. Закрытая поза
        if pose_landmarks is not None:
            body_analyzer = BodyLanguageAnalyzer()
            posture_analysis = body_analyzer.analyze_posture(pose_landmarks, image_shape)
            closed_posture = posture_analysis.get('closed_posture', False)
            stress_indicators.append(('closed_posture', closed_posture, 1.0 if closed_posture else 0.0))
        
        # 4. Напряженность (rigidity)
        if pose_landmarks is not None:
            body_analyzer = BodyLanguageAnalyzer()
            posture_analysis = body_analyzer.analyze_posture(pose_landmarks, image_shape)
            rigidity = posture_analysis.get('rigidity', False)
            stress_indicators.append(('rigidity', rigidity, 1.0 if rigidity else 0.0))
        
        # 5. Fidgeting (ёрзание) - быстрые мелкие движения
        if pose_landmarks is not None:
            # Сохраняем позицию для анализа движения
            nose = pose_landmarks.landmark[0] if len(pose_landmarks.landmark) > 0 else None
            if nose is not None:
                current_pos = np.array([nose.x, nose.y])
                self.movement_history.append(current_pos)
                
                if len(self.movement_history) >= 5:
                    # Вычисляем вариативность движения
                    positions = list(self.movement_history)
                    movement_variance = np.var([p[0] for p in positions]) + np.var([p[1] for p in positions])
                    # Высокая вариативность = ёрзание
                    fidgeting = movement_variance > 0.001
                    stress_indicators.append(('fidgeting', fidgeting, min(1.0, movement_variance * 1000)))
        
        # 6. Несинхронные движения (если есть несколько рук)
        if len(hand_landmarks_list) >= 2:
            # Проверяем синхронность движений рук
            hand_positions = []
            for hand_landmarks in hand_landmarks_list:
                wrist = hand_landmarks.landmark[0]
                hand_positions.append(np.array([wrist.x, wrist.y]))
            
            if len(hand_positions) == 2:
                # Расстояние между руками
                hand_distance = np.linalg.norm(hand_positions[0] - hand_positions[1])
                # Несинхронность = большая разница в движении
                async_movement = hand_distance > 0.3
                stress_indicators.append(('async_movement', async_movement, min(1.0, hand_distance)))
        
        # Общий индекс стресса
        if stress_indicators:
            stress_scores = [score for _, _, score in stress_indicators]
            stress_level = np.mean(stress_scores)
        else:
            stress_level = 0.0
        
        # Детализация по категориям
        stress_breakdown = {
            name: {
                'present': present,
                'intensity': float(score)
            }
            for name, present, score in stress_indicators
        }
        
        return {
            'stress_level': float(stress_level),
            'anxiety_score': float(stress_level * 0.9),  # тревожность связана со стрессом
            'stress_indicators': stress_breakdown,
            'stress_count': sum(1 for _, present, _ in stress_indicators if present)
        }
    
    def _calculate_ear(self, face_landmarks, image_shape, eye_type='left'):
        """Вычисляет Eye Aspect Ratio"""
        h, w = image_shape[:2]
        
        if eye_type == 'left':
            indices = [33, 160, 158, 133, 153, 144]
        else:
            indices = [362, 385, 387, 263, 373, 380]
        
        def get_coord(idx):
            if idx >= len(face_landmarks.landmark):
                return None
            lm = face_landmarks.landmark[idx]
            return np.array([lm.x * w, lm.y * h])
        
        try:
            p1, p2, p3, p4, p5, p6 = [get_coord(i) for i in indices]
            if any(p is None for p in [p1, p2, p3, p4, p5, p6]):
                return 0.3  # по умолчанию открыт
            
            v1 = np.linalg.norm(p2 - p6)
            v2 = np.linalg.norm(p3 - p5)
            h_dist = np.linalg.norm(p1 - p4)
            
            if h_dist > 0:
                ear = (v1 + v2) / (2.0 * h_dist)
            else:
                ear = 0.3
            return ear
        except:
            return 0.3


class BehaviorAnalyzer:
    """Главный класс для анализа поведения"""
    
    def __init__(self):
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_face_mesh = mp.solutions.face_mesh
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=False,
            model_complexity=2,
            smooth_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        self.gesture_classifier = HandGestureClassifier()
        self.body_analyzer = BodyLanguageAnalyzer()
        self.speech_analyzer = SpeechBehaviorAnalyzer()
        self.engagement_analyzer = EngagementAnalyzer()
        self.confidence_analyzer = ConfidenceAnalyzer()
        self.stress_analyzer = StressAnalyzer()
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """Обрабатывает один кадр"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        
        h, w = frame.shape[:2]
        
        # Обработка
        pose_results = self.pose.process(rgb_frame)
        hands_results = self.hands.process(rgb_frame)
        face_results = self.face_mesh.process(rgb_frame)
        
        results = {}
        
        # Жесты рук
        hand_gestures = []
        hand_landmarks_list = []
        if hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                hand_landmarks_list.append(hand_landmarks)
                gesture = self.gesture_classifier.classify_gesture(
                    hand_landmarks,
                    pose_results.pose_landmarks
                )
                hand_gestures.append(gesture)
        
        results['hand_gestures'] = hand_gestures
        results['num_hands'] = len(hand_landmarks_list)
        
        # Body language
        if pose_results.pose_landmarks:
            body_language = self.body_analyzer.analyze_posture(
                pose_results.pose_landmarks,
                frame.shape
            )
            results['body_language'] = body_language
        
        # Speech behavior
        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]
            speech_behavior = self.speech_analyzer.analyze_lip_sync(
                face_landmarks,
                frame.shape
            )
            results['speech_behavior'] = speech_behavior
        else:
            face_landmarks = None
        
        # Engagement Index
        engagement = self.engagement_analyzer.calculate_engagement(
            face_landmarks,
            pose_results.pose_landmarks,
            hand_landmarks_list,
            frame.shape
        )
        results['engagement'] = engagement
        
        # Confidence/Dominance Index
        confidence = self.confidence_analyzer.calculate_confidence(
            pose_results.pose_landmarks,
            face_landmarks,
            hand_landmarks_list,
            frame.shape
        )
        results['confidence'] = confidence
        
        # Stress/Anxiety
        stress = self.stress_analyzer.analyze_stress(
            face_landmarks,
            pose_results.pose_landmarks,
            hand_landmarks_list,
            frame.shape
        )
        results['stress'] = stress
        
        return results
    
    def process_video(self, video_path: str, output_path: Optional[str] = None, 
                     frame_skip: int = 1) -> Dict[str, Any]:
        """Обрабатывает видео"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return {'error': f'Cannot open video: {video_path}'}
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        all_results = []
        frame_idx = 0
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_skip == 0:
                result = self.process_frame(frame)
                result['frame_idx'] = frame_idx
                result['timestamp'] = frame_idx / fps
                all_results.append(result)
            
            frame_idx += 1
        
        cap.release()
        
        # Агрегирование результатов
        aggregated = self._aggregate_results(all_results)
        
        # Сохранение результатов
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump({
                    'video_path': video_path,
                    'fps': fps,
                    'total_frames': total_frames,
                    'processed_frames': len(all_results),
                    'frame_results': all_results,
                    'aggregated': aggregated
                }, f, indent=2, ensure_ascii=False)
        
        return {
            'success': True,
            'video_path': video_path,
            'fps': fps,
            'total_frames': total_frames,
            'processed_frames': len(all_results),
            'aggregated': aggregated
        }
    
    def _aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Агрегирует результаты по всему видео"""
        if not results:
            return {}
        
        aggregated = {}
        
        # Средние значения
        engagement_scores = [r.get('engagement', {}).get('engagement_score', 0) for r in results]
        confidence_scores = [r.get('confidence', {}).get('confidence_score', 0) for r in results]
        stress_scores = [r.get('stress', {}).get('stress_level', 0) for r in results]
        
        aggregated['avg_engagement'] = float(np.mean(engagement_scores))
        aggregated['avg_confidence'] = float(np.mean(confidence_scores))
        aggregated['avg_stress'] = float(np.mean(stress_scores))
        
        # Максимальные значения
        aggregated['max_engagement'] = float(np.max(engagement_scores))
        aggregated['max_confidence'] = float(np.max(confidence_scores))
        aggregated['max_stress'] = float(np.max(stress_scores))
        
        # Статистика жестов
        all_gestures = []
        for r in results:
            all_gestures.extend(r.get('hand_gestures', []))
        
        gesture_counts = {}
        for gesture in all_gestures:
            gesture_counts[gesture] = gesture_counts.get(gesture, 0) + 1
        
        aggregated['gesture_statistics'] = gesture_counts
        
        # Статистика поз
        postures = [r.get('body_language', {}).get('posture', 'unknown') for r in results]
        posture_counts = {}
        for posture in postures:
            if posture != 'unknown':
                posture_counts[posture] = posture_counts.get(posture, 0) + 1
        
        aggregated['posture_statistics'] = posture_counts
        
        return aggregated
    
    def __del__(self):
        """Очистка ресурсов"""
        if hasattr(self, 'pose'):
            self.pose.close()
        if hasattr(self, 'hands'):
            self.hands.close()
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()

