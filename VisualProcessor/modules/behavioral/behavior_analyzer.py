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
from typing import Dict, List, Any
from collections import deque

name = "behavior_analyzer"

from utils.logger import get_logger
logger = get_logger(name)

class HandGestureClassifier:
    """
    Детальная классификация жестов рук.

    ВАЖНО: Классификатор теперь используется только как источник
    вероятностного распределения по жестам (soft representation),
    без `unknown` класса. Для задач sequence-моделирования мы хотим
    гладкий вектор вероятностей, а не жёсткий one-hot/label.
    """
    
    def __init__(self):
        # Базовое множество "осмысленных" жестов (без unknown)
        self.gesture_types = {
            'pointing': self._is_pointing,
            'open_palm': self._is_open_palm,
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
    
    def classify_gesture_hard(self, hand_landmarks, pose_landmarks=None) -> str:
        """Жёсткая классификация жеста (для обратной совместимости/отладки)."""
        for gesture_name, check_func in self.gesture_types.items():
            try:
                if check_func(hand_landmarks, pose_landmarks):
                    return gesture_name
            except Exception:
                continue
        # В новой схеме стараемся не использовать unknown дальше по пайплайну,
        # но для дебага всё ещё возвращаем его здесь.
        return 'unknown'

    def classify_gesture_soft(self, hand_landmarks, pose_landmarks=None) -> Dict[str, float]:
        """
        Мягкое представление жеста: распределение вероятностей по предопределённым типам.

        Т.к. базовые правила детектора дискретные, мы эмулируем "мягкость":
        - все жесты получают небольшой базовый вес (epsilon),
        - найденный по правилам жест получает повышенный вес,
        - затем нормируем вектор до суммы 1.0.
        """
        epsilon = 1e-3
        scores = {g: epsilon for g in self.gesture_types.keys()}

        detected = None
        for gesture_name, check_func in self.gesture_types.items():
            try:
                if check_func(hand_landmarks, pose_landmarks):
                    detected = gesture_name
                    break
            except Exception:
                continue

        if detected is not None:
            # усиливаем найденный класс
            scores[detected] = 1.0

        total = float(sum(scores.values())) or 1.0
        probs = {k: float(v / total) for k, v in scores.items()}
        return probs


class BodyLanguageAnalyzer:
    """Анализ языка тела"""
    
    def __init__(self):
        pass
    
    def analyze_posture(self, pose_landmarks, image_shape):
        """
        Анализирует язык тела.

        В НОВОЙ СХЕМЕ:
        - вместо дискретных поз/ярлыков возвращаем непрерывные физические сигналы:
          * arm_openness
          * pose_expansion
          * body_lean_angle
          * balance_offset
          * shoulder_angle
        Старые флаги (`open_posture`, `closed_posture`, `power_pose`, `rigidity`, ...),
        а также posture='standing/sitting' используются только для обратной
        совместимости и могут быть убраны на следующих шагах.
        """
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
        
        # Базовые геометрические величины
        shoulder_width = np.linalg.norm(left_shoulder - right_shoulder)
        pelvis_center = (left_hip + right_hip) / 2
        shoulder_center = (left_shoulder + right_shoulder) / 2

        # ---------
        # Старые флаги (сохраняем временно для обратной совместимости)
        # ---------

        # Поза (стоя/сидя) – будет удалена из FEATURES_DESCRIPTION
        shoulder_hip_distance = np.mean([
            np.linalg.norm(left_shoulder - left_hip),
            np.linalg.norm(right_shoulder - right_hip)
        ])
        results['posture'] = 'standing' if shoulder_hip_distance > h * 0.2 else 'sitting'
        
        # Открытая/закрытая поза (будет удалено из внешнего API)
        if left_wrist is not None and right_wrist is not None:
            wrist_distance = np.linalg.norm(left_wrist - right_wrist)
            results['open_posture'] = wrist_distance > shoulder_width * 1.2
            results['closed_posture'] = wrist_distance < shoulder_width * 0.8
        else:
            results['open_posture'] = False
            results['closed_posture'] = False
        
        # Power pose (будет удалено из FEATURES_DESCRIPTION)
        if left_wrist is not None and right_wrist is not None:
            hip_center = pelvis_center
            wrist_center = (left_wrist + right_wrist) / 2
            vertical_distance = abs(wrist_center[1] - hip_center[1])
            horizontal_spread = np.linalg.norm(left_wrist - right_wrist)
            
            results['power_pose'] = (
                vertical_distance < h * 0.1 and
                horizontal_spread > shoulder_width * 1.5
            )
        else:
            results['power_pose'] = False
        
        # Напряженность (rigidity)
        shoulder_angle_deg = np.degrees(np.arctan2(
            right_shoulder[1] - left_shoulder[1],
            right_shoulder[0] - left_shoulder[0]
        ))
        results['rigidity'] = abs(shoulder_angle_deg) < 5.0
        
        # Расслабленность
        results['relaxed'] = not results.get('rigidity', False) and not results.get('closed_posture', False)
        
        # ---------
        # Новые непрерывные признаки
        # ---------

        # 1) Arm openness: wrist_distance / shoulder_width
        if left_wrist is not None and right_wrist is not None and shoulder_width > 1e-6:
            wrist_distance = np.linalg.norm(left_wrist - right_wrist)
            arm_openness = float(wrist_distance / shoulder_width)
        else:
            arm_openness = 0.0
        results['arm_openness'] = arm_openness

        # 2) Pose expansion: отношение площади bbox человека к площади кадра
        keypoints = [left_shoulder, right_shoulder, left_hip, right_hip]
        if left_wrist is not None:
            keypoints.append(left_wrist)
        if right_wrist is not None:
            keypoints.append(right_wrist)

        xs = [p[0] for p in keypoints]
        ys = [p[1] for p in keypoints]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        person_area = max(0.0, (max_x - min_x) * (max_y - min_y))
        frame_area = float(w * h) if w > 0 and h > 0 else 1.0
        pose_expansion = float(person_area / frame_area)
        results['pose_expansion'] = pose_expansion

        # 3) Body lean angle (backward → forward, нормировано в [-1, 1])
        if nose is not None:
            # Вектор от центра таза к носу
            body_vec = nose - pelvis_center
            # Рассматриваем наклон вперёд/назад вдоль оси Y; нормируем на высоту кадра
            lean_raw = -(body_vec[1]) / max(float(h), 1.0)
            body_lean_angle = float(np.clip(lean_raw * 5.0, -1.0, 1.0))
        else:
            body_lean_angle = 0.0
        results['body_lean_angle'] = body_lean_angle

        # 4) Balance offset (как и раньше, [-1,1] влево/вправо)
        center_top = shoulder_center
        center_bottom = pelvis_center
        center_of_mass = (center_top + center_bottom) / 2
        frame_center_x = w / 2
        results['balance_offset'] = float((center_of_mass[0] - frame_center_x) / max(float(w), 1.0))

        # 5) Shoulder angle (абсолютный угол в градусах, и служебно храним исходное значение)
        results['shoulder_angle'] = float(shoulder_angle_deg)
        
        return results


class SpeechBehaviorAnalyzer:
    """
    Анализ динамики рта/речи.

    В новой схеме храним только "сырые" непрерывные признаки и простую
    прокси-метрику речи, пригодные для подачи в VisualTransformer:
      - mouth_width_norm
      - mouth_height_norm
      - mouth_area_norm
      - mouth_velocity
      - mouth_open_ratio
      - speech_activity_proxy
    """
    
    def __init__(self, window_size=10):
        self.window_size = window_size
        self.mouth_history = deque(maxlen=window_size)
    
    def analyze_mouth_dynamics(self, face_landmarks, image_shape):
        """Анализирует динамику рта и прокси-активность речи."""
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
        last_area = self.mouth_history[-1]['area'] if len(self.mouth_history) > 0 else None
        self.mouth_history.append({
            'width': mouth_width,
            'height': mouth_height,
            'area': mouth_area
        })
        
        # Мгновенная скорость изменения площади рта (proxy mouth_velocity)
        if last_area is not None:
            mouth_velocity = abs(mouth_area - last_area)
        else:
            mouth_velocity = 0.0

        # Нормировки
        frame_diag = float(np.sqrt(w ** 2 + h ** 2)) or 1.0
        mouth_width_norm = float(mouth_width / frame_diag)
        mouth_height_norm = float(mouth_height / frame_diag)
        mouth_area_norm = float(mouth_area / (w * h + 1e-6))

        # Отношение открытия
        mouth_open_ratio = float(mouth_height / max(mouth_width, 1.0))

        # Прокси активности речи: sigmoid(z(mouth_velocity))
        # Масштабируем скорость и прогоняем через сигмоиду
        scaled = mouth_velocity / (w * 0.01 + 1e-6)
        speech_activity_proxy = float(1.0 / (1.0 + np.exp(-scaled)))
        
        return {
            'mouth_width_norm': mouth_width_norm,
            'mouth_height_norm': mouth_height_norm,
            'mouth_area_norm': mouth_area_norm,
            'mouth_velocity': float(mouth_velocity),
            'mouth_open_ratio': mouth_open_ratio,
            'speech_activity_proxy': speech_activity_proxy,
        }


class EngagementAnalyzer:
    """
    Ранее: hand-crafted индекс вовлеченности на кадр.

    Теперь: интерфейс-заглушка для обратной совместимости.
    Высокоуровневые метрики вовлеченности должны вычисляться уже
    на уровне финальной головы (MLP), а не внутри behavioral.

    Этот класс оставлен, чтобы не ломать импорт, но не используется
    в новой схеме sequence features.
    """
    
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.engagement_history = deque(maxlen=window_size)
    
    def calculate_engagement(self, *args, **kwargs):
        """
        Возвращает пустую структуру. Логика engagement перенесена
        на уровень агрегированных фичей (post-hoc).
        """
        return {}


class ConfidenceAnalyzer:
    """
    Ранее: кадровый индекс уверенности/доминантности.

    В новой схеме confidence/dominance считаются уже из латентных
    представлений модели (MLP head), а не внутри MediaPipe-пайплайна.
    """
    
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.confidence_history = deque(maxlen=window_size)
    
    def calculate_confidence(self, *args, **kwargs):
        """Возвращает пустую структуру, логика вынесена наружу."""
        return {}


class StressAnalyzer:
    """Детекция признаков стресса и тревожности"""
    
    def __init__(self, window_size=30):
        self.window_size = window_size
        self.blink_history = deque(maxlen=window_size)
        self.movement_history = deque(maxlen=window_size)
    
    def analyze_stress(self, face_landmarks, pose_landmarks, hand_landmarks_list, image_shape):
        """
        Анализирует "сырые" признаки стресса без интерпретаций:
          - blink_flag / blink_rate_short
          - self_touch_flag
          - fidgeting_energy
        """
        h, w = image_shape[:2]

        blink_flag = 0
        blink_rate_short = 0.0
        self_touch_flag = 0
        fidgeting_energy = 0.0
        
        # 1. Моргание (EAR)
        if face_landmarks is not None:
            left_ear = self._calculate_ear(face_landmarks, image_shape, 'left')
            right_ear = self._calculate_ear(face_landmarks, image_shape, 'right')
            avg_ear = (left_ear + right_ear) / 2
            
            is_blinking = avg_ear < 0.2
            blink_flag = int(is_blinking)
            self.blink_history.append(is_blinking)
            
            if len(self.blink_history) > 0:
                blink_rate_short = float(sum(self.blink_history) / len(self.blink_history))

        # 2. Self-touch (через классификацию жестов)
        if hand_landmarks_list:
            gesture_classifier = HandGestureClassifier()
            for hand_landmarks in hand_landmarks_list:
                gesture = gesture_classifier.classify_gesture_hard(hand_landmarks, pose_landmarks)
                if gesture == 'self_touch':
                    self_touch_flag = 1
                    break
        
        # 3. Fidgeting (вариативность позиции носа за последнее окно)
        if pose_landmarks is not None and len(pose_landmarks.landmark) > 0:
            nose = pose_landmarks.landmark[0]
            current_pos = np.array([nose.x, nose.y])
            self.movement_history.append(current_pos)
            
            if len(self.movement_history) >= 2:
                positions = np.stack(self.movement_history, axis=0)
                var_x = float(np.var(positions[:, 0]))
                var_y = float(np.var(positions[:, 1]))
                fidgeting_energy = var_x + var_y

        return {
            'blink_flag': int(blink_flag),
            'blink_rate_short': float(blink_rate_short),
            'self_touch_flag': int(self_touch_flag),
            'fidgeting_energy': float(fidgeting_energy),
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
    
    def __init__(
        self,
        # pose
        pose_static_image_mode: bool = False,
        pose_model_complexity: int = 2,
        pose_smooth_landmarks: bool = True,
        pose_min_detection_confidence: float = 0.5,
        pose_min_tracking_confidence: float = 0.5,
        # hands
        hands_static_image_mode: bool = False,
        hands_max_num_hands: int = 2,
        hands_model_complexity: int = 1,
        hands_min_detection_confidence: float = 0.5,
        hands_min_tracking_confidence: float = 0.5,
        # face
        face_static_image_mode: bool = False,
        face_max_num_faces: int = 1,
        face_refine_landmarks: bool = True,
        face_min_detection_confidence: float = 0.5,
        face_min_tracking_confidence: float = 0.5,
    ):
        self.mp_pose = mp.solutions.pose
        self.mp_hands = mp.solutions.hands
        self.mp_face_mesh = mp.solutions.face_mesh
        
        self.pose = self.mp_pose.Pose(
            static_image_mode=pose_static_image_mode,
            model_complexity=pose_model_complexity,
            smooth_landmarks=pose_smooth_landmarks,
            min_detection_confidence=pose_min_detection_confidence,
            min_tracking_confidence=pose_min_tracking_confidence
        )
        
        self.hands = self.mp_hands.Hands(
            static_image_mode=hands_static_image_mode,
            max_num_hands=hands_max_num_hands,
            model_complexity=hands_model_complexity,
            min_detection_confidence=hands_min_detection_confidence,
            min_tracking_confidence=hands_min_tracking_confidence
        )
        
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=face_static_image_mode,
            max_num_faces=face_max_num_faces,
            refine_landmarks=face_refine_landmarks,
            min_detection_confidence=face_min_detection_confidence,
            min_tracking_confidence=face_min_tracking_confidence
        )
        
        self.gesture_classifier = HandGestureClassifier()
        self.body_analyzer = BodyLanguageAnalyzer()
        self.speech_analyzer = SpeechBehaviorAnalyzer()
        self.engagement_analyzer = EngagementAnalyzer()
        self.confidence_analyzer = ConfidenceAnalyzer()
        self.stress_analyzer = StressAnalyzer()
        # для динамики головы/плеч/рук
        self._prev_head_center = None
        self._prev_shoulder_angle = None
        self._prev_hands_center = None
    
    def process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Обрабатывает один кадр и возвращает "сырые" непрерывные поведенческие сигналы.

        Результат содержит:
          - sequence_features: числовые каналы для VisualTransformer
          - вспомогательные поля (hand_gestures, body_language_raw, и т.п.) для отладки.
        """
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        
        h, w = frame.shape[:2]
        
        # Обработка
        pose_results = self.pose.process(rgb_frame)
        hands_results = self.hands.process(rgb_frame)
        face_results = self.face_mesh.process(rgb_frame)
        
        results: Dict[str, Any] = {}
        sequence_features: Dict[str, Any] = {}
        
        # -------------------------
        # 1. РУКИ И ЖЕСТЫ
        # -------------------------
        hand_gestures = []
        hand_landmarks_list = []
        if hands_results.multi_hand_landmarks:
            for hand_landmarks in hands_results.multi_hand_landmarks:
                hand_landmarks_list.append(hand_landmarks)
                gesture = self.gesture_classifier.classify_gesture_hard(
                    hand_landmarks,
                    pose_results.pose_landmarks
                )
                hand_gestures.append(gesture)
        
        results['hand_gestures'] = hand_gestures
        num_hands = len(hand_landmarks_list)
        results['num_hands'] = num_hands
        sequence_features['num_hands'] = int(num_hands)
        sequence_features['hands_visibility'] = 1 if num_hands > 0 else 0

        # Мягкое представление жестов: усредняем по всем рукам
        gesture_probs_accum = {g: 0.0 for g in self.gesture_classifier.gesture_types.keys()}
        if hand_landmarks_list:
            for hand_landmarks in hand_landmarks_list:
                probs = self.gesture_classifier.classify_gesture_soft(
                    hand_landmarks,
                    pose_results.pose_landmarks
                )
                for g, p in probs.items():
                    gesture_probs_accum[g] += float(p)
            # усредняем по числу рук
            for g in gesture_probs_accum.keys():
                gesture_probs_accum[g] /= float(len(hand_landmarks_list))
        # пишем как отдельные каналы
        sequence_features['gesture_probs'] = gesture_probs_accum

        # Hand motion energy: движение центра запястий между кадрами
        current_hands_center = None
        if hand_landmarks_list:
            wrist_points = []
            for hand_landmarks in hand_landmarks_list:
                wrist = hand_landmarks.landmark[0]
                wrist_points.append(np.array([wrist.x * w, wrist.y * h]))
            current_hands_center = np.mean(wrist_points, axis=0)

        if current_hands_center is not None and self._prev_hands_center is not None:
            hand_motion_energy = float(np.linalg.norm(current_hands_center - self._prev_hands_center))
        else:
            hand_motion_energy = 0.0
        sequence_features['hand_motion_energy'] = hand_motion_energy
        self._prev_hands_center = current_hands_center
        
        # -------------------------
        # 2. ТЕЛО / ПОЗА
        # -------------------------
        if pose_results.pose_landmarks:
            body_language = self.body_analyzer.analyze_posture(
                pose_results.pose_landmarks,
                frame.shape
            )
            results['body_language'] = body_language  # raw debug

            sequence_features['arm_openness'] = float(body_language.get('arm_openness', 0.0))
            sequence_features['pose_expansion'] = float(body_language.get('pose_expansion', 0.0))
            sequence_features['body_lean_angle'] = float(body_language.get('body_lean_angle', 0.0))
            sequence_features['balance_offset'] = float(body_language.get('balance_offset', 0.0))

            shoulder_angle = float(body_language.get('shoulder_angle', 0.0))
            sequence_features['shoulder_angle'] = shoulder_angle

            if self._prev_shoulder_angle is not None:
                shoulder_angle_velocity = abs(shoulder_angle - self._prev_shoulder_angle)
            else:
                shoulder_angle_velocity = 0.0
            sequence_features['shoulder_angle_velocity'] = float(shoulder_angle_velocity)
            self._prev_shoulder_angle = shoulder_angle
        
        # -------------------------
        # 3. ГОЛОВА / ВЗГЛЯД (proxy)
        # -------------------------
        head_position_x_norm = 0.0
        head_position_y_norm = 0.0
        head_motion_energy = 0.0
        head_stability = 1.0

        face_landmarks = None
        if face_results.multi_face_landmarks:
            face_landmarks = face_results.multi_face_landmarks[0]
            # центр головы как центр bbox по всем точкам
            xs = []
            ys = []
            for lm in face_landmarks.landmark:
                xs.append(lm.x * w)
                ys.append(lm.y * h)
            if xs and ys:
                cx = float(np.mean(xs))
                cy = float(np.mean(ys))
                head_position_x_norm = cx / max(float(w), 1.0)
                head_position_y_norm = cy / max(float(h), 1.0)

                current_head_center = np.array([cx, cy])
                if self._prev_head_center is not None:
                    head_motion_energy = float(np.linalg.norm(current_head_center - self._prev_head_center))
                self._prev_head_center = current_head_center

        sequence_features['head_position_x_norm'] = float(head_position_x_norm)
        sequence_features['head_position_y_norm'] = float(head_position_y_norm)
        sequence_features['head_motion_energy'] = float(head_motion_energy)
        head_stability = float(1.0 / (head_motion_energy + 1e-6))
        sequence_features['head_stability'] = head_stability

        # -------------------------
        # 4. РОТ / РЕЧЬ
        # -------------------------
        if face_landmarks is not None:
            mouth_dynamics = self.speech_analyzer.analyze_mouth_dynamics(
                face_landmarks,
                frame.shape
            )
            results['speech_behavior'] = mouth_dynamics
            sequence_features.update(mouth_dynamics)

        # -------------------------
        # 5. СТРЕСС: RAW СИГНАЛЫ
        # -------------------------
        stress = self.stress_analyzer.analyze_stress(
            face_landmarks,
            pose_results.pose_landmarks,
            hand_landmarks_list,
            frame.shape
        )
        results['stress'] = stress
        sequence_features.update(stress)

        # Записываем sequence_features
        results['sequence_features'] = sequence_features
        
        return results
    
    def process_video(self, frame_manager, frame_indices) -> Dict[str, Any]:
        """Обрабатывает видео"""
        import time

        fps = frame_manager.fps
        
        all_results: Dict[int, Dict[str, Any]] = {}
        c = 0
        t = time.time()
        
        for frame_idx in frame_indices:
            frame = frame_manager.get(frame_idx)
            
            result = self.process_frame(frame)
            timestamp = frame_idx / fps
            result['timestamp'] = float(timestamp)
            all_results[frame_idx] = result

            c += 1

            if c % 20 == 0:
                l = time.time()
                d = round(l - t, 2)
                t = l
                logger.info(f"Behavioral | Обработано кадров: {c}/{len(frame_indices)} | Time: {d}")
        
        # Нормализованный timestamp (t / video_duration)
        if all_results:
            min_f = min(frame_indices)
            max_f = max(frame_indices)
            video_duration = max((max_f - min_f) / fps, 1e-6)
            for idx, res in all_results.items():
                t_abs = res.get('timestamp', 0.0)
                t_rel = float(np.clip(t_abs / video_duration, 0.0, 1.0))
                seq = res.setdefault('sequence_features', {})
                seq['timestamp_norm'] = t_rel
        
        aggregated = self._aggregate_results(all_results)
        
        return {
            'frame_results': self.make_serializable(all_results),
            'aggregated': self.make_serializable(aggregated)
        }

    def make_serializable(self, obj):
        import numpy as np
        # numpy bool
        if isinstance(obj, (np.bool_,)):
            return bool(obj)
        # python bool
        if isinstance(obj, bool):
            return obj
        # numpy int/float
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        # numpy array
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # tuple -> list
        if isinstance(obj, tuple):
            return [self.make_serializable(x) for x in obj]
        # list
        if isinstance(obj, list):
            return [self.make_serializable(x) for x in obj]
        # dict
        if isinstance(obj, dict):
            return {k: self.make_serializable(v) for k, v in obj.items()}
        # objects with __dict__
        if hasattr(obj, "__dict__"):
            return self.make_serializable(obj.__dict__)
        return obj
    
    def _aggregate_results(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Агрегирует результаты по всему видео.

        Важно: здесь допустима интерпретация и high-level метрики,
        но они считаются из сырых sequence_features.
        """
        if not results:
            return {}
        
        aggregated: Dict[str, Any] = {}

        # Собираем последовательности по кадрам
        seq_list = [r.get('sequence_features', {}) for r in results.values()]

        def get_series(key: str):
            return np.array([float(s.get(key, 0.0)) for s in seq_list], dtype=float) if seq_list else np.zeros(0)

        # -----------------------------
        # 1. ENGAGEMENT (high-level)
        # -----------------------------
        # Простая прокси: комбинация активности рта, жестов и наклона корпуса
        speech_proxy = get_series('speech_activity_proxy')
        arm_open = get_series('arm_openness')
        body_lean = get_series('body_lean_angle')

        # нормировка
        def norm_sig(x):
            return 1.0 / (1.0 + np.exp(-x)) if np.isscalar(x) else 1.0 / (1.0 + np.exp(-x))

        engagement_signal = 0.5 * speech_proxy + 0.3 * norm_sig(arm_open) + 0.2 * norm_sig(body_lean)
        if engagement_signal.size == 0:
            engagement_signal = np.zeros(1)

        aggregated['avg_engagement'] = float(np.mean(engagement_signal))
        aggregated['max_engagement'] = float(np.max(engagement_signal))
        aggregated['engagement_variance'] = float(np.var(engagement_signal))

        # Пики вовлечённости (локальные максимумы)
        engagement_peaks = 0
        if engagement_signal.size >= 3:
            for i in range(1, engagement_signal.size - 1):
                if engagement_signal[i] > engagement_signal[i - 1] and engagement_signal[i] > engagement_signal[i + 1]:
                    engagement_peaks += 1
        aggregated['engagement_peaks'] = int(engagement_peaks)

        n = engagement_signal.size
        if n >= 5:
            split = max(int(0.2 * n), 1)
            early = engagement_signal[:split]
            late = engagement_signal[-split:]
            aggregated['early_engagement_mean'] = float(np.mean(early))
            aggregated['late_engagement_mean'] = float(np.mean(late))
        else:
            aggregated['early_engagement_mean'] = float(np.mean(engagement_signal))
            aggregated['late_engagement_mean'] = float(np.mean(engagement_signal))

        # -----------------------------
        # 2. CONFIDENCE / DOMINANCE
        # -----------------------------
        # Proxy: открытость рук + наклон корпуса вперёд
        confidence_signal = 0.6 * norm_sig(arm_open) + 0.4 * norm_sig(body_lean)
        if confidence_signal.size == 0:
            confidence_signal = np.zeros(1)

        aggregated['avg_confidence'] = float(np.mean(confidence_signal))
        aggregated['max_confidence'] = float(np.max(confidence_signal))
        aggregated['confidence_variance'] = float(np.var(confidence_signal))

        confidence_peaks = 0
        if confidence_signal.size >= 3:
            for i in range(1, confidence_signal.size - 1):
                if confidence_signal[i] > confidence_signal[i - 1] and confidence_signal[i] > confidence_signal[i + 1]:
                    confidence_peaks += 1
        aggregated['confidence_peak_count'] = int(confidence_peaks)

        # -----------------------------
        # 3. STRESS SUMMARY
        # -----------------------------
        blink_rate_short = get_series('blink_rate_short')
        self_touch_flag = get_series('self_touch_flag')
        fidgeting_energy = get_series('fidgeting_energy')

        stress_proxy = 0.4 * blink_rate_short + 0.3 * self_touch_flag + 0.3 * norm_sig(fidgeting_energy * 10.0)
        if stress_proxy.size == 0:
            stress_proxy = np.zeros(1)

        aggregated['avg_stress'] = float(np.mean(stress_proxy))
        aggregated['max_stress'] = float(np.max(stress_proxy))
        aggregated['stress_spike_count'] = int(np.sum(stress_proxy > (np.mean(stress_proxy) + np.std(stress_proxy))))
        aggregated['stress_duration_ratio'] = float(np.mean(stress_proxy > 0.5))

        # -----------------------------
        # 4. GESTURE STATISTICS
        # -----------------------------
        # Используем как gesture_counts (по hard-ярлыкам) и soft-характеристики
        all_gestures = []
        for r in results.values():
            all_gestures.extend(r.get('hand_gestures', []))
        gesture_counts: Dict[str, int] = {}
        for g in all_gestures:
            gesture_counts[g] = gesture_counts.get(g, 0) + 1
        aggregated['gesture_counts'] = gesture_counts

        total_frames = max(len(seq_list), 1)
        aggregated['gesture_rate_per_sec'] = float(len(all_gestures) / total_frames)

        # gesture entropy по soft распределениям
        entropies = []
        for s in seq_list:
            probs = s.get('gesture_probs', {})
            if not probs:
                continue
            p = np.array(list(probs.values()), dtype=float)
            p = p / (p.sum() + 1e-8)
            ent = float(-np.sum(p * np.log2(p + 1e-8)))
            entropies.append(ent)
        aggregated['gesture_entropy_mean'] = float(np.mean(entropies)) if entropies else 0.0

        if gesture_counts:
            dominant = max(gesture_counts.values())
            aggregated['dominant_gesture_ratio'] = float(dominant / max(sum(gesture_counts.values()), 1))
        else:
            aggregated['dominant_gesture_ratio'] = 0.0

        # Простая оценка скорости смены жестов
        gesture_switches = 0
        if all_gestures:
            for i in range(1, len(all_gestures)):
                if all_gestures[i] != all_gestures[i - 1]:
                    gesture_switches += 1
        aggregated['gesture_switching_rate'] = float(gesture_switches / max(total_frames - 1, 1))

        # -----------------------------
        # 5. BODY & MOTION SUMMARY
        # -----------------------------
        pose_expansion = get_series('pose_expansion')
        balance_offset = get_series('balance_offset')

        aggregated['avg_arm_openness'] = float(np.mean(arm_open)) if arm_open.size > 0 else 0.0
        aggregated['avg_pose_expansion'] = float(np.mean(pose_expansion)) if pose_expansion.size > 0 else 0.0

        # Энергия движения тела: используем head_motion_energy как прокси
        body_motion_energy = get_series('head_motion_energy')
        aggregated['body_motion_energy_mean'] = float(np.mean(body_motion_energy)) if body_motion_energy.size > 0 else 0.0
        aggregated['body_motion_energy_var'] = float(np.var(body_motion_energy)) if body_motion_energy.size > 0 else 0.0

        # -----------------------------
        # 6. SPEECH / RHYTHM SUMMARY
        # -----------------------------
        speech_proxy = get_series('speech_activity_proxy')
        aggregated['speech_activity_ratio'] = float(np.mean(speech_proxy > 0.5)) if speech_proxy.size > 0 else 0.0

        # burstiness: насколько активность речи сконцентрирована
        if speech_proxy.size > 0:
            mean_s = float(np.mean(speech_proxy))
            if mean_s > 0:
                aggregated['speech_burstiness'] = float(np.var(speech_proxy) / (mean_s ** 2 + 1e-8))
            else:
                aggregated['speech_burstiness'] = 0.0
        else:
            aggregated['speech_burstiness'] = 0.0

        aggregated['mouth_rhythm_score'] = float(np.std(speech_proxy)) if speech_proxy.size > 0 else 0.0

        # -----------------------------
        # 7. TEMPORAL CONTRAST
        # -----------------------------
        def temporal_contrast(sig):
            if sig.size == 0:
                return 0.0, 0.0, 0.0
            mean_val = float(np.mean(sig))
            max_val = float(np.max(sig))
            contrast = max_val - mean_val
            n_local = sig.size
            if n_local >= 5:
                split = max(int(0.2 * n_local), 1)
                early = sig[:split]
                late = sig[-split:]
                early_mean = float(np.mean(early))
                late_mean = float(np.mean(late))
            else:
                early_mean = late_mean = mean_val
            return contrast, early_mean, late_mean

        engagement_contrast, early_e, late_e = temporal_contrast(engagement_signal)
        confidence_contrast, early_c, late_c = temporal_contrast(confidence_signal)
        stress_contrast, early_s, late_s = temporal_contrast(stress_proxy)

        aggregated['engagement_contrast'] = float(engagement_contrast)
        aggregated['confidence_contrast'] = float(confidence_contrast)
        aggregated['stress_contrast'] = float(stress_contrast)

        aggregated['early_late_ratios'] = {
            'engagement': float((late_e + 1e-6) / (early_e + 1e-6)),
            'speech_activity': float((late_s + 1e-6) / (early_s + 1e-6)),
            'gesture_rate': float(aggregated['gesture_rate_per_sec'])
        }

        # -----------------------------
        # 8. PRESENCE & FRAMING
        # -----------------------------
        num_hands_series = get_series('num_hands')
        hands_visibility_ratio = float(np.mean(num_hands_series > 0)) if num_hands_series.size > 0 else 0.0
        aggregated['hands_visibility_ratio'] = hands_visibility_ratio

        # face_visibility_ratio – по наличию head_position (ненулевой x_norm)
        head_x = get_series('head_position_x_norm')
        aggregated['face_visibility_ratio'] = float(np.mean(head_x > 0)) if head_x.size > 0 else 0.0

        aggregated['center_bias_mean'] = float(np.mean(np.abs(balance_offset))) if balance_offset.size > 0 else 0.0

        return aggregated
    
    def __del__(self):
        """Очистка ресурсов"""
        if hasattr(self, 'pose'):
            self.pose.close()
        if hasattr(self, 'hands'):
            self.hands.close()
        if hasattr(self, 'face_mesh'):
            self.face_mesh.close()

