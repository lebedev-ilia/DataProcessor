"""
Модуль для извлечения профессиональных фич лица.
"""

from typing import Dict, List, Any, Optional
from collections import defaultdict, deque
import numpy as np

from _modules.base_module import FaceModule


class ProfessionalModule(FaceModule):
    """
    Модуль для извлечения профессиональных фич лица.
    Включает улучшенный fatigue score с полным анализом.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.fps = self.config.get("fps", 30.0)
        self._fatigue_history: Dict[int, deque] = defaultdict(
            lambda: deque(maxlen=int(self.fps * 5))  # История на 5 секунд для анализа трендов
        )

    def required_inputs(self) -> List[str]:
        """Требуются quality, eyes, motion, pose и опционально lip_reading."""
        return ["quality", "eyes", "motion", "pose"]

    def process(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Обрабатывает данные и возвращает профессиональные фичи."""
        quality = data["quality"]
        eyes = data["eyes"]
        motion = data["motion"]
        pose = data.get("pose", {})
        lip_reading = data.get("lip_reading", {})
        face_idx = data.get("face_idx", 0)

        # Face quality (computed as downstream meta-feature from base features)
        # Используем объединенные метрики из QualityModule
        face_sharpness = quality.get("face_sharpness", quality.get("sharpness_score", 0.0))
        face_noise = quality.get("face_noise_level", quality.get("noise_level", 0.0))
        face_exposure = quality.get("face_exposure_score", 0.5)
        
        face_quality = float(np.clip(
            0.4 * face_sharpness +
            0.3 * (1.0 - face_noise) +
            0.3 * face_exposure,
            0.0, 1.0
        ))

        # perceived_attractiveness_score удален - рискованная метрика, может быть дискриминационной
        # Вместо этого используем технические метрики качества

        # Emotion intensity
        emotion_intensity = float(motion.get("mouth_motion_score", 0.0) + motion.get("eyebrows_motion_score", 0.0))
        emotion_intensity = np.clip(emotion_intensity, 0.0, 1.0)

        # Eye-related metrics
        eye_opening_data = eyes.get("eye_opening_ratio", {})
        if isinstance(eye_opening_data, dict):
            avg_eye_opening = eye_opening_data.get("average", 0.5)
            left_eye_opening = eye_opening_data.get("left", 0.5)
            right_eye_opening = eye_opening_data.get("right", 0.5)
        else:
            avg_eye_opening = float(eye_opening_data) if isinstance(eye_opening_data, (int, float)) else 0.5
            left_eye_opening = right_eye_opening = avg_eye_opening

        gaze_score = eyes.get("gaze_at_camera_prob", 0.5)
        attention = eyes.get("attention_score", 0.5)
        blink_rate = eyes.get("blink_rate", 0.0)
        blink_intensity = eyes.get("blink_intensity", 0.0)

        # Lip reading / speaking - используем улучшенные фичи если доступны
        if lip_reading:
            lip_reading_features = {
                "mouth_openness": lip_reading.get("mouth_area", 0.0),
                "mouth_motion_intensity": lip_reading.get("mouth_motion_intensity", 0.0),
                "jaw_movement": motion.get("jaw_movement_intensity", 0.0),
                "speech_activity_prob": lip_reading.get("speech_activity_prob", 0.0),
                "phoneme_features": lip_reading.get("phoneme_features", {}),
                "cycle_strength": lip_reading.get("cycle_strength", 0.0),
            }
        else:
            # Fallback на базовые фичи из motion
            mouth_motion = motion.get("talking_motion_score", 0.0)
            jaw_motion = motion.get("jaw_movement_intensity", 0.0)
            lip_reading_features = {
                "mouth_openness": mouth_motion,
                "mouth_motion_intensity": mouth_motion,
                "jaw_movement": jaw_motion,
                "speech_activity_prob": float(np.clip(mouth_motion * 2.0, 0.0, 1.0)),
            }

        # Улучшенный Fatigue score - полный анализ
        face_speed = motion.get("face_speed", 0.0)
        head_motion = motion.get("head_motion_energy", 0.0)
        face_acceleration = motion.get("face_acceleration", 0.0)
        
        # 1. Eye-based fatigue indicators
        # Закрытые/полузакрытые глаза
        eye_closedness = 1.0 - np.clip(avg_eye_opening / 20.0, 0.0, 1.0)
        
        # Асимметрия открытия глаз (усталость может вызывать асимметрию)
        eye_asymmetry = abs(left_eye_opening - right_eye_opening) / max(max(left_eye_opening, right_eye_opening), 1e-6)
        
        # Низкая частота моргания или слишком высокая (оба признака усталости)
        blink_rate_normalized = np.clip(blink_rate * 60.0, 0.0, 30.0) / 30.0  # Нормализуем до 0-1
        # Нормальная частота: 15-20 в минуту, отклонения указывают на усталость
        blink_deviation = abs(blink_rate_normalized - 0.5) * 2.0  # Отклонение от нормы
        
        eye_fatigue_score = float(
            eye_closedness * 0.5 +
            eye_asymmetry * 0.2 +
            blink_deviation * 0.3
        )
        
        # 2. Pose-based fatigue indicators
        # Наклон головы вниз (pitch > 0 указывает на усталость)
        head_pitch = pose.get("pitch", 0.0)
        head_pitch_normalized = np.clip(head_pitch / 30.0, 0.0, 1.0)  # Наклон вниз = усталость
        
        # Нестабильность позы (усталость делает движения менее контролируемыми)
        pose_variability = pose.get("head_pose_variability", 0.0)
        pose_stability = 1.0 - np.clip(pose_variability / 15.0, 0.0, 1.0)
        
        # Снижение внимания к камере
        attention_to_camera = pose.get("attention_to_camera_ratio", 0.5)
        attention_loss = 1.0 - attention_to_camera
        
        pose_fatigue_score = float(
            head_pitch_normalized * 0.4 +
            (1.0 - pose_stability) * 0.3 +
            attention_loss * 0.3
        )
        
        # 3. Motion-based fatigue indicators
        # Медленные движения (усталость снижает скорость)
        speed_normalized = np.clip(face_speed / 10.0, 0.0, 1.0)
        low_speed_indicator = 1.0 - speed_normalized
        
        # Низкая активность (мало движений)
        motion_activity = motion.get("micro_expression_rate", 0.0)
        motion_activity_normalized = np.clip(motion_activity, 0.0, 1.0)
        low_activity_indicator = 1.0 - motion_activity_normalized
        
        # Неравномерные движения (усталость делает движения рывками)
        acceleration_variance = abs(face_acceleration)
        irregular_motion = np.clip(acceleration_variance / 5.0, 0.0, 1.0)
        
        motion_fatigue_score = float(
            low_speed_indicator * 0.4 +
            low_activity_indicator * 0.4 +
            irregular_motion * 0.2
        )
        
        # 4. Temporal patterns (анализ трендов во времени)
        current_fatigue_indicators = {
            "eye_closedness": eye_closedness,
            "head_pitch": head_pitch_normalized,
            "motion_speed": speed_normalized,
            "attention": attention_to_camera,
        }
        
        if face_idx not in self._fatigue_history:
            self._fatigue_history[face_idx] = deque(maxlen=int(self.fps * 5))
        
        history = self._fatigue_history[face_idx]
        history.append(current_fatigue_indicators)
        
        temporal_fatigue_score = 0.0
        if len(history) >= int(self.fps * 2):  # Минимум 2 секунды истории
            # Анализируем тренд: усталость обычно увеличивается со временем
            eye_closedness_history = [f["eye_closedness"] for f in history]
            motion_speed_history = [f["motion_speed"] for f in history]
            attention_history = [f["attention"] for f in history]
            
            # Проверяем тренды (увеличение усталости со временем)
            if len(eye_closedness_history) > 1:
                eye_trend = (eye_closedness_history[-1] - eye_closedness_history[0]) / len(eye_closedness_history)
                motion_trend = (1.0 - motion_speed_history[-1]) - (1.0 - motion_speed_history[0])
                attention_trend = attention_history[0] - attention_history[-1]
                
                # Положительные тренды = увеличение усталости
                temporal_fatigue_score = float(np.clip(
                    (eye_trend * 0.5 + motion_trend * 0.3 + attention_trend * 0.2) * 2.0,
                    0.0, 1.0
                ))
        
        # Комбинированный Fatigue Score
        fatigue_score = float(np.clip(
            eye_fatigue_score * 0.35 +
            pose_fatigue_score * 0.30 +
            motion_fatigue_score * 0.25 +
            temporal_fatigue_score * 0.10,
            0.0, 1.0
        ))
        
        # Детализированный fatigue breakdown
        fatigue_breakdown = {
            "eye_fatigue": float(eye_fatigue_score),
            "pose_fatigue": float(pose_fatigue_score),
            "motion_fatigue": float(motion_fatigue_score),
            "temporal_fatigue": float(temporal_fatigue_score),
            "eye_closedness": float(eye_closedness),
            "head_pitch_down": float(head_pitch_normalized),
            "low_motion_speed": float(low_speed_indicator),
            "blink_abnormality": float(blink_deviation),
        }

        # Engagement level
        micro_expressions = motion.get("micro_expression_rate", 0.0)
        mouth_motion = lip_reading_features.get("mouth_motion_intensity", motion.get("talking_motion_score", 0.0))
        
        engagement_level = float(np.clip(
            gaze_score * 0.35 +
            attention * 0.25 +
            min(micro_expressions, 1.0) * 0.2 +
            min(mouth_motion * 2.0, 1.0) * 0.1 +
            avg_eye_opening * 0.1,
            0.0, 1.0
        ))

        # Optional composite metrics
        alertness_score = float(np.clip(avg_eye_opening * 0.7 + face_speed / 15.0 * 0.3, 0.0, 1.0))
        expressiveness_score = float(np.clip(emotion_intensity + micro_expressions, 0.0, 1.0))

        return {
            "professional": {
                # High-level scores computed as downstream meta-features
                # (prefer learned classifier/ensembling on top of base features)
                "face_quality_score": face_quality,  # Computed from quality module features
                "emotion_intensity": emotion_intensity,  # Computed from motion/expression features
                "engagement_level": engagement_level,  # Computed from eyes/pose/motion features
                "alertness_score": alertness_score,  # Computed from eyes/motion features
                "expressiveness_score": expressiveness_score,  # Computed from motion/expression features
                # Detailed breakdowns
                "lip_reading_features": lip_reading_features,
                "fatigue_score": fatigue_score,
                "fatigue_breakdown": fatigue_breakdown,
                # Удалено: perceived_attractiveness_score - рискованная метрика
                # Рекомендация: compute high-level scores via learned model on module outputs
            }
        }

