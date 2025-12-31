"""
Утилиты для создания компактного набора фичей для VisualTransformer (~48 dims).
"""

from typing import Dict, List, Any, Optional
import numpy as np


def extract_compact_features(face_feature: Dict[str, Any]) -> np.ndarray:
    """
    Извлекает компактный набор фичей (~48 dims) для VisualTransformer.
    
    Структура:
    - face_embedding_proj (16 dims) - проекция PCA/learned от normalized landmarks
    - yaw_norm, pitch_norm, roll_norm (3 dims)
    - gaze_at_camera_prob (1 dim)
    - eye_opening_left, eye_opening_right (2 dims)
    - mouth_opening_ratio (1 dim)
    - face_size_rel (1 dim)
    - face_center_x_norm, face_center_y_norm (2 dims)
    - face_sharpness (1 dim)
    - face_noise_level (1 dim)
    - occlusion_proxy (1 dim)
    - glasses_prob (1 dim)
    - mask_prob (1 dim)
    - face_motion_speed_norm (1 dim)
    - head_motion_energy_norm (1 dim)
    - expression_proj (8 dims) - проекция expression_vector
    - detection_confidence (1 dim)
    
    Итого: ~42-48 dims
    """
    features = []
    
    # 1. face_embedding_proj (16 dims) - из face_shape_vector или structure.face_mesh_vector
    geometry = face_feature.get("geometry", {})
    structure = face_feature.get("structure", {})
    
    face_shape_vector = geometry.get("face_shape_vector", [])
    if len(face_shape_vector) >= 16:
        face_embedding_proj = face_shape_vector[:16]
    elif len(face_shape_vector) > 0:
        face_embedding_proj = face_shape_vector + [0.0] * (16 - len(face_shape_vector))
    else:
        # Fallback на face_mesh_vector
        face_mesh_vector = structure.get("face_mesh_vector", [])
        if len(face_mesh_vector) >= 16:
            face_embedding_proj = face_mesh_vector[:16]
        else:
            face_embedding_proj = [0.0] * 16
    
    features.extend(face_embedding_proj[:16])
    
    # 2. yaw_norm, pitch_norm, roll_norm (3 dims)
    pose = face_feature.get("pose", {})
    yaw_norm = pose.get("yaw_norm", pose.get("yaw", 0.0) / 90.0)
    pitch_norm = pose.get("pitch_norm", pose.get("pitch", 0.0) / 90.0)
    roll_norm = pose.get("roll_norm", pose.get("roll", 0.0) / 90.0)
    features.extend([yaw_norm, pitch_norm, roll_norm])
    
    # 3. gaze_at_camera_prob (1 dim)
    eyes = face_feature.get("eyes", {})
    gaze_at_camera_prob = eyes.get("gaze_at_camera_prob", 0.5)
    features.append(gaze_at_camera_prob)
    
    # 4. eye_opening_left, eye_opening_right (2 dims)
    eye_opening_ratio = eyes.get("eye_opening_ratio", {})
    if isinstance(eye_opening_ratio, dict):
        eye_opening_left = eye_opening_ratio.get("left", 0.5)
        eye_opening_right = eye_opening_ratio.get("right", 0.5)
    else:
        eye_opening_left = eyes.get("eye_opening_left", 0.5)
        eye_opening_right = eyes.get("eye_opening_right", 0.5)
    
    # Нормализуем (0-1)
    eye_opening_left_norm = np.clip(eye_opening_left / 20.0, 0.0, 1.0) if isinstance(eye_opening_left, (int, float)) else 0.5
    eye_opening_right_norm = np.clip(eye_opening_right / 20.0, 0.0, 1.0) if isinstance(eye_opening_right, (int, float)) else 0.5
    features.extend([eye_opening_left_norm, eye_opening_right_norm])
    
    # 5. mouth_opening_ratio (1 dim)
    lip_reading = face_feature.get("lip_reading", {})
    mouth_area = lip_reading.get("mouth_area", 0.0)
    mouth_width = lip_reading.get("mouth_width", 1.0)
    mouth_opening_ratio = np.clip(mouth_area / max(mouth_width * 10.0, 1e-6), 0.0, 1.0)
    features.append(mouth_opening_ratio)
    
    # 6. face_size_rel (1 dim)
    face_relative_size = geometry.get("face_relative_size", 0.0)
    features.append(face_relative_size)
    
    # 7. face_center_x_norm, face_center_y_norm (2 dims)
    face_center_x_norm = geometry.get("face_center_x_norm", 0.5)
    face_center_y_norm = geometry.get("face_center_y_norm", 0.5)
    features.extend([face_center_x_norm, face_center_y_norm])
    
    # 8. face_sharpness (1 dim)
    quality = face_feature.get("quality", {})
    face_sharpness = quality.get("face_sharpness", quality.get("sharpness_score", 0.5))
    features.append(face_sharpness)
    
    # 9. face_noise_level (1 dim)
    face_noise_level = quality.get("face_noise_level", quality.get("noise_level", 0.0))
    features.append(face_noise_level)
    
    # 10. occlusion_proxy (1 dim)
    occlusion_proxy = quality.get("occlusion_proxy", 0.0)
    features.append(occlusion_proxy)
    
    # 11. glasses_prob (1 dim)
    accessories = face_feature.get("accessories", {})
    glasses_prob = accessories.get("glasses_prob", 0.0)
    features.append(glasses_prob)
    
    # 12. mask_prob (1 dim)
    mask_prob = accessories.get("mask_prob", 0.0)
    features.append(mask_prob)
    
    # 13. face_motion_speed_norm (1 dim)
    motion = face_feature.get("motion", {})
    face_speed = motion.get("face_speed", 0.0)
    face_motion_speed_norm = np.clip(face_speed / 10.0, 0.0, 1.0)
    features.append(face_motion_speed_norm)
    
    # 14. head_motion_energy_norm (1 dim)
    head_motion_energy = motion.get("head_motion_energy", 0.0)
    head_motion_energy_norm = np.clip(head_motion_energy / 5.0, 0.0, 1.0)
    features.append(head_motion_energy_norm)
    
    # 15. expression_proj (8 dims) - из expression_vector
    expression_vector = structure.get("expression_vector", [])
    if len(expression_vector) >= 8:
        expression_proj = expression_vector[:8]
    elif len(expression_vector) > 0:
        expression_proj = expression_vector + [0.0] * (8 - len(expression_vector))
    else:
        expression_proj = [0.0] * 8
    features.extend(expression_proj[:8])
    
    # 16. detection_confidence (1 dim)
    detection_confidence = face_feature.get("detection_confidence", 1.0)
    features.append(detection_confidence)
    
    # Преобразуем в numpy array и обрезаем до нужной размерности
    features_array = np.array(features, dtype=np.float32)
    
    # Должно быть ~42-48 dims
    expected_dims = 16 + 3 + 1 + 2 + 1 + 1 + 2 + 1 + 1 + 1 + 1 + 1 + 1 + 1 + 8 + 1  # = 42
    if len(features_array) != expected_dims:
        # Дополняем или обрезаем до нужной размерности
        if len(features_array) < expected_dims:
            features_array = np.pad(features_array, (0, expected_dims - len(features_array)), mode='constant')
        else:
            features_array = features_array[:expected_dims]
    
    return features_array


def extract_per_face_aggregates(face_features_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Извлекает агрегаты на уровне лица (per-face aggregates).
    
    :param face_features_list: список фичей для одного лица по всем кадрам
    :return: словарь с агрегатами
    """
    if not face_features_list:
        return {}
    
    # Собираем метрики по кадрам
    yaws = []
    gaze_at_camera_probs = []
    blink_rates = []
    mouth_openings = []
    speech_activity_probs = []
    face_sharpnesses = []
    face_noise_levels = []
    face_motion_energies = []
    occlusion_proxies = []
    glasses_probs = []
    mask_probs = []
    expression_intensities = []
    
    for face_feat in face_features_list:
        pose = face_feat.get("pose", {})
        eyes = face_feat.get("eyes", {})
        lip_reading = face_feat.get("lip_reading", {})
        quality = face_feat.get("quality", {})
        motion = face_feat.get("motion", {})
        accessories = face_feat.get("accessories", {})
        structure = face_feat.get("structure", {})
        
        yaws.append(pose.get("yaw", 0.0))
        gaze_at_camera_probs.append(eyes.get("gaze_at_camera_prob", 0.0))
        blink_rates.append(eyes.get("blink_rate", 0.0))
        
        mouth_area = lip_reading.get("mouth_area", 0.0)
        mouth_width = lip_reading.get("mouth_width", 1.0)
        mouth_opening = np.clip(mouth_area / max(mouth_width * 10.0, 1e-6), 0.0, 1.0)
        mouth_openings.append(mouth_opening)
        
        speech_activity_probs.append(lip_reading.get("speech_activity_prob", 0.0))
        face_sharpnesses.append(quality.get("face_sharpness", 0.0))
        face_noise_levels.append(quality.get("face_noise_level", 0.0))
        face_motion_energies.append(motion.get("head_motion_energy", 0.0))
        occlusion_proxies.append(quality.get("occlusion_proxy", 0.0))
        glasses_probs.append(accessories.get("glasses_prob", 0.0))
        mask_probs.append(accessories.get("mask_prob", 0.0))
        
        # Expression intensity (norm of expression_proj)
        expression_vector = structure.get("expression_vector", [])
        if isinstance(expression_vector, list) and len(expression_vector) > 0:
            expression_intensity = np.linalg.norm(expression_vector[:8])
        else:
            expression_intensity = 0.0
        expression_intensities.append(expression_intensity)
    
    # Вычисляем статистики
    aggregates = {
        "avg_yaw": float(np.mean(yaws)) if yaws else 0.0,
        "std_yaw": float(np.std(yaws)) if yaws else 0.0,
        "head_turn_frequency": float(np.mean([abs(yaws[i] - yaws[i-1]) > 7.0 for i in range(1, len(yaws))])) if len(yaws) > 1 else 0.0,
        "gaze_at_camera_rate": float(np.mean([p > 0.7 for p in gaze_at_camera_probs])) if gaze_at_camera_probs else 0.0,
        "blink_rate": float(np.mean(blink_rates)) if blink_rates else 0.0,
        "avg_mouth_opening": float(np.mean(mouth_openings)) if mouth_openings else 0.0,
        "speech_activity_prob_mean": float(np.mean(speech_activity_probs)) if speech_activity_probs else 0.0,
        "speech_activity_prob_peak": float(np.max(speech_activity_probs)) if speech_activity_probs else 0.0,
        "face_visible_ratio": float(len(face_features_list) / max(len(face_features_list), 1)),  # Все кадры видны
        "avg_face_sharpness": float(np.mean(face_sharpnesses)) if face_sharpnesses else 0.0,
        "avg_noise_level": float(np.mean(face_noise_levels)) if face_noise_levels else 0.0,
        "face_motion_energy_mean": float(np.mean(face_motion_energies)) if face_motion_energies else 0.0,
        "face_motion_energy_std": float(np.std(face_motion_energies)) if face_motion_energies else 0.0,
        "expression_intensity_mean": float(np.mean(expression_intensities)) if expression_intensities else 0.0,
        "primary_accessory_flags": {
            "glasses": float(np.mean(glasses_probs)) > 0.5 if glasses_probs else False,
            "mask": float(np.mean(mask_probs)) > 0.5 if mask_probs else False,
            "hat": False,  # Можно добавить из accessories
        },
        "num_occlusion_events": int(np.sum([p > 0.5 for p in occlusion_proxies])) if occlusion_proxies else 0,
        "quality_proxy_score_mean": float(np.mean([q.get("quality_proxy_score", 0.0) for q in [f.get("quality", {}) for f in face_features_list]])) if face_features_list else 0.0,
        "min_quality_proxy_score": float(np.min([q.get("quality_proxy_score", 0.0) for q in [f.get("quality", {}) for f in face_features_list]])) if face_features_list else 0.0,
        "appearance_stability": 1.0,  # Можно улучшить, анализируя tracking_id
    }
    
    return aggregates

