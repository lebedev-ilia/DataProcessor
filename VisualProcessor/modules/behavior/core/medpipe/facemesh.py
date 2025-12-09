import mediapipe as mp
import numpy as np
import cv2

# Инициализация MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Конфигурация для Face Mesh
cfg = {
    'static_image_mode': False,
    'max_num_faces': 10,  # Максимальное количество лиц
    'refine_landmarks': True,  # Детальные лицевые точки
    'min_detection_confidence': 0.5,
    'min_tracking_confidence': 0.5
}

# Настройки для разных частей лица с цветовой кодировкой
FACE_PARTS_CONFIG = {
    'tesselation': {
        'connections': mp_face_mesh.FACEMESH_TESSELATION,
        'landmark_spec': None,
        'connection_spec': mp_drawing_styles.get_default_face_mesh_tesselation_style(),
        'enabled': True
    },
    'face_oval': {
        'connections': mp_face_mesh.FACEMESH_FACE_OVAL,
        'landmark_spec': None,
        'connection_spec': mp_drawing.DrawingSpec(color=(255, 255, 0), thickness=2),  # Голубой
        'enabled': True
    },
    'left_eye': {
        'connections': mp_face_mesh.FACEMESH_LEFT_EYE,
        'landmark_spec': mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=1),  # Синий
        'connection_spec': mp_drawing.DrawingSpec(color=(255, 0, 0), thickness=2),  # Синий
        'enabled': True
    },
    'right_eye': {
        'connections': mp_face_mesh.FACEMESH_RIGHT_EYE,
        'landmark_spec': mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1),  # Зеленый
        'connection_spec': mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2),  # Зеленый
        'enabled': True
    },
    'left_eyebrow': {
        'connections': mp_face_mesh.FACEMESH_LEFT_EYEBROW,
        'landmark_spec': None,
        'connection_spec': mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2),  # Желтый
        'enabled': True
    },
    'right_eyebrow': {
        'connections': mp_face_mesh.FACEMESH_RIGHT_EYEBROW,
        'landmark_spec': None,
        'connection_spec': mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=2),  # Желтый
        'enabled': True
    },
    'lips': {
        'connections': mp_face_mesh.FACEMESH_LIPS,
        'landmark_spec': mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=1),  # Красный
        'connection_spec': mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2),  # Красный
        'enabled': True
    },
    'nose': {
        'connections': mp_face_mesh.FACEMESH_NOSE,
        'landmark_spec': None,
        'connection_spec': mp_drawing.DrawingSpec(color=(255, 165, 0), thickness=2),  # Оранжевый
        'enabled': True
    },
    'irises': {
        'connections': mp_face_mesh.FACEMESH_IRISES,
        'landmark_spec': mp_drawing.DrawingSpec(color=(128, 0, 128), thickness=1),  # Фиолетовый
        'connection_spec': mp_drawing.DrawingSpec(color=(128, 0, 128), thickness=1),  # Фиолетовый
        'enabled': True
    },
    'contours': {
        'connections': mp_face_mesh.FACEMESH_CONTOURS,
        'landmark_spec': None,
        'connection_spec': mp_drawing.DrawingSpec(color=(192, 192, 192), thickness=1),  # Серый
        'enabled': False  # По умолчанию отключено (слишком много линий)
    }
}

def analyze_eye_aspect_ratio(face_landmarks, image_shape, eye_type='left'):
    """
    Вычисляет соотношение сторон глаза (Eye Aspect Ratio).
    Значение близкое к 0 - глаз закрыт, > 0.25 - открыт.
    
    Формула EAR = (||p2-p6|| + ||p3-p5||) / (2 * ||p1-p4||)
    где p1-p6 - ключевые точки глаза.
    """
    h, w = image_shape[:2]
    
    # Индексы точек для глаз (по алгоритму EAR)
    if eye_type == 'left':
        # Левый глаз: индексы [33, 133, 157, 158, 159, 160, 161, 173]
        # p1, p2, p3, p4, p5, p6
        indices = [33, 160, 158, 133, 153, 144]
    else:  # right
        # Правый глаз: индексы [362, 263, 386, 387, 388, 389, 390, 466]
        indices = [362, 385, 387, 263, 373, 380]
    
    # Получаем координаты
    def get_coord(idx):
        lm = face_landmarks.landmark[idx]
        return np.array([lm.x * w, lm.y * h])
    
    try:
        # Вычисляем расстояния по формуле EAR
        p1, p2, p3, p4, p5, p6 = [get_coord(i) for i in indices]
        
        # Вертикальные расстояния
        v1 = np.linalg.norm(p2 - p6)
        v2 = np.linalg.norm(p3 - p5)
        
        # Горизонтальное расстояние
        h_dist = np.linalg.norm(p1 - p4)
        
        # EAR
        if h_dist > 0:
            ear = (v1 + v2) / (2.0 * h_dist)
        else:
            ear = 0
            
        return ear
    except Exception as e:
        #print(f"Ошибка расчета EAR для {eye_type} глаза: {e}")
        return 0.0

def calculate_face_orientation(face_landmarks, image_shape):
    """
    Оценивает ориентацию лица (наклон головы).
    Возвращает примерный угол поворота головы.
    """
    h, w = image_shape[:2]
    
    def get_coord(idx):
        lm = face_landmarks.landmark[idx]
        return np.array([lm.x * w, lm.y * h])
    
    try:
        # Используем точки носа и глаз для оценки
        nose_tip = get_coord(1)  # Кончик носа
        left_eye = get_coord(33)  # Левая сторона левого глаза
        right_eye = get_coord(263)  # Правая сторона правого глаза
        
        # Вычисляем угол между горизонтальной линией и линией глаз
        eye_line = right_eye - left_eye
        angle = np.degrees(np.arctan2(eye_line[1], eye_line[0]))
        
        # Нормализуем угол
        if angle > 45:
            orientation = "Turned right"
        elif angle < -45:
            orientation = "Turned left"
        else:
            orientation = "Looking forward"
            
        return orientation, angle
    except:
        return "Не определено", 0

def draw_face_parts(annotated_frame, face_landmarks, config=FACE_PARTS_CONFIG):
    """
    Отрисовывает выбранные части лица согласно конфигурации.
    """
    for part_name, part_config in config.items():
        if part_config['enabled']:
            mp_drawing.draw_landmarks(
                image=annotated_frame,
                landmark_list=face_landmarks,
                connections=part_config['connections'],
                landmark_drawing_spec=part_config['landmark_spec'],
                connection_drawing_spec=part_config['connection_spec']
            )

def resize_frame(frame, scale_percent=50):
    """Изменяет размер кадра для отображения"""
    if scale_percent == 100:
        return frame
        
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    return cv2.resize(frame, (width, height))

def draw_statistics(frame, num_faces, ear_left, ear_right, orientation, display_scale):
    """Отображает статистику на кадре (английский текст)"""
    h, w = frame.shape[:2]
    
    # Полупрозрачная панель для статистики
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (400, 140), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    y_offset = 40
    line_height = 25
    
    # Информация о лицах
    cv2.putText(frame, f"Faces detected: {num_faces}", (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    y_offset += line_height
    
    # EAR для глаз
    cv2.putText(frame, f"Left eye EAR: {ear_left:.3f}", (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 1)
    y_offset += line_height
    
    cv2.putText(frame, f"Right eye EAR: {ear_right:.3f}", (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 1)
    y_offset += line_height
    
    # Ориентация лица
    cv2.putText(frame, f"Face: {orientation}", (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
    y_offset += line_height
    
    # Масштаб отображения
    cv2.putText(frame, f"Scale: {display_scale}%", (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 165, 0), 1)
    
    # Предупреждение о закрытых глазах
    if ear_left < 0.2 or ear_right < 0.2:
        cv2.putText(frame, "WARNING: Eyes closed!", (w//2 - 150, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

# Основной код
with mp_face_mesh.FaceMesh(**cfg) as face_mesh:
    # Открытие видеофайла
    cap = cv2.VideoCapture("-NSumhkOwSg.mp4")
    
    # Проверка успешного открытия видео
    if not cap.isOpened():
        print("Ошибка: не удалось открыть видеофайл.")
        exit()
    
    # Получаем параметры видео
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print("=" * 60)
    print("MEDIAPIPE FACE MESH DETECTOR")
    print("=" * 60)
    print(f"Разрешение видео: {width}x{height}")
    print(f"Частота кадров: {fps}")
    print(f"Максимум лиц: {cfg['max_num_faces']}")
    print("=" * 60)
    print("Управление:")
    print("  q / ESC - выход")
    print("  +/-     - изменить масштаб окна")
    print("  t       - переключить отображение сетки (tesselation)")
    print("  l       - переключить отображение губ")
    print("  e       - переключить отображение глаз")
    print("  r       - сбросить все настройки отображения")
    print("=" * 60)
    
    display_scale = 50  # Начальный масштаб отображения (50%)
    frame_count = 0
    
    # --- ОСНОВНОЙ ЦИКЛ ОБРАБОТКИ ВИДЕО ---
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:  # Если кадры закончились (или ошибка чтения)
            print("Видео закончилось.")
            break
        
        frame_count += 1
        annotated_frame = frame.copy()
        
        # Конвертация BGR в RGB для MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False  # Для оптимизации (только чтение)
        
        # Обработка кадра через Face Mesh
        face_mesh_results = face_mesh.process(rgb_frame)
        
        num_faces = 0
        avg_ear_left = 0
        avg_ear_right = 0
        face_orientation = "Не обнаружено"
        
        if face_mesh_results.multi_face_landmarks:
            num_faces = len(face_mesh_results.multi_face_landmarks)
            
            # Обработка каждого лица
            for i, face_landmarks in enumerate(face_mesh_results.multi_face_landmarks):
                # Отрисовка всех частей лица
                draw_face_parts(annotated_frame, face_landmarks)
                
                # Вычисление EAR для каждого глаза
                ear_left = analyze_eye_aspect_ratio(face_landmarks, annotated_frame.shape, 'left')
                ear_right = analyze_eye_aspect_ratio(face_landmarks, annotated_frame.shape, 'right')
                
                avg_ear_left += ear_left
                avg_ear_right += ear_right
                
                # Вычисление ориентации лица (только для первого лица)
                if i == 0:
                    face_orientation, angle = calculate_face_orientation(face_landmarks, annotated_frame.shape)
                
                # Отображение номера лица
                h, w, _ = annotated_frame.shape
                face_center = (
                    int(face_landmarks.landmark[1].x * w),  # Нос
                    int(face_landmarks.landmark[1].y * h - 30)
                )
                cv2.putText(annotated_frame, f"Лицо {i+1}", face_center,
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Вычисление средних значений EAR
            if num_faces > 0:
                avg_ear_left /= num_faces
                avg_ear_right /= num_faces
        
        # Отображение статистики
        draw_statistics(annotated_frame, num_faces, avg_ear_left, avg_ear_right, 
                       face_orientation, display_scale)
        
        # Изменение размера для отображения
        display_frame = resize_frame(annotated_frame, display_scale)
        
        # Показываем обработанный кадр
        cv2.imshow('MediaPipe FaceMesh', display_frame)
        
        # Сохранение кадра (раскомментировать при необходимости)
        # out.write(annotated_frame)
        
        # Обработка клавиш управления
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:  # q или ESC
            print("Выход по запросу пользователя.")
            break
        elif key == ord('+') or key == ord('='):
            display_scale = min(display_scale + 10, 200)
            print(f"Масштаб увеличен до {display_scale}%")
        elif key == ord('-') or key == ord('_'):
            display_scale = max(display_scale - 10, 20)
            print(f"Масштаб уменьшен до {display_scale}%")
        elif key == ord('t'):
            FACE_PARTS_CONFIG['tesselation']['enabled'] = not FACE_PARTS_CONFIG['tesselation']['enabled']
            status = "включена" if FACE_PARTS_CONFIG['tesselation']['enabled'] else "отключена"
            print(f"Сетка (tesselation) {status}")
        elif key == ord('l'):
            FACE_PARTS_CONFIG['lips']['enabled'] = not FACE_PARTS_CONFIG['lips']['enabled']
            status = "включены" if FACE_PARTS_CONFIG['lips']['enabled'] else "отключены"
            print(f"Губы {status}")
        elif key == ord('e'):
            # Переключаем отображение глаз и бровей
            eye_parts = ['left_eye', 'right_eye', 'left_eyebrow', 'right_eyebrow']
            new_state = not FACE_PARTS_CONFIG['left_eye']['enabled']
            for part in eye_parts:
                FACE_PARTS_CONFIG[part]['enabled'] = new_state
            status = "включены" if new_state else "отключены"
            print(f"Глаза и брови {status}")
        elif key == ord('r'):
            # Сброс всех настроек отображения
            for part in FACE_PARTS_CONFIG:
                FACE_PARTS_CONFIG[part]['enabled'] = True
            FACE_PARTS_CONFIG['contours']['enabled'] = False
            print("Все настройки отображения сброшены")
    
    # --- КОРРЕКТНОЕ ОСВОБОЖДЕНИЕ РЕСУРСОВ ---
    cap.release()  # Закрываем видеофайл
    # out.release()  # Раскомментировать, если используется VideoWriter
    cv2.destroyAllWindows()  # Закрываем все открытые окна OpenCV
    
    print("=" * 60)
    print("ОБРАБОТКА ЗАВЕРШЕНА")
    print(f"Всего обработано кадров: {frame_count}")
    print("Ресурсы освобождены.")
    print("=" * 60)