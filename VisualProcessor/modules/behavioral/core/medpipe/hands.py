import cv2
import mediapipe as mp
import numpy as np

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

cfg = {
    'static_image_mode': False,
    'max_num_hands': 10,
    'model_complexity': 1,
    'min_detection_confidence': 0.5,
    'min_tracking_confidence': 0.5
}

# Конфигурация для разных частей руки
HAND_CONFIG = {
    'landmarks': {
        'enabled': True,
        'spec': mp_drawing_styles.get_default_hand_landmarks_style()
    },
    'connections': {
        'enabled': True,
        'spec': mp_drawing_styles.get_default_hand_connections_style()
    }
}

# Жесты и их описания
GESTURES = {
    'OPEN_PALM': "Open palm",
    'FIST': "Fist",
    'THUMBS_UP': "Thumbs up",
    'THUMBS_DOWN': "Thumbs down",
    'VICTORY': "Victory (peace)",
    'POINTING': "Pointing",
    'OK': "OK sign",
    'ROCK': "Rock (horns)",
    'CALL_ME': "Call me",
    'LOVE': "Love you"
}

def calculate_finger_states(hand_landmarks):
    """
    Определяет состояние пальцев (поднят/опущен).
    Возвращает словарь с состояниями для каждого пальца.
    """
    finger_states = {
        'thumb': False,
        'index': False,
        'middle': False,
        'ring': False,
        'pinky': False
    }
    
    # Индексы ключевых точек для каждого пальца
    finger_tips = [4, 8, 12, 16, 20]  # Кончики пальцев (thumb, index, middle, ring, pinky)
    finger_pips = [2, 6, 10, 14, 18]  # Средние суставы
    
    for i, (tip_idx, pip_idx) in enumerate(zip(finger_tips, finger_pips)):
        finger_name = list(finger_states.keys())[i]
        
        # Для большого пальца используем другую логику
        if finger_name == 'thumb':
            # Сравниваем по оси X для большого пальца
            tip_x = hand_landmarks.landmark[tip_idx].x
            pip_x = hand_landmarks.landmark[pip_idx].x
            finger_states[finger_name] = tip_x < pip_x if tip_x < pip_x else tip_x > pip_x
        else:
            # Для остальных пальцев сравниваем по оси Y
            tip_y = hand_landmarks.landmark[tip_idx].y
            pip_y = hand_landmarks.landmark[pip_idx].y
            finger_states[finger_name] = tip_y < pip_y
    
    return finger_states

def count_raised_fingers(finger_states):
    """Подсчитывает количество поднятых пальцев"""
    return sum(1 for finger, raised in finger_states.items() if raised)

def recognize_gesture(finger_states, handedness):
    """
    Распознает жест на основе состояний пальцев.
    handedness: 'Left' или 'Right'
    """
    thumb = finger_states['thumb']
    index = finger_states['index']
    middle = finger_states['middle']
    ring = finger_states['ring']
    pinky = finger_states['pinky']
    
    raised_fingers = count_raised_fingers(finger_states)
    
    # Определение жестов
    if raised_fingers == 5:
        return 'OPEN_PALM'
    elif raised_fingers == 0 and not thumb:
        return 'FIST'
    elif thumb and raised_fingers == 0:
        # Для определения thumbs up/down нужна дополнительная логика
        # (нужно анализировать направление большого пальца)
        return 'THUMBS_UP'  # Упрощенно
    elif index and middle and raised_fingers == 2 and not ring and not pinky:
        return 'VICTORY'
    elif index and raised_fingers == 1 and not thumb and not middle and not ring and not pinky:
        return 'POINTING'
    elif thumb and index and raised_fingers == 2 and not middle and not ring and not pinky:
        # Проверка, что кончики большого и указательного пальцев близко
        return 'OK'
    elif index and pinky and raised_fingers == 2 and not thumb and not middle and not ring:
        return 'ROCK'
    elif pinky and thumb and raised_fingers == 2 and not index and not middle and not ring:
        return 'CALL_ME'
    elif index and pinky and raised_fingers == 2 and not thumb and not middle and not ring:
        # Или альтернативная комбинация для "I love you"
        return 'LOVE'
    
    return None

def calculate_hand_center(hand_landmarks, image_shape):
    """Вычисляет центр ладони"""
    h, w = image_shape[:2]
    
    # Используем точки ладони для вычисления центра
    palm_indices = [0, 1, 5, 9, 13, 17]  # Базовые точки ладони
    x_coords = []
    y_coords = []
    
    for idx in palm_indices:
        x_coords.append(hand_landmarks.landmark[idx].x * w)
        y_coords.append(hand_landmarks.landmark[idx].y * h)
    
    center_x = int(np.mean(x_coords))
    center_y = int(np.mean(y_coords))
    
    return (center_x, center_y)

def calculate_wrist_angle(hand_landmarks, image_shape):
    """Вычисляет угол наклона запястья"""
    h, w = image_shape[:2]
    
    # Точки запястья (0) и среднего пальца (9)
    wrist = np.array([
        hand_landmarks.landmark[0].x * w,
        hand_landmarks.landmark[0].y * h
    ])
    
    middle_base = np.array([
        hand_landmarks.landmark[9].x * w,
        hand_landmarks.landmark[9].y * h
    ])
    
    # Вычисляем угол относительно вертикали
    vector = middle_base - wrist
    angle = np.degrees(np.arctan2(vector[0], -vector[1]))  # Отрицательный Y для инвертированной системы координат
    
    return angle

def draw_hand_info(annotated_frame, hand_landmarks, handedness, gesture, finger_states):
    """Отрисовывает информацию о руке на кадре"""
    h, w = annotated_frame.shape[:2]
    
    # Центр ладони для отображения текста
    center = calculate_hand_center(hand_landmarks, annotated_frame.shape)
    
    # Цвет для разных рук
    if handedness == 'Left':
        color = (255, 0, 0)  # Синий для левой руки
    else:
        color = (0, 255, 0)  # Зеленый для правой руки
    
    # Отображение типа руки
    cv2.putText(annotated_frame, handedness, 
                (center[0] - 30, center[1] - 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    
    # Отображение жеста, если определен
    if gesture:
        cv2.putText(annotated_frame, GESTURES.get(gesture, gesture),
                    (center[0] - 50, center[1] - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    # Отображение количества поднятых пальцев
    raised_count = count_raised_fingers(finger_states)
    cv2.putText(annotated_frame, f"Fingers: {raised_count}",
                (center[0] - 40, center[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    # Отображение угла запястья
    wrist_angle = calculate_wrist_angle(hand_landmarks, annotated_frame.shape)
    cv2.putText(annotated_frame, f"Angle: {wrist_angle:.1f}°",
                (center[0] - 40, center[1] + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
    
    return center

def draw_statistics(frame, num_hands, gestures_detected, display_scale, fps=0):
    """Отображает статистику на кадре"""
    h, w = frame.shape[:2]
    
    # Полупрозрачная панель для статистики
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (350, 150), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)
    
    y_offset = 40
    line_height = 25
    
    # FPS
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    y_offset += line_height
    
    # Количество рук
    cv2.putText(frame, f"Hands: {num_hands}", (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 1)
    y_offset += line_height
    
    # Обнаруженные жесты
    if gestures_detected:
        gestures_text = ", ".join(gestures_detected[:3])  # Первые 3 жеста
        cv2.putText(frame, f"Gestures: {gestures_text}", (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
    else:
        cv2.putText(frame, "Gestures: None", (20, y_offset),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 128, 128), 1)
    y_offset += line_height
    
    # Масштаб отображения
    cv2.putText(frame, f"Scale: {display_scale}%", (20, y_offset),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 1)

def resize_frame(frame, scale_percent=50):
    """Изменяет размер кадра для отображения"""
    if scale_percent == 100:
        return frame
        
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    return cv2.resize(frame, (width, height))

# Основной код
with mp_hands.Hands(**cfg) as hands:
    # Открытие видеофайла
    cap = cv2.VideoCapture("-NSumhkOwSg.mp4")
    
    # Проверка успешного открытия видео
    if not cap.isOpened():
        print("Error: Failed to open video file.")
        exit()
    
    # Получаем параметры видео
    fps_source = cap.get(cv2.CAP_PROP_FPS)
    fps_video = int(fps_source) if fps_source > 0 else 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Для вычисления реального FPS
    frame_count = 0
    start_time = cv2.getTickCount()
    
    display_scale = 50  # Начальный масштаб отображения (50%)
    
    print("=" * 60)
    print("MEDIAPIPE HANDS DETECTOR")
    print("=" * 60)
    print(f"Video resolution: {width}x{height}")
    print(f"Video FPS: {fps_video}")
    print(f"Max hands: {cfg['max_num_hands']}")
    print("=" * 60)
    print("Controls:")
    print("  q / ESC - Quit")
    print("  +/-     - Change window scale")
    print("  l       - Toggle landmarks")
    print("  c       - Toggle connections")
    print("  r       - Reset display settings")
    print("=" * 60)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Video ended.")
            break
        
        frame_count += 1
        annotated_frame = frame.copy()
        
        # Конвертация BGR в RGB для MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False
        
        # Обработка кадра
        results = hands.process(rgb_frame)
        
        num_hands = 0
        all_gestures = []
        hand_centers = []
        
        if results.multi_hand_landmarks and results.multi_handedness:
            num_hands = len(results.multi_hand_landmarks)
            
            for hand_idx, (hand_landmarks, handedness_info) in enumerate(
                zip(results.multi_hand_landmarks, results.multi_handedness)
            ):
                handedness = handedness_info.classification[0].label
                
                # Отрисовка руки, если включено в конфигурации
                if HAND_CONFIG['landmarks']['enabled'] or HAND_CONFIG['connections']['enabled']:
                    mp_drawing.draw_landmarks(
                        annotated_frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        HAND_CONFIG['landmarks']['spec'] if HAND_CONFIG['landmarks']['enabled'] else None,
                        HAND_CONFIG['connections']['spec'] if HAND_CONFIG['connections']['enabled'] else None
                    )
                
                # Анализ пальцев и жестов
                finger_states = calculate_finger_states(hand_landmarks)
                gesture = recognize_gesture(finger_states, handedness)
                
                if gesture:
                    all_gestures.append(gesture)
                
                # Отрисовка информации о руке
                center = draw_hand_info(annotated_frame, hand_landmarks, 
                                        handedness, gesture, finger_states)
                hand_centers.append(center)
        
        # Вычисление реального FPS
        current_time = cv2.getTickCount()
        elapsed_time = (current_time - start_time) / cv2.getTickFrequency()
        current_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
        
        # Отображение статистики
        draw_statistics(annotated_frame, num_hands, all_gestures, display_scale, current_fps)
        
        # Изменение размера для отображения
        display_frame = resize_frame(annotated_frame, display_scale)
        
        # Показ кадра
        cv2.imshow('MediaPipe Hands', display_frame)
        
        # Обработка клавиш управления
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q') or key == 27:  # q или ESC
            print("Exit by user request.")
            break
        elif key == ord('+') or key == ord('='):
            display_scale = min(display_scale + 10, 200)
            print(f"Scale increased to {display_scale}%")
        elif key == ord('-') or key == ord('_'):
            display_scale = max(display_scale - 10, 20)
            print(f"Scale decreased to {display_scale}%")
        elif key == ord('l'):
            HAND_CONFIG['landmarks']['enabled'] = not HAND_CONFIG['landmarks']['enabled']
            status = "enabled" if HAND_CONFIG['landmarks']['enabled'] else "disabled"
            print(f"Landmarks {status}")
        elif key == ord('c'):
            HAND_CONFIG['connections']['enabled'] = not HAND_CONFIG['connections']['enabled']
            status = "enabled" if HAND_CONFIG['connections']['enabled'] else "disabled"
            print(f"Connections {status}")
        elif key == ord('r'):
            HAND_CONFIG['landmarks']['enabled'] = True
            HAND_CONFIG['connections']['enabled'] = True
            display_scale = 50
            print("Display settings reset")
    
    # Освобождение ресурсов
    cap.release()
    cv2.destroyAllWindows()
    
    print("=" * 60)
    print("PROCESSING COMPLETED")
    print(f"Total frames processed: {frame_count}")
    print(f"Average FPS: {frame_count/elapsed_time:.2f}" if elapsed_time > 0 else "Average FPS: N/A")
    print("Resources released.")
    print("=" * 60)