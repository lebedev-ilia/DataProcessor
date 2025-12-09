import mediapipe as mp
import numpy as np
import cv2

# Инициализация MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

cfg = {
    'static_image_mode': False,
    'model_complexity': 2,  # 0, 1, 2
    'smooth_landmarks': True,
    'min_detection_confidence': 0.5,
    'min_tracking_confidence': 0.5
}

# Используем контекстный менеджер 'with' для корректного управления ресурсами MediaPipe
with mp_pose.Pose(**cfg) as pose:

    def resize_frame(frame, scale_percent=50):
        """Изменяет размер кадра для отображения"""
        if scale_percent == 100:
            return frame
            
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        return cv2.resize(frame, (width, height))

    # Открытие видеофайла
    cap = cv2.VideoCapture("-NSumhkOwSg.mp4")

    # Проверка успешного открытия видео
    if not cap.isOpened():
        print("Ошибка: не удалось открыть видеофайл.")
        exit()

    def _landmarks_to_array(landmarks, image_shape) -> np.ndarray:
        """Преобразование landmarks в numpy массив"""
        h, w, _ = image_shape
        points = []
        for lm in landmarks.landmark:
            points.append([lm.x * w, lm.y * h, lm.z])
        return np.array(points)

    def _calculate_joint_angles(landmarks, image_shape):
        """Вычисление углов суставов"""
        angles = {}
        h, w = image_shape[:2]

        # Преобразование нормализованных координат в пиксельные
        def get_coord(idx):
            lm = landmarks.landmark[idx]
            return np.array([lm.x * w, lm.y * h])

        # Угол левого локтя (плечо-локоть-запястье)
        try:
            shoulder = get_coord(11)  # Левое плечо
            elbow = get_coord(13)     # Левый локоть
            wrist = get_coord(15)     # Левое запястье

            v1 = shoulder - elbow
            v2 = wrist - elbow
            # Добавляем малое значение (1e-8) для предотвращения деления на ноль
            angle = np.degrees(np.arccos(
                np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            ))
            angles['left_elbow_angle'] = float(angle)
        except Exception as e:
            #print(f"Ошибка при расчете левого локтя: {e}") # Раскомментировать для отладки
            angles['left_elbow_angle'] = 0.0

        # Угол правого локтя
        try:
            shoulder = get_coord(12)  # Правое плечо
            elbow = get_coord(14)     # Правый локоть
            wrist = get_coord(16)     # Правое запястье

            v1 = shoulder - elbow
            v2 = wrist - elbow
            angle = np.degrees(np.arccos(
                np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
            ))
            angles['right_elbow_angle'] = float(angle)
        except Exception as e:
            #print(f"Ошибка при расчете правого локтя: {e}")
            angles['right_elbow_angle'] = 0.0

        return angles

    def _draw_joint_angles(image, angles, landmarks):
        """Отрисовка углов суставов на изображении"""
        h, w = image.shape[:2]

        for angle_name, angle_value in angles.items():
            # Получаем координаты сустава для отображения угла
            if 'left_elbow' in angle_name:
                joint_idx = 13
            elif 'right_elbow' in angle_name:
                joint_idx = 14
            else:
                continue

            joint = landmarks.landmark[joint_idx]
            x, y = int(joint.x * w), int(joint.y * h)

            # Отображение угла
            cv2.putText(image, f"{angle_value:.1f}°",
                        (x + 20, y - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)

    # --- ОСНОВНОЙ ЦИКЛ ОБРАБОТКИ ВИДЕО ---
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:  # Если кадры закончились (или ошибка чтения)
            print("Видео закончилось или ошибка чтения.")
            break

        annotated_frame = frame.copy()

        # Конвертация BGR в RGB для MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb_frame.flags.writeable = False  # Для оптимизации (только чтение)

        pose_results = pose.process(rgb_frame)

        if pose_results.pose_landmarks:
            # 1. Отрисовка скелета MediaPipe
            mp_drawing.draw_landmarks(
                annotated_frame,
                pose_results.pose_landmarks,
                mp_pose.POSE_CONNECTIONS,
                landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
            )

            # 2. Вычисление и отрисовка углов
            angles = _calculate_joint_angles(pose_results.pose_landmarks, annotated_frame.shape)
            _draw_joint_angles(annotated_frame, angles, pose_results.pose_landmarks)

            # (Опционально) Вывод углов в консоль
            # print(f"Левый локоть: {angles.get('left_elbow_angle', 0):.1f}°, Правый локоть: {angles.get('right_elbow_angle', 0):.1f}°")

        annotated_frame = resize_frame(annotated_frame)

        # --- КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: ОТОБРАЖЕНИЕ КАДРА ---
        # Показываем обработанный кадр в окне с именем 'MediaPipe Pose'
        cv2.imshow('MediaPipe Pose', annotated_frame)

        # --- КЛЮЧЕВОЕ ИСПРАВЛЕНИЕ: УПРАВЛЕНИЕ ЦИКЛОМ И ВЫХОД ---
        # cv2.waitKey(1) ждет 1 миллисекунду и возвращает код нажатой клавиши
        # & 0xFF нужен для получения ASCII кода клавиши
        # Если нажата клавиша 'q' (quit) или ESC (27), выходим из цикла
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 27 - это код клавиши ESC
            print("Выход по запросу пользователя.")
            break

    # --- КОРРЕКТНОЕ ОСВОБОЖДЕНИЕ РЕСУРСОВ ---
    cap.release()  # Закрываем видеофайл
    cv2.destroyAllWindows()  # Закрываем все открытые окна OpenCV
    print("Обработка завершена. Ресурсы освобождены.")