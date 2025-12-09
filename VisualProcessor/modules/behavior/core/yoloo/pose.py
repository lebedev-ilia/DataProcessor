from ultralytics import YOLO
import cv2
import numpy as np

class VideoTracker:
    def __init__(self, model_path, tracker_config):
        self.model = YOLO(model_path, task="detect")
        self.tracker_config = tracker_config
        
        # COCO ключевые точки (17 точек) и соединения для скелета
        self.skeleton = [
            # Линии туловища
            (16, 14), (14, 12), (17, 15), (15, 13), (12, 13), (6, 12), (7, 13),
            # Линии рук
            (6, 8), (8, 10), (7, 9), (9, 11),
            # Линии ног
            (2, 1), (1, 0), (0, 4), (4, 6), (0, 5), (5, 7),
            # Линии лица
            (2, 3), (3, 4), (5, 3)
        ]
        
        # Цвета для разных частей тела
        self.skeleton_colors = [
            (255, 0, 0),    # Синий: плечи-шея
            (0, 255, 0),    # Зеленый: плечи-таз
            (0, 0, 255),    # Красный: таз-ноги
            (255, 255, 0),  # Голубой: руки
            (255, 0, 255),  # Фиолетовый: ноги
            (0, 255, 255),  # Желтый: лицо
        ]
    
    def draw_detections(self, frame, boxes, names, keypoints=None):
        """Отрисовка детекций и поз на кадре"""
        if boxes is None or len(boxes) == 0:
            return frame
        
        for i in range(len(boxes)):
            # Получаем координаты bounding box
            xyxy = boxes.xyxy[i].cpu().numpy().astype(int)
            x1, y1, x2, y2 = xyxy
            
            # Получаем ID трека (если есть)
            track_id = int(boxes.id[i].item()) if boxes.id is not None else None
            
            # Получаем уверенность и класс
            confidence = float(boxes.conf[i].item())
            cls_id = int(boxes.cls[i].item())
            class_name = names[cls_id]
            
            # Формируем подпись
            label = f"{f'ID:{track_id} ' if track_id is not None else ''}{class_name} {confidence:.2f}"
            
            # Рисуем bounding box и подпись
            self.draw_box_with_label(frame, x1, y1, x2, y2, label)
            
            # Рисуем позу (ключевые точки), если они есть
            if keypoints is not None and i < len(keypoints.xy):
                self.draw_pose(frame, keypoints.xy[i], track_id)
        
        return frame
    
    def draw_pose(self, frame, keypoints, track_id=None):
        """Рисует позу (скелет) по ключевым точкам"""
        if keypoints is None or len(keypoints) == 0:
            return
        
        # Преобразуем тензор в numpy массив
        kpts = keypoints.cpu().numpy() if hasattr(keypoints, 'cpu') else keypoints
        
        # Рисуем соединения (скелет)
        for idx, (start, end) in enumerate(self.skeleton):
            if start < len(kpts) and end < len(kpts):
                start_pt = (int(kpts[start][0]), int(kpts[start][1]))
                end_pt = (int(kpts[end][0]), int(kpts[end][1]))
                
                # Проверяем, что точки валидны (не нулевые)
                if start_pt != (0, 0) and end_pt != (0, 0):
                    # Выбираем цвет в зависимости от группы соединений
                    color_idx = min(idx // 3, len(self.skeleton_colors) - 1)
                    color = self.skeleton_colors[color_idx]
                    
                    # Рисуем линию
                    cv2.line(frame, start_pt, end_pt, color, 2)
        
        # Рисуем ключевые точки
        for i, kpt in enumerate(kpts):
            x, y = int(kpt[0]), int(kpt[1])
            
            # Пропускаем нулевые точки
            if x == 0 and y == 0:
                continue
            
            # Определяем цвет точки в зависимости от ее типа
            if i in [0, 1, 2, 3, 4]:  # Голова
                color = (0, 255, 255)  # Желтый
                radius = 4
            elif i in [5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]:  # Тело
                color = (0, 0, 255)  # Красный
                radius = 3
            else:  # Ноги
                color = (255, 0, 0)  # Синий
                radius = 3
            
            # Рисуем точку
            cv2.circle(frame, (x, y), radius, color, -1)
            
            # Для важных точек можно добавить номер
            if i in [0, 5, 6, 11, 12]:  # Нос, плечи, бедра
                cv2.putText(frame, str(i), (x+5, y-5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Если есть track_id, отображаем его в центре позы
        if track_id is not None:
            # Находим центр всех точек
            valid_points = [kpt for kpt in kpts if not (kpt[0] == 0 and kpt[1] == 0)]
            if valid_points:
                center_x = int(np.mean([p[0] for p in valid_points]))
                center_y = int(np.mean([p[1] for p in valid_points]))
                
                # Рисуем ID
                cv2.putText(frame, f"ID:{track_id}", (center_x-20, center_y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def draw_box_with_label(self, frame, x1, y1, x2, y2, label):
        """Рисует bounding box с подписью"""
        color = (0, 255, 0)  # Зеленый
        thickness = 2
        
        # Рисуем bounding box (полупрозрачный)
        overlay = frame.copy()
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
        
        # Добавляем прозрачность
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
        
        # Рисуем подпись с фоном
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        font_thickness = 1
        
        # Получаем размер текста
        (label_width, label_height), baseline = cv2.getTextSize(
            label, font, font_scale, font_thickness
        )
        
        # Рисуем фон для текста
        cv2.rectangle(
            frame,
            (x1, y1 - label_height - 10),
            (x1 + label_width, y1),
            color,
            -1  # Заливка
        )
        
        # Рисуем текст
        cv2.putText(
            frame,
            label,
            (x1, y1 - 5),
            font,
            font_scale,
            (0, 0, 0),  # Черный текст
            font_thickness
        )
    
    def resize_frame(self, frame, scale_percent=50):
        """Изменяет размер кадра для отображения"""
        if scale_percent == 100:
            return frame
            
        width = int(frame.shape[1] * scale_percent / 100)
        height = int(frame.shape[0] * scale_percent / 100)
        return cv2.resize(frame, (width, height))
    
    def process_video(self, video_path, output_path='output_video.mp4', 
                      display_scale=50, show_window=True, draw_boxes=True):
        """Основной метод обработки видео"""
        
        # Получаем параметры исходного видео
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        # Создаем видеописатель
        video_writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*'mp4v'),
            fps,
            (width, height)
        )
        
        print(f"Начинаем обработку видео: {video_path}")
        print(f"Размер: {width}x{height}, FPS: {fps}")
        print(f"Окно отображения: {display_scale}% от исходного")
        print(f"Отрисовка боксов: {'Включена' if draw_boxes else 'Выключена'}")
        print("-" * 50)
        
        # Обработка видео с трекингом
        results = self.model.track(
            source=video_path,
            stream=True,
            persist=True,
            tracker=self.tracker_config
        )
        
        frame_count = 0
        
        try:
            for r in results:
                frame_count += 1
                
                # Получаем исходный кадр
                frame = r.orig_img.copy()
                
                # Отрисовываем детекции и позы
                if r.boxes is not None and len(r.boxes) > 0:
                    # Если не нужно рисовать боксы, временно убираем их
                    if not draw_boxes:
                        # Сохраняем копию кадра без боксов
                        frame_no_boxes = frame.copy()
                        frame = self.draw_detections(frame, r.boxes, r.names, r.keypoints)
                        # Заменяем кадр с боксами на кадр без боксов, но с позами
                        if r.keypoints is not None:
                            for i in range(len(r.boxes)):
                                if i < len(r.keypoints.xy):
                                    track_id = int(r.boxes.id[i].item()) if r.boxes.id is not None else None
                                    self.draw_pose(frame_no_boxes, r.keypoints.xy[i], track_id)
                        frame = frame_no_boxes
                    else:
                        frame = self.draw_detections(frame, r.boxes, r.names, r.keypoints)
                
                # Сохраняем кадр в видеофайл
                video_writer.write(frame)
                
                # Отображаем кадр в окне
                if show_window:
                    display_frame = self.resize_frame(frame, display_scale)
                    cv2.imshow('YOLO Pose Tracking', display_frame)
                    
                    # Выход по нажатию клавиш
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:  # q или ESC
                        print("\nПрервано пользователем")
                        break
                    elif key == ord('b'):  # Переключение отображения боксов
                        draw_boxes = not draw_boxes
                        print(f"Отрисовка боксов: {'Включена' if draw_boxes else 'Выключена'}")
                    elif key == ord('+'):  # Увеличить окно
                        display_scale = min(display_scale + 10, 100)
                        print(f"Размер окна: {display_scale}%")
                    elif key == ord('-'):  # Уменьшить окно
                        display_scale = max(display_scale - 10, 20)
                        print(f"Размер окна: {display_scale}%")
                
                # Выводим прогресс
                if frame_count % 30 == 0:
                    print(f"Кадр: {frame_count} | Объектов: {len(r.boxes) if r.boxes else 0}")
        
        except KeyboardInterrupt:
            print("\nОбработка прервана")
        except Exception as e:
            print(f"\nОшибка: {e}")
        
        finally:
            # Освобождаем ресурсы
            video_writer.release()
            if show_window:
                cv2.destroyAllWindows()
            
            print("-" * 50)
            print(f"Обработка завершена!")
            print(f"Всего кадров: {frame_count}")
            print(f"Видео сохранено: '{output_path}'")
            print("\nУправление в окне:")
            print("  q / ESC - выход")
            print("  b - переключение боксов")
            print("  + - увеличить окно")
            print("  - - уменьшить окно")


def main():
    # Конфигурация
    MODEL_PATH = "yolo11x-pose.pt"
    TRACKER_CONFIG = ".venv/lib/python3.10/site-packages/ultralytics/cfg/trackers/bytetrack.yaml"
    VIDEO_PATH = "-NSumhkOwSg.mp4"
    OUTPUT_PATH = "output_pose_tracking.mp4"
    
    # Настройки отображения
    DISPLAY_SCALE = 60  # Размер окна (60% от исходного)
    DRAW_BOXES = True   # Отрисовывать bounding boxes
    
    # Создаем трекер и обрабатываем видео
    tracker = VideoTracker(MODEL_PATH, TRACKER_CONFIG)
    tracker.process_video(
        video_path=VIDEO_PATH,
        output_path=OUTPUT_PATH,
        display_scale=DISPLAY_SCALE,
        draw_boxes=DRAW_BOXES
    )


if __name__ == "__main__":
    main()