# from ultralytics import YOLO
# import cv2

# model = YOLO("yolo11n.pt", task="detect")

# results = model.track(source="-NSumhkOwSg.mp4", stream=True, persist=True, tracker=".venv/lib/python3.10/site-packages/ultralytics/cfg/trackers/bytetrack.yaml")

# for r in results:
#     '''
#     r - __dir__()
#     [
#         'orig_img',   - (1920, 1080, 3)
#         'orig_shape', - (1920, 1080)
#         'boxes',      - Может быть несколько объектов - boxes[i]
#                     [
#                         'data',        - tensor([[0.0000e+00, 1.8899e+02, 1.0800e+03, 1.9130e+03, 1.0000e+00, 9.3222e-01, 0.0000e+00]]) 
#                         'orig_shape',  - (1920, 1080)
#                         'is_track',    - True
#                         'xyxy',        - tensor([[   0.0000,  188.9927, 1080.0000, 1912.9863]])
#                         'conf',        - tensor([0.9322])
#                         'cls',         - tensor([0.]) (person)
#                         'id',          - tensor([1.])
#                         'xywh',        - tensor([[ 540.0000, 1050.9895, 1080.0000, 1723.9937]])
#                         'xyxyn',       - tensor([[0.0000, 0.0984, 1.0000, 0.9963]])
#                         'xywhn',       - tensor([[0.5000, 0.5474, 1.0000, 0.8979]])
#                         'shape',       - torch.Size([1, 7]) (shape от data)
#                         'cpu', 
#                         'numpy', 
#                         'cuda', 
#                         'to',  
#                     ]
#         'masks', 
#         'probs', 
#         'keypoints',  - None
#         'obb',        - None
#         'speed',      - {'preprocess': 2.78296299984504, 'inference': 59.363701000620495, 'postprocess': 14.278745000410709}
#         'names',    
#                     {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 
#                     6: 'train', 7: 'truck', 8: 'boat', 9: 'traffic light', 10: 'fire hydrant', 
#                     11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 16: 'dog', 
#                     17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 
#                     24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 
#                     30: 'skis', 31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 
#                     35: 'baseball glove', 36: 'skateboard', 37: 'surfboard', 38: 'tennis racket', 
#                     39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 44: 'spoon', 
#                     45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 
#                     51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 
#                     57: 'couch', 58: 'potted plant', 59: 'bed', 60: 'dining table', 61: 'toilet', 
#                     62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 67: 'cell phone', 
#                     68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 
#                     74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'} 
#         'cpu',
#         'numpy', 
#         'cuda',
#         'show',    - func - Выводит изображение
#         'to_df', 
#         'to_csv', 
#         'to_json'
#     ]
#     '''
from ultralytics import YOLO
import cv2
import json
import numpy as np

class VideoTracker:
    def __init__(self, model_path, tracker_config):
        self.model = YOLO(model_path, task="detect")
        self.tracker_config = tracker_config
        
    def draw_detections(self, frame, boxes, names):
        """Отрисовка детекций на кадре"""
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
        
        return frame
    
    def draw_box_with_label(self, frame, x1, y1, x2, y2, label):
        """Рисует bounding box с подписью"""
        color = (0, 255, 0)  # Зеленый
        thickness = 2
        
        # Рисуем bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
        
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
                      display_scale=50, show_window=True):
        """Основной метод обработки видео"""
        
        # Получаем параметры исходного видео
        cap = cv2.VideoCapture(video_path)
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        
        print(f"Начинаем обработку видео: {video_path}")
        print(f"Размер: {width}x{height}, FPS: {fps}")
        print(f"Окно отображения будет уменьшено до {display_scale}%")
        
        # Обработка видео с трекингом
        results = self.model.track(
            source=video_path,
            stream=True,
            persist=True,
            tracker=self.tracker_config
        )
        
        frame_count = 0

        uniq_person_frames = {}
        
        try:
            for r in results:

                for data in r.boxes:
                    cls_id = int(data.cls.item())          # класс объекта
                    if cls_id != 0:                        # 0 = CLASS "person" в COCO
                        continue                           # пропускаем всё кроме людей

                    if data.id is None:
                        continue                           # YOLO мог потерять трек на кадре

                    track_id = int(data.id.item())

                    if track_id not in uniq_person_frames:
                        uniq_person_frames[track_id] = []

                    uniq_person_frames[track_id].append(frame_count)

                # Получаем исходный кадр
                frame = r.orig_img.copy()
                
                # Отрисовываем детекции
                if r.boxes is not None and len(r.boxes) > 0:
                    frame = self.draw_detections(frame, r.boxes, r.names)
                
                # Отображаем кадр в окне (с уменьшенным размером)
                if show_window:
                    display_frame = self.resize_frame(frame, display_scale)
                    cv2.imshow('YOLO Tracking', display_frame)
                    
                    # Выход по нажатию 'q' или ESC
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q') or key == 27:  # 27 = ESC
                        print("\nПрервано пользователем")
                        break
                
                # Выводим прогресс каждые 50 кадров
                if frame_count % 50 == 0:
                    print(f"Обработано кадров: {frame_count}")

                frame_count += 1
        
        except KeyboardInterrupt:
            print("\nОбработка прервана")
        except Exception as e:
            print(f"\nПроизошла ошибка: {e}")
        
        finally:

            with open("unique_person_frames.json", "w") as f:
                json.dump(uniq_person_frames, f, ensure_ascii=False, indent=4)

            if show_window:
                cv2.destroyAllWindows()
            
            print(f"\nОбработка завершена!")
            print(f"Всего обработано кадров: {frame_count}")
            print(f"Видео сохранено как '{output_path}'")


def main():
    # Конфигурация
    MODEL_PATH = "yolo11x.pt"
    TRACKER_CONFIG = ".venv/lib/python3.10/site-packages/ultralytics/cfg/trackers/botsort.yaml"
    VIDEO_PATH = "-NSumhkOwSg.mp4"
    OUTPUT_PATH = "output_video.mp4"
    
    # Размер окна отображения (в процентах от исходного)
    # Можно установить от 10 до 100%
    DISPLAY_SCALE = 50  # 50% от исходного размера
    
    # Создаем трекер и обрабатываем видео
    tracker = VideoTracker(MODEL_PATH, TRACKER_CONFIG)
    tracker.process_video(
        video_path=VIDEO_PATH,
        output_path=OUTPUT_PATH,
        display_scale=DISPLAY_SCALE,
        show_window=True  # Установите False, если не нужно показывать окно
    )


if __name__ == "__main__":
    main()