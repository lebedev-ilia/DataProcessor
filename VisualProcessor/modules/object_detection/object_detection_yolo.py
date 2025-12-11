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
from utils.logger import get_logger

logger = get_logger("ObjectDetectionYOLO")

class ObjectDetectionYOLO:
    def __init__(self, model_path="yolo11x.pt", box_threshold=0.6, batch_size=16):
        import os
        p = os.path.dirname(__file__)

        self.threshold = box_threshold
        self.batch_size = batch_size
        self.model = YOLO(f"{p}/{model_path}")
        self.model.overrides['verbose'] = False
        self.names = self.model.names

    def run(self, frame_manager, frame_indices):
        results = {}

        # Бежим по индексам батчами
        for start in range(0, len(frame_indices), self.batch_size):
            end = start + self.batch_size
            batch_indices = frame_indices[start:end]

            # Загружаем только нужные кадры, не храним весь список
            frames = [frame_manager.get(i) for i in batch_indices]

            # Предсказываем прямо батчом
            preds = self.model.predict(
                frames, 
                stream=False,
                verbose=False
            )

            # Разбираем результаты батча
            for frame_index, pred in zip(batch_indices, preds):
                results[frame_index] = []
                boxes = pred.boxes.cpu()

                for box in boxes:
                    conf = float(box.conf)
                    if conf < self.threshold:
                        continue

                    cls_id = int(box.cls)
                    cls_name = self.names.get(cls_id, str(cls_id))
                    xyxy = box.xyxy.tolist()[0]

                    results[frame_index].append({
                        "class": cls_name,
                        "conf": conf,
                        "box": xyxy
                    })

            logger.info(
                f"ObjectDetectionYOLO | Обработано батчей: {end // self.batch_size}"
            )

        return results
