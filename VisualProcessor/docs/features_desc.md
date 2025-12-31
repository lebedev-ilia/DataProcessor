## object_detection

### Модели:

#### Вариант 1: YOLO (если категории объектов неизвестны)

```py
from ultralytics import YOLO
model = YOLO("yolo11x.pt")
```

#### Вариант 2: OWL-ViT / OWLv2 (open-vocabulary detection по текстовым запросам)

```py
from transformers import (
    OwlViTProcessor,
    OwlViTForObjectDetection,
    Owlv2Processor,
    Owlv2ForObjectDetection,
)

# OWL-ViT
processor = OwlViTProcessor.from_pretrained("google/owlvit-base-patch16")
model = OwlViTForObjectDetection.from_pretrained("google/owlvit-base-patch16")

# или OWLv2
processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16")
```

### Выход:

#### Вариант 1: YOLO

```json
{
    "0": [
        {
            "class": "person",
            "conf": 0.9593384861946106,
            "box": [43.025848388671875, 180.14501953125, 1077.9505615234375, 1914.7108154296875]
        }
    ],
    "3": [...],
    "6": [...]
}
```

#### Вариант 2: OWL-ViT / OWLv2

```json
{
    "frames": {
        "0": [
            {
                "bbox": [x_min, y_min, x_max, y_max],
                "score": 0.85,
                "label": "person",
                "color": {"B": 128, "G": 150, "R": 200},
                "semantic_tags": ["luxury", "danger"]
            }
        ]
    },
    "summary": {
        "total_detections": 42,
        "unique_categories": 5,
        "category_counts": {
            "person": 20,
            "car": 15,
            "bicycle": 7
        },
        "semantic_tag_counts": {
            "luxury": 5,
            "danger": 2
        },
        "brand_detections": [
            {
                "frame": 10,
                "brand": "nike logo",
                "score": 0.92,
                "bbox": [x_min, y_min, x_max, y_max]
            }
        ]
    },
    "frame_count": 11
}
```

### Фичи:

#### Вариант 1: YOLO

#### 1.1. - 1.3. class, conf, box

```py
def run(self, frame_manager, frame_indices):
    results = {}
    for start in range(0, len(frame_indices), self.batch_size):
        end = start + self.batch_size
        batch_indices = frame_indices[start:end]
        frames = [frame_manager.get(i) for i in batch_indices]
        
        preds = self.model.predict(frames, stream=False, verbose=False)
        
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
    
    return results
```

#### Вариант 2: OWL-ViT / OWLv2

#### 2.1. frames - детекции по кадрам

```py
def _detect_objects_in_frame(self, frame, text_queries=None):
    # Конвертация BGR -> RGB -> PIL
    if frame.ndim == 3:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
    
    h, w = rgb.shape[:2]
    image = Image.fromarray(rgb)
    
    # Подготовка запросов
    queries = self._prepare_text_queries(text_queries)
    
    # Обработка через processor и model
    inputs = self._processor(text=queries, images=image, return_tensors="pt").to(self.device)
    outputs = self._model(**inputs)
    
    # Post-processing
    target_sizes = torch.tensor([[h, w]], dtype=torch.long).to(self.device)
    results = self._processor.post_process_grounded_object_detection(
        outputs=outputs,
        target_sizes=target_sizes,
        threshold=self.box_threshold
    )
    
    # Извлечение детекций
    detections = []
    result = results[0]
    boxes = result.get("boxes", torch.tensor([])).cpu().numpy()
    scores = result.get("scores", torch.tensor([])).cpu().numpy()
    labels = result.get("labels", torch.tensor([])).cpu().numpy()
    
    for box, score, label_idx in zip(boxes, scores, labels):
        if float(score) < float(self.box_threshold):
            continue
        
        label_name = queries[int(label_idx)] if 0 <= int(label_idx) < len(queries) else f"class_{label_idx}"
        clamped = self._clamp_bbox(box.tolist(), width=w, height=h)
        
        det = {
            "bbox": [float(x_min), float(y_min), float(x_max), float(y_max)],
            "score": float(score),
            "label": label_name,
        }
        detections.append(det)
    
    return detections
```

#### 2.2. semantic_tags

```py
def _detect_semantic_tags(self, detections, frame):
    semantic_queries = {
        "luxury": ["luxury", "expensive", "premium", "luxury car", "luxury watch"],
        "danger": ["knife", "gun", "weapon", "dangerous"],
        "cute": ["cute", "adorable", "puppy", "kitten", "teddy"],
        "sport": ["sports", "athletic", "fitness", "equipment"],
        "food": ["food", "meal", "dish", "cuisine"],
        "technology": ["electronic device", "gadget", "smartphone", "laptop"]
    }
    
    for det in detections:
        tags = []
        label_lower = str(det.get("label", "")).lower()
        
        # Проверка по семантическим запросам
        for semantic, keywords in semantic_queries.items():
            if any(keyword in label_lower for keyword in keywords):
                tags.append(semantic)
        
        # Эвристики
        if any(w in label_lower for w in ["car", "vehicle", "motorcycle"]) and det.get("score", 0.0) >= 0.75:
            tags.append("luxury")
        if any(w in label_lower for w in ["knife", "gun", "weapon", "blade"]):
            tags.append("danger")
        if any(w in label_lower for w in ["cat", "dog", "puppy", "kitten", "teddy"]):
            tags.append("cute")
        
        det["semantic_tags"] = tags
    
    return detections
```

#### 2.3. color (доминирующий цвет объекта)

```py
def _extract_color_from_bbox(self, frame, bbox, k=3):
    x_min, y_min, x_max, y_max = [int(round(coord)) for coord in bbox]
    x_min = max(0, x_min)
    y_min = max(0, y_min)
    x_max = min(frame.shape[1], x_max)
    y_max = min(frame.shape[0], y_max)
    
    roi = frame[y_min:y_max, x_min:x_max]
    pixels = roi.reshape(-1, 3).astype(np.float32)
    
    # Средний цвет (fallback, если нет sklearn KMeans)
    mean_color = np.mean(pixels, axis=0).astype(int)
    return (int(mean_color[0]), int(mean_color[1]), int(mean_color[2]))  # BGR
```

#### 2.4. summary

```py
# Сбор статистики по всем кадрам
object_counts = {}
total_detections = 0
semantic_tag_counts = defaultdict(int)
brand_detections = []

for frame_idx, detections in all_detections.items():
    for det in detections:
        label = det.get("label", "unknown")
        object_counts[label] = object_counts.get(label, 0) + 1
        total_detections += 1
        
        for tag in det.get("semantic_tags", []) or []:
            semantic_tag_counts[tag] += 1
        
        if self.enable_brand_detection and "logo" in label.lower():
            brand_detections.append({
                "frame": frame_idx,
                "brand": label,
                "score": det.get("score", 0.0),
                "bbox": det.get("bbox", [])
            })

summary = {
    "total_detections": total_detections,
    "unique_categories": len(object_counts),
    "category_counts": object_counts,
    "semantic_tag_counts": dict(semantic_tag_counts),
    "brand_detections": brand_detections
}
```

## scene_classification

### Модели:

```py
# Places365 модели (ResNet18/ResNet50)
model = models.resnet18(pretrained=False)
# Загрузка весов Places365
checkpoint = torch.load("resnet18_places365.pth.tar", map_location=device)
state_dict = {k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()}
model.load_state_dict(state_dict)

# Современные архитектуры через timm
import timm
model = timm.create_model("efficientnet_b0", pretrained=True, num_classes=365)

# CLIP для семантических фичей (опционально)
from transformers import CLIPProcessor, CLIPModel
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Категории Places365
# categories_places365.txt - файл с 365 категориями сцен
```

### Выход:

```json
{
    "aquarium_5": {
        "indices": [0, 1, 2, 3, 4, 5],
        "start_frame": 0,
        "end_frame": 5,
        "length_frames": 6,
        "length_seconds": 0.2,
        "mean_score": 0.9139,
        "class_entropy_mean": 1.23,
        "top1_prob_mean": 0.91,
        "top1_vs_top2_gap_mean": 0.45,
        "fraction_high_confidence_frames": 0.83,
        "mean_indoor": 1.0,
        "mean_outdoor": 0.0,
        "mean_nature": 0.0,
        "mean_urban": 1.0,
        "mean_morning": 0.23,
        "mean_day": 0.27,
        "mean_evening": 0.28,
        "mean_night": 0.21,
        "time_of_day_probs": {
            "morning": 0.23,
            "day": 0.27,
            "evening": 0.28,
            "night": 0.21
        },
        "time_of_day_top": "evening",
        "time_of_day_confidence": 0.28,
        "mean_aesthetic_score": 0.60,
        "aesthetic_std": 0.07,
        "aesthetic_frac_high": 0.35,
        "mean_luxury_score": 0.68,
        "mean_cozy": 0.10,
        "mean_scary": 0.08,
        "mean_epic": 0.06,
        "mean_neutral": 0.75,
        "atmosphere_entropy": 1.12,
        "mean_openness": 0.69,
        "mean_clutter": 0.08,
        "mean_depth_cues": 0.26,
        "scene_change_score": 0.05,
        "label_stability": 0.95,
        "dominant_places_topk_ids": [5, 11, 42],
        "dominant_places_topk_probs": [3.2, 1.1, 0.5]
    },
    "trail_park_11": {
        "indices": [7, 8, 9, 10, 11],
        ...
    },
}
```

### Фичи:

#### 1. Базовые фичи сцены

#### 1.1. indices, start_frame, end_frame, length_frames, length_seconds
- **indices** — список индексов кадров, принадлежащих сцене;
- **start_frame / end_frame** — границы сцены в кадрах;
- **length_frames** — длина сцены в кадрах;
- **length_seconds** — длина сцены в секундах с учётом `fps`.

Сцены определяются как последовательные кадры с одинаковым предсказанным лейблом Places365, далее фильтруются по минимальной длительности в секундах (`min_scene_seconds` или эквивалент через `min_scene_length_frames / fps`).

#### 1.2. mean_score, class_entropy_mean, top1_prob_mean, top1_vs_top2_gap_mean, fraction_high_confidence_frames
Характеризуют уверенность модели в классификации сцены:

- **mean_score** — средний top‑1 score по кадрам;
- **class_entropy_mean** — средняя энтропия распределения по 365 классам;
- **top1_prob_mean** — средняя вероятность top‑1 класса;
- **top1_vs_top2_gap_mean** — средний разрыв между top‑1 и top‑2 вероятностями;
- **fraction_high_confidence_frames** — доля кадров, где top‑1 > 0.7.

```py
# Для одного кадра:
logits = model(tensor)  # Places365 модель
probs = F.softmax(logits, dim=1)

result_for_frame = []
for class_idx, prob in enumerate(probs):
    label = (
        categories[class_idx]
        if 0 <= class_idx < len(categories)
        else f"class_{class_idx}"
    )
    result_for_frame.append({
        "label": label,
        "score": float(prob)
    })

# Агрегация по сцене:
mean_score = np.mean([pred["score"] for pred in scene_predictions])
```

#### 2. Indoor/Outdoor классификация

#### 2.1. - 2.2. mean_indoor, mean_outdoor

```py
# categories - places365.txt

logits = model(tensor)
probs = F.softmax(logits, dim=1)

result_for_frame = []

for class_idx, prob in enumerate(probs): 
    # вернет классы и вероятности, можно поставить top_k или как далее будет браться лучшее предсказанием 
    label = (
        categories[class_idx]
        if 0 <= class_idx < len(categories)
        else f"class_{class_idx}"
    )
    result_for_frame.append({
        "label": label,
        "score": float(prob)
    })
```

```py
def _classify_indoor_outdoor(self, scene_label: str) -> Dict[str, float]:
    indoor_keywords = [
        "room", "bedroom", "kitchen", "bathroom", "living",
        "dining", "office", "hall", "corridor", "staircase",
        "attic", "basement", "garage", "shop", "store", "mall",
        "restaurant", "cafe", "bar", "pub", "hospital", "school",
        "classroom", "library", "museum", "theater", "cinema", 
        "gym", "stadium", "airport", "station", "subway",
        "train", "bus", "indoor"
    ]
    outdoor_keywords = [
        "outdoor", "street", "road", "highway", "bridge",
        "park", "garden", "forest", "beach", "mountain", 
        "desert", "field", "farm", "lake", "river", "ocean", 
        "sea", "sky", "cloud", "sunset", "sunrise", "outdoor"
    ]
    
    label_lower = scene_label.lower()
    indoor_score = sum(1 for keyword in indoor_keywords if keyword in label_lower)
    outdoor_score = sum(1 for keyword in outdoor_keywords if keyword in label_lower)
    
    total = indoor_score + outdoor_score
    if total == 0:
        if any(word in label_lower for word in ["room", "hall", "indoor"]):
            return {"indoor": 0.7, "outdoor": 0.3}
        else:
            return {"indoor": 0.5, "outdoor": 0.5}
    
    indoor_prob = indoor_score / total
    outdoor_prob = outdoor_score / total
    return {"indoor": indoor_prob, "outdoor": outdoor_prob}
```

#### 3. Nature/Urban классификация

#### 3.1. - 3.2. mean_nature, mean_urban

```py
def _classify_nature_urban(self, scene_label: str) -> Dict[str, float]:
    nature_keywords = [
        "forest", "jungle", "wood", "tree", 
        "beach", "coast", "shore", "mountain",
        "hill", "valley", "desert", "field", 
        "meadow", "grass", "flower", "garden",
        "park", "lake", "river", "stream", 
        "waterfall", "ocean", "sea", "island",
        "cave", "canyon", "cliff", "rock", "snow", 
        "ice", "sky", "cloud", "sunset", "sunrise", 
        "nature", "wild", "natural"
    ]
    urban_keywords = [
        "city", "urban", "street", "road", 
        "avenue", "boulevard", "alley", 
        "plaza", "square", "building", 
        "skyscraper", "tower", "bridge", 
        "highway", "subway", "station", 
        "airport", "mall", "shop", "store", 
        "restaurant", "cafe", "bar", "hotel", 
        "office", "factory", "warehouse", 
        "parking", "lot", "urban"
    ]
    
    label_lower = scene_label.lower()
    nature_score = sum(1 for keyword in nature_keywords if keyword in label_lower)
    urban_score = sum(1 for keyword in urban_keywords if keyword in label_lower)
    
    total = nature_score + urban_score
    if total == 0:
        return {"nature": 0.5, "urban": 0.5}
    
    nature_prob = nature_score / total
    urban_prob = urban_score / total
    return {"nature": nature_prob, "urban": urban_prob}
```

#### 4. Time of Day Detection

#### 4.1. - 4.4. mean_morning, mean_day, mean_evening, mean_night, time_of_day_probs, time_of_day_top, time_of_day_confidence

Для одного кадра:

```py
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
mean_brightness = np.mean(gray) / 255.0
hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
hue = hsv[:, :, 0].flatten()
warm_pixels = np.sum((hue < 30) | (hue > 150))
warm_ratio = warm_pixels / len(hue) if len(hue) > 0 else 0
morning_score = mean_brightness * 0.6 * (1 + warm_ratio * 0.5)
day_score = mean_brightness * (1 - warm_ratio * 0.3)
evening_score = (1 - mean_brightness) * 0.7 * (1 + warm_ratio * 0.8)
night_score = (1 - mean_brightness) * (1 - warm_ratio * 0.5)
total = morning_score + day_score + evening_score + night_score
if total == 0:
    return {"morning": 0.25, "day": 0.25, "evening": 0.25, "night": 0.25}
return {
    "morning": morning_score / total,
    "day": day_score / total,
    "evening": evening_score / total,
    "night": night_score / total
}
```

#### 5. Aesthetic Score

#### 5.1. mean_aesthetic_score, aesthetic_std, aesthetic_frac_high

Оценка эстетической привлекательности сцены. Если включена CLIP модель, используется zero-shot классификация. Иначе - эвристика на основе sharpness, contrast, colorfulness и brightness.

```py
# Если включена CLIP модель:
def _calculate_aesthetic_score_clip(self, frame: np.ndarray) -> float:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    texts = [
        "aesthetic beautiful scene",
        "professional photography",
        "ugly unappealing scene",
        "amateur photography"
    ]
    inputs = self._clip_processor(text=texts, images=image, return_tensors="pt", padding=True).to(self.device)
    with torch.no_grad():
        outputs = self._clip_model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
    aesthetic_score = (probs[0][0] + probs[0][1]).item()  # Сумма положительных классов
    return float(aesthetic_score)

# Если не включена CLIP (эвристика):
def _calculate_aesthetic_score_heuristic(self, frame: np.ndarray, scene_label: str) -> float:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # Sharpness (Laplacian variance)
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    sharpness_score = min(1.0, laplacian_var / 500.0)
    # Contrast
    contrast = np.std(gray) / 255.0
    # Colorfulness
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    rgb_flat = rgb.reshape(-1, 3)
    std_r = np.std(rgb_flat[:, 0])
    std_g = np.std(rgb_flat[:, 1])
    std_b = np.std(rgb_flat[:, 2])
    colorfulness = (std_r + std_g + std_b) / 3.0 / 255.0
    # Brightness balance
    mean_brightness = np.mean(gray) / 255.0
    brightness_score = 1.0 - abs(mean_brightness - 0.5) * 2.0
    # Комбинация метрик
    aesthetic = (sharpness_score * 0.3 + contrast * 0.3 + colorfulness * 0.2 + brightness_score * 0.2)
    return float(np.clip(aesthetic, 0.0, 1.0))
```

#### 6. Luxury Score

#### 6.1. mean_luxury_score

Оценка "роскошности" сцены. Если включена CLIP модель, используется zero-shot классификация. Иначе - эвристика на основе лейбла сцены, качества изображения и цветового богатства.

```py
# Если включена CLIP модель:
def _calculate_luxury_score_clip(self, frame: np.ndarray) -> float:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    texts = [
        "luxury expensive high-end scene",
        "premium elegant sophisticated",
        "cheap low-quality scene",
        "budget affordable scene"
    ]
    inputs = self._clip_processor(text=texts, images=image, return_tensors="pt", padding=True).to(self.device)
    with torch.no_grad():
        outputs = self._clip_model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
    luxury_score = (probs[0][0] + probs[0][1]).item()
    return float(luxury_score)

# Если не включена CLIP (эвристика):
def _calculate_luxury_score_heuristic(self, frame: np.ndarray, scene_label: str) -> float:
    label_lower = scene_label.lower()
    luxury_keywords = ["luxury", "premium", "elegant", "sophisticated", "high-end", "expensive"]
    label_score = 0.3 if any(kw in label_lower for kw in luxury_keywords) else 0.0
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    quality_score = min(1.0, sharpness / 500.0) * 0.4
    
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    std_colors = np.std(rgb.reshape(-1, 3), axis=0)
    color_score = np.mean(std_colors) / 255.0 * 0.3
    
    return float(np.clip(label_score + quality_score + color_score, 0.0, 1.0))
```

#### 7. Atmosphere Sentiment

#### 7.1. - 7.4. mean_cozy, mean_scary, mean_epic, mean_neutral, atmosphere_entropy

Оценка атмосферы сцены (уютная, страшная, эпическая, нейтральная). Если включена CLIP модель, используется zero-shot классификация. Иначе - эвристика на основе яркости, контраста и цветовой температуры.

```py
# Если включена CLIP модель:
def _detect_atmosphere_clip(self, frame: np.ndarray) -> Dict[str, float]:
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(rgb)
    texts = [
        "cozy warm comfortable scene",
        "scary frightening dark scene",
        "epic grand majestic scene",
        "neutral ordinary scene"
    ]
    inputs = self._clip_processor(text=texts, images=image, return_tensors="pt", padding=True).to(self.device)
    with torch.no_grad():
        outputs = self._clip_model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
    return {
        "cozy": float(probs[0][0].item()),
        "scary": float(probs[0][1].item()),
        "epic": float(probs[0][2].item()),
        "neutral": float(probs[0][3].item())
    }

# Если не включена CLIP (эвристика):
def _detect_atmosphere_heuristic(self, frame: np.ndarray) -> Dict[str, float]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean_brightness = np.mean(gray) / 255.0
    
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0].flatten()
    warm_ratio = np.sum((hue < 30) | (hue > 150)) / len(hue) if len(hue) > 0 else 0
    
    cozy_score = mean_brightness * 0.6 * (1 + warm_ratio * 0.5)
    contrast = np.std(gray) / 255.0
    scary_score = (1 - mean_brightness) * 0.7 * (1 + contrast * 0.5)
    dynamic_range = (np.max(gray) - np.min(gray)) / 255.0
    epic_score = mean_brightness * 0.8 * (1 + dynamic_range * 0.3)
    
    total = cozy_score + scary_score + epic_score
    if total == 0:
        return {"cozy": 0.33, "scary": 0.33, "epic": 0.34, "neutral": 0.0}
    
    return {
        "cozy": float(cozy_score / total),
        "scary": float(scary_score / total),
        "epic": float(epic_score / total),
        "neutral": float(1.0 - (cozy_score + scary_score + epic_score) / total)
    }
```

#### 8. Geometric Features

#### 8.1. - 8.3. mean_openness, mean_clutter, mean_depth_cues

Геометрические характеристики сцены: открытость (видимость неба/горизонта), загруженность (визуальная сложность) и признаки глубины.

```py
def _calculate_geometric_features(self, frame: np.ndarray) -> Dict[str, float]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    height, width = gray.shape
    
    # Openness: анализ верхней части изображения (небо/горизонт)
    top_portion = gray[:height//3, :]
    top_brightness = np.mean(top_portion) / 255.0
    openness = top_brightness * 0.6 + (1 - np.std(gray) / 255.0) * 0.4
    
    # Clutter: плотность краев (визуальная сложность)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (height * width)
    clutter = min(1.0, edge_density * 2.0)
    
    # Depth cues: анализ градиентов (сильные градиенты указывают на глубину)
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    depth_cues = min(1.0, np.mean(gradient_magnitude) / 100.0)
    
    return {
        "openness": float(np.clip(openness, 0.0, 1.0)),
        "clutter": float(np.clip(clutter, 0.0, 1.0)),
        "depth_cues": float(np.clip(depth_cues, 0.0, 1.0))
    }
```

#### 9. Агрегация по сценам

Все фичи с префиксом "mean_" вычисляются как среднее значение соответствующей фичи по всем кадрам, принадлежащим одной сцене. Сцены определяются как последовательные кадры с одинаковым предсказанным лейблом, после чего длина сцены переводится в секунды (`length_seconds`) с использованием `fps`, применяется fps‑aware порог `min_scene_seconds`, а также считаются дополнительные робастные агрегаты (энтропии, gaps, доли high‑confidence кадров, метрики стабильности лейблов).

```py
def aggregate_scenes(self, res, min_scene_length: int = 30):
    # Группировка последовательных кадров с одинаковым лейблом
    # Вычисление средних значений для каждой фичи по кадрам сцены
    # Возврат агрегированных данных с префиксом "mean_"
```
        
## face_detection

### Модели:

```py
from insightface.app import FaceAnalysis

# Автоматический выбор GPU/CPU
try:
    app = FaceAnalysis(providers=["CUDAExecutionProvider"])
    app.prepare(ctx_id=0, det_size=(640, 640))
except Exception:
    app = FaceAnalysis(providers=["CPUExecutionProvider"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
```

### Выход:

```json
{
    "frames_with_face": [0, 5, 12, 15, ...]
}
```

### Фичи:

#### 1. frames_with_face

Список индексов кадров, в которых обнаружено хотя бы одно лицо с confidence score >= detect_thr (по умолчанию 0.3). Для каждого кадра проверяется наличие лиц через FaceAnalysis, выбирается максимальный score среди всех обнаруженных лиц в кадре.

```py
class FaceDetector:
    def __init__(self, detect_thr: float = 0.3, det_size=(640, 640)):
        self.detect_thr = detect_thr
        self.face_app = self.init_face_app(det_size)
    
    def init_face_app(self, det_size=(640, 640)):
        from insightface.app import FaceAnalysis
        det_size = tuple(int(x) for x in det_size)
        
        # Попытка использовать GPU
        try:
            app = FaceAnalysis(providers=["CUDAExecutionProvider"])
            app.prepare(ctx_id=0, det_size=det_size)
            logger.info("GPU Detector")
        except Exception:
            # Fallback на CPU
            app = FaceAnalysis(providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=-1, det_size=det_size)
        return app
    
    def safe_det_score(self, face) -> float:
        """Безопасное извлечение confidence score из объекта face."""
        return float(getattr(face, "det_score", getattr(face, "score", 0.0) or 0.0))
    
    def detect_face(self, frame_bgr: np.ndarray, face_app, thr: float = 0.5) -> bool:
        """Определяет наличие лиц в кадре. Возвращает True если хотя бы одно лицо >= thr."""
        faces = face_app.get(frame_bgr)
        if not faces:
            return False
        best = max(self.safe_det_score(f) for f in faces)
        return best >= thr
    
    def run(self, frame_manager, frame_indices):
        """Сканирует видео на наличие лиц. Возвращает список индексов кадров с лицами."""
        timeline = {"frames_with_face": []}
        
        for idx in frame_indices:
            frame = frame_manager.get(idx)
            if self.detect_face(frame, self.face_app, thr=self.detect_thr):
                timeline["frames_with_face"].append(idx)
        
        return timeline
```

## detalize_face_modules

### Модели:

```py
import mediapipe as mp

_FACE_MESH = mp.solutions.face_mesh
face_mesh = _FACE_MESH.FaceMesh(
    max_num_faces=10,
    refine_landmarks=True,  # Использование 468 точек landmarks вместо 468
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7,
)
```

Модуль использует модульную архитектуру для извлечения различных типов фич лица:
- GeometryModule: геометрические фичи (bbox, размер, позиция, морфометрия, сжатые векторы формы)
- PoseModule: поза головы (yaw, pitch, roll, стабильность, внимание, нормализованные векторы)
- QualityModule: качество изображения (объединенные метрики: face_sharpness, face_noise_level, face_exposure_score, улучшенный occlusion_proxy)
- LightingModule: освещение (яркость, равномерность, баланс белого; skin_tone_index помечен как audit-only)
- SkinModule: кожа (макияж с осторожностью, гладкость, борода/усы, брови; удален skin_defect_score)
- AccessoriesModule: аксессуары (очки, маска, шапка, серьги)
- EyesModule: глаза (открытие, улучшенное моргание с hysteresis, взгляд, радужка; удален eye_redness_prob)
- MotionModule: движение (скорость лица, микро-выражения, движение рта/челюсти)
- StructureModule: структура (сжатые векторы формы, privacy-preserving identity, выражение, симметрия)
- ProfessionalModule: профессиональные фичи (качество, усталость, вовлеченность; удален perceived_attractiveness_score)
- LipReadingModule: чтение по губам (форма рта, фонемы опционально, активность речи)
- Face3DModule: 3D реконструкция (3D mesh сжатый, privacy-preserving identity shape, expression vector)

**Обновления:**
- Добавлены confidence метрики (detection_confidence, tracking_id) для каждого лица
- Поддержка primary face (наибольшая bbox_area) и мульти-лица
- Temporal history с фиксированными окнами (30 кадров ≈ 1-1.5 сек)
- Сжатие landmark-векторов (PCA/projection до 8-16 dims)
- Privacy-preserving для identity_shape_vector (хеширование)
- Удалены чувствительные метрики (perceived_attractiveness_score, skin_defect_score, eye_redness_prob)
- Объединены дублирующиеся метрики качества
- Нормализация всех метрик в стандартизованную шкалу [0..1]
- Компактный набор фичей для VisualTransformer (~48 dims) доступен через extract_compact_features()
    
### Выход:
```
{
    "0": [
            {
                "frame_index": 0,
                "face_index": 0,
                "bbox": [399.2803649902344, 415.4458312988281, 835.0166015625, 874.3395385742188],
                "detection_confidence": 0.9,
                "tracking_id": 1,
                "is_primary_face": true,
                "geometry": {
                    "face_bbox_area": 199956.61699487362,
                    "face_relative_size": 0.09642969569578376,
                    "face_box_ratio": 0.9495363080033962,
                    "face_bbox_position": {"cx": 617.1484832763672, "cy": 644.8926849365234},
                    "face_dist_from_center": 0.16896567724351552,
                    "face_rotation_in_frame": -8.832769393920898,
                    "aspect_ratio_stability": 0.0,
                    "jaw_width": 330.4817810058594,
                    "cheekbone_width": 432.2557373046875,
                    "cheek_width": 424.3172912597656,
                    "forehead_height": 166.32730102539062,
                    "face_shape_vector": [630.7454223632812, ...and 15 values... 432.874267578125],  # Сжато до 16 dims
                    "face_center_x_norm": 0.5,
                    "face_center_y_norm": 0.5,
                    "jaw_width_norm": 0.8,
                    "cheekbone_width_norm": 1.0,
                    "cheek_width_norm": 0.98,
                    "forehead_height_norm": 0.4,
                    "landmark_stability": 0.9
                },
                "pose": {
                    "yaw": 6.300705890788989,
                    "pitch": -17.35517438503278,
                    "roll": -8.068668365478516,
                    "yaw_norm": 0.07,
                    "pitch_norm": -0.19,
                    "roll_norm": -0.09,
                    "head_pose_variability": 0.0,
                    "pose_stability_score": 0.8599843135380225,
                    "head_turn_frequency": 0.0,
                    "attention_to_camera_ratio": 0.9384570096135878,
                    "looking_direction_vector": [0.97, -0.14, 0.24],  # Нормализованный unit-вектор
                    "pose_conf": 0.9,
                    "landmark_conf": 0.9,
                    "tracking_conf": 0.9
                },
                "quality": {
                    "face_sharpness": 0.04,  # Объединенная метрика
                    "face_noise_level": 0.038173782825469973,
                    "face_exposure_score": 0.85,
                    "occlusion_proxy": 0.1,  # Улучшенная оценка окклюзии
                    "quality_proxy_score": 0.75,
                    "quality_confidence": 0.9,
                    # Обратная совместимость (deprecated)
                    "face_blur_score": 0.06484930049356688,
                    "sharpness_score": 0.00678416907787323,
                    "noise_level": 0.038173782825469973,
                    "face_visibility_ratio": 0.09642969202112268
                },
                "lighting": {
                    "average_lighting_on_face": 0.5535850375306373,
                    "light_uniformity_score": 0.9283180386412377,
                    "face_contrast": 0.7607843137254902,
                    "white_balance_shift": {"r_minus_g": -0.325473830279182, "r_minus_b": -0.1098240571863511, "g_minus_b": 0.21564977309283087},
                    "skin_color_vector": [91.00738525390625, 149.82424926757812],
                    "highlight_intensity": 0.9607843137254902,
                    "shadow_depth": 0.0,
                    "glare_score": 0.8352941176470589,
                    "shading_score": 0.9254901960784314,
                    # "skin_tone_index": 1,  # Audit-only, по умолчанию отключен
                    "lighting_proxy_score": 0.7520862893497242,
                    "zone_lighting": {"forehead_brightness": 0.4262352438534007, "cheek_brightness": 0.6280032288794424, "chin_brightness": 0.4654477287741268}
                },
                "skin": {
                    "makeup_presence_prob": 0.0,
                    "lipstick_intensity": 0.0,
                    "eye_shadow_prob": 0.0,
                    "skin_smoothness": 0.6753284717684971,
                    # "skin_defect_score": удален (рискованная метрика)
                    "beard_prob": 0.602679947023671,
                    "mustache_prob": 0.4218759629165697,
                    "face_hair_density": 0.5122779549701204,
                    "eyebrow_shape_vector": [57.296722412109375, 137.78384399414062, 402.7089538574219, 82.4332275390625],
                    "eyelid_openness": 23.402135848999023
                },
                "accessories": {
                    "glasses_prob": 0.15,
                    "sunglasses_prob": 0.7283823529395642,
                    "mask_prob": 0.09963344644481419,
                    "hat_prob": 0.3827916733725834,
                    "helmet_prob": 0.0,
                    "earrings_presence": false,
                    "earrings_prob": 0.02601626014145019,
                    "jewelry_probability": 0.24869254758225162,
                    "hair_accessories": 0.0
                },
                "eyes": {
                    "eye_opening_ratio": {"left": 23.23587417602539, "right": 23.568397521972656, "average": 23.402135848999023},
                    "eye_opening_left": 0.95,
                    "eye_opening_right": 0.96,
                    "blink_rate": 0.0,  # blinks per minute
                    "blink_intensity": 0.0,
                    "blink_flag": false,
                    "last_blink_timestamp": 0.0,
                    "gaze_vector": [0.10475023421914995, -0.2982941465317203, 0.9487085909677143],
                    "gaze_at_camera_prob": 0.7899764703070337,
                    "attention_score": 0.87,
                    # "eye_redness_prob": удален (не нужен для модели)
                    "iris_position": {"left": 0.28678393061378304, "right": 0.2929455951149514}
                },
                "motion": {
                    "face_speed": 0.0,
                    "face_acceleration": 0.0,
                    "micro_expression_rate": 0.0,
                    "jaw_movement_intensity": 0.0,
                    "eyebrows_motion_score": 0.0,
                    "mouth_motion_score": 0.0,
                    "head_motion_energy": 0.0,
                    "talking_motion_score": 0.0
                },
                "structure": {
                    "face_mesh_vector": [0.02699216641485691, ...31 values... -98.80367279052734],  # Сжато до 32 dims
                    "identity_shape_vector": {"hash": "a3f5b2c1...", "dim": 16},  # Privacy-preserving hash
                    "expression_vector": [-0.36291444301605225, ...7 values... 11.020713806152344],  # Сжато до 8 dims
                    "jaw_pose_vector": [6.300705890788989, -17.35517438503278, -8.068668365478516],
                    "eye_pose_vector": [-17.35517438503278, -8.068668365478516],
                    "mouth_shape_params": [0.9495363235473633, 1.0],
                    "face_symmetry_score": 0.7370345642991275,
                    "face_uniqueness_score": 0.21873119473457336,
                    "identity_params_count": 150,
                    "expression_params_count": 150,
                    "face_dimensions": {"width": 0.9495363235473633, "height": 1.0, "depth": 677.1336059570312, "aspect_ratio": 0.9495363235473633}
                },
                "professional": {
                    "face_quality_score": 0.75,  # Computed from quality module features
                    # "perceived_attractiveness_score": удален (рискованная метрика)
                    "emotion_intensity": 0.0,
                    "lip_reading_features": {"mouth_openness": 0.0, "mouth_motion_intensity": 0.0, "jaw_movement": 0.0, "speech_activity_prob": 0.0},
                    "fatigue_score": 0.31152648970569663,
                    "fatigue_breakdown": {
                        "eye_fatigue": 0.30282177305977004,
                        "pose_fatigue": 0.01846289711592366,
                        "motion_fatigue": 0.8,
                        "temporal_fatigue": 0.0,
                        "eye_closedness": 0.0,
                        "head_pitch_down": 0.0,
                        "low_motion_speed": 1.0,
                        "blink_abnormality": 1.0
                    },
                    "engagement_level": 1.0,
                    "alertness_score": 1.0,
                    "expressiveness_score": 0.0
                },
                "lip_reading": {
                    "mouth_width": 149.02044677734375,
                    "mouth_height": 3.4132251739501953,
                    "mouth_area": 399.4851891593844,
                    "mouth_aspect_ratio": 0.022904408406787324,
                    "lip_separation": 3.4132251739501953,
                    "lip_asymmetry": 5.119696455245047e-08,
                    "lip_contour_compactness": 222.96712394074865,
                    "lip_prominence_ratio": 0.8110154271125793,
                    "phoneme_features": {"round_shape": 0.022904408406787324, "wide_shape": 1.0, "narrow_shape": 0.9770955915932127, "open_shape": 0.022904408406787324},
                    "mouth_motion_intensity": 0.0,
                    "width_velocity": 0.0,
                    "height_velocity": 0.0,
                    "area_velocity": 0.0,
                    "cycle_strength": 0.0,
                    "speech_activity_prob": 0.0,
                    "width_variance": 0.0,
                    "height_variance": 0.0,
                    "area_variance": 0.0,
                    "separation_variance": 0.0
                },
                "face_3d": {
                    "face_mesh_vector": [0.11840604245662689, ...298 values... 1.7961621284484863],
                    "identity_shape_vector": [-0.5973265171051025, ...198 values... 0.0],
                    "expression_vector": [5.086263143994074e-08, ...48 values... 0.0],
                    "jaw_pose_vector": [6.300705890788989, -17.35517438503278, -8.068668365478516],
                    "eye_pose_vector": [-17.35517438503278, -8.068668365478516],
                    "mouth_shape_params": [154.00621032714844, 8.721385955810547, -6.754088401794434, 17.658456019199924],
                    "face_symmetry_score": 0.5040507710822972,
                    "face_uniqueness_score": 1.0,
                    "mesh_num_vertices": 100,
                    "identity_params_count": 100,
                    "expression_params_count": 50
                }
            }
        ],
    "5": [...]
}
```

### Фичи:

#### 1. frame_index
            
индекс кадра относительно всех кадров видео

#### 2. face_index

индекс лица (в одном кадре может быть более одного лица)

#### 3. bbox

бокс лица

####  4. geometry
    
#### 4.1. face_bbox_area

```
width = float(bbox[2] - bbox[0])
height = float(bbox[3] - bbox[1])
area = width * height
```

#### 4.2. face_relative_size

```
face_relative_size = area / (frame_width * frame_height + 1e-6),
```

#### 4.3. face_box_ratio

```
ratio = width / max(height, 1e-6)
```

#### 4.4. face_bbox_position

```
cx = float(bbox[0] + width / 2)
cy = float(bbox[1] + height / 2)
face_bbox_position = {"cx": cx, "cy": cy}
```

#### 4.5. face_dist_from_center

```
frame_cx = frame_width / 2
frame_cy = frame_height / 2
face_dist_from_center = float(np.linalg.norm([cx - frame_cx, cy - frame_cy]) / max(frame_width, frame_height))
```

#### 4.6. face_rotation_in_frame

```
height, width = frame.shape[:2]
rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
results = face_mesh.process(rgb)
LANDMARKS = {
    ...
    "left_eye_outer": 33, 
    "right_eye_outer": 263,
    ...
}
for face_idx, landmark_list in enumerate(results.multi_face_landmarks):
    if not landmark_list or not landmark_list.landmark:
        return np.zeros((0, 3), dtype=np.float32)

    diag = math.sqrt(width**2 + height**2)
    coords = np.array([[lm.x * width, lm.y * height, lm.z * (diag if scale_z else 1.0)] for lm in landmark_list.landmark], dtype=np.float32)
    left = coords[LANDMARKS["left_eye_outer"], :2]
    right = coords[LANDMARKS["right_eye_outer"], :2]
    face_rotation_in_frame = float(np.degrees(np.arctan2(right[1] - left[1], right[0] - left[0])))
```

#### 4.7. aspect_ratio_stability

```
_aspect_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=history_size))
_aspect_history[face_idx].append(ratio)
history = _aspect_history[face_idx]
stability = (float(1.0 - (np.std(history) / (np.mean(history) + 1e-5))) if len(history) > 3 else 0.0)
```

#### 4.8. jaw_width

```python
left_jaw = 172
right_jaw = 397
def _safe_distance(coords, i, j):
    if i >= len(coords) or j >= len(coords):
        return 0.0
    return float(np.linalg.norm(coords[i][:2] - coords[j][:2]))
jaw_width = _safe_distance(coords, left_jaw, right_jaw)
```

#### 4.9. cheekbone_width

```python
left_zygom = 127
right_zygom = 356
cheekbone_width = _safe_distance(coords, left_zygom, right_zygom)
```

#### 4.10. cheek_width

```python
left_cheek = 234
right_cheek = 454
cheek_width = _safe_distance(coords, left_cheek, right_cheek)
```

#### 4.11. forehead_height

```python
left_brow = 70
right_brow = 301
forehead_point = (coords[left_brow] + coords[right_brow]) / 2
forehead_height = float(np.linalg.norm(forehead_point[:2] - nose_tip[:2]))
```

#### 4.12. face_shape_vector

```python
FACE_OVAL_LANDMARKS = [
    10, 338, 297, 332, 284, 251, 389, 356,
     454, 323, 361, 288, 397, 365, 379, 
     378, 400, 377, 152, 176, 149, 150, 
     136, 172, 58, 132, 93, 234, 127, 
     162, 21, 54, 103, 67, 109
]
face_shape_vector = coords[FACE_OVAL_LANDMARKS, :2].flatten().tolist()
```

#### 5. pose
    
#### 5.1. yaw

```python
LANDMARKS = {
    "nose_tip": 1, 
    "chin": 152, 
    "left_eye_inner": 133, 
    "left_eye_outer": 33, 
    "right_eye_inner": 362, 
    "right_eye_outer": 263,
    ...
}

def _average_coords(coords: np.ndarray, indices: List[int]) -> np.ndarray:
    if len(indices) == 0 or coords.size == 0:
        return np.zeros((coords.shape[1],), dtype=np.float32)
    return np.mean(coords[indices, :], axis=0)

left_eye = _average_coords(coords, [LANDMARKS["left_eye_inner"], LANDMARKS["left_eye_outer"]])
right_eye = _average_coords(coords, [LANDMARKS["right_eye_inner"], LANDMARKS["right_eye_outer"]])

nose = coords[LANDMARKS["nose_tip"], :3]
chin = coords[LANDMARKS["chin"], :3]
forehead = coords[10, :3] if 10 < len(coords) else nose.copy()
left_ear = coords[234, :3] if 234 < len(coords) else left_eye.copy()
right_ear = coords[454, :3] if 454 < len(coords) else right_eye.copy()
skull_width = np.linalg.norm(right_ear[:2] - left_ear[:2]) + 1e-6
mid_eye = (left_eye + right_eye) / 2
nose_horizontal_offset = nose[0] - mid_eye[0]
yaw_2d = np.degrees(np.arctan2(nose_horizontal_offset, skull_width * 0.55))
mid_face_z = (left_eye[2] + right_eye[2] + chin[2]) / 3
depth_offset = nose[2] - mid_face_z
yaw_3d = np.degrees(np.arctan2(nose_horizontal_offset, abs(depth_offset) + skull_width * 0.35))
yaw = yaw_2d * 0.7 + yaw_3d * 0.3
```

#### 5.2. pitch

```python
eye_y = mid_eye[1]
chin_y = chin[1]
face_vertical_span = abs(chin_y - eye_y) + 1e-6
expected_nose_y = eye_y + face_vertical_span * 0.35
nose_vertical_offset = nose[1] - expected_nose_y
pitch_2d = np.degrees(np.arctan2(nose_vertical_offset, face_vertical_span * 0.8))
forehead_chin_vertical = forehead[1] - chin[1]
pitch_3d = np.degrees(np.arctan2(forehead_chin_vertical, abs(depth_offset) + face_vertical_span * 0.3))
pitch = pitch_2d * 0.65 + pitch_3d * 0.35
```

#### 5.3. roll

```python
roll = np.degrees(np.arctan2(right_eye[1] - left_eye[1], right_eye[0] - left_eye[0]))
```

#### 5.4. avg_pose_angle

```python
avg_pose_angle = float(np.mean([yaw, pitch, roll]))
```

#### 5.5. head_pose_variability

```python
_pose_history: Dict[int, deque] = defaultdict(lambda: deque(maxlen=30))
if face_idx not in _pose_history:
    _pose_history[face_idx] = deque(maxlen=30)
current_pose = {"yaw": float(yaw), "pitch": float(pitch), "roll": float(roll)}
_pose_history[face_idx].append(current_pose)
hist = list(_pose_history[face_idx])
if len(hist) > 1:
    variability = float(np.mean([np.std([p["yaw"] for p in hist]), np.std([p["pitch"] for p in hist]), np.std([p["roll"] for p in hist])]))
else:
    variability = 0.0
```

#### 5.6. pose_stability_score

```python
pose_stability_score = float(np.clip(1 - abs(yaw) / 45.0, 0.0, 1.0))
```

#### 5.7. head_turn_frequency

```python
if len(hist) >= 3:
    yaw_diffs = [abs(hist[i]["yaw"] - hist[i - 1]["yaw"]) for i in range(1, len(hist))]
    turns = [1.0 if diff > 7.0 else 0.0 for diff in yaw_diffs]
    turn_frequency = float(np.mean(turns))
else:
    turn_frequency = 0.0
```

#### 5.8. attention_to_camera_ratio

```python
attention_to_camera_ratio = float(np.exp(-((yaw / 25.0) ** 2)))
```

#### 5.9. looking_direction_vector

```python
looking_direction_vector = [float(right_eye[0] - left_eye[0]), float(right_eye[1] - left_eye[1]), float(right_eye[2] - left_eye[2])]
```

#### 6. quality

#### 6.1. face_blur_score

```python
x_min, y_min, x_max, y_max = map(int, bbox)
if x_max < x_min:
    x_min, x_max = x_max, x_min
if y_max < y_min:
    y_min, y_max = y_max, y_min
h, w = frame.shape[:2]
x_min = max(0, min(w - 1, x_min))
x_max = max(0, min(w,     x_max))
y_min = max(0, min(h - 1, y_min))
y_max = max(0, min(h,     y_max))
if (x_max - x_min) < 2 or (y_max - y_min) < 2:
    return np.zeros((1, 1, 3), dtype=frame.dtype)
roi = frame[y_min:y_max, x_min:x_max].copy()

gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
blur_raw = cv2.Laplacian(gray, cv2.CV_64F).var()
blur_score = float(np.clip(blur_raw / 300.0, 0.0, 1.0))
```

#### 6.2. sharpness_score

```python
sobel = cv2.Sobel(gray, cv2.CV_32F, 1, 1, ksize=3)
sharpness = float(np.mean(np.abs(sobel)))
sharpness_score = float(np.clip(sharpness / 200.0, 0.0, 1.0))
```

#### 6.3. texture_quality

```python
texture_quality = float(np.std(gray) / 255.0)
```

#### 6.4. focus_metric

```python
mean, std = cv2.meanStdDev(gray)
focus_metric = float(np.clip(std[0][0] / 255.0, 0.0, 1.0))
```

#### 6.5. noise_level

```python
blurred = cv2.GaussianBlur(gray, (5, 5), 0)
noise = float(np.std(gray.astype(np.float32) - blurred.astype(np.float32)))
noise_level = float(np.clip(noise / 40.0, 0.0, 1.0))
```

#### 6.6. motion_blur_score

```python
motion_blur_score = float(np.clip(1.0 - blur_raw / 500.0, 0.0, 1.0))
```

#### 6.7. artifact_score

```python
artifact_score = float(np.clip(1.0 - texture_quality, 0.0, 1.0))
```

#### 6.8. resolution_of_face

```python
frame_area = frame_shape[0] * frame_shape[1]
face_area = max((bbox[2] - bbox[0]), 1) * max((bbox[3] - bbox[1]), 1)
resolution_of_face = float(face_area)
```

#### 6.9. face_visibility_ratio

```python
ratio = face_area / max(frame_area, 1)
face_visibility_ratio = float(np.clip(ratio, 0.0, 1.0))
```

#### 6.10. occlusion_score

```python
occlusion_score = float(np.clip(1.0 - face_visibility_ratio, 0.0, 1.0))
```

#### 6.11. quality_proxy_score

```python
quality_proxy_score = float(np.clip(
    0.4 * blur_score + 
    0.3 * sharpness_score + 
    0.2 * focus_metric + 
    0.1 * (1.0 - noise_level), 0.0, 1.0
))
```

#### 7. lighting

#### 7.1. average_lighting_on_face

```python
hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
value = hsv[:, :, 2].astype(np.float32)
avg_light = float(np.mean(value) / 255.0)
```

#### 7.2. light_uniformity_score

```python
h, w = roi.shape[:2]
mid_x = w // 2
left_mean = float(np.mean(value[:, :mid_x])) if mid_x > 1 else 0
right_mean = float(np.mean(value[:, mid_x:])) if w - mid_x > 1 else 0
uniformity = float(1.0 - abs(left_mean - right_mean) / 255.0)
uniformity = float(np.clip(uniformity, 0.0, 1.0))
```

#### 7.3. face_contrast

```python
p5 = np.percentile(value, 5)
p95 = np.percentile(value, 95)
contrast = float(np.clip((p95 - p5) / 255.0, 0.0, 1.0))
```

#### 7.4. white_balance_shift

```python
mean_rgb = np.mean(roi.reshape(-1, 3), axis=0).astype(np.float32)
b, g, r = mean_rgb
white_balance_shift = {"r_minus_g": float((r - g) / 255.0), "r_minus_b": float((r - b) / 255.0), "g_minus_b": float((g - b) / 255.0)}
```

#### 7.5. skin_color_vector

```python
lab = cv2.cvtColor(roi, cv2.COLOR_BGR2LAB)
L_channel = lab[:, :, 0].astype(np.float32)
A_channel = lab[:, :, 1].astype(np.float32)
B_channel = lab[:, :, 2].astype(np.float32)
skin_vector = [float(np.mean(A_channel)), float(np.mean(B_channel))]
```

#### 7.6. - 7.9. highlight_intensity, shadow_depth, glare_score, shading_score

```python
highlight = float(np.max(value) / 255.0)
shadow = float(np.min(value) / 255.0)
glare_score = float(np.clip(p95 / 255.0, 0.0, 1.0))
shading_score = float(np.clip(1.0 - p5 / 255.0, 0.0, 1.0))
```

#### 7.10. skin_tone_index

```python
mean_L = float(np.mean(L_channel))
if mean_L > 80:
    skin_tone_index = 1
elif mean_L > 70:
    skin_tone_index = 2
elif mean_L > 60:
    skin_tone_index = 3
elif mean_L > 50:
    skin_tone_index = 4
elif mean_L > 40:
    skin_tone_index = 5
else:
    skin_tone_index = 6
```

#### 7.11. lighting_proxy_score

```python
lighting_proxy = float(np.clip(
    0.4 * avg_light + 
    0.3 * uniformity + 
    0.2 * contrast + 
    0.1 * (1.0 - shadow), 0.0, 1.0
))
```

#### 7.12. zone_lighting

```python
LANDMARKS = {"nose_tip": 1, "chin": 152, "forehead": 10}
bbox_x_min, bbox_y_min = bbox[0], bbox[1]
coords_roi = coords.copy()
coords_roi[:, 0] -= bbox_x_min
coords_roi[:, 1] -= bbox_y_min
zone_lighting = {}
if coords_roi is not None and coords_roi.shape[0] > 0:
    try:
        yy = np.clip(coords_roi[:, 1].astype(int), 0, h - 1)
        forehead_y = yy[LANDMARKS["forehead"]]
        nose_y = yy[LANDMARKS["nose_tip"]]
        chin_y = yy[LANDMARKS["chin"]]

        top = max(0, int(forehead_y * 0.6))
        mid = int(nose_y)
        bot = int(chin_y)

        if top < forehead_y:
            zone = value[:forehead_y]
            if zone.size > 0:
                zone_lighting["forehead_brightness"] = float(np.mean(zone) / 255.0)

        if forehead_y < mid:
            zone = value[forehead_y:mid]
            if zone.size > 0:
                zone_lighting["cheek_brightness"] = float(np.mean(zone) / 255.0)

        if mid < bot:
            zone = value[mid:bot]
            if zone.size > 0:
                zone_lighting["chin_brightness"] = float(np.mean(zone) / 255.0)
    except Exception:
        pass
if zone_lighting:
    result["zone_lighting"] = zone_lighting
```

#### 8. skin

#### 8.1. - 8.3. makeup_presence_prob, lipstick_intensity, eye_shadow_prob

```python
LANDMARKS = {"upper_lip": 13, "lower_lip": 14,}
lip_idx = [LANDMARKS["upper_lip"], LANDMARKS["lower_lip"]]
lip_coords = coords_roi[lip_idx, :2]
lip_x1 = int(max(0, np.min(lip_coords[:, 0])))
lip_y1 = int(max(0, np.min(lip_coords[:, 1])))
lip_x2 = int(min(w - 1, np.max(lip_coords[:, 0])))
lip_y2 = int(min(h - 1, np.max(lip_coords[:, 1])))
lip_roi = roi[lip_y1:lip_y2, lip_x1:lip_x2] if lip_x2 > lip_x1 and lip_y2 > lip_y1 else None
if lip_roi is not None and lip_roi.size > 0:
    lip_hsv = cv2.cvtColor(lip_roi, cv2.COLOR_BGR2HSV)
    lip_sat = float(np.mean(lip_hsv[:, :, 1] / 255.0))
else:
    lip_sat = 0.0
lipstick_intensity = float(np.clip((lip_sat - avg_sat) * 1.5, 0.0, 1.0))
makeup_presence_prob = lipstick_intensity
eye_shadow_prob = float(lipstick_intensity * 0.4)
```

#### 8.4. - 8.5. skin_smoothness, skin_defect_score

```python
gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (5, 5), 0)
laplacian_var = cv2.Laplacian(blur, cv2.CV_64F).var()

skin_smoothness = float(1 / (1 + laplacian_var * 0.1))
skin_defect_score = float(min(max(laplacian_var * 10, 0), 100))
```

#### 8.6. - 8.10. beard_prob, mustache_prob, face_hair_density, eyebrow_shape_vector, eyelid_openness

```python
LANDMARKS = {
    "chin": 152,
    "left_brow": 70,
    "right_brow": 300
}
chin_idx = LANDMARKS["chin"]
chin_y = int(coords_roi[chin_idx, 1])
lower_roi = roi[chin_y: h] if 0 < chin_y < h else roi

if lower_roi.size > 0:
    lower_lab = cv2.cvtColor(lower_roi, cv2.COLOR_BGR2LAB)
    L_lower = lower_lab[:, :, 0]
    darkness = 1.0 - float(np.mean(L_lower) / 255.0)
    texture = float(cv2.Laplacian(cv2.cvtColor(lower_roi, cv2.COLOR_BGR2GRAY),
                                cv2.CV_64F).var() / 50.0)
    beard_prob = float(np.clip(0.6 * darkness + 0.4 * texture, 0.0, 1.0))
else:
    beard_prob = 0.0

mustache_prob = float(beard_prob * 0.7)
face_hair_density = float((beard_prob + mustache_prob) / 2)
eyebrow_pts = coords_roi[[LANDMARKS["left_brow"], LANDMARKS["right_brow"]], :2]
eyebrow_shape_vector = eyebrow_pts.flatten().tolist()
eyelid_openness = eye_opening(coords_roi)
```

#### 9. accessories

#### 9.1. - 9.2. glasses_prob, sunglasses_prob

```python
h, w = roi.shape[:2]
gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
glasses_prob = 0.0
sunglasses_prob = 0.0
try:
    ex1, ey1, ex2, ey2 = eye_box(coords_roi)
    ex1, ey1 = int(max(0, ex1)), int(max(0, ey1))
    ex2, ey2 = int(min(w - 1, ex2)), int(min(h - 1, ey2))
    eye_roi = gray[ey1:ey2, ex1:ex2] if ex2 > ex1 and ey2 > ey1 else np.array([], dtype=np.uint8)
    if eye_roi.size:
        edges = cv2.Canny(eye_roi, 50, 150)
        edge_density = float(np.sum(edges > 0) / (eye_roi.size + 1e-6))
        glasses_prob = float(np.clip(edge_density * 3.0, 0.0, 1.0))
        mean_eye = float(np.mean(eye_roi))
        sunglasses_prob = float(np.clip((1.0 - (mean_eye / 255.0)) * 1.2, 0.0, 1.0))
        if sunglasses_prob > 0.4 and edge_density > 0.02:
            sunglasses_prob = float(np.clip(sunglasses_prob + edge_density * 0.5, 0.0, 1.0))
            glasses_prob = max(glasses_prob, 0.15)
except Exception:
    pass
```

#### 9.3. mask_prob

```python
mask_prob = 0.0
try:
    lx1, ly1, lx2, ly2 = lower_face_box(coords_roi)
    lx1, ly1 = int(max(0, lx1)), int(max(0, ly1))
    lx2, ly2 = int(min(w - 1, lx2)), int(min(h - 1, ly2))
    lower_roi = roi[ly1:ly2, lx1:lx2] if lx2 > lx1 and ly2 > ly1 else np.array([], dtype=np.uint8)
    if lower_roi.size:
        lower_hsv = cv2.cvtColor(lower_roi, cv2.COLOR_BGR2HSV)
        v = lower_hsv[:, :, 2].astype(np.float32) / 255.0
        v_std = float(np.std(v))
        v_mean = float(np.mean(v))
        color_homogeneity = 1.0 - v_std
        coverage = float(lower_roi.size / (roi.size + 1e-6))
        mask_prob = float(np.clip(color_homogeneity * 1.2 * coverage, 0.0, 1.0))
except Exception:
    pass
```

#### 9.4. - 9.5. hat_prob, helmet_prob

```python
top_strip = roi[0:max(1, h // 6), :, :]
top_gray = cv2.cvtColor(top_strip, cv2.COLOR_BGR2GRAY) if top_strip.size else np.array([], dtype=np.uint8)
hat_prob = 0.0
helmet_prob = 0.0
if top_gray.size:
    top_mean = float(np.mean(top_gray))
    top_std = float(np.std(top_gray))
    hat_prob = float(np.clip((1.0 - top_std / (np.mean(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)) + 1e-6)) * (1.0 - top_mean / 255.0), 0.0, 1.0))
    helmet_prob = float(np.clip(hat_prob * 0.4 if top_mean < 80 and top_std < 30 else 0.0, 0.0, 1.0))
```

#### 9.6. earrings_presence

```python
earrings_prob = 0.0
try:
    left_cheek = coords_roi[LANDMARKS["left_cheek"], :2].astype(int)
    right_cheek = coords_roi[LANDMARKS["right_cheek"], :2].astype(int)
    ear_y = int((left_cheek[1] + right_cheek[1]) / 2)
    ear_region_size = min(30, max(10, w // 8))
    def detect_ear_spot(cx):
        x1 = max(0, cx - ear_region_size)
        x2 = min(w, cx + ear_region_size)
        y1 = max(0, ear_y - ear_region_size // 2)
        y2 = min(h, ear_y + ear_region_size // 2)
        region = roi[y1:y2, x1:x2]
        if region.size == 0:
            return 0.0
        gray_r = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY)
        bright_frac = float(np.sum(gray_r > np.mean(gray_r) * 1.25) / (gray_r.size + 1e-6))
        return bright_frac
    left_spot = detect_ear_spot(int(left_cheek[0]))
    right_spot = detect_ear_spot(int(right_cheek[0]))
    earrings_prob = float(np.clip((left_spot + right_spot) / 2.0, 0.0, 1.0))
except Exception:
    pass
earrings_presence = bool(earrings_prob > 0.03)
```

#### 9.7. earrings_prob

#### 9.8. jewelry_probability

```python
jewelry_probability = 0.0
try:
    chin_y = int(coords_roi[LANDMARKS["chin"], 1])
    neck_y = min(h - 1, chin_y + int(h * 0.06))
    if 0 <= neck_y < h:
        neck_h = min(30, h - neck_y)
        neck_roi = roi[neck_y:neck_y + neck_h, :]
        if neck_roi.size:
            neck_gray = cv2.cvtColor(neck_roi, cv2.COLOR_BGR2GRAY)
            jewelry_probability = float(np.clip(np.std(neck_gray) / 80.0, 0.0, 1.0))
except Exception:
    pass
```

#### 9.9. hair_accessories

```python
hair_accessories = 0.0
try:
    forehead_y = int(coords_roi[LANDMARKS["forehead"], 1])
    hair_top_y = max(0, forehead_y - 20)
    hair_roi = roi[hair_top_y:forehead_y, :]
    if hair_roi.size:
        gray_hr = cv2.cvtColor(hair_roi, cv2.COLOR_BGR2GRAY)
        hr_std = float(np.std(gray_hr))
        whole_std = float(np.std(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)))
        if whole_std > 1e-6:
            hair_accessories = float(np.clip(max(0.0, (hr_std - whole_std * 1.2) / 60.0), 0.0, 1.0))
except Exception:
    pass
```

#### 10. eyes

#### 10.1. eye_opening_ratio

```python
LANDMARKS = {
    ...
    "left_eye_upper": 159, 
    "left_eye_lower": 145, 
    "right_eye_upper": 386, 
    "right_eye_lower": 374,
    ...
}
left_open = safe_distance(coords, LANDMARKS["left_eye_upper"], LANDMARKS["left_eye_lower"])
right_open = safe_distance(coords, LANDMARKS["right_eye_upper"], LANDMARKS["right_eye_lower"])

avg_open = (left_open + right_open) / 2.0

result = {
    ...
    "eye_opening_ratio": {
        "left": float(left_open), 
        "right": float(right_open), 
        "average": float(avg_open)
    },
    ...
}
```

#### 10.2. - 10.3. blink_rate, blink_intensity

```python
def _estimate_blink_rate(history: deque) -> float:
    if len(history) < 2:
        return 0.0
    diffs = np.diff(list(history))
    threshold = -0.5 * np.std(list(history)) if np.std(list(history)) > 0 else -0.01
    blinks = np.sum(diffs < threshold)
    return float(blinks / max(len(history), 1))

blink_rate = _estimate_blink_rate(self._blink_history[face_idx])
blink_intensity = float(np.std(list(self._blink_history[face_idx]))) if len(self._blink_history[face_idx]) > 1 else 0.0
```

#### 10.4. gaze_vector

```python
yaw = float(np.radians(pose.get("yaw", 0.0)))
pitch = float(np.radians(pose.get("pitch", 0.0)))
gaze_x = np.sin(yaw) * np.cos(pitch)
gaze_y = np.sin(pitch)
gaze_z = np.cos(yaw) * np.cos(pitch)
gaze_vector = [float(gaze_x), float(gaze_y), float(gaze_z)]
```

#### 10.5. gaze_at_camera_prob

```python
gaze_at_camera_prob = float(np.clip(1 - abs(pose.get("yaw", 0)) / 30.0, 0.0, 1.0))
```

#### 10.6. attention_score

```python
attention_score = float((gaze_at_camera_prob + avg_open) / 2.0)
```

#### 10.7. - 10.8. eye_redness_prob, iris_position

```python
eye_redness_prob = float(np.clip((left_open + right_open) / 40.0, 0.0, 1.0))
left_width = safe_distance(coords, LANDMARKS["left_eye_inner"], LANDMARKS["left_eye_outer"])
right_width = safe_distance(coords, LANDMARKS["right_eye_inner"], LANDMARKS["right_eye_outer"])
left_iris = left_open / max(left_width, 1e-6)
right_iris = right_open / max(right_width, 1e-6)
iris_position = {"left": float(np.clip(left_iris, 0.0, 1.0)), "right": float(np.clip(right_iris, 0.0, 1.0))}
```

#### 11. motion

#### 11.1. face_speed

```python
coords = data["coords"]
geometry = data["geometry"] # из модуля geometry
center = geometry["face_bbox_position"]["cx"], geometry["face_bbox_position"]["cy"]
mouth_gap = safe_distance(coords, LANDMARKS["upper_lip"], LANDMARKS["lower_lip"])
jaw_distance = safe_distance(coords, LANDMARKS["chin"], LANDMARKS["nose_tip"])
eyebrow_height = coords[LANDMARKS["forehead"], 1] - coords[LANDMARKS["left_brow"], 1]
eye_opening_val = eye_opening(coords)
if face_idx not in _temporal_state_history:
    _temporal_state_history[face_idx] = deque(maxlen=int(fps // 2))
history = _temporal_state_history[face_idx]
history.append({"center": center, "mouth_gap": mouth_gap, "jaw_distance": jaw_distance, "eyebrow_height": eyebrow_height, "eye_opening": eye_opening_val})
centers = np.array([s["center"] for s in history])
displacements = np.linalg.norm(np.diff(centers, axis=0), axis=1)
speed_threshold = 0.5
filtered_displacements = displacements[displacements > speed_threshold]
face_speed = float(np.mean(filtered_displacements)) if filtered_displacements.size else 0.0
```

#### 11.2. face_acceleration

```python
face_acceleration = float(np.mean(np.diff(filtered_displacements))) if filtered_displacements.size > 1 else 0.0
```

#### 11.3. micro_expression_rate

```python
mouth_gaps = np.array([s["mouth_gap"] for s in history])
mouth_deltas = np.abs(np.diff(mouth_gaps))
motion_threshold = 0.3
filtered_mouth = mouth_deltas[mouth_deltas > motion_threshold]
mouth_motion = float(np.mean(filtered_mouth)) if filtered_mouth.size else 0.0
micro_expression_rate = mouth_motion + eyebrow_motion
```

#### 11.4. jaw_movement_intensity

```python
jaw_distances = np.array([s["jaw_distance"] for s in history])
jaw_deltas = np.abs(np.diff(jaw_distances))
filtered_jaw = jaw_deltas[jaw_deltas > motion_threshold]
jaw_movement_intensity = float(np.mean(filtered_jaw)) if filtered_jaw.size else 0.0
```

#### 11.5. eyebrows_motion_score

```python
eyebrow_heights = np.array([s["eyebrow_height"] for s in history])
eyebrow_deltas = np.abs(np.diff(eyebrow_heights))
filtered_eyebrow = eyebrow_deltas[eyebrow_deltas > motion_threshold]
eyebrow_motion = float(np.mean(filtered_eyebrow)) if filtered_eyebrow.size else 0.0
```

#### 11.6. mouth_motion_score

```python
mouth_motion = float(np.mean(filtered_mouth)) if filtered_mouth.size else 0.0
```

#### 11.7. head_motion_energy

```python
eye_openings = np.array([s["eye_opening"] for s in history])
eye_deltas = np.abs(np.diff(eye_openings))
filtered_eye = eye_deltas[eye_deltas > motion_threshold]
head_motion = float(np.mean(filtered_eye)) if filtered_eye.size else 0.0
```

#### 11.9. talking_motion_score

```python
talking_motion_score = mouth_motion
```

#### 12. structure

#### 12.1. face_mesh_vector

```python
normalized = coords[:, :3].copy()
normalized[:, 0] -= np.mean(normalized[:, 0])
normalized[:, 1] -= np.mean(normalized[:, 1])
normalized[:, :2] /= np.max(np.ptp(normalized[:, :2], axis=0) + 1e-6)
face_mesh_vector = normalized[:100, :].flatten().tolist()
```

#### 12.2. identity_shape_vector

```python
identity_shape_vector = normalized[:50, :3].flatten().tolist()
```

#### 12.3. expression_vector

```python
expression_vector = (normalized[50:100, :3] - normalized[:50, :3]).flatten().tolist()
```

#### 12.4. jaw_pose_vector

```python
jaw_pose_vector = [pose["yaw"], pose["pitch"], pose["roll"]] # из модуля pose
```

#### 12.5. eye_pose_vector

```python
eye_pose_vector = [pose["pitch"], pose["roll"]] # из модуля pose
```

#### 12.6. mouth_shape_params

```python
mouth_shape_params = [
    float(np.max(normalized[:, 0]) - np.min(normalized[:, 0])),
    float(np.max(normalized[:, 1]) - np.min(normalized[:, 1])),
]
```

#### 12.7. - 12.8. face_symmetry_score, face_uniqueness_score

```python
symmetry = 1.0 - float(np.mean(np.abs(normalized[:, 0] + normalized[::-1, 0])) / (np.max(np.abs([:, 0])) + 1e-6))
uniqueness = float(np.std(normalized[:, :2]))
```

#### 12.9. - 12.10. identity_params_count, expression_params_count

```python
identity_params_count = len(identity_shape_vector)
expression_params_count = len(expression_vector)
```

#### 12.11. face_dimensions

```python
face_width = float(np.max(normalized[:, 0]) - np.min(normalized[:, 0]))
face_height = float(np.max(normalized[:, 1]) - np.min(normalized[:, 1]))
face_depth = float(np.max(normalized[:, 2]) - np.min(normalized[:, 2])) if normalized.shape[1] > 2 else 0.0
face_aspect_ratio = face_width / max(face_height, 1e-6)
face_dimensions = {"width": face_width, "height": face_height, "depth": face_depth, "aspect_ratio": face_aspect_ratio}
```

#### 13. professional

#### 13.1. face_quality_score

```python
quality = data["quality"] # из модуля quality
blur = quality.get("face_blur_score", 0.0)
sharpness = quality.get("sharpness_score", 0.0)
face_quality = float(np.clip((sharpness * 2.0) / (1.0 + np.log1p(blur)), 0.0, 1.0))
```

#### 13.2. perceived_attractiveness_score

```python
perceived_attractiveness = float(np.clip(quality.get("face_contrast", 0.5), 0.0, 1.0))
```

#### 13.3. emotion_intensity

```python
motion = data["motion"]
emotion_intensity = float(motion.get("mouth_motion_score", 0.0) + motion.get("eyebrows_motion_score", 0.0))
emotion_intensity = np.clip(emotion_intensity, 0.0, 1.0)
```

#### 13.4. lip_reading_features

```python
lip_reading = data.get("lip_reading", {})
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
    mouth_motion = motion.get("talking_motion_score", 0.0)
    jaw_motion = motion.get("jaw_movement_intensity", 0.0)
    lip_reading_features = {
        "mouth_openness": mouth_motion,
        "mouth_motion_intensity": mouth_motion,
        "jaw_movement": jaw_motion,
        "speech_activity_prob": float(np.clip(mouth_motion * 2.0, 0.0, 1.0)),
    }
```

#### 13.5. fatigue_score

```python
eyes = data["eyes"]
eye_opening_data = eyes.get("eye_opening_ratio", {})
if isinstance(eye_opening_data, dict):
    avg_eye_opening = eye_opening_data.get("average", 0.5)
    left_eye_opening = eye_opening_data.get("left", 0.5)
    right_eye_opening = eye_opening_data.get("right", 0.5)
else:
    avg_eye_opening = float(eye_opening_data) if isinstance(eye_opening_data, (int, float)) else 0.5
    left_eye_opening = right_eye_opening = avg_eye_opening
blink_rate = eyes.get("blink_rate", 0.0)
eye_closedness = 1.0 - np.clip(avg_eye_opening / 20.0, 0.0, 1.0)
eye_asymmetry = abs(left_eye_opening - right_eye_opening) / max(max(left_eye_opening, right_eye_opening), 1e-6)
blink_rate_normalized = np.clip(blink_rate * 60.0, 0.0, 30.0) / 30.0  # Нормализуем до 0-1
blink_deviation = abs(blink_rate_normalized - 0.5) * 2.0  # Отклонение от нормы
eye_fatigue_score = float(eye_closedness * 0.5 + eye_asymmetry * 0.2 + blink_deviation * 0.3)

head_pitch = pose.get("pitch", 0.0)
head_pitch_normalized = np.clip(head_pitch / 30.0, 0.0, 1.0)  # Наклон вниз = усталость
pose_variability = pose.get("head_pose_variability", 0.0)
pose_stability = 1.0 - np.clip(pose_variability / 15.0, 0.0, 1.0)
attention_to_camera = pose.get("attention_to_camera_ratio", 0.5)
attention_loss = 1.0 - attention_to_camera
pose_fatigue_score = float(head_pitch_normalized * 0.4 + (1.0 - pose_stability) * 0.3 + attention_loss * 0.3)

face_speed = motion.get("face_speed", 0.0)
face_acceleration = motion.get("face_acceleration", 0.0)
speed_normalized = np.clip(face_speed / 10.0, 0.0, 1.0)
low_speed_indicator = 1.0 - speed_normalized
motion_activity = motion.get("micro_expression_rate", 0.0)
motion_activity_normalized = np.clip(motion_activity, 0.0, 1.0)
low_activity_indicator = 1.0 - motion_activity_normalized
acceleration_variance = abs(face_acceleration)
irregular_motion = np.clip(acceleration_variance / 5.0, 0.0, 1.0)
motion_fatigue_score = float(low_speed_indicator * 0.4 + low_activity_indicator * 0.4 + irregular_motion * 0.2)

current_fatigue_indicators = {"eye_closedness": eye_closedness, "head_pitch": head_pitch_normalized, "motion_speed": speed_normalized, "attention": attention_to_camera}
if face_idx not in self._fatigue_history:
    self._fatigue_history[face_idx] = deque(maxlen=int(self.fps * 5))
history = self._fatigue_history[face_idx]
history.append(current_fatigue_indicators)
temporal_fatigue_score = 0.0
if len(history) >= int(self.fps * 2):  # Минимум 2 секунды истории
    eye_closedness_history = [f["eye_closedness"] for f in history]
    motion_speed_history = [f["motion_speed"] for f in history]
    attention_history = [f["attention"] for f in history]
    if len(eye_closedness_history) > 1:
        eye_trend = (eye_closedness_history[-1] - eye_closedness_history[0]) / len(eye_closedness_history)
        motion_trend = (1.0 - motion_speed_history[-1]) - (1.0 - motion_speed_history[0])
        attention_trend = attention_history[0] - attention_history[-1]
        temporal_fatigue_score = float(np.clip((eye_trend * 0.5 + motion_trend * 0.3 + attention_trend * 0.2) * 2.0, 0.0, 1.0))
fatigue_score = float(np.clip(eye_fatigue_score * 0.35 + pose_fatigue_score * 0.30 + motion_fatigue_score * 0.25 + temporal_fatigue_score * 0.10, 0.0, 1.0))
```

#### 13.6. fatigue_breakdown

```python
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
```

#### 13.7. engagement_level

```python
gaze_score = eyes.get("gaze_at_camera_prob", 0.5)
attention = eyes.get("attention_score", 0.5)
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
```

#### 13.8. - 13.9. alertness_score, expressiveness_score

```python
    expressiveness_score
        alertness_score = float(np.clip(avg_eye_opening * 0.7 + face_speed / 15.0 * 0.3, 0.0, 1.0))
        expressiveness_score = float(np.clip(emotion_intensity + micro_expressions, 0.0, 1.0))
```

#### 14. lip_reading

#### 14.1. - 14.4. mouth_width, mouth_height, mouth_area, mouth_aspect_ratio

```python
LANDMARKS = {"mouth_left": 61, "mouth_right": 291}
mouth_left_idx = LANDMARKS.get("mouth_left", 61)
mouth_right_idx = LANDMARKS.get("mouth_right", 291)
mouth_width = safe_distance(coords, mouth_left_idx, mouth_right_idx)
mouth_width = safe_distance(coords, mouth_left_idx, mouth_right_idx)
mouth_height = safe_distance(coords, upper_lip_idx, lower_lip_idx)
mouth_area = np.pi * (mouth_width / 2.0) * (mouth_height / 2.0)
mouth_aspect_ratio = mouth_height / max(mouth_width, 1e-6)
```

#### 14.5. - 14.6. lip_separation, lip_asymmetry

```python
if (mouth_left_idx < len(coords) and mouth_right_idx < len(coords) and upper_lip_idx < len(coords) and lower_lip_idx < len(coords)):
    mouth_center = ((coords[mouth_left_idx][:2] + coords[mouth_right_idx][:2]) / 2.0)
    upper_lip_center = coords[upper_lip_idx][:2]
    lower_lip_center = coords[lower_lip_idx][:2]
    lip_separation = np.linalg.norm(upper_lip_center - lower_lip_center)
    left_half_width = np.linalg.norm(mouth_center - coords[mouth_left_idx][:2])
    right_half_width = np.linalg.norm(coords[mouth_right_idx][:2] - mouth_center)
    lip_asymmetry = abs(left_half_width - right_half_width) / max(mouth_width, 1e-6)
else:
    lip_separation = 0.0
    lip_asymmetry = 0.0
    mouth_center = np.array([0.0, 0.0])
```

#### 14.7. - 14.8. lip_contour_compactness, lip_prominence_ratio

```python
LANDMARKS = {
    "upper_lip": 13,
    "lower_lip": 14,
    "mouth_left": 61,
    "mouth_right": 291
}
lip_contour_points = []
for key in ["mouth_left", "upper_lip", "mouth_right", "lower_lip"]:
    idx = LANDMARKS.get(key)
    if idx is not None and idx < len(coords):
        lip_contour_points.append(coords[idx][:2])
if len(lip_contour_points) >= 4:
    lip_contour_points = np.array(lip_contour_points)
    perimeter = (
        np.linalg.norm(lip_contour_points[1] - lip_contour_points[0]) +
        np.linalg.norm(lip_contour_points[2] - lip_contour_points[1]) +
        np.linalg.norm(lip_contour_points[3] - lip_contour_points[2]) +
        np.linalg.norm(lip_contour_points[0] - lip_contour_points[3])
    )
    contour_compactness = (perimeter ** 2) / max(mouth_area, 1e-6)
    upper_lip_height = np.linalg.norm(lip_contour_points[1] - mouth_center)
    lower_lip_height = np.linalg.norm(lip_contour_points[3] - mouth_center)
    lip_prominence_ratio = upper_lip_height / max(lower_lip_height, 1e-6)
else:
    contour_compactness = 0.0
    lip_prominence_ratio = 1.0
```

#### 14.9. phoneme_features

```python
        phoneme_features = {
            "round_shape": float(np.clip(mouth_aspect_ratio, 0.0, 1.0)),  # Округлая форма (О, У)
            "wide_shape": float(np.clip(mouth_width / max(mouth_height, 1e-6), 0.0, 2.0) / 2.0),  # Широкая форма (И, Э)
            "narrow_shape": float(np.clip(1.0 - mouth_aspect_ratio, 0.0, 1.0)),  # Узкая форма
            "open_shape": float(np.clip(lip_separation / max(mouth_width, 1e-6), 0.0, 1.0)),  # Открытый рот (А, Э)
        }
   ```

#### 14.10. - 14.19. mouth_motion_intensity, width_velocity, height_velocity, area_velocity, cycle_strength, speech_activity_prob, width_variance, height_variance, area_variance, separation_variance

```python
current_frame_features = {
    "mouth_width": float(mouth_width),
    "mouth_height": float(mouth_height),
    "mouth_area": float(mouth_area),
    "lip_separation": float(lip_separation),
    "mouth_aspect_ratio": float(mouth_aspect_ratio),
}
if face_idx not in self._lip_history:
    self._lip_history[face_idx] = deque(maxlen=int(self.fps * 2))
history = self._lip_history[face_idx]
history.append(current_frame_features)
if len(history) >= 2:
    widths = [f["mouth_width"] for f in history]
    heights = [f["mouth_height"] for f in history]
    areas = [f["mouth_area"] for f in history]
    separations = [f["lip_separation"] for f in history]
    width_var = float(np.var(widths))
    height_var = float(np.var(heights))
    area_var = float(np.var(areas))
    separation_var = float(np.var(separations))
    width_velocity = float(np.mean(np.abs(np.diff(widths)))) if len(widths) > 1 else 0.0
    height_velocity = float(np.mean(np.abs(np.diff(heights)))) if len(heights) > 1 else 0.0
    area_velocity = float(np.mean(np.abs(np.diff(areas)))) if len(areas) > 1 else 0.0
    mouth_motion_intensity = float((width_velocity + height_velocity + area_velocity) / 3.0)
    if len(areas) >= 4:
        autocorr = np.correlate(areas, areas, mode='full')
        mid = len(autocorr) // 2
        if mid > 0:
            autocorr_peaks = autocorr[mid+1:mid+min(len(areas)//2, 30)]
            if len(autocorr_peaks) > 0:
                cycle_strength = float(np.max(autocorr_peaks) / max(autocorr[mid], 1e-6))
            else:
                cycle_strength = 0.0
        else:
            cycle_strength = 0.0
    else:
        cycle_strength = 0.0
    motion_from_motion_module = motion.get("mouth_motion_score", 0.0)
    talking_from_motion = motion.get("talking_motion_score", 0.0)
    speech_activity_prob = float(np.clip(
        (mouth_motion_intensity * 0.4 + 
        cycle_strength * 0.3 + 
        talking_from_motion * 0.3) * 2.0,
        0.0, 1.0
    ))
else:
    width_var = height_var = area_var = separation_var = 0.0
    width_velocity = height_velocity = area_velocity = 0.0
    mouth_motion_intensity = 0.0
    cycle_strength = 0.0
    speech_activity_prob = motion.get("speech_activity_prob", 0.0)
```

#### 15. face_3d

#### 15.1. face_mesh_vector

```python
coords = data["coords"]  # (N, 3) - 3D landmarks
mean_coords = np.mean(coords[:, :3], axis=0)
centered_coords = coords[:, :3] - mean_coords
scale = np.std(centered_coords) + 1e-6
normalized_coords = centered_coords / scale
max_landmarks_for_mesh = min(n_landmarks, 200)
face_mesh_vector_3d = normalized_coords[:max_landmarks_for_mesh, :3].flatten()
if len(face_mesh_vector_3d) > 300:
    step = len(face_mesh_vector_3d) // 300
    face_mesh_vector_3d = face_mesh_vector_3d[::step][:300]
face_mesh_vector = face_mesh_vector_3d.tolist()
```

#### 15.2. identity_shape_vector

```python
n_landmarks = len(coords)
identity_landmarks = min(100, n_landmarks)
identity_coords = normalized_coords[:identity_landmarks, :3]
identity_shape_vector = [
    float(np.mean(identity_coords[:, 0])),
    float(np.std(identity_coords[:, 0])),
    float(np.mean(identity_coords[:, 1])),
    float(np.std(identity_coords[:, 1])),
    float(np.mean(identity_coords[:, 2])),
    float(np.std(identity_coords[:, 2])),
]
if identity_coords.shape[0] > 3:
    cov_matrix = np.cov(identity_coords.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    n_pcs = min(self.n_identity_params - 6, len(eigenvalues))
    if n_pcs > 0:
        sorted_idx = np.argsort(eigenvalues)[::-1]
        top_eigenvectors = eigenvectors[:, sorted_idx[:n_pcs]]
        projected = identity_coords @ top_eigenvectors  # исправлено
        identity_shape_vector.extend([float(x) for x in projected.flatten()[:n_pcs * 3]])
while len(identity_shape_vector) < self.n_identity_params:
    identity_shape_vector.append(0.0)
identity_shape_vector = identity_shape_vector[:self.n_identity_params]
```

#### 15.3. expression_vector

```python
expression_landmarks = min(150, n_landmarks)
expression_coords = normalized_coords[:expression_landmarks, :3]
neutral_coords = np.mean(expression_coords, axis=0)
expression_deviations = expression_coords - neutral_coords
expression_vector = [
    float(np.mean(expression_deviations[:, 0])),
    float(np.std(expression_deviations[:, 0])),
    float(np.mean(expression_deviations[:, 1])),
    float(np.std(expression_deviations[:, 1])),
    float(np.mean(expression_deviations[:, 2])),
    float(np.std(expression_deviations[:, 2])),
]
if expression_deviations.shape[0] > 3:
    expr_cov = np.cov(expression_deviations.T)
    expr_eigenvalues, expr_eigenvectors = np.linalg.eigh(expr_cov)
    n_expr_pcs = min(self.n_expression_params - 6, len(expr_eigenvalues))
    if n_expr_pcs > 0:
        sorted_idx = np.argsort(expr_eigenvalues)[::-1]
        top_eigenvectors = expr_eigenvectors[:, sorted_idx[:n_expr_pcs]]
        projected = expression_deviations @ top_eigenvectors  # исправлено
        expression_vector.extend([float(x) for x in projected.flatten()[:n_expr_pcs * 3]])
while len(expression_vector) < self.n_expression_params:
    expression_vector.append(0.0)
expression_vector = expression_vector[:self.n_expression_params]
```

#### 15.4. - 15.5. jaw_pose_vector, eye_pose_vector

```python
yaw = pose.get("yaw", 0.0)
pitch = pose.get("pitch", 0.0)
roll = pose.get("roll", 0.0)
jaw_pose_vector = [float(yaw), float(pitch), float(roll)]
eye_pose_vector = [float(pitch), float(roll)]
```

#### 15.6. mouth_shape_params

```python
mouth_landmarks_indices = [13, 14, 61, 291]
mouth_coords = [coords[idx, :3] for idx in mouth_landmarks_indices if idx < len(coords)]
if len(mouth_coords) >= 2:
    mouth_coords = np.array(mouth_coords)
    mouth_width = float(np.linalg.norm(mouth_coords[2] - mouth_coords[3]) if len(mouth_coords) >= 4 else 0.0)
    mouth_height = float(np.linalg.norm(mouth_coords[0] - mouth_coords[1]))
    mouth_depth = float(np.mean(mouth_coords[:, 2]))
    mouth_shape_params = [
        mouth_width,
        mouth_height,
        mouth_depth,
        float(mouth_width / max(mouth_height, 1e-6)),
    ]
else:
    mouth_shape_params = [0.0, 0.0, 0.0, 0.0]
```

#### 15.7. - 15.8. face_symmetry_score, face_uniqueness_score

```python
face_center_x = np.mean(normalized_coords[:, 0])
left_half = normalized_coords[normalized_coords[:, 0] < face_center_x]
right_half = normalized_coords[normalized_coords[:, 0] >= face_center_x]
if len(left_half) > 0 and len(right_half) > 0:
    right_half_mirrored = right_half.copy()
    right_half_mirrored[:, 0] = face_center_x - (right_half_mirrored[:, 0] - face_center_x)
    min_len = min(len(left_half), len(right_half_mirrored))
    if min_len > 0:
        left_sampled = left_half[:min_len]
        right_sampled = right_half_mirrored[:min_len]
        symmetry_error = np.mean(np.abs(left_sampled - right_sampled))
        symmetry_score = float(1.0 / (1.0 + symmetry_error))
    else:
        symmetry_score = 0.5
else:
    symmetry_score = 0.5
uniqueness_score = float(np.std(normalized_coords.flatten()))
```

#### 15.9. - 15.11. mesh_num_vertices, identity_params_count, expression_params_count

```python
mesh_num_vertices = len(face_mesh_vector) // 3
identity_params_count = len(identity_shape_vector)
expression_params_count = len(expression_vector)
```

## emotion_face

### Модели:

```py
from models.emonet.emonet.models.emonet import EmoNet

# Загрузка модели EmoNet
model = EmoNet(n_expression=8)  # 8 базовых эмоций Ekman
state = torch.load("emonet_8.pth", map_location="cpu")
if isinstance(state, dict):
    state = {k.replace("module.", ""): v for k, v in state.items()}
model.load_state_dict(state, strict=False)
model.eval()
```

Модуль использует:
- **EmoNet**: Модель для распознавания эмоций (8 классов: Neutral, Happy, Sad, Surprise, Fear, Disgust, Anger, Contempt)
- **InsightFace**: Для детекции лиц (используется результат из модуля face_detection)
- **Адаптивная обработка**: Сегментация видео, выборка кадров, валидация качества

### Выход:

```json
{
    "metadata": {
        "output": "/home/ilya/Рабочий стол/TrendFlowML/DataProcessor/VisualProcessor/result_store/emotion_face",
        "analysis_timestamp": "20251214_161559",
        "processing_version": "1.0"
    },
    "summary": {
        "total_frames_analyzed": 125,
        "keyframes_count": 27,
        "dominant_emotion": null,
        "is_static_face": false
    },
    "keyframes": [
        {
            "global_index": 195,
            "local_index": 29,
            "type": "emotion_peak",
            "emotion": {
                "valence": 0.042205810546875,
                "arousal": 0.54443359375,
                "emotions": {
                    "Neutral": 3.3974647521972656e-06,
                    "Happy": 0.0004107952117919922,
                    "Sad": 0.0,
                    "Surprise": 5.960464477539063e-08,
                    "Fear": 3.504753112792969e-05,
                    "Disgust": 8.225440979003906e-06,
                    "Anger": 0.99951171875,
                    "Contempt": 0.0
                }
            }
        },
        {
            "global_index": 635,
            "local_index": 95,
            "type": "emotion_peak",
            "emotion": {...}
        },
        ...
    ],
    "emotion_profile": {
        "dominant_emotion": "Unknown",
        "neutral_percentage": 0,
        "valence_avg": 0,
        "arousal_avg": 0
    },
    "quality_metrics": {
        "is_valid": true,
        "is_monotonic": false,
        "overall_score": 0.586,
        "log_message": "Стандартные требования",
        "metrics": {
        "diversity_score": 0.596,
        "transition_score": 0.0,
        "monotonicity_score": 0.836,
        "variance_score": 0.071,
        "different_emotions": 5,
        "significant_transitions": 0,
        "max_monotonic_streak": 42,
        "sequence_length": 256,
        "neutral_percentage": 0.592
        }
    },
    "processing_stats": {
        "total_frames": 816,
        "faces_found": 164,
        "segments": 11,
        "selected_frames": 125,
        "final_length": 256,
        "keyframes_count": 27,
        "attempt_number": 1,
        "success": true,
        "video_type": "DYNAMIC_FACES"
    },
    "is_monotonic": false,
    "is_valid": true
}
```

### Фичи:

#### 1. Базовые эмоции (emotions)

Вероятности 8 базовых эмоций Ekman для каждого кадра, вычисляемые через EmoNet:

```py
def predict_emonet_batch(frames, model, batch_size=None, temperature=1.0, face_confidence=None):
    # Предобработка кадров (RGB, нормализация)
    tensors = [preprocess(f) for f in frames]
    batch_tensor = torch.stack(tensors).to(device)
    
    # Инференс через EmoNet
    out = model(batch_tensor)
    
    # Извлечение результатов
    logits = out["expression"].detach()
    
    # Temperature scaling для калибровки вероятностей
    if temperature != 1.0:
        logits = logits / temperature
    
    probs = F.softmax(logits, dim=1).cpu().numpy()
    
    # Маппинг на классы эмоций
    EMOTION_CLASSES = {
        0: "Neutral", 1: "Happy", 2: "Sad", 3: "Surprise",
        4: "Fear", 5: "Disgust", 6: "Anger", 7: "Contempt"
    }
    
    # Вычисление confidence: max_softmax * face_confidence
    for j in range(len(frames)):
        max_prob = np.max(probs[j])
        detection_conf = face_confidence[j] if face_confidence else 1.0
        emotion_confidence = max_prob * detection_conf
        is_valid = detection_conf >= 0.3
    
    return {
        "emotions": {EMOTION_CLASSES[k]: float(probs[j][k]) for k in range(8)},
        "emotion_confidence": float(emotion_confidence),
        "is_valid": bool(is_valid)
    }
```

#### 2. Valence и Arousal

Непрерывные значения валентности (позитивность, -1.0 до 1.0) и активации (возбуждение, -1.0 до 1.0), извлекаемые напрямую из выходов EmoNet:

```py
vals = out["valence"].detach().cpu().numpy()  # Валентность
arous = out["arousal"].detach().cpu().numpy()  # Активация
```

#### 3. Keyframes (ключевые кадры)

Ключевые кадры с эмоциональными переходами или пиками, определяемые через анализ изменений эмоций:

```py
from scipy import signal
from scipy.ndimage import gaussian_filter1d

def detect_keyframes(emotion_curve, threshold=0.3, prominence=0.1, min_distance=8):
    # Gaussian smoothing (σ = 1-3 кадра) вместо простого moving average
    sigma = 2.0
    valence_smooth = gaussian_filter1d(emotion_curve['valence'], sigma=sigma, mode='nearest')
    arousal_smooth = gaussian_filter1d(emotion_curve['arousal'], sigma=sigma, mode='nearest')
    
    # Вычисление интенсивности
    intensity = np.sqrt(valence_smooth**2 + arousal_smooth**2)
    intensity_norm = (intensity - np.min(intensity)) / (np.max(intensity) - np.min(intensity))
    
    # Детекция пиков через scipy.signal.find_peaks
    peaks, _ = signal.find_peaks(
        intensity_norm,
        prominence=prominence,  # ≥ 0.1 в нормализованной шкале
        distance=min_distance    # 8-12 кадров между пиками
    )
    
    # Детекция переходов
    valence_diff = np.abs(np.diff(valence_smooth))
    arousal_diff = np.abs(np.diff(arousal_smooth))
    combined_diff = np.sqrt(valence_diff**2 + arousal_diff**2)
    combined_diff_norm = (combined_diff - np.min(combined_diff)) / (np.max(combined_diff) - np.min(combined_diff))
    
    transitions, _ = signal.find_peaks(
        combined_diff_norm,
        prominence=prominence,
        distance=min_distance
    )
    
    return {
        "type": "emotion_peak" or "transition",
        "global_index": frame_index,
        "local_index": local_index,
        "emotion": {...}  # Полные данные эмоции для этого кадра
    }
```

#### 4. Emotion Profile

Профиль эмоций видео, включающий доминантную эмоцию, распределение эмоций и средние значения:

```py
def analyze_emotion_profile(emo_results, use_weighted_means=True):
    # Weighted aggregation по confidence
    emotion_totals = {}
    valence_sum = 0
    arousal_sum = 0
    confidence_sum = 0
    
    for e in emo_results:
        conf = e.get('emotion_confidence', e.get('face_confidence', 1.0))
        if not use_weighted_means:
            conf = 1.0
        
        valence_sum += e["valence"] * conf
        arousal_sum += e["arousal"] * conf
        confidence_sum += conf
        
        dominant = max(e["emotions"].items(), key=lambda x: x[1])[0]
        emotion_totals[dominant] = emotion_totals.get(dominant, 0) + conf
    
    valence_avg = valence_sum / confidence_sum if confidence_sum > 0 else 0
    arousal_avg = arousal_sum / confidence_sum if confidence_sum > 0 else 0
    
    dominant_emotion = max(emotion_totals.items(), key=lambda x: x[1])[0] if emotion_totals else None
    neutral_percentage = emotion_totals.get('Neutral', 0) / sum(emotion_totals.values()) if emotion_totals else 0
    
    return {
        "dominant_emotion": dominant_emotion,
        "neutral_percentage": neutral_percentage,
        "valence_avg": valence_avg,
        "arousal_avg": arousal_avg,
        "valence_std": np.std([e["valence"] for e in emo_results]),
        "arousal_std": np.std([e["arousal"] for e in emo_results])
    }
```

#### 5. Quality Metrics

Метрики качества последовательности эмоций для валидации:

```py
def validate_sequence_quality(emotions, min_diversity_threshold=0.2):
    # Diversity score: разнообразие эмоций
    unique_emotions = set()
    for e in emotions:
        dominant = max(e["emotions"].items(), key=lambda x: x[1])[0]
        unique_emotions.add(dominant)
    diversity_score = len(unique_emotions) / 8.0  # Нормализация на 8 эмоций
    
    # Transition score: количество значимых переходов
    transitions = detect_keyframes(emotions, threshold=0.3)
    transition_score = len(transitions) / len(emotions)
    
    # Monotonicity score: оценка монотонности (1.0 = полностью монотонно)
    valence_std = np.std([e["valence"] for e in emotions])
    arousal_std = np.std([e["arousal"] for e in emotions])
    monotonicity_score = 1.0 - min(1.0, (valence_std + arousal_std) / 2.0)
    
    # Variance score: вариативность эмоций
    variance_score = np.var([max(e["emotions"].values()) for e in emotions])
    
    # Overall score: комбинированная оценка
    overall_score = (diversity_score + transition_score + (1 - monotonicity_score) + variance_score) / 4.0
    
    return {
        "diversity_score": diversity_score,
        "transition_score": transition_score,
        "monotonicity_score": monotonicity_score,
        "variance_score": variance_score,
        "overall_score": overall_score,
        "is_valid": overall_score >= quality_threshold
    }
```

#### 6. Advanced Features (расширенные фичи)

##### 6.1. Microexpressions (микроэмоции)

Резкие эмоциональные изменения длительностью 0.03-0.5 секунды:

```py
def detect_micro_expressions(emotions, fps=30.0, min_frames=2, change_threshold=None):
    # Вычисление изменений между кадрами
    valence_changes = [abs(emotions[i]["valence"] - emotions[i-1]["valence"]) for i in range(1, len(emotions))]
    arousal_changes = [abs(emotions[i]["arousal"] - emotions[i-1]["arousal"]) for i in range(1, len(emotions))]
    
    # Комбинированное изменение: sqrt(Δv² + Δa²) + emotion_change
    combined_changes = [
        np.sqrt(v**2 + a**2) + e * 0.5
        for v, a, e in zip(valence_changes, arousal_changes, emotion_changes)
    ]
    
    # Adaptive threshold: 85th percentile если threshold не задан
    if change_threshold is None:
        change_threshold = np.percentile(combined_changes, 85)
    
    # Детекция резких изменений (микроэмоций)
    # min_frames: требуется ≥2 кадров (на 30fps 0.03s ≈ 1 кадр слишком мало)
    min_duration_frames = max(min_frames, int(0.03 * fps))
    max_duration_frames = min(int(0.5 * fps), 15)  # Cap at 15 frames
    
    # Поиск интервалов с резкими изменениями
    microexpressions = []
    # ... детекция интервалов с combined_change >= change_threshold
    # длительностью между min_duration_frames и max_duration_frames
    
    return {
        "microexpressions_count": len(microexpressions),
        "microexpression_rate": len(microexpressions) / total_duration_sec,
        "avg_duration": np.mean([m["duration_sec"] for m in microexpressions])
    }
```

##### 6.2. Physiological Signals (физиологические сигналы)

⚠️ **ВАЖНО**: Эти метрики основаны на эвристических правилах и не являются валидированными клиническими индикаторами. Для продакшн-использования рекомендуется обучить learned meta-model (X → label) с размеченными данными или weak supervision.

Оценка стресса, уверенности, нервозности и напряжения:

```py
# ⚠️ NOTE: Heuristic-based scores. For production, consider training a learned meta-model.
def compute_physiological_signals(emotions):
    valence = np.array([e["valence"] for e in emotions])
    arousal = np.array([e["arousal"] for e in emotions])
    
    # Stress: высокий arousal + отрицательная валентность + страх/гнев
    stress_score = np.mean((arousal > 0.3) & (valence < 0)) + \
                   np.mean([e["emotions"]["Fear"] + e["emotions"]["Anger"] for e in emotions])
    
    # Confidence: положительная валентность + умеренный arousal + счастье/нейтральность
    confidence_score = np.mean(valence > 0.2) + \
                       np.mean((arousal > -0.2) & (arousal < 0.5)) + \
                       np.mean([e["emotions"]["Happy"] + e["emotions"]["Neutral"] for e in emotions])
    
    # Nervousness: высокая вариативность + высокий arousal + страх/удивление
    variability = (np.var(valence) + np.var(arousal)) / 2.0
    nervousness_score = min(1.0, variability * 3.0) + \
                        np.mean(arousal > 0.4) + \
                        np.mean([e["emotions"]["Fear"] + e["emotions"]["Surprise"] for e in emotions])
    
    return {
        "stress_level_score": stress_score / 3.0,  # ⚠️ Heuristic-based, not validated
        "confidence_face_score": confidence_score / 3.0,  # ⚠️ Heuristic-based, not validated
        "nervousness_score": nervousness_score / 3.0  # ⚠️ Heuristic-based, not validated
    }
```

##### 6.3. Face Asymmetry (асимметрия лица)

Анализ симметрии лица (требует landmarks от face_detection):

```py
def compute_face_asymmetry(landmarks):
    # Сравнение левой и правой половин лица
    # Анализ асимметрии бровей, глаз, рта
    
    return {
        "asymmetry_score": float,
        "sincerity_score": float  # ⚠️ AUDIT-ONLY / RESEARCH: Not validated for production use
    }
```

⚠️ **sincerity_score**: Эта метрика является **псевдопсихологической** и **НЕ должна использоваться в продакшн-моделях** без клинической валидации и юридической проверки. Используется только для исследовательских целей или аудита.

##### 6.4. Emotional Individuality (индивидуальность выражения)

Анализ стиля и интенсивности выражения эмоций:

```py
def compute_emotional_individuality(emotions):
    # Базовая интенсивность эмоций
    intensity_baseline = np.mean([max(e["emotions"].values()) for e in emotions])
    
    # Индекс выразительности
    expressivity = np.std([max(e["emotions"].values()) for e in emotions])
    
    # Эмоциональный диапазон
    emotional_range = max([max(e["emotions"].values()) for e in emotions]) - \
                      min([max(e["emotions"].values()) for e in emotions])
    
    return {
        "emotional_intensity_baseline": intensity_baseline,
        "expressivity_index": expressivity,
        "emotional_range": emotional_range
    }
```

#### 7. Processing Pipeline

Модуль использует адаптивную обработку с сегментацией и выборкой кадров:

```py
# 1. Сегментация видео на сегменты с лицами
segments = segmentation(timeline, fps, max_gap_seconds=0.5)

# 2. Адаптивная выборка кадров из сегментов
selected_indices = select_from_segments(segments, total_frames, fps, max_samples_per_segment=10)

# 3. Обработка эмоций батчами
emo_results = process_frames_in_batches(frame_manager, selected_indices, model, batch_size_load=50, batch_size_process=16)

# 4. Детекция ключевых кадров
keyframes = detect_keyframes(emo_results, threshold=0.3)

# 5. Валидация и нормализация до target_length (256)
validated_sequence = validate_sequence_quality(emo_results, min_diversity_threshold=0.2)
```

## behavioral

### Модели:

```python
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_hands = mp.solutions.hands
mp_face_mesh = mp.solutions.face_mesh

pose = mp_pose.Pose(
    static_image_mode=True,
    model_complexity=pose_model_complexity,
    smooth_landmarks=pose_smooth_landmarks,
    min_detection_confidence=pose_min_detection_confidence,
    min_tracking_confidence=pose_min_tracking_confidence
)

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=hands_max_num_hands,
    model_complexity=hands_model_complexity,
    min_detection_confidence=hands_min_detection_confidence,
    min_tracking_confidence=hands_min_tracking_confidence
)

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=face_max_num_faces,
    refine_landmarks=face_refine_landmarks,
    min_detection_confidence=face_min_detection_confidence,
    min_tracking_confidence=face_min_tracking_confidence
)
```

### Выход (пример, упрощённый):

```json
{
  "frame_results": {
    "0": {
      "hand_gestures": ["open_palm"],
      "num_hands": 2,
      "body_language": {
        "posture": "standing",
        "arm_openness": 1.3,
        "pose_expansion": 0.18,
        "body_lean_angle": 0.2,
        "balance_offset": -0.05,
        "shoulder_angle": 1.7
      },
      "speech_behavior": {
        "mouth_width_norm": 0.12,
        "mouth_height_norm": 0.03,
        "mouth_area_norm": 0.004,
        "mouth_velocity": 35.2,
        "mouth_open_ratio": 0.25,
        "speech_activity_proxy": 0.71
      },
      "stress": {
        "blink_flag": 0,
        "blink_rate_short": 0.1,
        "self_touch_flag": 0,
        "fidgeting_energy": 0.0004
      },
      "sequence_features": {
        "num_hands": 2,
        "hands_visibility": 1,
        "hand_motion_energy": 12.5,
        "gesture_probs": {
          "pointing": 0.05,
          "open_palm": 0.7,
          "fist": 0.1,
          "thumbs_up": 0.05,
          "thumbs_down": 0.0,
          "victory": 0.0,
          "ok": 0.05,
          "rock": 0.0,
          "call_me": 0.0,
          "love": 0.05
        },
        "arm_openness": 1.3,
        "pose_expansion": 0.18,
        "body_lean_angle": 0.2,
        "balance_offset": -0.05,
        "shoulder_angle": 1.7,
        "shoulder_angle_velocity": 0.03,
        "head_position_x_norm": 0.48,
        "head_position_y_norm": 0.32,
        "head_motion_energy": 4.1,
        "head_stability": 0.19,
        "mouth_width_norm": 0.12,
        "mouth_height_norm": 0.03,
        "mouth_area_norm": 0.004,
        "mouth_velocity": 35.2,
        "mouth_open_ratio": 0.25,
        "speech_activity_proxy": 0.71,
        "blink_flag": 0,
        "blink_rate_short": 0.1,
        "self_touch_flag": 0,
        "fidgeting_energy": 0.0004,
        "timestamp_norm": 0.03
      },
      "timestamp": 0.12
    },
    "5": { "...": "..." }
  },
  "aggregated": {
    "avg_engagement": 0.62,
    "max_engagement": 0.91,
    "engagement_variance": 0.04,
    "engagement_peaks": 3,
    "early_engagement_mean": 0.55,
    "late_engagement_mean": 0.68,

    "avg_confidence": 0.58,
    "max_confidence": 0.87,
    "confidence_variance": 0.03,
    "confidence_peak_count": 2,

    "avg_stress": 0.31,
    "max_stress": 0.74,
    "stress_spike_count": 4,
    "stress_duration_ratio": 0.27,

    "gesture_rate_per_sec": 1.8,
    "gesture_entropy_mean": 2.3,
    "dominant_gesture_ratio": 0.55,
    "gesture_switching_rate": 0.4,
    "gesture_counts": {
      "pointing": 12,
      "open_palm": 45,
      "thumbs_up": 8
    },

    "avg_arm_openness": 1.1,
    "avg_pose_expansion": 0.16,
    "body_motion_energy_mean": 3.2,
    "body_motion_energy_var": 1.1,

    "speech_activity_ratio": 0.64,
    "speech_burstiness": 0.85,
    "mouth_rhythm_score": 0.19,

    "engagement_contrast": 0.29,
    "confidence_contrast": 0.24,
    "stress_contrast": 0.33,
    "early_late_ratios": {
      "engagement": 1.23,
      "speech_activity": 1.15,
      "gesture_rate": 1.05
    },

    "hands_visibility_ratio": 0.78,
    "face_visibility_ratio": 0.96,
    "center_bias_mean": 0.18
  }
}
```

### Фичи

#### 1. frame_results / sequence_features

Модуль возвращает для каждого кадра:

- **`hand_gestures`** — список распознанных жестов (hard‑labels, для дебага).
- **`body_language` / `speech_behavior` / `stress`** — сырые промежуточные структуры (для анализа и отладки).
- **`sequence_features`** — компактный словарь числовых каналов, который и подаётся в VisualTransformer.

Ключевые группы признаков в `sequence_features`:

- **Hand & Gesture Dynamics**
  - `num_hands`, `hands_visibility`, `hand_motion_energy`.
  - `gesture_probs` — распределение по жестам: `pointing`, `open_palm`, `fist`, `thumbs_up`, `thumbs_down`, `victory`, `ok`, `rock`, `call_me`, `love`.
- **Arm & Body Openness**
  - `arm_openness = wrist_distance / shoulder_width`.
  - `pose_expansion = person_bbox_area / frame_area`.
- **Body Orientation & Stability**
  - `body_lean_angle` \([-1,1]\), `balance_offset` \([-1,1]`, `shoulder_angle`, `shoulder_angle_velocity`.
- **Head & Gaze Proxies**
  - `head_position_x_norm`, `head_position_y_norm`, `head_motion_energy`, `head_stability`.
- **Speech / Mouth Dynamics**
  - `mouth_width_norm`, `mouth_height_norm`, `mouth_area_norm`, `mouth_velocity`, `mouth_open_ratio`, `speech_activity_proxy`.
- **Stress RAW Signals**
  - `blink_flag`, `blink_rate_short`, `self_touch_flag`, `fidgeting_energy`.
- **Temporal Context**
  - `timestamp_norm = t / video_duration`.

Все эти признаки являются **сырой физикой/геометрией**, без frame‑level индексов вовлечённости/уверенности/стресса.

#### 2. aggregated

Агрегированные признаки строятся **только** из `sequence_features`:

- **Engagement**: `avg_engagement`, `max_engagement`, `engagement_variance`, `engagement_peaks`, `early_engagement_mean`, `late_engagement_mean`.
- **Confidence / Dominance**: `avg_confidence`, `max_confidence`, `confidence_variance`, `confidence_peak_count`.
- **Stress Summary**: `avg_stress`, `max_stress`, `stress_spike_count`, `stress_duration_ratio`.
- **Gesture Statistics**: `gesture_rate_per_sec`, `gesture_entropy_mean`, `dominant_gesture_ratio`, `gesture_switching_rate`, `gesture_counts`.
- **Body & Motion Summary**: `avg_arm_openness`, `avg_pose_expansion`, `body_motion_energy_mean`, `body_motion_energy_var`.
- **Speech / Rhythm Summary**: `speech_activity_ratio`, `speech_burstiness`, `mouth_rhythm_score`.
- **Temporal Contrast**: `engagement_contrast`, `confidence_contrast`, `stress_contrast`, `early_late_ratios`.
- **Presence & Framing**: `hands_visibility_ratio`, `face_visibility_ratio`, `center_bias_mean`.

Frame‑level `engagement_score / confidence_score / dominance_score / anxiety_score`, а также агрегаты `posture_statistics` **больше не считаются** и не должны использоваться downstream‑модулями.

## optical_flow

> Обновление: покадровый вектор сжат до ~12 численных признаков; углы считаем через sin/cos, moving_pixels теперь в px/sec + адаптивный порог, добавлены forward-backward (fb) метрики качества потока и occlusion_fraction. Строковые категории убраны из последовательности.

### Покадровый вектор (для трансформера)
- `magnitude_mean_px_sec_norm`, `magnitude_std_px_sec_norm`, `magnitude_p95_px_sec_norm`
- `dir_sin_mean`, `dir_cos_mean`, `dir_dispersion`, `direction_std_circular`
- `dx_mean_norm`, `dy_mean_norm`
- `moving_pixels_1.0` (px/sec), `moving_pixels_rel` (median+K*MAD)
- `flow_confidence_mean_norm`, `fb_error_mean_norm`, `occlusion_fraction`, `flow_consistency`

### Основные обновления расчётов
- Направления: sin/cos, результирующая длина, circular entropy (36 bins, ln, Laplace).
- Motion thresholds: px/sec (учёт fps и frame_skip), адаптивный порог median+K*MAD.
- Качество флоу: forward-backward error (`fb_error_mean`, `fb_error_fraction`), `flow_confidence_mean = 1/(1+fb_error)`, `occlusion_fraction`, `valid_ratio`.
- MEI/MHI: decay учитывает fps (`decay = exp(-dt/tau)`).
- Spatial grid: multi-scale 1x1, 4x4, 8x8 с числовой `direction_histogram`; ROI отдаёт `direction_hist_summed`.
- Periodicity: detrend + Hann window перед FFT, значимость к медиане мощности.

### Модели:

```py
import torchvision.models.optical_flow as models

if model_type == "large":
    model = models.raft_large(
        weights=models.Raft_Large_Weights.DEFAULT,
        progress=True
    )
else:
    model = models.raft_small(
        weights=models.Raft_Small_Weights.DEFAULT,
        progress=True
    )
```

### Выход:

```json
{
    "analysis_info": {
        "version": "1.0.0",
        "timestamp": "2025-12-14T16:18:14.149175"
    },
    "processing_info": {
        "total_frames_analyzed": 10,
        "frames_with_spatial_analysis": 1,
        "analysis_duration_seconds": 2e-05
    },
    "statistics": {
        "frame_statistics": [
            {
                "magnitude_mean": 2289.622314453125,
                "magnitude_std": 1044.5616455078125,
                "magnitude_max": 9501.55859375,
                "magnitude_min": 6.5795159339904785,
                "magnitude_median": 2036.3526000976562,
                "magnitude_iqr": 869.2304382324219,
                "magnitude_p90": 3678.285595703125,
                "magnitude_p95": 4545.842163085936,
                "direction_mean": -2.953230857849121,
                "direction_std": 0.47795698046684265,
                "direction_entropy": 3.3794058255641497,
                "dx_mean": -2077.046142578125,
                "dy_mean": -341.9216613769531,
                "dx_std": 1120.7344970703125,
                "dy_std": 803.9885864257812,
                "dx_abs_mean": 2088.0634765625,
                "dy_abs_mean": 731.2765502929688,
                "moving_pixels_0.5": 1.0,
                "moving_pixels_1.0": 1.0,
                "moving_pixels_2.0": 1.0,
                "magnitude_skew": 1.7333471775054932,
                "magnitude_kurtosis": 4.330869197845459,
                "spatial_gradient": 18.3243408203125,
                "flow_consistency": 0.07220868021249771,
                "flow_divergence_mean": -0.22143946588039398,
                "motion_intensity": "very_high",
                "dominant_direction": "left",
                "frame_shape": "(1920, 1080)",
                "pixel_count": 2073600,
                "flow_filename": "flow_000000.pt",
                "frame_index": 0,
                "camera_motion_mean": 2289.622314453125,
                "camera_motion_std": 1044.5616455078125,
                "camera_motion_max": 9501.55859375,
                "camera_motion_energy": 13133094584320.0,
                "camera_motion_entropy": 2.342425623521644,
                "camera_shake_var": 0.0,
                "camera_shake_mean": 0.0,
                "camera_shake_max": 0.0,
                "camera_affine_scale": 0.0,
                "camera_affine_rotation": 0.0,
                "camera_affine_tx": 0.0,
                "camera_affine_ty": 0.0,
                "camera_background_ratio": 0.0,
                "camera_rotation_speed": 0.0
            },
            {...},
        ],
        "spatial_analysis": {
            "frame_000000": {
                "regional_stats": [
                    {
                        "region_id": "R0_0",
                        "grid_position": "0,0",
                        "pixel_coords": "0:480,0:270",
                        "region_size": 129600,
                        "region_magnitude_mean": 1430.4569091796875,
                        "region_magnitude_std": 461.14105224609375,
                        "region_dx_mean": -932.2562866210938,
                        "region_dy_mean": -917.7066650390625,
                        "relative_activity": 0.6247566938400269,
                        "motion_dominance": 0.04173611111111111,
                        "region_gradient": 16.52753257751465,
                        "flow_divergence": 3.9261422157287598,
                        "activity_level": "low",
                        "direction_category": "diagonal_up_left"
                    },
                    {
                        "region_id": "R0_1",
                        ...
                    },
                    {...}
                ],
                "roi_analysis": {
                    "top_regions": [
                        "R2_2",
                        "R2_3",
                        "R1_3"
                    ],
                    "activity_concentration": 0.2792715451470355,
                    "spatial_distribution": "concentrated",
                    "dominant_directions": {
                        "horizontal_left": 3
                    }
                },
                "flow_filename": "flow_000000.pt"
            }
        },
        "temporal_analysis": {
            "trends": {
                "magnitude": {
                    "slope": -15.445449366714104,
                    "trend_type": "decreasing",
                    "mean": 1520.1704711914062,
                    "std": 359.28048600672884,
                    "range": 1157.2615966796875,
                    "has_trend": "True"
                },
                "moving_pixels": {
                    "slope": -1.9029915270644353e-16,
                    "trend_type": "stable",
                    "mean": 0.9999999035493827,
                    "std": 1.9290123458581833e-07,
                    "range": 4.822530864645458e-07,
                    "has_trend": "False"
                },
                "direction_std": {...},
                "spatial_gradient": {...}
            },
            "periodicity": {
                "magnitude": {
                    "has_periodicity": false,
                    "reason": "insufficient_data"
                },
                "moving_pixels": {
                    "has_periodicity": false,
                    "reason": "insufficient_data"
                },
                "direction_std": {...},
                "spatial_gradient": {...}
            },
            "transitions": {
                "magnitude": {
                    "transition_count": 0
                },
                "moving_pixels": {
                    "transition_count": 0
                },
                "direction_std": {...},
                "spatial_gradient": {...}
            },
            "segments": {
                "segments": [
                    {
                        "start_frame": 0,
                        "end_frame": 1,
                        "length_frames": 1,
                        "mean_magnitude": 2289.622314453125,
                        "std_magnitude": 0.0,
                        "cluster_label": 2
                    },
                    {
                        "start_frame": 1,
                        "end_frame": 4,
                        ...
                    },
                    {...}
                ],
                "boundary_frames": [1, 4, 6, 8, 9],
                "method": "kmeans"
            },
            "summary": {
                "total_duration_seconds": 1.8,
                "avg_magnitude": 1520.1704711914062,
                "magnitude_variability": 0.23634223451605765,
                "activity_peaks_count": 0,
                "stability_score": 0.9008772214806994
            }
        },
        "summary_metrics": {
            "overall_magnitude_mean": 1520.1704711914062,
            "overall_magnitude_std": 378.71488487783887,
            "activity_variability": 2.0333575491654438e-07,
            "dominant_motion_intensity": "very_high",
            "dominant_direction": "left",
            "temporal_stability": 0.9008772214806994,
            "peak_activity_frames": 0,
            "dominant_trend": "decreasing",
            "has_periodicity": false,
            "transition_count": 0
        },
        "camera_motion": {
            "summary": {
                "motion_mean_mean": 1520.1704711914062,
                "motion_mean_std": 359.28048600672884,
                "motion_mean_max": 2289.622314453125,
                "motion_mean_min": 1132.3607177734375,
                "motion_std_mean": 796.5256469726562,
                "motion_std_std": 133.45570201460475,
                "motion_std_max": 1044.5616455078125,
                "motion_std_min": 591.6805419921875,
                "motion_energy_sum": 64121176326144.0,
                "motion_entropy_mean": 2.4576866470164473,
                "shake_mean": 0.0,
                "shake_std": 0.0,
                "shake_max": 0.0,
                "zoom_in_count": 0,
                "zoom_out_count": 0,
                "zoom_speed_mean": 0.0,
                "rotation_speed_mean": 0.0,
                "rotation_speed_std": 0.0,
                "pan_ratio": 0.0,
                "truck_ratio": 0.0,
                "pedestal_ratio": 0.0,
                "static_ratio": 1.0,
                "chaos_index": 4.760271739686495,
                "style_handheld": 0.0,
                "style_tripod": 1.0,
                "style_cinematic": 1.0,
                "style_drone": 0.47602717396864946,
                "style_action_cam": 1.0,
                "n_frames": 10
            },
            "per_frame": [
                {
                    "motion_mean": 2289.622314453125,
                    "motion_std": 1044.5616455078125,
                    "motion_max": 9501.55859375,
                    "motion_energy": 13133094584320.0,
                    "motion_entropy": 2.342425623521644,
                    "shake_var": 0.0,
                    "shake_mean": 0.0,
                    "shake_max": 0.0,
                    "affine_scale": 0.0,
                    "affine_rotation": 0.0,
                    "affine_tx": 0.0,
                    "affine_ty": 0.0,
                    "background_ratio": 0.0,
                    "rotation_speed": 0.0
                },
                {
                    "motion_mean": 1431.7615966796875,
                    ...
                },
                {...},
            ]
        },
        "advanced_features": {
            "motion_energy_image": {
                "features": {
                    "mei_total_energy": 2066193.0,
                    "mei_coverage_ratio": 0.9964279532432556,
                    "mei_max_energy": 1.0,
                    "mei_std": 0.05965975299477577,
                    "mhi_contrast": 1.0,
                    "mhi_entropy": 1.4952185011992494,
                    "mhi_mean": 0.9273028373718262,
                    "mhi_max": 1.0,
                    "motion_persistence": 0.9471952160493827
                },
                "mei_shape": [1920, 1080]
            },
            "foreground_background_motion": {
                "summary": {
                    "foreground_motion_energy_mean": 6412117842329.6,
                    "background_motion_energy_mean": 0.03635804653167725,
                    "ratio_foreground_background_mean": 5693603682713.6,
                    "foreground_coverage_mean": 0.9999999035493827
                },
                "per_frame_count": 10
            },
            "motion_clusters": {
                "summary": {
                    "num_clusters_mean": 5.0,
                    "largest_cluster_coverage_mean": 0.3808238811728395,
                    "cluster_diversity_mean": 1.0
                },
                "per_frame_count": 10
            },
            "smoothness_jerkiness": {
                "smoothness_index": 0.003525036619976163,
                "jerkiness_index": 289.8276062011719,
                "flow_temporal_entropy": 3.169925000124696,
                "movement_stability": 0.8088375329971313,
                "mean_acceleration": 282.68499755859375,
                "std_acceleration": 286.2464904785156,
                "max_acceleration": 857.8607177734375,
                "mean_jerk": 289.8276062011719,
                "max_jerk": 740.99072265625
            }
        }
    }
}
```

### Фичи:

#### Общее:

```py
# model - raft model

list_of_flows = model(
    frame1,
    frame2
)
flow_tensor = list_of_flows[-1].squeeze(0)
```

#### 1. analysis_info

#### 1.1. version

#### 1.2. timestamp

#### 2. processing_info

#### 2.1. total_frames_analyzed

#### 2.2. frames_with_spatial_analysis

#### 2.3. analysis_duration_seconds

#### 3. statistics

#### 3.1. frame_statistics

#### 3.1.1. - 3.1.8. magnitude_mean, magnitude_std, magnitude_max, magnitude_min, magnitude_median, magnitude_iqr, magnitude_p90, magnitude_p95

```py
# flow_tensor: Тензор [2, H, W]

dx = flow_tensor[0].numpy().astype(np.float32)
dy = flow_tensor[1].numpy().astype(np.float32)

magnitude = np.sqrt(dx**2 + dy**2)

flat_mag = magnitude.flatten()
percentiles = np.percentile(flat_mag, [25, 50, 75, 90, 95])

return {
    'magnitude_mean': float(np.mean(flat_mag)),
    'magnitude_std': float(np.std(flat_mag)),
    'magnitude_max': float(np.max(flat_mag)),
    'magnitude_min': float(np.min(flat_mag)),
    'magnitude_median': float(percentiles[1]),
    'magnitude_iqr': float(percentiles[2] - percentiles[0]),
    'magnitude_p90': float(percentiles[3]),
    'magnitude_p95': float(percentiles[4])
}
```

#### 3.1.9 - 3.1.11. direction_mean, direction_std, direction_entropy

```py
direction_bins = 36

direction = np.arctan2(dy, dx)

flat_dir = direction.flatten()
        
def _circular_mean(angles: np.ndarray):
    sin_sum = np.mean(np.sin(angles))
    cos_sum = np.mean(np.cos(angles))
    return np.arctan2(sin_sum, cos_sum)

def _circular_std(angles: np.ndarray):
    R = np.sqrt(np.mean(np.sin(angles))**2 + np.mean(np.cos(angles))**2)
    return np.sqrt(-2 * np.log(R + 1e-10))

def _directional_entropy(angles: np.ndarray, bins: int):
    try:
        hist, _ = np.histogram(angles, bins=bins, range=(-np.pi, np.pi))
        hist = hist / (hist.sum() + 1e-10)
        hist = hist[hist > 0]
        return float(-np.sum(hist * np.log2(hist + 1e-10)))
    except:
        return 0.0

return {
    'direction_mean': float(_circular_mean(flat_dir)),
    'direction_std': float(_circular_std(flat_dir)),
    'direction_entropy': float(_directional_entropy(flat_dir, direction_bins))
}
```

#### 3.1.12. - 3.1.17. dx_mean, dy_mean, dx_std, dy_std, dx_abs_mean, dy_abs_mean

```py
return {
    'dx_mean': float(np.mean(dx)),
    'dy_mean': float(np.mean(dy)),
    'dx_std': float(np.std(dx)),
    'dy_std': float(np.std(dy)),
    'dx_abs_mean': float(np.mean(np.abs(dx))),
    'dy_abs_mean': float(np.mean(np.abs(dy)))
}
```

#### 3.1.18. - 3.1.20. moving_pixels_0, moving_pixels_1, moving_pixels_2

```py
motion_thresholds = [0, 1.0, 2.0]

magnitude = np.sqrt(dx**2 + dy**2)

stats = {}
for threshold in motion_thresholds:
    moving_pixels = np.sum(magnitude > threshold) / magnitude.size
    stats[f'moving_pixels_{threshold}'] = float(moving_pixels)
```

#### 3.1.21. - 3.1.22. magnitude_skew, magnitude_kurtosis

```py
from scipy import stats

flat_mag = magnitude.flatten()
if len(flat_mag) < 4:
    return {'magnitude_skew': 0.0, 'magnitude_kurtosis': 0.0}
try:
    return {
        'magnitude_skew': float(stats.skew(flat_mag)),
        'magnitude_kurtosis': float(stats.kurtosis(flat_mag))
    }
except:
    return {'magnitude_skew': 0.0, 'magnitude_kurtosis': 0.0}
```

#### 3.1.23. - 3.1.25. spatial_gradient, flow_consistency, flow_divergence_mean

```py
try:
    grad_y, grad_x = np.gradient(magnitude)
    spatial_gradient = np.mean(np.sqrt(grad_x**2 + grad_y**2))
    div = np.gradient(dx, axis=1) + np.gradient(dy, axis=0)
    flow_consistency = 1.0 / (1.0 + np.mean(np.abs(div)))
    return {
        'spatial_gradient': float(spatial_gradient),
        'flow_consistency': float(flow_consistency),
        'flow_divergence_mean': float(np.mean(div))
    }
except:
    return {
        'spatial_gradient': 0.0,
        'flow_consistency': 0.0,
        'flow_divergence_mean': 0.0
    }
```

#### 3.1.26. - 3.1.27. motion_intensity, dominant_direction

```py
def _categorize_intensity(magnitude: np.ndarray):
    mean_mag = np.mean(magnitude)
    if mean_mag < 0.5:
        return 'very_low'
    elif mean_mag < 1.0:
        return 'low'
    elif mean_mag < 2.0:
        return 'medium'
    elif mean_mag < 5.0:
        return 'high'
    else:
        return 'very_high'

def _categorize_direction(direction: np.ndarray):
    deg = np.degrees(direction) % 360
    bins = np.arange(0, 361, 45)
    hist, _ = np.histogram(deg, bins=bins)
    
    if np.sum(hist) == 0:
        return 'undefined'
    
    dominant_bin = np.argmax(hist)
    directions = [
        'right', 'up_right', 'up', 'up_left', 
        'left', 'down_left', 'down', 'down_right'
    ]
    return directions[dominant_bin % len(directions)]

return {
    'motion_intensity': _categorize_intensity(magnitude),
    'dominant_direction': _categorize_direction(direction)
}
```

#### 3.1.28. - 3.1.29. frame_shape, pixel_count

```py
stats_dict.update({
    'frame_shape': f"{dx.shape}",
    'pixel_count': int(magnitude.size)
})
```

#### 3.1.30 camera_motion_mean

Среднее значение движения камеры (величина движения фона). Доступно только при `enable_camera_motion=True`.

```py
def _safe_float(x):
    try:
        xf = float(x)
        if math.isnan(xf) or math.isinf(xf):
            return 0.0
        return xf
    except Exception:
        return 0.0

def flow_magnitude_angle(flow: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    dx = flow[..., 0]
    dy = flow[..., 1]
    mag = np.sqrt(dx * dx + dy * dy)
    ang = np.arctan2(dy, dx) 
    return mag, ang

mag, ang = flow_magnitude_angle(flow_tensor)

flat = mag.ravel()

motion_mean = _safe_float(np.mean(flat)) if flat.size else 0.0
```

#### 3.1.31. camera_motion_std

Стандартное отклонение движения камеры.

#### 3.1.32. camera_motion_max

Максимальное значение движения камеры.

#### 3.1.33. camera_motion_energy

Энергия движения камеры (сумма квадратов величин движения).

#### 3.1.34. camera_motion_entropy

Энтропия движения камеры, показывающая разнообразие направлений.

```py
motion_std = _safe_float(np.std(flat)) if flat.size else 0.0
motion_max = _safe_float(np.max(flat)) if flat.size else 0.0
motion_energy = _safe_float(np.sum(flat ** 2))

try:
    hist, _ = np.histogram(ang.ravel(), bins=36, range=(-math.pi, math.pi))
    p = hist / (hist.sum() + 1e-9)
    motion_entropy = _safe_float(-np.sum([float(x) * math.log(float(x) + 1e-12) for x in p if x > 0]))
except Exception:
    motion_entropy = 0.0
```

#### 3.1.35. camera_shake_var

Дисперсия тряски камеры (вариативность движения фона).

#### 3.1.36. camera_shake_mean

Среднее значение тряски камеры.

#### 3.1.37. camera_shake_max

Максимальное значение тряски камеры.

```py
mag_thresh = 0.5

def background_mask_by_magnitude(flow: np.ndarray, mag_thresh: float = 0.5) -> np.ndarray:
    mag, _ = flow_magnitude_angle(flow)
    return mag <= mag_thresh

bg_mask = background_mask_by_magnitude(flow_tensor)

try:
    mag, _ = flow_magnitude_angle(flow)
    if background_mask is not None:
        arr = mag[background_mask]
    else:
        arr = mag.ravel()
    shakiness = dict(shake_var=float(np.var(arr)), shake_mean=float(np.mean(arr)), shake_max=float(np.max(arr)))
except Exception:
    shakiness = dict(shake_var=0.0, shake_mean=0.0, shake_max=0.0)
```

#### 3.1.38. camera_affine_scale

Масштаб аффинного преобразования камеры (zoom). Значение > 1.0 означает приближение, < 1.0 — отдаление.

#### 3.1.39. camera_affine_rotation

Угол поворота камеры в радианах.

#### 3.1.40. camera_affine_tx

Горизонтальное смещение камеры в пикселях.

#### 3.1.41. camera_affine_ty

Вертикальное смещение камеры в пикселях.

```py

def estimate_affine_from_flow(flow_tensor, bg_mask)
    sample_n = 2000
    ransac_thresh = 3.0

    h, w, _ = flow_tensor.shape
    ys, xs = np.mgrid[0:h, 0:w]
    pts = np.stack([xs.ravel().astype(np.float32), ys.ravel().astype(np.float32)], axis=-1)
    disp = flow_tensor.reshape(-1, 2).astype(np.float32)
    if bg_mask is not None:
        m = bg_mask.ravel().astype(bool)
        if m.sum() < 10:
            return None
        pts = pts[m]
        disp = disp[m]
    if pts.shape[0] > sample_n:
        idx = np.random.choice(pts.shape[0], sample_n, replace=False)
        pts = pts[idx]
        disp = disp[idx]
    src = pts
    dst = pts + disp
    M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh)
    return M

M = estimate_affine_from_flow(flow_tensor, bg_mask)

def decompose_affine(M)
    if M is None:
        return dict(scale=np.nan, rotation=np.nan, tx=np.nan, ty=np.nan)

    a, b, tx = M[0]
    c, d, ty = M[1]
    scale = math.sqrt(a * a + b * b)

    rotation = math.atan2(c, a)

    return dict(scale=scale, rotation=rotation, tx=float(tx), ty=float(ty))

affine = decompose_affine(M)
```

#### 3.1.42. camera_background_ratio

Доля пикселей, классифицированных как фон (по порогу величины движения, по умолчанию 0.5).

```py
background_ratio = _safe_float(np.mean(bg_mask))
```

#### 3.1.43. camera_rotation_speed

Скорость вращения камеры (изменение угла поворота между кадрами) в радианах на кадр.

```py
# flow_prev - предыдущий flow_tensor

rotation_speed = 0.0
if flow_prev is not None:
    try:
        Mprev = estimate_affine_from_flow(
            flow_prev,
            mask=background_mask_by_magnitude(flow_prev)
        )
        prev_affine = decompose_affine(Mprev)
        rotation_speed = _safe_float(affine["rotation"] - prev_affine["rotation"])
    except Exception:
        rotation_speed = 0.0
```

#### 3.2 spatial_analysis

#### 3.2.1. regional_stats

Общее:

```py
grid_size = (4, 4)

dx = flow_tensor[0].numpy().astype(np.float32)
dy = flow_tensor[1].numpy().astype(np.float32)
magnitude = np.sqrt(dx**2 + dy**2)

H, W = magnitude.shape
rows, cols = grid_size
region_h, region_w = H // rows, W // cols

regional_stats = []

for i in range(rows):
    for j in range(cols):
        y_start = i * region_h
        y_end = (i + 1) * region_h if i < rows - 1 else H
        x_start = j * region_w
        x_end = (j + 1) * region_w if j < cols - 1 else W
        region_mag = magnitude[y_start:y_end, x_start:x_end]
        region_dx = dx[y_start:y_end, x_start:x_end]
        region_dy = dy[y_start:y_end, x_start:x_end]
        region_stats = _calculate_region_stats(
            region_mag, region_dx, region_dy, i, j,
            (y_start, y_end, x_start, x_end), magnitude
        )
        regional_stats.append(region_stats)

spatial_df = pd.DataFrame(regional_stats)

regional_stats = spatial_df.to_dict('records')
```

#### 3.2.1.1. region_id

```py
region_id = f"R{i}_{j}",
```

#### 3.2.1.2. grid_position

```py
grid_position = f"{i},{j}",
```

#### 3.2.1.3. pixel_coords

```py
coords = (y_start, y_end, x_start, x_end)

pixel_coords = f"{coords[0]}:{coords[1]},{coords[2]}:{coords[3]}"
```

#### 3.2.1.4. region_size

```py
region_size = int(region_mag.size)
```

#### 3.2.1.5. - 3.2.1.8. region_magnitude_mean, region_magnitude_std, region_dx_mean, region_dy_mean

```py
region_magnitude_mean = np.mean(region_mag)
region_magnitude_std = float(np.std(region_mag))
region_dx_mean = float(np.mean(region_dx))
region_dy_mean = float(np.mean(region_dy))
```

#### 3.2.1.9. - 3.2.1.10. relative_activity, motion_dominance

```py
global_mean = np.mean(magnitude)

relative_activity = float(region_mean / (global_mean + 1e-10))
motion_dominance = float(np.sum(region_mag > global_mean) / region_mag.size)
```

#### 3.2.1.11. - 3.2.1.12. region_gradient, flow_divergence

```py
def _calculate_region_gradient(region_mag: np.ndarray) -> float:
    if region_mag.size < 4:
        return 0.0
    try:
        grad_y, grad_x = np.gradient(region_mag)
        return np.mean(np.sqrt(grad_x**2 + grad_y**2))
    except:
        return 0.0

def _calculate_region_divergence(dx: np.ndarray, dy: np.ndarray) -> float:
    if dx.size < 4:
        return 0.0
    try:
        div = np.gradient(dx, axis=1) + np.gradient(dy, axis=0)
        return np.mean(div)
    except:
        return 0.0

region_gradient = float(_calculate_region_gradient(region_mag))
flow_divergence = float(_calculate_region_divergence(region_dx, region_dy))
```

#### 3.2.1.13. - 3.2.1.14. activity_level, direction_category

```py
def _categorize_region_activity(region_mean: float, global_mean: float) -> str:
    ratio = region_mean / (global_mean + 1e-10)
    if ratio < 0.3:
        return 'very_low'
    elif ratio < 0.7:
        return 'low'
    elif ratio < 1.3:
        return 'average'
    elif ratio < 2.0:
        return 'high'
    else:
        return 'very_high'

def _categorize_region_direction(dx: np.ndarray, dy: np.ndarray) -> str:
    mean_dx = np.mean(dx)
    mean_dy = np.mean(dy)
    
    if abs(mean_dx) > abs(mean_dy) * 1.5:
        return 'horizontal_right' if mean_dx > 0 else 'horizontal_left'
    elif abs(mean_dy) > abs(mean_dx) * 1.5:
        return 'vertical_down' if mean_dy > 0 else 'vertical_up'
    else:
        if mean_dx > 0 and mean_dy > 0:
            return 'diagonal_down_right'
        elif mean_dx < 0 and mean_dy > 0:
            return 'diagonal_down_left'
        elif mean_dx > 0 and mean_dy < 0:
            return 'diagonal_up_right'
        else:
            return 'diagonal_up_left'

activity_level = _categorize_region_activity(region_mean, global_mean)
direction_category = _categorize_region_direction(region_dx, region_dy)
``` 

#### 3.2.2 roi_analysis

```py
top_regions_count = 3

top_regions = spatial_df.nlargest(top_regions_count, 'region_magnitude_mean')
```

#### 3.2.2.1 top_regions

```py
top_regions = top_regions['region_id'].tolist(),
```

#### 3.2.2.2 activity_concentration

```py
activity_concentration = float(
    top_regions['region_magnitude_mean'].sum() / 
    spatial_df['region_magnitude_mean'].sum()
),
```

#### 3.2.2.3 spatial_distribution

```py
def _analyze_spatial_distribution(top_regions: pd.DataFrame) -> str:
    if len(top_regions) < 2:
        return 'single_region'
    
    positions = []
    for pos in top_regions['grid_position']:
        if isinstance(pos, str) and ',' in pos:
            try:
                row, col = map(int, pos.split(','))
                positions.append([row, col])
            except:
                continue
    
    if len(positions) < 2:
        return 'single_region'
    
    positions_array = np.array(positions)
    variance = np.var(positions_array, axis=0).sum()
    
    if variance < 1.0:
        return 'concentrated'
    elif variance < 4.0:
        return 'scattered'
    else:
        return 'distributed'

spatial_distribution = _analyze_spatial_distribution(top_regions)
```

#### 3.2.2.4 dominant_directions

```py
dominant_directions = top_regions['direction_category'].value_counts().to_dict()
```

#### 3.3. temporal_analysis

Общее:

```py
# frame_stats_list - (3.1. frame_statistics)
# video_metadata - береться из алгоритма создания flow

fps = video_metadata.get('video_properties', {}).get('fps', 25.0)
skip = video_metadata.get('processing_parameters', {}).get('frame_skip', 5)

df = pd.DataFrame(frame_stats_list)

time_series = {
    'magnitude': df['magnitude_mean'].values,
    'moving_pixels': df['moving_pixels_0.5'].values,
    'direction_std': df['direction_std'].values,
    'spatial_gradient': df['spatial_gradient'].values
}

time_seconds = np.arange(len(df)) * skip / fps
```

#### 3.3.1. trends

#### 3.3.1.1. - 3.3.1.4. magnitude, moving_pixels, direction_std, spatial_gradient

#### 3.3.1.1-4.1. - 3.3.1.1-4.6. slope, trend_type, mean, std, range, has_trend

```py
trends = {}
savgol_window = 11

for metric_name, values in time_series.items():
    try:
        coeffs = np.polyfit(time_seconds, values, 1)
        slope = coeffs[0]
    except:
        slope = 0.0
    
    # Сглаживание
    if len(values) > savgol_window:
        window = min(savgol_window, len(values))
        if window % 2 == 0:
            window -= 1 
        try:
            smoothed = savgol_filter(values, window, 2)
        except:
            smoothed = values
    else:
        smoothed = values
    
    if abs(slope) < 0.001:
        trend_type = 'stable'
    elif slope > 0:
        trend_type = 'increasing'
    else:
        trend_type = 'decreasing'
    
    trends[metric_name] = {
        'slope': float(slope),
        'trend_type': trend_type,
        'mean': float(np.mean(values)),
        'std': float(np.std(values)),
        'range': float(np.max(values) - np.min(values)),
        'has_trend': abs(slope) > 0.001
    }
```

#### 3.3.2. periodicity

#### 3.3.2.1 - 3.3.2.4 magnitude, moving_pixels, direction_std, spatial_gradient

#### 3.3.1.1-4.1. - 3.3.1.1-4.2. has_periodicity, reason

```py
results = {}
        
for metric_name, values in time_series.items():
    if len(values) < 20:
        results[metric_name] = {'has_periodicity': False, 'reason': 'insufficient_data'}
        continue
    
    try:
        n = len(values)
        yf = np.fft.fft(values - np.mean(values))
        xf = np.fft.fftfreq(n, d=(skip/fps))
        
        power = np.abs(yf[:n//2])**2
        freqs = xf[:n//2]
        
        mask = (freqs > 0.1) & (freqs < 5.0)
        if np.any(mask) and len(power[mask]) > 0:
            dominant_idx = np.argmax(power[mask])
            dominant_freq = freqs[mask][dominant_idx]
            
            if dominant_freq > 0:
                dominant_period = 1.0 / dominant_freq
                significance = power[mask][dominant_idx] / np.mean(power[mask])
                
                if significance > 2.0:
                    results[metric_name] = {
                        'has_periodicity': True,
                        'dominant_frequency_hz': float(dominant_freq),
                        'dominant_period_seconds': float(dominant_period),
                        'significance': float(significance)
                    }
                    continue
    
        results[metric_name] = {'has_periodicity': False}
    except:
        results[metric_name] = {'has_periodicity': False, 'reason': 'analysis_error'}

return results
```

#### 3.3.3. transitions

#### 3.3.2.1 - 3.3.2.4 magnitude, moving_pixels, direction_std, spatial_gradient

```py
transitions = {}
        
for metric_name, values in time_series.items():
    if len(values) < window_size * 2:
        transitions[metric_name] = {'transition_count': 0, 'reason': 'insufficient_data'}
        continue
    
    try:
        series = pd.Series(values)
        rolling_mean = series.rolling(window=window_size, center=True, min_periods=1).mean()
        rolling_std = series.rolling(window=window_size, center=True, min_periods=1).std()
        z_scores = np.abs((values - rolling_mean) / (rolling_std.replace(0, 1e-10)))
        anomaly_mask = z_scores > 2.0
        
        if np.any(anomaly_mask):
            anomaly_indices = np.where(anomaly_mask)[0]
            transition_points = []
            current_group = []
            
            for idx in anomaly_indices:
                if not current_group or idx - current_group[-1] <= window_size:
                    current_group.append(idx)
                else:
                    if current_group:
                        center = int(np.mean(current_group))
                        transition_points.append({
                            'frame_index': int(center),
                            'z_score': float(z_scores[center]),
                            'value_change': float(values[center] - np.mean(values))
                        })
                    current_group = [idx]
            
            if current_group:
                center = int(np.mean(current_group))
                transition_points.append({
                    'frame_index': int(center),
                    'z_score': float(z_scores[center]),
                    'value_change': float(values[center] - np.mean(values))
                })
            
            transitions[metric_name] = {
                'transition_count': len(transition_points),
                'transition_points': transition_points,
                'max_z_score': float(np.max(z_scores))
            }
        else:
            transitions[metric_name] = {'transition_count': 0}
    except:
        transitions[metric_name] = {'transition_count': 0, 'reason': 'analysis_error'}
```

#### 3.3.4. segments

#### 3.3.4.1. - 3.3.4.3. segments, boundary_frames, method

```py
def _temporal_segmentation(magnitude_series: np.ndarray, n_segments: int = 5) -> Dict[str, Any]:
    if len(magnitude_series) < n_segments * 2:
        return {'segments': [], 'error': 'insufficient_data'}
    
    try:
        from sklearn.cluster import KMeans
        
        X = magnitude_series.reshape(-1, 1)
        kmeans = KMeans(n_clusters=n_segments, random_state=42, n_init=10)
        labels = kmeans.fit_predict(X)
        boundaries = []
        for i in range(1, len(labels)):
            if labels[i] != labels[i-1]:
                boundaries.append(i)
        
        segments = []
        start_idx = 0
        
        for boundary in boundaries:
            segment_data = magnitude_series[start_idx:boundary]
            segments.append({
                'start_frame': int(start_idx),
                'end_frame': int(boundary),
                'length_frames': len(segment_data),
                'mean_magnitude': float(np.mean(segment_data)),
                'std_magnitude': float(np.std(segment_data)),
                'cluster_label': int(labels[start_idx])
            })
            start_idx = boundary
        
        if start_idx < len(magnitude_series):
            segment_data = magnitude_series[start_idx:]
            segments.append({
                'start_frame': int(start_idx),
                'end_frame': len(magnitude_series),
                'length_frames': len(segment_data),
                'mean_magnitude': float(np.mean(segment_data)),
                'std_magnitude': float(np.std(segment_data)),
                'cluster_label': int(labels[start_idx])
            })
        
        return {
            'segments': segments,
            'boundary_frames': boundaries,
            'method': 'kmeans'
        }
    except Exception as e:
        return TemporalAnalyzer._uniform_segmentation(magnitude_series, n_segments)

@staticmethod
def _uniform_segmentation(magnitude_series: np.ndarray, n_segments: int = 5) -> Dict[str, Any]:
    segment_length = len(magnitude_series) // n_segments
    segments = []
    
    for i in range(n_segments):
        start_idx = i * segment_length
        end_idx = (i + 1) * segment_length if i < n_segments - 1 else len(magnitude_series)
        
        segment_data = magnitude_series[start_idx:end_idx]
        if len(segment_data) > 0:
            segments.append({
                'start_frame': int(start_idx),
                'end_frame': int(end_idx),
                'length_frames': len(segment_data),
                'mean_magnitude': float(np.mean(segment_data)),
                'std_magnitude': float(np.std(segment_data))
            })
    
    return {'segments': segments, 'method': 'uniform'}

segments = _temporal_segmentation(time_series['magnitude'])
```

#### 3.3.5. summary

#### 3.3.5.1. - 3.3.5.5. total_duration_seconds, avg_magnitude, magnitude_variability, activity_peaks_count, stability_score

```py
magnitude = time_series['magnitude']
        
try:
    peaks, properties = find_peaks(magnitude, 
                                    height=np.mean(magnitude) * 1.5,
                                    distance=5)
    peak_count = len(peaks)
except:
    peak_count = 0

stability_metrics = []
for values in time_series.values():
    cv = np.std(values) / (np.mean(values) + 1e-10)
    stability_metrics.append(1.0 / (1.0 + cv))

summary = {
    'total_duration_seconds': float(time_seconds[-1]) if len(time_seconds) > 0 else 0.0,
    'avg_magnitude': float(np.mean(magnitude)),
    'magnitude_variability': float(np.std(magnitude) / (np.mean(magnitude) + 1e-10)),
    'activity_peaks_count': peak_count,
    'stability_score': float(np.mean(stability_metrics)) if stability_metrics else 0.0
}
```

#### 3.4. summary_metrics

#### 3.4.1. - 3.4.10. overall_magnitude_mean, overall_magnitude_std, activity_variability, dominant_motion_intensity, dominant_direction, temporal_stability, peak_activity_frames, dominant_trend, has_periodicity, transition_count

```py
df = pd.DataFrame(frame_stats_list)

metrics = {
    'overall_magnitude_mean': float(df['magnitude_mean'].mean()),
    'overall_magnitude_std': float(df['magnitude_mean'].std()),
    'activity_variability': float(df['moving_pixels_0.5'].std()),
    'dominant_motion_intensity': df['motion_intensity'].mode()[0] if not df['motion_intensity'].mode().empty else 'unknown',
    'dominant_direction': df['dominant_direction'].mode()[0] if not df['dominant_direction'].mode().empty else 'unknown'
}

if 'summary' in temporal_results and not isinstance(temporal_results.get('error'), str):
    temp_summary = temporal_results['summary']
    metrics.update({
        'temporal_stability': temp_summary.get('stability_score', 0.0),
        'peak_activity_frames': temp_summary.get('activity_peaks_count', 0),
        'dominant_trend': temporal_results.get('trends', {}).get('magnitude', {}).get('trend_type', 'unknown'),
        'has_periodicity': any(
            r.get('has_periodicity', False) for r in temporal_results.get('periodicity', {}).values()
        ),
        'transition_count': sum(
            t.get('transition_count', 0) for t in temporal_results.get('transitions', {}).values()
        )
    })
```

#### 3.5. camera_motion

```py
camera_motion_results = None
if getattr(self.config, 'enable_camera_motion', False):
    camera_motion_results = self._analyze_camera_motion(flow_files)
```

#### 3.5.1. summary

#### 3.5.1.1. - 3.5.1.29. motion_mean_mean, motion_mean_std, motion_mean_max, motion_mean_min, motion_std_mean, motion_std_std, motion_std_max, motion_std_min, motion_energy_sum, motion_entropy_mean, shake_mean, shake_std, shake_max, zoom_in_count, zoom_out_count, zoom_speed_mean, rotation_speed_mean, rotation_speed_std, pan_ratio, truck_ratio, pedestal_ratio, static_ratio, chaos_index, style_handheld, style_tripod, style_cinematic, style_drone, style_action_cam, n_frames

```py
def compute_frame_motion_features(flow: np.ndarray, flow_prev: Optional[np.ndarray] = None, mag_bg_thresh: float = 0.5) -> Dict[str, float]:
    mag, ang = flow_magnitude_angle(flow)
    flat = mag.ravel()

    motion_mean = _safe_float(np.mean(flat)) if flat.size else 0.0
    motion_std = _safe_float(np.std(flat)) if flat.size else 0.0
    motion_max = _safe_float(np.max(flat)) if flat.size else 0.0
    motion_energy = _safe_float(np.sum(flat ** 2))

    try:
        hist, _ = np.histogram(ang.ravel(), bins=36, range=(-math.pi, math.pi))
        p = hist / (hist.sum() + 1e-9)
        motion_entropy = _safe_float(-np.sum([float(x) * math.log(float(x) + 1e-12) for x in p if x > 0]))
    except Exception:
        motion_entropy = 0.0

    try:
        bg_mask = background_mask_by_magnitude(flow, mag_bg_thresh)
        background_ratio = _safe_float(np.mean(bg_mask))
    except Exception:
        bg_mask = None
        background_ratio = 0.0

    try:
        shakiness = compute_shakiness(flow, background_mask=bg_mask)
        shakiness = _safe_dict(shakiness)
    except Exception:
        shakiness = dict(shake_var=0.0, shake_mean=0.0, shake_max=0.0)

    try:
        M = estimate_affine_from_flow(flow, mask=bg_mask)
        affine = _safe_dict(decompose_affine(M))
    except Exception:
        affine = dict(scale=1.0, rotation=0.0, tx=0.0, ty=0.0)

    rotation_speed = 0.0
    if flow_prev is not None:
        try:
            Mprev = estimate_affine_from_flow(
                flow_prev,
                mask=background_mask_by_magnitude(flow_prev, mag_bg_thresh)
            )
            prev_affine = _safe_dict(decompose_affine(Mprev))
            rotation_speed = _safe_float(affine["rotation"] - prev_affine["rotation"])
        except Exception:
            rotation_speed = 0.0

    return dict(
        motion_mean=motion_mean,
        motion_std=motion_std,
        motion_max=motion_max,
        motion_energy=motion_energy,
        motion_entropy=motion_entropy,

        shake_var=shakiness['shake_var'],
        shake_mean=shakiness['shake_mean'],
        shake_max=shakiness['shake_max'],

        affine_scale=affine['scale'],
        affine_rotation=affine['rotation'],
        affine_tx=affine['tx'],
        affine_ty=affine['ty'],

        background_ratio=background_ratio,
        rotation_speed=rotation_speed
    )

def background_mask_by_magnitude(flow: np.ndarray, mag_thresh: float = 0.5) -> np.ndarray:
    mag, _ = flow_magnitude_angle(flow)
    return mag <= mag_thresh

def estimate_affine_from_flow(flow: np.ndarray, mask = None, sample_n = 2000, ransac_thresh = 3.0):
    import cv2

    h, w, _ = flow.shape
    ys, xs = np.mgrid[0:h, 0:w]
    pts = np.stack([xs.ravel().astype(np.float32), ys.ravel().astype(np.float32)], axis=-1)
    disp = flow.reshape(-1, 2).astype(np.float32)
    if mask is not None:
        m = mask.ravel().astype(bool)
        if m.sum() < 10:
            return None
        pts = pts[m]
        disp = disp[m]
    if pts.shape[0] > sample_n:
        idx = np.random.choice(pts.shape[0], sample_n, replace=False)
        pts = pts[idx]
        disp = disp[idx]
    src = pts
    dst = pts + disp
    M, _ = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=ransac_thresh)
    return M 

def decompose_affine(M: np.ndarray):
    if M is None:
        return dict(scale=np.nan, rotation=np.nan, tx=np.nan, ty=np.nan)
    a, b, tx = M[0]
    c, d, ty = M[1]
    scale = math.sqrt(a * a + b * b)
    rotation = math.atan2(c, a)
    return dict(scale=scale, rotation=rotation, tx=float(tx), ty=float(ty))

def detect_zoom_from_affines(prev_affine, cur_affine, eps = 1e-6):
    if prev_affine is None or cur_affine is None:
        return 0.0
    prev = decompose_affine(prev_affine)
    cur = decompose_affine(cur_affine)
    if math.isnan(prev['scale']) or math.isnan(cur['scale']):
        return 0.0
    return float(cur['scale'] - prev['scale'])

mag_bg_thresh = config.get("mag_bg_thresh", 0.5)
zoom_eps = config.get("zoom_eps", 1e-3)
sharp_angle_thresh = config.get("sharp_angle_thresh_deg", 15.0)

per_frame: List[Dict[str, float]] = []
affines: List[np.ndarray] = []
flows: List[np.ndarray] = []

for p in sorted(flow_paths):
    try:
        f = load_flow_tensor(p)
    except Exception as e:
        print(f"[camera_motion] failed to load {p}: {e}")
        continue
    flows.append(f)

n = len(flows)
if n == 0:
    return {}

prev_flow = None
prev_affine = None
zoom_ins = 0
zoom_outs = 0
zoom_deltas = []
rotation_speeds = []
shake_vars = []
pan_cnt = tilt_cnt = roll_cnt = dolly_cnt = truck_cnt = pedestal_cnt = static_cnt = 0

for f in flows:
    feats = compute_frame_motion_features(f, flow_prev=prev_flow, mag_bg_thresh=mag_bg_thresh)
    per_frame.append(feats)
    shake_vars.append(feats['shake_var'])
    rotation_speeds.append(feats.get('rotation_speed', 0.0))
    bg_mask = background_mask_by_magnitude(f, mag_bg_thresh)
    M = estimate_affine_from_flow(f, mask=bg_mask)
    affines.append(M)
    if prev_affine is not None:
        dz = detect_zoom_from_affines(prev_affine, M)
        zoom_deltas.append(dz)
        if dz > zoom_eps:
            zoom_ins += 1
        elif dz < -zoom_eps:
            zoom_outs += 1
    prev_affine = M
    dec = decompose_affine(M)
    rot = abs(dec['rotation']) if not math.isnan(dec['rotation']) else 0.0
    tnorm = math.hypot(dec['tx'], dec['ty'])
    if rot > math.radians(0.05):
        pan_cnt += 1
    elif tnorm > 0.5:
        if abs(dec['tx']) > abs(dec['ty']):
            truck_cnt += 1
        else:
            pedestal_cnt += 1
    else:
        static_cnt += 1
    prev_flow = f

motion_means = [x['motion_mean'] for x in per_frame]
motion_stds = [x['motion_std'] for x in per_frame]
rotation_speeds_arr = np.array(rotation_speeds)
zoom_deltas_arr = np.array(zoom_deltas) if zoom_deltas else np.array([0.0])

def safe_stats(arr):
    arr = np.array(arr)
    return dict(mean=float(np.nanmean(arr)), std=float(np.nanstd(arr)), max=float(np.nanmax(arr)), min=float(np.nanmin(arr)))

res: Dict[str, float] = {}
res.update({f"motion_mean_{k}": v for k, v in safe_stats(motion_means).items()})
res.update({f"motion_std_{k}": v for k, v in safe_stats(motion_stds).items()})
res["motion_energy_sum"] = float(sum([x["motion_energy"] for x in per_frame]))
res["motion_entropy_mean"] = float(np.nanmean([x["motion_entropy"] for x in per_frame]))
res["shake_mean"] = float(np.nanmean(shake_vars))
res["shake_std"] = float(np.nanstd(shake_vars))
res["shake_max"] = float(np.nanmax(shake_vars))
res["zoom_in_count"] = int(zoom_ins)
res["zoom_out_count"] = int(zoom_outs)
res["zoom_speed_mean"] = float(np.mean(np.abs(zoom_deltas_arr))) if zoom_deltas_arr.size else 0.0
res["rotation_speed_mean"] = float(np.nanmean(rotation_speeds_arr))
res["rotation_speed_std"] = float(np.nanstd(rotation_speeds_arr))
total_moves = pan_cnt + tilt_cnt + roll_cnt + dolly_cnt + truck_cnt + pedestal_cnt + static_cnt
total_moves = total_moves or 1
res["pan_ratio"] = float(pan_cnt / total_moves)
res["truck_ratio"] = float(truck_cnt / total_moves)
res["pedestal_ratio"] = float(pedestal_cnt / total_moves)
res["static_ratio"] = float(static_cnt / total_moves)
all_dirs = np.hstack([np.histogram(np.arctan2(f[..., 1], f[..., 0]).ravel(), bins=36, range=(-math.pi, math.pi))[0] for f in flows])
p = all_dirs / (all_dirs.sum() + 1e-9)
chaos = -np.sum([x * math.log(x + 1e-12) for x in p if x > 0])
res["chaos_index"] = float(chaos)
res["style_handheld"] = float(min(1.0, res["shake_mean"] * 2.0))
res["style_tripod"] = float(max(0.0, 1.0 - res["shake_mean"] * 2.0))
res["style_cinematic"] = float(max(0.0, 1.0 - res["shake_mean"]))
res["style_drone"] = float(min(1.0, res["chaos_index"] / 10.0))
res["style_action_cam"] = float(min(1.0, (res["shake_mean"] + res["motion_energy_sum"] / 1e4)))
res["n_frames"] = int(len(per_frame))
```

#### 3.5.2. per_frame

#### 3.5.2.1. - 3.5.2.14. motion_mean, motion_std, motion_max, motion_energy, motion_entropy, shake_var, shake_mean, shake_max, affine_scale, affine_rotation, affine_tx, affine_ty, background_ratio, rotation_speed

```py
per_frame = []
prev = None
for flow in flows:
    feats = compute_frame_motion_features(flow, flow_prev=prev, mag_bg_thresh=mag_bg_thresh)
    per_frame.append(feats)
    prev = flow
```

#### 3.6. advanced_features

```py
flows = []
magnitudes = []
for flow_file in flow_files:
    try:
        flow = torch.load(flow_file, map_location='cpu')
        if isinstance(flow, torch.Tensor):
            flow = flow.numpy()
        if flow.shape[0] == 2:
            flow = np.transpose(flow, (1, 2, 0))
        flows.append(flow)
        mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        magnitudes.append(mag)
    except Exception as e:
        logger.warning(f"Ошибка загрузки {flow_file}: {e}")
        continue
```

#### 3.6.1. motion_energy_image

#### 3.6.1.1. features

```py
if getattr(self.config, 'enable_mei', True):
    try:
        mei, mei_features = compute_mei(magnitudes)
        results['motion_energy_image'] = {
            'features': mei_features,
            'mei_shape': list(mei.shape)
        }
    except Exception as e:
        logger.warning(f"Ошибка MEI: {e}")
        results['motion_energy_image'] = {'error': str(e)}
```

#### 3.6.1.1.1. - 3.6.1.1.9. mei_total_energy, mei_coverage_ratio, mei_max_energy, mei_std, mhi_contrast, mhi_entropy, mhi_mean, mhi_max, motion_persistence

```py
def compute_mei(flow_magnitudes: List[np.ndarray], decay_factor: float = 0.9) -> Tuple[np.ndarray, Dict[str, float]]:
    if not flow_magnitudes:
        return np.zeros((1, 1)), {}
    
    h, w = flow_magnitudes[0].shape
    for mag in flow_magnitudes:
        if mag.shape != (h, w):
            mag = cv2.resize(mag, (w, h), interpolation=cv2.INTER_LINEAR)
    
    threshold = np.percentile(np.concatenate([m.flatten() for m in flow_magnitudes]), 50)
    
    mei = np.zeros((h, w), dtype=np.float32)
    mhi = np.zeros((h, w), dtype=np.float32) 
    
    for i, mag in enumerate(flow_magnitudes):
        motion_mask = mag > threshold
        mei = np.maximum(mei, motion_mask.astype(np.float32))
        
        mhi = np.maximum(mhi * decay_factor, motion_mask.astype(np.float32))
    
    features = {
        'mei_total_energy': float(np.sum(mei)),
        'mei_coverage_ratio': float(np.mean(mei)),
        'mei_max_energy': float(np.max(mei)),
        'mei_std': float(np.std(mei)),
        'mhi_contrast': float((np.max(mhi) - np.min(mhi)) / (np.max(mhi) + 1e-10)),
        'mhi_entropy': float(MotionEnergyImage._compute_entropy(mhi)),
        'mhi_mean': float(np.mean(mhi)),
        'mhi_max': float(np.max(mhi)),
        'motion_persistence': float(np.mean(mhi > 0.5))
    }
    
    return mei, features
```

#### 3.6.1.2. mei_shape

```py
mei_shape = list(mei.shape)
```

#### 3.6.2. foreground_background_motion

```py
if getattr(self.config, 'enable_fg_bg', True):
    try:
        fg_bg_results = []
        for flow in flows[:min(50, len(flows))]: 
            fg_bg = separate_motion(
                flow,
                method=getattr(self.config, 'fg_bg_method', 'magnitude_threshold'),
                threshold=getattr(self.config, 'fg_bg_threshold', 0.5)
            )
            fg_bg_results.append(fg_bg['features'])
        
        # Агрегируем статистики
        if fg_bg_results:
            avg_fg_bg = {
                'foreground_motion_energy_mean': float(np.mean([r['foreground_motion_energy'] for r in fg_bg_results])),
                'background_motion_energy_mean': float(np.mean([r['background_motion_energy'] for r in fg_bg_results])),
                'ratio_foreground_background_mean': float(np.mean([r['ratio_foreground_background_flow'] for r in fg_bg_results])),
                'foreground_coverage_mean': float(np.mean([r['foreground_coverage_ratio'] for r in fg_bg_results]))
            }
            results['foreground_background_motion'] = {
                'summary': avg_fg_bg,
                'per_frame_count': len(fg_bg_results)
            }
    except Exception as e:
        logger.warning(f"Ошибка FG/BG: {e}")
        results['foreground_background_motion'] = {'error': str(e)}
```

#### 3.6.2.1. summary

#### 3.6.2.1.1. - 3.6.2.1.4. foreground_motion_energy_mean, background_motion_energy_mean, ratio_foreground_background_mean, foreground_coverage_mean

```py
def separate_motion(flow: np.ndarray, method: str = 'magnitude_threshold', threshold: float = 0.5, segmentation_mask: Optional[np.ndarray] = None) -> Dict[str, Any]:
    if flow.shape[0] == 2:
        flow = np.transpose(flow, (1, 2, 0))
    
    dx = flow[..., 0]
    dy = flow[..., 1]
    magnitude = np.sqrt(dx**2 + dy**2)
    
    if method == 'segmentation' and segmentation_mask is not None:
        fg_mask = segmentation_mask.astype(bool)
        bg_mask = ~fg_mask
    elif method == 'spatial_clustering':
        fg_mask, bg_mask = _spatial_clustering(magnitude)
    else:
        fg_mask = magnitude > threshold
        bg_mask = magnitude <= threshold
    
    fg_magnitude = magnitude[fg_mask] if np.any(fg_mask) else np.array([])
    bg_magnitude = magnitude[bg_mask] if np.any(bg_mask) else np.array([])
    
    features = {
        'foreground_motion_energy': float(np.sum(fg_magnitude**2)) if len(fg_magnitude) > 0 else 0.0,
        'background_motion_energy': float(np.sum(bg_magnitude**2)) if len(bg_magnitude) > 0 else 0.0,
        'foreground_motion_mean': float(np.mean(fg_magnitude)) if len(fg_magnitude) > 0 else 0.0,
        'background_motion_mean': float(np.mean(bg_magnitude)) if len(bg_magnitude) > 0 else 0.0,
        'foreground_motion_std': float(np.std(fg_magnitude)) if len(fg_magnitude) > 0 else 0.0,
        'background_motion_std': float(np.std(bg_magnitude)) if len(bg_magnitude) > 0 else 0.0,
        'foreground_coverage_ratio': float(np.mean(fg_mask)),
        'background_coverage_ratio': float(np.mean(bg_mask)),
        'ratio_foreground_background_flow': float(
            (np.sum(fg_magnitude**2) / (np.sum(bg_magnitude**2) + 1e-10)) 
            if len(fg_magnitude) > 0 and len(bg_magnitude) > 0 else 0.0
        ),
        'foreground_max': float(np.max(fg_magnitude)) if len(fg_magnitude) > 0 else 0.0,
        'background_max': float(np.max(bg_magnitude)) if len(bg_magnitude) > 0 else 0.0
    }
    
    return {
        'foreground_mask': fg_mask,
        'background_mask': bg_mask,
        'foreground_flow': flow[fg_mask] if np.any(fg_mask) else np.array([]),
        'background_flow': flow[bg_mask] if np.any(bg_mask) else np.array([]),
        'features': features
    }

def _spatial_clustering(magnitude: np.ndarray, n_clusters: int = 2) -> Tuple[np.ndarray, np.ndarray]:
    h, w = magnitude.shape
    y_coords, x_coords = np.mgrid[0:h, 0:w]
    features = np.stack([
        magnitude.flatten(),
        x_coords.flatten() / w, 
        y_coords.flatten() / h
    ], axis=1)
    
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(features)
    
    cluster_means = [magnitude.flatten()[labels == i].mean() for i in range(n_clusters)]
    fg_cluster = np.argmax(cluster_means)
    
    fg_mask = (labels == fg_cluster).reshape(h, w)
    bg_mask = ~fg_mask
    
    return fg_mask, bg_mask
```

#### 3.6.2.2. per_frame_count

```py
per_frame_count = len(fg_bg_results)
```

#### 3.6.3. motion_clusters

```py
if getattr(self.config, 'enable_clusters', True):
    try:
        cluster_results = []
        n_clusters = getattr(self.config, 'motion_clusters_n', 5)
        for flow in flows[:min(20, len(flows))]:  # Ограничиваем для скорости
            clusters = cluster_motion(
                flow,
                n_clusters=n_clusters
            )
            cluster_results.append(clusters['features'])
        
        if cluster_results:
            avg_clusters = {
                'num_clusters_mean': float(np.mean([r.get('num_motion_clusters', 0) for r in cluster_results])),
                'largest_cluster_coverage_mean': float(np.mean([r.get('largest_cluster_coverage', 0) for r in cluster_results])),
                'cluster_diversity_mean': float(np.mean([r.get('cluster_diversity', 0) for r in cluster_results]))
            }
            results['motion_clusters'] = {
                'summary': avg_clusters,
                'per_frame_count': len(cluster_results)
            }
    except Exception as e:
        logger.warning(f"Ошибка кластеров: {e}")
        results['motion_clusters'] = {'error': str(e)}
```

#### 3.6.3.1. summary

#### 3.6.3.1.1 - 3.6.3.1.3. num_clusters_mean, largest_cluster_coverage_mean, cluster_diversity_mean

```py
def cluster_motion(flow: np.ndarray, n_clusters: int = 5, method: str = 'direction_speed', sample_ratio: float = 0.1) -> Dict[str, Any]:
    if flow.shape[0] == 2:
        flow = np.transpose(flow, (1, 2, 0))
    
    dx = flow[..., 0]
    dy = flow[..., 1]
    magnitude = np.sqrt(dx**2 + dy**2)
    direction = np.arctan2(dy, dx)
    
    h, w = magnitude.shape
    
    total_pixels = h * w
    n_sample = max(1000, int(total_pixels * sample_ratio))
    
    if method == 'direction_speed':
        features = np.stack([
            magnitude.flatten(),
            np.cos(direction.flatten()),
            np.sin(direction.flatten())
        ], axis=1)
    else:
        features = np.stack([
            dx.flatten(),
            dy.flatten(),
            magnitude.flatten()
        ], axis=1)
    
    if n_sample < total_pixels:
        indices = np.random.choice(total_pixels, n_sample, replace=False)
        features_sample = features[indices]
    else:
        features_sample = features
        indices = np.arange(total_pixels)
    
    features_norm = (features_sample - features_sample.mean(axis=0)) / (features_sample.std(axis=0) + 1e-10)
    
    try:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        cluster_labels = kmeans.fit_predict(features_norm)
        
        if n_sample < total_pixels:
            all_features_norm = (features - features_sample.mean(axis=0)) / (features_sample.std(axis=0) + 1e-10)
            all_labels = kmeans.predict(all_features_norm)
        else:
            all_labels = cluster_labels
        
        cluster_mask = all_labels.reshape(h, w)
        
        cluster_stats = []
        for i in range(n_clusters):
            cluster_mask_i = cluster_mask == i
            cluster_mag = magnitude[cluster_mask_i]
            
            if len(cluster_mag) > 0:
                cluster_dir = direction[cluster_mask_i]
                cluster_stats.append({
                    'cluster_id': i,
                    'size': int(np.sum(cluster_mask_i)),
                    'coverage_ratio': float(np.mean(cluster_mask_i)),
                    'mean_magnitude': float(np.mean(cluster_mag)),
                    'std_magnitude': float(np.std(cluster_mag)),
                    'mean_direction': float(np.arctan2(
                        np.mean(np.sin(cluster_dir)),
                        np.mean(np.cos(cluster_dir))
                    )),
                    'max_magnitude': float(np.max(cluster_mag))
                })
        
        cluster_stats.sort(key=lambda x: x['size'], reverse=True)
        
        features = {
            'num_motion_clusters': len(cluster_stats),
            'largest_cluster_size': cluster_stats[0]['size'] if cluster_stats else 0,
            'largest_cluster_coverage': cluster_stats[0]['coverage_ratio'] if cluster_stats else 0.0,
            'cluster_size_distribution': [s['size'] for s in cluster_stats],
            'cluster_coverage_distribution': [s['coverage_ratio'] for s in cluster_stats],
            'dominant_cluster_magnitude': cluster_stats[0]['mean_magnitude'] if cluster_stats else 0.0,
            'cluster_diversity': float(len(cluster_stats) / n_clusters)
        }
        
        return {
            'cluster_mask': cluster_mask,
            'cluster_stats': cluster_stats,
            'features': features
        }
    except Exception as e:
        logger.warning(f"Ошибка кластеризации движения: {e}")
        return {
            'cluster_mask': np.zeros((h, w), dtype=int),
            'cluster_stats': [],
            'features': {
                'num_motion_clusters': 0,
                'error': str(e)
            }
        }
```

#### 3.6.3.2. per_frame_count

```py
per_frame_count = len(cluster_results)
```

#### 3.6.4. smoothness_jerkiness

```py
if getattr(self.config, 'enable_smoothness', True):
    try:
        fps = video_metadata.get('video_properties', {}).get('fps', 25.0)
        skip = video_metadata.get('processing_parameters', {}).get('frame_skip', 5)
        smoothness = compute_smoothness_metrics(
            flows,
            fps=fps,
            frame_skip=skip
        )
        results['smoothness_jerkiness'] = smoothness
    except Exception as e:
        logger.warning(f"Ошибка smoothness: {e}")
        results['smoothness_jerkiness'] = {'error': str(e)}
```

#### 3.6.4.1. - 3.6.4.9. smoothness_index, jerkiness_index, flow_temporal_entropy, movement_stability, mean_acceleration, std_acceleration, max_acceleration, mean_jerk, max_jerk

```py
def compute_smoothness_metrics(flow_sequence: List[np.ndarray], fps: float = 25.0, frame_skip: int = 1) -> Dict[str, Any]:
    if len(flow_sequence) < 3:
        return {'error': 'insufficient_frames'}
    
    magnitudes = []
    accelerations = []
    jerks = [] 
    
    for i in range(len(flow_sequence)):
        flow = flow_sequence[i]
        if flow.shape[0] == 2:
            flow = np.transpose(flow, (1, 2, 0))
        
        mag = np.sqrt(flow[..., 0]**2 + flow[..., 1]**2)
        magnitudes.append(np.mean(mag))
        
        if i > 0:
            prev_mag = magnitudes[i-1]
            accel = abs(magnitudes[i] - prev_mag)
            accelerations.append(accel)
            
            if i > 1:
                prev_accel = accelerations[i-2] if len(accelerations) > 1 else 0
                jerk = abs(accel - prev_accel)
                jerks.append(jerk)
    
    if len(accelerations) > 0:
        accel_array = np.array(accelerations)
        smoothness_index = 1.0 / (1.0 + np.mean(accel_array))
        jerkiness_index = np.mean(jerks) if len(jerks) > 0 else 0.0
    else:
        smoothness_index = 1.0
        jerkiness_index = 0.0
    
    if len(magnitudes) > 1:
        mag_array = np.array(magnitudes)
        changes = np.diff(mag_array)
        hist, _ = np.histogram(changes, bins=50)
        hist = hist / (hist.sum() + 1e-10)
        hist = hist[hist > 0]
        flow_temporal_entropy = float(-np.sum(hist * np.log2(hist + 1e-10)))
    else:
        flow_temporal_entropy = 0.0
    
    if len(magnitudes) > 1:
        mag_array = np.array(magnitudes)
        cv = np.std(mag_array) / (np.mean(mag_array) + 1e-10)
        movement_stability = 1.0 / (1.0 + cv)
    else:
        movement_stability = 1.0
    
    return {
        'smoothness_index': float(smoothness_index),
        'jerkiness_index': float(jerkiness_index),
        'flow_temporal_entropy': float(flow_temporal_entropy),
        'movement_stability': float(movement_stability),
        'mean_acceleration': float(np.mean(accelerations)) if len(accelerations) > 0 else 0.0,
        'std_acceleration': float(np.std(accelerations)) if len(accelerations) > 0 else 0.0,
        'max_acceleration': float(np.max(accelerations)) if len(accelerations) > 0 else 0.0,
        'mean_jerk': float(np.mean(jerks)) if len(jerks) > 0 else 0.0,
        'max_jerk': float(np.max(jerks)) if len(jerks) > 0 else 0.0
    }
```

## action_recognition

### Модели:

```py
from torchvision.models.video import slowfast_r50

model = slowfast_r50(pretrained=True)
```

### Выход:

```json
{
    "total_frames": 164,
    "num_tracks": 5,
    "processing_params": {
        "clip_len": 32,
        "batch_size": 4
    },
    "results": {
        "1": {
            "sequence_features": {
                "embedding_normed_256d": [[...], [...], ...],
                "temporal_diff_normalized": [0.1, 0.3, 0.2, ...]
            },
            "mean_embedding_norm_raw": 1.2,
            "std_embedding_norm_raw": 0.3,
            "temporal_variance": 0.8,
            "max_temporal_jump": 1.5,
            "stability": 0.85,
            "switch_rate_per_sec": 0.5,
            "early_late_embedding_shift": 0.3,
            "motion_entropy": 1.2,
            "num_unique_actions": 3,
            "dominant_action_ratio": 0.6,
            "embedding_entropy": 2.1,
            "is_multi_person": true,
            "num_persons": 5,
            "action_synchronization": 0.75
        },
        "2": {...}
    }
}
```

### Фичи:

Модуль использует **SlowFast** (Meta AI) — dual-pathway CNN архитектуру для распознавания действий. SlowFast имеет встроенную обработку motion через fast pathway, что делает внешнее вычисление motion избыточным.

Модуль извлекает два типа фичей:
1. **Sequence Features** — для VisualTransformer (L2-нормализованные embeddings)
2. **Aggregate Features** — для MLP/Tabular Head (агрегированные статистики)

#### 1. total_frames

Общее количество кадров видео, переданных в модуль для обработки.

#### 2. num_tracks

Количество обработанных треков (людей) в видео.

```py
num_tracks = len(results)
```

#### 3. processing_params

Параметры обработки видео.

**clip_len**: Длина клипа в кадрах (по умолчанию 32 для SlowFast).

**batch_size**: Размер батча для inference (по умолчанию 4).

#### 4. results

```py
# В production-трафике треки приходят из отдельного модуля трекинга
# (например, YOLO + ByteTrack), который возвращает:
# frame_indices_per_person: Dict[track_id, List[frame_idx]]

frame_indices_per_person = {
    track_id: [... кадры человека ...],
    # ...
}
```

```py
all_clips = []
meta = []

for track_id, indices in frame_indices_per_person.items():
    if len(indices) == 0:
        continue
    frames = self._load_frames(indices)  # Без motion (SlowFast обрабатывает motion)
    clips = self._make_clips(frames)

    all_clips.extend(clips)
    meta.extend([track_id]*len(clips))

if not all_clips:
    return {}

# Извлечение embeddings (raw и normed)
raw_embeddings_all, normed_embeddings_all = self._extract_embeddings(all_clips)

per_track_raw = defaultdict(list)
per_track_normed = defaultdict(list)
for tid, raw_emb, normed_emb in zip(meta, raw_embeddings_all, normed_embeddings_all):
    per_track_raw[tid].append(raw_emb)
    per_track_normed[tid].append(normed_emb)
```

#### 4.1. sequence_features

Временные последовательности фичей для VisualTransformer.

##### 4.1.1. embedding_normed_256d

Массив L2-нормализованных embeddings для каждого клипа. Формат: `[num_clips, 256]`.

```py
# Извлечение features из SlowFast (2048d)
raw_embeddings = slowfast_model.forward_features(clips)  # [num_clips, 2048]
# Проекция в 256d
embeddings_raw_256d = linear_projection(raw_embeddings)  # [num_clips, 256]
# L2 нормализация для VisualTransformer
norms = ||embeddings_raw_256d||_2
embedding_normed_256d = embeddings_raw_256d / norms  # [num_clips, 256], ||e|| = 1
```

**Важно**: Используются нормализованные embeddings (||e|| = 1), чтобы избежать влияния масштаба.

##### 4.1.2. temporal_diff_normalized

Массив нормализованных временных различий между соседними клипами, основанный на косинусной дистанции. Формат: `[num_clips]`.

```py
temporal_diff_normalized[0] = 0.0
for i in range(1, num_clips):
    # Косинусная дистанция: 1.0 - cosine_similarity(e_t, e_{t-1})
    temporal_diff_normalized[i] = 1.0 - cosine_similarity(
        embedding_normed[i],
        embedding_normed[i - 1],
    )
```

**Диапазон**: [0.0, 2.0]

**Преимущества**:
- устойчива к масштабу,
- интерпретируема (0 = одинаково, 1 ≈ ортогонально, 2 = противоположно),
- лучше работает в attention-модулях.

#### 4.2. CORE DYNAMICS

##### 4.2.1. mean_embedding_norm_raw

Средняя норма **raw** embeddings (до L2 нормализации) по всем клипам. Показывает общую "энергию" действий.

```py
# Используются raw embeddings (до нормализации)
embedding_norms_raw = [||e_raw||_2 for e_raw in raw_embeddings]
mean_embedding_norm_raw = mean(embedding_norms_raw)
```

**Важно**: Используются raw embeddings, а не нормализованные, иначе норма всегда ≈ 1.

##### 4.2.2. std_embedding_norm_raw

Стандартное отклонение норм **raw** embeddings. Показывает вариативность "энергии" действий.

```py
embedding_norms_raw = [||e_raw||_2 for e_raw in raw_embeddings]
std_embedding_norm_raw = std(embedding_norms_raw)
```

##### 4.2.3. temporal_variance

Временная вариация embeddings. Показывает, насколько embeddings отклоняются от среднего значения. Использует нормализованные embeddings.

```py
mean_embedding_normed = mean(normed_embeddings, axis=0)
temporal_variance = mean([||e - mean_embedding_normed||_2 for e in normed_embeddings])
```

##### 4.2.4. max_temporal_jump

Максимальный скачок между соседними клипами. Показывает максимальное изменение действий (hook moment).

```py
temporal_jumps = [||normed_embeddings[i] - normed_embeddings[i-1]||_2 
                 for i in range(1, len(normed_embeddings))]
max_temporal_jump = max(temporal_jumps)
```

#### 4.3. TEMPORAL STRUCTURE

##### 4.3.1. stability

Стабильность действий (0.0-1.0), вычисляемая через кластеризацию embeddings (k-means) с фиксацией пространства через PCA.

```py
if num_clips < 3:
    stability = 1.0
else:
    # Фиксация пространства через PCA перед кластеризацией
    n_pca_components = min(32, embedding_dim, num_clips - 1)
    pca = PCA(n_components=n_pca_components)
    embeddings_for_cluster = pca.fit_transform(normed_embeddings)

    # Выбор k: min(5, num_clips // 2)
    k = min(5, max(1, num_clips // 2))
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(embeddings_for_cluster)

    stability = longest_run_fraction(labels)
```

##### 4.3.2. switch_rate_per_sec

Частота смены действий в секунду, вычисленная через кластеризацию embeddings (используются те же labels, что и для stability).

```py
if num_clips < 3:
    switch_rate_per_sec = 0.0
else:
    transitions = sum(labels[1:] != labels[:-1])
    total_time_sec = num_clips * clip_len / fps
    switch_rate_per_sec = transitions / total_time_sec
```

##### 4.3.3. early_late_embedding_shift

Сдвиг embeddings между первой и второй половиной последовательности. Вычисляется как `1 - cosine_similarity` для лучшей интерпретации в MLP.

```py
mid = len(normed_embeddings) // 2
early_embedding = mean(normed_embeddings[:mid], axis=0)
late_embedding = mean(normed_embeddings[mid:], axis=0)
cosine_sim = cosine_similarity(early_embedding, late_embedding)
early_late_embedding_shift = 1.0 - cosine_sim  # 0 = одинаково, 1 = сильно изменилось
```

#### 4.4. MOTION-AWARE FEATURES (упрощенные)

**Важно**: SlowFast уже имеет встроенную обработку motion через fast pathway. Поэтому используется только легковесный сигнал на основе temporal differences.

##### 4.4.1. motion_entropy

Энтропия распределения временных различий между клипами, нормализованная в [0, 1]. Показывает структурированность движения.

```py
temporal_diffs = [||normed_embeddings[i] - normed_embeddings[i-1]||_2 
                 for i in range(1, len(normed_embeddings))]
temporal_diffs_norm = temporal_diffs / (max(temporal_diffs) + 1e-6)
temporal_diffs_probs = temporal_diffs_norm / (sum(temporal_diffs_norm) + 1e-6)
motion_entropy_raw = -sum(temporal_diffs_probs * log(temporal_diffs_probs + 1e-12))

T = len(temporal_diffs)
motion_entropy = motion_entropy_raw / (log(T + 1e-6) + 1e-12)
```

**Диапазон**: [0.0, 1.0]

**Убрано**: Optical Flow и motion_weighted_embedding_energy (избыточно, SlowFast обрабатывает motion внутренне).

#### 4.5. DIVERSITY METRICS

##### 4.5.1. num_unique_actions

Количество уникальных кластеров действий (через k-means кластеризацию embeddings).

```py
labels = kmeans.fit_predict(embeddings_for_cluster)  # Те же labels, что и для stability
num_unique_actions = len(unique(labels))
```

##### 4.5.2. dominant_action_ratio

Доля клипов в доминирующем кластере. Показывает стабильность действия.

```py
unique, counts = unique(labels, return_counts=True)
dominant_action_ratio = max(counts) / len(labels)
```

##### 4.5.3. embedding_entropy

Энтропия eigenvalues ковариационной матрицы embeddings. Показывает глобальное разнообразие.

```py
cov_matrix = cov(normed_embeddings.T)

# Численная стабилизация: небольшой diagonal jitter
cov_matrix += 1e-5 * I

eigenvalues = eigh(cov_matrix).eigenvalues
eigenvalues = abs(eigenvalues[eigenvalues > 1e-6])
eigenvalues_norm = eigenvalues / (sum(eigenvalues) + 1e-6)
embedding_entropy = -sum(eigenvalues_norm * log(eigenvalues_norm + 1e-12))
```

#### 4.6. MULTI-PERSON CONTEXT

Доступны только если обрабатывается несколько треков (≥2).

##### 4.6.1. is_multi_person

Булево значение, указывающее на наличие нескольких людей в сцене.

- `False`, если треков < 2 **или трекер не используется**.
- `True`, только если есть ≥2 реальных трека из модуля трекинга.

##### 4.6.2. num_persons

Количество людей (треков) в сцене, ограничено максимумом 5.

```py
num_persons = min(num_tracks, 5)
```

##### 4.6.3. action_synchronization

Синхронизация действий между людьми (0.0-1.0), где 1.0 — полная синхронизация. Вычисляется через cosine similarity между mean embeddings треков.

```py
track_embeddings = [mean(track['sequence_features']['embedding_normed_256d'], axis=0) 
                   for track in tracks]
similarities = [cosine_similarity(track_embeddings[i], track_embeddings[j]) 
               for i in range(len(tracks)) for j in range(i+1, len(tracks))]
action_synchronization = mean(similarities)
```

**Важно**: если треки не приходят из внешнего трекера (YOLO + ByteTrack и т.п.), модуль работает в single-person режиме и multi-person фичи фактически отключены.

## color_light

Модуль для комплексного анализа цвета и освещения видео. Выдаёт:
- компактную последовательность покадровых фич (frame-level) для VisualTransformer;
- сценовые агрегаты (scene-level);
- агрегаты по всему видео (video-level).

### Параметры обработки:

- **`max_frames_per_scene`**: максимальное количество кадров для обработки на сцену (по умолчанию 350).  
- **`stride`**: шаг выборки кадров (по умолчанию 5). При `stride > 1` используются кадры через шаг, иначе — равномерный сэмплинг до `max_frames_per_scene`.

### Выход:

```json
{
  "frames": {
    "shoe_shop_45": {
      "45": {
        "frame_idx": 45,
        "features": {
          "hue_mean_norm": 0.47,
          "hue_std_norm": 0.18,
          "hue_entropy": 2.82,
          "hue_entropy_weighted": 2.75,
          "sat_mean_norm": 0.19,
          "val_mean_norm": 0.55,
          "L_mean_norm": 0.54,
          "colorfulness_norm": 0.45,
          "global_contrast_norm": 0.29,
          "local_contrast_mean_norm": 0.07,
          "skin_tone_ratio": 0.0012,
          "overexposed_ratio": 0.0023,
          "underexposed_ratio": 0.0136,
          "vignetting_score_norm": 0.0,
          "soft_light_prob": 0.95,
          "dominant_lab_a_norm": 0.61,
          "dominant_lab_b_norm": 0.52,
          "...": "дополнительные frame-level фичи (brightness_mean, colorfulness_index и т.д.)"
        }
      },
      "54": { "...": "аналогично" }
    },
    "ice_floe_115": {
      "115": { "...": "фичи кадра 115" }
    }
  },
  "scenes": {
    "shoe_shop_45": {
      "num_frames": 9,
      "num_frames_norm": 0.03,
      "hue_mean_norm_mean": 0.46,
      "hue_mean_norm_std": 0.01,
      "val_mean_norm_mean": 0.56,
      "val_mean_norm_std": 0.02,
      "colorfulness_norm_mean": 0.48,
      "colorfulness_norm_std": 0.02,
      "global_contrast_norm_mean": 0.29,
      "global_contrast_norm_std": 0.00,
      "brightness_change_speed": 0.17,
      "scene_flicker_intensity": 0.25,
      "flash_events_count": 0.0,
      "flash_events_count_norm": 0.0,
      "color_change_speed": 0.18,
      "color_transition_variance": 0.01,
      "color_stability": 3.52,
      "color_temporal_entropy": 2.04,
      "color_pattern_periodicity": 0.0,
      "scene_color_shift_speed": 0.18,
      "scene_contrast": 73.38,
      "dynamic_range": 0.92,
      "...": "агрегаты для других числовых frame-level фич"
    },
    "ice_floe_115": {
      "num_frames": 9,
      "...": "аналогичные агрегаты"
    }
  },
  "video_features": {
    "num_frames_mean": 6.44,
    "num_frames_std": 2.27,
    "num_frames_min": 4.0,
    "num_frames_max": 10.0,
    "hue_entropy_mean_mean": 2.82,
    "hue_entropy_mean_std": 0.14,
    "colorfulness_norm_mean_mean": 0.63,
    "colorfulness_norm_mean_std": 0.11,
    "global_contrast_norm_mean_mean": 0.25,
    "global_contrast_norm_mean_std": 0.03,
    "color_distribution_entropy": 2.75,
    "color_distribution_gini": 0.21,
    "style_teal_orange_prob": 0.19,
    "style_film_prob": 1.0,
    "style_desaturated_prob": 0.46,
    "style_hyper_saturated_prob": 0.0,
    "style_vintage_prob": 0.41,
    "style_tiktok_prob": 0.2,
    "nima_mean": 1.33,
    "nima_std": 0.14,
    "laion_mean": 0.64,
    "laion_std": 0.11,
    "cinematic_lighting_score": 0.95,
    "professional_look_score": 0.88,
    "global_brightness_change_speed": 0.0,
    "global_color_change_speed": 0.0,
    "strobe_transition_frequency": 0.0,
    "global_color_periodicity": 0.0,
    "global_color_shift": 0.0
  },
  "sequence_inputs": {
    "frames": [
      [
        0.47,   // hue_mean_norm
        0.18,   // hue_std_norm
        2.80,   // hue_entropy_weighted
        0.19,   // sat_mean_norm
        0.55,   // val_mean_norm
        0.54,   // L_mean_norm
        0.29,   // global_contrast_norm
        0.07,   // local_contrast_mean_norm
        0.45,   // colorfulness_norm
        0.0012, // skin_tone_ratio
        0.0023, // overexposed_ratio
        0.0136, // underexposed_ratio
        0.0,    // vignetting_score_norm
        0.95,   // soft_light_prob
        0.61,   // dominant_lab_a_norm
        0.52    // dominant_lab_b_norm
      ],
      "... more frames ..."
    ],
    "scenes": [
      [
        9,      // num_frames
        0.03,   // num_frames_norm
        0.46,   // hue_mean_norm_mean
        0.56,   // val_mean_norm_mean
        0.48,   // colorfulness_norm_mean
        0.29    // global_contrast_norm_mean
        "... side‑features по сцене ..."
      ],
      "... more scenes ..."
    ],
    "global": [
      6.44, 2.27, 0.63, 0.11, 0.95, 0.88, 0.19, 1.0, 0.2, 0.0
      "... агрегированные video-level фичи ..."
    ]
  }
}
```

### Фичи:

#### 1. Frame-level (покадровые фичи)

- **hue_mean_norm, hue_std_norm** — средний hue и его std, нормированные в [0,1].  
- **hue_entropy, hue_entropy_weighted** — энтропия hue по 36 бинам и её версия, взвешенная по saturation.  
- **sat_mean_norm, val_mean_norm** — средняя насыщенность и яркость Value в [0,1].  
- **L_mean, L_contrast, L_mean_norm** — средняя яркость и контраст в Lab, плюс нормированная яркость.  
- **ab_balance** — баланс тёплых/холодных тонов в Lab.  
- **dominant_lab_a/b, dominant_lab_a_norm/b_norm** — координаты доминантного кластера в Lab и их нормализованные версии.  
- **colorfulness_index, colorfulness_norm** — индекс цветности и его нормализованный вариант.  
- **warm_vs_cold_ratio** — отношение тёплых к холодным тонам по hue.  
- **skin_tone_ratio** — доля пикселей кожи.  
- **color_palette_entropy** — энтропия палитры по hue.  
- **color_harmony_complementary_prob, color_harmony_analogous_prob** — вероятности базовых цветовых гармоний.

Освещение:

- **brightness_mean, brightness_std** — средняя яркость и std по серому.  
- **global_contrast, local_contrast, local_contrast_std** — глобальный и локальный контраст.  
- **brightness_entropy, contrast_entropy** — энтропия яркости и контраста.  
- **dynamic_range_db** — динамический диапазон в децибелах.  
- **overexposed_pixels, underexposed_pixels** — доля пере/недоэкспонированных пикселей.  
- **overexposed_ratio, underexposed_ratio** — алиасы тех же долей для удобства.  
- **highlight_clipping_ratio, shadow_clipping_ratio** — доля клиппинга.  
- **lighting_uniformity_index, center_brightness, corner_brightness, vignetting_score, vignetting_score_norm** — равномерность освещения и виньетирование.  
- **global_contrast_norm, local_contrast_mean_norm** — нормализованные контрасты для трансформера.

Источники света:

- **light_source_count_estimate** — оценка числа источников света (0–5).  
- **soft_light_probability, hard_light_probability** — вероятности мягкого/жёсткого света.  
- **soft_light_prob** — компактный алиас soft_light_probability (0–1).

#### 2. Scene-level (сценовые фичи)

- **num_frames, num_frames_norm** — количество и нормированная длина сцены.  
- `{feature}_mean`, `{feature}_std` — агрегаты по всем числовым покадровым фичам.  
- **brightness_change_speed, scene_flicker_intensity** — \(\mathbb{E}|\Delta value\_mean|\) и его std по кадрам.  
- **flash_events_count, flash_events_count_norm** — количество вспышек и его нормированная версия.  
- **color_change_speed, color_transition_variance** — скорость и вариативность изменения hue (с учётом цикличности).  
- **color_stability** — \(1 / (1 + mean\_color\_diff)\) по RGB‑средним соседних кадров.  
- **color_temporal_entropy** — энтропия hue‑последовательности (18 бинов).  
- **color_pattern_periodicity** — периодичность цветовых паттернов по автокорреляции.  
- **scene_color_shift_speed** — средняя \(|Δ hue\_mean|\) по сцене.  
- **scene_contrast** — средний global_contrast по сцене.  
- **dynamic_range** — \(\max(brightness\_mean) - \min(brightness\_mean)\) в сцене.

#### 3. Video-level (агрегаты по видео)

- Агрегаты `{feature}_mean`, `{feature}_std`, `{feature}_min`, `{feature}_max` для всех числовых сценовых фич.  
- **color_distribution_entropy, color_distribution_gini** — распределение hue по всему видео.  
- **style_teal_orange_prob, style_film_prob, style_desaturated_prob, style_hyper_saturated_prob, style_vintage_prob, style_tiktok_prob** — компактный вектор стилей цветокоррекции (0–1).  
- **nima_mean/std, laion_mean/std, cinematic_lighting_score, professional_look_score** — упрощённые эстетические и «кинематографичные» скоры (0–1).  
- **global_brightness_change_speed, global_color_change_speed, strobe_transition_frequency, global_color_periodicity, global_color_shift** — глобальная динамика яркости и цвета.

#### 4. sequence_inputs

- **`sequence_inputs["frames"]`** — компактная N x D последовательность нормализованных покадровых фич (основной вход VisualTransformer).  
- **`sequence_inputs["scenes"]`** — последовательность сценовых агрегатов (side‑features).  
- **`sequence_inputs["global"]`** — вектор видеоуровневых агрегатов (side‑features для головы/MLP).

## frames_composition

Модуль для комплексного анализа композиции кадров видео. Извлекает покадровые фичи и агрегирует их по всему видео для оценки качества композиции.

### Основные улучшения (v2.0)

- **Обнаружение объектов и лиц**: Маски (сегментация), face_pose, eye_gaze, landmarks_visibility_ratio, bbox_area_ratio
- **Composition Anchors**: Объединены Rule of Thirds и Golden Ratio в `composition_anchor_distance`
- **Balance**: Saliency map (с fallback на brightness+object_mask)
- **Depth Analysis**: Опциональный, компактный, с флагом `depth_reliable`
- **Symmetry**: Упрощена (horizontal/vertical в fast_mode)
- **Negative Space**: Сегментация вместо bbox
- **Visual Complexity**: Local variance вместо SLIC
- **Leading Lines**: Edge thinning + saliency mask
- **Per-frame вектор**: ~20 dims для VisualTransformer

### Модели:

```py
yolo_model = YOLO("yolo11n.pt")
```

```py
mp_face = mp.solutions.face_mesh
face_mesh = mp_face.FaceMesh(
    static_image_mode=True,
    max_num_faces=self.config.max_num_faces,
    min_detection_confidence=self.config.min_detection_confidence
)
```

```py
torch.hub.set_dir("./models")
midas_model = torch.hub.load(
    "intel-isl/MiDaS",
    "MiDaS_small",
    pretrained=True,
    trust_repo=True,
    verbose=False
)
transforms = torch.hub.load(
    "intel-isl/MiDaS",
    "transforms",
    trust_repo=True,
    verbose=False
)
```

```py
style_model = models.resnet50(pretrained=True)
style_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

### Выход:

```json
{
    "frame_count": 11,
    "video_composition_score": 0.5939282539387515,
    "numeric_features": {
        "frame_dimensions.width_mean": 1080.0,
        "frame_dimensions.width_std": 0.0,
        "frame_dimensions.width_min": 1080.0,
        "frame_dimensions.width_max": 1080.0,
        "frame_dimensions.width_median": 1080.0,
        "frame_dimensions.width_range": 0.0,
        "rule_of_thirds.quadrant_distribution.top_right_mean": 0.0,
        "rule_of_thirds.quadrant_distribution.top_right_std": 0.0,
        "rule_of_thirds.quadrant_distribution.top_right_min": 0.0,
        "rule_of_thirds.quadrant_distribution.top_right_max": 0.0,
        "rule_of_thirds.quadrant_distribution.top_right_median": 0.0,
        "rule_of_thirds.quadrant_distribution.top_right_range": 0.0,
        "golden_ratio.min_distance_normalized_mean": 0.07851574223086831,
        "golden_ratio.min_distance_normalized_std": 0.01778366199329337,
        "golden_ratio.min_distance_normalized_min": 0.054954209642960626,
        "golden_ratio.min_distance_normalized_max": 0.10096014386943827,
        "golden_ratio.min_distance_normalized_median": 0.08087995404786164,
        "golden_ratio.min_distance_normalized_range": 0.04600593422647765,
        "composition_style.style_probabilities.cinematic_mean": 0.26089885652105815,
        "composition_style.style_probabilities.cinematic_std": 0.004353131904611979,
        "composition_style.style_probabilities.cinematic_min": 0.25419352394147143,
        "composition_style.style_probabilities.cinematic_max": 0.2663283986924103,
        "composition_style.style_probabilities.cinematic_median": 0.2617053189767343,
        "composition_style.style_probabilities.cinematic_range": 0.012134874750938884,
        "balance.quadrant_weights.top_right_mean": 0.2403288818242248,
        "balance.quadrant_weights.top_right_std": 0.011951895622634335,
        "balance.quadrant_weights.top_right_min": 0.22367222778090048,
        "balance.quadrant_weights.top_right_max": 0.2722487468932586,
        "balance.quadrant_weights.top_right_median": 0.2345782087471911,
        "balance.quadrant_weights.top_right_range": 0.048576519112358135,
        "rule_of_thirds.quadrant_distribution.bottom_left_mean": 0.0,
        "rule_of_thirds.quadrant_distribution.bottom_left_std": 0.0,
        "rule_of_thirds.quadrant_distribution.bottom_left_min": 0.0,
        "rule_of_thirds.quadrant_distribution.bottom_left_max": 0.0,
        "rule_of_thirds.quadrant_distribution.bottom_left_median": 0.0,
        "rule_of_thirds.quadrant_distribution.bottom_left_range": 0.0,
        "depth.depth_std_mean": 0.2990017275918614,
        "depth.depth_std_std": 0.030900558481459194,
        "depth.depth_std_min": 0.24097318947315216,
        "depth.depth_std_max": 0.3375455141067505,
        "depth.depth_std_median": 0.29595935344696045,
        "depth.depth_std_range": 0.09657232463359833,
        "negative_space.quadrant_distribution.bottom_right_mean": 0.038453984328291634,
        "negative_space.quadrant_distribution.bottom_right_std": 0.033401313432571564,
        "negative_space.quadrant_distribution.bottom_right_min": 0.008892746642231941,
        "negative_space.quadrant_distribution.bottom_right_max": 0.0931520089507103,
        "negative_space.quadrant_distribution.bottom_right_median": 0.01666666753590107,
        "negative_space.quadrant_distribution.bottom_right_range": 0.08425926230847836,
        "negative_space.negative_space_entropy_mean": 0.6039543179948109,
        "negative_space.negative_space_entropy_std": 0.20803570764099683,
        "negative_space.negative_space_entropy_min": 0.34513927605942707,
        "negative_space.negative_space_entropy_max": 0.8662676045939778,
        "negative_space.negative_space_entropy_median": 0.5309776874026139,
        "negative_space.negative_space_entropy_range": 0.5211283285345507,
        "face_data.face_count_mean": 0.0,
        "face_data.face_count_std": 0.0,
        "face_data.face_count_min": 0.0,
        "face_data.face_count_max": 0.0,
        "face_data.face_count_median": 0.0,
        "face_data.face_count_range": 0.0,
        "depth.depth_entropy_mean": 4.754644707750797,
        "depth.depth_entropy_std": 0.2583742780039493,
        "depth.depth_entropy_min": 4.061392016397191,
        "depth.depth_entropy_max": 5.062659025630369,
        "depth.depth_entropy_median": 4.801505064299754,
        "depth.depth_entropy_range": 1.0012670092331781,
        "depth.midground_ratio_mean": 0.7984179906705949,
        "depth.midground_ratio_std": 0.0050015327578627235,
        "depth.midground_ratio_min": 0.7826017554012346,
        "depth.midground_ratio_max": 0.8,
        "depth.midground_ratio_median": 0.7999995177469136,
        "depth.midground_ratio_range": 0.01739824459876549,
        "depth.background_ratio_mean": 0.10000026304713806,
        "depth.background_ratio_std": 3.161433146934429e-07,
        "depth.background_ratio_min": 0.1,
        "depth.background_ratio_max": 0.10000096450617284,
        "depth.background_ratio_median": 0.1,
        "depth.background_ratio_range": 9.64506172831947e-07,
        "complexity.color_complexity_mean": 53.92323716982799,
        "complexity.color_complexity_std": 1.9168915952274224,
        "complexity.color_complexity_min": 48.91945594709912,
        "complexity.color_complexity_max": 56.10800354667701,
        "complexity.color_complexity_median": 54.05417418520958,
        "complexity.color_complexity_range": 7.188547599577895,
        "rule_of_thirds.main_subject_x_mean": 0.5084175084175084,
        "rule_of_thirds.main_subject_x_std": 0.0023756171366447966,
        "rule_of_thirds.main_subject_x_min": 0.5050925925925925,
        "rule_of_thirds.main_subject_x_max": 0.5120370370370371,
        "rule_of_thirds.main_subject_x_median": 0.5078703703703704,
        "rule_of_thirds.main_subject_x_range": 0.006944444444444531,
        "complexity.edge_density_mean": 0.06125968890291808,
        "complexity.edge_density_std": 0.026501548277549317,
        "complexity.edge_density_min": 0.013010706018518519,
        "complexity.edge_density_max": 0.09464265046296295,
        "complexity.edge_density_median": 0.07680314429012346,
        "complexity.edge_density_range": 0.08163194444444444,
        "symmetry.symmetry_details.diagonal_mean": 1.0,
        "symmetry.symmetry_details.diagonal_std": 1.0042345108077676e-16,
        "symmetry.symmetry_details.diagonal_min": 0.9999999999999998,
        "symmetry.symmetry_details.diagonal_max": 1.0,
        "symmetry.symmetry_details.diagonal_median": 1.0,
        "symmetry.symmetry_details.diagonal_range": 2.220446049250313e-16,
        "leading_lines.diagonal_lines_mean": 267.1818181818182,
        "leading_lines.diagonal_lines_std": 138.2072345046434,
        "leading_lines.diagonal_lines_min": 2.0,
        "leading_lines.diagonal_lines_max": 455.0,
        "leading_lines.diagonal_lines_median": 343.0,
        "leading_lines.diagonal_lines_range": 453.0,
        "negative_space.quadrant_distribution.top_right_mean": 0.24750701608982953,
        "negative_space.quadrant_distribution.top_right_std": 0.13075554523325938,
        "negative_space.quadrant_distribution.top_right_min": 0.09479166567325592,
        "negative_space.quadrant_distribution.top_right_max": 0.4146643579006195,
        "negative_space.quadrant_distribution.top_right_median": 0.20400077104568481,
        "negative_space.quadrant_distribution.top_right_range": 0.3198726922273636,
        "balance.mass_center_y_mean": 0.5031680108146049,
        "balance.mass_center_y_std": 0.01673879389292671,
        "balance.mass_center_y_min": 0.47888303545060534,
        "balance.mass_center_y_max": 0.5222466120004002,
        "balance.mass_center_y_median": 0.5030600078902262,
        "balance.mass_center_y_range": 0.04336357654979489,
        "depth.depth_p90_mean": 0.8347233631394125,
        "depth.depth_p90_std": 0.05217763451560173,
        "depth.depth_p90_min": 0.780290138721466,
        "depth.depth_p90_max": 0.9020773351192475,
        "depth.depth_p90_median": 0.7994986057281495,
        "depth.depth_p90_range": 0.12178719639778146,
        "frame_index_mean": 25.0,
        "frame_index_std": 15.811388300841896,
        "frame_index_min": 0.0,
        "frame_index_max": 50.0,
        "frame_index_median": 25.0,
        "frame_index_range": 50.0,
        "composition_style.style_probabilities.product_centered_mean": 0.4308780782212294,
        "composition_style.style_probabilities.product_centered_std": 0.0112427012567362,
        "composition_style.style_probabilities.product_centered_min": 0.41090789042010134,
        "composition_style.style_probabilities.product_centered_max": 0.44287462509961956,
        "composition_style.style_probabilities.product_centered_median": 0.4354306038858417,
        "composition_style.style_probabilities.product_centered_range": 0.031966734679518216,
        "rule_of_thirds.quadrant_distribution.bottom_right_mean": 1.6363636363636365,
        "rule_of_thirds.quadrant_distribution.bottom_right_std": 0.8813963377120598,
        "rule_of_thirds.quadrant_distribution.bottom_right_min": 1.0,
        "rule_of_thirds.quadrant_distribution.bottom_right_max": 3.0,
        "rule_of_thirds.quadrant_distribution.bottom_right_median": 1.0,
        "rule_of_thirds.quadrant_distribution.bottom_right_range": 2.0,
        "frame_dimensions.height_mean": 1920.0,
        "frame_dimensions.height_std": 0.0,
        "frame_dimensions.height_min": 1920.0,
        "frame_dimensions.height_max": 1920.0,
        "frame_dimensions.height_median": 1920.0,
        "frame_dimensions.height_range": 0.0,
        "balance.mass_center_x_mean": 0.4951597219268931,
        "balance.mass_center_x_std": 0.0015332728161178745,
        "balance.mass_center_x_min": 0.4920581065345277,
        "balance.mass_center_x_max": 0.4976804859145418,
        "balance.mass_center_x_median": 0.49518009985532274,
        "balance.mass_center_x_range": 0.0056223793800140864,
        "leading_lines.convergence_score_mean": 0.7487571163409755,
        "leading_lines.convergence_score_std": 0.02469699556617532,
        "leading_lines.convergence_score_min": 0.7232577320344191,
        "leading_lines.convergence_score_max": 0.7828909461035958,
        "leading_lines.convergence_score_median": 0.7292724321077895,
        "leading_lines.convergence_score_range": 0.059633214069176654,
        "balance.top_bottom_balance_mean": 0.9428412792084653,
        "balance.top_bottom_balance_std": 0.020843382505952288,
        "balance.top_bottom_balance_min": 0.9083173749437763,
        "balance.top_bottom_balance_max": 0.9925088803000872,
        "balance.top_bottom_balance_median": 0.9439306662257643,
        "balance.top_bottom_balance_range": 0.08419150535631093,
        "golden_ratio.golden_ratio_score_mean": 0.9214842577691317,
        "golden_ratio.golden_ratio_score_std": 0.017783661993293377,
        "golden_ratio.golden_ratio_score_min": 0.8990398561305617,
        "golden_ratio.golden_ratio_score_max": 0.9450457903570394,
        "golden_ratio.golden_ratio_score_median": 0.9191200459521384,
        "golden_ratio.golden_ratio_score_range": 0.046005934226477696,
        "negative_space.negative_space_balance_mean": 0.9073416472158649,
        "negative_space.negative_space_balance_std": 0.050362931760428505,
        "negative_space.negative_space_balance_min": 0.8320832997560501,
        "negative_space.negative_space_balance_max": 0.9568171203136444,
        "negative_space.negative_space_balance_median": 0.930015429854393,
        "negative_space.negative_space_balance_range": 0.1247338205575943,
        "rule_of_thirds.distance_to_thirds_mean": 0.2351205552974478,
        "rule_of_thirds.distance_to_thirds_std": 0.05839783696209945,
        "rule_of_thirds.distance_to_thirds_min": 0.15934718868363562,
        "rule_of_thirds.distance_to_thirds_max": 0.2990564483742331,
        "rule_of_thirds.distance_to_thirds_median": 0.25757282524235753,
        "rule_of_thirds.distance_to_thirds_range": 0.1397092596905975,
        "composition_style.style_probabilities.minimalist_mean": 0.3082224825241052,
        "composition_style.style_probabilities.minimalist_std": 0.01291619117026405,
        "composition_style.style_probabilities.minimalist_min": 0.2907964013677237,
        "composition_style.style_probabilities.minimalist_max": 0.3315573011211144,
        "composition_style.style_probabilities.minimalist_median": 0.3082139381907811,
        "composition_style.style_probabilities.minimalist_range": 0.0407608997533907,
        "composition_style.style_probabilities.vlog_mean": 0.0,
        "composition_style.style_probabilities.vlog_std": 0.0,
        "composition_style.style_probabilities.vlog_min": 0.0,
        "composition_style.style_probabilities.vlog_max": 0.0,
        "composition_style.style_probabilities.vlog_median": 0.0,
        "composition_style.style_probabilities.vlog_range": 0.0,
        "depth.depth_edge_density_mean": 0.0004504243827160493,
        "depth.depth_edge_density_std": 0.00031614168533946496,
        "depth.depth_edge_density_min": 0.0,
        "depth.depth_edge_density_max": 0.00083960262345679,
        "depth.depth_edge_density_median": 0.0003872492283950617,
        "depth.depth_edge_density_range": 0.00083960262345679,
        "negative_space.quadrant_distribution.bottom_left_mean": 0.07044069164178589,
        "negative_space.quadrant_distribution.bottom_left_std": 0.036020059122175464,
        "negative_space.quadrant_distribution.bottom_left_min": 0.035256557166576385,
        "negative_space.quadrant_distribution.bottom_left_max": 0.12350308895111084,
        "negative_space.quadrant_distribution.bottom_left_median": 0.048070985823869705,
        "negative_space.quadrant_distribution.bottom_left_range": 0.08824653178453445,
        "composition_style.style_confidence_mean": 0.4308780782212294,
        "composition_style.style_confidence_std": 0.0112427012567362,
        "composition_style.style_confidence_min": 0.41090789042010134,
        "composition_style.style_confidence_max": 0.44287462509961956,
        "composition_style.style_confidence_median": 0.4354306038858417,
        "composition_style.style_confidence_range": 0.031966734679518216,
        "depth.foreground_ratio_mean": 0.10158174628226711,
        "depth.foreground_ratio_std": 0.005001463431457804,
        "depth.foreground_ratio_min": 0.1,
        "depth.foreground_ratio_max": 0.11739776234567902,
        "depth.foreground_ratio_median": 0.1,
        "depth.foreground_ratio_range": 0.01739776234567901,
        "leading_lines.horizontal_lines_mean": 870.1818181818181,
        "leading_lines.horizontal_lines_std": 693.0588087577376,
        "leading_lines.horizontal_lines_min": 37.0,
        "leading_lines.horizontal_lines_max": 1747.0,
        "leading_lines.horizontal_lines_median": 1306.0,
        "leading_lines.horizontal_lines_range": 1710.0,
        "balance.center_offset_mean": 0.02820252937091807,
        "balance.center_offset_std": 0.010548374768285391,
        "balance.center_offset_min": 0.00787394388164879,
        "balance.center_offset_max": 0.039298766850727286,
        "balance.center_offset_median": 0.024701941541111935,
        "balance.center_offset_range": 0.031424822969078496,
        "object_data.object_count_mean": 1.6363636363636365,
        "object_data.object_count_std": 0.8813963377120598,
        "object_data.object_count_min": 1.0,
        "object_data.object_count_max": 3.0,
        "object_data.object_count_median": 1.0,
        "object_data.object_count_range": 2.0,
        "balance.left_right_balance_mean": 0.9812963393039061,
        "balance.left_right_balance_std": 0.009489324340335299,
        "balance.left_right_balance_min": 0.9660001909690099,
        "balance.left_right_balance_max": 0.9935575379969275,
        "balance.left_right_balance_median": 0.98352287967748,
        "balance.left_right_balance_range": 0.027557347027917567,
        "symmetry.radial_symmetry_mean": -0.15406138589939405,
        "symmetry.radial_symmetry_std": 0.15883123945378944,
        "symmetry.radial_symmetry_min": -0.2889291590914214,
        "symmetry.radial_symmetry_max": 0.16417738594905595,
        "symmetry.radial_symmetry_median": -0.19889096157601133,
        "symmetry.radial_symmetry_range": 0.45310654504047737,
        "leading_lines.avg_length_mean": 100.30412927987847,
        "leading_lines.avg_length_std": 16.018786524952212,
        "leading_lines.avg_length_min": 81.37990567240418,
        "leading_lines.avg_length_max": 118.21989871055314,
        "leading_lines.avg_length_median": 111.09152554630627,
        "leading_lines.avg_length_range": 36.83999303814896,
        "balance.quadrant_weights.bottom_right_mean": 0.2503192878277283,
        "balance.quadrant_weights.bottom_right_std": 0.010316211011330782,
        "balance.quadrant_weights.bottom_right_min": 0.2225703950305872,
        "balance.quadrant_weights.bottom_right_max": 0.26808921205783953,
        "balance.quadrant_weights.bottom_right_median": 0.2508236120074452,
        "balance.quadrant_weights.bottom_right_range": 0.04551881702725233,
        "depth.depth_p10_mean": 0.07791719819334421,
        "depth.depth_p10_std": 0.02703734882523542,
        "depth.depth_p10_min": 0.043678414076566696,
        "depth.depth_p10_max": 0.1533492237329483,
        "depth.depth_p10_median": 0.07535671293735505,
        "depth.depth_p10_range": 0.1096708096563816,
        "depth.bokeh_potential_mean": 0.5980034551837228,
        "depth.bokeh_potential_std": 0.06180111696291839,
        "depth.bokeh_potential_min": 0.4819463789463043,
        "depth.bokeh_potential_max": 0.675091028213501,
        "depth.bokeh_potential_median": 0.5919187068939209,
        "depth.bokeh_potential_range": 0.19314464926719666,
        "leading_lines.line_strength_mean": 0.06984720090797204,
        "leading_lines.line_strength_std": 0.05436743353936683,
        "leading_lines.line_strength_min": 0.001669616272195384,
        "leading_lines.line_strength_max": 0.13694261029260657,
        "leading_lines.line_strength_median": 0.10685969367265095,
        "leading_lines.line_strength_range": 0.13527299402041118,
        "balance.quadrant_weights.top_left_mean": 0.2681537207801162,
        "balance.quadrant_weights.top_left_std": 0.02328836378708892,
        "balance.quadrant_weights.top_left_min": 0.23767321474252293,
        "balance.quadrant_weights.top_left_max": 0.2914446945372662,
        "balance.quadrant_weights.top_left_median": 0.28007333206905594,
        "balance.quadrant_weights.top_left_range": 0.053771479794743254,
        "balance.overall_balance_score_mean": 0.9620688092561855,
        "balance.overall_balance_score_std": 0.010116211773867541,
        "balance.overall_balance_score_min": 0.948977829395734,
        "balance.overall_balance_score_max": 0.9880158799887836,
        "balance.overall_balance_score_median": 0.9581061621419761,
        "balance.overall_balance_score_range": 0.039038050593049545,
        "symmetry.symmetry_score_mean": 0.25626145821079976,
        "symmetry.symmetry_score_std": 0.08649112687724157,
        "symmetry.symmetry_score_min": 0.14732554165767717,
        "symmetry.symmetry_score_max": 0.35703973584227616,
        "symmetry.symmetry_score_median": 0.26186235594925744,
        "symmetry.symmetry_score_range": 0.209714194184599,
        "leading_lines.total_length_mean": 144835.15580277078,
        "leading_lines.total_length_std": 112736.31018723107,
        "leading_lines.total_length_min": 3462.116302024348,
        "leading_lines.total_length_max": 283964.196702749,
        "leading_lines.total_length_median": 221584.260799609,
        "leading_lines.total_length_range": 280502.08040072466,
        "rule_of_thirds.balance_score_mean": 0.0,
        "rule_of_thirds.balance_score_std": 0.0,
        "rule_of_thirds.balance_score_min": 0.0,
        "rule_of_thirds.balance_score_max": 0.0,
        "rule_of_thirds.balance_score_median": 0.0,
        "rule_of_thirds.balance_score_range": 0.0,
        "depth.depth_p50_mean": 0.7273010557348077,
        "depth.depth_p50_std": 0.03642305723021649,
        "depth.depth_p50_min": 0.6564784049987793,
        "depth.depth_p50_max": 0.7760311663150787,
        "depth.depth_p50_median": 0.7215226292610168,
        "depth.depth_p50_range": 0.11955276131629944,
        "symmetry.symmetry_details.vertical_mean": -0.06486244149148471,
        "symmetry.symmetry_details.vertical_std": 0.25547762059776813,
        "symmetry.symmetry_details.vertical_min": -0.35466816225705405,
        "symmetry.symmetry_details.vertical_max": 0.2736582360974301,
        "symmetry.symmetry_details.vertical_median": -0.16861509445238812,
        "symmetry.symmetry_details.vertical_range": 0.6283263983544842,
        "negative_space.object_background_ratio_mean": 0.8338549110022458,
        "negative_space.object_background_ratio_std": 0.09301953693579061,
        "negative_space.object_background_ratio_min": 0.711887538433075,
        "negative_space.object_background_ratio_max": 0.9354774281382561,
        "negative_space.object_background_ratio_median": 0.8794357627630234,
        "negative_space.object_background_ratio_range": 0.22358988970518112,
        "complexity.texture_entropy_mean": 5.622032847504382,
        "complexity.texture_entropy_std": 0.23920100952075335,
        "complexity.texture_entropy_min": 5.276426564500914,
        "complexity.texture_entropy_max": 5.924068308122954,
        "complexity.texture_entropy_median": 5.514911240547224,
        "complexity.texture_entropy_range": 0.6476417436220396,
        "overall_composition_score_mean": 0.5939282539387515,
        "overall_composition_score_std": 0.01175150273100835,
        "overall_composition_score_min": 0.5783730210278396,
        "overall_composition_score_max": 0.6084782920179257,
        "overall_composition_score_median": 0.5926452866501422,
        "overall_composition_score_range": 0.030105270990086086,
        "complexity.saturation_level_mean": 0.3072997449646245,
        "complexity.saturation_level_std": 0.09063161019913792,
        "complexity.saturation_level_min": 0.18852060071713872,
        "complexity.saturation_level_max": 0.3854184897724522,
        "complexity.saturation_level_median": 0.3800970557976277,
        "complexity.saturation_level_range": 0.19689788905531347,
        "complexity.overall_complexity_mean": 0.30767883782857797,
        "complexity.overall_complexity_std": 0.005817550244067053,
        "complexity.overall_complexity_min": 0.2923974306974919,
        "complexity.overall_complexity_max": 0.31490995622234025,
        "complexity.overall_complexity_median": 0.30804672164886787,
        "complexity.overall_complexity_range": 0.022512525524848337,
        "rule_of_thirds.quadrant_distribution.top_left_mean": 0.0,
        "rule_of_thirds.quadrant_distribution.top_left_std": 0.0,
        "rule_of_thirds.quadrant_distribution.top_left_min": 0.0,
        "rule_of_thirds.quadrant_distribution.top_left_max": 0.0,
        "rule_of_thirds.quadrant_distribution.top_left_median": 0.0,
        "rule_of_thirds.quadrant_distribution.top_left_range": 0.0,
        "leading_lines.line_count_mean": 1295.909090909091,
        "leading_lines.line_count_std": 943.1242523506121,
        "leading_lines.line_count_min": 41.0,
        "leading_lines.line_count_max": 2402.0,
        "leading_lines.line_count_median": 1953.0,
        "leading_lines.line_count_range": 2361.0,
        "depth.depth_mean_mean": 0.5803525393659418,
        "depth.depth_mean_std": 0.01779802073493067,
        "depth.depth_mean_min": 0.5414716005325317,
        "depth.depth_mean_max": 0.6048058271408081,
        "depth.depth_mean_median": 0.5808888673782349,
        "depth.depth_mean_range": 0.06333422660827637,
        "symmetry.symmetry_details.horizontal_mean": 0.24396966023407793,
        "symmetry.symmetry_details.horizontal_std": 0.19551307232860796,
        "symmetry.symmetry_details.horizontal_min": 0.004294421999348588,
        "symmetry.symmetry_details.horizontal_max": 0.5136815075156177,
        "symmetry.symmetry_details.horizontal_median": 0.10093023378748527,
        "symmetry.symmetry_details.horizontal_range": 0.5093870855162691,
        "symmetry.symmetry_details.radial_mean": -0.15406138589939405,
        "symmetry.symmetry_details.radial_std": 0.15883123945378944,
        "symmetry.symmetry_details.radial_min": -0.2889291590914214,
        "symmetry.symmetry_details.radial_max": 0.16417738594905595,
        "symmetry.symmetry_details.radial_median": -0.19889096157601133,
        "symmetry.symmetry_details.radial_range": 0.45310654504047737,
        "negative_space.negative_space_ratio_mean": 0.16614508899775418,
        "negative_space.negative_space_ratio_std": 0.09301953693579061,
        "negative_space.negative_space_ratio_min": 0.06452257186174393,
        "negative_space.negative_space_ratio_max": 0.28811246156692505,
        "negative_space.negative_space_ratio_median": 0.12056423723697662,
        "negative_space.negative_space_ratio_range": 0.22358988970518112,
        "negative_space.quadrant_distribution.top_left_mean": 0.30817866596308624,
        "negative_space.quadrant_distribution.top_left_std": 0.1753209151458269,
        "negative_space.quadrant_distribution.top_left_min": 0.11658371984958649,
        "negative_space.quadrant_distribution.top_left_max": 0.5326601266860962,
        "negative_space.quadrant_distribution.top_left_median": 0.22915315628051758,
        "negative_space.quadrant_distribution.top_left_range": 0.4160764068365097,
        "rule_of_thirds.main_subject_y_mean": 0.573035037878788,
        "rule_of_thirds.main_subject_y_std": 0.05121775579210053,
        "rule_of_thirds.main_subject_y_min": 0.5208333333333334,
        "rule_of_thirds.main_subject_y_max": 0.640625,
        "rule_of_thirds.main_subject_y_median": 0.5489583333333333,
        "rule_of_thirds.main_subject_y_range": 0.11979166666666663,
        "symmetry.horizontal_symmetry_mean": 0.24396966023407793,
        "symmetry.horizontal_symmetry_std": 0.19551307232860796,
        "symmetry.horizontal_symmetry_min": 0.004294421999348588,
        "symmetry.horizontal_symmetry_max": 0.5136815075156177,
        "symmetry.horizontal_symmetry_median": 0.10093023378748527,
        "symmetry.horizontal_symmetry_range": 0.5093870855162691,
        "leading_lines.vertical_lines_mean": 158.54545454545453,
        "leading_lines.vertical_lines_std": 128.38674785503912,
        "leading_lines.vertical_lines_min": 2.0,
        "leading_lines.vertical_lines_max": 343.0,
        "leading_lines.vertical_lines_median": 192.0,
        "leading_lines.vertical_lines_range": 341.0,
        "symmetry.diagonal_symmetry_mean": 1.0,
        "symmetry.diagonal_symmetry_std": 1.0042345108077676e-16,
        "symmetry.diagonal_symmetry_min": 0.9999999999999998,
        "symmetry.diagonal_symmetry_max": 1.0,
        "symmetry.diagonal_symmetry_median": 1.0,
        "symmetry.diagonal_symmetry_range": 2.220446049250313e-16,
        "rule_of_thirds.alignment_score_mean": 0.7648794447025522,
        "rule_of_thirds.alignment_score_std": 0.058397836962099436,
        "rule_of_thirds.alignment_score_min": 0.7009435516257669,
        "rule_of_thirds.alignment_score_max": 0.8406528113163644,
        "rule_of_thirds.alignment_score_median": 0.7424271747576425,
        "rule_of_thirds.alignment_score_range": 0.13970925969059744,
        "depth.depth_dynamic_range_mean": 0.7568061649460684,
        "depth.depth_dynamic_range_std": 0.06757434602712831,
        "depth.depth_dynamic_range_min": 0.6309524267911912,
        "depth.depth_dynamic_range_max": 0.8366201832890511,
        "depth.depth_dynamic_range_median": 0.7456775043159724,
        "depth.depth_dynamic_range_range": 0.20566775649785984,
        "symmetry.vertical_symmetry_mean": -0.06486244149148471,
        "symmetry.vertical_symmetry_std": 0.25547762059776813,
        "symmetry.vertical_symmetry_min": -0.35466816225705405,
        "symmetry.vertical_symmetry_max": 0.2736582360974301,
        "symmetry.vertical_symmetry_median": -0.16861509445238812,
        "symmetry.vertical_symmetry_range": 0.6283263983544842,
        "balance.quadrant_weights.bottom_left_mean": 0.2411981095679307,
        "balance.quadrant_weights.bottom_left_std": 0.027826142882528795,
        "balance.quadrant_weights.bottom_left_min": 0.21374600788423462,
        "balance.quadrant_weights.bottom_left_max": 0.2783947046670408,
        "balance.quadrant_weights.bottom_left_median": 0.22816522809220405,
        "balance.quadrant_weights.bottom_left_range": 0.06464869678280619
    },
    "qualitative_features": {
        "dominant_composition_style": "product_centered",
        "style_distribution": {
            "product_centered": 11
        },
        "dominant_symmetry_type": "diagonal",
        "symmetry_distribution": {
            "diagonal": 11
        },
        "style_consistency": 1.0
    },
    "frame_analysis_summary": {
        "total_frames_analyzed": 11,
        "best_frames": [
            {
                "index": 7,
                "score": 0.6084782920179257
            },
            {
                "index": 9,
                "score": 0.6064247949494282
            },
            {
                "index": 8,
                "score": 0.6063174124603956
            }
            ],
            "worst_frames": [
            {
                "index": 3,
                "score": 0.5810663163591584
            },
            {
                "index": 4,
                "score": 0.579738719360331
            },
            {
                "index": 5,
                "score": 0.5783730210278396
            }
        ],
        "style_summary": {
            "product_centered": {
                "count": 11,
                "avg_score": 0.5939282539387515,
                "best_score": 0.6084782920179257
            }
        },
        "score_range": {
            "min": 0.5783730210278396,
            "max": 0.6084782920179257,
            "mean": 0.5939282539387516
        }
    }
}
```

### Фичи:

#### Per-frame вектор для VisualTransformer (~20 dims)

Компактный вектор для подачи в VisualTransformer (последовательность кадров):

```py
def build_per_frame_vector(frame_analysis, frame_idx, total_frames):
    """
    Строит компактный per-frame вектор для VisualTransformer.
    Все значения нормализуются по train-set или per-video.
    """
    # time_norm
    time_norm = frame_idx / max(total_frames - 1, 1)
    
    # Главный субъект
    main_subject_x = frame_analysis['composition_anchors']['main_subject_x']
    main_subject_y = frame_analysis['composition_anchors']['main_subject_y']
    main_subject_area_ratio = frame_analysis.get('object_data', {}).get('main_subject', {}).get('bbox_area_ratio', 0.0)
    
    # Лица
    face_count = frame_analysis['face_data'].get('face_count', 0)
    face_count_norm = min(face_count / 5.0, 1.0)
    
    main_face = frame_analysis['face_data'].get('main_face')
    face_size_ratio = main_face.get('face_size_ratio', 0.0) if main_face else 0.0
    face_pose_yaw = main_face.get('face_pose', {}).get('yaw', 0.0) if main_face else 0.0
    face_pose_pitch = main_face.get('face_pose', {}).get('pitch', 0.0) if main_face else 0.0
    
    # Balance
    saliency_center_offset = frame_analysis['balance'].get('saliency_center_offset', 0.0)
    mass_center_offset = frame_analysis['balance'].get('center_offset', 0.0)
    
    # Depth
    depth_data = frame_analysis.get('depth', {})
    depth_mean_norm = depth_data.get('depth_mean', 0.5)
    depth_reliable_flag = 1.0 if depth_data.get('depth_reliable', False) else 0.0
    
    # Complexity
    edge_density = frame_analysis['complexity'].get('edge_density', 0.0)
    color_complexity = frame_analysis['complexity'].get('color_complexity', 0.0) / 180.0
    
    # Negative space
    negative_space_ratio = frame_analysis['negative_space'].get('negative_space_ratio', 0.0)
    
    # Composition anchors
    composition_anchor_distance = frame_analysis['composition_anchors'].get('composition_anchor_distance', 0.5)
    
    # Style embedding (упрощенная версия - можно заменить на learned embedding)
    style_probs = frame_analysis['composition_style'].get('style_probabilities', {})
    style_embedding = [
        style_probs.get('minimalist', 0.0),
        style_probs.get('cinematic', 0.0),
        style_probs.get('vlog', 0.0),
        style_probs.get('product_centered', 0.0)
    ]
    
    return np.array([
        time_norm,                    # 1
        main_subject_x,               # 2
        main_subject_y,               # 3
        main_subject_area_ratio,      # 4
        face_count_norm,              # 5
        face_size_ratio,              # 6
        face_pose_yaw,                # 7
        face_pose_pitch,              # 8
        saliency_center_offset,       # 9
        mass_center_offset,           # 10
        depth_mean_norm,              # 11
        depth_reliable_flag,          # 12
        edge_density,                 # 13
        color_complexity,             # 14
        negative_space_ratio,         # 15
        composition_anchor_distance,  # 16
        *style_embedding              # 17-20 (4 dims)
    ])  # Итого: 20 dims
```

⚠️ **ВАЖНО**: `overall_composition_score` НЕ включается в per-frame вектор, так как это агрегированная метрика с фиксированными весами, которая лишит трансформер возможности учиться.

#### Video-level агрегаты (не для трансформера)

#### 1. - 5. frame_count, video_composition_score, numeric_features, qualitative_features, frame_analysis_summary

В конце анализа по всем кадрам идет агрегация

```py
numeric_features = {}

all_keys = set()
for analysis in frame_analyses:
    all_keys.update(self._extract_numeric_keys(analysis))

for key in all_keys:
    values = []
    for analysis in frame_analyses:
        val = self._get_nested_value(analysis, key)
        if val is not None:
            values.append(val)
    
    if values:
        values = np.array(values)
        numeric_features[f'{key}_mean'] = float(values.mean())
        numeric_features[f'{key}_std'] = float(values.std())
        numeric_features[f'{key}_min'] = float(values.min())
        numeric_features[f'{key}_max'] = float(values.max())
        numeric_features[f'{key}_median'] = float(np.median(values))
        numeric_features[f'{key}_range'] = float(values.max() - values.min())

style_counts = {}
symmetry_types = {}

for analysis in frame_analyses:
    if 'composition_style' in analysis:
        style = analysis['composition_style'].get('dominant_style', 'unknown')
        style_counts[style] = style_counts.get(style, 0) + 1
    
    if 'symmetry' in analysis:
        sym_type = analysis['symmetry'].get('dominant_symmetry_type', 'unknown')
        symmetry_types[sym_type] = symmetry_types.get(sym_type, 0) + 1

dominant_style = max(style_counts.items(), key=lambda x: x[1])[0] if style_counts else 'unknown'
dominant_symmetry = max(symmetry_types.items(), key=lambda x: x[1])[0] if symmetry_types else 'unknown'

consistency_score = 0.0
if style_counts:
    total_frames = len(frame_analyses)
    max_style_count = max(style_counts.values())
    consistency_score = max_style_count / total_frames

qualitative = {
    'dominant_composition_style': dominant_style,
    'style_distribution': style_counts,
    'dominant_symmetry_type': dominant_symmetry,
    'symmetry_distribution': symmetry_types,
    'style_consistency': float(consistency_score)
}

video_score = float(np.mean([a.get('overall_composition_score', 0) for a in frame_analyses]))

scores = []
for i, analysis in enumerate(frame_analyses):
    score = analysis.get('overall_composition_score', 0)
    scores.append((i, score))

scores.sort(key=lambda x: x[1], reverse=True)

best_frames = scores[:3]
worst_frames = scores[-3:] if len(scores) >= 3 else scores

styles_summary = {}
for analysis in frame_analyses:
    if 'composition_style' in analysis:
        style = analysis['composition_style'].get('dominant_style', 'unknown')
        if style not in styles_summary:
            styles_summary[style] = {
                'count': 0,
                'avg_score': 0,
                'best_score': 0
            }
        
        styles_summary[style]['count'] += 1
        score = analysis.get('overall_composition_score', 0)
        styles_summary[style]['avg_score'] += score
        styles_summary[style]['best_score'] = max(
            styles_summary[style]['best_score'], score
        )

for style in styles_summary:
    if styles_summary[style]['count'] > 0:
        styles_summary[style]['avg_score'] /= styles_summary[style]['count']

frame_analysis_summary = {
    'total_frames_analyzed': len(frame_analyses),
    'best_frames': [{'index': idx, 'score': score} for idx, score in best_frames],
    'worst_frames': [{'index': idx, 'score': score} for idx, score in worst_frames],
    'style_summary': styles_summary,
    'score_range': {
        'min': min([s[1] for s in scores]) if scores else 0,
        'max': max([s[1] for s in scores]) if scores else 0,
        'mean': np.mean([s[1] for s in scores]) if scores else 0
    }
}

return {
    'frame_count': len(frame_analyses),
    'video_composition_score': video_score,
    'numeric_features': numeric_features,
    'qualitative_features': qualitative,
    'frame_analysis_summary': frame_analysis_summary
}
```

#### 3. numeric_features

Обшщее:

```py
def extract_objects(self, frame: np.ndarray) -> Dict:
    """
    Улучшенная детекция объектов с масками (сегментация) и дополнительными фичами.
    """
    H, W = frame.shape[:2]
    frame_area = H * W
    results = self.models.yolo_model(
        frame, 
        conf=self.config.yolo_conf_threshold,
        verbose=False
    )[0]
    
    objects = []
    object_mask = np.zeros((H, W), dtype=np.float32)
    object_centers = []
    
    # Сортируем по confidence и ограничиваем количество
    boxes_sorted = sorted(
        zip(results.boxes.xyxy, results.boxes.conf, results.boxes.cls),
        key=lambda x: x[1],
        reverse=True
    )[:self.config.max_detections]
    
    for box_xyxy, conf, cls in boxes_sorted:
        x1, y1, x2, y2 = map(int, box_xyxy)
        bbox_area = (x2 - x1) * (y2 - y1)
        bbox_area_ratio = bbox_area / frame_area
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        
        # Пытаемся получить маску (если доступна сегментация)
        if self.config.use_segmentation and hasattr(results, 'masks'):
            # YOLOv8-seg предоставляет маски
            mask = results.masks.data[...].cpu().numpy()
            object_mask = np.maximum(object_mask, mask)
        
        objects.append({
            'bbox': [x1, y1, x2, y2],
            'center': (cx, cy),
            'center_x_norm': cx / W,
            'center_y_norm': cy / H,
            'confidence': conf,
            'class': label,
            'class_id': cls,
            'bbox_area_ratio': bbox_area_ratio
        })
    
    if results.boxes is not None:
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = self.models.yolo_model.names[cls]
            
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            
            objects.append({
                'bbox': [x1, y1, x2, y2],
                'center': (cx, cy),
                'confidence': conf,
                'class': label,
                'class_id': cls
            })
            
            object_centers.append((cx, cy))
            object_mask[y1:y2, x1:x2] = 1.0
    
    return {
        'objects': objects,
        'object_mask': object_mask,
        'object_centers': object_centers,
        'object_count': len(objects)
    }

def extract_faces(self, frame: np.ndarray) -> Dict:
    H, W = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = self.models.face_mesh.process(rgb_frame)
    
    faces = []
    face_landmarks_list = []
    
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            landmarks = []
            for landmark in face_landmarks.landmark:
                x, y = int(landmark.x * W), int(landmark.y * H)
                landmarks.append((x, y))
            
            xs = [lm[0] for lm in landmarks]
            ys = [lm[1] for lm in landmarks]
            x1, x2 = min(xs), max(xs)
            y1, y2 = min(ys), max(ys)
            
            face_center = (np.mean(xs), np.mean(ys))
            
            faces.append({
                'bbox': [x1, y1, x2, y2],
                'center': face_center,
                'landmarks': landmarks[:10]
            })
            
            face_landmarks_list.append(face_landmarks.landmark)
    
    return {
        'faces': faces,
        'face_landmarks': face_landmarks_list[0] if face_landmarks_list else None,
        'face_count': len(faces)
    }
```

```py
H, W = frame.shape[:2]

object_data = self.extract_objects(frame)
face_data = self.extract_faces(frame)

main_subject = None
if face_data['face_landmarks']:
    main_subject = face_data['faces'][0]['center']
elif object_data['object_centers']:
    main_subject = object_data['object_centers'][0]

main_subject_norm = None
if main_subject:
    main_subject_norm = (main_subject[0] / W, main_subject[1] / H)
```

#### 3.1. frame_dimensions

```py
H, W = frame.shape[:2]

frame_dimensions = {'height': H, 'width': W}
```

#### 3.2. rule_of_thirds

#### 3.2.1. quadrant_distribution

#### 3.2.1.1. - 3.2.1.4. top_right, bottom_left, bottom_right, top_left

```py
H, W = frame.shape[:2]
quadrants = {
    'top_left': 0, 'top_right': 0,
    'bottom_left': 0, 'bottom_right': 0
}

for obj in object_data['objects']:
    cx, cy = obj['center']
    if cy < H/2:
        if cx < W/2:
            quadrants['top_left'] += 1
        else:
            quadrants['top_right'] += 1
    else:
        if cx < W/2:
            quadrants['bottom_left'] += 1
        else:
            quadrants['bottom_right'] += 1
```

#### 3.2.2. alignment_score

```py
third_x = [W / 3, 2 * W / 3]
third_y = [H / 3, 2 * H / 3]

main_subject = None

if face_data['faces']:
    main_subject = face_data['faces'][0]['center']
elif object_data['objects']:
    objects = object_data['objects']
    areas = [(obj['bbox'][2] - obj['bbox'][0]) * 
            (obj['bbox'][3] - obj['bbox'][1]) 
            for obj in objects]
    main_idx = np.argmax(areas)
    main_subject = objects[main_idx]['center']
else:
    main_subject = (W / 2, H / 2)

mx, my = main_subject

min_dist = float('inf')
best_point = None

for tx in third_x:
    for ty in third_y:
        dist = np.sqrt((mx - tx)**2 + (my - ty)**2)
        if dist < min_dist:
            min_dist = dist
            best_point = (tx, ty)

max_dist = np.sqrt((W/2)**2 + (H/2)**2)
alignment_score = max(0, 1.0 - (min_dist / max_dist))
```

#### 3.2.3. - 3.2.4. main_subject_x, main_subject_y

```py
main_subject_x = float(mx / W)
main_subject_y = float(my / H)
```

#### 3.2.5. distance_to_thirds

```py
distance_to_thirds = float(min_dist / max_dist)
```

#### 3.2.6. balance_score

```py
total_objs = sum(quadrants.values())
balance_score = 1.0
```

#### 3.3. composition_anchors (новое: объединенные Rule of Thirds + Golden Ratio)

```py
def analyze_composition_anchors(frame, object_data, face_data):
    """
    Объединенный анализ композиционных якорей.
    Возвращает минимальное расстояние до любого эстетического якоря.
    """
    H, W = frame.shape[:2]
    
    # Определяем главный субъект (приоритет лицу)
    if face_data.get('main_face'):
        main_subject_x_norm = face_data['main_face']['center_x_norm']
        main_subject_y_norm = face_data['main_face']['center_y_norm']
    elif object_data.get('main_subject'):
        main_subject_x_norm = object_data['main_subject'][0] / W
        main_subject_y_norm = object_data['main_subject'][1] / H
    else:
        main_subject_x_norm, main_subject_y_norm = 0.5, 0.5
    
    # Эстетические якоря: Rule of Thirds, Golden Ratio, Center
    aesthetic_points = []
    
    # Rule of Thirds (4 точки)
    for tx in [W/3, 2*W/3]:
        for ty in [H/3, 2*H/3]:
            aesthetic_points.append({
                'point': (tx, ty),
                'type': 'rule_of_thirds',
                'normalized': (tx/W, ty/H)
            })
    
    # Golden Ratio (4 точки)
    phi = 1.618033988749895
    for gx, gy in [(W/phi, H/phi), (W*(phi-1), H/phi), 
                    (W/phi, H*(phi-1)), (W*(phi-1), H*(phi-1))]:
        aesthetic_points.append({
            'point': (gx, gy),
            'type': 'golden_ratio',
            'normalized': (gx/W, gy/H)
        })
    
    # Center
    aesthetic_points.append({
        'point': (W/2, H/2),
        'type': 'center',
        'normalized': (0.5, 0.5)
    })
    
    # Находим ближайший якорь
    mx, my = main_subject_x_norm * W, main_subject_y_norm * H
    min_distance = min(
        np.sqrt((mx - ax)**2 + (my - ay)**2)
        for anchor in aesthetic_points
        for ax, ay in [anchor['point']]
    )
    
    max_possible = np.sqrt((W/2)**2 + (H/2)**2)
    composition_anchor_distance = min_distance / max_possible
    
    return {
        'composition_anchor_distance': composition_anchor_distance,
        'closest_anchor_type': closest_type,
        'alignment_score': 1.0 - composition_anchor_distance
    }
if total_objs > 0:
    quadrant_balance = [quadrants['top_left'] + quadrants['bottom_right'],
                        quadrants['top_right'] + quadrants['bottom_left']]
    balance_score = 1.0 - abs(quadrant_balance[0] - quadrant_balance[1]) / total_objs
```

#### 3.2.7. main_subject_position

```py
grid_x = 1 if mx < third_x[0] else 2 if mx < third_x[1] else 3
grid_y = 1 if my < third_y[0] else 2 if my < third_y[1] else 3
grid_position = f"{grid_x}-{grid_y}"
```

#### 3.3. balance

#### 3.3.1. quadrant_weights

#### 3.3.1.1. - 3.3.1.4. top_right, bottom_right, top_left, bottom_left

```py
# Улучшенная версия: использует saliency map (если доступно)
H, W = frame.shape[:2]

if use_saliency:
    # Saliency через градиенты и контраст (lightweight proxy)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]) / 8.0
    contrast = np.abs(cv2.filter2D(gray, -1, kernel))
    saliency_map = (gradient_magnitude + contrast) / 2.0
    saliency_map = (saliency_map - saliency_map.min()) / (saliency_map.max() - saliency_map.min() + 1e-6)
    weight_map = saliency_map.astype(np.float32)
else:
    # Fallback: brightness + object_mask
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    obj_norm = object_mask / (object_mask.max() + 1e-6)
    weight_map = (brightness_weight * gray + object_weight * obj_norm)
quadrant_weights = {
    'top_left': weight_map[:H//2, :W//2].sum(),
    'top_right': weight_map[:H//2, W//2:].sum(),
    'bottom_left': weight_map[H//2:, :W//2].sum(),
    'bottom_right': weight_map[H//2:, W//2:].sum()
}
```

#### 3.3.2. - 3.3.3. mass_center_y, mass_center_x

```py
y_coords, x_coords = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
mass_x = np.sum(x_coords * weight_map) / (np.sum(weight_map) + 1e-6)
mass_y = np.sum(y_coords * weight_map) / (np.sum(weight_map) + 1e-6)

mass_center_x = float(mass_x / W)
mass_center_y = float(mass_y / H)
```

#### 3.3.4. center_offset

```py
center_x, center_y = W / 2, H / 2
offset_distance = np.sqrt((mass_x - center_x)**2 + (mass_y - center_y)**2)
max_offset = np.sqrt((W/2)**2 + (H/2)**2)
normalized_offset = offset_distance / max_offset
center_offset = float(normalized_offset)
```

#### 3.3.5. - 3.3.7. left_right_balance, top_bottom_balance, overall_balance_score

```py
left_right_balance = abs(quadrants['top_left'] + quadrants['bottom_left'] - quadrants['top_right'] - quadrants['bottom_right'])
top_bottom_balance = abs(quadrants['top_left'] + quadrants['top_right'] - quadrants['bottom_left'] - quadrants['bottom_right'])
overall_balance_score = 1.0 - (left_right_balance + top_bottom_balance) / 2.0
```

#### 3.4. depth

#### 3.4.1. - 3.4.12. depth_std, depth_entropy, midground_ratio, background_ratio, depth_p90, depth_edge_density, foreground_ratio, depth_p10, bokeh_potential, depth_p50, depth_mean, depth_dynamic_range

```py
# Улучшенная версия: опциональный depth с проверкой разрешения и флагом надежности
H, W = frame.shape[:2]

# Проверяем условия для depth analysis
if not use_midas or min(H, W) < min_resolution_for_depth:
    depth_reliable = False
    depth_mean = 0.5
    depth_std = 0.0
    foreground_ratio = 0.0
    bokeh_potential = 0.0
else:
    try:
        mm = ModelManager()
        model, transform = mm.midas
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        input_tensor = transform(rgb).to(device)

        with torch.no_grad():
            depth = model(input_tensor)          # [1, h, w]
            depth = depth.unsqueeze(1)           # [1, 1, h, w]
            depth = F.interpolate(
                depth,
                size=(H, W),
                mode="bicubic",
        align_corners=False
    ).squeeze()

depth = depth.cpu().numpy()

d_min, d_max = depth.min(), depth.max()
depth = (depth - d_min) / (d_max - d_min + 1e-6)
mean = float(depth.mean())
std = float(depth.std())
p10, p50, p90 = np.percentile(depth, [10, 50, 90])
dynamic_range = float(p90 - p10)
fg_mask = depth <= p10
bg_mask = depth >= p90
mg_mask = (~fg_mask) & (~bg_mask)
fg_ratio = float(fg_mask.mean())
mg_ratio = float(mg_mask.mean())
bg_ratio = float(bg_mask.mean())
depth_uint8 = (depth * 255).astype(np.uint8)
edges = cv2.Canny(depth_uint8, 50, 150)
depth_edge_density = float(edges.mean() / 255.0)
hist, _ = np.histogram(depth, bins=64, range=(0, 1))
prob = hist / (hist.sum() + 1e-8)
entropy = float(-np.sum(prob * np.log2(prob + 1e-8)))
bokeh_potential = float(np.clip(std * 2.0, 0.0, 1.0))

return {
    "depth_mean": mean,
    "depth_std": std,
    "depth_p10": float(p10),
    "depth_p50": float(p50),
    "depth_p90": float(p90),
    "depth_dynamic_range": dynamic_range,
    "depth_entropy": entropy,
    "depth_edge_density": depth_edge_density,
    "foreground_ratio": fg_ratio,
    "midground_ratio": mg_ratio,
    "background_ratio": bg_ratio,
    "bokeh_potential": bokeh_potential,
    "depth_method": "midas_small"
}
```

#### 3.5. symmetry

#### 3.5.1. - 3.5.7. symmetry_score, dominant_symmetry_type, horizontal_symmetry, vertical_symmetry, diagonal_symmetry, radial_symmetry, symmetry_details

```py
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
H, W = gray.shape
h_flip = cv2.flip(gray, 1)
horizontal_corr = np.corrcoef(gray.flatten(), h_flip.flatten())[0, 1]
horizontal_score = float(np.nan_to_num(horizontal_corr, nan=0.0))
v_flip = cv2.flip(gray, 0)
vertical_corr = np.corrcoef(gray.flatten(), v_flip.flatten())[0, 1]
vertical_score = float(np.nan_to_num(vertical_corr, nan=0.0))
diag_flip = cv2.flip(cv2.flip(gray, -1), -1)
diag_corr = np.corrcoef(gray.flatten(), diag_flip.flatten())[0, 1]
diag_score = float(np.nan_to_num(diag_corr, nan=0.0))
center = (W // 2, H // 2)
max_radius = min(W, H) // 2
polar = cv2.linearPolar(gray, center, max_radius, cv2.WARP_FILL_OUTLIERS)
radial_flip = cv2.flip(polar, 1)
radial_corr = np.corrcoef(polar.flatten(), radial_flip.flatten())[0, 1]
radial_score = float(np.nan_to_num(radial_corr, nan=0.0))

scores = {
    'horizontal': horizontal_score,
    'vertical': vertical_score,
    'diagonal': diag_score,
    'radial': radial_score
}

best_symmetry = max(scores.items(), key=lambda x: x[1])
symmetry_score = float(np.mean([horizontal_score, vertical_score, diag_score, radial_score]))

return {
    'symmetry_score': symmetry_score,
    'dominant_symmetry_type': best_symmetry[0],
    'horizontal_symmetry': horizontal_score,
    'vertical_symmetry': vertical_score,
    'diagonal_symmetry': diag_score,
    'radial_symmetry': radial_score,
    'symmetry_details': scores
}
```

#### 3.6. negative_space

#### 3.6.1. - 3.6.5. quadrant_distribution, negative_space_entropy, negative_space_balance, object_background_ratio, negative_space_ratio

```py
H, W = frame.shape[:2]
negative_space_mask = 1.0 - object_mask
negative_space_ratio = float(negative_space_mask.mean())
quadrants = {
    'top_left': negative_space_mask[:H//2, :W//2].mean(),
    'top_right': negative_space_mask[:H//2, W//2:].mean(),
    'bottom_left': negative_space_mask[H//2:, :W//2].mean(),
    'bottom_right': negative_space_mask[H//2:, W//2:].mean()
}
left_balance = abs(quadrants['top_left'] + quadrants['bottom_left'] - quadrants['top_right'] - quadrants['bottom_right'])
negative_space_balance = 1.0 - left_balance
hist, _ = np.histogram(negative_space_mask, bins=256, range=(0, 1))
hist_norm = hist / (hist.sum() + 1e-6)
entropy = float(-np.sum(hist_norm * np.log2(hist_norm + 1e-6)))
object_background_ratio = 1.0 - negative_space_ratio

return {
    'negative_space_ratio': negative_space_ratio,
    'negative_space_balance': float(negative_space_balance),
    'negative_space_entropy': entropy,
    'object_background_ratio': float(object_background_ratio),
    'quadrant_distribution': {k: float(v) for k, v in quadrants.items()}
}
```

#### 3.7. complexity

#### 3.7.1. - 3.7.5. edge_density, texture_entropy, color_complexity, saturation_level, overall_complexity

```py
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)
edge_density = float(edges.mean() / 255.0)

try:
    segments = slic(frame, n_segments=self.config.slic_n_segments,
                    compactness=self.config.slic_compactness,
                    start_label=1)
    texture_entropy = float(shannon_entropy(segments))
except:
    texture_entropy = 0.0

hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
hue_std = float(hsv[:, :, 0].std())
saturation_mean = float(hsv[:, :, 1].mean() / 255.0)
complexity_score = (edge_density + texture_entropy / 10.0 + hue_std / 180.0) / 3.0

return {
    'edge_density': edge_density,
    'texture_entropy': texture_entropy,
    'color_complexity': hue_std,
    'saturation_level': saturation_mean,
    'overall_complexity': float(complexity_score)
}
```

#### 3.8. leading_lines

#### 3.8.1. - 3.8.7. line_count, total_length, avg_length, horizontal_lines, vertical_lines, diagonal_lines, convergence_score

```py
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)
lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=80, minLineLength=50, maxLineGap=10)

line_features = {'line_count': 0,'total_length': 0.0,'avg_length': 0.0,'horizontal_lines': 0,'vertical_lines': 0,'diagonal_lines': 0,'convergence_score': 0.0}

if lines is not None:
    lines = lines.reshape(-1, 4)
    line_features['line_count'] = len(lines)
    
    lengths = []
    angles = []
    endpoints = []
    
    for x1, y1, x2, y2 in lines:
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        lengths.append(length)
        
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        angle = (angle + 180) % 180
        angles.append(angle)
        
        endpoints.append(((x1, y1), (x2, y2)))
    
    line_features['total_length'] = float(sum(lengths))
    line_features['avg_length'] = float(np.mean(lengths))
    
    for angle in angles:
        if angle < 30 or angle > 150:
            line_features['vertical_lines'] += 1
        elif 60 < angle < 120:
            line_features['horizontal_lines'] += 1
        else:
            line_features['diagonal_lines'] += 1
    
    if len(endpoints) > 1:
        convergence_points = []
        for i in range(len(endpoints)):
            for j in range(i+1, len(endpoints)):
                line1 = endpoints[i]
                line2 = endpoints[j]
                mid1 = ((line1[0][0] + line1[1][0]) / 2, 
                        (line1[0][1] + line1[1][1]) / 2)
                mid2 = ((line2[0][0] + line2[1][0]) / 2, 
                        (line2[0][1] + line2[1][1]) / 2)
                dist = np.sqrt((mid1[0] - mid2[0])**2 + (mid1[1] - mid2[1])**2)
                convergence_points.append(dist)
        
        if convergence_points:
            avg_convergence = np.mean(convergence_points)
            max_dist = np.sqrt(frame.shape[0]**2 + frame.shape[1]**2)
            line_features['convergence_score'] = float(1.0 - avg_convergence / max_dist)

if line_features['line_count'] > 0:
    line_strength = min(line_features['total_length'] / (frame.shape[0] * frame.shape[1]), 1.0)
else:
    line_strength = 0.0

line_features['line_strength'] = float(line_strength)

return line_features
```

#### 3.9. golden_ratio

#### 3.9.1. - 3.9.3. golden_ratio_score, closest_orientation, min_distance_normalized

```py
H, W = frame.shape[:2]
phi = 1.618033988749895
golden_points = [
    (W / phi, H / phi),
    (W * (phi - 1), H / phi),
    (W / phi, H * (phi - 1)),
    (W * (phi - 1), H * (phi - 1))
]
mx, my = main_subject_pos
mx, my = mx * W, my * H

distances = []
for gx, gy in golden_points:
    dist = np.sqrt((mx - gx)**2 + (my - gy)**2)
    distances.append(dist)

min_dist = min(distances)
max_possible = np.sqrt(W**2 + H**2)
golden_score = max(0, 1.0 - (min_dist / max_possible))
orientations = ['top_left', 'top_right', 'bottom_left', 'bottom_right']
closest_idx = np.argmin(distances)

return {
    'golden_ratio_score': float(golden_score),
    'closest_orientation': orientations[closest_idx],
    'min_distance_normalized': float(min_dist / max_possible)
}
```

#### 3.10. composition_style

#### 3.10.1. - 3.10.3. style_probabilities, dominant_style, style_confidence

```py
eps = 1e-6

def g(path, default=0.0):
    cur = analysis
    for p in path:
        if not isinstance(cur, dict) or p not in cur:
            return default
        cur = cur[p]
    return cur

complexity = np.clip(g(["complexity", "overall_complexity"]), 0, 1)
neg_space = np.clip(g(["negative_space", "negative_space_ratio"]), 0, 1)
obj_count = g(["object_data", "object_count"], 0)
obj_density = np.clip(obj_count / 8.0, 0, 1)
depth_std = np.clip(g(["depth", "depth_std"]), 0, 1)
depth_edges = np.clip(g(["depth", "depth_edge_density"]), 0, 1)
bokeh = np.clip(g(["depth", "bokeh_potential"]), 0, 1)
center_offset = np.clip(g(["balance", "center_offset"]), 0, 1)
symmetry = np.clip(g(["symmetry", "symmetry_score"]), 0, 1)
thirds = np.clip(g(["rule_of_thirds", "alignment_score"]), 0, 1)
face_count = g(["face_data", "face_count"], 0)
faces = g(["face_data", "faces"], [])

styles = {}
styles["minimalist"] = (
    0.45 * (1.0 - complexity) +
    0.35 * neg_space +
    0.20 * (1.0 - obj_density)
)

styles["cinematic"] = (
    0.35 * depth_std +
    0.25 * depth_edges +
    0.20 * (1.0 - center_offset) +
    0.20 * (1.0 - symmetry)
)

vlog_score = 0.0
if face_count > 0 and len(faces) > 0:
    fx = faces[0]["center"][0] / (W + eps)
    face_centering = 1.0 - abs(fx - 0.5) * 2.0  # [0..1]
    vlog_score = (
        0.45 * face_centering +
        0.35 * (1.0 - complexity) +
        0.20 * obj_density
    )
styles["vlog"] = vlog_score

product_score = 0.0
objs = g(["object_data", "objects"], [])
if objs:
    max_area = 0.0
    frame_area = H * W
    for o in objs:
        x1, y1, x2, y2 = o["bbox"]
        area = max(0, (x2 - x1) * (y2 - y1))
        max_area = max(max_area, area)

    size_ratio = np.clip(max_area / (frame_area + eps), 0, 1)

    product_score = (
        0.45 * size_ratio +
        0.30 * thirds +
        0.25 * bokeh
    )

styles["product_centered"] = product_score

for k in styles:
    styles[k] = float(np.clip(styles[k], 0.0, 1.0))

total = sum(styles.values()) + eps
styles = {k: v / total for k, v in styles.items()}

dominant_style = max(styles.items(), key=lambda x: x[1])[0]

return {
    "style_probabilities": styles,
    "dominant_style": dominant_style,
    "style_confidence": float(styles[dominant_style])
}
```

#### 3.11. overall_composition_score

```py
weights = {
    'rule_of_thirds': 0.2,
    'balance': 0.15,
    'symmetry': 0.1,
    'negative_space': 0.15,
    'depth': 0.15,
    'leading_lines': 0.1,
    'complexity': 0.1,
    'style_confidence': 0.05
}

weighted_scores = []
used_weights = []

rot = analysis.get('rule_of_thirds')
if rot and 'alignment_score' in rot:
    weighted_scores.append(rot['alignment_score'] * weights['rule_of_thirds'])
    used_weights.append(weights['rule_of_thirds'])

balance = analysis.get('balance')
if balance and 'overall_balance_score' in balance:
    weighted_scores.append(balance['overall_balance_score'] * weights['balance'])
    used_weights.append(weights['balance'])

symmetry = analysis.get('symmetry')
if symmetry and 'symmetry_score' in symmetry:
    weighted_scores.append(symmetry['symmetry_score'] * weights['symmetry'])
    used_weights.append(weights['symmetry'])

neg = analysis.get('negative_space')
if neg and 'negative_space_balance' in neg:
    weighted_scores.append(neg['negative_space_balance'] * weights['negative_space'])
    used_weights.append(weights['negative_space'])

depth = analysis.get('depth')
if depth:
    depth_contrast = float(np.clip(depth.get('depth_contrast', 0.0), 0.0, 1.0))
    bokeh_potential = float(np.clip(depth.get('bokeh_potential', 0.0), 0.0, 1.0))

    depth_score = 0.5 * depth_contrast + 0.5 * bokeh_potential
    weighted_scores.append(depth_score * weights['depth'])
    used_weights.append(weights['depth'])

lines = analysis.get('leading_lines')
if lines and 'line_strength' in lines:
    weighted_scores.append(lines['line_strength'] * weights['leading_lines'])
    used_weights.append(weights['leading_lines'])

complexity_block = analysis.get('complexity')
if complexity_block and 'overall_complexity' in complexity_block:
    complexity = np.clip(complexity_block['overall_complexity'], 0.0, 1.0)
    complexity_score = max(0.0, 1.0 - abs(complexity - 0.5) * 2.0)
    weighted_scores.append(complexity_score * weights['complexity'])
    used_weights.append(weights['complexity'])

style = analysis.get('composition_style')
if style and 'style_confidence' in style:
    weighted_scores.append(style['style_confidence'] * weights['style_confidence'])
    used_weights.append(weights['style_confidence'])

if not weighted_scores:
    return 0.0

total_weight = sum(used_weights)
final_score = sum(weighted_scores) / max(total_weight, 1e-6)

return float(np.clip(final_score, 0.0, 1.0))
```

## shot_quality

### Выход:

```json
{
    "frames": {
        "0": {
            "sharpness_laplacian": 282.23291107833217,
            "sharpness_tenengrad": 6191.513206983024,
            "sharpness_smd2": 105.48051298725967,
            "motion_blur_probability": 0.5524605952141297,
            "edge_clarity_index": 0.010444155092592593,
            "blur_score": 0.28223291107833215,
            "focus_accuracy_score": 0.40541117706986635,
            "spatial_frequency_mean": 0.14036138545732907,
            "noise_level": 0.00873942393809557,
            "noise_cbdnet_stats": {
                "noise_mean": 0.0,
                "noise_std": 0.0,
                "noise_energy": 0.0
            },
            "noise_level_luma": 0.004086665343493223,
            "noise_level_chroma": 0.0017770176054909825,
            "iso_estimated_value": 125.7459916640073,
            "grain_strength": 0.9062851071357727,
            "noise_spatial_entropy": 1.593319296836853,
            "underexposure_ratio": 0.4451524019241333,
            "overexposure_ratio": 0.006647858768701553,
            "midtones_balance": 0.3385782837867737,
            "exposure_histogram_skewness": 4.9329023361206055,
            "highlight_recovery_potential": 0.9933521151542664,
            "shadow_recovery_potential": 0.5548475980758667,
            "contrast_global": 68.95363835868497,
            "contrast_local": 3.297163387345679,
            "contrast_dynamic_range": 1.0,
            "contrast_clarity_score": 1.0,
            "microcontrast": 0.5599928498268127,
            "wb_r": 72.71403694058642,
            "wb_g": 82.16442226080247,
            "wb_b": 60.715373263888885,
            "color_cast_type": "green",
            "skin_tone_accuracy_score": 0.9203651716916404,
            "color_fidelity_index": 0.539660632610321,
            "color_noise_level": 0.01642541214823723,
            "color_uniformity_score": 0.747990746532204,
            "compression_blockiness_score": 106.4944402337272,
            "banding_intensity": 0.7414174968984507,
            "ringing_artifacts_level": 1.0,
            "bitrate_estimation_score": 0.0,
            "codec_artifact_entropy": 0.9176333155832297,
            "vignetting_level": 4.653134645061726,
            "chromatic_aberration_level": 1.4464187885802469,
            "distortion_type": "pincushion",
            "lens_sharpness_drop_off": 0.9785833024667806,
            "lens_obstruction_probability": 9.615180948922779e-05,
            "lens_dirt_probability": 1.0,
            "veiling_glare_score": 0.029050804295987206,
            "fog_haziness_score": 0.0030145643350185892,
            "temporal_flicker_score": 0.0,
            "rolling_shutter_artifacts_score": 0.0,
            "quality_cinematic_prob": 0.1470947265625,
            "quality_lowlight_cinematic_prob": 0.1395263671875,
            "quality_smartphone_good_prob": 0.144775390625,
            "quality_smartphone_poor_prob": 0.1419677734375,
            "quality_webcam_prob": 0.1448974609375,
            "quality_screenrecord_prob": 0.1427001953125,
            "quality_surveillance_prob": 0.138916015625,
            "aesthetic_score": 0.6381813611557134,
            "clip_embedding": [
                0.00797271728515625,
                ...and 766 values...
                -0.0132904052734375
            ]
        },
        "5": {
            "sharpness_laplacian": ...,
            ...
        },
        "10": {...}
        ...
    },
    "frame_features": {
        "avg_iso_estimated_value": 163.45513916015625,
        "std_iso_estimated_value": 24.137897491455078,
        "min_iso_estimated_value": 125.59379577636719,
        "max_iso_estimated_value": 194.0288543701172,
        "avg_contrast_dynamic_range": 0.997504472732544,
        "std_contrast_dynamic_range": 0.0053594643250107765,
        "min_contrast_dynamic_range": 0.9843137264251709,
        "max_contrast_dynamic_range": 1.0,
        "avg_midtones_balance": 0.3231237232685089,
        "std_midtones_balance": 0.05360579490661621,
        "min_midtones_balance": 0.2098563015460968,
        "max_midtones_balance": 0.36032116413116455,
        "avg_veiling_glare_score": 0.019700147211551666,
        "std_veiling_glare_score": 0.02385139651596546,
        "min_veiling_glare_score": 0.003953271079808474,
        "max_veiling_glare_score": 0.06836672872304916,
        "avg_quality_surveillance_prob": 0.1399591565132141,
        "std_quality_surveillance_prob": 0.0010336586274206638,
        "min_quality_surveillance_prob": 0.138671875,
        "max_quality_surveillance_prob": 0.14208984375,
        "avg_quality_smartphone_poor_prob": 0.14163485169410706,
        "std_quality_smartphone_poor_prob": 0.0005913935019634664,
        "min_quality_smartphone_poor_prob": 0.140869140625,
        "max_quality_smartphone_poor_prob": 0.1427001953125,
        "avg_wb_b": 86.67137908935547,
        "std_wb_b": 20.426462173461914,
        "min_wb_b": 60.71537399291992,
        "max_wb_b": 128.3212127685547,
        "avg_sharpness_tenengrad": 6723.2841796875,
        "std_sharpness_tenengrad": 1719.2457275390625,
        "min_sharpness_tenengrad": 3607.310546875,
        "max_sharpness_tenengrad": 9637.0166015625,
        "avg_contrast_clarity_score": 1.0,
        "std_contrast_clarity_score": 0.0,
        "min_contrast_clarity_score": 1.0,
        "max_contrast_clarity_score": 1.0,
        "avg_wb_r": 84.93345642089844,
        "std_wb_r": 19.804636001586914,
        "min_wb_r": 72.71403503417969,
        "max_wb_r": 126.74759674072266,
        "avg_quality_webcam_prob": 0.14627352356910706,
        "std_quality_webcam_prob": 0.0006907123024575412,
        "min_quality_webcam_prob": 0.1448974609375,
        "max_quality_webcam_prob": 0.1470947265625,
        "avg_exposure_histogram_skewness": 5.109077453613281,
        "std_exposure_histogram_skewness": 0.06552010774612427,
        "min_exposure_histogram_skewness": 4.9329023361206055,
        "max_exposure_histogram_skewness": 5.177124977111816,
        "avg_underexposure_ratio": 0.2105618119239807,
        "std_underexposure_ratio": 0.07439949363470078,
        "min_underexposure_ratio": 0.17186874151229858,
        "max_underexposure_ratio": 0.4451524019241333,
        "avg_color_uniformity_score": 0.7698869705200195,
        "std_color_uniformity_score": 0.007714730687439442,
        "min_color_uniformity_score": 0.7479907274246216,
        "max_color_uniformity_score": 0.7755506038665771,
        "avg_chromatic_aberration_level": 9.693726539611816,
        "std_chromatic_aberration_level": 4.707952499389648,
        "min_chromatic_aberration_level": 1.4464187622070312,
        "max_chromatic_aberration_level": 15.212334632873535,
        "avg_noise_level_chroma": 0.0034708837047219276,
        "std_noise_level_chroma": 0.0012645239476114511,
        "min_noise_level_chroma": 0.0013660754775628448,
        "max_noise_level_chroma": 0.005023794714361429,
        "avg_quality_screenrecord_prob": 0.1406915783882141,
        "std_quality_screenrecord_prob": 0.0009020952275022864,
        "min_quality_screenrecord_prob": 0.1395263671875,
        "max_quality_screenrecord_prob": 0.1427001953125,
        "avg_spatial_frequency_mean": 0.15300093591213226,
        "std_spatial_frequency_mean": 0.021734915673732758,
        "min_spatial_frequency_mean": 0.11015830188989639,
        "max_spatial_frequency_mean": 0.17356278002262115,
        "avg_motion_blur_probability": 0.5588899254798889,
        "std_motion_blur_probability": 0.019921449944376945,
        "min_motion_blur_probability": 0.5409846901893616,
        "max_motion_blur_probability": 0.6000564694404602,
        "avg_contrast_global": 60.75063705444336,
        "std_contrast_global": 6.992555141448975,
        "min_contrast_global": 54.22650909423828,
        "max_contrast_global": 73.39219665527344,
        "avg_banding_intensity": 0.7051272988319397,
        "std_banding_intensity": 0.027981191873550415,
        "min_banding_intensity": 0.6788219213485718,
        "max_banding_intensity": 0.7539072036743164,
        "avg_fog_haziness_score": 0.003318234346807003,
        "std_fog_haziness_score": 0.003387270262464881,
        "min_fog_haziness_score": 0.0010787744540721178,
        "max_fog_haziness_score": 0.010499908588826656,
        "avg_lens_sharpness_drop_off": 0.2164309173822403,
        "std_lens_sharpness_drop_off": 0.29956331849098206,
        "min_lens_sharpness_drop_off": 0.0,
        "max_lens_sharpness_drop_off": 0.9785832762718201,
        "avg_edge_clarity_index": 0.04837949573993683,
        "std_edge_clarity_index": 0.020303472876548767,
        "min_edge_clarity_index": 0.0104441549628973,
        "max_edge_clarity_index": 0.07125916332006454,
        "avg_aesthetic_score": 0.6532429456710815,
        "std_aesthetic_score": 0.02822086028754711,
        "min_aesthetic_score": 0.6339597105979919,
        "max_aesthetic_score": 0.7127498388290405,
        "avg_noise_level_luma": 0.010072245262563229,
        "std_noise_level_luma": 0.003831412410363555,
        "min_noise_level_luma": 0.004062507767230272,
        "max_noise_level_luma": 0.014925215393304825,
        "avg_microcontrast": 0.6785458922386169,
        "std_microcontrast": 0.20111891627311707,
        "min_microcontrast": 0.29730695486068726,
        "max_microcontrast": 0.940936803817749,
        "avg_color_noise_level": 0.02456689067184925,
        "std_color_noise_level": 0.00816378928720951,
        "min_color_noise_level": 0.010038466192781925,
        "max_color_noise_level": 0.03457053750753403,
        "avg_codec_artifact_entropy": 1.372809648513794,
        "std_codec_artifact_entropy": 0.1830337643623352,
        "min_codec_artifact_entropy": 0.9176332950592041,
        "max_codec_artifact_entropy": 1.571119785308838,
        "avg_quality_lowlight_cinematic_prob": 0.13917125761508942,
        "std_quality_lowlight_cinematic_prob": 0.0002520822163205594,
        "min_quality_lowlight_cinematic_prob": 0.138671875,
        "max_quality_lowlight_cinematic_prob": 0.1395263671875,
        "avg_lens_dirt_probability": 1.0,
        "std_lens_dirt_probability": 0.0,
        "min_lens_dirt_probability": 1.0,
        "max_lens_dirt_probability": 1.0,
        "avg_sharpness_laplacian": 450.7858581542969,
        "std_sharpness_laplacian": 213.7984161376953,
        "min_sharpness_laplacian": 79.55229949951172,
        "max_sharpness_laplacian": 796.82568359375,
        "avg_blur_score": 0.4507858455181122,
        "std_blur_score": 0.2137984186410904,
        "min_blur_score": 0.0795523002743721,
        "max_blur_score": 0.7968257069587708,
        "avg_quality_cinematic_prob": 0.14749422669410706,
        "std_quality_cinematic_prob": 0.0005048967432230711,
        "min_quality_cinematic_prob": 0.1468505859375,
        "max_quality_cinematic_prob": 0.1485595703125,
        "avg_vignetting_level": -19.843406677246094,
        "std_vignetting_level": 35.91448211669922,
        "min_vignetting_level": -95.5359878540039,
        "max_vignetting_level": 4.653134822845459,
        "avg_quality_smartphone_good_prob": 0.14478649199008942,
        "std_quality_smartphone_good_prob": 0.0008121526334434748,
        "min_quality_smartphone_good_prob": 0.14306640625,
        "max_quality_smartphone_good_prob": 0.1458740234375,
        "avg_bitrate_estimation_score": 0.0,
        "std_bitrate_estimation_score": 0.0,
        "min_bitrate_estimation_score": 0.0,
        "max_bitrate_estimation_score": 0.0,
        "avg_temporal_flicker_score": 91.14862060546875,
        "std_temporal_flicker_score": 41.501068115234375,
        "min_temporal_flicker_score": 0.0,
        "max_temporal_flicker_score": 144.38487243652344,
        "avg_highlight_recovery_potential": 0.9949933886528015,
        "std_highlight_recovery_potential": 0.00527555588632822,
        "min_highlight_recovery_potential": 0.98414546251297,
        "max_highlight_recovery_potential": 0.9982233643531799,
        "avg_noise_level": 0.014954076148569584,
        "std_noise_level": 0.002965194871649146,
        "min_noise_level": 0.00873942393809557,
        "max_noise_level": 0.017800865694880486,
        "avg_wb_g": 98.07890319824219,
        "std_wb_g": 18.343427658081055,
        "min_wb_g": 82.16442108154297,
        "max_wb_g": 136.33859252929688,
        "avg_ringing_artifacts_level": 1.0,
        "std_ringing_artifacts_level": 0.0,
        "min_ringing_artifacts_level": 1.0,
        "max_ringing_artifacts_level": 1.0,
        "avg_focus_accuracy_score": 0.7285686135292053,
        "std_focus_accuracy_score": 0.17812378704547882,
        "min_focus_accuracy_score": 0.4054111838340759,
        "max_focus_accuracy_score": 0.9744747281074524,
        "avg_grain_strength": 0.8993720412254333,
        "std_grain_strength": 0.19322222471237183,
        "min_grain_strength": 0.49157145619392395,
        "max_grain_strength": 1.0,
        "avg_sharpness_smd2": 131.39300537109375,
        "std_sharpness_smd2": 16.326906204223633,
        "min_sharpness_smd2": 105.13397979736328,
        "max_sharpness_smd2": 146.42242431640625,
        "avg_compression_blockiness_score": 95.86984252929688,
        "std_compression_blockiness_score": 3.5033082962036133,
        "min_compression_blockiness_score": 92.3011245727539,
        "max_compression_blockiness_score": 106.49443817138672,
        "avg_overexposure_ratio": 0.005006575956940651,
        "std_overexposure_ratio": 0.005275558680295944,
        "min_overexposure_ratio": 0.0017766205128282309,
        "max_overexposure_ratio": 0.015854552388191223,
        "avg_lens_obstruction_probability": 0.00013327239139471203,
        "std_lens_obstruction_probability": 4.601048203767277e-05,
        "min_lens_obstruction_probability": 5.3114032198209316e-05,
        "max_lens_obstruction_probability": 0.00020761278574354947,
        "avg_rolling_shutter_artifacts_score": 0.13339251279830933,
        "std_rolling_shutter_artifacts_score": 0.11742038279771805,
        "min_rolling_shutter_artifacts_score": 0.0,
        "max_rolling_shutter_artifacts_score": 0.4169347882270813,
        "avg_shadow_recovery_potential": 0.7894381880760193,
        "std_shadow_recovery_potential": 0.07439949363470078,
        "min_shadow_recovery_potential": 0.5548475980758667,
        "max_shadow_recovery_potential": 0.8281312584877014,
        "avg_skin_tone_accuracy_score": 0.6664265990257263,
        "std_skin_tone_accuracy_score": 0.22054459154605865,
        "min_skin_tone_accuracy_score": 0.5,
        "max_skin_tone_accuracy_score": 0.9703139066696167,
        "avg_color_fidelity_index": 0.5775403380393982,
        "std_color_fidelity_index": 0.012185280211269855,
        "min_color_fidelity_index": 0.539660632610321,
        "max_color_fidelity_index": 0.584287166595459,
        "avg_noise_spatial_entropy": 1.9421532154083252,
        "std_noise_spatial_entropy": 0.1899554282426834,
        "min_noise_spatial_entropy": 1.593319296836853,
        "max_noise_spatial_entropy": 2.1466665267944336,
        "avg_contrast_local": 7.934553623199463,
        "std_contrast_local": 3.0667600631713867,
        "min_contrast_local": 3.072570323944092,
        "max_contrast_local": 11.83682632446289
    },
    "temporal_features": {
        "temporal_sharpness_stability": 0.5257206559181213,
        "temporal_exposure_stability": 0.8341013193130493,
        "exposure_consistency_over_time": 0.9463942030525008,
        "temporal_noise_variation": 0.3803927004337311
    },
    "total_frames_processed": 11
}
```

### Фичи:

#### 1. frames

#### 1.1. - 1.3. sharpness_laplacian, sharpness_tenengrad, sharpness_smd2

```py
def sharpness_laplacian(frame):
    return cv2.Laplacian(frame, cv2.CV_64F).var()

def sharpness_tenengrad(frame):
    gx = cv2.Sobel(frame, cv2.CV_64F, 1, 0)
    gy = cv2.Sobel(frame, cv2.CV_64F, 0, 1)
    g = gx*gx + gy*gy
    return np.mean(g)

def smd2(frame):
    diff1 = np.abs(frame[:, 1:] - frame[:, :-1])
    diff2 = np.abs(frame[1:, :] - frame[:-1, :])
    return np.mean(diff1) + np.mean(diff2)
```

#### 1.4. motion_blur_probability

```py
def motion_blur_probability(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    fft = np.fft.fft2(gray)
    mag = np.log(np.abs(fft) + 1)
    blur = 1 - (np.mean(mag) / np.max(mag))
    return float(np.clip(blur, 0, 1))
```

#### 1.5. edge_clarity_index

```py
def edge_clarity(frame):
    edges = cv2.Canny(frame, 100, 200)
    return np.mean(edges) / 255
```

#### 1.6. blur_score

```py
def blur_score(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    return float(np.clip(laplacian_var / 1000.0, 0, 1))
```

#### 1.7. focus_accuracy_score

```py
def focus_accuracy_score(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(gx**2 + gy**2)
    return float(np.clip(np.mean(gradient_magnitude) / 50.0, 0, 1))
```

#### 1.8. spatial_frequency_mean

```py
def spatial_frequency_mean(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    fft = np.fft.fft2(gray)
    fft_shift = np.fft.fftshift(fft)
    magnitude = np.abs(fft_shift)
    h, w = gray.shape
    y, x = np.ogrid[:h, :w]
    center_y, center_x = h // 2, w // 2
    distances = np.sqrt((x - center_x)**2 + (y - center_y)**2)
    weighted_freq = np.sum(magnitude * distances) / (np.sum(magnitude) + 1e-10)
    return float(weighted_freq / max(h, w))
```

#### 1.9. - 1.15. noise_level, noise_cbdnet_stats, noise_level_luma, noise_level_chroma, iso_estimated_value, grain_strength, noise_spatial_entropy

```py

def noise_estimation_dncnn(
    dncnn: torch.nn.Module,
    frame_rgb: np.ndarray,
    device: str = "cuda"
) -> float:
    if frame_rgb.ndim == 3:
        gray = (
            0.299 * frame_rgb[..., 0] +
            0.587 * frame_rgb[..., 1] +
            0.114 * frame_rgb[..., 2]
        )
    else:
        gray = frame_rgb

    gray = gray.astype(np.float32) / 255.0

    tensor = (torch.from_numpy(gray).unsqueeze(0).unsqueeze(0).to(device))

    with torch.no_grad():
        denoised = dncnn(tensor)
        noise_map = torch.abs(denoised - tensor)
        noise_level = float(noise_map.mean().item())

    return noise_level

def noise_estimation_cbdnet(cbdnet, frame_rgb, device="cuda"):
    img = frame_rgb.astype(np.float32) / 255.0
    tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)

    with torch.no_grad():
        noise_map, _ = cbdnet(tensor)

    noise_mean = noise_map.mean().item()
    noise_std = noise_map.std().item()
    noise_energy = (noise_map ** 2).mean().item()

    return {
        "noise_mean": float(noise_mean),
        "noise_std": float(noise_std),
        "noise_energy": float(noise_energy)
    }

def noise_level_luma(gray):
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    diff = np.abs(gray.astype(np.float32) - blurred.astype(np.float32))
    return float(np.mean(diff) / 255.0)

def noise_level_chroma(frame):
    yuv = cv2.cvtColor(frame, cv2.COLOR_BGR2YUV)
    u = yuv[:, :, 1].astype(np.float32)
    v = yuv[:, :, 2].astype(np.float32)
    u_blur = cv2.GaussianBlur(u, (5, 5), 0)
    v_blur = cv2.GaussianBlur(v, (5, 5), 0)
    u_noise = np.mean(np.abs(u - u_blur))
    v_noise = np.mean(np.abs(v - v_blur))
    return float((u_noise + v_noise) / 2.0 / 255.0)

def iso_estimated_value(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    noise = noise_level_luma(gray)
    iso_est = 100 + noise * 6300
    return float(np.clip(iso_est, 100, 6400))

def grain_strength(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    filtered = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    grain = np.std(filtered)
    return float(np.clip(grain / 50.0, 0, 1))

def noise_spatial_entropy(gray):
    h, w = gray.shape
    block_size = 8
    entropies = []
    for i in range(0, h - block_size, block_size):
        for j in range(0, w - block_size, block_size):
            block = gray[i:i+block_size, j:j+block_size]
            hist = cv2.calcHist([block], [0], None, [256], [0, 256]).ravel()
            hist = hist / (hist.sum() + 1e-10)
            entropies.append(entropy(hist))
    return float(np.mean(entropies) if entropies else 0.0)
```

#### 1.16. 1.21. underexposure_ratio, overexposure_ratio, midtones_balance, exposure_histogram_skewness, highlight_recovery_potential, shadow_recovery_potential

```py
def exposure_metrics(gray):
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    hist /= hist.sum()

    under = hist[:30].sum()
    over = hist[230:].sum()
    mid = hist[80:170].sum()
    skew = entropy(hist)

    highlight_recovery = 1.0 - over 
    shadow_recovery = 1.0 - under

    return {
        "underexposure_ratio": float(under),
        "overexposure_ratio": float(over),
        "midtones_balance": float(mid),
        "exposure_histogram_skewness": float(skew),
        "highlight_recovery_potential": float(highlight_recovery),
        "shadow_recovery_potential": float(shadow_recovery),
    }
```

```py
def contrast_global(gray):
    return float(gray.std())

def contrast_local(gray):
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(np.mean(np.abs(lap)))

def contrast_dynamic_range(gray):
    min_val = np.min(gray)
    max_val = np.max(gray)
    if max_val > min_val:
        return float((max_val - min_val) / 255.0)
    return 0.0

def contrast_clarity_score(gray):
    local_contrast = contrast_local(gray)
    gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_mag = np.mean(np.sqrt(gx**2 + gy**2))
    return float(np.clip((local_contrast + gradient_mag / 100.0) / 2.0, 0, 1))

def microcontrast(gray):
    kernel = np.array([[0, -1, 0], [-1, 4, -1], [0, -1, 0]])
    filtered = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    micro = np.std(filtered)
    return float(np.clip(micro / 30.0, 0, 1))
```

#### 1.22. - 1.24. wb_r, wb_g, wb_b

```py
def white_balance_shift(frame):
    means = frame.mean(axis=(0, 1))
    return {
        "wb_r": float(means[2]),  # Red channel mean
        "wb_g": float(means[1]),  # Green channel mean
        "wb_b": float(means[0])   # Blue channel mean
    }
```

#### 1.25. color_cast_type

```py
def color_cast(frame):
    b, g, r = frame.mean(axis=(0, 1))
    if r > g and r > b:
        return "red"
    if g > r and g > b:
        return "green"
    if b > g and b > r:
        return "blue"
    return "neutral"
```

#### 1.26. skin_tone_accuracy_score

```py
def skin_tone_accuracy_score(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    skin_ratio = np.sum(mask > 0) / (frame.shape[0] * frame.shape[1])
    if skin_ratio > 0.01:
        skin_pixels = frame[mask > 0]
        r_mean = np.mean(skin_pixels[:, 2])
        g_mean = np.mean(skin_pixels[:, 1])
        b_mean = np.mean(skin_pixels[:, 0])
        if r_mean > g_mean > b_mean:
            return float(1.0 - abs(r_mean - g_mean) / 255.0)
    return 0.5
```

#### 1.27. color_fidelity_index

```py
def color_fidelity_index(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180]).ravel()
    h_hist = h_hist / (h_hist.sum() + 1e-10)
    color_entropy = entropy(h_hist)
    return float(np.clip(color_entropy / 7.0, 0, 1))
```

#### 1.28. color_noise_level

```py
def color_noise_level(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    a = lab[:, :, 1].astype(np.float32)
    b = lab[:, :, 2].astype(np.float32)
    a_blur = cv2.GaussianBlur(a, (5, 5), 0)
    b_blur = cv2.GaussianBlur(b, (5, 5), 0)
    a_noise = np.std(a - a_blur)
    b_noise = np.std(b - b_blur)
    return float((a_noise + b_noise) / 2.0 / 128.0)
```

#### 1.29. color_uniformity_score

```py
def color_uniformity_score(frame):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h_std = np.std(hsv[:, :, 0])
    s_std = np.std(hsv[:, :, 1])
    v_std = np.std(hsv[:, :, 2])
    uniformity = 1.0 - np.clip((h_std + s_std + v_std) / (180 + 255 + 255), 0, 1)
    return float(uniformity)
```

#### 1.30. - 1.34. compression_blockiness_score, banding_intensity, ringing_artifacts_level, bitrate_estimation_score, codec_artifact_entropy

```py
def blockiness(frame):
    diffs = np.abs(frame[:, 8:] - frame[:, :-8]).mean()
    return float(diffs)

def banding(gray):
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    diff = np.abs(gray - blurred).mean()
    return float(1 - diff / 255)

def ringing_artifacts_level(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    kernel = np.array([[-1, -1, -1, -1, -1],
                       [-1,  2,  2,  2, -1],
                       [-1,  2,  8,  2, -1],
                       [-1,  2,  2,  2, -1],
                       [-1, -1, -1, -1, -1]])
    filtered = cv2.filter2D(gray.astype(np.float32), -1, kernel)
    ringing = np.std(filtered)
    return float(np.clip(ringing / 50.0, 0, 1))

def bitrate_estimation_score(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    block = blockiness(frame)
    band = banding(gray)
    artifacts = (block + band) / 2.0
    bitrate_score = 1.0 - np.clip(artifacts, 0, 1)
    return float(bitrate_score)

def codec_artifact_entropy(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    h, w = gray.shape
    block_size = 8
    entropies = []
    for i in range(0, h - block_size, block_size):
        for j in range(0, w - block_size, block_size):
            block = gray[i:i+block_size, j:j+block_size]
            block_std = np.std(block)
            entropies.append(block_std)
    if entropies:
        hist = np.histogram(entropies, bins=20)[0]
        hist = hist / (hist.sum() + 1e-10)
        return float(entropy(hist))
    return 0.0
```

#### 1.35. - 1.41. vignetting_level, chromatic_aberration_level, distortion_type, lens_sharpness_drop_off, lens_obstruction_probability, lens_dirt_probability, veiling_glare_score

```py
def vignetting(frame):
    h, w = frame.shape[:2]
    center = frame[h//4:3*h//4, w//4:3*w//4].mean()
    corners = np.mean([frame[0:h//4, 0:w//4],
                       frame[0:h//4, 3*w//4:w],
                       frame[3*h//4:h, 0:w//4],
                       frame[3*h//4:h, 3*w//4:w]])
    return float(center - corners)

def chromatic_aberration(frame):
    shift = np.abs(cv2.Canny(frame[:,:,2],100,200) - cv2.Canny(frame[:,:,0],100,200))
    return float(shift.mean())

def distortion_type(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    h, w = gray.shape
    center_y, center_x = h // 2, w // 2
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=50, minLineLength=30, maxLineGap=10)
    
    if lines is not None and len(lines) > 0:
        curvatures = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dist1 = np.sqrt((x1 - center_x)**2 + (y1 - center_y)**2)
            dist2 = np.sqrt((x2 - center_x)**2 + (y2 - center_y)**2)
            if dist1 > 0 and dist2 > 0:
                curvature = (dist1 + dist2) / 2.0
                curvatures.append(curvature)
        
        if curvatures:
            avg_curvature = np.mean(curvatures)
            if avg_curvature > np.sqrt(h**2 + w**2) * 0.4:
                return "barrel"
            elif avg_curvature < np.sqrt(h**2 + w**2) * 0.2:
                return "pincushion"
    
    return "none"

def lens_sharpness_drop_off(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    h, w = gray.shape
    center_y, center_x = h // 2, w // 2
    
    center_region = gray[center_y-h//4:center_y+h//4, center_x-w//4:center_x+w//4]
    center_sharpness = cv2.Laplacian(center_region, cv2.CV_64F).var()
    
    corners = [
        gray[0:h//4, 0:w//4],
        gray[0:h//4, 3*w//4:w],
        gray[3*h//4:h, 0:w//4],
        gray[3*h//4:h, 3*w//4:w]
    ]
    corner_sharpness = np.mean([cv2.Laplacian(c, cv2.CV_64F).var() for c in corners])
    
    if center_sharpness > 0:
        drop_off = 1.0 - (corner_sharpness / center_sharpness)
        return float(np.clip(drop_off, 0, 1))
    return 0.0

def lens_obstruction_probability(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    blurred = cv2.GaussianBlur(gray, (15, 15), 0)
    diff = np.abs(gray.astype(np.float32) - blurred.astype(np.float32))
    obstruction_score = np.mean(diff > 30) / 255.0
    return float(np.clip(obstruction_score, 0, 1))

def lens_dirt_probability(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    threshold = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY_INV, 11, 2)
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    small_dark_areas = sum(1 for c in contours if cv2.contourArea(c) < 100)
    dirt_prob = min(small_dark_areas / 50.0, 1.0)
    return float(dirt_prob)

def veiling_glare_score(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).ravel()
    hist = hist / (hist.sum() + 1e-10)
    bright_ratio = hist[200:].sum()
    contrast = gray.std()
    glare_score = bright_ratio * (1.0 - contrast / 128.0)
    return float(np.clip(glare_score, 0, 1))
```

#### 1.42. fog_haziness_score

```py
def fog_score(frame):
    lap = cv2.Laplacian(frame, cv2.CV_64F).var()
    return float(1.0 / (lap + 1))
```

#### 1.43. - 1.44. temporal_flicker_score, rolling_shutter_artifacts_score

```py
def temporal_flicker(prev_gray, gray):
    if prev_gray is None:
        return 0.0
    return float(np.mean(np.abs(prev_gray - gray)))

def rolling_shutter_artifacts_score(prev_frame, curr_frame):
    if prev_frame is None or curr_frame is None:
        return 0.0
    
    gray_prev = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY) if len(prev_frame.shape) == 3 else prev_frame
    gray_curr = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY) if len(curr_frame.shape) == 3 else curr_frame
    
    flow = cv2.calcOpticalFlowFarneback(gray_prev, gray_curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    
    h, w = flow.shape[:2]
    vertical_flow = flow[:, :, 1]
    
    num_strips = 5
    strip_width = w // num_strips
    strip_flows = []
    for i in range(num_strips):
        strip = vertical_flow[:, i*strip_width:(i+1)*strip_width]
        strip_flows.append(np.mean(np.abs(strip)))
    
    flow_variation = np.std(strip_flows) if len(strip_flows) > 1 else 0.0
    rolling_shutter_score = float(np.clip(flow_variation / 5.0, 0, 1))
    
    return rolling_shutter_score
```

#### 1.45. - 1.51. quality_cinematic_prob, quality_lowlight_cinematic_prob, quality_smartphone_good_prob, quality_smartphone_poor_prob, quality_webcam_prob, quality_screenrecord_prob, quality_surveillance_prob

```py
class ShotQualityZeroShot:
    def __init__(self, device="cuda"):
        self.device = device
        self.clip_model, self.preprocess = clip.load("ViT-L/14", device=device)
        
        self.prompts = [
            "cinematic shot, high-quality professional footage",
            "professional low-light cinematic footage",
            "good smartphone video quality",
            "poor smartphone video quality, grainy, noisy",
            "webcam low resolution footage",
            "screen recording of display",
            "cctv surveillance camera footage low quality"
        ]
        self.text = clip.tokenize(self.prompts).to(device)

    def predict(self, frame):
        pil = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        img_clip = self.preprocess(pil).unsqueeze(0).to(self.device)

        with torch.no_grad():
            img_feat = self.clip_model.encode_image(img_clip)
            txt_feat = self.clip_model.encode_text(self.text)

        img_feat /= img_feat.norm(dim=-1, keepdim=True)
        txt_feat /= txt_feat.norm(dim=-1, keepdim=True)

        logits = (img_feat @ txt_feat.T).softmax(dim=-1).cpu().numpy()[0]

        return {
            "quality_cinematic_prob": float(logits[0]),
            "quality_lowlight_cinematic_prob": float(logits[1]),
            "quality_smartphone_good_prob": float(logits[2]),
            "quality_smartphone_poor_prob": float(logits[3]),
            "quality_webcam_prob": float(logits[4]),
            "quality_screenrecord_prob": float(logits[5]),
            "quality_surveillance_prob": float(logits[6])
        }
```

#### 1.52. aesthetic_score

```py
class AestheticPredictor:
    def predict(self, pil_image):
        img_array = np.array(pil_image)
        brightness = np.mean(img_array) / 255.0
        contrast = np.std(img_array) / 255.0
        return float(0.5 + 0.3 * brightness + 0.2 * contrast)
```

#### 1.53. clip_embedding

```py
# CLIP embedding - вектор размерности 768 из модели CLIP ViT-L/14
# Получается через self.clip_model.encode_image(img_clip)
# Представляет семантическое представление кадра в пространстве CLIP
```

#### 2. frame_features

Агрегированные статистики по всем кадрам. Для каждой числовой фичи из `frames` вычисляются:
- `avg_{feature_name}` - среднее значение
- `std_{feature_name}` - стандартное отклонение
- `min_{feature_name}` - минимальное значение
- `max_{feature_name}` - максимальное значение

```py
def aggregate_frame_features(frame_results):
    frame_features = {}
    numeric_keys = set()
    
    for result in frame_results.values():
        for k, v in result.items():
            if isinstance(v, (int, float, np.number)) and not isinstance(v, bool):
                numeric_keys.add(k)
    
    for key in numeric_keys:
        values = [float(r[key]) for r in frame_results.values() if key in r]
        if not values:
            continue
        
        values_np = np.array(values, dtype=np.float32)
        frame_features[f"avg_{key}"] = float(values_np.mean())
        frame_features[f"std_{key}"] = float(values_np.std())
        frame_features[f"min_{key}"] = float(values_np.min())
        frame_features[f"max_{key}"] = float(values_np.max())
    
    return frame_features
```

#### 3. temporal_features

Временные метрики стабильности качества между кадрами:

```py
def stability(x):
    x = np.array(x, dtype=np.float32)
    return float(1.0 - np.std(x) / (np.mean(x) + 1e-8))

# temporal_sharpness_stability - стабильность резкости во времени
temporal_features["temporal_sharpness_stability"] = stability(sharpness_history)

# temporal_exposure_stability - стабильность экспозиции во времени
temporal_features["temporal_exposure_stability"] = stability(exposure_history)

# exposure_consistency_over_time - консистентность экспозиции
temporal_features["exposure_consistency_over_time"] = float(
    np.clip(1.0 - np.std(exposure_history), 0.0, 1.0)
)

# temporal_noise_variation - вариация уровня шума во времени
temporal_features["temporal_noise_variation"] = float(
    np.std(noise_history) / (np.mean(noise_history) + 1e-8)
)
```

### Модели:

- **DnCNN** - модель для оценки уровня шума (grayscale)
- **CBDNet** - модель для оценки статистик шума (color)
- **CLIP ViT-L/14** - модель для классификации типа качества видео и генерации эмбеддингов
- **AestheticPredictor** - модель для оценки эстетической привлекательности кадра





## cut_detection

### Модели:

```py
# ResNet18/ResNet50 для deep feature extraction
model = models.resnet18(pretrained=True)
model = torch.nn.Sequential(*list(model.children())[:-1])

# CLIP ViT-B/32 для zero-shot классификации переходов
clip_model, preprocess = clip.load("ViT-B/32", device=device)

# MediaPipe для детекции лиц и позы (для jump cuts)
face_detector = mp_face.FaceMesh(static_image_mode=True)
pose_detector = mp_pose.Pose(static_image_mode=True)
```

### Выход:

```json
{
    "features": {
        "hard_cuts_count": 0,
        "hard_cut_strength_mean": 0.0,
        "hard_cuts_per_minute": 0.0,
        "fade_in_count": 1,
        "fade_out_count": 0,
        "dissolve_count": 0,
        "avg_fade_duration": 0.4,
        "motion_cuts_count": 1,
        "motion_cut_intensity_score": 8.339471817016602,
        "flow_spike_ratio": 999999999.9999999,
        "whip_pan_transitions_count": 0,
        "zoom_transition_count": 1,
        "speed_ramp_cuts_count": 0,
        "transition_glitch_transition_count": 9,
        "jump_cuts_count": 0,
        "jump_cut_intensity": 0.0,
        "jump_cut_ratio_per_minute": 0.0,
        "cuts_per_minute": 0.0,
        "median_cut_interval": null,
        "min_cut_interval": null,
        "max_cut_interval": null,
        "cut_interval_std": null,
        "cut_interval_entropy": null,
        "cut_rhythm_uniformity_score": null,
        "avg_shot_length": 0.44,
        "median_shot_length": 0.44,
        "short_shots_ratio": 1.0,
        "long_shots_ratio": 0.0,
        "very_long_shots_count": 0,
        "extremely_short_shots_count": 0,
        "scene_count": 1,
        "avg_scene_length_shots": 1.0,
        "scene_to_shot_ratio": 0.9999999989999999,
        "scene_hard_cut_transitions": 0,
        "scene_fade_transitions": 0,
        "scene_dissolve_transitions": 0,
        "scene_motion_transitions": 0,
        "scene_stylized_transitions": 0,
        "audio_cut_alignment_score": 0.0,
        "audio_spike_cut_ratio": 0.10432931156841424,
        "scene_whoosh_transition_prob": 0.3488177266452295,
        "edit_style_hard_cut_prob": 0.09905259874131946,
        "edit_style_fade_prob": 0.10022854275173612,
        "edit_style_dissolve_prob": 0.09956868489583333,
        "edit_style_whip_pan_prob": 0.09845479329427081,
        "edit_style_zoom_transition_prob": 0.10187513563368056,
        "edit_style_wipe_transition_prob": 0.09960089789496526,
        "edit_style_slide_transition_prob": 0.0986663818359375,
        "edit_style_glitch_transition_prob": 0.10257297092013888,
        "edit_style_flash_transition_prob": 0.10074903700086806,
        "edit_style_luma_wipe_transition_prob": 0.09923027886284723,
        "edit_style_fast_prob": 0.0,
        "edit_style_slow_prob": 0.0,
        "edit_style_cinematic_prob": 0.0,
        "edit_style_meme_prob": 0.0,
        "edit_style_social_prob": 0.0,
        "edit_style_high_action_prob": 0.0
    },
    "detections": {
        "hard_cut_indices": [],
        "hard_cut_strengths": [],
        "soft_events": [{"type": "fade_in", "start": 0, "end": 10, "duration_s": 0.4}],
        "motion_cut_indices": [9],
        "motion_cut_intensities": [8.339471817016602],
        "motion_cut_types": ["zoom"],
        "stylized_counts": {"glitch transition": 9},
        "jump_cut_indices": [],
        "jump_cut_scores": [],
        "shot_boundaries_frames": [0, 11],
        "scene_boundaries_shot_idx": [[0, 0]]
    }
}
```

### Фичи:

#### 1. Hard Cuts (Жесткие склейки)

#### 1.1. - 1.4. hard_cuts_count, hard_cut_strength_mean, hard_cut_strength_p25/p50/p75, hard_cuts_per_minute

```py
def detect_hard_cuts(frame_manager, frame_indices, hist_thresh=None, ssim_thresh=None, 
                     flow_thresh=None, use_deep_features=True, use_adaptive_thresholds=True,
                     temporal_smoothing=True, embed_model=None, transform=None, device='cpu'):
    # Вычисляем различия между соседними кадрами
    hdiffs = []  # Различия гистограмм HSV
    ssim_diffs = []  # Различия SSIM
    flow_mags = []  # Величины оптического потока
    deep_diffs = []  # Различия deep embeddings
    
    for i in range(1, n):
        fA = frame_manager.get(frame_indices[i-1])
        fB = frame_manager.get(frame_indices[i])
        hdiff = frame_histogram_diff(fA, fB)  # L1 норма разности гистограмм HSV
        s = frame_ssim(fA, fB)  # 1 - SSIM
        flow_mag, _, _ = optical_flow_magnitude(prev_gray, gray)
        deep_diff = feature_embedding_diff(fA, fB, embed_model, transform, device)
        
        hdiffs.append(hdiff)
        ssim_diffs.append(s)
        flow_mags.append(flow_mag)
        deep_diffs.append(deep_diff)
    
    # Адаптивные пороги на основе локальной статистики
    if use_adaptive_thresholds:
        hist_thresh = np.median(hdiffs) + 2.0 * np.std(hdiffs)
        ssim_thresh = np.median(ssim_diffs) + 1.5 * np.std(ssim_diffs)
        flow_thresh = np.median(flow_mags) + 2.0 * np.std(flow_mags)
        deep_thresh = np.median(deep_diffs) + 1.5 * np.std(deep_diffs)
    
    # Комбинируем сигналы: каждый сигнал добавляет 1 к счету
    scores = np.zeros(len(hdiffs))
    scores += (hdiffs > hist_thresh).astype(float)
    scores += (ssim_diffs > ssim_thresh).astype(float)
    scores += (flow_mags > flow_thresh).astype(float)
    if use_deep_features:
        scores += (deep_diffs > deep_thresh).astype(float)
    
    # Временное сглаживание для уменьшения ложных срабатываний
    if temporal_smoothing:
        # Медианная фильтрация для робастности
        scores_median = medfilt(scores.astype(float), kernel_size=3)
        scores_smooth = gaussian_filter1d(scores_median, sigma=1.0)
        cut_candidates = [(i+1, scores_smooth[i]) for i in range(1, len(scores_smooth)-1)
                          if scores_smooth[i] > scores_smooth[i-1] and 
                             scores_smooth[i] > scores_smooth[i+1] and
                             scores_smooth[i] >= 2.0]
    
    # Морфологическая очистка: удаление изолированных детекций
    cut_flag_array = morphological_clean_cuts(cut_flags, min_neighbors=0)
    
    return cut_idxs, strengths
```

#### 2. Soft Cuts (Мягкие переходы)

#### 2.1. - 2.4. fade_in_count, fade_out_count, dissolve_count, avg_fade_duration

```py
def detect_soft_cuts(frame_manager, frame_indices, fps, fade_threshold=0.02, 
                     min_duration_frames=4, use_flow_consistency=True):
    # Анализ градиентов яркости и гистограмм
    hsv_values = []  # Средняя яркость в HSV
    lab_values = []  # Средняя яркость в LAB
    hist_diffs = []  # Различия гистограмм HSV
    flow_mags = []  # Величины оптического потока
    
    for i, idx in enumerate(frame_indices):
        f = frame_manager.get(idx)
        hsv = cv2.cvtColor(f, cv2.COLOR_BGR2HSV)
        v = hsv[:,:,2].mean() / 255.0
        hsv_values.append(v)
        
        lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)
        l = lab[:,:,0].mean() / 255.0
        lab_values.append(l)
        
        # Градиент гистограммы
        hsv_hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 8, 8], 
                                [0, 180, 0, 256, 0, 256]).flatten()
        hsv_hist = hsv_hist / (hsv_hist.sum() + 1e-9)
        if prev_hsv_hist is not None:
            hist_diff = float(np.linalg.norm(hsv_hist - prev_hsv_hist, ord=1))
            hist_diffs.append(hist_diff)
    
    # Детекция fade-in: монотонное увеличение яркости
    # Детекция fade-out: монотонное уменьшение яркости
        # Детекция dissolve: плавное изменение гистограммы без резких скачков
        # Улучшенная детекция: исключение экспозиционных изменений через проверку
        # корреляции гистограмм и анализа градиентов яркости
    
    return events  # [{'type': 'fade_in', 'start': 0, 'end': 10, 'duration_s': 0.4}, ...]
```

#### 3. Motion-based Cuts (Движение-основанные переходы)

#### 3.1. - 3.5. motion_cuts_count, motion_cut_intensity_score, flow_spike_ratio, whip_pan_transitions_count, zoom_transition_count, speed_ramp_cuts_count

```py
def detect_motion_based_cuts(frame_manager, frame_indices, flow_spike_factor=None,
                              use_direction_analysis=True, adaptive_threshold=True,
                              detect_speed_ramps=True):
    mags = []  # Величины оптического потока
    angles_list = []  # Углы направления потока
    direction_consistencies = []  # Консистентность направления
    mag_variances = []  # Дисперсия величины потока (для speed ramp)
    
    # Компенсация движения камеры через RANSAC homography
    for i in range(1, n):
        gray = cv2.cvtColor(frame_manager.get(frame_indices[i]), cv2.COLOR_BGR2GRAY)
        
        # Оценка глобального движения камеры
        if use_camera_motion_compensation:
            H, inlier_ratio = estimate_global_motion_homography(prev_gray, gray)
            is_camera_motion = inlier_ratio > 0.7 if H is not None else False
        
        # Оптический поток (на уменьшенном разрешении для больших кадров)
        if use_low_res:
            prev_gray_small = cv2.resize(prev_gray, (256, 256))
            gray_small = cv2.resize(gray, (256, 256))
            mag, mag_map, angles = optical_flow_magnitude(prev_gray_small, gray_small)
            mag = mag * scale_factor  # Масштабирование обратно
        else:
            mag, mag_map, angles = optical_flow_magnitude(prev_gray, gray)
        
        mags.append((mag, is_camera_motion))
        
        # Консистентность направления через круговую статистику
        if angles is not None:
            consistency = optical_flow_direction_consistency(angles)
            direction_consistencies.append(consistency)
        
        # Дисперсия величины потока для детекции speed ramp
        if detect_speed_ramps and mag_map is not None:
            mag_variance = float(np.var(mag_map))
            mag_variances.append(mag_variance)
    
    # Адаптивный порог на основе перцентилей
    mags_array = np.array([m[0] for m in mags])
    threshold = np.percentile(mags_array, 95) if adaptive_threshold else median + 3.0 * std
    
    # Классификация типов переходов:
    # - whip_pan: высокая консистентность направления + высокая величина
    # - zoom: низкая консистентность + высокая величина
    # - speed_ramp: высокая величина + высокая дисперсия
    
    return spike_idxs, intensities, types
```

#### 4. Stylized Transitions (Стилизованные переходы)

#### 4.1. transition_*_count (glitch, flash, wipe, slide, luma_wipe и т.д.)

```py
class StylizedTransitionZeroShot:
    def predict_transition(self, frames_window):
        # Создание мультимодального входа: оригинал | разность | оптический поток
        if self.use_multimodal:
            combined = self._create_multimodal_input(frames_window)
            # Конкатенация: оригинал | diff | flow визуализация
        
        # CLIP zero-shot классификация
        # Оптимизация: candidate-first подход - CLIP запускается только на
        # кандидатных окнах, отфильтрованных через легковесные сигналы
        # (изменение гистограмм, SSIM drop, flow spike)
        image_input = self.preprocess(pil).unsqueeze(0).to(self.device)
        img_feat = self.model.encode_image(image_input)
        txt_feat = self.model.encode_text(self.text_tokens)
        
        logits = (img_feat @ txt_feat.T).softmax(dim=-1)
        return {label: prob for label, prob in zip(self.labels, logits)}
```

#### 5. Jump Cuts (Прыжковые склейки)

#### 5.1. - 5.3. jump_cuts_count, jump_cut_intensity, jump_cut_ratio_per_minute

```py
def detect_jump_cuts(frame_manager, frame_indices, use_background_embedding=True,
                     use_pose_estimation=True, embed_model=None, transform=None, device='cuda'):
    # MediaPipe для детекции лиц и позы
    face_detector = mp_face.FaceMesh(static_image_mode=True)
    pose_detector = mp_pose.Pose(static_image_mode=True)
    
    for i, idx in enumerate(frame_indices):
        f = frame_manager.get(idx)
        img_rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
        
        # Детекция лица и позы
        face_res = face_detector.process(img_rgb)
        pose_res = pose_detector.process(img_rgb)
        
        # Face ID embedding для более точного определения того же человека
        if embed_model is not None and face_bbox is not None:
            face_roi = extract_face_region(f, face_bbox)
            face_emb = embed_model(transform(ImageFromCV(face_roi)))
        
        # Embedding фона через ResNet (с маскированием области лица для чистоты сравнения)
        if use_background_embedding:
            bg_frame = mask_foreground(f, face_bbox)  # Исключаем лицо/позу
            bg_emb = embed_model(transform(ImageFromCV(bg_frame)))
        
        # Проверка jump cut: большое изменение позы/лица + похожий фон
        if prev_landmarks is not None:
            # Используем face ID embedding если доступен
            if face_emb is not None and prev_face_emb is not None:
                face_change = 1.0 - cosine_similarity(face_emb, prev_face_emb)
            else:
                face_change = 1.0 - cosine_similarity(face_landmarks, prev_landmarks)
            pose_change = 1.0 - cosine_similarity(pose_landmarks, prev_pose_landmarks)
            background_similar = cosine_similarity(bg_emb, prev_bg_emb) > 0.85
            
            # Адаптивный порог на основе уверенности
            threshold = 0.3 / confidence if confidence > 0 else 0.3
            if (face_change + pose_change) / 2.0 > threshold and background_similar:
                jump_idxs.append(i)
                jump_scores.append((face_change + pose_change) / 2.0 * bg_sim)
    
    return jump_idxs, jump_scores
```

#### 6. Cut Timing Statistics (Статистика времени склеек)

#### 6.1. - 6.8. cuts_per_minute, median_cut_interval, min_cut_interval, max_cut_interval, cut_interval_std, cut_interval_cv, cut_interval_entropy, cut_rhythm_uniformity_score

```py
def cut_timing_statistics(cut_frame_indices, fps, video_length_s):
    times = np.array(cut_frame_indices, dtype=np.float32) / float(fps)
    intervals = np.diff(times)  # Интервалы между склейками
    
    cpm = len(cut_frame_indices) / video_length_s * 60.0  # Только per_minute (cuts_per_second удален)
    median = float(np.median(intervals))
    mn = float(np.min(intervals))
    mx = float(np.max(intervals))
    std = float(np.std(intervals))
    mean_int = float(np.mean(intervals))
    
    # Коэффициент вариации (CV)
    cv = std / (mean_int + 1e-9)
    
    # Нормализованная энтропия интервалов через гистограмму
    n_bins = min(20, len(intervals))
    hist, _ = np.histogram(intervals, bins=n_bins)
    hist = hist + 1e-9
    hist = hist / hist.sum()  # Нормализация к вероятностям
    ent = float(scipy.stats.entropy(hist))
    # Нормализация энтропии делением на log(n_bins) для диапазона [0, 1]
    max_entropy = np.log(n_bins) if n_bins > 1 else 1.0
    ent_normalized = ent / (max_entropy + 1e-9)
    
    # Равномерность ритма: 1 - CV (coefficient of variation)
    cv_clipped = np.clip(cv, 0.0, 1.0)
    uniformity = float(1.0 - cv_clipped)
    
    return {
        'cuts_per_minute': float(cpm),
        'median_cut_interval': median,
        'min_cut_interval': mn,
        'max_cut_interval': mx,
        'cut_interval_std': std,
        'cut_interval_cv': float(cv),
        'cut_interval_entropy': float(ent_normalized),
        'cut_rhythm_uniformity_score': uniformity
    }
```

#### 7. Shot Length Statistics (Статистика длины кадров)

#### 7.1. - 7.11. avg_shot_length, median_shot_length, shot_length_p10/p25/p75/p90, short_shots_ratio, long_shots_ratio, very_long_shots_count, extremely_short_shots_count, shot_length_histogram

```py
def shot_length_stats(shot_frame_lengths, fps):
    durations_s = np.array([seconds_from_fps(l, fps) for l in shot_frame_lengths])
    
    avg = float(durations_s.mean())
    med = float(np.median(durations_s))
    
    # Перцентили распределения
    percentiles = np.percentile(durations_s, [10, 25, 75, 90])
    
    short_ratio = float((durations_s < 1.0).sum() / durations_s.size)
    long_ratio = float((durations_s > 4.0).sum() / durations_s.size)
    very_long = int((durations_s > 10.0).sum())
    extremely_short = int((durations_s < 0.25).sum())
    
    # Гистограмма (8 бинов) для компактного представления распределения
    hist, bin_edges = np.histogram(durations_s, bins=8)
    hist_normalized = hist / (hist.sum() + 1e-9)
    
    return {
        'avg_shot_length': avg,
        'median_shot_length': med,
        'short_shots_ratio': short_ratio,
        'long_shots_ratio': long_ratio,
        'very_long_shots_count': very_long,
        'extremely_short_shots_count': extremely_short
    }
```

#### 8. Scene Detection (Детекция сцен)

#### 8.1. - 8.6. scene_count, avg_scene_length_shots, scene_to_shot_ratio, scene_hard_cut_transitions, scene_fade_transitions, scene_dissolve_transitions, scene_motion_transitions, scene_stylized_transitions

```py
def scene_boundaries_from_shots(shot_cut_indices, shots_duration_frames, fps,
                                min_scene_shots=2, use_semantic_clustering=True,
                                frame_embeddings=None, audio_events=None):
    # Семантическая кластеризация через DBSCAN на embeddings кадров
    if use_semantic_clustering and frame_embeddings is not None:
        shot_embeddings = []
        for i, (start_idx, duration) in enumerate(zip([0] + shot_cut_indices, shots_duration_frames)):
            mid_frame_idx = start_idx + duration // 2
            shot_embeddings.append(frame_embeddings[mid_frame_idx])
        
        scaler = StandardScaler()
        shot_embeddings_scaled = scaler.fit_transform(shot_embeddings)
        
        # DBSCAN кластеризация для группировки похожих кадров в сцены
        clustering = DBSCAN(eps=0.5, min_samples=min_scene_shots).fit(shot_embeddings_scaled)
        labels = clustering.labels_
        
        # Группировка кадров по кластерам
        scenes = group_shots_by_cluster(labels)
    
    # Альтернатива: аудио-визуальное слияние
    # Использование аудио событий (onsets, паузы) для определения границ сцен
    
    return scenes  # [(start_shot_idx, end_shot_idx), ...]
```

#### 9. Audio Features (Аудио фичи)

#### 9.1. - 9.3. audio_cut_alignment_score, audio_spike_cut_ratio, scene_whoosh_transition_prob

```py
def audio_onset_strength(audio_path, sr=22050, hop_length=512, use_multiband=True):
    y, sr = librosa.load(audio_path, sr=sr)
    
    # Многополосный анализ: разделение на низкие и высокие частоты
    if use_multiband:
        y_low = librosa.effects.preemphasis(y, coef=0.97)
        y_high = y - y_low
        onset_env_low = librosa.onset.onset_strength(y=y_low, sr=sr, hop_length=hop_length)
        onset_env_high = librosa.onset.onset_strength(y=y_high, sr=sr, hop_length=hop_length)
        onset_env = 0.6 * onset_env_low + 0.4 * onset_env_high
    else:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop_length)
    
    times = librosa.times_like(onset_env, sr=sr, hop_length=hop_length)
    rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
    
    return onset_env, times, rms, loudness

def audio_cut_alignment_score(cut_times_seconds, onset_env, onset_times, window=0.5,
                               use_dynamic_threshold=True, rms=None, use_clustering=True):
    # Динамический порог на основе RMS
    if use_dynamic_threshold and rms is not None:
        rms_normalized = (rms - rms.min()) / (rms.max() - rms.min() + 1e-9)
        threshold = np.mean(onset_env) + np.std(onset_env) * (1.0 + rms_normalized)
    else:
        threshold = np.mean(onset_env) + np.std(onset_env)
    
    # Кластеризация onset пиков для стабильного сопоставления
    if use_clustering:
        onset_clusters = cluster_onsets(onset_env, onset_times, window=window)
        aligned = sum(1 for ct in cut_times_seconds 
                     if any(np.abs(cluster_time - ct) <= window for cluster_time in onset_clusters))
    else:
        aligned = sum(1 for ct in cut_times_seconds
                     if np.any((np.abs(onset_times - ct) <= window) & (onset_env > threshold)))
    
    return float(aligned / len(cut_times_seconds))

def detect_scene_whoosh_transitions(audio_path, scene_boundaries_times, sr=22050):
    # STFT анализ
    stft = librosa.stft(y, n_fft=2048, hop_length=512)
    magnitude = np.abs(stft)
    
    # Спектральные фичи
    spectral_centroid = librosa.feature.spectral_centroid(S=magnitude, sr=sr)[0]
    spectral_rolloff = librosa.feature.spectral_rolloff(S=magnitude, sr=sr, roll_percent=0.85)[0]
    spectral_flux = np.sum(np.diff(magnitude, axis=1) ** 2, axis=0)
    
    # Высокочастотная энергия (>5000 Hz)
    freqs = librosa.fft_frequencies(sr=sr, n_fft=2048)
    hf_mask = freqs > 5000
    high_freq_energy = magnitude[hf_mask].sum(axis=0)
    
    # Оценка whoosh: растущий rolloff + высокий spectral flux + HF энергия
    for scene_time in scene_boundaries_times:
        mask = (times >= scene_time - 0.5) & (times <= scene_time + 0.5)
        roll_diff = np.diff(spectral_rolloff[mask])
        roll_score = np.mean(roll_diff[roll_diff > 0]) if np.any(roll_diff > 0) else 0.0
        flux_score = np.mean(spectral_flux[mask] > np.percentile(spectral_flux, 85))
        hf_score = np.mean(high_freq_energy[mask] > np.percentile(high_freq_energy, 85))
        
        prob = 0.35 * np.tanh(roll_score) + 0.35 * flux_score + 0.30 * hf_score
        whoosh_probs.append(float(np.clip(prob, 0.0, 1.0)))
    
    return whoosh_probs
```

#### 10. Edit Style Classification (Классификация стиля монтажа)

#### 10.1. - 10.6. edit_style_fast_prob, edit_style_slow_prob, edit_style_cinematic_prob, edit_style_meme_prob, edit_style_social_prob, edit_style_high_action_prob

```py
def classify_edit_style(cut_timing_stats, shot_stats, motion_cuts_count, jump_cuts_count,
                        stylized_counts, hard_cuts_count, duration_s):
    cpm = cut_timing_stats.get('cuts_per_minute', 0.0)
    median_interval = cut_timing_stats.get('median_cut_interval', 0.0)
    cut_std = cut_timing_stats.get('cut_interval_std', 0.0)
    uniformity = cut_timing_stats.get('cut_rhythm_uniformity_score', 0.0)
    avg_shot_length = shot_stats.get('avg_shot_length', 0.0)
    short_shots_ratio = shot_stats.get('short_shots_ratio', 0.0)
    
    # Fast-cut montage: очень высокая частота склеек, короткие кадры, высокая вариация
    if cpm > 20 and avg_shot_length < 2.0 and cut_std > 0.5:
        styles['fast'] = min(1.0, (cpm / 60.0) * 0.5 + (1.0 - avg_shot_length / 3.0) * 0.3 + cut_std * 0.2)
    
    # Slow-paced: низкая частота, длинные кадры, равномерный ритм
    if cpm < 8 and avg_shot_length > 5.0 and uniformity > 0.7:
        styles['slow'] = min(1.0, (1.0 - cpm / 15.0) * 0.4 + (avg_shot_length / 10.0) * 0.3 + uniformity * 0.3)
    
    # Social media: много jump cuts, средняя-высокая частота, короткие кадры
    if jump_cut_ratio > 3.0 and cpm > 15 and short_shots_ratio > 0.3:
        styles['social'] = min(1.0, (jump_cut_ratio / 10.0) * 0.4 + (cpm / 40.0) * 0.3 + short_shots_ratio * 0.3)
    
    # Cinematic: средняя частота, длинные кадры, стилизованные переходы
    if 5 < cpm < 15 and avg_shot_length > 3.0 and stylized_count_total > total_cuts * 0.2:
        styles['cinematic'] = min(1.0, (avg_shot_length / 6.0) * 0.4 + (stylized_count_total / max(total_cuts, 1)) * 0.4)
    
    # Meme-style: очень короткие кадры, высокая частота, много jump cuts, хаотичный ритм
    if extremely_short_count > 5 and cpm > 25 and jump_cut_ratio > 2.0 and uniformity < 0.5:
        styles['meme'] = min(1.0, (extremely_short_count / 20.0) * 0.3 + (cpm / 50.0) * 0.3)
    
    # High-action: много motion transitions, высокая частота, короткие-средние кадры
    if motion_transition_ratio > 0.3 and cpm > 18 and 1.0 < avg_shot_length < 4.0:
        styles['high_action'] = min(1.0, motion_transition_ratio * 0.4 + (cpm / 40.0) * 0.3)
    
    # Нормализация вероятностей
    total = sum(styles.values()) + 1e-9
    for key in styles:
        styles[key] = styles[key] / total
    
    return styles
```

#### 11. edit_style_*_prob (CLIP-based transition probabilities)

```py
# Вероятности типов переходов через CLIP zero-shot классификацию
# Вычисляются как средние вероятности по всем переходам в видео
# edit_style_hard_cut_prob, edit_style_fade_prob, edit_style_dissolve_prob,
# edit_style_whip_pan_prob, edit_style_zoom_transition_prob, edit_style_wipe_transition_prob,
# edit_style_slide_transition_prob, edit_style_glitch_transition_prob,
# edit_style_flash_transition_prob, edit_style_luma_wipe_transition_prob

if self.clip_detector is not None and stylized_probs_per_cut:
    avg_probs = {lbl: 0.0 for lbl in labels}
    for p in stylized_probs_per_cut:
        for lbl, val in p.items():
            avg_probs[lbl] += val
    for lbl in labels:
        avg_probs[lbl] /= count
        features[f"edit_style_{lbl.replace(' ','_').lower()}_prob"] = float(avg_probs[lbl])
```

## video_pacing

### Модели:

```py
# CLIP ViT-B/32 для извлечения семантических embeddings кадров
clip_model, _ = clip.load("ViT-B/32", device=device)
clip_model.eval()

# SSIM для детекции склеек между кадрами
from skimage.metrics import structural_similarity as ssim

# Оптический поток Farneback для анализа движения
flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, pyr_scale=0.5, levels=3, 
                                     winsize=15, iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
```

### Выход:

```json
{
    "shot_duration_mean": 0.85,
    "shot_duration_median": 0.7,
    "shot_duration_min": 0.12,
    "shot_duration_max": 3.5,
    "shot_duration_std": 0.6,
    "shot_duration_entropy": 1.2,
    "shot_duration_mean_normalized": 0.18,
    "shot_length_gini": 0.42,
    "cuts_per_10s": 3.2,
    "cuts_per_10s_max": 7.0,
    "cuts_per_10s_median": 2.5,
    "cuts_variance": 0.36,
    "short_shot_fraction": 0.4,
    "quick_cut_burst_count": 2,
    "shot_length_histogram_5bins": [0.25, 0.35, 0.25, 0.1, 0.05],
    "tempo_entropy": 1.35,
    "cut_density_map_8bins": [0.5, 0.8, 0.3, 0.2, 0.1, 0.4, 0.6, 0.7],
    "pace_curve_mean": 0.85,
    "pace_curve_slope": -0.02,
    "pace_curve_slope_normalized": -0.017,
    "pace_curve_peaks": 3,
    "pace_curve_peaks_mean_prominence": 0.4,
    "pace_curve_peak_positions": [0.15, 0.5, 0.85],
    "pace_curve_dominant_period_sec": 2.4,
    "pace_curve_power_at_period": 0.62,
    "scene_changes_per_minute": 14.0,
    "average_scene_duration": 0.85,
    "scene_duration_variance": 0.36,
    "mean_motion_speed_per_shot": 4.36,
    "motion_speed_median": 3.45,
    "motion_speed_variance": 12.48,
    "motion_speed_90perc": 8.83,
    "share_of_high_motion_frames": 0.3,
    "share_of_high_motion_shots": 0.4,
    "motion_shot_corr": -0.35,
    "optical_flow_direction_changes_per_second": 2.1,
    "frame_embedding_diff_mean": 0.3,
    "frame_embedding_diff_std": 0.08,
    "high_change_frames_ratio": 0.3,
    "scene_embedding_jumps": 10,
    "semantic_change_burst_count": 2,
    "color_histogram_diff_mean": 14.64,
    "color_histogram_diff_std": 14.47,
    "saturation_change_rate": 18.82,
    "brightness_change_rate": 16.6,
    "color_change_bursts": 3,
    "luminance_spikes_per_minute": 1.2,
    "high_frequency_flash_ratio": 0.28,
    "intro_speed": 0.9,
    "main_speed": 0.8,
    "climax_speed": 0.7,
    "pacing_symmetry": -0.25
}
```

### Фичи:

#### 1. Shot Features (Характеристики кадров)

#### 1.1. - ... shot_duration_mean, shot_duration_median, shot_duration_min, shot_duration_max, shot_duration_std, shot_duration_entropy, shot_duration_mean_normalized, shot_length_gini, cuts_per_10s, cuts_per_10s_max, cuts_per_10s_median, cuts_variance, short_shot_fraction, quick_cut_burst_count, shot_length_histogram_5bins, tempo_entropy, cut_density_map_8bins

```py
def _detect_shots(self) -> List[int]:
    shot_indices = [0]
    prev_frame = self._get_resize_frame(0)
    for idx in self.frame_indices:
        curr_frame = self._get_resize_frame(idx)
        score = self._safe_ssim(prev_frame, curr_frame)  # SSIM между кадрами
        if score < 0.95:  # threshold для hard cut
            shot_indices.append(idx)
            prev_frame = curr_frame
    return shot_indices

def extract_shot_features(self) -> Dict:
    durations = np.diff([0] + self.shot_boundaries + [self.total_frames])
    return {
        "shot_duration_mean": float(np.mean(durations)),
        "shot_duration_median": float(np.median(durations)),
        "shot_duration_min": float(np.min(durations)),
        "shot_duration_max": float(np.max(durations)),
        "shot_duration_std": float(np.std(durations)),
        "shot_duration_entropy": float(entropy(np.histogram(durations, bins=20)[0])),
        "cuts_per_10s": float(len(self.shot_boundaries) / (self.total_frames / self.fps / 10)),
        "cuts_variance": float(np.var(durations)),
        "longest_shot_duration": float(np.max(durations)),
        "shortest_shot_duration": float(np.min(durations))
    }
```

#### 2. Pace Curve (Кривая темпа)

#### 2.1. - ... pace_curve_mean, pace_curve_slope, pace_curve_slope_normalized, pace_curve_peaks, pace_curve_peaks_mean_prominence, pace_curve_peak_positions, pace_curve_dominant_period_sec, pace_curve_power_at_period

```py
def extract_pace_curve(self) -> Dict:
    durations = np.diff([0] + self.shot_boundaries + [self.total_frames])
    # Линейная регрессия для определения наклона кривой темпа
    curve_slope = np.polyfit(np.arange(len(durations)), durations, 1)[0]
    # Подсчет локальных максимумов (пиков)
    peaks = ((durations[1:-1] > durations[:-2]) & (durations[1:-1] > durations[2:])).sum()
    # Автокорреляция для определения периодичности
    autocorr = np.correlate(durations - np.mean(durations), durations - np.mean(durations), mode="full")
    autocorr /= autocorr.max()
    period = np.argmax(autocorr[len(autocorr)//2+1:]) + 1
    return {
        "pace_curve_mean": float(np.mean(durations)),
        "pace_curve_slope": float(curve_slope),
        "pace_curve_peaks": int(peaks),
        "pace_curve_periodicity": int(period)
    }
```

#### 3. Scene Pacing (Темп сцен)

#### 3.1. - 3.3. scene_changes_per_minute, average_scene_duration, scene_duration_variance

```py
def extract_scene_pacing(self) -> Dict:
    durations = np.diff([0] + self.scene_boundaries + [self.total_frames])
    return {
        "scene_changes_per_minute": float(len(self.scene_boundaries) / ((self.total_frames/self.fps)/60)),
        "average_scene_duration": float(np.mean(durations)),
        "scene_duration_variance": float(np.var(durations))
    }
```

#### 4. Motion Features (Характеристики движения)

#### 4.1. - 4.6. mean_motion_speed_per_shot, motion_speed_median, motion_speed_variance, motion_speed_90perc, share_of_high_motion_frames, optical_flow_direction_changes_per_second

```py
def extract_motion_features(self) -> Dict:
    flow_mags = []  # Величины оптического потока
    dir_changes = []  # Изменения направления потока
    prev_gray = cv2.cvtColor(self._get_resize_frame(0), cv2.COLOR_RGB2GRAY)
    for idx in self.frame_indices[1:]:
        frame = self._get_resize_frame(idx)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # Оптический поток Farneback
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                            pyr_scale=0.5, levels=3, winsize=15,
                                            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
        mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
        flow_mags.append(np.mean(mag))
        dir_changes.append(np.std(ang))  # Стандартное отклонение углов = вариация направления
        prev_gray = gray
    flow_mags = np.array(flow_mags)
    dir_changes = np.array(dir_changes)
    return {
        "mean_motion_speed_per_shot": float(np.mean(flow_mags)),
        "motion_speed_median": float(np.median(flow_mags)),
        "motion_speed_variance": float(np.var(flow_mags)),
        "motion_speed_90perc": float(np.percentile(flow_mags, 90)),
        "share_of_high_motion_frames": float(np.mean(flow_mags > np.percentile(flow_mags, 75))),
        "optical_flow_direction_changes_per_second": float(np.mean(dir_changes)*self.fps)
    }
```

#### 5. Content Change Rate (Скорость изменения контента)

#### 5.1. - 5.4. frame_embedding_diff_mean, frame_embedding_diff_std, high_change_frames_ratio, scene_embedding_jumps

```py
def extract_content_change_rate(self) -> Dict:
    embeddings = []
    # Батчовая обработка кадров через CLIP
    for i in range(0, self.total_frames, self.batch_size):
        batch_frames = [self._get_clip_frame(idx) for idx in self.frame_indices[i:i+self.batch_size]]
        batch_tensor = torch.tensor(np.stack(batch_frames)/255.0).permute(0,3,1,2).float().to(device)
        with torch.no_grad():
            emb = self.clip_model.encode_image(batch_tensor.half() if device=="cuda" else batch_tensor)
            embeddings.append(emb.cpu().numpy())
    embeddings = np.vstack(embeddings)
    # L2 норма разности embeddings между соседними кадрами
    diff = np.linalg.norm(np.diff(embeddings, axis=0), axis=1)
    # Сглаживание через скользящее среднее (окно 5)
    diff_smooth = np.convolve(diff, np.ones(5)/5, mode='same')
    return {
        "frame_embedding_diff_mean": float(np.mean(diff_smooth)),
        "frame_embedding_diff_std": float(np.std(diff_smooth)),
        "high_change_frames_ratio": float(np.mean(diff_smooth > np.percentile(diff_smooth, 75))),
        "scene_embedding_jumps": int(np.sum(diff_smooth > 2*np.std(diff_smooth)))
    }
```

#### 6. Color Pacing (Темп изменения цвета)

#### 6.1. - 6.4. color_histogram_diff_mean, color_histogram_diff_std, saturation_change_rate, brightness_change_rate

```py
def extract_color_pacing(self) -> Dict:
    hist_diffs = []
    prev_frame = self._get_resize_frame(0)
    for idx in self.frame_indices[1:]:
        frame = self._get_resize_frame(idx)
        # DeltaE в LAB пространстве для perceptual цветового различия
        lab1 = rgb2lab(prev_frame)
        lab2 = rgb2lab(frame)
        deltaE = np.sqrt(np.sum((lab1-lab2)**2, axis=2))
        hist_diffs.append(np.mean(deltaE))
        prev_frame = frame
    hist_diffs = np.array(hist_diffs)
    # Насыщенность и яркость из HSV
    saturation = [np.mean(cv2.cvtColor(self._get_resize_frame(idx), cv2.COLOR_RGB2HSV)[:,:,1]) 
                  for idx in self.frame_indices]
    brightness = [np.mean(cv2.cvtColor(self._get_resize_frame(idx), cv2.COLOR_RGB2HSV)[:,:,2]) 
                  for idx in self.frame_indices]
    return {
        "color_histogram_diff_mean": float(np.mean(hist_diffs)),
        "color_histogram_diff_std": float(np.std(hist_diffs)),
        "saturation_change_rate": float(np.std(saturation)),
        "brightness_change_rate": float(np.std(brightness))
    }
```

#### 7. Lighting Pacing (Темп изменения освещения)

#### 7.1. - 7.2. luminance_spikes_per_minute, high_frequency_flash_ratio

```py
def extract_lighting_pacing(self) -> Dict:
    # Яркость из grayscale
    lum = [np.mean(cv2.cvtColor(self._get_resize_frame(idx), cv2.COLOR_RGB2GRAY)) 
           for idx in self.frame_indices]
    lum_diff = np.diff(lum)
    # FFT анализ для высокочастотных изменений (вспышки)
    lum_fft = np.fft.fft(lum_diff)
    # Доля высокочастотной энергии (последняя четверть спектра)
    hf_ratio = np.sum(np.abs(lum_fft[len(lum_fft)//4:len(lum_fft)//2])) / (np.sum(np.abs(lum_fft))+1e-9)
    return {
        "luminance_spikes_per_minute": float(np.sum(np.abs(lum_diff) > np.std(lum_diff)) / (len(self.frame_indices)/self.fps*60)),
        "high_frequency_flash_ratio": float(hf_ratio)
    }
```

#### 8. Structural Pacing (Структурный темп)

#### 8.1. - 8.4. intro_speed, main_speed, climax_speed, pacing_symmetry

```py
def extract_structural_pacing(self) -> Dict:
    durations = np.diff([0] + self.shot_boundaries + [self.total_frames])
    n = len(durations)
    quarter = max(n//4, 1)
    # Разделение на три части: intro (первая четверть), main (середина), climax (последняя четверть)
    return {
        "intro_speed": float(np.median(durations[:quarter])),  # Медианная длительность кадров в начале
        "main_speed": float(np.median(durations[quarter:3*quarter])),  # Медианная длительность в середине
        "climax_speed": float(np.median(durations[3*quarter:])),  # Медианная длительность в конце
        "pacing_symmetry": float(np.mean(np.diff(durations)))  # Среднее изменение длительности (симметрия ритма)
    }
```

## story_structure

### Модели:

```py
# CLIP ViT-B/32 для извлечения визуальных embeddings кадров
clip_model, clip_preprocess = clip.load("ViT-B/32", device=device)

# SentenceTransformer для извлечения текстовых embeddings (если есть субтитры)
from sentence_transformers import SentenceTransformer
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

# MediaPipe FaceMesh для детекции лиц и подсчета количества людей
mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)

# AgglomerativeClustering для сегментации истории и тем
from sklearn.cluster import AgglomerativeClustering
clustering = AgglomerativeClustering(n_clusters=n_segments, metric='cosine', linkage='average')

# Оптический поток Farneback для анализа движения
flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
```

### Выход:

```json
{
    "number_of_story_segments": 5,
    "avg_story_segment_duration": 2.2,
    "avg_story_segment_duration_normalized": 0.18,
    "abrupt_story_transition_count": 4,
    "narrative_continuity_score": 0.7054986040083917,
    "narrative_continuity_std": 0.0692671015474594,
    "hook_motion_intensity": 5.746213912963867,
    "hook_cut_rate": 0.2,
    "hook_motion_spikes": 1,
    "hook_rhythm_score": 1.5,
    "hook_face_presence": 0.0,
    "hook_visual_surprise_score": 0.12894346546282595,
    "hook_visual_surprise_std": 0.09942302256606807,
    "hook_brightness_spike": 1.1,
    "hook_saturation_spike": 0.9,
    "climax_timestamp": 0,
    "climax_position_normalized": 0.0,
    "climax_strength": 7.072592735290527,
    "climax_strength_normalized": 1.9,
    "number_of_peaks": 1,
    "climax_duration": 5,
    "time_from_hook_to_climax": 0.0,
    "hook_to_avg_energy_ratio": 1.2,
    "story_energy_curve": [
        7.072592735290527,
        6.375296592712402,
        5.405511856079102,
        4.016316890716553,
        3.457134485244751,
        2.384732961654663,
        1.9452251195907593,
        3.6477019786834717,
        3.280890464782715,
        2.912097454071045
    ],
    "story_energy_curve_downsampled_128": [...],
    "number_of_unique_identities": 0,
    "main_character_screen_time": 0.0,
    "speaker_switch_rate": 0.0,
    "face_presence_curve": [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "face_area_fraction_curve": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "has_subtitles": true,
    "number_of_topics": 3,
    "avg_topic_duration": 1.0,
    "avg_topic_duration_normalized": 0.33,
    "topic_shift_times": [0, 1],
    "topic_diversity": 1.0,
    "topic_diversity_normalized": 1.58,
    "semantic_coherence_score": 0.7,
    "topic_coherence_std": 0.1
}
```

### Фичи:

#### 1. Story Segmentation (Сегментация истории)

#### 1.1. - 1.4. number_of_story_segments, avg_story_segment_duration, abrupt_story_transition_count, narrative_continuity_score, narrative_continuity_std

```py
def story_segmentation(self, n_segments=5):
    # Сглаживание embeddings через скользящее среднее
    smooth_emb = smooth_signal(self.clip_embeddings, window=3)
    # Иерархическая кластеризация на основе cosine similarity
    clustering = AgglomerativeClustering(n_clusters=n_segments, metric='cosine', linkage='average')
    labels = clustering.fit_predict(smooth_emb)
    
    # Длительности сегментов
    segment_durations = [np.sum(labels==i) for i in range(n_segments)]
    
    # Narrative continuity: средняя similarity между последовательными сегментами
    cont_scores = []
    for i in range(n_segments-1):
        idx1 = np.where(labels==i)[0]
        idx2 = np.where(labels==i+1)[0]
        sim = cosine_similarity(
            self.clip_embeddings[idx1].mean(axis=0).reshape(1,-1),
            self.clip_embeddings[idx2].mean(axis=0).reshape(1,-1)
        )[0][0]
        cont_scores.append(sim)
    
    return {
        "number_of_story_segments": n_segments,
        "avg_story_segment_duration": np.mean(segment_durations),
        "abrupt_story_transition_count": np.sum(np.diff(labels)!=0),  # Количество резких переходов
        "narrative_continuity_score": np.mean(cont_scores),
        "narrative_continuity_std": np.std(cont_scores)
    }
```

#### 2. Hook Features (Фичи начала видео)

#### 2.1. - 2.7. hook_motion_intensity, hook_cut_rate, hook_motion_spikes, hook_face_presence, hook_visual_surprise_score, hook_visual_surprise_std, hook_brightness_spike, hook_saturation_spike

```py
def hook_features(self, hook_seconds=5):
    n_frames = min(len(self.frame_indices), hook_seconds)
    hook_frames = self.frame_indices[:n_frames]
    
    # Оптический поток для анализа движения
    hook_flow = compute_optical_flow(self.frame_manager, hook_frames)
    hook_flow_smooth = smooth_signal(hook_flow)
    
    # Motion intensity и spikes
    hook_motion_intensity = np.mean(hook_flow_smooth)
    hook_cut_rate = np.sum(hook_flow_smooth > np.percentile(hook_flow_smooth, 75)) / hook_seconds
    hook_motion_spikes = np.sum(hook_flow_smooth > np.percentile(hook_flow_smooth, 90))
    
    # Face presence через MediaPipe
    face_count = 0
    for idx in hook_frames:
        frame = self.frame_manager.get(idx)
        results = self.mp_face.process(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        if results.multi_face_landmarks:
            face_count += 1
    hook_face_presence = face_count / n_frames
    
    # Visual surprise: изменения CLIP embeddings
    hook_emb = self.clip_embeddings[:n_frames]
    diff = embedding_diff(hook_emb)  # Cosine distance между соседними embeddings
    hook_visual_surprise_score = np.mean(diff)
    hook_visual_surprise_std = np.std(diff)
    
    # Brightness и saturation spikes
    brightness = [np.mean(cv2.cvtColor(self.frame_manager.get(idx), cv2.COLOR_RGB2HSV)[:,:,2]) 
                  for idx in hook_frames]
    saturation = [np.mean(cv2.cvtColor(self.frame_manager.get(idx), cv2.COLOR_RGB2HSV)[:,:,1]) 
                  for idx in hook_frames]
    hook_brightness_spike = max(brightness) - np.mean(brightness)
    hook_saturation_spike = max(saturation) - np.mean(saturation)
    
    return features
```

#### 3. Climax Detection (Детекция кульминации)

#### 3.1. - 3.5. climax_timestamp, climax_strength, number_of_peaks, climax_duration, story_energy_curve

```py
def climax_detection(self):
    # Комбинация сигналов: движение + изменение embeddings
    motion = compute_optical_flow(self.frame_manager, self.frame_indices)
    motion_smooth = smooth_signal(motion)
    embed_diff = embedding_diff(self.clip_embeddings)  # Cosine distance
    embed_diff_smooth = smooth_signal(embed_diff)
    
    # Комбинированный сигнал энергии истории
    combined_signal = motion_smooth[:len(embed_diff_smooth)] + embed_diff_smooth
    
    # Пик = максимальное значение комбинированного сигнала
    peak_idx = np.argmax(combined_signal)
    climax_timestamp = peak_idx  # в кадрах
    climax_strength = combined_signal[peak_idx]
    
    # Количество пиков (выше 90-го перцентиля)
    number_of_peaks = np.sum(combined_signal > np.percentile(combined_signal, 90))
    
    # Длительность кульминации (выше 50-го перцентиля)
    climax_duration = np.sum(combined_signal > np.percentile(combined_signal, 50))
    
    return {
        "climax_timestamp": climax_timestamp,
        "climax_strength": climax_strength,
        "number_of_peaks": number_of_peaks,
        "climax_duration": climax_duration,
        "story_energy_curve": combined_signal.tolist()
    }
```

#### 4. Character Features (Фичи персонажей)

#### 4.1. - 4.4. number_of_speakers, main_character_screen_time, speaker_switch_rate, face_presence_curve

```py
def character_features(self):
    face_tracks = []
    for idx in self.frame_indices:
        frame = self.frame_manager.get(idx)
        results = self.mp_face.process(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        # Количество лиц в кадре
        face_count = len(results.multi_face_landmarks) if results.multi_face_landmarks else 0
        face_tracks.append(face_count)
    
    face_tracks = np.array(face_tracks)
    
    return {
        "number_of_speakers": max(face_tracks),  # Максимальное количество людей одновременно
        "main_character_screen_time": np.sum(face_tracks > 0) / len(face_tracks),  # Доля кадров с лицами
        "speaker_switch_rate": np.sum(np.diff(face_tracks > 0) != 0) / len(face_tracks),  # Частота появления/исчезновения лиц
        "face_presence_curve": face_tracks.tolist()  # Кривая присутствия лиц по кадрам
    }
```

#### 5. Topic Features (Фичи тем)

#### 5.1. - 5.5. number_of_topics, avg_topic_duration, topic_shift_times, topic_diversity, semantic_coherence_score

```py
def topic_features(self, subtitles=None):
    if subtitles is None or len(subtitles) == 0:
        # Если нет субтитров, возвращаем нули
        return {
            "number_of_topics": 0,
            "avg_topic_duration": 0,
            "topic_shift_times": [],
            "topic_diversity": 0,
            "semantic_coherence_score": 0
        }
    
    # SentenceTransformer embeddings для субтитров
    embeddings = self.sentence_model.encode(subtitles)
    
    # Иерархическая кластеризация для выделения тем
    clustering = AgglomerativeClustering(n_clusters=min(5, len(subtitles)), metric='cosine', linkage='average')
    labels = clustering.fit_predict(embeddings)
    
    # Статистика тем
    number_of_topics = len(np.unique(labels))
    durations = [sum(np.array(labels) == i) for i in range(number_of_topics)]
    avg_topic_duration = np.mean(durations)
    topic_shift_times = np.where(np.diff(labels) != 0)[0].tolist()  # Индексы смены тем
    
    # Topic diversity: доля уникальных тем
    topic_diversity = len(np.unique(labels)) / len(labels)
    
    # Semantic coherence: средняя similarity внутри каждой темы
    coherences = []
    for i in range(number_of_topics):
        idx = np.where(labels == i)[0]
        if len(idx) > 1:
            sim = cosine_similarity(embeddings[idx]).mean()
            coherences.append(sim)
    semantic_coherence_score = np.mean(coherences) if coherences else 0
    
    return {
        "number_of_topics": number_of_topics,
        "avg_topic_duration": avg_topic_duration,
        "topic_shift_times": topic_shift_times,
        "topic_diversity": topic_diversity,
        "semantic_coherence_score": semantic_coherence_score
    }
```

## text_scoring

### Модели:

#### EasyOCR
```py
import easyocr
reader = easyocr.Reader(['en', 'ru'], gpu=True)
results = reader.readtext(frame)
```

### Core‑интеграция:

- **motion_peaks** — приоритетно из `core_optical_flow`  
  (`result_store/optical_flow/statistical_analysis.json`, поле  
  `statistics.frame_statistics[*].magnitude_mean_px_sec_norm`  
  или `magnitude_mean_px_sec` / `magnitude_mean`); при отсутствии core‑слоя  
  используется старый результат модуля `optical_flow` (`motion_intensity_curve`).  
- **face_peaks** — пока считываются из `emotion_face`  
  (`result_store/emotion_face/*.json`, поле `emotion_curve`);  
  планируется перенос на провайдер поверх `core_face_landmarks`.  
- **audio_peaks** — зарезервировано под будущий `core_audio_embeddings`  
  (энергия/эмбеддинги аудио по кадрам), сейчас не используется.  

### Выход:

```json
{
    "text_action_sync_score": 0.45,
    "text_motion_alignment": 0.52,
    "text_motion_alignment_windowed": 0.61,
    "multimodal_attention_boost_score": 0.78,
    "multimodal_attention_boost_position": 0.32,
    "text_on_screen_continuity": 2.3,
    "text_on_screen_continuity_normalized": 0.18,
    "text_on_screen_continuity_median": 1.9,
    "text_on_screen_continuity_max": 5.4,
    "text_on_screen_continuity_std": 0.7,
    "text_switch_rate": 0.15,
    "num_unique_texts": 12,
    "time_to_first_text_sec": 0.8,
    "time_to_first_text_position": 0.05,
    "text_area_fraction": 0.12,
    "cta_presence": 0.92,
    "cta_timestamp": 45.2,
    "cta_first_timestamp": 30.1,
    "cta_mean_timestamp": 45.2,
    "cta_last_timestamp": 58.7,
    "cta_first_position": 0.21,
    "cta_mean_position": 0.32,
    "cta_last_position": 0.41,
    "cta_strength": 0.65,
    "persistent_cta_flag": true,
    "text_emphasis_peak_flags": [2, 5, 9],
    "text_emphasis_peak_prominence": [0.23, 0.31, 0.19],
    "text_emphasis_peak_positions": [0.12, 0.35, 0.7],
    "text_readability_score": 0.74,
    "ocr_language_entropy": 0.45,
    "text_movement_speed": 0.03,
    "metadata": {
        "total_frames": 900,
        "fps": 30,
        "ocr_detections_count": 156,
        "device": "cuda"
    }
}
```

### Фичи:

#### 1. Text → Action / Motion Correlation

#### 1.1. text_action_sync_score
Робастная оценка синхронизации текста с движением, основанная на z-score энергии движения в окне вокруг появления текста и усечённом среднем по всем текстовым элементам.

```py
text_action_scores = []
for entry in ocr_data:
    frame_idx = entry["frame"]
    motion_peak_val = motion_signal[frame_idx] if frame_idx < len(motion_signal) else 0
    text_action_scores.append(motion_peak_val)

text_action_sync_score = np.mean(text_action_scores) if text_action_scores else 0
```

#### 1.2. text_motion_alignment
Средняя оценка мультимодального выравнивания текста (движение + лицо + аудио) в момент появления текста. Веса каналов задаются в пайплайне и нормализуются до суммы 1.0.

```py
multimodal_score = 0.4*motion_peak_val + 0.3*face_peak_val + 0.3*audio_peak_val
text_motion_alignment = np.mean(text_motion_align_scores)
```

#### 1.3. text_motion_alignment_windowed
Оконная версия text_motion_alignment — усреднение максимальных мультимодальных скорингов в окне \([t-w, t+w]\) вокруг появления текста.

#### 1.4. multimodal_attention_boost_score
Максимальная оценка мультимодального выравнивания текста по всем уникальным текстовым элементам.

#### 1.5. multimodal_attention_boost_position
Относительная позиция (0..1) текстового элемента, дающего максимум по `multimodal_attention_boost_score`.

```py
multimodal_attention_boost_score = np.max(text_motion_align_scores) if text_motion_align_scores else 0
```

#### 2. Text Duration and Continuity

#### 2.1. text_on_screen_continuity
Средняя длительность отображения уникальных текстовых элементов в секундах (с учётом дедупликации по IoU+тексту).

Дополнительно считаются:  
- `text_on_screen_continuity_normalized` — средняя длительность, делённая на длину видео,  
- `text_on_screen_continuity_median`, `text_on_screen_continuity_max`, `text_on_screen_continuity_std`.

#### 2.2. text_switch_rate
Частота смены текста на экране — число уникальных текстовых элементов в секунду видео: `num_unique_texts / video_length_seconds`.

Также возвращаются `num_unique_texts`, `time_to_first_text_sec`, `time_to_first_text_position` и `text_area_fraction` (средняя доля площади кадра с текстом).

#### 3. Call-to-Action (CTA) Detection

#### 3.1. - 3.3. cta_presence, cta_timestamp, cta_strength
Детекция призывов к действию через комбинацию:
- флагов `is_cta_candidate` из OCR,  
- fuzzy-match по CTA-ключевым словам (EN/RU) и нормализованному тексту.  
`cta_presence` — вероятность наличия CTA (0..1), `cta_timestamp` / `cta_mean_timestamp` — среднее время появления CTA, `cta_strength` — средний мультимодальный скор в CTA-моментах.  
Дополнительно возвращаются `cta_first_timestamp` / `cta_last_timestamp` и их нормализованные версии, а также `persistent_cta_flag` для долгих CTA.

```py
cta_keywords = ['subscribe', 'follow', 'like', 'link in bio', 'click', 'watch']
is_cta = any(keyword in text.lower() for keyword in cta_keywords)

if is_cta:
    cta_times.append(frame_idx)
    cta_strengths.append(multimodal_score)

cta_presence = 1 if cta_times else 0
cta_timestamp = np.mean(cta_times)/self.video_fps if cta_times else None
cta_strength = np.mean(cta_strengths) if cta_strengths else 0
```

#### 4. Text Emphasis Peaks

#### 4.1. text_emphasis_peak_flags
Список индексов текстовых элементов, где мультимодальный текстовый скор образует пики (через `find_peaks`), а также их prominence и нормализованные позиции по видео.

## high_level_semantic

### Модели:

#### CLIP (OpenAI или OpenCLIP) + Trainable Projection
```py
import clip
model, preprocess = clip.load("ViT-L/14", device="cuda")
# или
import open_clip
model, _, preprocess = open_clip.create_model_and_transforms("ViT-B-32", pretrained='laion2b_s13b_b90k')

# Trainable linear projection для CLIP embeddings
projection = nn.Linear(clip_dim, projection_dim)  # например, 512→64
```

### Выход:

```json
{
    "features": {
        "scene_count": 15,
        "avg_scene_duration": 2.5,
        "scene_sim_adjacent_mean": 0.72,
        "scene_sim_adjacent_std": 0.15,
        "video_embedding_norm_mean": 1.0,
        "video_embedding_norm_weighted": 0.98,
        "video_embedding_var_mean": 0.12,
        "topic_probabilities": {
            "gaming": 0.45,
            "tutorial": 0.30,
            "vlog": 0.25
        },
        "topic_diversity_score": 0.85,
        "topic_transition_rate": 0.20,
        "number_of_events": 8,
        "event_rate_per_minute": 2.5,
        "event_strength_max": 0.85,
        "event_types": ["face", "audio", "scene_jump"],
        "event_timestamps": [12.5, 25.3, 38.7],
        "emotion_correlation": 0.65,
        "emotion_lag_seconds": 0.3,
        "emotion_alignment_score": 0.70,
        "emotion_alignment_reliability": 0.75,
        "avg_emotion_valence": 0.45,
        "emotion_variance": 0.25,
        "peak_emotion_intensity": 0.90,
        "face_presence_ratio": 0.85,
        "avg_face_valence": 0.50,
        "avg_face_arousal": 0.60,
        "story_flow_score": 0.75,
        "narrative_complexity_score": 0.20,
        "cross_modal_novelty_score": 0.35,
        "multimodal_attention_overall": 0.68,
        "genre_probabilities": {
            "gaming": 0.50,
            "tutorial": 0.30,
            "vlog": 0.20
        },
        "dominant_genre": "gaming",
        "genre_confidence": 0.50,
        "emotion_reliable": true,
        "audio_reliable": true,
        "ocr_reliable": false,
        "clip_confidence": 0.98
    },
    "scene_embeddings": [[...], [...]],  # Projected embeddings [n_scenes, projection_dim]
    "scene_metadata": [
        {
            "scene_id": 0,
            "start_ts": 0.0,
            "end_ts": 2.5,
            "representative_frame_idx": 37,
            "scene_duration": 2.5,
            "scene_source_confidence": 0.95
        },
        ...
    ],
    "video_embeddings": {
        "mean_embedding": [...],
        "weighted_mean_embedding": [...],
        "max_embedding": [...],
        "var_embedding": [...],
        "attention_weights": [...]
    },
    "per_scene_vectors": [[...], [...]],  # [n_scenes, ~64] для VisualTransformer
    "reliability_flags": {
        "emotion_reliable": true,
        "audio_reliable": true,
        "ocr_reliable": false,
        "clip_confidence": 0.98
    },
    "metadata": {
        "total_frames": 900,
        "fps": 30,
        "n_scenes": 15,
        "device": "cuda",
        "clip_model": "ViT-L/14",
        "projection_dim": 64,
        "mode": "full"
    }
}
```

### Фичи:

#### 1. Scene-level Semantic Embeddings

#### 1.1. scene_embeddings
Массив проектированных CLIP embeddings для каждой сцены. CLIP embedding → L2-нормализация → trainable linear projection (512/768/1024 → 64/128). Каждая сцена представлена репрезентативным кадром.

```py
# CLIP encoding с L2-нормализацией
clip_embs = clip_model.encode_image(frames)
clip_embs = clip_embs / clip_embs.norm(dim=-1, keepdim=True)

# Projection через trainable layer
projected_embs = projection_layer(clip_embs)  # [n_scenes, projection_dim]
projected_embs = projected_embs / (projected_embs.norm(dim=-1, keepdim=True) + 1e-9)
```

#### 1.2. scene_metadata
Метаданные для каждой сцены: scene_id, start_ts, end_ts, representative_frame_idx, scene_duration, scene_source_confidence.

#### 1.3. - 1.4. scene_sim_adjacent_mean, scene_sim_adjacent_std
Средняя и стандартное отклонение схожести между соседними сценами (на проектированных embeddings).

```py
sims = []
for i in range(n - 1):
    sim = cosine_similarity(scene_embeddings[i:i+1], scene_embeddings[i+1:i+2])[0][0]
    sims.append(float(sim))

scene_sim_adjacent_mean = float(np.mean(sims))
scene_sim_adjacent_std = float(np.std(sims))
```

#### 2. Video-level Embeddings

#### 2.1. - 2.5. mean_embedding, weighted_mean_embedding, max_embedding, var_embedding, attention_weights
Агрегированные embeddings на уровне видео. Weighted mean использует learnable attention fusion вместо жёстких весов 45/35/20.

```py
mean_emb = scene_embeddings.mean(axis=0)
max_emb = scene_embeddings.max(axis=0)
var_emb = scene_embeddings.var(axis=0)
video_embedding_var_mean = float(np.mean(var_emb))

# Learnable attention fusion для weighted mean
attention_weights = multimodal_attention_fusion.compute_attention_weights(
    face_signal=face_w,
    audio_signal=audio_w,
    text_signal=text_w,
    ocr_signal=ocr_w,
)
weighted_mean = (scene_embeddings * attention_weights[:, None]).sum(axis=0)
```

#### 3. Topic / Concept Detection

#### 3.1. - 3.4. topic_probabilities, per_scene_dominant_topic, topic_diversity_score, topic_transition_rate
Детекция тем через схожесть с topic vectors и softmax с температурой для калибровки.

```py
if topic_vectors:
    vecs = np.array([topic_vectors[k] for k in names])
    vecs = vecs / (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)  # Normalize
    sims = cosine_similarity(scene_embeddings, vecs)
    
    # Softmax с температурой
    sims_scaled = sims / temperature
    exps = np.exp(sims_scaled - sims_scaled.max(axis=1, keepdims=True))
    probs = exps / (exps.sum(axis=1, keepdims=True) + 1e-9)
    
    agg = probs.mean(axis=0)
    agg = agg / (agg.sum() + 1e-9)
    topic_probabilities = {names[i]: float(agg[i]) for i in range(len(names))}
    
    # Normalized entropy для diversity
    topic_diversity_score = _entropy(agg)  # 0..1
    
    dominant = [names[i] for i in sims.argmax(axis=1)]
    transitions = int(np.sum(np.array([names.index(d) for d in dominant[1:]]) != np.array([names.index(d) for d in dominant[:-1]])))
    topic_transition_rate = transitions / max(1, n - 1)
```

#### 4. Events / Key Moments Detection

#### 4.1. - 4.6. number_of_events, event_rate_per_minute, event_strength_max, event_types, event_timestamps, event_context_embeddings
Детекция ключевых событий через анализ пиков в комбинированном мультимодальном сигнале с learnable весами и улучшенным алгоритмом.

```py
# Нормализованные кривые
f_face = _norm01(_smooth(face_emotion_curve, sigma))
f_audio = _norm01(_smooth(audio_energy_curve, sigma))
scene_jump = _scene_jump_signal(scene_embeddings, L=L)

# Learnable attention weights
attention_weights = attention_fusion.compute_attention_weights(f_face, f_audio, f_text, f_ocr)
combined = attention_weights * f_face + attention_weights * f_audio + 0.3 * scene_jump + 0.1 * f_text + 0.05 * f_pose
combined = _norm01(_smooth(combined, sigma=sigma))

# Улучшенный поиск пиков
thresh = combined.mean() + k_std * combined.std()
peaks, properties = find_peaks(combined, height=thresh, distance=min_distance_frames)

number_of_events = int(len(peaks))
event_rate_per_minute = len(peaks) / (L / (fps * 60.0))
event_strengths = combined[peaks].tolist()
event_strength_max = float(max(event_strengths))
event_timestamps = (peaks / float(fps)).tolist()

# Event context embeddings (±1s вокруг события)
event_context_embeddings = []
for t in event_timestamps:
    context_scenes = [scene_embeddings[i] for i, meta in enumerate(scene_metadata) 
                      if meta["start_ts"] <= t + 1.0 and meta["end_ts"] >= t - 1.0]
    if context_scenes:
        event_context_embeddings.append(np.mean(context_scenes, axis=0))
```

#### 5. Emotion Alignment

#### 5.1. - 5.4. emotion_correlation, emotion_lag_seconds, emotion_alignment_score, emotion_alignment_reliability
Выравнивание эмоций между визуальным и текстовым каналами через кросс-корреляцию с поиском оптимального лага.

```py
# Кросс-корреляция для поиска лага
max_lag_frames = int(max_lag_seconds * fps)
corr = np.correlate(f_r - f_r.mean(), t_r - t_r.mean(), mode='full')
center = len(corr) // 2
lag_range = slice(max(0, center - max_lag_frames), min(len(corr), center + max_lag_frames + 1))
corr_window = corr[lag_range]
lag_idx = lag_range.start + corr_window.argmax() - center
emotion_lag_seconds = lag_idx / float(fps)

# Нормализованная корреляция
corr_max = corr_window.max()
corr_norm = corr_max / (np.sqrt(np.sum((f_r - f_r.mean())**2) * np.sum((t_r - t_r.mean())**2)) + 1e-9)
emotion_alignment_score = float(np.clip(corr_norm, 0.0, 1.0))

# Pearson correlation
emotion_correlation, _ = pearsonr(f_r, t_r)

# Reliability: доля совпадающих пиков
f_peaks, _ = find_peaks(f_r, height=f_r.mean() + 0.5 * f_r.std(), distance=int(0.5 * fps))
t_peaks, _ = find_peaks(t_r, height=t_r.mean() + 0.5 * t_r.std(), distance=int(0.5 * fps))
aligned_peaks = sum(1 for fp in f_peaks if any(abs(tp - (fp + lag_idx)) < int(0.5 * fps) for tp in t_peaks))
emotion_alignment_reliability = aligned_peaks / max(1, len(f_peaks))
```

#### 6. Emotion Features

#### 6.1. - 6.6. avg_emotion_valence, emotion_variance, peak_emotion_intensity, face_presence_ratio, avg_face_valence, avg_face_arousal
Фичи эмоций с добавлением face_presence_ratio и отдельных кривых валентности/активации.

```py
curve = _norm01(_smooth(face_emotion_curve, sigma))
avg_emotion_valence = float(curve.mean())
emotion_variance = float(curve.var())
peak_emotion_intensity = float(curve.max())

# Face presence ratio
face_presence_ratio = float(np.mean(face_presence_per_frame > 0.5)) if face_presence_per_frame is not None else float(np.mean(curve > 0.1))

# Valence и arousal (если доступны)
avg_face_valence = float(np.mean(_norm01(face_valence_curve))) if face_valence_curve is not None else avg_emotion_valence
avg_face_arousal = float(np.mean(_norm01(face_arousal_curve))) if face_arousal_curve is not None else 0.0
```

#### 7. Narrative / Story Features

#### 7.1. - 7.3. narrative_embedding, story_flow_score, narrative_complexity_score
Фичи повествования через multimodal attention fusion визуальных и текстовых embeddings.

```py
# Story flow: mean cosine similarity между соседними сценами (на проектированных embeddings)
sims = []
for i in range(len(scene_embeddings) - 1):
    sim = cosine_similarity(scene_embeddings[i:i+1], scene_embeddings[i+1:i+2])[0][0]
    sims.append(float(sim))
story_flow_score = float(np.mean(sims))

# Narrative complexity: std(similarity) + topic_transition_rate + topic_diversity
narrative_complexity_score = float(np.std(sims)) + topic_transition_rate + topic_diversity_score

# Multimodal attention fusion для narrative embedding
visual_mean = scene_embeddings.mean(axis=0)
if scene_caption_embeddings is not None:
    text_normalized = scene_caption_embeddings / (np.linalg.norm(scene_caption_embeddings, axis=1, keepdims=True) + 1e-9)
    text_mean = text_normalized.mean(axis=0)
    narrative_emb = 0.6 * visual_mean + 0.4 * text_mean  # Можно использовать learnable weights
    narrative_emb = narrative_emb / (np.linalg.norm(narrative_emb) + 1e-9)
else:
    narrative_emb = visual_mean
```

#### 8. Multimodal Features

#### 8.1. - 8.2. cross_modal_novelty_score, multimodal_attention_overall
Новизна контента и согласованность мультимодальных сигналов.

```py
# Novelty: mean dissimilarity между соседними сценами
novelty = float(np.mean(1 - np.array(sims)))

# Multimodal attention: корреляция между нормализованными кривыми
R = np.corrcoef(np.vstack([face_norm, audio_norm, text_norm]))
off_diag = R[np.triu_indices(R.shape[0], k=1)]
multimodal_attention_overall = float(np.clip(np.mean(np.maximum(off_diag, 0.0)), 0.0, 1.0))
```

#### 9. Genre / Style Classification

#### 9.1. - 9.4. genre_probabilities, per_scene_top_class, dominant_genre, genre_confidence
Zero-shot классификация через CLIP с temperature scaling для калибровки.

```py
# Encode text prompts
text_feats = clip_model.encode_text(clip.tokenize(class_prompts))
text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
text_projected = projection_layer(text_feats)

# Similarity с temperature scaling
sims = cosine_similarity(scene_embeddings, text_projected)
sims_scaled = sims / temperature
exps = np.exp(sims_scaled - sims_scaled.max(axis=1, keepdims=True))
probs_per_scene = exps / (exps.sum(axis=1, keepdims=True) + 1e-9)

agg = probs_per_scene.mean(axis=0)
agg = agg / (agg.sum() + 1e-9)
genre_probabilities = {class_prompts[i]: float(agg[i]) for i in range(len(class_prompts))}
dominant_genre = class_prompts[agg.argmax()]
genre_confidence = float(agg.max())
per_scene_top_class = [class_prompts[i] for i in probs_per_scene.argmax(axis=1)]
```

#### 10. Per-Scene Vectors для VisualTransformer

#### 10.1. per_scene_vectors
Компактный per-scene вектор (≈64 dims) для VisualTransformer.

```py
per_scene_vectors = []
for i in range(n_scenes):
    vec = []
    # Projected CLIP embedding (32-48 dims)
    vec.extend(scene_embeddings[i][:projection_dim].tolist())
    # Additional features
    vec.append(scene_duration_norm)
    vec.append(scene_position_norm)
    vec.append(audio_energy_norm)
    vec.append(face_presence_flag)
    vec.append(avg_face_valence)
    vec.append(avg_face_arousal)
    vec.append(text_activity_flag)
    vec.append(text_sentiment)
    vec.extend(topic_embedding)  # 4 dims
    vec.append(scene_novelty_score)
    vec.append(multimodal_attention_score)
    vec.append(scene_visual_confidence)
    per_scene_vectors.append(vec)
```

#### 11. Reliability Flags

#### 11.1. - 11.4. emotion_reliable, audio_reliable, ocr_reliable, clip_confidence
Флаги надёжности модальностей для downstream моделей.

```py
emotion_reliable = face_emotion_curve is not None and len(face_emotion_curve) > 0 and np.mean(face_emotion_curve) > 0.01
audio_reliable = audio_energy_curve is not None and len(audio_energy_curve) > 0 and np.mean(audio_energy_curve) > 0.01
ocr_reliable = ocr_activity_curve is not None and len(ocr_activity_curve) > 0 and np.mean(ocr_activity_curve) > 0.01

# CLIP confidence: средняя норма embeddings (должна быть близка к 1.0)
norms = np.linalg.norm(scene_embeddings, axis=1)
clip_confidence = float(np.mean(norms))
```

## similarity_metrics

### Модели:

Модуль не использует модели напрямую, а работает с embeddings и фичами из других модулей.

### Выход:

```json
{
    "features": {
        "semantic_similarity_mean": 0.65,
        "semantic_similarity_max": 0.82,
        "semantic_similarity_p10": 0.40,
        "semantic_novelty_score": 0.55,
        "topic_overlap_score": 0.45,
        "topic_diversity_comparison": 0.20,
        "key_concept_match_ratio": 0.60,
        "color_histogram_similarity": 0.70,
        "lighting_pattern_similarity": 0.65,
        "shot_type_distribution_similarity": 0.55,
        "cut_rate_similarity": 0.80,
        "motion_pattern_similarity": 0.72,
        "ocr_text_semantic_similarity": 0.50,
        "text_layout_similarity": 0.45,
        "text_timing_similarity": 0.55,
        "audio_embedding_similarity": 0.68,
        "speech_content_similarity": 0.62,
        "music_tempo_similarity": 0.75,
        "audio_energy_pattern_similarity": 0.70,
        "emotion_curve_similarity": 0.65,
        "pose_motion_similarity": 0.58,
        "behavior_pattern_similarity": 0.60,
        "pacing_curve_similarity": 0.72,
        "shot_duration_distribution_similarity": 0.68,
        "scene_length_similarity": 0.70,
        "temporal_pattern_novelty": 0.28,
        "overall_similarity_score": 0.65,
        "uniqueness_score": 0.35,
        "trend_alignment_score": 0.65,
        "viral_pattern_score": 0.68
    },
    "metadata": {
        "total_frames": 900,
        "top_n": 10,
        "reference_videos_count": 25
    }
}
```

### Фичи:

#### 1. Semantic Similarity

#### 1.1. - 1.4. semantic_similarity_mean, semantic_similarity_max, semantic_similarity_p10, semantic_novelty_score
Семантическая схожесть через косинусную схожесть embeddings (mean/max/p10) и нормализованная новизна на основе \(1 - \mathrm{max\_similarity}\).

```py
video_emb_norm = video_embedding / (np.linalg.norm(video_embedding) + 1e-10)
similarities = []
for ref_emb in reference_embeddings:
    ref_emb_norm = ref_emb / (np.linalg.norm(ref_emb) + 1e-10)
    sim = np.dot(video_emb_norm, ref_emb_norm)
    similarities.append(float(sim))

similarities = np.array(similarities)
top_similarities = np.sort(similarities)[-self.top_n:] if len(similarities) > self.top_n else similarities

semantic_similarity_mean = float(np.mean(top_similarities))
semantic_similarity_max = float(np.max(similarities))
semantic_similarity_p10 = float(np.percentile(similarities, 10.0))
semantic_novelty_score = float(np.clip((1.0 - semantic_similarity_max) * 0.5 + 0.5, 0.0, 1.0))
```

#### 2. Topic / Concept Overlap

#### 2.1. - 2.3. topic_overlap_score, topic_diversity_comparison, key_concept_match_ratio
Тематическое пересечение через взвешенный Jaccard и энтропию распределений тем.

```py
def to_weighted_dict(topics):
    # dict: {topic: weight}, list[str]: weight=1.0, list[float]/np.ndarray: индексы с prob>0
    ...

video_topics_w = to_weighted_dict(video_topics)
video_keys = set(video_topics_w.keys())

ref_topics_w = to_weighted_dict(ref_topics)
ref_keys = set(ref_topics_w.keys())

# Взвешенный Jaccard
inter = sum(min(video_topics_w.get(k, 0.0), ref_topics_w.get(k, 0.0)) for k in video_keys | ref_keys)
union = sum(max(video_topics_w.get(k, 0.0), ref_topics_w.get(k, 0.0)) for k in video_keys | ref_keys)
topic_overlap_score = inter / (union + 1e-10) if union > 0 else 0.0

# Разница в diversity через энтропию весов
def entropy_from_weights(w):
    arr = np.asarray(list(w.values()), dtype=np.float32)
    arr = arr / (arr.sum() + 1e-10)
    return float(-np.sum(arr * np.log(arr + 1e-10)))

div_video = entropy_from_weights(video_topics_w)
div_ref = entropy_from_weights(ref_topics_w)
topic_diversity_comparison = abs(div_video - div_ref) / max(div_video, div_ref, 1e-6)

# Взвешенная доля совпадающих ключевых концептов
inter_weight = sum(min(video_topics_w.get(k, 0.0), ref_topics_w.get(k, 0.0)) for k in video_keys)
total_weight = sum(video_topics_w.values()) + 1e-10
key_concept_match_ratio = inter_weight / total_weight
```

#### 3. Style & Composition Similarity

#### 3.1. - 3.5. color_histogram_similarity, lighting_pattern_similarity, shot_type_distribution_similarity, cut_rate_similarity, motion_pattern_similarity
Схожесть визуального стиля через различные метрики.

```py
# Color histogram similarity
hist1_norm = hist1 / (np.linalg.norm(hist1) + 1e-10)
hist2_norm = hist2 / (np.linalg.norm(hist2) + 1e-10)
color_sim = np.dot(hist1_norm, hist2_norm)

# Shot type distribution similarity
dist1_norm = dist1 / (dist1.sum() + 1e-10)
dist2_norm = dist2 / (dist2.sum() + 1e-10)
shot_sim = 1.0 - wasserstein_distance(dist1_norm, dist2_norm) / (np.max(dist1_norm) + np.max(dist2_norm) + 1e-10)

# Motion pattern similarity
corr, _ = pearsonr(motion1, motion2)
motion_sim = max(0.0, corr) if not np.isnan(corr) else 0.0
```

#### 4. Text & OCR Similarity

#### 4.1. - 4.3. ocr_text_semantic_similarity, text_layout_similarity, text_timing_similarity
Схожесть текстовых характеристик.

```py
# OCR text semantic similarity
emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-10)
emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-10)
ocr_sim = np.dot(emb1_norm, emb2_norm)

# Text timing similarity
corr, _ = pearsonr(timing1, timing2)
timing_sim = max(0.0, corr) if not np.isnan(corr) else 0.0
```

#### 5. Audio / Speech Similarity

#### 5.1. - 5.4. audio_embedding_similarity, speech_content_similarity, music_tempo_similarity, audio_energy_pattern_similarity
Схожесть аудио характеристик.

```py
# Audio embedding similarity
audio_sim = np.dot(audio_emb1_norm, audio_emb2_norm)

# Music tempo similarity
max_tempo = max(abs(tempo1), abs(tempo2), 1.0)
tempo_sim = 1.0 - abs(tempo1 - tempo2) / max_tempo

# Audio energy pattern similarity
corr, _ = pearsonr(energy1, energy2)
energy_sim = max(0.0, corr) if not np.isnan(corr) else 0.0
```

#### 6. Emotion & Behavior Similarity

#### 6.1. - 6.3. emotion_curve_similarity, pose_motion_similarity, behavior_pattern_similarity
Схожесть эмоций и поведения.

```py
# Emotion curve similarity
corr, _ = pearsonr(curve1, curve2)
emotion_sim = max(0.0, corr) if not np.isnan(corr) else 0.0

# Pose motion similarity
pose_sim = np.dot(pose1_norm.flatten(), pose2_norm.flatten())
```

#### 7. Temporal / Pacing Similarity

#### 7.1. - 7.4. pacing_curve_similarity, shot_duration_distribution_similarity, scene_length_similarity, temporal_pattern_novelty
Схожесть временного ритма.

```py
# Pacing curve similarity
corr, _ = pearsonr(curve1, curve2)
pacing_sim = max(0.0, corr) if not np.isnan(corr) else 0.0

# Shot duration distribution similarity
wd = wasserstein_distance(dist1_norm, dist2_norm)
shot_sim = 1.0 - wd / (max_wd + 1e-10)

# Temporal pattern novelty
temporal_pattern_novelty = 1.0 - mean_pacing_sim
```

#### 8. High-level Comparative Scores

#### 8.1. - 8.4. overall_similarity_score, uniqueness_score, trend_alignment_score, viral_pattern_score
Высокоуровневые сравнительные оценки (эвристический агрегат, рекомендовано дообучать).

```py
overall_similarity = (
    weights['semantic'] * semantic_score +
    weights['topics'] * topics_score +
    weights['visual'] * visual_score +
    weights['text'] * text_score +
    weights['audio'] * audio_score +
    weights['emotion'] * emotion_score +
    weights['temporal'] * temporal_score
)
overall_similarity = float(np.clip(overall_similarity, 0.0, 1.0))

uniqueness_score = 1.0 - overall_similarity
trend_alignment_score = overall_similarity  # В бою лучше обучать отдельно
viral_pattern_score = overall_similarity   # В бою лучше обучать отдельно
```

## uniqueness

### Модели:

Модуль не использует модели напрямую, а работает с embeddings и фичами из других модулей.

### Выход:

```json
{
    "features": {
        "semantic_novelty_score": 0.75,
        "semantic_novelty_max": 0.75,
        "semantic_novelty_topk_mean": 0.68,
        "semantic_novelty_topk_median": 0.70,
        "topic_novelty_score": 0.60,
        "concept_diversity_score": 0.85,
        "concept_diversity_entropy": 0.85,
        "concept_diversity_unique_norm": 0.42,
        "color_palette_novelty": 0.70,
        "lighting_style_novelty": 0.65,
        "shot_type_novelty": 0.55,
        "camera_motion_novelty": 0.72,
        "cut_rate_novelty": 0.80,
        "shot_duration_distribution_novelty": 0.68,
        "scene_length_novelty": 0.70,
        "pacing_pattern_novelty": 0.72,
        "music_track_novelty": 0.68,
        "voice_style_novelty": 0.62,
        "sound_effects_novelty": 0.75,
        "audio_energy_pattern_novelty": 0.70,
        "ocr_text_novelty": 0.50,
        "text_layout_novelty": 0.45,
        "text_style_novelty": 0.55,
        "pose_motion_novelty": 0.65,
        "object_interaction_novelty": 0.58,
        "action_sequence_novelty": 0.60,
        "multimodal_novelty_score": 0.68,
        "novel_event_alignment_score": 0.55,
        "overall_novelty_index": 0.65,
        "trend_alignment_score": 0.35,
        "historical_similarity_score": 0.50,
        "early_adopter_score": 0.65
    },
    "metadata": {
        "total_frames": 900,
        "top_n": 100,
        "reference_videos_count": 25
    }
}
```

### Фичи:

#### 1. Semantic / Content Novelty

#### 1.1. - 1.3. semantic_novelty_score, topic_novelty_score, concept_diversity_score
Семантическая новизна контента.

```py
video_emb_norm = video_embedding / (np.linalg.norm(video_embedding) + 1e-10)
similarities = []
for ref_emb in reference_embeddings:
    ref_emb_norm = ref_emb / (np.linalg.norm(ref_emb) + 1e-10)
    sim = np.dot(video_emb_norm, ref_emb_norm)
    similarities.append(float(sim))

max_similarity = np.max(similarities)
semantic_novelty_score = 1.0 - max_similarity

# Topic novelty: доля новых концептов
new_concepts = video_topics_set - all_reference_topics
topic_novelty_score = len(new_concepts) / len(video_topics_set) if len(video_topics_set) > 0 else 0.0

# Concept diversity: энтропия распределения тем
probs = np.array(list(video_topics.values()))
probs = probs / (probs.sum() + 1e-10)
concept_diversity_score = entropy(probs) / np.log(len(probs) + 1e-10)
```

#### 2. Visual / Style Novelty

#### 2.1. - 2.4. color_palette_novelty, lighting_style_novelty, shot_type_novelty, camera_motion_novelty
Визуальная новизна стиля.

```py
# Color palette novelty
hist1_norm = hist1 / (np.linalg.norm(hist1) + 1e-10)
hist2_norm = hist2 / (np.linalg.norm(hist2) + 1e-10)
sim = np.dot(hist1_norm, hist2_norm)
color_palette_novelty = 1.0 - sim

# Shot type novelty
dist1_norm = dist1 / (dist1.sum() + 1e-10)
dist2_norm = dist2 / (dist2.sum() + 1e-10)
wd = wasserstein_distance(dist1_norm, dist2_norm)
max_wd = np.max(dist1_norm) + np.max(dist2_norm)
sim = 1.0 - wd / (max_wd + 1e-10)
shot_type_novelty = 1.0 - sim
```

#### 3. Editing & Pacing Novelty

#### 3.1. - 3.4. cut_rate_novelty, shot_duration_distribution_novelty, scene_length_novelty, pacing_pattern_novelty
Новизна монтажа и ритма.

```py
# Cut rate novelty
max_cut = max(abs(cut1), abs(cut2), 1.0)
diff = abs(cut1 - cut2) / max_cut
cut_rate_novelty = min(1.0, diff)

# Pacing pattern novelty
corr, _ = pearsonr(curve1, curve2)
pacing_pattern_novelty = 1.0 - max(0.0, corr) if not np.isnan(corr) else 1.0
```

#### 4. Audio Novelty

#### 4.1. - 4.4. music_track_novelty, voice_style_novelty, sound_effects_novelty, audio_energy_pattern_novelty
Аудио новизна.

```py
# Music track novelty
emb1_norm = emb1 / (np.linalg.norm(emb1) + 1e-10)
emb2_norm = emb2 / (np.linalg.norm(emb2) + 1e-10)
sim = np.dot(emb1_norm, emb2_norm)
music_track_novelty = 1.0 - sim
```

#### 5. Text / OCR Novelty

#### 5.1. - 5.3. ocr_text_novelty, text_layout_novelty, text_style_novelty
Текстовая новизна.

```py
# OCR text novelty
ocr_text_novelty = 1.0 - ocr_semantic_similarity

# Text layout novelty
layout_sim = 1.0 - cosine(layout1.flatten(), layout2.flatten())
text_layout_novelty = 1.0 - layout_sim
```

#### 6. Behavioral & Motion Novelty

#### 6.1. - 6.3. pose_motion_novelty, object_interaction_novelty, action_sequence_novelty
Поведенческая новизна.

```py
# Pose motion novelty
pose_sim = np.dot(pose1_norm.flatten(), pose2_norm.flatten())
pose_motion_novelty = 1.0 - pose_sim
```

#### 7. Multimodal Novelty

#### 7.1. - 7.3. multimodal_novelty_score, novel_event_alignment_score, overall_novelty_index
Мультимодальная новизна.

```py
multimodal_novelty = (
    weights['semantic'] * semantic_novelty +
    weights['visual'] * visual_novelty +
    weights['editing'] * editing_novelty +
    weights['audio'] * audio_novelty +
    weights['text'] * text_novelty +
    weights['behavioral'] * behavioral_novelty
)

# Novel event alignment
new_event_types = video_event_types - all_reference_event_types
novel_event_alignment_score = len(new_event_types) / len(video_event_types) if len(video_event_types) > 0 else 0.0

# Overall novelty index
overall_novelty_index = (
    weights['semantic'] * semantic_novelty +
    weights['visual'] * visual_novelty +
    weights['editing'] * editing_novelty +
    weights['audio'] * audio_novelty +
    weights['text'] * text_novelty +
    weights['behavioral'] * behavioral_novelty +
    weights['multimodal'] * multimodal_novelty +
    weights['temporal'] * temporal_novelty
)
```

## micro_emotion

### Модели:

#### OpenFace (через Docker)
```bash
docker pull openface/openface:latest
```

### Выход:

```json
{
    "success": true,
    "face_count": 850,
    "success_rate": 0.94,
    "features": {
        "AU06_intensity_mean": 1.25,
        "AU06_intensity_std": 0.45,
        "AU06_intensity_delta_mean": 0.85,
        "AU06_presence_rate": 0.65,
        "AU06_peak_count": 12,
        "AU12_intensity_mean": 1.85,
        "AU12_intensity_std": 0.50,
        "AU12_intensity_delta_mean": 1.20,
        "AU12_presence_rate": 0.75,
        "AU12_peak_count": 15,
        "au_pca_1": 0.45,
        "au_pca_2": -0.23,
        "au_pca_3": 0.12,
        "pose_Ry_mean": -1.2,
        "pose_Rx_mean": 2.5,
        "pose_Ry_std": 8.5,
        "pose_Rx_std": 5.2,
        "pose_stability_score": 0.72,
        "pose_Tz_mean": 450.0,
        "gaze_x_mean": 2.3,
        "gaze_y_mean": -1.5,
        "gaze_x_std": 8.5,
        "gaze_y_std": 5.2,
        "gaze_centered_ratio": 0.68,
        "blink_rate_per_min": 18.5,
        "eye_contact_score": 0.75,
        "mouth_opening_mean": 0.15,
        "mouth_opening_std": 0.05,
        "smile_width_mean": 45.2,
        "smile_width_std": 8.5,
        "face_asymmetry_score": 0.12,
        "landmarks_pca_1": 0.35,
        "landmarks_pca_2": -0.18,
        "head_depth_variation": 12.5,
        "microexpr_count": 25,
        "microexpr_rate_per_min": 1.67,
        "microexpr_max_intensity": 0.85,
        "microexpr_types_distribution": {
            "smile": 12,
            "surprise": 5,
            "frown": 4,
            "disgust": 4
        },
        "smile_ratio": 0.45,
        "eye_contact_ratio": 0.68,
        "face_presence_ratio": 0.94,
        "au_quality_overall": 0.82,
        "landmark_visibility_mean": 0.91
    },
    "per_frame_vectors": [[...], [...]],  # [n_frames, ~22] для VisualTransformer
    "reliability_flags": {
        "au_quality_reliable": true,
        "landmark_visibility_reliable": true,
        "occlusion_flag": false,
        "lighting_flag": false
    },
    "microexpr_features": {
        "microexpr_count": 25,
        "microexpr_rate_per_min": 1.67,
        "microexpr_max_intensity": 0.85,
        "microexpr_types_distribution": {...},
        "microexpr_timestamps": [12.5, 25.3, 38.7, ...],
        "microexpr_types": ["smile", "surprise", ...]
    },
    "summary": {
        "total_frames": 900,
        "frames_processed": 900,
        "frames_with_face": 850,
        "au_count": 14,
        "landmarks_2d_count": 68,
        "landmarks_3d_count": 68
    },
    "metadata": {
        "processing_mode": "optimized",
        "openface_version": "2.2.0",
        "docker_image_tag": "latest"
    }
}
```

### Фичи:

#### 1. Ключевые Action Units (AU) — оптимизировано

#### 1.1. Key AU (AU06, AU12, AU04, AU01, AU02, AU25, AU26, AU07, AU23, AU45, AU43, AU15, AU20, AU10)
Выбранные 10-14 ключевых AU вместо всех 45. Для каждого AU:

```py
# Baseline subtraction для уменьшения межсубъектного сдвига
neutral_frames = df[total_activity <= np.percentile(total_activity, 20)]
baseline = neutral_frames[au_col].mean()

# Интенсивность с baseline correction
intensities = df[au_col].values
intensities_delta = intensities - baseline

features[f'{au}_intensity_mean'] = float(np.mean(intensities))
features[f'{au}_intensity_std'] = float(np.std(intensities))
features[f'{au}_intensity_delta_mean'] = float(np.mean(intensities_delta))
features[f'{au}_presence_rate'] = float(df[presence_col].mean())

# Peak detection для вспышек
smoothed = gaussian_filter1d(intensities, sigma=sigma)
peaks, _ = find_peaks(smoothed, height=baseline + 1.5*std, distance=min_distance)
features[f'{au}_peak_count'] = len(peaks)
```

#### 1.2. PCA для остальных AU
Остальные AU проецируются через PCA:

```py
non_key_au_cols = [col for col in au_cols if col not in KEY_AUS]
pca = PCA(n_components=3)
au_pca = pca.fit_transform(df[non_key_au_cols].values)

features['au_pca_1'] = float(np.mean(au_pca[:, 0]))
features['au_pca_2'] = float(np.mean(au_pca[:, 1]))
features['au_pca_3'] = float(np.mean(au_pca[:, 2]))
features['au_pca_var_explained_1'] = float(pca.explained_variance_ratio_[0])
```

#### 2. Head Pose — оптимизировано

#### 2.1. - 2.3. pose_Ry_mean, pose_Rx_mean, pose_stability_score
Основные метрики для трансформера:

```py
features['pose_Ry_mean'] = float(df['pose_Ry'].mean())  # Горизонтальный поворот
features['pose_Rx_mean'] = float(df['pose_Rx'].mean())  # Наклон/кивок
features['pose_Ry_std'] = float(df['pose_Ry'].std())
features['pose_Rx_std'] = float(df['pose_Rx'].std())

# Pose stability score
total_std = np.sqrt(rx_std**2 + ry_std**2 + rz_std**2)
pose_stability_score = 1.0 - (total_std / max_expected_std)  # Normalized
features['pose_stability_score'] = float(np.clip(pose_stability_score, 0.0, 1.0))

# Tz (приближение/удаление)
features['pose_Tz_mean'] = float(df['pose_Tz'].mean())
```

#### 3. Gaze Direction — улучшено

#### 3.1. - 3.4. gaze_x_mean, gaze_y_mean, gaze_centered_ratio, eye_contact_score
Направление взгляда с метриками зрительного контакта:

```py
features['gaze_x_mean'] = float(df['gaze_angle_x'].mean())
features['gaze_y_mean'] = float(df['gaze_angle_y'].mean())
features['gaze_x_std'] = float(df['gaze_angle_x'].std())
features['gaze_y_std'] = float(df['gaze_angle_y'].std())

# Gaze centered ratio (взгляд в камеру)
centered = (np.abs(df['gaze_angle_x']) < threshold) & (np.abs(df['gaze_angle_y']) < threshold)
features['gaze_centered_ratio'] = float(centered.mean())

# Blink rate (AU45 presence < 0.25s)
blink_frames = int(0.25 * fps)
blink_count = count_blinks(df['AU45_c'], blink_frames)
features['blink_rate_per_min'] = blink_count / duration_minutes

# Eye contact score
eye_contact_score = (gaze_centered_ratio * 0.7) + (normalized_blink_rate * 0.3)
features['eye_contact_score'] = float(eye_contact_score)
```

#### 4. Facial Landmarks — оптимизировано

#### 4.1. - 4.4. mouth_opening_mean, smile_width_mean, face_asymmetry_score, landmarks_pca_1..5
Компактные геометрические признаки вместо всех координат:

```py
# Mouth opening (нормализованное по межглазному расстоянию)
upper_lip_y = df[['y_51', 'y_52', 'y_53']].mean(axis=1)
lower_lip_y = df[['y_57', 'y_58', 'y_59']].mean(axis=1)
mouth_opening = np.abs(upper_lip_y - lower_lip_y)
interocular_dist = np.sqrt((df['x_36'] - df['x_45'])**2 + (df['y_36'] - df['y_45'])**2).mean()
mouth_opening_norm = mouth_opening / interocular_dist
features['mouth_opening_mean'] = float(mouth_opening_norm.mean())

# Smile width
smile_width = np.sqrt((df['x_48'] - df['x_54'])**2 + (df['y_48'] - df['y_54'])**2)
features['smile_width_mean'] = float(smile_width.mean())

# Face asymmetry (корреляция L-R distances)
asymmetry_scores = []
for i in range(17):
    left_dist = np.abs(df[f'x_{i}'] - center_x)
    right_dist = np.abs(df[f'x_{16-i}'] - center_x)
    corr = np.corrcoef(left_dist, right_dist)[0, 1]
    asymmetry_scores.append(1.0 - abs(corr))
features['face_asymmetry_score'] = float(np.mean(asymmetry_scores))

# PCA для landmarks
landmark_matrix = np.array([df[f'x_{i}'], df[f'y_{i}']] for i in range(68)).T
pca_landmarks = PCA(n_components=5)
landmarks_pca = pca_landmarks.fit_transform(landmark_matrix)
for i in range(5):
    features[f'landmarks_pca_{i+1}'] = float(np.mean(landmarks_pca[:, i]))
```

#### 5. Micro-expressions Detection — новый функционал

#### 5.1. - 5.6. microexpr_count, microexpr_rate_per_min, microexpr_max_intensity, microexpr_types_distribution, microexpr_timestamps, microexpr_types
Детекция быстрых вспышек AU (0.03-0.5s):

```py
# Сглаживание и поиск пиков
smoothed = gaussian_filter1d(combined_intensity, sigma=0.05*fps)
threshold = np.mean(smoothed) + 1.5 * np.std(smoothed)
peaks, properties = find_peaks(
    smoothed,
    height=threshold,
    distance=int(0.2*fps),  # min_distance
    width=(1, int(0.5*fps))  # max_duration
)

# Типы micro-expressions через комбинации AU
microexpr_types = {
    'smile': ['AU06', 'AU12'],
    'surprise': ['AU01', 'AU02', 'AU25', 'AU26'],
    'frown': ['AU04', 'AU15'],
    'disgust': ['AU09', 'AU10']
}

features['microexpr_count'] = len(peaks)
features['microexpr_rate_per_min'] = len(peaks) / duration_minutes
features['microexpr_max_intensity'] = float(smoothed[peaks].max())
features['microexpr_types_distribution'] = count_by_type(peaks, microexpr_types)
features['microexpr_timestamps'] = (peaks / fps).tolist()
```

#### 6. Per-Frame Vectors для VisualTransformer

#### 6.1. per_frame_vectors
Компактный вектор (~18-22 числа) на кадр:

```py
per_frame_vectors = []
for idx in range(n_frames):
    vec = [
        idx / n_frames,  # time_norm
        face_presence_flag,
        au12_intensity_delta,
        au6_intensity_delta,
        au4_intensity_delta,
        au25_presence_rate_short_window,
        blink_flag,
        pose_Ry_norm,
        pose_Rx_norm,
        gaze_centered_flag,
        gaze_x_norm,
        gaze_y_norm,
        mouth_opening_norm,
        face_asymmetry_score,
        microexpr_recent_count,
        au_pca_1, au_pca_2, au_pca_3,
        au_quality_flag
    ]
    per_frame_vectors.append(vec)
```

#### 7. Reliability Flags и Quality Scores

#### 7.1. - 7.5. au_quality_overall, landmark_visibility_mean, occlusion_flag, lighting_flag
Флаги надёжности для downstream моделей:

```py
# AU quality
au_quality_overall = float(df[[col for col in df.columns if col.endswith('_c')]].mean().mean())
features['au_quality_overall'] = au_quality_overall
features['au_quality_reliable'] = au_quality_overall > 0.5

# Landmark visibility
visible_count = sum((df[col] != 0.0).sum() for col in landmark_cols)
landmark_visibility_mean = visible_count / (len(landmark_cols) * len(df))
features['landmark_visibility_mean'] = landmark_visibility_mean
features['landmark_visibility_reliable'] = landmark_visibility_mean > 0.8
features['occlusion_flag'] = landmark_visibility_mean < 0.7
```