"""
Object Detection Module using OWL-ViT / OWLv2.

Улучшенная и более устойчивая версия оригинального модуля.
Поддерживает:
 - lazy loading модели (OWL-ViT / OWLv2)
 - open-vocabulary detection через text queries
 - brand detection, semantic tags, базовые атрибуты (цвет)
 - трекинг (скелет для расширения)
 - корректную обработку входных изображений (cv2 numpy -> PIL)
 - безопасный постпроцессинг и обрезку bbox
"""
import time
from typing import Any, Dict, List, Optional, Union, Tuple
import cv2
import numpy as np
import torch
from PIL import Image
from collections import defaultdict
from dataclasses import dataclass, field
import logging

# Try/except импорта трансформеров - предполагается что transformers установлен
from transformers import (
    OwlViTProcessor,
    OwlViTForObjectDetection,
    Owlv2Processor,
    Owlv2ForObjectDetection,
)


from utils.logger import get_logger
logger = get_logger("ObjectDetectionModule")


@dataclass
class TrackedObject:
    """Represents a tracked object across frames."""
    track_id: int
    label: str
    first_frame: int
    last_frame: int
    frames: List[int] = field(default_factory=list)
    bboxes: List[List[float]] = field(default_factory=list)
    scores: List[float] = field(default_factory=list)
    center_positions: List[Tuple[float, float]] = field(default_factory=list)
    colors: Optional[List[Tuple[int, int, int]]] = None
    semantic_tags: Optional[List[str]] = None


class ObjectDetectionModule:
    """
    Object detection module using OWL-ViT / OWLv2.

    Основные улучшения:
      - стабильная работа с PIL (processor требует PIL)
      - корректное вычисление target_sizes из numpy
      - безопасный постпроцессинг и обрезка bbox по краям изображения
      - fallback для извлечения цвета, если sklearn не установлен
    """

    name = "objects/detection"

    def __init__(
        self,
        model_name: str = "google/owlv2-large-patch14",
        model_family: str = "owlv2",  # "owlvit" or "owlv2"
        device: Optional[str] = None,
        default_categories: Optional[List[str]] = None,
        box_threshold: float = 0.1,
        enable_tracking: bool = True,
        enable_brand_detection: bool = True,
        enable_semantic_tags: bool = True,
        enable_attributes: bool = True,
    ):
        self.model_name = model_name
        self.model_family = model_family.lower()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.box_threshold = float(box_threshold)

        # Default COCO-like categories (kept from original)
        self.default_categories = default_categories or [
            "person", "car", "truck", "bicycle", "motorcycle", "bus", "train", "airplane",
            "boat", "traffic light", "stop sign", "bench", "bird", "cat", "dog", "horse",
            "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
            "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
            "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
            "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
            "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
            "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush",
        ]

        self.brand_queries = [
            "nike logo", "adidas logo", "apple logo", "samsung logo", "coca cola logo",
            "pepsi logo", "mcdonalds logo", "starbucks logo", "google logo", "microsoft logo",
            "amazon logo", "facebook logo", "instagram logo", "youtube logo", "tesla logo",
            "bmw logo", "mercedes logo", "audi logo", "toyota logo", "ford logo",
            "gucci logo", "prada logo", "versace logo", "chanel logo", "louis vuitton logo"
        ]

        self.semantic_queries = {
            "luxury": ["luxury", "expensive", "premium", "luxury car", "luxury watch"],
            "danger": ["knife", "gun", "weapon", "dangerous"],
            "cute": ["cute", "adorable", "puppy", "kitten", "teddy"],
            "sport": ["sports", "athletic", "fitness", "equipment"],
            "food": ["food", "meal", "dish", "cuisine"],
            "technology": ["electronic device", "gadget", "smartphone", "laptop"]
        }

        # Flags
        self.enable_tracking = enable_tracking
        self.enable_brand_detection = enable_brand_detection
        self.enable_semantic_tags = enable_semantic_tags
        self.enable_attributes = enable_attributes

        # Lazy-loaded objects
        self._model = None
        self._processor = None
        self._model_loaded = False

        logger.info(
            "ObjectDetectionModule initialized: model=%s family=%s device=%s threshold=%s",
            self.model_name, self.model_family, self.device, self.box_threshold
        )

    def _load_model(self) -> None:
        """Lazy load the processor and the model. Safe to call multiple times."""
        if self._model_loaded:
            return

        logger.info("Loading model %s (family=%s) on device=%s", self.model_name, self.model_family, self.device)

        # Automatic model name mapping for common older name
        if self.model_family == "owlv2":
            # map old accidental name to owlv2 if necessary
            if self.model_name == "google/owlvit-base-patch16":
                self.model_name = "google/owlv2-base-patch16"
            self._processor = Owlv2Processor.from_pretrained(self.model_name)
            self._model = Owlv2ForObjectDetection.from_pretrained(self.model_name).to(self.device)
        else:
            self._processor = OwlViTProcessor.from_pretrained(self.model_name)
            self._model = OwlViTForObjectDetection.from_pretrained(self.model_name).to(self.device)

        self._model.eval()
        self._model_loaded = True
        logger.info("Model loaded successfully")

    def _prepare_text_queries(self, text_queries: Optional[Union[str, List[str]]]) -> List[str]:
        """Normalize text_queries into a list of trimmed strings."""
        if text_queries is None:
            return list(self.default_categories)
        if isinstance(text_queries, str):
            # allow comma-separated string
            items = [q.strip() for q in text_queries.split(",") if q.strip()]
            return items or list(self.default_categories)
        # already a list
        return [str(q).strip() for q in text_queries if str(q).strip()]

    def _clamp_bbox(self, bbox: List[float], width: int, height: int) -> List[float]:
        """Clamp bbox coordinates to image boundaries and ensure valid box."""
        x_min, y_min, x_max, y_max = map(float, bbox)
        x_min = max(0.0, min(x_min, width - 1.0))
        x_max = max(0.0, min(x_max, width - 1.0))
        y_min = max(0.0, min(y_min, height - 1.0))
        y_max = max(0.0, min(y_max, height - 1.0))
        # Ensure min < max
        if x_max <= x_min or y_max <= y_min:
            # return degenerate box (handled by caller)
            return [0.0, 0.0, 0.0, 0.0]
        return [x_min, y_min, x_max, y_max]

    def _extract_color_from_bbox(self, frame: np.ndarray, bbox: List[float], k: int = 3) -> Tuple[int, int, int]:
        """
        Extract dominant color from the BGR ROI.
        Fallback to mean color if sklearn.cluster.KMeans not available.
        """
        x_min, y_min, x_max, y_max = [int(round(coord)) for coord in bbox]
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(frame.shape[1], x_max)
        y_max = min(frame.shape[0], y_max)

        if x_max <= x_min or y_max <= y_min:
            return (128, 128, 128)

        roi = frame[y_min:y_max, x_min:x_max]
        if roi.size == 0:
            return (128, 128, 128)

        pixels = roi.reshape(-1, 3).astype(np.float32)

        # # Try sklearn KMeans for better dominant color
        # try:
        #     from sklearn.cluster import KMeans
        #     n_clusters = min(k, len(pixels))
        #     if n_clusters <= 0:
        #         mean_color = np.mean(pixels, axis=0).astype(int)
        #         return tuple(int(c) for c in mean_color.tolist())
        #     kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
        #     kmeans.fit(pixels)
        #     centers = kmeans.cluster_centers_.astype(int)
        #     # choose the largest cluster center (first one is fine, but be explicit)
        #     labels, counts = np.unique(kmeans.labels_, return_counts=True)
        #     dominant_index = labels[np.argmax(counts)]
        #     dominant_color = centers[dominant_index]
        #     return (int(dominant_color[0]), int(dominant_color[1]), int(dominant_color[2]))
        # except Exception:
        #     # fallback to mean color (B, G, R)
        mean_color = np.mean(pixels, axis=0).astype(int)
        return (int(mean_color[0]), int(mean_color[1]), int(mean_color[2]))

    def _detect_semantic_tags(self, detections: List[Dict[str, Any]], frame: np.ndarray) -> List[Dict[str, Any]]:
        """Enrich detections with basic semantic tags using label and heuristics."""
        for det in detections:
            tags: List[str] = []
            label_lower = str(det.get("label", "")).lower()

            for semantic, keywords in self.semantic_queries.items():
                if any(keyword in label_lower for keyword in keywords):
                    tags.append(semantic)

            # Heuristics examples
            if any(w in label_lower for w in ["car", "vehicle", "motorcycle"]):
                if det.get("score", 0.0) >= 0.75:
                    tags.append("luxury")

            if any(w in label_lower for w in ["knife", "gun", "weapon", "blade"]):
                tags.append("danger")

            if any(w in label_lower for w in ["cat", "dog", "puppy", "kitten", "teddy"]):
                tags.append("cute")

            det["semantic_tags"] = tags

        return detections

    def _detect_objects_in_frame(
        self,
        frame: np.ndarray,
        text_queries: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Detect objects in a single frame.

        :param frame: BGR numpy array (OpenCV format)
        :param text_queries: list of text queries (strings)
        :return: list of detections {bbox, score, label, ...}
        """
        if frame is None:
            return []

        # ensure model is loaded
        self._load_model()

        # Convert to RGB (from BGR OpenCV)
        if frame.ndim == 3:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            # grayscale? convert to 3-channel
            rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        # Keep original width/height from numpy to compute target_sizes
        h, w = rgb.shape[:2]

        # Use PIL Image for processor (this is critical)
        image = Image.fromarray(rgb)

        queries = self._prepare_text_queries(text_queries)
        if not queries:
            queries = list(self.default_categories)

        # Run processor + model on device
        tik = time.time()
        try:
            inputs = self._processor(text=queries, images=image, return_tensors="pt").to(self.device)
        except Exception as e:
            logger.exception("Processor failed for the given image. Falling back to CPU processor call: %s", e)
            # fallback: build inputs on CPU and then move tensors to device
            inputs = self._processor(text=queries, images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        tok = round(time.time() - tik, 2)

        with torch.no_grad():
            outputs = self._model(**inputs)

        # target_sizes must be [height, width]
        target_sizes = torch.tensor([[h, w]], dtype=torch.long).to(self.device)

        # Post-process (returns list with dict per image)
        results = self._processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=self.box_threshold
        )

        detections: List[Dict[str, Any]] = []

        if not results or len(results) == 0:
            return []

        result = results[0]
        boxes = result.get("boxes", torch.tensor([]))
        scores = result.get("scores", torch.tensor([]))
        labels = result.get("labels", torch.tensor([]))

        # Bring to CPU and numpy for iteration
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy()
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()

        # Iterate detections
        for box, score, label_idx in zip(boxes, scores, labels):
            score_val = float(score)
            if score_val < float(self.box_threshold):
                continue

            # label_idx is index into queries (sometimes long)
            try:
                label_i = int(label_idx)
                if 0 <= label_i < len(queries):
                    label_name = queries[label_i]
                else:
                    label_name = f"class_{label_i}"
            except Exception:
                label_name = str(label_idx)

            # clamp bbox
            clamped = self._clamp_bbox(box.tolist(), width=w, height=h)
            if clamped[2] <= clamped[0] or clamped[3] <= clamped[1]:
                # invalid box after clamp
                continue

            x_min, y_min, x_max, y_max = clamped

            det = {
                "bbox": [float(x_min), float(y_min), float(x_max), float(y_max)],
                "score": score_val,
                "label": label_name,
            }
            detections.append(det)

        return detections, tok

    def run(self, frame_manager, frame_indices: List[int]) -> Dict[str, Any]:
        """
        Main execution method.

        :param frame_manager: object with .fps and .get(index) -> numpy frame (BGR)
        :param frame_indices: list of frame indices to process
        :return: result dictionary containing per-frame detections and summary
        """
        import time

        if frame_manager is None:
            raise ValueError("frame_manager is None")

        fps = getattr(frame_manager, "fps", 30)
        frame_time = 1.0 / fps if fps > 0 else 1.0 / 30.0

        self._load_model()

        # Prepare text queries: base categories + brands if enabled
        queries = list(self.default_categories)
        if self.enable_brand_detection:
            queries.extend(self.brand_queries)

        all_detections: Dict[int, List[Dict[str, Any]]] = {}
        skipped = 0

        t = time.time()

        for k, frame_idx in enumerate(frame_indices):
            m_tik = time.time()
            frame = frame_manager.get(frame_idx)

            if frame is None:
                all_detections[frame_idx] = []
                skipped += 1
                continue

            try:
                tik = time.time()
                detections, pred_time = self._detect_objects_in_frame(frame, text_queries=queries)
                det_tok = round(time.time() - tik, 2)

                # Add semantic tags
                tik = time.time()
                if self.enable_semantic_tags and detections:
                    detections = self._detect_semantic_tags(detections, frame)
                sem_tok = round(time.time() - tik, 2)

                tik = time.time()
                # Extract attributes (dominant color)
                if self.enable_attributes and detections:
                    for det in detections:
                        color_bgr = self._extract_color_from_bbox(frame, det["bbox"])
                        det["color"] = {"B": int(color_bgr[0]), "G": int(color_bgr[1]), "R": int(color_bgr[2])}
                col_tok = round(time.time() - tik, 2)
                m_tok = round(time.time() - m_tik, 2)

                logger.debug(f"pred_time: {pred_time} | det_tok: {det_tok} | sem_tok: {sem_tok} | color_time: {col_tok} | all_frame_time: {m_tok}")

                all_detections[frame_idx] = detections

                # Periodic log
                if k % 20 == 0:
                    processed_nonempty = sum(1 for v in all_detections.values() if v)
                    c_t = time.time()
                    logger.info(
                        "run | processed_frames=%d | current_index=%d | skipped=%d | nonempty_frames=%d | all_time:%d",
                        k, frame_idx, skipped, processed_nonempty, round(c_t - t, 2)
                    )
                    t = c_t

            except Exception as e:
                logger.exception("Error detecting objects in frame %s: %s", frame_idx, e)
                all_detections[frame_idx] = []
                skipped += 1

        # Build summary
        object_counts: Dict[str, int] = {}
        total_detections = 0
        semantic_tag_counts = defaultdict(int)
        brand_detections: List[Dict[str, Any]] = []

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

        result = {
            "frames": all_detections,
            "summary": {
                "total_detections": total_detections,
                "unique_categories": len(object_counts),
                "category_counts": object_counts,
                "semantic_tag_counts": dict(semantic_tag_counts),
                "brand_detections": brand_detections,
            },
            "frame_count": len(frame_indices),
        }

        return result
