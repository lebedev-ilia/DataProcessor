"""
Object Detection Module using OWL-ViT.

This module detects objects in video frames using OWL-ViT (Vision Transformer for Open-Vocabulary Object Detection).
It follows the BaseModule interface and integrates with the core architecture.
"""

from typing import Any, Dict, List, Optional, Union, Tuple
import cv2
import numpy as np
import torch
from PIL import Image
from collections import defaultdict
from dataclasses import dataclass
from transformers import (
    OwlViTProcessor,
    OwlViTForObjectDetection,
    Owlv2Processor,
    Owlv2ForObjectDetection,
)

from core.base_module import BaseModule


@dataclass
class TrackedObject:
    """Represents a tracked object across frames."""
    track_id: int
    label: str
    first_frame: int
    last_frame: int
    frames: List[int]  # List of frame indices where object was detected
    bboxes: List[List[float]]  # Bounding boxes for each frame
    scores: List[float]  # Confidence scores
    center_positions: List[Tuple[float, float]]  # Center (x, y) for each frame
    colors: Optional[List[Tuple[int, int, int]]] = None  # Dominant colors
    semantic_tags: Optional[List[str]] = None  # Semantic tags (luxury, danger, etc.)


class SimpleTracker:
    """Simple IoU-based object tracker."""
    
    def __init__(self, iou_threshold: float = 0.3, max_age: int = 5):
        """
        Initialize tracker.
        
        :param iou_threshold: Minimum IoU to match detections to tracks
        :param max_age: Maximum frames a track can be missing before deletion
        """
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.tracks: Dict[int, TrackedObject] = {}
        self.next_id = 1
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection over Union (IoU) between two boxes."""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        # Calculate intersection
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        
        # Calculate union
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area
    
    def _get_center(self, bbox: List[float]) -> Tuple[float, float]:
        """Get center point of bounding box."""
        x_min, y_min, x_max, y_max = bbox
        return ((x_min + x_max) / 2, (y_min + y_max) / 2)
    
    def update(self, detections: List[Dict[str, Any]], frame_idx: int) -> List[Dict[str, Any]]:
        """
        Update tracker with new detections.
        
        :param detections: List of detections with bbox, score, label
        :param frame_idx: Current frame index
        :return: List of detections with added track_id
        """
        # Mark all tracks as unmatched
        unmatched_tracks = set(self.tracks.keys())
        matched_detections = set()
        
        # Try to match detections to existing tracks
        for det_idx, det in enumerate(detections):
            if det_idx in matched_detections:
                continue
            
            best_iou = 0.0
            best_track_id = None
            
            for track_id, track in self.tracks.items():
                if track.label != det["label"]:
                    continue  # Only match same label
                if track_id not in unmatched_tracks:
                    continue
                
                # Calculate IoU with last known bbox
                if track.bboxes:
                    last_bbox = track.bboxes[-1]
                    iou = self._calculate_iou(det["bbox"], last_bbox)
                    if iou > best_iou and iou >= self.iou_threshold:
                        best_iou = iou
                        best_track_id = track_id
            
            if best_track_id is not None:
                # Match found
                track = self.tracks[best_track_id]
                track.last_frame = frame_idx
                track.frames.append(frame_idx)
                track.bboxes.append(det["bbox"])
                track.scores.append(det["score"])
                track.center_positions.append(self._get_center(det["bbox"]))
                det["track_id"] = best_track_id
                unmatched_tracks.remove(best_track_id)
                matched_detections.add(det_idx)
        
        # Create new tracks for unmatched detections
        for det_idx, det in enumerate(detections):
            if det_idx not in matched_detections:
                track_id = self.next_id
                self.next_id += 1
                track = TrackedObject(
                    track_id=track_id,
                    label=det["label"],
                    first_frame=frame_idx,
                    last_frame=frame_idx,
                    frames=[frame_idx],
                    bboxes=[det["bbox"]],
                    scores=[det["score"]],
                    center_positions=[self._get_center(det["bbox"])]
                )
                self.tracks[track_id] = track
                det["track_id"] = track_id
        
        # Remove old tracks that haven't been seen for max_age frames
        tracks_to_remove = []
        for track_id, track in self.tracks.items():
            if frame_idx - track.last_frame > self.max_age:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            del self.tracks[track_id]
        
        return detections
    
    def get_all_tracks(self) -> List[TrackedObject]:
        """Get all active and completed tracks."""
        return list(self.tracks.values())


class ObjectDetectionModule(BaseModule):
    """
    Object detection module using OWL-ViT.
    
    Detects objects in video frames based on text queries.
    Integrates with ModelRegistry, FrameReader, and ResultStore.
    """

    name = "objects/detection"

    def __init__(
        self,
        model_name: str = "google/owlvit-base-patch16",
        model_family: str = "owlvit",  # "owlvit" or "owlv2"
        device: Optional[str] = None,
        default_categories: Optional[List[str]] = None,
        box_threshold: float = 0.3,
        logger=None,
    ):
        """
        Initialize the object detection module.

        :param model_name: Hugging Face model name or path
        :param model_family: "owlvit" or "owlv2"
        :param device: "cuda" / "cpu" / None (auto-detect)
        :param default_categories: Default object categories to detect
        :param box_threshold: Confidence threshold for bounding boxes
        :param logger: Optional logger function
        """
        super().__init__(logger=logger)
        
        self.model_name = model_name
        self.model_family = model_family
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.box_threshold = box_threshold
        
        # Default categories if not specified
        self.default_categories = default_categories or [
            "person",
            "car",
            "truck",
            "bicycle",
            "motorcycle",
            "bus",
            "train",
            "airplane",
            "boat",
            "traffic light",
            "stop sign",
            "parking meter",
            "bench",
            "bird",
            "cat",
            "dog",
            "horse",
            "sheep",
            "cow",
            "elephant",
            "bear",
            "zebra",
            "giraffe",
            "backpack",
            "umbrella",
            "handbag",
            "tie",
            "suitcase",
            "frisbee",
            "skis",
            "snowboard",
            "sports ball",
            "kite",
            "baseball bat",
            "baseball glove",
            "skateboard",
            "surfboard",
            "tennis racket",
            "bottle",
            "wine glass",
            "cup",
            "fork",
            "knife",
            "spoon",
            "bowl",
            "banana",
            "apple",
            "sandwich",
            "orange",
            "broccoli",
            "carrot",
            "hot dog",
            "pizza",
            "donut",
            "cake",
            "chair",
            "couch",
            "potted plant",
            "bed",
            "dining table",
            "toilet",
            "tv",
            "laptop",
            "mouse",
            "remote",
            "keyboard",
            "cell phone",
            "microwave",
            "oven",
            "toaster",
            "sink",
            "refrigerator",
            "book",
            "clock",
            "vase",
            "scissors",
            "teddy bear",
            "hair drier",
            "toothbrush",
        ]
        
        # Model will be loaded lazily via ModelRegistry
        self._model = None
        self._processor = None
        self._model_loaded = False
        
        # Tracking and advanced features
        self.enable_tracking = True
        self.enable_brand_detection = True
        self.enable_semantic_tags = True
        self.enable_attributes = True
        
        # Brand and semantic queries
        self.brand_queries = [
            "nike logo", "adidas logo", "apple logo", "samsung logo", "coca cola logo",
            "pepsi logo", "mcdonalds logo", "starbucks logo", "google logo", "microsoft logo",
            "amazon logo", "facebook logo", "instagram logo", "youtube logo", "tesla logo",
            "bmw logo", "mercedes logo", "audi logo", "toyota logo", "ford logo",
            "gucci logo", "prada logo", "versace logo", "chanel logo", "louis vuitton logo"
        ]
        
        self.semantic_queries = {
            "luxury": ["luxury car", "luxury watch", "luxury bag", "luxury item", "expensive item"],
            "danger": ["knife", "gun", "weapon", "dangerous object", "sharp object"],
            "cute": ["cute animal", "cute toy", "cute pet", "adorable"],
            "sport": ["sports equipment", "athletic gear", "fitness equipment"],
            "food": ["food", "meal", "dish", "cuisine"],
            "technology": ["electronic device", "gadget", "tech device", "smartphone", "laptop"]
        }

    def _load_model(self):
        """Load the OWL-ViT model and processor."""
        if self._model_loaded:
            return

        self.log(f"Loading OWL-ViT model: {self.model_name} ({self.model_family})")
        
        if self.model_family == "owlv2":
            if self.model_name == "google/owlvit-base-patch16":
                self.model_name = "google/owlv2-base-patch16"
            self._processor = Owlv2Processor.from_pretrained(self.model_name)
            self._model = Owlv2ForObjectDetection.from_pretrained(self.model_name).to(self.device)
        else:
            self._processor = OwlViTProcessor.from_pretrained(self.model_name)
            self._model = OwlViTForObjectDetection.from_pretrained(self.model_name).to(self.device)
        
        self._model.eval()
        self._model_loaded = True
        self.log(f"Model loaded on device: {self.device}")

    def _prepare_text_queries(self, text_queries: Optional[Union[str, List[str]]]) -> List[str]:
        """Prepare text queries for OWL-ViT."""
        if text_queries is None:
            return self.default_categories.copy()
        if isinstance(text_queries, str):
            return [q.strip() for q in text_queries.split(",")]
        return text_queries

    def _detect_objects_in_frame(
        self,
        frame: np.ndarray,
        text_queries: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Detect objects in a single frame.

        :param frame: BGR numpy array (OpenCV format)
        :param text_queries: List of object categories to detect
        :return: List of detections with bbox, score, label
        """
        if frame is None:
            return []

        # Convert BGR to RGB
        if frame.ndim == 3:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            rgb = frame
        
        image = Image.fromarray(rgb)
        text_queries = self._prepare_text_queries(text_queries)

        # Process inputs
        inputs = self._processor(text=text_queries, images=image, return_tensors="pt").to(self.device)

        # Run inference
        with torch.no_grad():
            outputs = self._model(**inputs)

        # Post-process results
        target_sizes = torch.tensor([image.size[::-1]]).to(self.device)
        results = self._processor.post_process_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=self.box_threshold
        )

        detections = []
        if results and len(results) > 0:
            result = results[0]
            boxes = result.get("boxes", torch.tensor([]))
            scores = result.get("scores", torch.tensor([]))
            labels = result.get("labels", torch.tensor([]))

            # Convert tensors to CPU
            if isinstance(boxes, torch.Tensor):
                boxes = boxes.cpu()
            if isinstance(scores, torch.Tensor):
                scores = scores.cpu()
            if isinstance(labels, torch.Tensor):
                labels = labels.cpu()

            # Map labels to category names
            for box, score, label_idx in zip(boxes, scores, labels):
                score_val = float(score)
                if score_val < self.box_threshold:
                    continue

                label_idx_int = int(label_idx)
                if 0 <= label_idx_int < len(text_queries):
                    label_name = text_queries[label_idx_int]
                else:
                    label_name = f"class_{label_idx_int}"

                x_min, y_min, x_max, y_max = box.tolist() if isinstance(box, torch.Tensor) else box
                detections.append({
                    "bbox": [float(x_min), float(y_min), float(x_max), float(y_max)],
                    "score": score_val,
                    "label": label_name,
                })

        return detections
    
    def _extract_color_from_bbox(self, frame: np.ndarray, bbox: List[float], k: int = 3) -> Tuple[int, int, int]:
        """
        Extract dominant color from object region.
        
        :param frame: BGR frame
        :param bbox: [x_min, y_min, x_max, y_max]
        :param k: Number of dominant colors to extract
        :return: Dominant color as (B, G, R)
        """
        x_min, y_min, x_max, y_max = [int(coord) for coord in bbox]
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(frame.shape[1], x_max)
        y_max = min(frame.shape[0], y_max)
        
        if x_max <= x_min or y_max <= y_min:
            return (128, 128, 128)  # Default gray
        
        roi = frame[y_min:y_max, x_min:x_max]
        if roi.size == 0:
            return (128, 128, 128)
        
        # Reshape to list of pixels
        pixels = roi.reshape(-1, 3)
        
        # Use K-means to find dominant color
        try:
            from sklearn.cluster import KMeans
            kmeans = KMeans(n_clusters=min(k, len(pixels)), n_init=10, random_state=42)
            kmeans.fit(pixels)
            dominant_color = kmeans.cluster_centers_[0].astype(int)
            return tuple(dominant_color.tolist())
        except ImportError:
            # Fallback: use mean color
            mean_color = np.mean(pixels, axis=0).astype(int)
            return tuple(mean_color.tolist())
    
    def _detect_semantic_tags(self, detections: List[Dict[str, Any]], frame: np.ndarray) -> List[Dict[str, Any]]:
        """
        Add semantic tags to detections based on label and visual features.
        
        :param detections: List of detections
        :param frame: Current frame
        :return: Detections with added semantic_tags
        """
        for det in detections:
            tags = []
            label_lower = det["label"].lower()
            
            # Check semantic categories
            for semantic, keywords in self.semantic_queries.items():
                if any(keyword in label_lower for keyword in keywords):
                    tags.append(semantic)
            
            # Additional heuristics
            if any(word in label_lower for word in ["car", "vehicle", "motorcycle"]):
                # Check if it's luxury (simplified - could use additional model)
                if det.get("score", 0) > 0.7:  # High confidence might indicate quality
                    tags.append("luxury")
            
            if any(word in label_lower for word in ["knife", "gun", "weapon"]):
                tags.append("danger")
            
            if any(word in label_lower for word in ["cat", "dog", "puppy", "kitten", "teddy"]):
                tags.append("cute")
            
            det["semantic_tags"] = tags
        
        return detections
    
    def _calculate_overlap(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Calculate overlap ratio between two bounding boxes."""
        return self._calculate_iou(bbox1, bbox2)
    
    def _calculate_density(self, detections: List[Dict[str, Any]], frame_width: int, frame_height: int) -> float:
        """
        Calculate object density (objects per unit area).
        
        :param detections: List of detections
        :param frame_width: Frame width
        :param frame_height: Frame height
        :return: Density value
        """
        if frame_width == 0 or frame_height == 0:
            return 0.0
        
        total_area = frame_width * frame_height
        if total_area == 0:
            return 0.0
        
        # Calculate total area covered by objects
        covered_area = 0.0
        for det in detections:
            bbox = det["bbox"]
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            covered_area += area
        
        # Density as ratio of covered area to total area
        density = covered_area / total_area
        return density
    
    def _calculate_overlapping_objects(self, detections: List[Dict[str, Any]], overlap_threshold: float = 0.1) -> Dict[str, Any]:
        """
        Calculate overlapping objects statistics.
        
        :param detections: List of detections
        :param overlap_threshold: Minimum IoU to consider objects overlapping
        :return: Statistics about overlapping
        """
        num_overlaps = 0
        overlap_pairs = []
        
        for i, det1 in enumerate(detections):
            for j, det2 in enumerate(detections[i+1:], start=i+1):
                iou = self._calculate_iou(det1["bbox"], det2["bbox"])
                if iou >= overlap_threshold:
                    num_overlaps += 1
                    overlap_pairs.append({
                        "object1": det1["label"],
                        "object2": det2["label"],
                        "iou": iou
                    })
        
        return {
            "num_overlapping_pairs": num_overlaps,
            "overlap_pairs": overlap_pairs,
            "overlap_ratio": num_overlaps / max(1, len(detections) * (len(detections) - 1) / 2)
        }
    
    def _calculate_iou(self, box1: List[float], box2: List[float]) -> float:
        """Calculate Intersection over Union (IoU) between two boxes."""
        x1_min, y1_min, x1_max, y1_max = box1
        x2_min, y2_min, x2_max, y2_max = box2
        
        inter_x_min = max(x1_min, x2_min)
        inter_y_min = max(y1_min, y2_min)
        inter_x_max = min(x1_max, x2_max)
        inter_y_max = min(y1_max, y2_max)
        
        if inter_x_max < inter_x_min or inter_y_max < inter_y_min:
            return 0.0
        
        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        box1_area = (x1_max - x1_min) * (y1_max - y1_min)
        box2_area = (x2_max - x2_min) * (y2_max - y2_min)
        union_area = box1_area + box2_area - inter_area
        
        if union_area == 0:
            return 0.0
        
        return inter_area / union_area

    def should_run(self, state: Dict[str, Any]) -> bool:
        """Check if module should run based on state."""
        # Check if sampler_result exists and has frames
        sampler_result = state.get("sampler_result", {})
        if not sampler_result:
            self.log("No sampler_result in state, skipping")
            return False
        
        # Check if we have any frame indices to process
        # Look for "global" or any other strategy
        has_frames = False
        for strategy, frames in sampler_result.items():
            if frames and len(frames) > 0:
                has_frames = True
                break
        
        if not has_frames:
            self.log("No frames to process, skipping")
            return False
        
        return True

    def run(self, state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Main execution method.

        :param state: State dictionary containing:
            - video_context: VideoContext with video metadata
            - sampler_result: Dict with frame indices per strategy
            - frame_reader: FrameReader instance (optional, will create if not present)
            - model_registry: ModelRegistry instance (optional)
            - result_store: ResultStore instance (optional)
        :return: Dictionary with results or None
        """
        self.log("Starting object detection")

        # Get components from state
        video_context = state.get("video_context")
        sampler_result = state.get("sampler_result", {})
        frame_reader = state.get("frame_reader")
        model_registry = state.get("model_registry")
        result_store = state.get("result_store")

        if video_context is None:
            self.log("ERROR: video_context not found in state")
            return None

        # Get video path from context
        video_path = getattr(video_context, "video_path", None)
        if not video_path:
            self.log("ERROR: video_path not found in video_context")
            return None

        # Create or use FrameReader
        if frame_reader is None:
            from core.frame_reader import FrameReader
            frame_reader = FrameReader(video_path)
            state["frame_reader"] = frame_reader

        # Load model (via ModelRegistry if available, otherwise directly)
        if model_registry:
            model_key = f"owlvit-{self.model_name}"
            try:
                if not model_registry.registry.get(model_key):
                    def load_model():
                        self._load_model()
                        return {
                            "model": self._model,
                            "processor": self._processor,
                        }
                    model_registry.register_model(model_key, load_model)
                
                model_data = model_registry.get_model(model_key, device=self.device)
                self._model = model_data["model"]
                self._processor = model_data["processor"]
            except Exception as e:
                self.log(f"ERROR loading model from registry: {e}, loading directly")
                self._load_model()
        else:
            self._load_model()

        # Get frame indices from sampler_result
        # Use "global" strategy by default, or first available
        frame_indices = []
        for strategy in ["global", "GLOBAL"]:
            if strategy in sampler_result:
                frames_data = sampler_result[strategy]
                # Extract indices from frame descriptors
                for frame_desc in frames_data:
                    if isinstance(frame_desc, dict):
                        if "idx" in frame_desc:
                            frame_indices.append(int(frame_desc["idx"]))
                        elif "frame_index" in frame_desc:
                            frame_indices.append(int(frame_desc["frame_index"]))
                    elif isinstance(frame_desc, int):
                        frame_indices.append(frame_desc)
                break
        
        # If no indices found, try to get from any strategy
        if not frame_indices:
            for strategy, frames_data in sampler_result.items():
                for frame_desc in frames_data:
                    if isinstance(frame_desc, dict):
                        if "idx" in frame_desc:
                            frame_indices.append(int(frame_desc["idx"]))
                        elif "frame_index" in frame_desc:
                            frame_indices.append(int(frame_desc["frame_index"]))
                    elif isinstance(frame_desc, int):
                        frame_indices.append(frame_desc)
                if frame_indices:
                    break

        if not frame_indices:
            self.log("WARNING: No frame indices found in sampler_result")
            return None

        self.log(f"Processing {len(frame_indices)} frames")

        # Get video FPS for duration calculations
        fps = getattr(video_context, "fps", 30)
        frame_time = 1.0 / fps if fps > 0 else 1.0 / 30.0

        # Read frames
        frames = []
        for idx in frame_indices:
            try:
                frame = frame_reader.read_frame(idx)
                frames.append(frame)
            except Exception as e:
                self.log(f"ERROR reading frame {idx}: {e}")
                frames.append(None)

        # Initialize tracker
        tracker = SimpleTracker(iou_threshold=0.3, max_age=5) if self.enable_tracking else None

        # Prepare text queries (include brands if enabled)
        text_queries = self.default_categories.copy()
        if self.enable_brand_detection:
            text_queries.extend(self.brand_queries)

        # Detect objects in each frame
        all_detections = {}
        frame_metadata = {}
        
        for frame_idx, frame in zip(frame_indices, frames):
            if frame is None:
                all_detections[frame_idx] = []
                continue
            
            try:
                detections = self._detect_objects_in_frame(frame, text_queries=text_queries)
                
                # Add semantic tags
                if self.enable_semantic_tags:
                    detections = self._detect_semantic_tags(detections, frame)
                
                # Extract attributes (color)
                if self.enable_attributes:
                    for det in detections:
                        color = self._extract_color_from_bbox(frame, det["bbox"])
                        det["color"] = {"B": int(color[0]), "G": int(color[1]), "R": int(color[2])}
                
                # Update tracker
                if tracker:
                    detections = tracker.update(detections, frame_idx)
                
                all_detections[frame_idx] = detections
                
                # Calculate frame-level metrics
                frame_height, frame_width = frame.shape[:2]
                density = self._calculate_density(detections, frame_width, frame_height)
                overlap_stats = self._calculate_overlapping_objects(detections)
                
                frame_metadata[frame_idx] = {
                    "density": density,
                    "num_objects": len(detections),
                    "overlapping": overlap_stats
                }
                
            except Exception as e:
                self.log(f"ERROR detecting objects in frame {frame_idx}: {e}")
                all_detections[frame_idx] = []
                frame_metadata[frame_idx] = {
                    "density": 0.0,
                    "num_objects": 0,
                    "overlapping": {"num_overlapping_pairs": 0, "overlap_pairs": [], "overlap_ratio": 0.0}
                }

        # Get tracking results
        tracks = []
        if tracker:
            tracks = tracker.get_all_tracks()
            # Add color and semantic tags to tracks
            for track in tracks:
                if track.frames and frames:
                    # Get color from first frame
                    first_frame_idx = track.frames[0]
                    if first_frame_idx in frame_indices:
                        frame_idx_pos = frame_indices.index(first_frame_idx)
                        if frame_idx_pos < len(frames) and frames[frame_idx_pos] is not None:
                            frame = frames[frame_idx_pos]
                            if track.bboxes:
                                color = self._extract_color_from_bbox(frame, track.bboxes[0])
                                track.colors = [color]
                
                # Add semantic tags
                semantic_tags = []
                for semantic, keywords in self.semantic_queries.items():
                    if any(keyword in track.label.lower() for keyword in keywords):
                        semantic_tags.append(semantic)
                track.semantic_tags = semantic_tags

        # Calculate tracking metrics
        tracking_metrics = {}
        if tracks:
            # Duration of presence
            track_durations = {}
            track_durations_seconds = {}
            for track in tracks:
                duration_frames = track.last_frame - track.first_frame + 1
                duration_seconds = duration_frames * frame_time
                track_durations[track.track_id] = duration_frames
                track_durations_seconds[track.track_id] = duration_seconds
            
            # Object turnover rate (birth/death events)
            birth_events = {}  # frame -> list of track_ids
            death_events = {}  # frame -> list of track_ids
            
            for track in tracks:
                # Birth event
                if track.first_frame not in birth_events:
                    birth_events[track.first_frame] = []
                birth_events[track.first_frame].append(track.track_id)
                
                # Death event (if track ended)
                if track.last_frame < frame_indices[-1]:
                    death_frame = track.last_frame + 1
                    if death_frame not in death_events:
                        death_events[death_frame] = []
                    death_events[death_frame].append(track.track_id)
            
            # Calculate turnover rate
            total_births = sum(len(tracks) for tracks in birth_events.values())
            total_deaths = sum(len(tracks) for tracks in death_events.values())
            avg_objects_per_frame = len(tracks) / max(1, len(frame_indices))
            turnover_rate = (total_births + total_deaths) / max(1, len(frame_indices)) if frame_indices else 0.0
            
            tracking_metrics = {
                "num_tracks": len(tracks),
                "track_durations_frames": track_durations,
                "track_durations_seconds": track_durations_seconds,
                "avg_duration_frames": np.mean(list(track_durations.values())) if track_durations else 0.0,
                "avg_duration_seconds": np.mean(list(track_durations_seconds.values())) if track_durations_seconds else 0.0,
                "birth_events": {str(k): v for k, v in birth_events.items()},
                "death_events": {str(k): v for k, v in death_events.items()},
                "total_births": total_births,
                "total_deaths": total_deaths,
                "turnover_rate": turnover_rate,
                "avg_objects_per_frame": avg_objects_per_frame
            }
        
        # Aggregate results
        object_counts = {}
        total_detections = 0
        semantic_tag_counts = defaultdict(int)
        brand_detections = []
        
        for frame_idx, detections in all_detections.items():
            for det in detections:
                label = det["label"]
                object_counts[label] = object_counts.get(label, 0) + 1
                total_detections += 1
                
                # Count semantic tags
                if "semantic_tags" in det:
                    for tag in det["semantic_tags"]:
                        semantic_tag_counts[tag] += 1
                
                # Check for brands
                if self.enable_brand_detection and "logo" in label.lower():
                    brand_detections.append({
                        "frame": frame_idx,
                        "brand": label,
                        "score": det.get("score", 0.0),
                        "bbox": det.get("bbox", [])
                    })

        # Calculate average density and overlapping
        avg_density = np.mean([meta["density"] for meta in frame_metadata.values()]) if frame_metadata else 0.0
        avg_overlap_ratio = np.mean([meta["overlapping"]["overlap_ratio"] for meta in frame_metadata.values()]) if frame_metadata else 0.0

        # Prepare result
        result = {
            "object_detections": {
                "frames": all_detections,  # Per-frame detections
                "frame_metadata": frame_metadata,  # Density, overlapping per frame
                "summary": {
                    "total_detections": total_detections,
                    "unique_categories": len(object_counts),
                    "category_counts": object_counts,
                    "semantic_tag_counts": dict(semantic_tag_counts),
                    "brand_detections": brand_detections,
                    "avg_density": float(avg_density),
                    "avg_overlap_ratio": float(avg_overlap_ratio),
                },
                "tracking": tracking_metrics if tracking_metrics else None,
                "tracks": [
                    {
                        "track_id": track.track_id,
                        "label": track.label,
                        "first_frame": track.first_frame,
                        "last_frame": track.last_frame,
                        "duration_frames": track.last_frame - track.first_frame + 1,
                        "duration_seconds": (track.last_frame - track.first_frame + 1) * frame_time,
                        "num_detections": len(track.frames),
                        "semantic_tags": track.semantic_tags or [],
                        "colors": [{"B": int(c[0]), "G": int(c[1]), "R": int(c[2])} for c in (track.colors or [])]
                    }
                    for track in tracks
                ] if tracks else [],
                "frame_count": len(frame_indices),
            }
        }

        # Store results in ResultStore if available
        if result_store:
            result_store.store(
                self.name,
                result["object_detections"],
                level="video"
            )
            
            # Also store per-frame results
            for frame_idx, detections in all_detections.items():
                result_store.store(
                    self.name,
                    detections,
                    level="frame",
                    key=frame_idx
                )

        self.log(f"Object detection completed: {total_detections} detections across {len(frame_indices)} frames")
        
        # Release model from registry if used
        if model_registry:
            try:
                model_registry.release_model(f"owlvit-{self.model_name}")
            except:
                pass

        return result

