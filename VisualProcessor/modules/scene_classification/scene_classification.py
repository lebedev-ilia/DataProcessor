"""Scene classification extractor powered by Places365 models."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union, Tuple
from collections import defaultdict

import cv2
import numpy as np
import requests
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False

try:
    from transformers import CLIPProcessor, CLIPModel
    CLIP_AVAILABLE = True
except ImportError:
    CLIP_AVAILABLE = False
    

class Places365SceneClassifier:
    """
    Scene classifier built on top of Places365 checkpoints.

    The extractor accepts a list of OpenCV frames (BGR numpy arrays) and returns
    the top-K scene predictions per frame.
    """

    CATEGORY_URL = "https://raw.githubusercontent.com/CSAILVision/places365/master/categories_places365.txt"
    MODEL_URLS = {
        "resnet18": "http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar",
        "resnet50": "http://places2.csail.mit.edu/models_places365/resnet50_places365.pth.tar",
    }
    # Современные модели через timm (предобученные на ImageNet, можно дообучить на Places365)
    # Или использовать предобученные на Places365 из timm
    TIMM_MODELS = {
        # EfficientNet - эффективные и точные
        "efficientnet_b0": "efficientnet_b0",
        "efficientnet_b1": "efficientnet_b1",
        "efficientnet_b2": "efficientnet_b2",
        "efficientnet_b3": "efficientnet_b3",
        # ConvNeXt - современная архитектура, превосходит ResNet
        "convnext_tiny": "convnext_tiny",
        "convnext_small": "convnext_small",
        "convnext_base": "convnext_base",
        # Vision Transformers
        "vit_base_patch16_224": "vit_base_patch16_224",
        "vit_large_patch16_224": "vit_large_patch16_224",
        # RegNet - эффективные модели от Facebook
        "regnetx_002": "regnetx_002",
        "regnetx_004": "regnetx_004",
        "regnetx_006": "regnetx_006",
        # ResNet улучшенные версии
        "resnet50": "resnet50",  # через timm для лучшей оптимизации
        "resnet101": "resnet101",
    }
    DEFAULT_MEAN = [0.485, 0.456, 0.406]
    DEFAULT_STD = [0.229, 0.224, 0.225]

    def __init__(
        self,
        *,
        model_arch: str = "resnet50",
        use_timm: bool = False,
        top_k: int = 5,
        batch_size: int = 1,
        device: Optional[str] = None,
        categories_path: Optional[str] = None,
        cache_dir: Optional[str] = None,
        gpu_memory_threshold: float = 0.9,
        log_metrics_every_n_frames: int = 10,
        # Quality improvement options
        input_size: int = 224,
        use_tta: bool = False,
        use_multi_crop: bool = False,
        temporal_smoothing: bool = False,
        smoothing_window: int = 5,
        # Advanced features
        enable_advanced_features: bool = True,
        use_clip_for_semantics: bool = True,
    ) -> None:
        """
        :param model_arch: model architecture name
            - For Places365: 'resnet18', 'resnet50'
            - For timm (use_timm=True): 'efficientnet_b0', 'convnext_tiny', 'vit_base_patch16_224', etc.
        :param use_timm: use timm library for modern architectures (EfficientNet, ConvNeXt, ViT, etc.)
            If True, model will be pretrained on ImageNet (can be fine-tuned on Places365)
        :param top_k: number of predictions to return per frame
        :param batch_size: number of frames to process simultaneously
        :param device: torch device ('cuda', 'cpu', etc.), autodetected when None
        :param categories_path: optional local path to categories_places365.txt
        :param cache_dir: optional directory where helper files will be cached
        :param gpu_memory_threshold: BaseExtractor GPU memory threshold
        :param log_metrics_every_n_frames: resource logging cadence
        :param input_size: input image size (224, 256, 320, etc.). Larger = better accuracy, slower
        :param use_tta: enable Test-Time Augmentation (multiple augmentations + averaging)
        :param use_multi_crop: enable multi-crop inference (5 crops: center + 4 corners)
        :param temporal_smoothing: enable temporal smoothing for video sequences
        :param smoothing_window: window size for temporal smoothing (number of frames)
        :param enable_advanced_features: enable advanced features (indoor/outdoor, time of day, etc.)
        :param use_clip_for_semantics: use CLIP for semantic features (aesthetic, atmosphere)
        """
        super().__init__(
            gpu_memory_threshold=gpu_memory_threshold,
            log_metrics_every_n_frames=log_metrics_every_n_frames,
        )

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.top_k = max(1, top_k)
        self.batch_size = max(1, batch_size)
        self.input_size = input_size
        self.use_tta = use_tta
        self.use_multi_crop = use_multi_crop
        self.temporal_smoothing = temporal_smoothing
        self.smoothing_window = max(1, smoothing_window)
        self.use_timm = use_timm
        self.cache_dir = Path(cache_dir) if cache_dir else Path.home() / ".cache" / "places365"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Advanced features
        self.enable_advanced_features = enable_advanced_features
        self.use_clip_for_semantics = use_clip_for_semantics and CLIP_AVAILABLE
        self._clip_model = None
        self._clip_processor = None
        
        # Initialize indoor/outdoor and nature/urban mappings
        self._init_scene_mappings()

        if use_timm and not TIMM_AVAILABLE:
            raise ImportError(
                "timm library is required for modern architectures. "
                "Install it with: pip install timm"
            )

        self.categories = self._load_categories(categories_path)
        self.model = self._load_model(model_arch)
        
        # Base preprocessing (used for single inference)
        resize_size = int(self.input_size * 1.143)  # ~256 for 224, ~366 for 320
        self.preprocess = transforms.Compose(
            [
                transforms.Resize(resize_size),
                transforms.CenterCrop(self.input_size),
                transforms.ToTensor(),
                transforms.Normalize(mean=self.DEFAULT_MEAN, std=self.DEFAULT_STD),
            ]
        )
        
        # TTA augmentations (if enabled)
        if self.use_tta:
            self.tta_transforms = [
                transforms.Compose([
                    transforms.Resize(resize_size),
                    transforms.CenterCrop(self.input_size),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.DEFAULT_MEAN, std=self.DEFAULT_STD),
                ]),
                transforms.Compose([
                    transforms.Resize(resize_size),
                    transforms.CenterCrop(self.input_size),
                    transforms.RandomHorizontalFlip(p=1.0),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=self.DEFAULT_MEAN, std=self.DEFAULT_STD),
                ]),
            ]
        
        self.model.eval()
        
        # Load CLIP if needed
        if self.enable_advanced_features and self.use_clip_for_semantics:
            self._load_clip_model()

    def classify(
        self, frame_manager, frame_indices, top_k: Optional[int] = None
    ) -> List[List[Dict[str, Any]]]:
        """
        Runs scene classification over the provided frames.

        :param top_k: override default number of predictions
        :return: list where each element contains predictions dicts for a frame:
                 { "label": str, "score": float }
        """
        k = max(1, top_k or self.top_k)
        raw_predictions: List[List[Dict[str, Any]]] = [[] for _ in range(len(frame_indices))]

        # Process frames with batching
        batch_tensors: List[torch.Tensor] = []
        batch_indices: List[int] = []

        def flush_batch() -> None:
            if not batch_tensors:
                return
            batch_tensor = torch.cat(batch_tensors, dim=0)
            batch_results = self._infer_batch(batch_tensor, k)
            
            # Group results by frame index (for TTA/multi-crop averaging)
            frame_results: Dict[int, List[List[Dict[str, Any]]]] = {}
            for idx, preds in zip(batch_indices, batch_results):
                if idx not in frame_results:
                    frame_results[idx] = []
                frame_results[idx].append(preds)
            
            # Average predictions for frames with multiple inferences (TTA/multi-crop)
            for idx, pred_list in frame_results.items():
                if len(pred_list) > 1:
                    raw_predictions[idx] = self._average_predictions(pred_list, k)
                else:
                    raw_predictions[idx] = pred_list[0]
            
            batch_tensors.clear()
            batch_indices.clear()

        for frame_idx in frame_indices:
            
            frame = frame_manager.get(frame_idx)
            
            self.check_system_state(frame_idx)
            if frame is None:
                continue

            # Prepare frame(s) - may return multiple tensors for TTA/multi-crop
            tensors = self._prepare_frame(frame)
            if tensors is None:
                continue

            if isinstance(tensors, list):
                # Multiple tensors (TTA or multi-crop)
                for tensor in tensors:
                    batch_tensors.append(tensor)
                    batch_indices.append(frame_idx)
            else:
                # Single tensor
                batch_tensors.append(tensors)
                batch_indices.append(frame_idx)

            if len(batch_tensors) >= self.batch_size:
                flush_batch()

        flush_batch()

        # Apply temporal smoothing if enabled
        if self.temporal_smoothing and len(raw_predictions) > 1:
            return self._apply_temporal_smoothing(raw_predictions, k)
        
        return raw_predictions

    __call__ = classify

    def _prepare_frame(self, frame: np.ndarray) -> Optional[torch.Tensor | List[torch.Tensor]]:
        if frame is None or not isinstance(frame, np.ndarray):
            self.logger.warning("Frame is not a numpy array – skipping")
            return None

        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 4:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)

        # Multi-crop: 5 crops (center + 4 corners)
        if self.use_multi_crop:
            crops = self._get_multi_crops(image)
            tensors = []
            for crop in crops:
                tensor = self.preprocess(crop).unsqueeze(0).to(self.device)
                tensors.append(tensor)
            return tensors

        # TTA: multiple augmentations
        if self.use_tta:
            tensors = []
            for transform in self.tta_transforms:
                tensor = transform(image).unsqueeze(0).to(self.device)
                tensors.append(tensor)
            return tensors

        # Standard single inference
        tensor = self.preprocess(image).unsqueeze(0).to(self.device)
        return tensor

    def _infer_batch(self, tensor: torch.Tensor, top_k: int) -> List[List[Dict[str, Any]]]:
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)
            top_probs, top_indices = probs.topk(top_k, dim=1)

        batch_predictions: List[List[Dict[str, Any]]] = []
        for sample_probs, sample_indices in zip(top_probs, top_indices):
            frame_predictions: List[Dict[str, Any]] = []
            for prob, idx in zip(sample_probs, sample_indices):
                class_idx = int(idx)
                label = (
                    self.categories[class_idx]
                    if 0 <= class_idx < len(self.categories)
                    else f"class_{class_idx}"
                )
                frame_predictions.append({"label": label, "score": float(prob.item())})
            batch_predictions.append(frame_predictions)

        return batch_predictions

    def _load_model(self, model_arch: str) -> torch.nn.Module:
        model_arch = model_arch.lower()
        
        # Load from timm (modern architectures)
        if self.use_timm:
            if model_arch not in self.TIMM_MODELS:
                available = ", ".join(self.TIMM_MODELS.keys())
                raise ValueError(
                    f"Unsupported timm model_arch '{model_arch}'. "
                    f"Available options: {available}"
                )
            
            timm_name = self.TIMM_MODELS[model_arch]
            self.logger.info(f"Loading timm model: {timm_name} (pretrained on ImageNet)")
            
            # Create model with ImageNet pretrained weights and Places365 classifier
            model = timm.create_model(
                timm_name,
                pretrained=True,
                num_classes=len(self.categories),  # Directly set to Places365 classes
            )
            
            self.logger.info(
                f"Model loaded from timm. Note: Using ImageNet pretrained weights. "
                f"For best results, fine-tune on Places365 dataset."
            )
            return model.to(self.device)
        
        # Load from Places365 (original method)
        if model_arch not in self.MODEL_URLS:
            raise ValueError(
                f"Unsupported model_arch '{model_arch}'. "
                f"Available options: {', '.join(self.MODEL_URLS.keys())}"
            )

        if not hasattr(models, model_arch):
            raise ValueError(f"torchvision.models has no '{model_arch}' implementation")

        constructor = getattr(models, model_arch)
        try:
            model = constructor(weights=None, num_classes=len(self.categories))
        except TypeError:
            model = constructor(weights=None)
            if not hasattr(model, "fc"):
                raise RuntimeError(f"Model '{model_arch}' does not expose an 'fc' attribute")
            in_features = model.fc.in_features
            model.fc = torch.nn.Linear(in_features, len(self.categories))

        state_dict = torch.hub.load_state_dict_from_url(
            self.MODEL_URLS[model_arch], map_location="cpu", progress=True
        )
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]

        cleaned_state = {}
        for key, value in state_dict.items():
            new_key = key.replace("module.", "") if key.startswith("module.") else key
            cleaned_state[new_key] = value

        missing, unexpected = model.load_state_dict(cleaned_state, strict=False)
        if missing:
            self.logger.warning("Missing keys while loading Places365 weights: %s", missing)
        if unexpected:
            self.logger.warning("Unexpected keys while loading Places365 weights: %s", unexpected)

        return model.to(self.device)

    def _load_categories(self, categories_path: Optional[str]) -> List[str]:
        if categories_path:
            path = Path(categories_path)
            if not path.exists():
                raise FileNotFoundError(f"Categories file not found: {categories_path}")
            return self._parse_categories(path.read_text(encoding="utf-8"))

        cache_file = self.cache_dir / "categories_places365.txt"
        if cache_file.exists():
            return self._parse_categories(cache_file.read_text(encoding="utf-8"))

        try:
            response = requests.get(self.CATEGORY_URL, timeout=30)
            response.raise_for_status()
            cache_file.write_text(response.text, encoding="utf-8")
            return self._parse_categories(response.text)
        except Exception as exc:
            self.logger.warning(
                "Failed to load Places365 categories (%s). Falling back to generic labels.",
                exc,
            )
            return [f"class_{idx}" for idx in range(365)]

    def _get_multi_crops(self, image: Image.Image) -> List[Image.Image]:
        """Generate 5 crops: center + 4 corners."""
        width, height = image.size
        # Use input_size as crop size, but ensure it fits
        crop_size = min(self.input_size, width, height)
        crops = []
        
        # Center crop
        left = (width - crop_size) // 2
        top = (height - crop_size) // 2
        crops.append(image.crop((left, top, left + crop_size, top + crop_size)))
        
        # 4 corner crops
        corners = [
            (0, 0),  # top-left
            (width - crop_size, 0),  # top-right
            (0, height - crop_size),  # bottom-left
            (width - crop_size, height - crop_size),  # bottom-right
        ]
        for x, y in corners:
            if x >= 0 and y >= 0 and x + crop_size <= width and y + crop_size <= height:
                crops.append(image.crop((x, y, x + crop_size, y + crop_size)))
        
        return crops[:5]  # Ensure exactly 5 crops

    def _average_predictions(
        self, predictions_list: List[List[Dict[str, Any]]], top_k: int
    ) -> List[Dict[str, Any]]:
        """Average multiple predictions (from TTA or multi-crop)."""
        # Aggregate scores by label
        label_scores: Dict[str, List[float]] = {}
        
        for preds in predictions_list:
            for pred in preds:
                label = pred["label"]
                score = pred["score"]
                if label not in label_scores:
                    label_scores[label] = []
                label_scores[label].append(score)
        
        # Average scores and sort
        averaged = [
            {"label": label, "score": sum(scores) / len(scores)}
            for label, scores in label_scores.items()
        ]
        averaged.sort(key=lambda x: x["score"], reverse=True)
        
        return averaged[:top_k]

    def _apply_temporal_smoothing(
        self, predictions: List[List[Dict[str, Any]]], top_k: int
    ) -> List[List[Dict[str, Any]]]:
        """Apply temporal smoothing using moving average."""
        if len(predictions) == 0:
            return predictions
        
        smoothed: List[List[Dict[str, Any]]] = []
        window = self.smoothing_window
        
        for i in range(len(predictions)):
            # Get window of frames
            start = max(0, i - window // 2)
            end = min(len(predictions), i + window // 2 + 1)
            window_predictions = predictions[start:end]
            
            # Aggregate scores across window
            label_scores: Dict[str, List[float]] = {}
            for frame_preds in window_predictions:
                for pred in frame_preds:
                    label = pred["label"]
                    score = pred["score"]
                    if label not in label_scores:
                        label_scores[label] = []
                    label_scores[label].append(score)
            
            # Average and sort
            averaged = [
                {"label": label, "score": sum(scores) / len(scores)}
                for label, scores in label_scores.items()
            ]
            averaged.sort(key=lambda x: x["score"], reverse=True)
            smoothed.append(averaged[:top_k])
        
        return smoothed

    @staticmethod
    def _parse_categories(raw_text: str) -> List[str]:
        categories: List[str] = []
        for line in raw_text.splitlines():
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            label = " ".join(parts[:-1]).split("/")[-1]
            categories.append(label)
        return categories
    
    def _init_scene_mappings(self) -> None:
        """Initialize mappings for indoor/outdoor and nature/urban classification."""
        # Indoor keywords
        self.indoor_keywords = [
            "room", "bedroom", "kitchen", "bathroom", "living", "dining", "office",
            "hall", "corridor", "staircase", "attic", "basement", "garage", "shop",
            "store", "mall", "restaurant", "cafe", "bar", "pub", "hospital", "school",
            "classroom", "library", "museum", "theater", "cinema", "gym", "stadium",
            "airport", "station", "subway", "train", "bus", "indoor"
        ]
        
        # Outdoor keywords
        self.outdoor_keywords = [
            "outdoor", "street", "road", "highway", "bridge", "park", "garden",
            "forest", "beach", "mountain", "desert", "field", "farm", "lake", "river",
            "ocean", "sea", "sky", "cloud", "sunset", "sunrise", "outdoor"
        ]
        
        # Nature keywords
        self.nature_keywords = [
            "forest", "jungle", "wood", "tree", "beach", "coast", "shore", "mountain",
            "hill", "valley", "desert", "field", "meadow", "grass", "flower", "garden",
            "park", "lake", "river", "stream", "waterfall", "ocean", "sea", "island",
            "cave", "canyon", "cliff", "rock", "snow", "ice", "sky", "cloud", "sunset",
            "sunrise", "nature", "wild", "natural"
        ]
        
        # Urban keywords
        self.urban_keywords = [
            "city", "urban", "street", "road", "avenue", "boulevard", "alley", "plaza",
            "square", "building", "skyscraper", "tower", "bridge", "highway", "subway",
            "station", "airport", "mall", "shop", "store", "restaurant", "cafe", "bar",
            "hotel", "office", "factory", "warehouse", "parking", "lot", "urban"
        ]
    
    def _load_clip_model(self) -> None:
        """Load CLIP model for semantic features."""
        if not CLIP_AVAILABLE:
            self.logger.warning("CLIP not available. Install transformers for semantic features.")
            return
        
        try:
            self.logger.info("Loading CLIP model for semantic features...")
            self._clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            self._clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self._clip_model = self._clip_model.to(self.device)
            self._clip_model.eval()
            self.logger.info("CLIP model loaded successfully")
        except Exception as e:
            self.logger.warning(f"Failed to load CLIP model: {e}. Semantic features will be limited.")
            self.use_clip_for_semantics = False
    
    def _classify_indoor_outdoor(self, scene_label: str) -> Dict[str, float]:
        """
        Classify scene as indoor or outdoor based on Places365 label.
        
        :param scene_label: Scene label from Places365
        :return: Dictionary with indoor/outdoor probabilities
        """
        label_lower = scene_label.lower()
        
        indoor_score = sum(1 for keyword in self.indoor_keywords if keyword in label_lower)
        outdoor_score = sum(1 for keyword in self.outdoor_keywords if keyword in label_lower)
        
        total = indoor_score + outdoor_score
        if total == 0:
            # Default: try to infer from label structure
            if any(word in label_lower for word in ["room", "hall", "indoor"]):
                return {"indoor": 0.7, "outdoor": 0.3}
            else:
                return {"indoor": 0.5, "outdoor": 0.5}
        
        indoor_prob = indoor_score / total
        outdoor_prob = outdoor_score / total
        
        return {"indoor": indoor_prob, "outdoor": outdoor_prob}
    
    def _classify_nature_urban(self, scene_label: str) -> Dict[str, float]:
        """
        Classify scene as nature or urban based on Places365 label.
        
        :param scene_label: Scene label from Places365
        :return: Dictionary with nature/urban probabilities
        """
        label_lower = scene_label.lower()
        
        nature_score = sum(1 for keyword in self.nature_keywords if keyword in label_lower)
        urban_score = sum(1 for keyword in self.urban_keywords if keyword in label_lower)
        
        total = nature_score + urban_score
        if total == 0:
            return {"nature": 0.5, "urban": 0.5}
        
        nature_prob = nature_score / total
        urban_prob = urban_score / total
        
        return {"nature": nature_prob, "urban": urban_prob}
    
    def _detect_time_of_day(self, frame: np.ndarray) -> Dict[str, float]:
        """
        Detect time of day from frame brightness and color analysis.
        
        :param frame: BGR frame
        :return: Dictionary with time of day probabilities
        """
        # Convert to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Calculate brightness
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray) / 255.0
        
        # Analyze color distribution (warm vs cool)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0].flatten()
        
        # Warm colors (sunset/sunrise): hue 0-30, 150-180
        warm_pixels = np.sum((hue < 30) | (hue > 150))
        warm_ratio = warm_pixels / len(hue) if len(hue) > 0 else 0
        
        # Calculate probabilities
        # Morning: medium brightness, warm colors
        morning_score = mean_brightness * 0.6 * (1 + warm_ratio * 0.5)
        
        # Day: high brightness, neutral colors
        day_score = mean_brightness * (1 - warm_ratio * 0.3)
        
        # Evening: medium-low brightness, warm colors
        evening_score = (1 - mean_brightness) * 0.7 * (1 + warm_ratio * 0.8)
        
        # Night: low brightness, cool colors
        night_score = (1 - mean_brightness) * (1 - warm_ratio * 0.5)
        
        # Normalize
        total = morning_score + day_score + evening_score + night_score
        if total == 0:
            return {"morning": 0.25, "day": 0.25, "evening": 0.25, "night": 0.25}
        
        return {
            "morning": morning_score / total,
            "day": day_score / total,
            "evening": evening_score / total,
            "night": night_score / total
        }
    
    def _calculate_aesthetic_score(self, frame: np.ndarray, scene_label: str) -> float:
        """
        Calculate aesthetic score for the scene.
        
        :param frame: BGR frame
        :param scene_label: Scene label
        :return: Aesthetic score (0-1)
        """
        if self.use_clip_for_semantics and self._clip_model is not None:
            return self._calculate_aesthetic_score_clip(frame)
        else:
            return self._calculate_aesthetic_score_heuristic(frame, scene_label)
    
    def _calculate_aesthetic_score_clip(self, frame: np.ndarray) -> float:
        """Calculate aesthetic score using CLIP."""
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            
            texts = [
                "aesthetic beautiful scene",
                "professional photography",
                "ugly unappealing scene",
                "amateur photography"
            ]
            
            inputs = self._clip_processor(
                text=texts,
                images=image,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self._clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
            
            # Positive scores (aesthetic, professional)
            aesthetic_score = (probs[0][0] + probs[0][1]).item()
            return float(aesthetic_score)
        except Exception as e:
            self.logger.warning(f"CLIP aesthetic score failed: {e}, using heuristic")
            return self._calculate_aesthetic_score_heuristic(frame, "")
    
    def _calculate_aesthetic_score_heuristic(self, frame: np.ndarray, scene_label: str) -> float:
        """Calculate aesthetic score using heuristics."""
        # Analyze image quality metrics
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
        
        # Combine scores
        aesthetic = (sharpness_score * 0.3 + contrast * 0.3 + colorfulness * 0.2 + brightness_score * 0.2)
        return float(np.clip(aesthetic, 0.0, 1.0))
    
    def _calculate_luxury_score(self, frame: np.ndarray, scene_label: str) -> float:
        """
        Calculate luxury score for the scene.
        
        :param frame: BGR frame
        :param scene_label: Scene label
        :return: Luxury score (0-1)
        """
        if self.use_clip_for_semantics and self._clip_model is not None:
            return self._calculate_luxury_score_clip(frame)
        else:
            return self._calculate_luxury_score_heuristic(frame, scene_label)
    
    def _calculate_luxury_score_clip(self, frame: np.ndarray) -> float:
        """Calculate luxury score using CLIP."""
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            
            texts = [
                "luxury expensive high-end scene",
                "premium elegant sophisticated",
                "cheap low-quality scene",
                "budget affordable scene"
            ]
            
            inputs = self._clip_processor(
                text=texts,
                images=image,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
            with torch.no_grad():
                outputs = self._clip_model(**inputs)
                logits_per_image = outputs.logits_per_image
                probs = logits_per_image.softmax(dim=1)
            
            luxury_score = (probs[0][0] + probs[0][1]).item()
            return float(luxury_score)
        except Exception as e:
            self.logger.warning(f"CLIP luxury score failed: {e}, using heuristic")
            return self._calculate_luxury_score_heuristic(frame, "")
    
    def _calculate_luxury_score_heuristic(self, frame: np.ndarray, scene_label: str) -> float:
        """Calculate luxury score using heuristics."""
        label_lower = scene_label.lower()
        luxury_keywords = ["luxury", "premium", "elegant", "sophisticated", "high-end", "expensive"]
        
        # Check label
        label_score = 0.3 if any(kw in label_lower for kw in luxury_keywords) else 0.0
        
        # Analyze image quality (luxury scenes often have high quality)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
        quality_score = min(1.0, sharpness / 500.0) * 0.4
        
        # Color richness
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        std_colors = np.std(rgb.reshape(-1, 3), axis=0)
        color_score = np.mean(std_colors) / 255.0 * 0.3
        
        return float(np.clip(label_score + quality_score + color_score, 0.0, 1.0))
    
    def _detect_atmosphere_sentiment(self, frame: np.ndarray) -> Dict[str, float]:
        """
        Detect atmosphere sentiment (cozy, scary, epic).
        
        :param frame: BGR frame
        :return: Dictionary with atmosphere probabilities
        """
        if self.use_clip_for_semantics and self._clip_model is not None:
            return self._detect_atmosphere_clip(frame)
        else:
            return self._detect_atmosphere_heuristic(frame)
    
    def _detect_atmosphere_clip(self, frame: np.ndarray) -> Dict[str, float]:
        """Detect atmosphere using CLIP."""
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(rgb)
            
            texts = [
                "cozy warm comfortable scene",
                "scary frightening dark scene",
                "epic grand majestic scene",
                "neutral ordinary scene"
            ]
            
            inputs = self._clip_processor(
                text=texts,
                images=image,
                return_tensors="pt",
                padding=True
            ).to(self.device)
            
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
        except Exception as e:
            self.logger.warning(f"CLIP atmosphere detection failed: {e}, using heuristic")
            return self._detect_atmosphere_heuristic(frame)
    
    def _detect_atmosphere_heuristic(self, frame: np.ndarray) -> Dict[str, float]:
        """Detect atmosphere using heuristics."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray) / 255.0
        
        # Cozy: medium brightness, warm colors
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0].flatten()
        warm_ratio = np.sum((hue < 30) | (hue > 150)) / len(hue) if len(hue) > 0 else 0
        cozy_score = mean_brightness * 0.6 * (1 + warm_ratio * 0.5)
        
        # Scary: low brightness, high contrast
        contrast = np.std(gray) / 255.0
        scary_score = (1 - mean_brightness) * 0.7 * (1 + contrast * 0.5)
        
        # Epic: high brightness, wide dynamic range
        dynamic_range = (np.max(gray) - np.min(gray)) / 255.0
        epic_score = mean_brightness * 0.8 * (1 + dynamic_range * 0.3)
        
        # Normalize
        total = cozy_score + scary_score + epic_score
        if total == 0:
            return {"cozy": 0.33, "scary": 0.33, "epic": 0.34, "neutral": 0.0}
        
        return {
            "cozy": float(cozy_score / total),
            "scary": float(scary_score / total),
            "epic": float(epic_score / total),
            "neutral": float(1.0 - (cozy_score + scary_score + epic_score) / total)
        }
    
    def _calculate_geometric_features(self, frame: np.ndarray) -> Dict[str, float]:
        """
        Calculate geometric features: openness, clutter, depth cues.
        
        :param frame: BGR frame
        :return: Dictionary with geometric features
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        height, width = gray.shape
        
        # Openness: measure of visible sky/horizon
        # Simplified: analyze top portion of image
        top_portion = gray[:height//3, :]
        top_brightness = np.mean(top_portion) / 255.0
        openness = top_brightness * 0.6 + (1 - np.std(gray) / 255.0) * 0.4
        
        # Clutter: measure of visual complexity
        # Use edge density
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / (height * width)
        clutter = min(1.0, edge_density * 2.0)
        
        # Depth cues: simplified using gradient analysis
        # Strong gradients suggest depth
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        gradient_magnitude = np.sqrt(grad_x**2 + grad_y**2)
        depth_cues = min(1.0, np.mean(gradient_magnitude) / 100.0)
        
        return {
            "openness": float(np.clip(openness, 0.0, 1.0)),
            "clutter": float(np.clip(clutter, 0.0, 1.0)),
            "depth_cues": float(np.clip(depth_cues, 0.0, 1.0))
        }
    
    def classify_with_advanced_features(
        self, frame_manager, frame_indices, top_k: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Classify scenes with advanced features.
        
        :param frames: iterable of OpenCV frames (BGR numpy arrays)
        :param top_k: override default number of predictions
        :return: list of dictionaries with scene predictions and advanced features
        """
        # Get base predictions
        base_predictions = self.classify(frame_manager, frame_indices, top_k)
        
        if not self.enable_advanced_features:
            return [
                {"predictions": preds, "advanced_features": None}
                for preds in base_predictions
            ]
        
        results = {}
        for frame_idx, preds in (zip(frame_indices, base_predictions)):
            
            frame = frame_manager.get(frame_idx)
            
            if frame is None or not preds:
                results[frame_idx] = {
                    "predictions": preds,
                    "advanced_features": None
                }
                continue
            
            # Get top prediction
            top_scene = preds[0]["label"] if preds else ""
            
            # Calculate advanced features
            advanced = {}
            
            # Indoor/outdoor
            indoor_outdoor = self._classify_indoor_outdoor(top_scene)
            advanced["indoor_outdoor"] = indoor_outdoor
            
            # Nature/urban
            nature_urban = self._classify_nature_urban(top_scene)
            advanced["nature_urban"] = nature_urban
            
            # Time of day
            time_of_day = self._detect_time_of_day(frame)
            advanced["time_of_day"] = time_of_day
            
            # Aesthetic score
            aesthetic_score = self._calculate_aesthetic_score(frame, top_scene)
            advanced["aesthetic_score"] = aesthetic_score
            
            # Luxury score
            luxury_score = self._calculate_luxury_score(frame, top_scene)
            advanced["luxury_score"] = luxury_score
            
            # Atmosphere sentiment
            atmosphere = self._detect_atmosphere_sentiment(frame)
            advanced["atmosphere_sentiment"] = atmosphere
            
            # Geometric features
            geometric = self._calculate_geometric_features(frame)
            advanced["geometric_features"] = geometric
            
            results[frame_idx] = {
                "predictions": preds,
                "advanced_features": advanced
            }
        
        return results

