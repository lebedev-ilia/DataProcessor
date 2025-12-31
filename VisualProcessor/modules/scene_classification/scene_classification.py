"""
Scene classification extractor powered by Places365 models.

Обновления:
    - Модуль приведён к интерфейсу `BaseModule` (есть `process()`, поддержка `run()`/`save_results()`).
    - Интеграция с `core_clip` оптимизирована: `embeddings.npz` загружается один раз (mmap) вместо чтения на каждый кадр.
    - Выход приведён к npz-дружелюбному формату: числовые признаки → numpy массивы, переменной длины поля → object arrays.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union, Tuple
from collections import defaultdict

import os
import cv2
import numpy as np
import requests
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

from modules.base_module import BaseModule

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
    
from utils.logger import get_logger
logger = get_logger("Places365SceneClassifier")


def _load_core_clip_embeddings(rs_path: Optional[str], frame_index: int) -> Optional[np.ndarray]:
    """
    Загружает CLIP эмбеддинги из core_clip для конкретного кадра.
    Возвращает None, если core данные недоступны.
    """
    if not rs_path:
        return None
    
    core_path = os.path.join(rs_path, "core_clip", "embeddings.npz")
    if not os.path.isfile(core_path):
        return None
    
    try:
        # Fallback path: avoid full read when possible.
        data = np.load(core_path, mmap_mode="r")
        emb = data.get("frame_embeddings")
        if emb is None:
            return None
        emb = np.asarray(emb, dtype=np.float32)
        
        # Проверяем, что frame_index в пределах массива
        if frame_index < 0 or frame_index >= emb.shape[0]:
            return None
        
        return emb[frame_index]
    except Exception as e:
        logger.warning(f"Places365SceneClassifier | _load_core_clip_embeddings | Error loading core data: {e}")
        return None

class Places365SceneClassifier(BaseModule):
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
        min_scene_length: int = 30,
        min_scene_seconds: Optional[float] = None,
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
        # core-данные
        rs_path: Optional[str] = None,
        **kwargs: Any,
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
        :param min_scene_seconds: minimal scene length in seconds (fps‑aware). If None,
            value will be derived from ``min_scene_length`` and runtime FPS.
        :param enable_advanced_features: enable advanced features (indoor/outdoor, time of day, etc.)
        :param use_clip_for_semantics: use CLIP for semantic features (aesthetic, atmosphere)
        """
        # BaseModule init (results store, logging, metadata helpers)
        super().__init__(rs_path=rs_path, logger_name="scene_classification", **kwargs)

        # Store both frame‑based and time‑based scene length thresholds.
        # Frame threshold is kept for backwards compatibility, but aggregation
        # logic is fps‑aware and primarily uses seconds.
        self.min_scene_length_frames = max(1, int(min_scene_length))
        self.min_scene_seconds = float(min_scene_seconds) if min_scene_seconds is not None else None
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
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
        # CLIP features may still partially work via heuristics when transformers aren't available.
        self.use_clip_for_semantics = bool(use_clip_for_semantics)
        self._clip_model = None
        self._clip_processor = None

        # core_clip integration (cache provider output once, not per-frame)
        self._core_clip_path: Optional[str] = None
        self._core_clip_frame_embeddings: Optional[np.ndarray] = None  # may be memmap
        self._use_core_clip = False
        if rs_path:
            core_path = os.path.join(rs_path, "core_clip", "embeddings.npz")
            if os.path.isfile(core_path):
                self._use_core_clip = True
                self._core_clip_path = core_path
                try:
                    # Memory-map to avoid loading whole array into RAM.
                    data = np.load(core_path, mmap_mode="r")
                    emb = data.get("frame_embeddings")
                    if emb is not None:
                        self._core_clip_frame_embeddings = emb
                except Exception as e:
                    logger.warning(
                        f"Places365SceneClassifier | core_clip preload failed: {e}. "
                        "Will fallback to per-frame loader."
                    )

        # Cached (normalized) text embeddings for core_clip mode
        self._core_text_embeddings: Dict[str, np.ndarray] = {}
        
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
        
        # Load CLIP if needed (только если не используем core_clip).
        # If transformers isn't available, we'll silently fallback to heuristics.
        if self.enable_advanced_features and self.use_clip_for_semantics and (not self._use_core_clip) and CLIP_AVAILABLE:
            self._load_clip_model()
        elif self._use_core_clip:
            logger.info("Places365SceneClassifier | Используется core_clip для семантических фичей")
            # Precompute text embeddings once (optional; closes TODOs for core_clip mode)
            self._maybe_prepare_core_text_embeddings()

    @property
    def module_name(self) -> str:
        # Keep stable module id for metadata section and results folder.
        return "scene_classification"

    def required_dependencies(self) -> List[str]:
        # core_clip is optional. If it's present we use it; otherwise we fallback to internal CLIP or heuristics.
        return []

    def process(self, frame_manager, frame_indices: List[int], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        BaseModule entrypoint.
        Returns a npz-friendly dict (numeric arrays where possible).
        """
        # Apply lightweight runtime overrides (do not rebuild model by default)
        if config:
            self._apply_runtime_config(config)

        agg = self.classify_with_advanced_features(frame_manager, frame_indices)
        return self._pack_npz_result(agg)

    def _apply_runtime_config(self, config: Dict[str, Any]) -> None:
        """Apply safe runtime overrides that don't require model rebuild."""
        try:
            if "min_scene_seconds" in config and config["min_scene_seconds"] is not None:
                self.min_scene_seconds = float(config["min_scene_seconds"])
            if "min_scene_length" in config and config["min_scene_length"] is not None:
                self.min_scene_length_frames = max(1, int(config["min_scene_length"]))
            if "enable_advanced_features" in config and config["enable_advanced_features"] is not None:
                self.enable_advanced_features = bool(config["enable_advanced_features"])
            if "use_clip_for_semantics" in config and config["use_clip_for_semantics"] is not None:
                self.use_clip_for_semantics = bool(config["use_clip_for_semantics"])
        except Exception as e:
            logger.warning(f"Places365SceneClassifier | _apply_runtime_config | Failed to apply overrides: {e}")

    def _get_core_clip_embedding(self, frame_index: int) -> Optional[np.ndarray]:
        """Fast path: use cached core_clip embeddings when available."""
        if self._core_clip_frame_embeddings is not None:
            try:
                if 0 <= frame_index < int(self._core_clip_frame_embeddings.shape[0]):
                    emb = np.asarray(self._core_clip_frame_embeddings[frame_index], dtype=np.float32)
                    return emb
            except Exception:
                return None
        # Fallback to legacy per-frame loader
        return _load_core_clip_embeddings(self.rs_path, frame_index)

    def _maybe_prepare_core_text_embeddings(self) -> None:
        """
        Precompute normalized text embeddings for core_clip mode.
        This allows computing aesthetic/luxury/atmosphere scores without loading CLIP per-frame.
        """
        if not (self._use_core_clip and self.use_clip_for_semantics):
            return
        if not CLIP_AVAILABLE:
            return
        if self._core_text_embeddings:
            return

        try:
            # Compute on CPU to reduce GPU memory pressure (core_clip already provides image embeddings).
            model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            model.eval()

            prompt_sets = {
                "aesthetic": [
                    "aesthetic beautiful scene",
                    "professional photography",
                    "ugly unappealing scene",
                    "amateur photography",
                ],
                "luxury": [
                    "luxury expensive high-end scene",
                    "premium elegant sophisticated",
                    "cheap low-quality scene",
                    "budget affordable scene",
                ],
                "atmosphere": [
                    "cozy warm comfortable scene",
                    "scary frightening dark scene",
                    "epic grand majestic scene",
                    "neutral ordinary scene",
                ],
            }

            for key, texts in prompt_sets.items():
                inputs = processor(text=texts, return_tensors="pt", padding=True)
                with torch.no_grad():
                    feats = model.get_text_features(**inputs)
                feats = feats / (feats.norm(dim=-1, keepdim=True) + 1e-9)
                self._core_text_embeddings[key] = feats.cpu().numpy().astype(np.float32)

            # Free memory
            del model
            del processor
            if torch.cuda.is_available():
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
        except Exception as e:
            logger.warning(f"Places365SceneClassifier | core text embeddings init failed: {e}")

    def _pack_npz_result(self, agg: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert aggregated scene dict to npz-friendly payload:
        - numeric arrays for scalar features
        - object arrays for variable-length lists (indices, dominant_topk_ids, etc.)
        """
        if not agg:
            return {
                "scene_ids": np.asarray([], dtype=object),
                "scene_label": np.asarray([], dtype=object),
                "start_frame": np.asarray([], dtype=np.int32),
                "end_frame": np.asarray([], dtype=np.int32),
                "length_frames": np.asarray([], dtype=np.int32),
                "length_seconds": np.asarray([], dtype=np.float32),
                "scenes_raw": np.asarray({}, dtype=object),
            }

        scene_ids = list(agg.keys())
        scenes = [agg[sid] for sid in scene_ids]

        def f32(name: str) -> np.ndarray:
            return np.asarray([float(s.get(name, 0.0)) for s in scenes], dtype=np.float32)

        def i32(name: str) -> np.ndarray:
            return np.asarray([int(s.get(name, 0)) for s in scenes], dtype=np.int32)

        payload: Dict[str, Any] = {
            "scene_ids": np.asarray(scene_ids, dtype=object),
            "scene_label": np.asarray([s.get("scene_label", "") for s in scenes], dtype=object),
            "start_frame": i32("start_frame"),
            "end_frame": i32("end_frame"),
            "length_frames": i32("length_frames"),
            "length_seconds": f32("length_seconds"),
            # Places metrics
            "mean_score": f32("mean_score"),
            "class_entropy_mean": f32("class_entropy_mean"),
            "top1_prob_mean": f32("top1_prob_mean"),
            "top1_vs_top2_gap_mean": f32("top1_vs_top2_gap_mean"),
            "fraction_high_confidence_frames": f32("fraction_high_confidence_frames"),
            # indoor/outdoor + nature/urban
            "mean_indoor": f32("mean_indoor"),
            "mean_outdoor": f32("mean_outdoor"),
            "mean_nature": f32("mean_nature"),
            "mean_urban": f32("mean_urban"),
            # time-of-day
            "mean_morning": f32("mean_morning"),
            "mean_day": f32("mean_day"),
            "mean_evening": f32("mean_evening"),
            "mean_night": f32("mean_night"),
            "time_of_day_top": np.asarray([s.get("time_of_day_top", "") for s in scenes], dtype=object),
            "time_of_day_confidence": f32("time_of_day_confidence"),
            # aesthetics / luxury
            "mean_aesthetic_score": f32("mean_aesthetic_score"),
            "aesthetic_std": f32("aesthetic_std"),
            "aesthetic_frac_high": f32("aesthetic_frac_high"),
            "mean_luxury_score": f32("mean_luxury_score"),
            # atmosphere
            "mean_cozy": f32("mean_cozy"),
            "mean_scary": f32("mean_scary"),
            "mean_epic": f32("mean_epic"),
            "mean_neutral": f32("mean_neutral"),
            "atmosphere_entropy": f32("atmosphere_entropy"),
            # geometry
            "mean_openness": f32("mean_openness"),
            "mean_clutter": f32("mean_clutter"),
            "mean_depth_cues": f32("mean_depth_cues"),
            # stability
            "scene_change_score": f32("scene_change_score"),
            "label_stability": f32("label_stability"),
            # variable-length lists
            "indices": np.asarray([s.get("indices", []) for s in scenes], dtype=object),
            "dominant_places_topk_ids": np.asarray([s.get("dominant_places_topk_ids", []) for s in scenes], dtype=object),
            "dominant_places_topk_probs": np.asarray([s.get("dominant_places_topk_probs", []) for s in scenes], dtype=object),
            # keep raw for backwards compatibility/debug
            "scenes_raw": np.asarray(agg, dtype=object),
        }
        return payload

    def classify(
        self, frame_manager, frame_indices
    ) -> List[Dict[str, Any]]:
        """
        Runs scene classification over the provided frames.

        Returns a list of dicts, one per frame:
            { "label": str, "score": float }
        """

        # Output results indexed by position (not frame index)
        raw_predictions: List[Dict[str, Any]] = [None] * len(frame_indices)

        # Map frame_index → output array position
        index_map = {frame_idx: i for i, frame_idx in enumerate(frame_indices)}

        # Batch accumulators
        batch_tensors: List[torch.Tensor] = []
        batch_frame_indices: List[int] = []

        def select_best(preds: List[Dict[str, Any]]) -> Dict[str, Any]:
            """Return the highest-score prediction."""
            return max(preds, key=lambda x: x["score"])

        def flush_batch() -> None:
            """Run inference for accumulated tensors and distribute results."""
            if not batch_tensors:
                return

            batch_tensor = torch.cat(batch_tensors, dim=0)
            batch_results = self._infer_batch(batch_tensor)  # returns list[list[preds]]

            # Group predictions by frame index (in case of TTA/multi-crop)
            frame_groups: Dict[int, List[List[Dict[str, Any]]]] = {}

            for frame_idx, preds in zip(batch_frame_indices, batch_results):
                frame_groups.setdefault(frame_idx, []).append(preds)

            # Merge predictions (if multi-crop/TTA) → pick best overall
            for frame_idx, pred_list in frame_groups.items():
                all_preds = [p for group in pred_list for p in group]
                best = select_best(all_preds)

                pos = index_map[frame_idx]
                raw_predictions[pos] = best

            batch_tensors.clear()
            batch_frame_indices.clear()

        # Main loop
        for frame_idx in frame_indices:

            frame = frame_manager.get(frame_idx)
            if frame is None:
                continue

            tensors = self._prepare_frame(frame)
            if tensors is None:
                continue

            # Multi-crop/TTA returns list
            if isinstance(tensors, list):
                for tensor in tensors:
                    batch_tensors.append(tensor)
                    batch_frame_indices.append(frame_idx)
            else:
                batch_tensors.append(tensors)
                batch_frame_indices.append(frame_idx)

            if len(batch_tensors) >= self.batch_size:
                flush_batch()

        # Last batch
        flush_batch()

        return raw_predictions

    __call__ = classify


    def _prepare_frame(self, frame: np.ndarray) -> Optional[torch.Tensor | List[torch.Tensor]]:
        if frame is None or not isinstance(frame, np.ndarray):
            logger.warning("Frame is not a numpy array – skipping")
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

    def _infer_batch(self, tensor: torch.Tensor) -> List[List[Dict[str, Any]]]:
        with torch.no_grad():
            logits = self.model(tensor)
            probs = F.softmax(logits, dim=1)

        batch_predictions: List[List[Dict[str, Any]]] = []

        for sample_probs in probs:  # sample_probs: [num_classes]
            # --- Global confidence statistics for this frame ---
            # Shannon entropy over class probabilities
            entropy_val = float(
                -torch.sum(sample_probs * torch.log(sample_probs + 1e-8)).item()
            )

            # Top‑K (for ontology‑based features later)
            top_k = min(5, sample_probs.shape[0])
            topk_vals, topk_indices = torch.topk(sample_probs, k=top_k)
            top1_prob_val = float(topk_vals[0].item())
            top2_prob_val = float(topk_vals[1].item()) if top_k > 1 else 0.0
            top1_top2_gap = float(max(0.0, top1_prob_val - top2_prob_val))
            topk_indices_list = [int(i.item()) for i in topk_indices]
            topk_probs_list = [float(v.item()) for v in topk_vals]
            top1_idx = topk_indices_list[0]

            frame_predictions: List[Dict[str, Any]] = []
            for class_idx, prob in enumerate(sample_probs):
                label = (
                    self.categories[class_idx]
                    if 0 <= class_idx < len(self.categories)
                    else f"class_{class_idx}"
                )
                entry: Dict[str, Any] = {
                    "label": label,
                    "score": float(prob),
                    # Per‑frame confidence summary (same for all classes for this frame)
                    "entropy": entropy_val,
                    "top1_prob": top1_prob_val,
                    "top2_prob": top2_prob_val,
                    "top1_top2_gap": top1_top2_gap,
                    "class_idx": int(class_idx),
                }
                # Store compact top‑K only once (on the top‑1 entry) to avoid bloat.
                if class_idx == top1_idx:
                    entry["topk_class_indices"] = topk_indices_list
                    entry["topk_class_probs"] = topk_probs_list

                frame_predictions.append(entry)

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
            logger.info(f"Loading timm model: {timm_name} (pretrained on ImageNet)")
            
            # Create model with ImageNet pretrained weights and Places365 classifier
            model = timm.create_model(
                timm_name,
                pretrained=True,
                num_classes=len(self.categories),  # Directly set to Places365 classes
            )
            
            logger.info(
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
            logger.warning("Missing keys while loading Places365 weights: %s", missing)
        if unexpected:
            logger.warning("Unexpected keys while loading Places365 weights: %s", unexpected)

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
            logger.warning(
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
        # Indoor keywords (used as a simple ontology over Places365 labels)
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
        import os
        p = os.path.dirname(__file__)

        if not CLIP_AVAILABLE:
            logger.warning("CLIP not available. Install transformers for semantic features.")
            return
        try:
            logger.info("Loading CLIP model for semantic features...")
            self._clip_model = CLIPModel.from_pretrained(f"{p}/models/clip_vit_base_patch32")
            self._clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
            self._clip_model = self._clip_model.to(self.device)
            self._clip_model.eval()
            logger.info("CLIP model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load CLIP model: {e}. Semantic features will be limited.")
            self.use_clip_for_semantics = False
    
    def _ontology_indoor_outdoor(
        self,
        topk_labels: Sequence[str],
        topk_probs: Sequence[float],
    ) -> Dict[str, float]:
        """
        Soft indoor/outdoor score based on ontology mapping of top‑K Places labels.

        indoor_score = sum(prob_i * I(label_i is indoor))
        outdoor_score = sum(prob_i * I(label_i is outdoor))
        """
        if not topk_labels or not topk_probs:
            return {"indoor": 0.5, "outdoor": 0.5}

        indoor_score = 0.0
        outdoor_score = 0.0

        for label, p in zip(topk_labels, topk_probs):
            p = float(p)
            lname = label.lower()
            has_indoor = any(k in lname for k in self.indoor_keywords)
            has_outdoor = any(k in lname for k in self.outdoor_keywords)

            if has_indoor and not has_outdoor:
                indoor_score += p
            elif has_outdoor and not has_indoor:
                outdoor_score += p
            elif has_indoor and has_outdoor:
                indoor_score += 0.5 * p
                outdoor_score += 0.5 * p

        total = indoor_score + outdoor_score
        if total <= 0.0:
            return {"indoor": 0.5, "outdoor": 0.5}

        return {
            "indoor": float(indoor_score / total),
            "outdoor": float(outdoor_score / total),
        }

    def _ontology_nature_urban(
        self,
        topk_labels: Sequence[str],
        topk_probs: Sequence[float],
    ) -> Dict[str, float]:
        """
        Soft nature/urban score based on ontology mapping of top‑K Places labels.
        """
        if not topk_labels or not topk_probs:
            return {"nature": 0.5, "urban": 0.5}

        nature_score = 0.0
        urban_score = 0.0

        for label, p in zip(topk_labels, topk_probs):
            p = float(p)
            lname = label.lower()
            has_nature = any(k in lname for k in self.nature_keywords)
            has_urban = any(k in lname for k in self.urban_keywords)

            if has_nature and not has_urban:
                nature_score += p
            elif has_urban and not has_nature:
                urban_score += p
            elif has_nature and has_urban:
                nature_score += 0.5 * p
                urban_score += 0.5 * p

        total = nature_score + urban_score
        if total <= 0.0:
            return {"nature": 0.5, "urban": 0.5}

        return {
            "nature": float(nature_score / total),
            "urban": float(urban_score / total),
        }
    
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
    
    def _calculate_aesthetic_score(self, frame: np.ndarray, scene_label: str, frame_index: Optional[int] = None) -> float:
        """
        Calculate aesthetic score for the scene.
        
        :param frame: BGR frame
        :param scene_label: Scene label
        :param frame_index: Frame index for core_clip lookup
        :return: Aesthetic score (0-1)
        """
        if not self.use_clip_for_semantics:
            return self._calculate_aesthetic_score_heuristic(frame, scene_label)
            if self._use_core_clip and frame_index is not None:
                return self._calculate_aesthetic_score_core_clip(frame_index)
        if self._clip_model is not None:
                return self._calculate_aesthetic_score_clip(frame)
            return self._calculate_aesthetic_score_heuristic(frame, scene_label)
    
    def _calculate_aesthetic_score_core_clip(self, frame_index: int) -> float:
        """Calculate aesthetic score using core_clip embeddings."""
        try:
            img_feat = self._get_core_clip_embedding(frame_index)
            if img_feat is None:
                logger.warning(f"Places365SceneClassifier | _calculate_aesthetic_score_core_clip | core_clip not found for frame {frame_index}, using heuristic")
                return 0.5  # Fallback to neutral score
            
            img_feat = np.asarray(img_feat, dtype=np.float32)
            img_feat = img_feat / (np.linalg.norm(img_feat) + 1e-9)

            self._maybe_prepare_core_text_embeddings()
            text_emb = self._core_text_embeddings.get("aesthetic")
            if text_emb is None:
                # Fallback heuristic: stable neutral
                return 0.5

            logits = img_feat @ text_emb.T  # (4,)
            logits = logits - float(np.max(logits))
            probs = np.exp(logits) / (np.sum(np.exp(logits)) + 1e-9)
            # Positive prompts: [0,1]
            return float(np.clip(float(probs[0] + probs[1]), 0.0, 1.0))
        except Exception as e:
            logger.warning(f"Places365SceneClassifier | _calculate_aesthetic_score_core_clip | Error: {e}, using heuristic")
            return 0.5
    
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
            logger.warning(f"CLIP aesthetic score failed: {e}, using heuristic")
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
    
    def _calculate_luxury_score(self, frame: np.ndarray, scene_label: str, frame_index: Optional[int] = None) -> float:
        """
        Calculate luxury score for the scene.
        
        :param frame: BGR frame
        :param scene_label: Scene label
        :param frame_index: Frame index for core_clip lookup
        :return: Luxury score (0-1)
        """
        if not self.use_clip_for_semantics:
            return self._calculate_luxury_score_heuristic(frame, scene_label)
            if self._use_core_clip and frame_index is not None:
                return self._calculate_luxury_score_core_clip(frame_index)
        if self._clip_model is not None:
                return self._calculate_luxury_score_clip(frame)
            return self._calculate_luxury_score_heuristic(frame, scene_label)
    
    def _calculate_luxury_score_core_clip(self, frame_index: int) -> float:
        """Calculate luxury score using core_clip embeddings."""
        try:
            img_feat = self._get_core_clip_embedding(frame_index)
            if img_feat is None:
                logger.warning(f"Places365SceneClassifier | _calculate_luxury_score_core_clip | core_clip not found for frame {frame_index}, using heuristic")
                return 0.5

            img_feat = np.asarray(img_feat, dtype=np.float32)
            img_feat = img_feat / (np.linalg.norm(img_feat) + 1e-9)

            self._maybe_prepare_core_text_embeddings()
            text_emb = self._core_text_embeddings.get("luxury")
            if text_emb is None:
                return 0.5

            logits = img_feat @ text_emb.T
            logits = logits - float(np.max(logits))
            probs = np.exp(logits) / (np.sum(np.exp(logits)) + 1e-9)
            return float(np.clip(float(probs[0] + probs[1]), 0.0, 1.0))
        except Exception as e:
            logger.warning(f"Places365SceneClassifier | _calculate_luxury_score_core_clip | Error: {e}, using heuristic")
            return 0.5
    
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
            logger.warning(f"CLIP luxury score failed: {e}, using heuristic")
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
    
    def _detect_atmosphere_sentiment(self, frame: np.ndarray, frame_index: Optional[int] = None) -> Dict[str, float]:
        """
        Detect atmosphere sentiment (cozy, scary, epic, neutral).
        
        :param frame: BGR frame
        :param frame_index: Frame index for core_clip lookup
        :return: Dict with atmosphere probabilities
        """
        if not self.use_clip_for_semantics:
            return self._detect_atmosphere_heuristic(frame)
            if self._use_core_clip and frame_index is not None:
                return self._detect_atmosphere_core_clip(frame_index)
        if self._clip_model is not None:
                return self._detect_atmosphere_clip(frame)
            return self._detect_atmosphere_heuristic(frame)
    
    def _detect_atmosphere_core_clip(self, frame_index: int) -> Dict[str, float]:
        """Detect atmosphere using core_clip embeddings."""
        try:
            img_feat = self._get_core_clip_embedding(frame_index)
            if img_feat is None:
                logger.warning(f"Places365SceneClassifier | _detect_atmosphere_core_clip | core_clip not found for frame {frame_index}, using heuristic")
                return {"cozy": 0.25, "scary": 0.25, "epic": 0.25, "neutral": 0.25}

            img_feat = np.asarray(img_feat, dtype=np.float32)
            img_feat = img_feat / (np.linalg.norm(img_feat) + 1e-9)

            self._maybe_prepare_core_text_embeddings()
            text_emb = self._core_text_embeddings.get("atmosphere")
            if text_emb is None:
            return {"cozy": 0.25, "scary": 0.25, "epic": 0.25, "neutral": 0.25}

            logits = img_feat @ text_emb.T
            logits = logits - float(np.max(logits))
            probs = np.exp(logits) / (np.sum(np.exp(logits)) + 1e-9)
            return {
                "cozy": float(probs[0]),
                "scary": float(probs[1]),
                "epic": float(probs[2]),
                "neutral": float(probs[3]),
            }
        except Exception as e:
            logger.warning(f"Places365SceneClassifier | _detect_atmosphere_core_clip | Error: {e}, using heuristic")
            return {"cozy": 0.25, "scary": 0.25, "epic": 0.25, "neutral": 0.25}
    
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
            logger.warning(f"CLIP atmosphere detection failed: {e}, using heuristic")
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
    
    def aggregate_scenes(self, res, fps: float) -> Dict[str, Any]:
        """
        Aggregate consecutive frames with the same scene label.

        Args:
            res: dict[frame_idx] = {
                "predictions": {
                    "label": str,
                    "score": float,
                    "entropy": float,
                    "top1_prob": float,
                    "top2_prob": float,
                    "top1_top2_gap": float,
                    "class_idx": int,
                    "topk_class_indices": Optional[List[int]],
                    "topk_class_probs": Optional[List[float]],
                },
                "advanced_features": {
                    "indoor_outdoor": {"indoor": float, "outdoor": float},
                    "nature_urban": {"nature": float, "urban": float},
                    "time_of_day": {"morning": float, "day": float, "evening": float, "night": float},
                    "aesthetic_score": float,
                    "luxury_score": float,
                    "atmosphere_sentiment": {"cozy": float, "scary": float, "epic": float, "neutral": float},
                    "geometric_features": {"openness": float, "clutter": float, "depth_cues": float}
                }
            }
            fps: frames per second for current video (used for time‑based stats)

        Returns:
            dict: aggregated segments with means and indices
        """
        import numpy as np

        if not res:
            return {}

        aggregated: Dict[str, Any] = {}
        current_label = None
        current_indices = []
        current_values = None
        current_topk_indices: List[Sequence[int]] = []
        current_topk_probs: List[Sequence[float]] = []

        def reset_values():
            return {
                "score": [], "entropy": [],
                "top1_prob": [], "top2_prob": [], "top1_top2_gap": [],
                "indoor": [], "outdoor": [],
                "nature": [], "urban": [],
                "morning": [], "day": [], "evening": [], "night": [],
                "aesthetic_score": [], "luxury_score": [],
                "cozy": [], "scary": [], "epic": [], "neutral": [],
                "openness": [], "clutter": [], "depth_cues": [],
                "labels": [],
            }

        def finalize_segment(label, indices, values, topk_idx_seq, topk_prob_seq):
            """Return aggregated segment dict."""
            if not indices:
                return None
            # FPS‑aware duration
            length_frames = len(indices)
            fps_safe = float(fps) if fps and fps > 0 else 30.0
            length_seconds = float(length_frames) / fps_safe

            # Determine minimal duration in seconds
            if self.min_scene_seconds is not None:
                min_len_s = self.min_scene_seconds
            else:
                # Backwards‑compatible: interpret frame threshold at runtime FPS
                min_len_s = float(self.min_scene_length_frames) / fps_safe

            if length_seconds < min_len_s:
                return None

            start_frame = int(indices[0])
            end_frame = int(indices[-1])

            # Scene‑level time‑of‑day distribution
            tod_vec = np.array([
                np.mean(values["morning"]),
                np.mean(values["day"]),
                np.mean(values["evening"]),
                np.mean(values["night"]),
            ], dtype=np.float32)
            tod_sum = float(tod_vec.sum()) or 1.0
            tod_probs = (tod_vec / tod_sum).tolist()
            tod_top_idx = int(np.argmax(tod_vec))
            tod_labels = ["morning", "day", "evening", "night"]
            tod_top_label = tod_labels[tod_top_idx]
            tod_conf = float(tod_vec[tod_top_idx])

            # Aesthetic / luxury aggregates
            aesthetic_arr = np.asarray(values["aesthetic_score"], dtype=np.float32)
            luxury_arr = np.asarray(values["luxury_score"], dtype=np.float32)
            aesthetic_mean = float(np.mean(aesthetic_arr)) if aesthetic_arr.size else 0.0
            aesthetic_std = float(np.std(aesthetic_arr)) if aesthetic_arr.size else 0.0
            aesthetic_frac_high = float(np.mean(aesthetic_arr > 0.8)) if aesthetic_arr.size else 0.0
            luxury_mean = float(np.mean(luxury_arr)) if luxury_arr.size else 0.0

            # Atmosphere entropy
            atm_mat = np.stack(
                [
                    np.asarray(values["cozy"], dtype=np.float32),
                    np.asarray(values["scary"], dtype=np.float32),
                    np.asarray(values["epic"], dtype=np.float32),
                    np.asarray(values["neutral"], dtype=np.float32),
                ],
                axis=0,
            )
            atm_mean = atm_mat.mean(axis=1)
            atm_sum = float(atm_mean.sum()) or 1.0
            atm_probs = atm_mean / atm_sum
            atm_entropy = float(-np.sum(atm_probs * np.log(atm_probs + 1e-8)))

            # Label stability
            labels = values["labels"]
            if labels:
                from collections import Counter
                cnt = Counter(labels)
                scene_label = cnt.most_common(1)[0][0]
                label_stability = float(cnt[scene_label] / len(labels))
            else:
                scene_label = label
                label_stability = 0.0

            # Scene change score: within‑scene variance of confidence
            score_arr = np.asarray(values["score"], dtype=np.float32)
            scene_change_score = float(np.std(score_arr)) if score_arr.size else 0.0

            # Places confidence aggregates
            top1_arr = np.asarray(values["top1_prob"], dtype=np.float32)
            entropy_arr = np.asarray(values["entropy"], dtype=np.float32)
            gap_arr = np.asarray(values["top1_top2_gap"], dtype=np.float32)
            places_top1_prob_mean = float(np.mean(top1_arr)) if top1_arr.size else 0.0
            places_entropy_mean = float(np.mean(entropy_arr)) if entropy_arr.size else 0.0
            places_top1_vs_top2_gap_mean = float(np.mean(gap_arr)) if gap_arr.size else 0.0
            fraction_high_confidence_frames = (
                float(np.mean(top1_arr > 0.7)) if top1_arr.size else 0.0
            )

            # Dominant Places top‑K ids aggregated over scene
            dominant_topk_ids: List[int] = []
            dominant_topk_probs: List[float] = []
            if topk_idx_seq:
                from collections import Counter

                # Flatten indices and accumulate weights using probs
                weight_by_class: Dict[int, float] = {}
                for idx_list, prob_list in zip(topk_idx_seq, topk_prob_seq):
                    for cid, p in zip(idx_list, prob_list):
                        weight_by_class[int(cid)] = weight_by_class.get(int(cid), 0.0) + float(p)
                if weight_by_class:
                    # Take top‑5 by accumulated weight
                    sorted_items = sorted(
                        weight_by_class.items(), key=lambda x: x[1], reverse=True
                    )
                    for cid, w in sorted_items[:5]:
                        dominant_topk_ids.append(int(cid))
                        dominant_topk_probs.append(float(w))

            return {
                "scene_label": scene_label,
                "indices": indices,
                "start_frame": start_frame,
                "end_frame": end_frame,
                "length_frames": length_frames,
                "length_seconds": length_seconds,
                "mean_score": float(np.mean(values["score"])),
                "class_entropy_mean": places_entropy_mean,
                "top1_prob_mean": places_top1_prob_mean,
                "top1_vs_top2_gap_mean": places_top1_vs_top2_gap_mean,
                "fraction_high_confidence_frames": fraction_high_confidence_frames,
                "mean_indoor": float(np.mean(values["indoor"])),
                "mean_outdoor": float(np.mean(values["outdoor"])),
                "mean_nature": float(np.mean(values["nature"])),
                "mean_urban": float(np.mean(values["urban"])),
                "mean_morning": float(np.mean(values["morning"])),
                "mean_day": float(np.mean(values["day"])),
                "mean_evening": float(np.mean(values["evening"])),
                "mean_night": float(np.mean(values["night"])),
                "time_of_day_probs": {
                    "morning": tod_probs[0],
                    "day": tod_probs[1],
                    "evening": tod_probs[2],
                    "night": tod_probs[3],
                },
                "time_of_day_top": tod_top_label,
                "time_of_day_confidence": tod_conf,
                "mean_aesthetic_score": aesthetic_mean,
                "aesthetic_std": aesthetic_std,
                "aesthetic_frac_high": aesthetic_frac_high,
                "mean_luxury_score": luxury_mean,
                "mean_cozy": float(np.mean(values["cozy"])),
                "mean_scary": float(np.mean(values["scary"])),
                "mean_epic": float(np.mean(values["epic"])),
                "mean_neutral": float(np.mean(values["neutral"])),
                "atmosphere_entropy": atm_entropy,
                "mean_openness": float(np.mean(values["openness"])),
                "mean_clutter": float(np.mean(values["clutter"])),
                "mean_depth_cues": float(np.mean(values["depth_cues"])),
                "scene_change_score": scene_change_score,
                "label_stability": label_stability,
                "dominant_places_topk_ids": dominant_topk_ids,
                "dominant_places_topk_probs": dominant_topk_probs,
            }

        for idx in sorted(res.keys()):
            d = res[idx]
            label = d["predictions"]["label"]

            if current_label != label:
                if current_label is not None:
                    segment = finalize_segment(
                        current_label, current_indices, current_values,
                        current_topk_indices, current_topk_probs,
                    )
                    if segment:
                        aggregated[f"{current_label}_{current_indices[0]}"] = segment
                # Start new segment
                current_label = label
                current_indices = [idx]
                current_values = reset_values()
                current_topk_indices = []
                current_topk_probs = []
            else:
                current_indices.append(idx)

            # Fill values
            adv = d["advanced_features"]
            current_values["score"].append(d["predictions"]["score"])
            current_values["entropy"].append(d["predictions"].get("entropy", 0.0))
            current_values["top1_prob"].append(d["predictions"].get("top1_prob", d["predictions"]["score"]))
            current_values["top2_prob"].append(d["predictions"].get("top2_prob", 0.0))
            current_values["top1_top2_gap"].append(d["predictions"].get("top1_top2_gap", 0.0))
            current_values["labels"].append(label)

            tk_idx = d["predictions"].get("topk_class_indices")
            tk_prob = d["predictions"].get("topk_class_probs")
            if tk_idx is not None and tk_prob is not None:
                current_topk_indices.append(tk_idx)
                current_topk_probs.append(tk_prob)

            current_values["indoor"].append(adv["indoor_outdoor"]["indoor"])
            current_values["outdoor"].append(adv["indoor_outdoor"]["outdoor"])
            current_values["nature"].append(adv["nature_urban"]["nature"])
            current_values["urban"].append(adv["nature_urban"]["urban"])
            tod = adv["time_of_day"]
            current_values["morning"].append(tod["morning"])
            current_values["day"].append(tod["day"])
            current_values["evening"].append(tod["evening"])
            current_values["night"].append(tod["night"])
            current_values["aesthetic_score"].append(adv["aesthetic_score"])
            current_values["luxury_score"].append(adv["luxury_score"])
            atm = adv["atmosphere_sentiment"]
            current_values["cozy"].append(atm["cozy"])
            current_values["scary"].append(atm["scary"])
            current_values["epic"].append(atm["epic"])
            current_values["neutral"].append(atm["neutral"])
            geo = adv["geometric_features"]
            current_values["openness"].append(geo["openness"])
            current_values["clutter"].append(geo["clutter"])
            current_values["depth_cues"].append(geo["depth_cues"])

        # Final segment
        if current_label is not None:
            segment = finalize_segment(
                current_label, current_indices, current_values,
                current_topk_indices, current_topk_probs,
            )
            if segment:
                aggregated[f"{current_label}_{current_indices[0]}"] = segment

        return aggregated

    def classify_with_advanced_features(
        self, frame_manager, frame_indices
    ) -> List[Dict[str, Any]]:
        """
        Classify scenes and compute advanced features (if enabled).

        Returns list aligned with frame_indices:
            {
                "predictions": { "label": str, "score": float } or None,
                "advanced_features": dict or None
            }
        """
        
        # Base predictions (already best-only, but enriched with confidence stats)
        # Ensure module is ready even if user calls this directly.
        try:
            self.initialize()
        except Exception:
            # If BaseModule init path isn't used in some legacy contexts, ignore.
            pass

        base_predictions = self.classify(frame_manager, frame_indices)

        # If advanced features disabled — simple scene aggregation with numeric stats
        if not self.enable_advanced_features:
            results = {
                frame_idx: {
                    "predictions": pred,
                    "advanced_features": None
                }
                for frame_idx, pred in zip(frame_indices, base_predictions)
                if pred is not None
            }
            fps = getattr(frame_manager, "fps", 30.0)
            return self.aggregate_scenes(results, fps=fps)

        # Allocate output dict indexed by frame ID
        results: Dict[int, Dict[str, Any]] = {}

        for frame_idx, pred in zip(frame_indices, base_predictions):
            frame = frame_manager.get(frame_idx)

            # No frame OR no prediction → no features
            if frame is None or pred is None:
                results[frame_idx] = {
                    "predictions": pred,
                    "advanced_features": None
                }
                continue

            # Best scene prediction and top‑K info
            top_scene = pred["label"]
            topk_indices = pred.get("topk_class_indices") or []
            topk_probs = pred.get("topk_class_probs") or []
            topk_labels = [
                self.categories[i] if 0 <= i < len(self.categories) else f"class_{i}"
                for i in topk_indices
            ]

            # Compute advanced features
            advanced: Dict[str, Any] = {}

            advanced["indoor_outdoor"] = self._ontology_indoor_outdoor(topk_labels, topk_probs)
            advanced["nature_urban"] = self._ontology_nature_urban(topk_labels, topk_probs)
            advanced["time_of_day"] = self._detect_time_of_day(frame)
            advanced["aesthetic_score"] = self._calculate_aesthetic_score(frame, top_scene, frame_index=frame_idx)
            advanced["luxury_score"] = self._calculate_luxury_score(frame, top_scene, frame_index=frame_idx)
            advanced["atmosphere_sentiment"] = self._detect_atmosphere_sentiment(frame, frame_index=frame_idx)
            advanced["geometric_features"] = self._calculate_geometric_features(frame)

            results[frame_idx] = {
                "predictions": pred,
                "advanced_features": advanced
            }

        fps = getattr(frame_manager, "fps", 30.0)
        agg_result = self.aggregate_scenes(results, fps=fps)

        return agg_result

