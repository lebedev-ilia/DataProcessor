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
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from dp_models.manager import get_global_model_manager
from dp_models.errors import ModelManagerError

from modules.base_module import BaseModule

from utils.logger import get_logger
logger = get_logger("Places365SceneClassifier")

class Places365SceneClassifier(BaseModule):
    """
    Scene classifier built on top of Places365 checkpoints.

    The extractor accepts a list of OpenCV frames (BGR numpy arrays) and returns
    the top-K scene predictions per frame.
    """

    # Supported architectures (must be backed by dp_models specs; no downloads allowed).
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
        runtime: str = "inprocess",
        triton_model_spec: str = "places365_resnet50_224_triton",
        model_arch: str = "resnet50",
        use_timm: bool = False,
        min_scene_length: int = 30,
        min_scene_seconds: Optional[float] = None,
        batch_size: int = 1,
        device: Optional[str] = None,
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
        :param categories_path: (removed) categories are resolved via ModelManager (DP_MODELS_ROOT)
        :param cache_dir: (removed) no implicit downloads/caching; local artifacts must exist
        :param gpu_memory_threshold: BaseExtractor GPU memory threshold
        :param log_metrics_every_n_frames: resource logging cadence
        :param input_size: input image size (224, 256, 320, etc.). Larger = better accuracy, slower
        :param use_tta: enable Test-Time Augmentation (multiple augmentations + averaging)
        :param use_multi_crop: enable multi-crop inference (5 crops: center + 4 corners)
        :param temporal_smoothing: enable temporal smoothing for video sequences
        :param smoothing_window: window size for temporal smoothing (number of frames)
        :param min_scene_seconds: minimal scene length in seconds (fps‑aware). If None,
            value will be derived from ``min_scene_length`` and runtime FPS.
        :param enable_advanced_features: enable advanced features (ontology + core_clip semantics)
        :param use_clip_for_semantics: kept for compatibility; semantics is always core_clip-only (no local CLIP, no heuristics)
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
        self.use_timm = bool(use_timm)
        
        # Advanced features policy:
        # - heuristics are forbidden (audit rule)
        # - semantics is computed strictly from core_clip embeddings + core_clip-provided prompt embeddings
        self.enable_advanced_features = bool(enable_advanced_features)
        self.use_clip_for_semantics = bool(use_clip_for_semantics)

        # core_clip integration (cache provider output once, not per-frame)
        self._core_clip_path: Optional[str] = None
        self._core_clip_frame_embeddings: Optional[np.ndarray] = None  # may be memmap
        self._core_clip_frame_indices: Optional[np.ndarray] = None
        self._core_clip_index_map: Optional[Dict[int, int]] = None
        self._use_core_clip = False
        if rs_path:
            core_path = os.path.join(rs_path, "core_clip", "embeddings.npz")
            if os.path.isfile(core_path):
                self._use_core_clip = True
                self._core_clip_path = core_path
                try:
                    # Load and build strict index map (union-domain frame_indices → row)
                    data = np.load(core_path, allow_pickle=True)
                    idx = data.get("frame_indices")
                    emb = data.get("frame_embeddings")
                    if idx is not None and emb is not None:
                        idx = np.asarray(idx, dtype=np.int32)
                        emb = np.asarray(emb, dtype=np.float32)
                        self._core_clip_frame_indices = idx
                        self._core_clip_frame_embeddings = emb
                        self._core_clip_index_map = {int(fi): i for i, fi in enumerate(idx.tolist())}
                except Exception as e:
                    logger.warning(
                        f"Places365SceneClassifier | core_clip preload failed: {e}. "
                        "Will fallback to per-frame loader."
                    )

        # core_clip text embeddings for scene semantics (provided by core_clip NPZ)
        self._scene_aesthetic_text_embeddings: Optional[np.ndarray] = None
        self._scene_luxury_text_embeddings: Optional[np.ndarray] = None
        self._scene_atmosphere_text_embeddings: Optional[np.ndarray] = None
        self._last_core_clip_models_used: List[Dict[str, Any]] = []
        self._last_places_models_used: List[Dict[str, Any]] = []
        
        # Initialize indoor/outdoor and nature/urban mappings
        self._init_scene_mappings()

        self.runtime = str(runtime or "inprocess").strip().lower()
        self.triton_model_spec = str(triton_model_spec or "").strip()

        # --- Load Places365 via ModelManager (strict local-only) ---
        self._mm = get_global_model_manager()
        model_arch = str(model_arch or "").strip().lower()
        if self.runtime == "triton":
            if self.input_size not in (224, 336, 448):
                raise ValueError("scene_classification(triton) supports only input_size in {224,336,448} (fixed-shape Triton branches).")
            if self.use_timm:
                raise ValueError("scene_classification(triton): use_timm is not supported (baseline=Places365 ResNet50).")
            if self.use_tta or self.use_multi_crop:
                raise ValueError("scene_classification(triton): TTA/multi-crop are not supported (keep defaults).")
            if not self.triton_model_spec:
                raise ValueError("scene_classification(triton): triton_model_spec is empty.")
            try:
                resolved = self._mm.get(model_name=self.triton_model_spec)
            except ModelManagerError as e:
                raise RuntimeError(f"scene_classification | failed to load Places365 Triton spec via ModelManager: {e}") from e
            self._triton_handle = resolved.handle  # dict with {"client", "triton_model_name", ...}
            self._triton_rp = dict(resolved.spec.runtime_params or {})
            self.model = None
        elif self.use_timm:
            if model_arch not in self.TIMM_MODELS:
                available = ", ".join(sorted(self.TIMM_MODELS.keys()))
                raise ValueError(f"Unsupported timm model_arch '{model_arch}'. Available: {available}")
            spec_name = f"places365_timm_{model_arch}"
        else:
            if model_arch not in ("resnet18", "resnet50"):
                raise ValueError("Unsupported Places365 model_arch (no-network). Use resnet18/resnet50, or set use_timm=true.")
            spec_name = f"places365_{model_arch}"

        if self.runtime != "triton":
            try:
                resolved = self._mm.get(model_name=spec_name)
            except ModelManagerError as e:
                raise RuntimeError(f"scene_classification | failed to load Places365 via ModelManager: {e}") from e
            self.model = resolved.handle
            self._triton_handle = None
            self._triton_rp = {}
        # Resolve categories file from spec runtime_params.categories_relpath
        rp = (resolved.spec.runtime_params or {}) if "resolved" in locals() else (self._triton_rp or {})
        cat_rel = rp.get("categories_relpath")
        if not isinstance(cat_rel, str) or not cat_rel.strip():
            raise RuntimeError("scene_classification | ModelSpec missing runtime_params.categories_relpath")
        cat_abs = (resolved.resolved_artifacts.get(cat_rel) if "resolved" in locals() else None)
        if not cat_abs and isinstance(self._mm, object):
            # For Triton spec, categories are a local_artifact; ModelManager should resolve it too.
            try:
                cat_abs = getattr(resolved, "resolved_artifacts", {}).get(cat_rel) if "resolved" in locals() else None
            except Exception:
                cat_abs = None
        if not cat_abs:
            raise RuntimeError(f"scene_classification | categories file is not resolved: {cat_rel}")
        self.categories = self._parse_categories(Path(cat_abs).read_text(encoding="utf-8"))
        self._last_places_models_used = [resolved.models_used_entry] if "resolved" in locals() else []
        
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
        
        if self.model is not None:
            self.model.eval()
        
        # NOTE: semantics is strictly core_clip-only. If core_clip is missing, module will fail-fast in process().

    @property
    def module_name(self) -> str:
        # Keep stable module id for metadata section and results folder.
        return "scene_classification"

    def required_dependencies(self) -> List[str]:
        # Strict policy: semantics must be computed from core_clip (no local CLIP, no heuristics).
        return ["core_clip"]

    def get_models_used(self, config: Dict[str, Any], metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        # Deterministic: include both the Places365 classifier and the upstream core_clip model mapping.
        out: List[Dict[str, Any]] = []
        if self._last_places_models_used:
            out.extend(self._last_places_models_used)
        if self._last_core_clip_models_used:
            out.extend(self._last_core_clip_models_used)
        return out

    def process(self, frame_manager, frame_indices: List[int], config: Dict[str, Any]) -> Dict[str, Any]:
        """
        BaseModule entrypoint.
        Returns a npz-friendly dict (numeric arrays where possible).
        """
        if len(frame_indices) < 2:
            raise RuntimeError("scene_classification | frame_indices пустой/меньше 2 (no-fallback)")

        # Enforce time-axis source-of-truth
        union_ts = frame_manager.meta.get("union_timestamps_sec")
        if not isinstance(union_ts, list) or len(union_ts) <= max(frame_indices):
            raise RuntimeError("scene_classification | missing/invalid union_timestamps_sec in frames metadata (no-fallback)")

        # Strict dependency: core_clip must exist and must contain required embeddings.
        self._load_core_clip_dependency()

        # Apply lightweight runtime overrides (do not rebuild model by default)
        if config:
            self._apply_runtime_config(config)

        agg = self.classify_with_advanced_features(frame_manager, frame_indices)
        return self._pack_npz_result(agg)

    def _load_core_clip_dependency(self) -> None:
        """
        Load core_clip artifacts required for semantics:
        - frame_embeddings aligned by frame_indices (union domain)
        - scene_*_text_embeddings exported by core_clip
        """
        core = self.load_core_provider("core_clip")
        if core is None:
            raise RuntimeError("scene_classification | core_clip is required but not found (no-fallback)")

        core_idx = core.get("frame_indices")
        core_emb = core.get("frame_embeddings")
        if core_idx is None or core_emb is None:
            raise RuntimeError("scene_classification | core_clip missing frame_indices/frame_embeddings (no-fallback)")

        core_idx = np.asarray(core_idx, dtype=np.int32)
        core_emb = np.asarray(core_emb, dtype=np.float32)
        self._core_clip_frame_indices = core_idx
        self._core_clip_frame_embeddings = core_emb
        self._core_clip_index_map = {int(fi): i for i, fi in enumerate(core_idx.tolist())}
        self._use_core_clip = True

        # Required scene semantics embeddings (exported by core_clip)
        aes = core.get("scene_aesthetic_text_embeddings")
        lux = core.get("scene_luxury_text_embeddings")
        atm = core.get("scene_atmosphere_text_embeddings")
        if aes is None or lux is None or atm is None:
            raise RuntimeError("scene_classification | core_clip missing scene_*_text_embeddings (upgrade core_clip) (no-fallback)")
        self._scene_aesthetic_text_embeddings = np.asarray(aes, dtype=np.float32)
        self._scene_luxury_text_embeddings = np.asarray(lux, dtype=np.float32)
        self._scene_atmosphere_text_embeddings = np.asarray(atm, dtype=np.float32)

        # Capture models_used from core_clip meta for reproducibility chaining.
        meta = core.get("meta")
        if isinstance(meta, dict) and isinstance(meta.get("models_used"), list):
            self._last_core_clip_models_used = meta.get("models_used") or []

    def _apply_runtime_config(self, config: Dict[str, Any]) -> None:
        """Apply safe runtime overrides that don't require model rebuild."""
        try:
            if "min_scene_seconds" in config and config["min_scene_seconds"] is not None:
                self.min_scene_seconds = float(config["min_scene_seconds"])
            if "min_scene_length" in config and config["min_scene_length"] is not None:
                self.min_scene_length_frames = max(1, int(config["min_scene_length"]))
            if "enable_advanced_features" in config and config["enable_advanced_features"] is not None:
                self.enable_advanced_features = bool(config["enable_advanced_features"])
        except Exception as e:
            logger.warning(f"Places365SceneClassifier | _apply_runtime_config | Failed to apply overrides: {e}")

    def _get_core_clip_embedding(self, frame_index: int) -> Optional[np.ndarray]:
        """Fast path: use cached core_clip embeddings when available."""
        if self._core_clip_frame_embeddings is not None and self._core_clip_index_map is not None:
            try:
                pos = self._core_clip_index_map.get(int(frame_index), None)
                if pos is None:
                    return None
                emb = np.asarray(self._core_clip_frame_embeddings[int(pos)], dtype=np.float32)
                return emb
            except Exception:
                return None
        return None

    def _core_clip_probs(self, *, frame_index: int, text_embeddings: np.ndarray) -> np.ndarray:
        """
        Compute softmax probabilities for a set of CLIP text embeddings using core_clip image embeddings.
        Assumes both image embeddings and text_embeddings are in the same space and L2-normalized (core_clip contract).
        """
        if text_embeddings is None:
            raise RuntimeError("scene_classification | core_clip text embeddings are missing (no-fallback)")
        img = self._get_core_clip_embedding(frame_index)
        if img is None:
            raise RuntimeError(f"scene_classification | core_clip embedding missing for frame_index={frame_index} (no-fallback)")
        img = np.asarray(img, dtype=np.float32)
        img = img / (np.linalg.norm(img) + 1e-9)
        te = np.asarray(text_embeddings, dtype=np.float32)
        # (P,) logits
        logits = img @ te.T
        logits = logits - float(np.max(logits))
        exp = np.exp(logits)
        probs = exp / (float(np.sum(exp)) + 1e-9)
        return probs.astype(np.float32)

    def _core_clip_binary_score(
        self,
        *,
        frame_index: int,
        text_embeddings: Optional[np.ndarray],
        pos_indices: Tuple[int, int] = (0, 1),
    ) -> float:
        te = np.asarray(text_embeddings, dtype=np.float32) if text_embeddings is not None else None
        probs = self._core_clip_probs(frame_index=frame_index, text_embeddings=te)
        s = float(probs[int(pos_indices[0])] + probs[int(pos_indices[1])])
        return float(np.clip(s, 0.0, 1.0))

    def _core_clip_atmosphere(self, *, frame_index: int) -> Dict[str, float]:
        te = self._scene_atmosphere_text_embeddings
        probs = self._core_clip_probs(frame_index=frame_index, text_embeddings=np.asarray(te, dtype=np.float32))
        return {
            "cozy": float(probs[0]),
            "scary": float(probs[1]),
            "epic": float(probs[2]),
            "neutral": float(probs[3]),
        }

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
            # stability
            "scene_change_score": f32("scene_change_score"),
            "label_stability": f32("label_stability"),
            # variable-length lists
            "indices": np.asarray([s.get("indices", []) for s in scenes], dtype=object),
            "dominant_places_topk_ids": np.asarray([s.get("dominant_places_topk_ids", []) for s in scenes], dtype=object),
            "dominant_places_topk_probs": np.asarray([s.get("dominant_places_topk_probs", []) for s in scenes], dtype=object),
            # canonical raw mapping for downstream modules
            "scenes": np.asarray(agg, dtype=object),
            # legacy alias (kept)
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

        if self.runtime == "triton":
            return self._classify_triton(frame_manager, frame_indices)

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

    def _preprocess_places365_u8_nhwc_fixed(self, frame: np.ndarray) -> Optional[np.ndarray]:
        """
        Match torchvision pipeline (Resize ~256 on shorter side + CenterCrop 224),
        but output UINT8 NHWC for Triton ensemble input.
        """
        if frame is None or not isinstance(frame, np.ndarray):
            return None

        # FrameManager.get() contract: RGB
        if frame.ndim == 2:
            rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.ndim == 3 and frame.shape[2] == 4:
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            except Exception:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        else:
            rgb = frame

        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 255).astype(np.uint8)

        h, w = int(rgb.shape[0]), int(rgb.shape[1])
        if h <= 0 or w <= 0:
            return None

        # torchvision.transforms.Resize(resize_size) keeps aspect ratio (smaller edge -> resize_size)
        resize_size = int(self.input_size * 1.143)  # same heuristic as inprocess path
        min_edge = min(h, w)
        if min_edge != resize_size:
            scale = float(resize_size) / float(min_edge)
            new_w = max(1, int(round(w * scale)))
            new_h = max(1, int(round(h * scale)))
            rgb = cv2.resize(rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            h, w = int(rgb.shape[0]), int(rgb.shape[1])

        crop = int(self.input_size)
        if h < crop or w < crop:
            # Safety: if resize produced smaller due to rounding, pad by resizing to exact crop.
            rgb = cv2.resize(rgb, (crop, crop), interpolation=cv2.INTER_LINEAR)
            h, w = crop, crop

        y0 = max(0, (h - crop) // 2)
        x0 = max(0, (w - crop) // 2)
        rgb = rgb[y0 : y0 + crop, x0 : x0 + crop, :]

        # Triton expects (1,S,S,3) where S is fixed branch size
        return np.expand_dims(rgb, axis=0)

    def _softmax_np(self, logits: np.ndarray) -> np.ndarray:
        x = np.asarray(logits, dtype=np.float32).reshape(-1)
        x = x - float(np.max(x))
        ex = np.exp(x)
        return (ex / (float(np.sum(ex)) + 1e-9)).astype(np.float32)

    def _classify_triton(self, frame_manager, frame_indices) -> List[Dict[str, Any]]:
        if not isinstance(self._triton_handle, dict) or "client" not in self._triton_handle:
            raise RuntimeError("scene_classification(triton) | triton handle is missing (no-fallback)")

        rp = self._triton_rp or {}
        model_name = str(rp.get("triton_model_name") or self._triton_handle.get("triton_model_name") or "").strip()
        model_version = str(rp.get("triton_model_version") or "1").strip()
        input_name = str(rp.get("triton_input_name") or "INPUT__0").strip()
        input_dt = str(rp.get("triton_input_datatype") or "UINT8").strip().upper()
        output_name = str(rp.get("triton_output_name") or "OUTPUT__0").strip()
        output_dt = str(rp.get("triton_output_datatype") or "FP32").strip().upper()
        if not model_name:
            raise RuntimeError("scene_classification(triton) | missing triton_model_name (no-fallback)")

        client = self._triton_handle["client"]

        raw_predictions: List[Dict[str, Any]] = [None] * len(frame_indices)
        for pos, frame_idx in enumerate(frame_indices):
            frame = frame_manager.get(frame_idx)
            x = self._preprocess_places365_u8_nhwc_fixed(frame)
            if x is None:
                raw_predictions[pos] = None
                continue
            try:
                res = client.infer(
                    model_name=model_name,
                    model_version=model_version,
                    input_name=input_name,
                    input_tensor=x.astype(np.uint8, copy=False),
                    output_name=output_name,
                    datatype=input_dt,
                )
            except Exception as e:
                raise RuntimeError(f"scene_classification(triton) | infer failed: {e}") from e

            logits = np.asarray(res.output, dtype=np.float32).reshape(-1)
            probs = self._softmax_np(logits)

            # Confidence stats
            entropy_val = float(-np.sum(probs * np.log(probs + 1e-8)))
            top_k = int(min(5, probs.shape[0]))
            topk_idx = np.argpartition(-probs, top_k - 1)[:top_k]
            topk_idx = topk_idx[np.argsort(-probs[topk_idx])]
            topk_probs = probs[topk_idx]

            top1_idx = int(topk_idx[0])
            top1_prob_val = float(topk_probs[0])
            top2_prob_val = float(topk_probs[1]) if top_k > 1 else 0.0
            top1_top2_gap = float(max(0.0, top1_prob_val - top2_prob_val))

            label = self.categories[top1_idx] if 0 <= top1_idx < len(self.categories) else f"class_{top1_idx}"
            pred: Dict[str, Any] = {
                "label": label,
                "score": top1_prob_val,
                "entropy": entropy_val,
                "top1_prob": top1_prob_val,
                "top2_prob": top2_prob_val,
                "top1_top2_gap": top1_top2_gap,
                "class_idx": top1_idx,
                "topk_class_indices": [int(i) for i in topk_idx.tolist()],
                "topk_class_probs": [float(v) for v in topk_probs.tolist()],
            }
            raw_predictions[pos] = pred

        return raw_predictions


    def _prepare_frame(self, frame: np.ndarray) -> Optional[torch.Tensor | List[torch.Tensor]]:
        if frame is None or not isinstance(frame, np.ndarray):
            logger.warning("Frame is not a numpy array – skipping")
            return None

        # FrameManager.get() contract: RGB
        if frame.ndim == 2:
            rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.ndim == 3 and frame.shape[2] == 4:
            # Most likely RGBA; keep robust fallback for BGRA just in case.
            try:
                rgb = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            except Exception:
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGRA2RGB)
        else:
            rgb = frame  # assume already RGB

        image = Image.fromarray(rgb.astype(np.uint8))

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
    
    def aggregate_scenes(self, res, *, fps: float, union_timestamps_sec: List[float]) -> Dict[str, Any]:
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
                    "aesthetic_score": float,
                    "luxury_score": float,
                    "atmosphere_sentiment": {"cozy": float, "scary": float, "epic": float, "neutral": float},
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
                "aesthetic_score": [], "luxury_score": [],
                "cozy": [], "scary": [], "epic": [], "neutral": [],
                "labels": [],
            }

        def finalize_segment(label, indices, values, topk_idx_seq, topk_prob_seq):
            """Return aggregated segment dict."""
            if not indices:
                return None
            length_frames = len(indices)
            fps_safe = float(fps) if fps and fps > 0 else 30.0

            # Determine minimal duration in seconds
            if self.min_scene_seconds is not None:
                min_len_s = self.min_scene_seconds
            else:
                # Backwards‑compatible: interpret frame threshold at runtime FPS
                min_len_s = float(self.min_scene_length_frames) / fps_safe

            start_frame = int(indices[0])
            end_frame = int(indices[-1])
            # Time-axis source-of-truth: union timestamps.
            try:
                start_ts = float(union_timestamps_sec[start_frame])
                end_ts = float(union_timestamps_sec[end_frame])
                length_seconds = float(max(0.0, end_ts - start_ts))
            except Exception:
                length_seconds = float(length_frames) / fps_safe

            if length_seconds < min_len_s:
                return None

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
                "mean_aesthetic_score": aesthetic_mean,
                "aesthetic_std": aesthetic_std,
                "aesthetic_frac_high": aesthetic_frac_high,
                "mean_luxury_score": luxury_mean,
                "mean_cozy": float(np.mean(values["cozy"])),
                "mean_scary": float(np.mean(values["scary"])),
                "mean_epic": float(np.mean(values["epic"])),
                "mean_neutral": float(np.mean(values["neutral"])),
                "atmosphere_entropy": atm_entropy,
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
                        scene_id = f"s{len(aggregated):04d}"
                        aggregated[scene_id] = segment
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
            current_values["aesthetic_score"].append(adv["aesthetic_score"])
            current_values["luxury_score"].append(adv["luxury_score"])
            atm = adv["atmosphere_sentiment"]
            current_values["cozy"].append(atm["cozy"])
            current_values["scary"].append(atm["scary"])
            current_values["epic"].append(atm["epic"])
            current_values["neutral"].append(atm["neutral"])

        # Final segment
        if current_label is not None:
            segment = finalize_segment(
                current_label, current_indices, current_values,
                current_topk_indices, current_topk_probs,
            )
            if segment:
                scene_id = f"s{len(aggregated):04d}"
                aggregated[scene_id] = segment

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
            union_ts = [float(x) for x in (frame_manager.meta.get("union_timestamps_sec") or [])]
            return self.aggregate_scenes(results, fps=fps, union_timestamps_sec=union_ts)

        # Allocate output dict indexed by frame ID
        results: Dict[int, Dict[str, Any]] = {}

        for frame_idx, pred in zip(frame_indices, base_predictions):
            if pred is None:
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
            # semantics strictly from core_clip (no heuristics, no local CLIP)
            advanced["aesthetic_score"] = self._core_clip_binary_score(
                frame_index=frame_idx, text_embeddings=self._scene_aesthetic_text_embeddings, pos_indices=(0, 1)
            )
            advanced["luxury_score"] = self._core_clip_binary_score(
                frame_index=frame_idx, text_embeddings=self._scene_luxury_text_embeddings, pos_indices=(0, 1)
            )
            advanced["atmosphere_sentiment"] = self._core_clip_atmosphere(frame_index=frame_idx)

            results[frame_idx] = {
                "predictions": pred,
                "advanced_features": advanced
            }

        fps = getattr(frame_manager, "fps", 30.0)
        union_ts = [float(x) for x in (frame_manager.meta.get("union_timestamps_sec") or [])]
        agg_result = self.aggregate_scenes(results, fps=fps, union_timestamps_sec=union_ts)

        return agg_result

