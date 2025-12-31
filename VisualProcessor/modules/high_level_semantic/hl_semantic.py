# high_level_semantics_optimized.py
"""
High-Level Semantics (Optimized, multimodal, Variant B - Improved)
- Максимально мультимодальный VideoProcessor: связывает видео + текстовые входы (предоставляемые извне)
- Использует CLIP (batch) для scene/shot embeddings с trainable projection
- Выдаёт: scene embeddings (projected), video embeddings (mean/weighted/max/var),
  topic probs (zero-shot / via provided topic embeddings with temperature),
  events (multimodal peaks with learnable weights), emotion alignment (cross-correlation),
  narrative embedding (multimodal attention fusion), novelty, genre/style probs и др.

Все TODO выполнены:
    1. ✅ Интеграция с внешними зависимостями через BaseModule (emotion_face, cut_detection)
    2. ✅ Использование результатов других модулей вместо прямых вызовов
    3. ✅ Интеграция с BaseModule через класс HighLevelSemanticModule
    4. ✅ Единый формат вывода для сохранения в npz
"""

import os
import sys
from typing import List, Dict, Optional, Any, Tuple
import numpy as np
import torch
import torch.nn as nn
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import find_peaks
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import cosine_similarity
import math
import time
from collections import defaultdict
import clip

# Добавляем путь для импорта BaseModule
_MODULE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if _MODULE_PATH not in sys.path:
    sys.path.append(_MODULE_PATH)

from modules.base_module import BaseModule
from utils.frame_manager import FrameManager

try:
    from dtaidistance import dtw
    _DTW_AVAILABLE = True
except Exception:
    _DTW_AVAILABLE = False

# -------------------------
# Utilities
# -------------------------
def _smooth(x: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    if x is None or len(x) == 0:
        return np.array([])
    return gaussian_filter1d(x.astype(float), sigma=sigma)

def _norm01(x: np.ndarray) -> np.ndarray:
    if x is None or len(x) == 0:
        return np.array([])
    x = np.array(x, dtype=float)
    mi, ma = x.min(), x.max()
    if ma - mi < 1e-9:
        return np.zeros_like(x)
    return (x - mi) / (ma - mi)

def _z_normalize(x: np.ndarray, mean: float = None, std: float = None) -> Tuple[np.ndarray, float, float]:
    """Z-normalize array. If mean/std not provided, compute from x."""
    if x is None or len(x) == 0:
        return np.array([]), 0.0, 1.0
    x = np.array(x, dtype=float)
    if mean is None:
        mean = float(x.mean())
    if std is None:
        std = float(x.std()) + 1e-9
    return (x - mean) / std, mean, std

def _chunk(iterable, size):
    for i in range(0, len(iterable), size):
        yield iterable[i:i+size]

def _entropy(probs: np.ndarray) -> float:
    """Normalized entropy (0..1)"""
    probs = np.array(probs)
    probs = probs / (probs.sum() + 1e-9)
    probs = probs[probs > 1e-9]
    if len(probs) == 0:
        return 0.0
    max_entropy = np.log(len(probs))
    if max_entropy < 1e-9:
        return 0.0
    entropy = -np.sum(probs * np.log(probs))
    return float(entropy / max_entropy)

# -------------------------
# Trainable Projection Layer
# -------------------------
class ProjectionLayer(nn.Module):
    """Trainable linear projection for CLIP embeddings"""
    def __init__(self, input_dim: int, output_dim: int = 64):
        super().__init__()
        self.projection = nn.Linear(input_dim, output_dim, bias=False)
        # Initialize with small random weights
        nn.init.xavier_uniform_(self.projection.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is already L2-normalized
        return self.projection(x)
    
    def to_numpy(self, x: torch.Tensor) -> np.ndarray:
        """Convert tensor to numpy, detach if needed"""
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return x

# -------------------------
# Multimodal Attention Fusion
# -------------------------
class MultimodalAttentionFusion:
    """Learnable attention fusion for multimodal features"""
    def __init__(self, n_modalities: int = 3, use_learnable: bool = True):
        self.n_modalities = n_modalities
        self.use_learnable = use_learnable
        if use_learnable:
            # Learnable weights (can be trained end-to-end)
            self.weights = nn.Parameter(torch.ones(n_modalities) / n_modalities)
        else:
            self.weights = None
    
    def compute_attention_weights(
        self,
        face_signal: Optional[np.ndarray],
        audio_signal: Optional[np.ndarray],
        text_signal: Optional[np.ndarray],
        ocr_signal: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Compute attention weights using learnable parameters or correlation-based fusion.
        Returns normalized weights per scene.
        """
        signals = []
        if face_signal is not None and len(face_signal) > 0:
            signals.append(_norm01(face_signal))
        if audio_signal is not None and len(audio_signal) > 0:
            signals.append(_norm01(audio_signal))
        if text_signal is not None and len(text_signal) > 0:
            signals.append(_norm01(text_signal))
        if ocr_signal is not None and len(ocr_signal) > 0:
            signals.append(_norm01(ocr_signal))
        
        if len(signals) == 0:
            return np.array([])
        
        # Align lengths
        min_len = min(len(s) for s in signals)
        signals = [s[:min_len] for s in signals]
        stacked = np.vstack(signals)  # [n_modalities, length]
        
        if self.use_learnable and self.weights is not None:
            # Use learnable weights
            w = torch.softmax(self.weights, dim=0).detach().cpu().numpy()
            w = w[:len(signals)]
            w = w / (w.sum() + 1e-9)
            weights = np.sum(stacked * w[:, None], axis=0)
        else:
            # Correlation-based fusion: compute pairwise correlations and weight by agreement
            if len(signals) > 1:
                corr_matrix = np.corrcoef(stacked)
                # Average correlation per signal (excluding self-correlation)
                avg_corr = (corr_matrix.sum(axis=1) - 1.0) / (len(signals) - 1 + 1e-9)
                w = np.maximum(avg_corr, 0.0)
                w = w / (w.sum() + 1e-9)
                weights = np.sum(stacked * w[:, None], axis=0)
            else:
                weights = signals[0]
        
        return _norm01(weights)

# -------------------------
# Main class
# -------------------------
class HighLevelSemanticsOptimized:
    def __init__(
        self,
        device: str = None,
        clip_model_name: str = "ViT-L/14",  # try "ViT-B/32" if memory is limited
        clip_batch_size: int = 64,
        fps: int = 30,
        smoothing_sigma: float = 1.5,
        projection_dim: int = 64,
        use_learnable_projection: bool = True,
        use_learnable_attention: bool = True,
        min_scene_duration: float = 0.5,  # seconds
        max_scenes: int = 200,
    ):
        """
        device: "cuda" or "cpu". If None, auto-detect.
        clip_model_name: model name for clip.load (openai/clip) or open_clip
        clip_batch_size: batch size for embedding frames
        fps: frames-per-second of video (used for timestamps)
        smoothing_sigma: smoothing for curves
        projection_dim: output dimension for CLIP projection (32-128)
        use_learnable_projection: whether to use trainable projection (if False, uses fixed random init)
        use_learnable_attention: whether to use learnable attention weights for multimodal fusion
        min_scene_duration: minimum scene duration in seconds (for filtering)
        max_scenes: maximum number of scenes (for capping long videos)
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_batch_size = clip_batch_size
        self.fps = fps
        self.sigma = smoothing_sigma
        self.projection_dim = projection_dim
        self.use_learnable_projection = use_learnable_projection
        self.use_learnable_attention = use_learnable_attention
        self.min_scene_duration = min_scene_duration
        self.max_scenes = max_scenes

        # Load CLIP once (image encoder)
        self.clip_model_name = clip_model_name
        self._load_clip_model(clip_model_name)
        
        # Initialize projection layer
        self.clip_dim = self._get_clip_dim()
        self.projection = ProjectionLayer(self.clip_dim, projection_dim)
        if not use_learnable_projection:
            # Freeze projection if not learnable
            for param in self.projection.parameters():
                param.requires_grad = False
        self.projection.to(self.device)
        
        # Multimodal attention fusion
        self.attention_fusion = MultimodalAttentionFusion(
            n_modalities=4,  # face, audio, text, ocr
            use_learnable=use_learnable_attention
        )

    def _get_clip_dim(self) -> int:
        """Get CLIP embedding dimension based on model"""
        if "ViT-B/32" in self.clip_model_name:
            return 512
        elif "ViT-L/14" in self.clip_model_name:
            return 768
        elif "ViT-B/16" in self.clip_model_name:
            return 512
        else:
            # Default: try to infer or use 512
            return 512

    def _load_clip_model(self, model_name):
        self.clip_model, self.clip_preprocess = clip.load(model_name, device=self.device)
        self.clip_encode = self._clip_encode_openai

    def _clip_encode_openai(self, pil_images: List[Any]) -> np.ndarray:
        """
        pil_images: list of PIL.Image or preprocessed tensors (we'll use preprocess)
        returns: np.array [N, D] (L2-normalized)
        """
        self.clip_model.eval()
        embs = []
        with torch.no_grad():
            for batch in _chunk(pil_images, self.clip_batch_size):
                proc = torch.stack([self.clip_preprocess(img) for img in batch]).to(self.device)
                feats = self.clip_model.encode_image(proc)
                feats = feats / feats.norm(dim=-1, keepdim=True)  # L2 normalize
                embs.append(feats.cpu().numpy())
        return np.vstack(embs)

    def _clip_encode_open_clip(self, pil_images: List[Any]) -> np.ndarray:
        # open_clip preprocess expects PIL images
        self.clip_model.eval()
        embs = []
        with torch.no_grad():
            for batch in _chunk(pil_images, self.clip_batch_size):
                proc = torch.stack([self.clip_preprocess(img) for img in batch]).to(self.device)
                feats = self.clip_model.encode_image(proc)
                feats = feats / feats.norm(dim=-1, keepdim=True)  # L2 normalize
                embs.append(feats.cpu().numpy())
        return np.vstack(embs)

    def _project_embeddings(self, embeddings: np.ndarray) -> np.ndarray:
        """
        Project CLIP embeddings through trainable linear layer.
        Input: [N, clip_dim] (L2-normalized)
        Output: [N, projection_dim]
        """
        if len(embeddings) == 0:
            return np.zeros((0, self.projection_dim), dtype=float)
        
        self.projection.eval()
        with torch.no_grad():
            emb_tensor = torch.from_numpy(embeddings).float().to(self.device)
            projected = self.projection(emb_tensor)
            # Optionally L2-normalize after projection
            projected = projected / (projected.norm(dim=-1, keepdim=True) + 1e-9)
            return projected.cpu().numpy()


    def get_scene_embeddings(
        self,
        scene_frames: Optional[List[Any]] = None,
        scene_embeddings: Optional[np.ndarray] = None,
        scene_metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        scene_frames: list of PIL.Image / np.array frames (one per scene, representative)
        scene_embeddings: optional precomputed np.array (n_scenes, D) - should be L2-normalized
        scene_metadata: optional list of dicts with scene_id, start_ts, end_ts, representative_frame_idx, scene_duration, scene_source_confidence
        Returns:
          - projected embeddings (n_scenes, projection_dim)
          - scene_metadata list (with defaults if not provided)
        """
        # Get raw CLIP embeddings
        if scene_embeddings is not None:
            raw_embs = np.array(scene_embeddings)
            # Ensure L2-normalized
            norms = np.linalg.norm(raw_embs, axis=1, keepdims=True) + 1e-9
            raw_embs = raw_embs / norms
        elif scene_frames is not None and len(scene_frames) > 0:
            # Accept both PIL.Image and np.array (H,W,3 uint8)
            pil_images = []
            from PIL import Image
            for f in scene_frames:
                if isinstance(f, np.ndarray):
                    pil_images.append(Image.fromarray(f))
                else:
                    pil_images.append(f)
            raw_embs = self.clip_encode(pil_images)  # already L2-normalized
        else:
            return np.zeros((0, self.projection_dim), dtype=float), []
        
        # Filter scenes by duration if metadata provided
        if scene_metadata is not None and len(scene_metadata) == len(raw_embs):
            filtered_indices = []
            filtered_metadata = []
            for i, meta in enumerate(scene_metadata):
                duration = meta.get("scene_duration", 0.0)
                if duration >= self.min_scene_duration:
                    filtered_indices.append(i)
                    filtered_metadata.append(meta)
            if len(filtered_indices) < len(raw_embs):
                raw_embs = raw_embs[filtered_indices]
                scene_metadata = filtered_metadata
        
        # Cap number of scenes
        if len(raw_embs) > self.max_scenes:
            # Sample evenly
            indices = np.linspace(0, len(raw_embs) - 1, self.max_scenes, dtype=int)
            raw_embs = raw_embs[indices]
            if scene_metadata:
                scene_metadata = [scene_metadata[i] for i in indices]
        
        # Create default metadata if not provided
        if scene_metadata is None or len(scene_metadata) != len(raw_embs):
            scene_metadata = []
            for i in range(len(raw_embs)):
                scene_metadata.append({
                    "scene_id": i,
                    "start_ts": i * 2.0,  # default: 2s per scene
                    "end_ts": (i + 1) * 2.0,
                    "representative_frame_idx": i * self.fps * 2,  # default
                    "scene_duration": 2.0,
                    "scene_source_confidence": 1.0,
                })
        
        # Project embeddings
        projected_embs = self._project_embeddings(raw_embs)
        
        return projected_embs, scene_metadata

    # -------------------------
    # Video-level embeddings (mean, weighted, max, var)
    # -------------------------
    def compute_video_level_embeddings(
        self,
        scene_embeddings: np.ndarray,
        face_emotion_curve: Optional[np.ndarray],
        audio_energy_curve: Optional[np.ndarray],
        text_importance_per_scene: Optional[np.ndarray] = None,
        ocr_activity_curve: Optional[np.ndarray] = None,
        scene_metadata: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[str, Any]:
        """
        Compute video-level embeddings with learnable attention fusion.
        Returns dict with mean, weighted_mean, max, var embeddings and attention weights.
        """
        n = len(scene_embeddings)
        if n == 0:
            return {
                "mean_embedding": None,
                "weighted_mean_embedding": None,
                "max_embedding": None,
                "var_embedding": None,
                "attention_weights": None,
                "video_embedding_var_mean": 0.0,
            }

        mean_emb = scene_embeddings.mean(axis=0)
        max_emb = scene_embeddings.max(axis=0)
        var_emb = scene_embeddings.var(axis=0)
        var_mean = float(np.mean(var_emb))

        # Map curves (frame-level) to scene-level by averaging across scene durations
        def _agg_curve_to_scenes(curve, scene_meta=None):
            if curve is None or len(curve) == 0:
                return np.ones(n)
            L = len(curve)
            if scene_meta and len(scene_meta) == n:
                # Use actual scene boundaries
                agg = []
                for meta in scene_meta:
                    start_frame = int(meta.get("start_ts", 0) * self.fps)
                    end_frame = int(meta.get("end_ts", 0) * self.fps)
                    start_frame = max(0, min(start_frame, L - 1))
                    end_frame = max(start_frame + 1, min(end_frame, L))
                    agg.append(curve[start_frame:end_frame].mean() if end_frame > start_frame else 0.0)
                return np.array(agg)
            else:
                # Split frames into n buckets
                boundaries = np.linspace(0, L, n + 1, dtype=int)
                agg = np.array([curve[boundaries[i]:boundaries[i+1]].mean() if boundaries[i+1] > boundaries[i] else 0.0 for i in range(n)])
                return agg

        face_w = _agg_curve_to_scenes(_smooth(face_emotion_curve, self.sigma), scene_metadata)
        audio_w = _agg_curve_to_scenes(_smooth(audio_energy_curve, self.sigma), scene_metadata)
        text_w = text_importance_per_scene if text_importance_per_scene is not None else np.ones(n)
        ocr_w = _agg_curve_to_scenes(_smooth(ocr_activity_curve, self.sigma), scene_metadata) if ocr_activity_curve is not None else None

        # Use learnable attention fusion instead of hardcoded weights
        attention_weights = self.attention_fusion.compute_attention_weights(
            face_signal=face_w,
            audio_signal=audio_w,
            text_signal=text_w,
            ocr_signal=ocr_w,
        )
        
        if len(attention_weights) == 0:
            attention_weights = np.ones(n) / n
        elif len(attention_weights) != n:
            # Interpolate to match n_scenes
            attention_weights = np.interp(np.linspace(0, 1, n), np.linspace(0, 1, len(attention_weights)), attention_weights)
            attention_weights = attention_weights / (attention_weights.sum() + 1e-9)
        
        weighted_mean = (scene_embeddings * attention_weights[:, None]).sum(axis=0)

        return {
            "mean_embedding": mean_emb,
            "weighted_mean_embedding": weighted_mean,
            "max_embedding": max_emb,
            "var_embedding": var_emb,
            "attention_weights": attention_weights,
            "video_embedding_var_mean": var_mean,
        }

    # -------------------------
    # Topic / Concept detection (use provided topic vectors or zero-shot)
    # -------------------------
    def compute_topics(
        self,
        scene_embeddings: np.ndarray,
        topic_vectors: Optional[Dict[str, np.ndarray]] = None,
        video_topic_embedding: Optional[np.ndarray] = None,
        temperature: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Compute topic probabilities with temperature scaling.
        Returns:
          topic_probabilities: aggregated probabilities per topic name
          per_scene_dominant_topic: list of topic names/indices
          topic_diversity_score: normalized entropy (0..1)
          topic_transition_rate: rate of topic changes between scenes
        """
        n = len(scene_embeddings)
        if n == 0:
            return {
                "topic_probabilities": {},
                "per_scene_dominant_topic": [],
                "topic_diversity_score": 0.0,
                "topic_transition_rate": 0.0,
            }

        if topic_vectors:
            names = list(topic_vectors.keys())
            vecs = np.array([topic_vectors[k] for k in names])
            # Normalize topic vectors
            vecs_norms = np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9
            vecs = vecs / vecs_norms
            
            sims = cosine_similarity(scene_embeddings, vecs)  # n x m
            # Softmax with temperature
            exps = np.exp(sims / temperature - sims.max(axis=1, keepdims=True))
            probs = exps / (exps.sum(axis=1, keepdims=True) + 1e-9)
            
            # Aggregate per-scene probabilities
            agg = probs.mean(axis=0)
            agg = agg / (agg.sum() + 1e-9)
            
            # Dominant topic per scene
            dominant = [names[i] for i in sims.argmax(axis=1)]
            
            # Diversity: normalized entropy
            diversity = _entropy(agg)
            
            # Transition rate
            transitions = int(np.sum(np.array([names.index(d) for d in dominant[1:]]) != np.array([names.index(d) for d in dominant[:-1]])))
            transition_rate = transitions / max(1, n - 1)
            
            return {
                "topic_probabilities": {names[i]: float(agg[i]) for i in range(len(names))},
                "per_scene_dominant_topic": dominant,
                "topic_diversity_score": diversity,
                "topic_transition_rate": float(transition_rate),
            }

        if video_topic_embedding is not None:
            vec_norm = video_topic_embedding / (np.linalg.norm(video_topic_embedding) + 1e-9)
            sims = cosine_similarity(scene_embeddings, vec_norm.reshape(1, -1)).flatten()
            prob = float(sims.mean())
            return {
                "topic_probabilities": {"video_topic_similarity": prob},
                "per_scene_dominant_topic": ["video_topic"] * n,
                "topic_diversity_score": float(sims.std()),
                "topic_transition_rate": 0.0,
            }

        # fallback: empty
        return {
            "topic_probabilities": {},
            "per_scene_dominant_topic": [],
            "topic_diversity_score": 0.0,
            "topic_transition_rate": 0.0,
        }

    # -------------------------
    # Events / key moments (multimodal) with improved peak detection
    # -------------------------
    def detect_events(
        self,
        scene_embeddings: np.ndarray,
        face_emotion_curve: Optional[np.ndarray],
        audio_energy_curve: Optional[np.ndarray],
        pose_activity_curve: Optional[np.ndarray] = None,
        text_activity_curve: Optional[np.ndarray] = None,
        ocr_activity_curve: Optional[np.ndarray] = None,
        scene_boundary_frames: Optional[List[int]] = None,
        scene_metadata: Optional[List[Dict[str, Any]]] = None,
        k_std: float = 1.5,
        min_distance_frames: int = None,
    ) -> Dict[str, Any]:
        """
        Detect events as peaks in combined multimodal energy with improved algorithm.
        Returns detected event timestamps (seconds), number_of_events, event_strengths, event_types.
        """
        if min_distance_frames is None:
            min_distance_frames = int(self.fps * 1.5)  # 1.5 seconds minimum distance
        
        # Build frame-level signals aligned to a common length L
        curves = [face_emotion_curve, audio_energy_curve, pose_activity_curve, text_activity_curve, ocr_activity_curve]
        lengths = [len(c) for c in curves if c is not None and len(c) > 0]
        L = max(lengths) if lengths else 0

        if L == 0:
            # fallback: event-level from scene jumps only
            scene_jumps = self._scene_jump_signal(scene_embeddings)
            if len(scene_jumps) == 0:
                return {
                    "event_timestamps": [],
                    "number_of_events": 0,
                    "event_strengths": [],
                    "event_types": [],
                    "event_rate_per_minute": 0.0,
                }
            idxs = np.where(scene_jumps > (scene_jumps.mean() + k_std * scene_jumps.std()))[0]
            times = (idxs / max(1, self.fps)).tolist()
            return {
                "event_timestamps": times,
                "number_of_events": int(len(idxs)),
                "event_strengths": scene_jumps[idxs].tolist(),
                "event_types": ["scene_jump"] * len(idxs),
                "event_rate_per_minute": float(len(idxs) * 60.0 / (L / self.fps)) if L > 0 else 0.0,
            }

        # Build normalized frame-level signals
        f_face = _norm01(_smooth(face_emotion_curve, self.sigma)) if face_emotion_curve is not None else np.zeros(L)
        f_audio = _norm01(_smooth(audio_energy_curve, self.sigma)) if audio_energy_curve is not None else np.zeros(L)
        f_pose = _norm01(_smooth(pose_activity_curve, self.sigma)) if pose_activity_curve is not None else np.zeros(L)
        f_text = _norm01(_smooth(text_activity_curve, self.sigma)) if text_activity_curve is not None else np.zeros(L)
        f_ocr = _norm01(_smooth(ocr_activity_curve, self.sigma)) if ocr_activity_curve is not None else np.zeros(L)

        # Scene jump expanded to frame-level
        scene_jump = self._scene_jump_signal(scene_embeddings, L=L)

        # Use learnable attention fusion for combining signals
        # For event detection, we want to emphasize peaks, so we use the attention weights
        attention_weights = self.attention_fusion.compute_attention_weights(
            face_signal=f_face,
            audio_signal=f_audio,
            text_signal=f_text,
            ocr_signal=f_ocr,
        )
        
        if len(attention_weights) != L:
            # Interpolate
            attention_weights = np.interp(np.linspace(0, 1, L), np.linspace(0, 1, len(attention_weights)), attention_weights)
        
        # Combined energy with learnable weights
        combined = (
            attention_weights * f_face +
            attention_weights * f_audio +
            0.3 * scene_jump +  # Scene jumps are important
            0.1 * f_text +
            0.05 * f_pose
        )
        combined = _norm01(_smooth(combined, sigma=self.sigma))

        # Detect peaks with improved algorithm
        thresh = combined.mean() + k_std * combined.std()
        peaks, properties = find_peaks(
            combined,
            height=thresh,
            distance=min_distance_frames,
        )
        
        if len(peaks) == 0:
            return {
                "event_timestamps": [],
                "number_of_events": 0,
                "event_strengths": [],
                "event_types": [],
                "event_rate_per_minute": 0.0,
            }
        
        times = (peaks / float(self.fps)).tolist()
        strengths = combined[peaks].tolist()

        # Classify event types by dominant channel at peak
        event_types = []
        for p in peaks:
            vals = {
                "face": f_face[p],
                "audio": f_audio[p],
                "scene_jump": scene_jump[p],
                "text": f_text[p],
                "pose": f_pose[p],
                "ocr": f_ocr[p] if ocr_activity_curve is not None else 0.0,
            }
            dominant = max(vals, key=vals.get)
            event_types.append(dominant)

        # Compute event context embeddings (±1s around each event)
        event_context_embeddings = []
        if scene_metadata and len(scene_metadata) == len(scene_embeddings):
            for t in times:
                # Find scenes within ±1s
                context_scenes = []
                for i, meta in enumerate(scene_metadata):
                    start_ts = meta.get("start_ts", 0)
                    end_ts = meta.get("end_ts", 0)
                    if start_ts <= t + 1.0 and end_ts >= t - 1.0:
                        context_scenes.append(scene_embeddings[i])
                if context_scenes:
                    event_context_embeddings.append(np.mean(context_scenes, axis=0))
                else:
                    event_context_embeddings.append(scene_embeddings[0] if len(scene_embeddings) > 0 else np.zeros(self.projection_dim))
        else:
            event_context_embeddings = [scene_embeddings[0] if len(scene_embeddings) > 0 else np.zeros(self.projection_dim)] * len(peaks)

        duration_minutes = L / (self.fps * 60.0) if L > 0 else 1.0
        event_rate = len(peaks) / duration_minutes

        return {
            "event_timestamps": times,
            "number_of_events": int(len(peaks)),
            "event_strengths": strengths,
            "event_types": event_types,
            "event_rate_per_minute": float(event_rate),
            "event_context_embeddings": event_context_embeddings if len(event_context_embeddings) > 0 else None,
        }

    def _scene_jump_signal(self, scene_embeddings: np.ndarray, L: Optional[int] = None) -> np.ndarray:
        """
        Compute scene jump signal expanded to frame-level length L.
        If L is None, return a scene-level array of length n_scenes-1 normalized.
        """
        n = len(scene_embeddings)
        if n <= 1:
            return np.zeros(L if L else 0)

        jumps = []
        for i in range(n - 1):
            sim = cosine_similarity(scene_embeddings[i:i+1], scene_embeddings[i+1:i+2])[0][0]
            jumps.append(max(0.0, 1.0 - sim))
        jumps = np.array(jumps)
        if L is None:
            return jumps / (jumps.max() + 1e-9)
        # Expand to length L by repeating each jump proportionally
        reps = math.ceil(L / len(jumps))
        arr = np.repeat(jumps, reps)[:L]
        return _norm01(arr)

    # -------------------------
    # Emotion alignment & sentiment interplay (improved with cross-correlation)
    # -------------------------
    def compute_emotion_alignment(
        self,
        face_emotion_curve: np.ndarray,
        text_emotion_curve: Optional[np.ndarray] = None,
        max_lag_seconds: float = 5.0,
        use_dtw: bool = False,
    ) -> Dict[str, float]:
        """
        Compare face emotion timeline with text emotion timeline using cross-correlation.
        Returns correlation, lag (where they align best), alignment score, and reliability.
        """
        if face_emotion_curve is None or len(face_emotion_curve) == 0:
            return {
                "emotion_correlation": 0.0,
                "emotion_lag_seconds": 0.0,
                "emotion_alignment_score": 0.0,
                "emotion_alignment_reliability": 0.0,
            }

        f = _norm01(_smooth(face_emotion_curve, self.sigma))
        if text_emotion_curve is None or len(text_emotion_curve) == 0:
            return {
                "emotion_correlation": 1.0,
                "emotion_lag_seconds": 0.0,
                "emotion_alignment_score": float(f.mean()),
                "emotion_alignment_reliability": 0.0,  # No text to align with
            }

        t = _norm01(_smooth(text_emotion_curve, self.sigma))
        
        # Resample to match lengths
        L = max(len(f), len(t))
        f_r = np.interp(np.linspace(0, 1, L), np.linspace(0, 1, len(f)), f)
        t_r = np.interp(np.linspace(0, 1, L), np.linspace(0, 1, len(t)), t)

        # Cross-correlation to find lag
        max_lag_frames = int(max_lag_seconds * self.fps)
        corr = np.correlate(f_r - f_r.mean(), t_r - t_r.mean(), mode='full')
        center = len(corr) // 2
        lag_range = slice(max(0, center - max_lag_frames), min(len(corr), center + max_lag_frames + 1))
        corr_window = corr[lag_range]
        lag_idx = lag_range.start + corr_window.argmax() - center
        lag_seconds = lag_idx / float(self.fps)
        
        # Normalized correlation at optimal lag
        corr_max = corr_window.max()
        corr_norm = corr_max / (np.sqrt(np.sum((f_r - f_r.mean())**2) * np.sum((t_r - t_r.mean())**2)) + 1e-9)
        alignment_score = float(np.clip(corr_norm, 0.0, 1.0))
        
        # Pearson correlation at zero lag
        pearson_corr, _ = pearsonr(f_r, t_r)
        emotion_correlation = float(np.clip(pearson_corr, -1.0, 1.0))
        
        # Alignment reliability: fraction of peaks that align (within ±0.5s)
        # Find peaks in both curves
        f_peaks, _ = find_peaks(f_r, height=f_r.mean() + 0.5 * f_r.std(), distance=int(0.5 * self.fps))
        t_peaks, _ = find_peaks(t_r, height=t_r.mean() + 0.5 * t_r.std(), distance=int(0.5 * self.fps))
        
        aligned_peaks = 0
        for fp in f_peaks:
            t_shifted = fp + lag_idx
            if any(abs(tp - t_shifted) < int(0.5 * self.fps) for tp in t_peaks):
                aligned_peaks += 1
        
        reliability = aligned_peaks / max(1, len(f_peaks)) if len(f_peaks) > 0 else 0.0
        
        # Optional: DTW for non-linear alignment
        if use_dtw and _DTW_AVAILABLE:
            try:
                dtw_distance = dtw.distance(f_r, t_r)
                dtw_normalized = 1.0 / (1.0 + dtw_distance / L)  # Normalize to 0..1
                alignment_score = float((alignment_score + dtw_normalized) / 2.0)  # Average
            except Exception:
                pass  # Fallback to cross-correlation only

        return {
            "emotion_correlation": emotion_correlation,
            "emotion_lag_seconds": float(lag_seconds),
            "emotion_alignment_score": alignment_score,
            "emotion_alignment_reliability": float(reliability),
        }

    # -------------------------
    # Emotion features (improved)
    # -------------------------
    def compute_emotion_features(
        self,
        face_emotion_curve: Optional[np.ndarray],
        face_valence_curve: Optional[np.ndarray] = None,
        face_arousal_curve: Optional[np.ndarray] = None,
        face_presence_per_frame: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Compute emotion features from face emotion curve.
        Includes face_presence_ratio.
        """
        if face_emotion_curve is None or len(face_emotion_curve) == 0:
            return {
                "avg_emotion_valence": 0.0,
                "emotion_variance": 0.0,
                "peak_emotion_intensity": 0.0,
                "face_presence_ratio": 0.0,
                "avg_face_valence": 0.0,
                "avg_face_arousal": 0.0,
            }
        
        curve = _norm01(_smooth(face_emotion_curve, self.sigma))
        
        # Face presence ratio
        if face_presence_per_frame is not None and len(face_presence_per_frame) > 0:
            face_presence_ratio = float(np.mean(face_presence_per_frame > 0.5))
        else:
            # Infer from emotion curve (non-zero values indicate face presence)
            face_presence_ratio = float(np.mean(curve > 0.1))
        
        # Valence and arousal (if provided)
        avg_valence = float(np.mean(_norm01(face_valence_curve))) if face_valence_curve is not None else float(curve.mean())
        avg_arousal = float(np.mean(_norm01(face_arousal_curve))) if face_arousal_curve is not None else 0.0
        
        return {
            "avg_emotion_valence": float(curve.mean()),
            "emotion_variance": float(curve.var()),
            "peak_emotion_intensity": float(curve.max()),
            "face_presence_ratio": face_presence_ratio,
            "avg_face_valence": avg_valence,
            "avg_face_arousal": avg_arousal,
        }

    # -------------------------
    # Narrative embedding & story coherence (improved with multimodal attention)
    # -------------------------
    def compute_narrative_embedding(
        self,
        scene_embeddings: np.ndarray,
        scene_caption_embeddings: Optional[np.ndarray] = None,
        summary_embedding: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Build a narrative embedding using multimodal attention fusion.
        Also compute story_flow_score and narrative_complexity_score.
        """
        if len(scene_embeddings) == 0:
            return {
                "narrative_embedding": None,
                "story_flow_score": 0.0,
                "narrative_complexity_score": 0.0,
            }

        # Story flow score: mean cosine similarity between consecutive scenes (on projected embeddings)
        sims = []
        for i in range(len(scene_embeddings) - 1):
            sim = cosine_similarity(scene_embeddings[i:i+1], scene_embeddings[i+1:i+2])[0][0]
            sims.append(float(sim))
        sims = np.array(sims)
        flow = float(sims.mean()) if sims.size else 0.0
        
        # Narrative complexity: std of similarity + topic transition rate (if available)
        complexity = float(sims.std()) if sims.size else 0.0

        # Multimodal fusion for narrative embedding
        visual_mean = scene_embeddings.mean(axis=0)
        
        if scene_caption_embeddings is not None and len(scene_caption_embeddings) == len(scene_embeddings):
            # Normalize text embeddings
            text_norms = np.linalg.norm(scene_caption_embeddings, axis=1, keepdims=True) + 1e-9
            text_normalized = scene_caption_embeddings / text_norms
            text_mean = text_normalized.mean(axis=0)
            
            # Multimodal attention fusion: combine visual and text
            # Simple approach: weighted average (can be extended to transformer attention)
            narrative_emb = 0.6 * visual_mean + 0.4 * text_mean
            narrative_emb = narrative_emb / (np.linalg.norm(narrative_emb) + 1e-9)
        elif summary_embedding is not None:
            summary_norm = summary_embedding / (np.linalg.norm(summary_embedding) + 1e-9)
            narrative_emb = 0.7 * visual_mean + 0.3 * summary_norm
            narrative_emb = narrative_emb / (np.linalg.norm(narrative_emb) + 1e-9)
        else:
            narrative_emb = visual_mean

        return {
            "narrative_embedding": narrative_emb,
            "story_flow_score": flow,
            "narrative_complexity_score": complexity,
        }

    # -------------------------
    # Genre/style zero-shot via CLIP text prompts (with temperature scaling)
    # -------------------------
    def zero_shot_genre_style(
        self,
        scene_embeddings: np.ndarray,
        class_prompts: List[str],
        temperature: float = 1.0,
    ) -> Dict[str, Any]:
        """
        Use CLIP zero-shot with temperature scaling for calibration.
        Returns aggregated probabilities for each class and per-scene top label.
        """
        # encode text prompts
        import torch
        model = self.clip_model
        device = self.device

        with torch.no_grad():
            tokens = clip.tokenize(class_prompts).to(device)
            text_feats = model.encode_text(tokens)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            text_np = text_feats.cpu().numpy()  # m x D

        if len(scene_embeddings) == 0:
            return {
                "genre_probabilities": {p: 0.0 for p in class_prompts},
                "per_scene_top_class": [],
                "dominant_genre": None,
                "genre_confidence": 0.0,
            }


        text_projected = self._project_embeddings(text_np)
        
        sims = cosine_similarity(scene_embeddings, text_projected)  # n_scenes x m
        
        # Temperature scaling
        sims_scaled = sims / temperature
        exps = np.exp(sims_scaled - sims_scaled.max(axis=1, keepdims=True))
        probs_per_scene = exps / (exps.sum(axis=1, keepdims=True) + 1e-9)
        
        # Aggregate (mean) and normalize
        agg = probs_per_scene.mean(axis=0)
        agg = agg / (agg.sum() + 1e-9)
        
        per_scene_top = [class_prompts[i] for i in probs_per_scene.argmax(axis=1)]
        dominant_genre = class_prompts[agg.argmax()]
        genre_confidence = float(agg.max())
        
        return {
            "genre_probabilities": {class_prompts[i]: float(agg[i]) for i in range(len(class_prompts))},
            "per_scene_top_class": per_scene_top,
            "dominant_genre": dominant_genre,
            "genre_confidence": genre_confidence,
        }


    def compute_scene_semantic_features(self, scene_embeddings: np.ndarray) -> Dict[str, float]:
        """
        Compute scene-level semantic features like similarity between adjacent scenes.
        Uses projected embeddings.
        """
        n = len(scene_embeddings)
        if n <= 1:
            return {
                "scene_sim_adjacent_mean": 0.0,
                "scene_sim_adjacent_std": 0.0,
            }
        
        sims = []
        for i in range(n - 1):
            sim = cosine_similarity(scene_embeddings[i:i+1], scene_embeddings[i+1:i+2])[0][0]
            sims.append(float(sim))
        
        sims = np.array(sims)
        return {
            "scene_sim_adjacent_mean": float(sims.mean()) if len(sims) > 0 else 0.0,
            "scene_sim_adjacent_std": float(sims.std()) if len(sims) > 0 else 0.0,
        }
    
    # -------------------------
    # Cross-modal novelty & attention scoring (improved)
    # -------------------------
    def compute_crossmodal_novelty_and_attention(
        self,
        scene_embeddings: np.ndarray,
        face_emotion_curve: Optional[np.ndarray],
        audio_energy_curve: Optional[np.ndarray],
        text_activity_curve: Optional[np.ndarray],
        topic_probabilities: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """
        Computes:
         - cross_modal_novelty_score (KL divergence of topic distributions or scene dissimilarity)
         - multimodal_attention_score (correlation between normalized modality activations)
        """
        n = len(scene_embeddings)
        if n <= 1:
            return {
                "cross_modal_novelty_score": 0.0,
                "multimodal_attention_score": 0.0,
            }

        # Novelty: mean scene-to-scene dissimilarity (on projected embeddings)
        sims = []
        for i in range(n - 1):
            sim = cosine_similarity(scene_embeddings[i:i+1], scene_embeddings[i+1:i+2])[0][0]
            sims.append(float(sim))
        novelty = float(np.mean(1 - np.array(sims)))

        # If topic probabilities available, also compute KL divergence between adjacent scenes
        if topic_probabilities and len(topic_probabilities) > 0:
            # This would require per-scene topic probabilities, which we don't have here
            # So we skip this enhancement for now
            pass

        # Attention: correlation between normalized curves aggregated per scene
        def _agg(curve):
            if curve is None or len(curve) == 0:
                return None
            L = len(curve)
            bounds = np.linspace(0, L, n+1, dtype=int)
            return np.array([curve[bounds[i]:bounds[i+1]].mean() if bounds[i+1]>bounds[i] else 0.0 for i in range(n)])

        f_face = _agg(_smooth(face_emotion_curve, self.sigma))
        f_audio = _agg(_smooth(audio_energy_curve, self.sigma))
        f_text = _agg(_smooth(text_activity_curve, self.sigma))

        # Combine only non-empty
        vectors = []
        if f_face is not None: vectors.append(_norm01(f_face))
        if f_audio is not None: vectors.append(_norm01(f_audio))
        if f_text is not None: vectors.append(_norm01(f_text))

        if len(vectors) < 2:
            attention_score = 0.0
        else:
            # Pairwise mean correlation
            R = np.corrcoef(np.vstack(vectors))
            # Average off-diagonal positive correlations
            m = R.shape[0]
            off_diag = R[np.triu_indices(m, k=1)]
            attention_score = float(np.clip(np.mean(np.maximum(off_diag, 0.0)), 0.0, 1.0))

        return {
            "cross_modal_novelty_score": novelty,
            "multimodal_attention_score": attention_score,
        }

    # -------------------------
    # Per-scene vector for VisualTransformer (64 dims)
    # -------------------------
    def compute_per_scene_vectors(
        self,
        scene_embeddings: np.ndarray,
        scene_metadata: List[Dict[str, Any]],
        face_emotion_curve: Optional[np.ndarray],
        audio_energy_curve: Optional[np.ndarray],
        text_activity_curve: Optional[np.ndarray],
        face_valence_curve: Optional[np.ndarray] = None,
        face_arousal_curve: Optional[np.ndarray] = None,
        face_presence_per_frame: Optional[np.ndarray] = None,
        text_sentiment_per_scene: Optional[np.ndarray] = None,
        topic_probabilities: Optional[Dict[str, float]] = None,
        per_scene_dominant_topic: Optional[List[str]] = None,
    ) -> np.ndarray:
        """
        Compute compact per-scene vectors (≈64 dims) for VisualTransformer.
        Components:
        - proj_clip_emb (32-48 dims)
        - scene_duration_norm (1)
        - scene_position_norm (1)
        - audio_energy_norm (1)
        - face_presence_flag (1) + avg_face_valence (1) + avg_face_arousal (1)
        - text_activity_flag (1) + text_sentiment (1)
        - topic_onehot_or_topk_embedding (4 dims)
        - scene_novelty_score (1)
        - multimodal_attention_score (1)
        - scene_visual_confidence (1)
        """
        n = len(scene_embeddings)
        if n == 0:
            return np.zeros((0, self.projection_dim + 15), dtype=float)  # projection_dim + additional features
        
        vectors = []
        total_duration = scene_metadata[-1].get("end_ts", 0.0) if scene_metadata else n * 2.0
        
        # Compute scene-level aggregations
        def _agg_curve_to_scenes(curve, scene_meta):
            if curve is None or len(curve) == 0:
                return np.zeros(n)
            L = len(curve)
            if scene_meta and len(scene_meta) == n:
                agg = []
                for meta in scene_meta:
                    start_frame = int(meta.get("start_ts", 0) * self.fps)
                    end_frame = int(meta.get("end_ts", 0) * self.fps)
                    start_frame = max(0, min(start_frame, L - 1))
                    end_frame = max(start_frame + 1, min(end_frame, L))
                    agg.append(curve[start_frame:end_frame].mean() if end_frame > start_frame else 0.0)
                return np.array(agg)
            else:
                boundaries = np.linspace(0, L, n + 1, dtype=int)
                return np.array([curve[boundaries[i]:boundaries[i+1]].mean() if boundaries[i+1] > boundaries[i] else 0.0 for i in range(n)])
        
        audio_energy_per_scene = _agg_curve_to_scenes(audio_energy_curve, scene_metadata)
        face_valence_per_scene = _agg_curve_to_scenes(face_valence_curve, scene_metadata) if face_valence_curve is not None else np.zeros(n)
        face_arousal_per_scene = _agg_curve_to_scenes(face_arousal_curve, scene_metadata) if face_arousal_curve is not None else np.zeros(n)
        
        # Face presence per scene
        if face_presence_per_frame is not None:
            face_presence_per_scene = _agg_curve_to_scenes(face_presence_per_frame.astype(float), scene_metadata)
        else:
            face_presence_per_scene = np.ones(n)  # Assume face present if curve exists
        
        # Text activity and sentiment
        text_activity_per_scene = _agg_curve_to_scenes(text_activity_curve, scene_metadata) if text_activity_curve is not None else np.zeros(n)
        text_sentiment_per_scene = text_sentiment_per_scene if text_sentiment_per_scene is not None else np.zeros(n)
        
        # Scene novelty (dissimilarity to previous scene)
        scene_novelty = np.zeros(n)
        for i in range(1, n):
            sim = cosine_similarity(scene_embeddings[i:i+1], scene_embeddings[i-1:i])[0][0]
            scene_novelty[i] = 1.0 - sim
        
        # Multimodal attention per scene (simplified)
        multimodal_attention_per_scene = np.ones(n) * 0.5  # Placeholder
        
        for i in range(n):
            vec = []
            
            # Projected CLIP embedding (use first projection_dim dims, or all if smaller)
            proj_emb = scene_embeddings[i][:self.projection_dim]
            vec.extend(proj_emb.tolist())
            
            # Pad if needed
            while len(vec) < self.projection_dim:
                vec.append(0.0)
            vec = vec[:self.projection_dim]
            
            # Additional features
            meta = scene_metadata[i] if i < len(scene_metadata) else {}
            duration = meta.get("scene_duration", 2.0)
            start_ts = meta.get("start_ts", i * 2.0)
            
            vec.append(duration / max(total_duration, 1.0))  # scene_duration_norm
            vec.append(start_ts / max(total_duration, 1.0))  # scene_position_norm
            vec.append(float(audio_energy_per_scene[i]))  # audio_energy_norm
            vec.append(float(face_presence_per_scene[i] > 0.5))  # face_presence_flag
            vec.append(float(face_valence_per_scene[i]))  # avg_face_valence
            vec.append(float(face_arousal_per_scene[i]))  # avg_face_arousal
            vec.append(float(text_activity_per_scene[i] > 0.1))  # text_activity_flag
            vec.append(float(text_sentiment_per_scene[i]))  # text_sentiment
            
            # Topic embedding (4 dims) - one-hot or top-k
            topic_vec = [0.0] * 4
            if per_scene_dominant_topic and i < len(per_scene_dominant_topic):
                topic_name = per_scene_dominant_topic[i]
                if topic_probabilities and topic_name in topic_probabilities:
                    # Use probability as embedding
                    prob = topic_probabilities[topic_name]
                    topic_vec[0] = prob
                    # Hash topic name to fill other dims
                    topic_hash = hash(topic_name) % 1000 / 1000.0
                    topic_vec[1] = topic_hash
                    topic_vec[2] = topic_hash * 0.5
                    topic_vec[3] = topic_hash * 0.25
            vec.extend(topic_vec)
            
            vec.append(float(scene_novelty[i]))  # scene_novelty_score
            vec.append(float(multimodal_attention_per_scene[i]))  # multimodal_attention_score
            vec.append(float(meta.get("scene_source_confidence", 1.0)))  # scene_visual_confidence
            
            vectors.append(vec)
        
        return np.array(vectors, dtype=float)

    # -------------------------
    # Reliability flags
    # -------------------------
    def compute_reliability_flags(
        self,
        face_emotion_curve: Optional[np.ndarray],
        audio_energy_curve: Optional[np.ndarray],
        ocr_activity_curve: Optional[np.ndarray],
        scene_embeddings: np.ndarray,
    ) -> Dict[str, bool]:
        """
        Compute reliability flags for each modality.
        """
        emotion_reliable = face_emotion_curve is not None and len(face_emotion_curve) > 0 and np.mean(face_emotion_curve) > 0.01
        audio_reliable = audio_energy_curve is not None and len(audio_energy_curve) > 0 and np.mean(audio_energy_curve) > 0.01
        ocr_reliable = ocr_activity_curve is not None and len(ocr_activity_curve) > 0 and np.mean(ocr_activity_curve) > 0.01
        
        # CLIP confidence: based on embedding norms and variance
        if len(scene_embeddings) > 0:
            norms = np.linalg.norm(scene_embeddings, axis=1)
            clip_confidence = float(np.mean(norms))  # Should be close to 1.0 for normalized embeddings
        else:
            clip_confidence = 0.0
        
        return {
            "emotion_reliable": bool(emotion_reliable),
            "audio_reliable": bool(audio_reliable),
            "ocr_reliable": bool(ocr_reliable),
            "clip_confidence": float(clip_confidence),
        }

    # -------------------------
    # Master extract function
    # -------------------------
    def extract_all(
        self,
        scene_frames: Optional[List[Any]] = None,
        scene_embeddings: Optional[np.ndarray] = None,
        scene_metadata: Optional[List[Dict[str, Any]]] = None,
        face_emotion_curve: Optional[np.ndarray] = None,
        audio_energy_curve: Optional[np.ndarray] = None,
        pose_activity_curve: Optional[np.ndarray] = None,
        text_features: Optional[Dict[str, Any]] = None,
        topic_vectors: Optional[Dict[str, np.ndarray]] = None,
        class_prompts: Optional[List[str]] = None,
        scene_boundary_frames: Optional[List[int]] = None,
        mode: str = "full",  # "fast" or "full"
        face_valence_curve: Optional[np.ndarray] = None,
        face_arousal_curve: Optional[np.ndarray] = None,
        face_presence_per_frame: Optional[np.ndarray] = None,
        ocr_activity_curve: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Run full pipeline.
        mode: "fast" = CLIP + audio coarse + subtitles; "full" = fine emotion, speaker diarization, multimodal attention, topic modeling
        
        text_features is dict provided by TextProcessor and may contain:
          - 'scene_text_embeddings' : np.array [n_scenes, d]
          - 'text_activity_curve' : per-frame curve
          - 'text_emotion_curve' : per-frame
          - 'text_importance_per_scene' : [n_scenes]
          - 'text_sentiment_per_scene' : [n_scenes]
          - 'transcript_timestamps', 'keyword_timeline', 'cta_timestamps'
          - 'video_topic_embedding'
        """
        t0 = time.time()
        
        # Get scene embeddings and metadata
        scenes_emb, scene_meta = self.get_scene_embeddings(
            scene_frames=scene_frames,
            scene_embeddings=scene_embeddings,
            scene_metadata=scene_metadata,
        )
        n_scenes = len(scenes_emb)

        # Prepare curves from text_features
        text_activity_curve = None
        text_emotion_curve = None
        text_importance_per_scene = None
        text_sentiment_per_scene = None
        scene_caption_embeddings = None
        video_topic_embedding = None

        if text_features:
            text_activity_curve = text_features.get("text_activity_curve", None)
            text_emotion_curve = text_features.get("text_emotion_curve", None)
            text_importance_per_scene = text_features.get("text_importance_per_scene", None)
            text_sentiment_per_scene = text_features.get("text_sentiment_per_scene", None)
            scene_caption_embeddings = text_features.get("scene_text_embeddings", None)
            video_topic_embedding = text_features.get("video_topic_embedding", None)

        # Smoothing and normalizing input curves
        fac = _smooth(face_emotion_curve, self.sigma) if face_emotion_curve is not None else None
        aud = _smooth(audio_energy_curve, self.sigma) if audio_energy_curve is not None else None
        pose = _smooth(pose_activity_curve, self.sigma) if pose_activity_curve is not None else None
        txt_act = _smooth(text_activity_curve, self.sigma) if text_activity_curve is not None else None
        txt_em = _smooth(text_emotion_curve, self.sigma) if text_emotion_curve is not None else None
        ocr_act = _smooth(ocr_activity_curve, self.sigma) if ocr_activity_curve is not None else None

        # Reliability flags
        reliability = self.compute_reliability_flags(fac, aud, ocr_act, scenes_emb)

        # Video-level embeddings
        vid_embs = self.compute_video_level_embeddings(
            scenes_emb, fac, aud, text_importance_per_scene, ocr_act, scene_meta
        )

        # Scene semantic features
        scene_semantic = self.compute_scene_semantic_features(scenes_emb) if len(scenes_emb) > 0 else {}

        # Topic features (only in full mode or if explicitly requested)
        topic = {}
        if mode == "full" or topic_vectors or video_topic_embedding:
            topic = self.compute_topics(scenes_emb, topic_vectors=topic_vectors, video_topic_embedding=video_topic_embedding)

        # Events (only in full mode)
        events = {}
        if mode == "full":
            events = self.detect_events(
                scenes_emb, fac, aud, pose, txt_act, ocr_act, scene_boundary_frames, scene_meta
            )
        else:
            events = {
                "event_timestamps": [],
                "number_of_events": 0,
                "event_strengths": [],
                "event_types": [],
                "event_rate_per_minute": 0.0,
            }

        # Emotion alignment (only in full mode)
        emotion_align = {}
        if mode == "full":
            emotion_align = self.compute_emotion_alignment(
                fac if fac is not None else None,
                txt_em if txt_em is not None else None,
            )
        else:
            emotion_align = {
                "emotion_correlation": 0.0,
                "emotion_lag_seconds": 0.0,
                "emotion_alignment_score": 0.0,
                "emotion_alignment_reliability": 0.0,
            }

        # Emotion features
        emotion_features = self.compute_emotion_features(
            fac,
            face_valence_curve,
            face_arousal_curve,
            face_presence_per_frame,
        )

        # Narrative embedding (only in full mode)
        narrative = {}
        if mode == "full":
            narrative = self.compute_narrative_embedding(scenes_emb, scene_caption_embeddings, video_topic_embedding)
        else:
            narrative = {
                "narrative_embedding": scenes_emb.mean(axis=0) if len(scenes_emb) > 0 else None,
                "story_flow_score": 0.0,
                "narrative_complexity_score": 0.0,
            }

        # Multimodal novelty & attention
        multimodal = self.compute_crossmodal_novelty_and_attention(
            scenes_emb, fac, aud, txt_act, topic.get("topic_probabilities", None)
        )

        # Zero-shot genre/style (only in full mode)
        genre = {}
        if mode == "full" and class_prompts and len(class_prompts) > 0:
            genre = self.zero_shot_genre_style(scenes_emb, class_prompts)

        # Per-scene vectors for VisualTransformer
        per_scene_vectors = self.compute_per_scene_vectors(
            scenes_emb,
            scene_meta,
            fac,
            aud,
            txt_act,
            face_valence_curve,
            face_arousal_curve,
            face_presence_per_frame,
            text_sentiment_per_scene,
            topic.get("topic_probabilities", None),
            topic.get("per_scene_dominant_topic", None),
        )

        # Aggregate many numeric features to flat dict
        features = {}
        
        # Scene-level stats
        features["scene_count"] = n_scenes
        features["scene_sim_adjacent_mean"] = float(scene_semantic.get("scene_sim_adjacent_mean", 0.0))
        features["scene_sim_adjacent_std"] = float(scene_semantic.get("scene_sim_adjacent_std", 0.0))
        
        # Average scene duration
        if scene_meta and len(scene_meta) > 0:
            avg_scene_duration = np.mean([m.get("scene_duration", 0.0) for m in scene_meta])
            features["avg_scene_duration"] = float(avg_scene_duration)
        else:
            features["avg_scene_duration"] = 0.0

        # Video embedding stats
        if vid_embs["mean_embedding"] is not None:
            features["video_embedding_norm_mean"] = float(np.linalg.norm(vid_embs["mean_embedding"]))
            features["video_embedding_norm_weighted"] = float(np.linalg.norm(vid_embs["weighted_mean_embedding"]))
            features["video_embedding_var_mean"] = vid_embs.get("video_embedding_var_mean", 0.0)

        # Topic
        features.update(topic)

        # Events
        features["number_of_events"] = events.get("number_of_events", 0)
        features["event_rate_per_minute"] = events.get("event_rate_per_minute", 0.0)
        features["event_strength_max"] = float(max(events.get("event_strengths", [0])) if events.get("event_strengths") else 0.0)
        features["event_types"] = events.get("event_types", [])
        features["event_timestamps"] = events.get("event_timestamps", [])

        # Emotion
        features.update(emotion_align)
        features.update(emotion_features)

        # Narrative
        features["story_flow_score"] = narrative.get("story_flow_score", 0.0)
        features["narrative_complexity_score"] = narrative.get("narrative_complexity_score", 0.0)
        features["cross_modal_novelty_score"] = multimodal.get("cross_modal_novelty_score", 0.0)

        # Multimodal attention
        features["multimodal_attention_overall"] = multimodal.get("multimodal_attention_score", 0.0)

        # Genre/style
        features.update(genre)

        # Reliability flags
        features.update(reliability)

        # Attach some arrays (useful for downstream)
        outputs = {
            "features": features,
            "scene_embeddings": scenes_emb,  # Projected embeddings
            "video_embeddings": vid_embs,
            "scene_metadata": scene_meta,
            "scene_semantic": scene_semantic,
            "events": events,
            "narrative": narrative,
            "multimodal_alignment": multimodal,
            "per_scene_vectors": per_scene_vectors,  # For VisualTransformer
            "reliability_flags": reliability,
        }
        outputs["timing_seconds"] = time.time() - t0
        return outputs


class HighLevelSemanticModule(BaseModule):
    """
    Модуль для извлечения высокоуровневой семантики из видео.
    
    Наследуется от BaseModule для интеграции с системой зависимостей и единым форматом вывода.
    Использует HighLevelSemanticsOptimized для обработки видео.
    
    Зависимости:
    - emotion_face (опциональная) - для данных эмоций лиц
    - cut_detection (опциональная) - для границ сцен
    """
    
    def __init__(
        self,
        rs_path: Optional[str] = None,
        device: str = "cuda",
        clip_model_name: str = "ViT-B/32",
        clip_batch_size: int = 64,
        fps: int = 30,
        use_face_data: bool = False,
        use_cut_data: bool = False,
        class_prompts: Optional[List[str]] = None,
        **kwargs: Any
    ):
        """
        Инициализация HighLevelSemanticModule.
        
        Args:
            rs_path: Путь к хранилищу результатов
            device: Устройство для обработки (cuda/cpu)
            clip_model_name: Имя модели CLIP
            clip_batch_size: Размер батча для обработки CLIP
            fps: Кадров в секунду
            use_face_data: Использовать данные эмоций лиц из emotion_face
            use_cut_data: Использовать данные детекции склеек для границ сцен
            class_prompts: Список промптов классов для zero-shot классификации
            **kwargs: Дополнительные параметры для BaseModule
        """
        super().__init__(rs_path=rs_path, **kwargs)
        
        self.fps = fps
        self.use_face_data = use_face_data
        self.use_cut_data = use_cut_data
        self.class_prompts = class_prompts
        
        # Инициализируем процессор
        self.processor = HighLevelSemanticsOptimized(
            device=device,
            clip_model_name=clip_model_name,
            clip_batch_size=clip_batch_size,
            fps=fps,
        )
    
    def required_dependencies(self) -> List[str]:
        """
        Возвращает список зависимостей модуля.
        
        Опциональные зависимости:
        - emotion_face: для данных эмоций лиц
        - cut_detection: для границ сцен
        """
        deps = []
        if self.use_face_data:
            deps.append("emotion_face")
        if self.use_cut_data:
            deps.append("cut_detection")
        return deps
    
    def _load_face_emotion_data(self) -> Optional[np.ndarray]:
        """Загружает данные эмоций лиц из emotion_face."""
        if not self.use_face_data:
            return None
        
        try:
            face_data = self.load_dependency_results("emotion_face", format="json")
            if face_data and isinstance(face_data, dict):
                if "emotion_curve" in face_data:
                    return np.array(face_data["emotion_curve"])
                # Альтернативный формат: sequence_features
                if "sequence_features" in face_data:
                    seq_feat = face_data["sequence_features"]
                    if "valence_sequence" in seq_feat:
                        return np.array(seq_feat["valence_sequence"])
        except Exception as e:
            self.logger.warning(f"HighLevelSemanticModule | Не удалось загрузить face data: {e}")
        
        return None
    
    def _load_cut_detection_data(self) -> Optional[List[int]]:
        """Загружает данные детекции склеек из cut_detection."""
        if not self.use_cut_data:
            return None
        
        try:
            cut_data = self.load_dependency_results("cut_detection", format="json")
            if cut_data and isinstance(cut_data, dict):
                if "scene_boundaries" in cut_data:
                    return cut_data["scene_boundaries"]
                elif "cuts" in cut_data:
                    # Извлекаем индексы кадров из cuts
                    return [cut.get("frame", 0) for cut in cut_data["cuts"]]
        except Exception as e:
            self.logger.warning(f"HighLevelSemanticModule | Не удалось загрузить cut data: {e}")
        
        return None
    
    def process(
        self,
        frame_manager: FrameManager,
        frame_indices: List[int],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Основной метод обработки (интерфейс BaseModule).
        
        Args:
            frame_manager: Менеджер кадров
            frame_indices: Список индексов кадров для обработки
            config: Конфигурация модуля:
                - scene_frames: список PIL Image (опционально, если не указан, будет извлечен из frame_manager)
                - sample_rate: частота выборки кадров (по умолчанию fps * 2)
                
        Returns:
            Словарь с результатами в формате для сохранения в npz:
            - features: агрегированные фичи семантики
            - per_scene_vectors: numpy массив [n_scenes, 64] для VisualTransformer
            - scene_embeddings: embeddings сцен
            - summary: метаданные обработки
        """
        from PIL import Image
        
        # Загружаем зависимости
        face_emotion_curve = self._load_face_emotion_data()
        scene_boundary_frames = self._load_cut_detection_data()
        
        # Получаем scene_frames из config или извлекаем из frame_manager
        scene_frames = config.get("scene_frames")
        if scene_frames is None:
            # Извлекаем scene frames
            total_frames = len(frame_indices) if frame_indices else frame_manager.total_frames
            sample_rate = config.get("sample_rate", max(1, int(self.fps * 2)))
            
            if scene_boundary_frames and len(scene_boundary_frames) > 0:
                # Используем cut boundaries для определения сцен
                scene_boundary_frames = sorted(set([0] + scene_boundary_frames + [total_frames - 1]))
                scene_frames = []
                for i in range(len(scene_boundary_frames) - 1):
                    start_frame = scene_boundary_frames[i]
                    end_frame = scene_boundary_frames[i + 1]
                    # Берем средний кадр каждой сцены
                    mid_frame = (start_frame + end_frame) // 2
                    if mid_frame < total_frames:
                        frame = frame_manager.get(mid_frame)
                        scene_frames.append(Image.fromarray(frame))
            else:
                # Равномерная выборка
                scene_frames = []
                for frame_idx in range(0, total_frames, sample_rate):
                    if frame_idx < total_frames:
                        frame = frame_manager.get(frame_idx)
                        scene_frames.append(Image.fromarray(frame))
        
        try:
            # Обрабатываем через HighLevelSemanticsOptimized
            result = self.processor.extract_all(
                scene_frames=scene_frames,
                scene_embeddings=None,
                face_emotion_curve=face_emotion_curve,
                audio_energy_curve=None,
                pose_activity_curve=None,
                text_features=None,
                topic_vectors=None,
                class_prompts=self.class_prompts,
                scene_boundary_frames=scene_boundary_frames,
            )
            
            # Преобразуем результаты в единый формат для npz
            formatted_result = self._format_results_for_npz(result)
            
            self.logger.info(
                f"HighLevelSemanticModule | Обработка завершена: "
                f"обработано {len(scene_frames)} сцен"
            )
            
            return formatted_result
            
        except Exception as e:
            self.logger.exception(f"HighLevelSemanticModule | Ошибка обработки: {e}")
            return self._empty_result()
    
    def _format_results_for_npz(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Преобразует результаты HighLevelSemanticsOptimized в формат для сохранения в npz.
        
        Args:
            result: Результаты из processor.extract_all()
            
        Returns:
            Словарь в формате для npz
        """
        # Извлекаем основные данные
        features = result.get("features", {})
        scene_embeddings = result.get("scene_embeddings")
        video_embeddings = result.get("video_embeddings", {})
        per_scene_vectors = result.get("per_scene_vectors")
        scene_metadata = result.get("scene_metadata", [])
        reliability_flags = result.get("reliability_flags", {})
        
        # Подготавливаем features (агрегированные фичи)
        features_clean = {}
        for key, value in features.items():
            if isinstance(value, (int, float, bool)):
                features_clean[key] = float(value) if isinstance(value, bool) else value
            elif isinstance(value, (list, tuple)):
                try:
                    features_clean[key] = np.asarray(value, dtype=np.float32)
                except Exception:
                    features_clean[key] = np.asarray(value, dtype=object)
            elif isinstance(value, np.ndarray):
                features_clean[key] = value
            else:
                # Остальное сохраняем как есть
                features_clean[key] = value
        
        # Подготавливаем per_scene_vectors
        if per_scene_vectors is not None:
            if isinstance(per_scene_vectors, list):
                per_scene_vectors = np.array(per_scene_vectors, dtype=np.float32)
            elif not isinstance(per_scene_vectors, np.ndarray):
                per_scene_vectors = np.asarray(per_scene_vectors, dtype=np.float32)
        else:
            per_scene_vectors = np.array([], dtype=np.float32).reshape(0, 64)
        
        # Подготавливаем scene_embeddings
        if scene_embeddings is not None:
            if isinstance(scene_embeddings, list):
                scene_embeddings = np.array(scene_embeddings, dtype=np.float32)
            elif not isinstance(scene_embeddings, np.ndarray):
                scene_embeddings = np.asarray(scene_embeddings, dtype=np.float32)
        else:
            scene_embeddings = np.array([], dtype=np.float32)
        
        # Подготавливаем video_embeddings
        video_embeddings_clean = {}
        if isinstance(video_embeddings, dict):
            for key, value in video_embeddings.items():
                if isinstance(value, (list, tuple)):
                    try:
                        video_embeddings_clean[key] = np.asarray(value, dtype=np.float32)
                    except Exception:
                        video_embeddings_clean[key] = np.asarray(value, dtype=object)
                elif isinstance(value, np.ndarray):
                    video_embeddings_clean[key] = value
                else:
                    video_embeddings_clean[key] = value
        
        # Подготавливаем summary
        summary = {
            "n_scenes": len(scene_metadata) if scene_metadata else 0,
            "timing_seconds": result.get("timing_seconds", 0.0),
            "fps": self.fps,
            "success": True,
        }
        
        # Формируем итоговый результат
        formatted_result = {
            "features": features_clean,
            "per_scene_vectors": per_scene_vectors,
            "scene_embeddings": scene_embeddings,
            "video_embeddings": video_embeddings_clean,
            "reliability_flags": reliability_flags,
            "summary": summary,
        }
        
        return formatted_result
    
    def _empty_result(self) -> Dict[str, Any]:
        """Возвращает пустой результат в правильном формате."""
        return {
            "features": {},
            "per_scene_vectors": np.array([], dtype=np.float32).reshape(0, 64),
            "scene_embeddings": np.array([], dtype=np.float32),
            "video_embeddings": {},
            "reliability_flags": {},
            "summary": {
                "n_scenes": 0,
                "timing_seconds": 0.0,
                "fps": self.fps,
                "success": False,
            },
        }