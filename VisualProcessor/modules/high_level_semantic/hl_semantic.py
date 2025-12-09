# high_level_semantics_optimized.py
"""
High-Level Semantics (Optimized, multimodal, Variant B)
- Максимально мультимодальный VideoProcessor: связывает видео + текстовые входы (предоставляемые извне)
- Использует CLIP (batch) для scene/shot embeddings (можно подать precomputed embeddings)
- Выдаёт: scene embeddings, video embeddings (mean/weighted/max/var),
  topic probs (zero-shot / via provided topic embeddings),
  events (multimodal peaks), emotion alignment, narrative embedding, novelty, genre/style probs и др.

Inputs (examples):
- scene_frames: list[list[np.uint8]] or list of representative PIL/np frames per scene (each scene may be 1 frame or an average frame)
- OR scene_embeddings: np.array [n_scenes, D] (if precomputed)
- face_emotion_curve: np.array per-frame (0..1)
- audio_energy_curve: np.array per-frame (0..1)
- pose_activity_curve: np.array per-frame (0..1) (optional)
- text_features: dict from TextProcessor (keys described below)

TextFeatures expected (examples — module doesn't compute them):
{
  "transcript_timestamps": [(start_s, end_s, text, embedding_vec), ...],
  "scene_text_embeddings": np.array [n_scenes, d_text] (optional),
  "topic_vectors": {"sports": vec, "cooking": vec, ...} (optional),
  "text_emotion_curve": np.array per-frame (0..1) (optional),
  "keyword_timeline": [{"frame": idx, "keyword": "subscribe"}, ...] (optional),
  "cta_timestamps": [t1_s, t2_s],
  "video_topic_embedding": np.array (d,)
}
Outputs:
- features dict with many numeric features and useful arrays (see usage)
"""

from typing import List, Dict, Optional, Any, Tuple
import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d
from sklearn.metrics.pairwise import cosine_similarity
import math
import time
from collections import defaultdict

# Optional: CLIP (OpenAI or open_clip). Try openai/clip first.
try:
    import clip  # openai/clip
    _CLIP_BACKEND = "openai"
except Exception:
    try:
        import open_clip  # open_clip (if installed)
        _CLIP_BACKEND = "open_clip"
    except Exception:
        _CLIP_BACKEND = None

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

def _chunk(iterable, size):
    for i in range(0, len(iterable), size):
        yield iterable[i:i+size]

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
    ):
        """
        device: "cuda" or "cpu". If None, auto-detect.
        clip_model_name: model name for clip.load (openai/clip) or open_clip
        clip_batch_size: batch size for embedding frames
        fps: frames-per-second of video (used for timestamps)
        smoothing_sigma: smoothing for curves
        """
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.clip_batch_size = clip_batch_size
        self.fps = fps
        self.sigma = smoothing_sigma

        if _CLIP_BACKEND is None:
            raise ImportError("CLIP backend not found. Install 'clip' or 'open_clip' package.")

        # Load CLIP once (image encoder)
        self.clip_model_name = clip_model_name
        self._load_clip_model(clip_model_name)

    def _load_clip_model(self, model_name):
        if _CLIP_BACKEND == "openai":
            self.clip_model, self.clip_preprocess = clip.load(model_name, device=self.device)
            self.clip_encode = self._clip_encode_openai
            # note: openai clip returns 512- or 1024-d embeddings depending on model
        else:
            # open_clip
            import open_clip
            self.clip_model, _, self.clip_preprocess = open_clip.create_model_and_transforms(model_name, pretrained='laion2b_s13b_b90k')
            self.clip_model.to(self.device).eval()
            self.clip_encode = self._clip_encode_open_clip

    def _clip_encode_openai(self, pil_images: List[Any]) -> np.ndarray:
        """
        pil_images: list of PIL.Image or preprocessed tensors (we'll use preprocess)
        returns: np.array [N, D]
        """
        self.clip_model.eval()
        embs = []
        with torch.no_grad():
            for batch in _chunk(pil_images, self.clip_batch_size):
                proc = torch.stack([self.clip_preprocess(img) for img in batch]).to(self.device)
                feats = self.clip_model.encode_image(proc)
                feats = feats / feats.norm(dim=-1, keepdim=True)
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
                feats = feats / feats.norm(dim=-1, keepdim=True)
                embs.append(feats.cpu().numpy())
        return np.vstack(embs)

    # -------------------------
    # Helper: get scene embeddings either from frames or accept precomputed
    # -------------------------
    def get_scene_embeddings(
        self,
        scene_frames: Optional[List[Any]] = None,
        scene_embeddings: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        scene_frames: list of PIL.Image / np.array frames (one per scene, representative)
        scene_embeddings: optional precomputed np.array (n_scenes, D)
        Returns normalized embeddings (n_scenes, D)
        """
        if scene_embeddings is not None:
            embs = np.array(scene_embeddings)
            # normalize
            norms = np.linalg.norm(embs, axis=1, keepdims=True) + 1e-9
            return embs / norms

        if scene_frames is None or len(scene_frames) == 0:
            return np.zeros((0, 512), dtype=float)

        # Accept both PIL.Image and np.array (H,W,3 uint8)
        pil_images = []
        from PIL import Image
        for f in scene_frames:
            if isinstance(f, np.ndarray):
                pil_images.append(Image.fromarray(f))
            else:
                pil_images.append(f)

        embs = self.clip_encode(pil_images)  # already normalized
        return embs

    # -------------------------
    # Video-level embeddings (mean, weighted, max, var)
    # -------------------------
    def compute_video_level_embeddings(
        self,
        scene_embeddings: np.ndarray,
        face_emotion_curve: Optional[np.ndarray],
        audio_energy_curve: Optional[np.ndarray],
        text_importance_per_scene: Optional[np.ndarray] = None,
    ) -> Dict[str, np.ndarray]:
        """
        text_importance_per_scene: optional array [n_scenes] with importance weights from TextProcessor
        Returns dict with mean, weighted_mean, max, var embeddings and attention weights
        """
        n = len(scene_embeddings)
        if n == 0:
            return {
                "mean_embedding": None,
                "weighted_mean_embedding": None,
                "max_embedding": None,
                "var_embedding": None,
                "attention_weights": None,
            }

        mean_emb = scene_embeddings.mean(axis=0)
        max_emb = scene_embeddings.max(axis=0)
        var_emb = scene_embeddings.var(axis=0)

        # compute attention weights per scene using face_emotion, audio, pose and text importance
        # Map curves (frame-level) to scene-level by averaging across approximate scene durations.
        # If curves are empty -> fallback equal weights
        def _agg_curve_to_scenes(curve):
            if curve is None or len(curve) == 0:
                return np.ones(n)
            L = len(curve)
            # split frames into n buckets
            boundaries = np.linspace(0, L, n + 1, dtype=int)
            agg = np.array([curve[boundaries[i]:boundaries[i+1]].mean() if boundaries[i+1] > boundaries[i] else 0.0 for i in range(n)])
            return agg

        face_w = _agg_curve_to_scenes(_smooth(face_emotion_curve, self.sigma))
        audio_w = _agg_curve_to_scenes(_smooth(audio_energy_curve, self.sigma))
        text_w = text_importance_per_scene if text_importance_per_scene is not None else np.ones(n)

        # combine with weights, normalize
        raw = (0.45 * _norm01(face_w) + 0.35 * _norm01(audio_w) + 0.2 * _norm01(text_w))
        att = raw / (raw.sum() + 1e-9)
        weighted_mean = (scene_embeddings * att[:, None]).sum(axis=0)

        return {
            "mean_embedding": mean_emb,
            "weighted_mean_embedding": weighted_mean,
            "max_embedding": max_emb,
            "var_embedding": var_emb,
            "attention_weights": att,
        }

    # -------------------------
    # Topic / Concept detection (use provided topic vectors or zero-shot)
    # -------------------------
    def compute_topics(
        self,
        scene_embeddings: np.ndarray,
        topic_vectors: Optional[Dict[str, np.ndarray]] = None,
        video_topic_embedding: Optional[np.ndarray] = None,
        k_top: int = 3,
    ) -> Dict[str, Any]:
        """
        If topic_vectors provided (name->vec), compute per-scene similarity and aggregate.
        If not provided, but video_topic_embedding provided, compute similarity to it.
        Returns:
          topic_probabilities: aggregated probabilities per topic name
          per_scene_dominant_topic: list indices
          topic_diversity_score
          topic_transition_rate
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
            sims = cosine_similarity(scene_embeddings, vecs)  # n x m
            # softmax per scene
            exps = np.exp(sims - sims.max(axis=1, keepdims=True))
            probs = exps / (exps.sum(axis=1, keepdims=True) + 1e-9)
            agg = probs.mean(axis=0)
            agg = agg / (agg.sum() + 1e-9)
            dominant = sims.argmax(axis=1).tolist()
            # diversity: entropy-like
            diversity = float(-(agg * np.log(agg + 1e-9)).sum())
            # transition rate
            transitions = int(np.sum(np.array(dominant[1:]) != np.array(dominant[:-1])))
            transition_rate = transitions / max(1, n - 1)
            return {
                "topic_probabilities": {names[i]: float(agg[i]) for i in range(len(names))},
                "per_scene_dominant_topic": dominant,
                "topic_diversity_score": diversity,
                "topic_transition_rate": float(transition_rate),
            }

        if video_topic_embedding is not None:
            sims = cosine_similarity(scene_embeddings, video_topic_embedding.reshape(1, -1)).flatten()
            prob = float(sims.mean())
            return {
                "topic_probabilities": {"video_topic_similarity": prob},
                "per_scene_dominant_topic": (sims.argmax(axis=0),),
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
    # Events / key moments (multimodal)
    # -------------------------
    def detect_events(
        self,
        scene_embeddings: np.ndarray,
        face_emotion_curve: Optional[np.ndarray],
        audio_energy_curve: Optional[np.ndarray],
        pose_activity_curve: Optional[np.ndarray] = None,
        text_activity_curve: Optional[np.ndarray] = None,
        scene_boundary_frames: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Compute events as peaks in combined multimodal energy:
         - scene_jump (embedding derivative)
         - face_emotion peaks
         - audio peaks
         - pose peaks
         - text activity peaks (text show/keyword)
        scene_boundary_frames: optional list of frame indices dividing scenes (to map scene-level to frames)
        Returns detected event timestamps (seconds), number_of_events, event_strengths, event_types (simple labels)
        """
        # Build frame-level signals aligned to a common length L (use max length from provided curves)
        curves = [face_emotion_curve, audio_energy_curve, pose_activity_curve, text_activity_curve]
        lengths = [len(c) for c in curves if c is not None and len(c) > 0]
        L = max(lengths) if lengths else 0

        if L == 0:
            # fallback: event-level from scene jumps only
            scene_jumps = self._scene_jump_signal(scene_embeddings)
            idxs = np.where(scene_jumps > (scene_jumps.mean() + scene_jumps.std()))[0]
            times = (idxs / max(1, self.fps)).tolist()
            return {
                "event_timestamps": times,
                "number_of_events": int(len(idxs)),
                "event_strengths": scene_jumps[idxs].tolist(),
                "event_types": ["scene_jump"] * len(idxs),
            }

        # build normalized frame-level signals
        f_face = _norm01(_smooth(face_emotion_curve, self.sigma)) if face_emotion_curve is not None else np.zeros(L)
        f_audio = _norm01(_smooth(audio_energy_curve, self.sigma)) if audio_energy_curve is not None else np.zeros(L)
        f_pose = _norm01(_smooth(pose_activity_curve, self.sigma)) if pose_activity_curve is not None else np.zeros(L)
        f_text = _norm01(_smooth(text_activity_curve, self.sigma)) if text_activity_curve is not None else np.zeros(L)

        # scene jump expanded to frame-level
        scene_jump = self._scene_jump_signal(scene_embeddings, L=L)

        # combined energy (weights tuned: face=0.25, audio=0.25, scene_jump=0.3, text=0.15, pose=0.05)
        combined = 0.25 * f_face + 0.25 * f_audio + 0.3 * scene_jump + 0.15 * f_text + 0.05 * f_pose
        combined = _norm01(_smooth(combined, sigma=self.sigma))

        # detect peaks (local maxima above threshold)
        # simple approach: threshold = mean + 0.7*std
        thresh = combined.mean() + 0.7 * combined.std()
        peaks = np.where((combined >= thresh) & (combined >= np.roll(combined, 1)) & (combined >= np.roll(combined, -1)))[0]
        times = (peaks / float(self.fps)).tolist()
        strengths = combined[peaks].tolist()

        # classify event types by dominant channel at peak
        event_types = []
        for p in peaks:
            vals = {"face": f_face[p], "audio": f_audio[p], "scene_jump": scene_jump[p], "text": f_text[p], "pose": f_pose[p]}
            dominant = max(vals, key=vals.get)
            event_types.append(dominant)

        return {
            "event_timestamps": times,
            "number_of_events": int(len(peaks)),
            "event_strengths": strengths,
            "event_types": event_types,
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
        # expand to length L by repeating each jump proportionally
        reps = math.ceil(L / len(jumps))
        arr = np.repeat(jumps, reps)[:L]
        return _norm01(arr)

    # -------------------------
    # Emotion alignment & sentiment interplay
    # -------------------------
    def compute_emotion_alignment(
        self,
        face_emotion_curve: np.ndarray,
        text_emotion_curve: Optional[np.ndarray] = None,
    ) -> Dict[str, float]:
        """
        Compare face emotion timeline with text emotion timeline (text emotion from TextProcessor).
        Returns correlation, lag (where they align best), and alignment score.
        """
        if face_emotion_curve is None or len(face_emotion_curve) == 0:
            return {"emotion_correlation": 0.0, "emotion_lag_seconds": 0.0, "emotion_alignment_score": 0.0}

        f = _norm01(_smooth(face_emotion_curve, self.sigma))
        if text_emotion_curve is None or len(text_emotion_curve) == 0:
            # correlation of face with itself -> 1
            return {"emotion_correlation": float(np.corrcoef(f, f)[0, 1]), "emotion_lag_seconds": 0.0, "emotion_alignment_score": float(f.mean())}

        t = _norm01(_smooth(text_emotion_curve, self.sigma))
        # resample shorter one to match longer
        L = max(len(f), len(t))
        f_r = np.interp(np.linspace(0, 1, L), np.linspace(0, 1, len(f)), f)
        t_r = np.interp(np.linspace(0, 1, L), np.linspace(0, 1, len(t)), t)

        # cross-correlation to find lag
        corr = np.correlate(f_r - f_r.mean(), t_r - t_r.mean(), mode='full')
        lag = corr.argmax() - (L - 1)
        lag_seconds = lag / float(self.fps)
        # normalized correlation at zero lag
        corr_norm = corr.max() / (np.sqrt(np.sum((f_r - f_r.mean())**2) * np.sum((t_r - t_r.mean())**2)) + 1e-9)
        alignment_score = float(corr_norm)
        return {"emotion_correlation": float(corr_norm), "emotion_lag_seconds": float(lag_seconds), "emotion_alignment_score": alignment_score}

    # -------------------------
    # Narrative embedding & story coherence
    # -------------------------
    def compute_narrative_embedding(
        self,
        scene_embeddings: np.ndarray,
        scene_caption_embeddings: Optional[np.ndarray] = None,
        summary_embedding: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """
        Build a narrative embedding by combining visual scene embeddings and textual scene captions embeddings (if provided).
        Also compute story_flow_score (smoothness of embeddings) and narrative_complexity.
        """
        if len(scene_embeddings) == 0:
            return {"narrative_embedding": None, "story_flow_score": 0.0, "narrative_complexity_score": 0.0}

        visual_mean = scene_embeddings.mean(axis=0)
        if scene_caption_embeddings is not None and len(scene_caption_embeddings) == len(scene_embeddings):
            text_mean = scene_caption_embeddings.mean(axis=0)
            narrative_emb = (visual_mean + text_mean) / 2.0
        elif summary_embedding is not None:
            narrative_emb = (visual_mean + summary_embedding) / 2.0
        else:
            narrative_emb = visual_mean

        # story flow score: mean cosine similarity between consecutive scenes (higher = more coherent / less jumpy)
        sims = []
        for i in range(len(scene_embeddings) - 1):
            sims.append(cosine_similarity(scene_embeddings[i:i+1], scene_embeddings[i+1:i+2])[0][0])
        sims = np.array(sims)
        flow = float(sims.mean()) if sims.size else 0.0
        complexity = float(sims.std()) if sims.size else 0.0

        return {"narrative_embedding": narrative_emb, "story_flow_score": flow, "narrative_complexity_score": complexity}

    # -------------------------
    # Genre/style zero-shot via CLIP text prompts
    # -------------------------
    def zero_shot_genre_style(
        self,
        scene_embeddings: np.ndarray,
        class_prompts: List[str],
    ) -> Dict[str, Any]:
        """
        Use CLIP zero-shot: encode class_prompts as text embeddings and compute similarity to scene embeddings.
        Returns aggregated probabilities for each class and per-scene top label.
        """
        if _CLIP_BACKEND is None:
            return {"genre_probabilities": {}, "per_scene_top_class": []}

        # encode text prompts
        import torch
        model = self.clip_model
        tokenizer = clip.tokenize if _CLIP_BACKEND == "openai" else None
        device = self.device

        with torch.no_grad():
            if _CLIP_BACKEND == "openai":
                tokens = clip.tokenize(class_prompts).to(device)
                text_feats = model.encode_text(tokens)
            else:
                # open_clip tokenization / encoding
                import open_clip
                tokens = open_clip.tokenize(class_prompts).to(device)
                text_feats = model.encode_text(tokens)
            text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)
            text_np = text_feats.cpu().numpy()  # m x D

        if len(scene_embeddings) == 0:
            return {"genre_probabilities": {p: 0.0 for p in class_prompts}, "per_scene_top_class": []}

        sims = cosine_similarity(scene_embeddings, text_np)  # n_scenes x m
        # aggregate (mean) and normalize
        agg = sims.mean(axis=0)
        agg = np.maximum(agg, 0.0)
        if agg.sum() > 0:
            probs = agg / agg.sum()
        else:
            probs = np.ones_like(agg) / len(agg)

        per_scene_top = sims.argmax(axis=1).tolist()
        return {"genre_probabilities": {class_prompts[i]: float(probs[i]) for i in range(len(class_prompts))},
                "per_scene_top_class": per_scene_top}

    # -------------------------
    # Cross-modal novelty & attention scoring
    # -------------------------
    def compute_crossmodal_novelty_and_attention(
        self,
        scene_embeddings: np.ndarray,
        face_emotion_curve: Optional[np.ndarray],
        audio_energy_curve: Optional[np.ndarray],
        text_activity_curve: Optional[np.ndarray]
    ) -> Dict[str, float]:
        """
        Computes:
         - cross_modal_novelty_score (how often scenes diverge semantically)
         - multimodal_attention_score (agreement between face/audio/text peaks)
        """
        n = len(scene_embeddings)
        if n <= 1:
            return {"cross_modal_novelty_score": 0.0, "multimodal_attention_score": 0.0}

        # novelty: mean scene-to-scene dissimilarity
        sims = []
        for i in range(n - 1):
            sims.append(cosine_similarity(scene_embeddings[i:i+1], scene_embeddings[i+1:i+2])[0][0])
        novelty = float(np.mean(1 - np.array(sims)))

        # attention: correlation between normalized curves aggregated per scene
        def _agg(curve):
            if curve is None or len(curve) == 0:
                return None
            L = len(curve)
            bounds = np.linspace(0, L, n+1, dtype=int)
            return np.array([curve[bounds[i]:bounds[i+1]].mean() if bounds[i+1]>bounds[i] else 0.0 for i in range(n)])

        f_face = _agg(_smooth(face_emotion_curve, self.sigma))
        f_audio = _agg(_smooth(audio_energy_curve, self.sigma))
        f_text = _agg(_smooth(text_activity_curve, self.sigma))

        # combine only non-empty
        vectors = []
        if f_face is not None: vectors.append(_norm01(f_face))
        if f_audio is not None: vectors.append(_norm01(f_audio))
        if f_text is not None: vectors.append(_norm01(f_text))

        if len(vectors) < 2:
            attention_score = 0.0
        else:
            # pairwise mean correlation
            R = np.corrcoef(np.vstack(vectors))
            # average off-diagonal positive correlations
            m = R.shape[0]
            s = (R.sum() - m) / (m*(m-1) + 1e-9)
            attention_score = float(np.clip(s, -1.0, 1.0))

        return {"cross_modal_novelty_score": novelty, "multimodal_attention_score": attention_score}

    # -------------------------
    # Master extract function
    # -------------------------
    def extract_all(
        self,
        scene_frames: Optional[List[Any]] = None,
        scene_embeddings: Optional[np.ndarray] = None,
        face_emotion_curve: Optional[np.ndarray] = None,
        audio_energy_curve: Optional[np.ndarray] = None,
        pose_activity_curve: Optional[np.ndarray] = None,
        text_features: Optional[Dict[str, Any]] = None,
        topic_vectors: Optional[Dict[str, np.ndarray]] = None,
        class_prompts: Optional[List[str]] = None,
        scene_boundary_frames: Optional[List[int]] = None,
    ) -> Dict[str, Any]:
        """
        Run full pipeline. text_features is dict provided by TextProcessor and may contain:
          - 'scene_text_embeddings' : np.array [n_scenes, d]
          - 'text_activity_curve' : per-frame curve
          - 'text_emotion_curve' : per-frame
          - 'text_importance_per_scene' : [n_scenes]
          - 'transcript_timestamps', 'keyword_timeline', 'cta_timestamps'
          - 'video_topic_embedding'
        """
        t0 = time.time()
        scenes_emb = self.get_scene_embeddings(scene_frames=scene_frames, scene_embeddings=scene_embeddings)
        n_scenes = len(scenes_emb)

        # prepare curves from text_features
        text_activity_curve = None
        text_emotion_curve = None
        text_importance_per_scene = None
        scene_caption_embeddings = None
        video_topic_embedding = None

        if text_features:
            text_activity_curve = text_features.get("text_activity_curve", None)
            text_emotion_curve = text_features.get("text_emotion_curve", None)
            text_importance_per_scene = text_features.get("text_importance_per_scene", None)
            scene_caption_embeddings = text_features.get("scene_text_embeddings", None)
            video_topic_embedding = text_features.get("video_topic_embedding", None)

        # smoothing and normalizing input curves
        fac = _smooth(face_emotion_curve, self.sigma) if face_emotion_curve is not None else None
        aud = _smooth(audio_energy_curve, self.sigma) if audio_energy_curve is not None else None
        pose = _smooth(pose_activity_curve, self.sigma) if pose_activity_curve is not None else None
        txt_act = _smooth(text_activity_curve, self.sigma) if text_activity_curve is not None else None
        txt_em = _smooth(text_emotion_curve, self.sigma) if text_emotion_curve is not None else None

        # video-level embeddings
        vid_embs = self.compute_video_level_embeddings(scenes_emb, fac, aud, text_importance_per_scene)

        # scene semantic features
        scene_semantic = self.compute_scene_semantic_features() if len(scenes_emb)>0 else {}

        # topic features
        topic = self.compute_topics(scenes_emb, topic_vectors=topic_vectors, video_topic_embedding=video_topic_embedding)

        # events
        events = self.detect_events(scenes_emb, fac, aud, pose, txt_act, scene_boundary_frames)

        # emotion alignment
        emotion_align = self.compute_emotion_alignment(fac if fac is not None else None, txt_em if txt_em is not None else None)

        # narrative embedding
        narrative = self.compute_narrative_embedding(scenes_emb, scene_caption_embeddings, video_topic_embedding)

        # multimodal novelty & attention
        multimodal = self.compute_crossmodal_novelty_and_attention(scenes_emb, fac, aud, txt_act)

        # zero-shot genre/style
        genre = {}
        if class_prompts and len(class_prompts)>0:
            genre = self.zero_shot_genre_style(scenes_emb, class_prompts)

        # aggregate many numeric features to flat dict
        features = {}
        # scene-level stats
        features["n_scenes"] = n_scenes
        features["scene_similarity_mean"] = float(scene_semantic.get("scene_similarity_mean", 0.0))
        features["scene_similarity_std"] = float(scene_semantic.get("scene_similarity_std", 0.0))

        # video embedding stats length and norms
        if vid_embs["mean_embedding"] is not None:
            features["video_embedding_norm_mean"] = float(np.linalg.norm(vid_embs["mean_embedding"]))
            features["video_embedding_norm_weighted"] = float(np.linalg.norm(vid_embs["weighted_mean_embedding"]))
            features["video_embedding_var_mean"] = float(np.mean(vid_embs["var_embedding"]))

        # topic
        features.update(topic)

        # events
        features["number_of_events"] = events.get("number_of_events", 0)
        features["event_strength_max"] = float(max(events.get("event_strengths", [0])) if events.get("event_strengths") else 0.0)
        features["event_types"] = events.get("event_types", [])
        features["event_timestamps"] = events.get("event_timestamps", [])

        # emotion
        features.update(emotion_align)
        features.update(self.compute_emotion_features() if fac is not None else {"avg_emotion_valence":0,"emotion_variance":0,"peak_emotion_intensity":0})

        # narrative
        features.update(narrative)

        # multimodal
        features.update(multimodal)

        # genre/style
        features.update(genre)

        # attach some arrays (useful for downstream)
        outputs = {
            "features": features,
            "scene_embeddings": scenes_emb,
            "video_embeddings": vid_embs,
            "scene_semantic": scene_semantic,
            "events": events,
            "narrative": narrative,
            "multimodal_alignment": multimodal,
        }
        outputs["timing_seconds"] = time.time() - t0
        return outputs

# -------------------------
# Example (demo) usage:
# -------------------------
if __name__ == "__main__":
    # Demo: you must provide scene_frames (list of representative frames per scene)
    # and curves from your other processors (face_emotion_curve, audio_energy_curve, etc.)
    import numpy as np
    from PIL import Image

    # fake demo data
    n_scenes = 8
    # create blank images as placeholders (replace with real frames)
    scene_frames = [Image.new("RGB", (224, 224), color=(int(255*i/n_scenes), 120, 120)) for i in range(n_scenes)]
    face_emotion_curve = np.zeros(300)
    audio_energy_curve = np.zeros(300)
    # add some peaks
    face_emotion_curve[50] = 1.0
    audio_energy_curve[120] = 1.0

    # text_features stub (provided by TextProcessor)
    text_features = {
        "text_activity_curve": np.zeros(300),
        "text_emotion_curve": np.zeros(300),
        "text_importance_per_scene": np.ones(n_scenes),
        "scene_text_embeddings": None,
        "video_topic_embedding": None,
    }

    topic_vectors = {
        "sports": np.random.randn(512),
        "cooking": np.random.randn(512),
        "gaming": np.random.randn(512),
    }

    hl = HighLevelSemanticsOptimized(device=None, clip_model_name="ViT-B/32", fps=30)
    out = hl.extract_all(
        scene_frames=scene_frames,
        scene_embeddings=None,
        face_emotion_curve=face_emotion_curve,
        audio_energy_curve=audio_energy_curve,
        pose_activity_curve=None,
        text_features=text_features,
        topic_vectors=topic_vectors,
        class_prompts=["gaming", "vlog", "tutorial"],
        scene_boundary_frames=None,
    )

    print("Features keys:", list(out["features"].keys()))
    print("Timing (s):", out["timing_seconds"])
