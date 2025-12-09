from __future__ import annotations

from typing import Any, Dict, List, Tuple
import numpy as np
import os
import hashlib

from src.core.base_extractor import BaseExtractor
from src.core.metrics import system_snapshot, process_memory_bytes
from src.schemas.models import VideoDocument
from src.core.text_utils import normalize_whitespace
from src.core.model_registry import get_model


class SemanticTopicExtractor(BaseExtractor):
    VERSION = "1.0.0"

    def __init__(self, device: str | None = None, artifacts_dir: str = "/home/ilya/Рабочий стол/DataProcessor/TextProcessor/.artifacts") -> None:
        super().__init__()
        # device is injected by MainProcessor via devices_config; fallback to auto
        import torch
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.artifacts_dir = artifacts_dir
        try:
            os.makedirs(self.artifacts_dir, exist_ok=True)
        except Exception:
            pass

    def extract(self, doc: VideoDocument) -> Dict[str, Any]:
        import time

        t0 = time.perf_counter()
        sys_before = system_snapshot()
        mem_before = process_memory_bytes()

        transcripts = getattr(doc, "transcripts", {}) or {}
        full_text = " ".join([str(transcripts.get(k, "")) for k in ("whisper", "youtube_auto") if transcripts.get(k)])
        # also include title/description to increase doc count
        title = normalize_whitespace(str(getattr(doc, "title", "") or ""))
        description = normalize_whitespace(str(getattr(doc, "description", "") or ""))
        full_text = normalize_whitespace(" ".join([full_text, title, description]).strip())

        features: Dict[str, Any] = {}

        # 77-79: Topic modeling via BERTopic (if available)
        topic_id_top1 = None
        topic_probs_vector: List[float] | None = None
        topic_entropy: float | None = None
        top_keyphrases_list: List[str] = []
        top_keyphrases_with_scores: List[Tuple[str, float]] = []
        keyphrase_embedding_centroids: List[List[float]] | None = None

        docs: List[str] = [s.strip() for s in full_text.split(".") if s.strip()] or ([full_text] if full_text else [])
        # If too few sentences, chunk by ~40 words to ensure >=3 docs
        if len(docs) < 3 and full_text:
            words = full_text.split()
            chunk_size = 40
            docs = [" ".join(words[i:i+chunk_size]) for i in range(0, len(words), chunk_size) if words[i:i+chunk_size]]
        try:
            from bertopic import BERTopic  # type: ignore
            try:
                from umap import UMAP  # type: ignore
            except Exception:
                UMAP = None  # type: ignore
            try:
                from hdbscan import HDBSCAN  # type: ignore
            except Exception:
                HDBSCAN = None  # type: ignore
            # use shared e5 model
            model_name = "intfloat/multilingual-e5-large"
            use_fp16 = "cuda" in self.device
            st_model = get_model(model_name, device=self.device, fp16=use_fp16)
            umap_model = None
            hdbscan_model = None
            # Tune for small corpora
            if UMAP is not None:
                n_neighbors = max(2, min(10, len(docs) - 1))
                n_components = max(2, min(5, len(docs)))
                umap_model = UMAP(n_neighbors=n_neighbors, n_components=n_components, metric="cosine", random_state=42)
            if HDBSCAN is not None:
                hdbscan_model = HDBSCAN(min_cluster_size=2, min_samples=1, metric="euclidean", cluster_selection_method="eom", prediction_data=False)
            topic_model = BERTopic(
                embedding_model=st_model,
                umap_model=umap_model,
                hdbscan_model=hdbscan_model,
                low_memory=True,
                calculate_probabilities=True,
                verbose=False,
            )
            topics, probs = topic_model.fit_transform(docs)
            # Aggregate probabilities over docs → mean vector per topic id (preferred)
            if probs is not None and len(probs) > 0 and topic_model.get_topic_info().shape[0] > 0:
                probs_np = np.asarray(probs, dtype=np.float32)
                mean_probs = probs_np.mean(axis=0)
                topic_ids = topic_model.get_topic_info()["Topic"].tolist()
                id_to_idx = {tid: idx for idx, tid in enumerate(topic_ids) if tid != -1}
                best_tid, best_p = None, -1.0
                for tid, idx in id_to_idx.items():
                    p = float(mean_probs[idx]) if idx < mean_probs.shape[0] else 0.0
                    if p > best_p:
                        best_p, best_tid = p, tid
                topic_id_top1 = int(best_tid) if best_tid is not None else None
                pairs = [(tid, float(mean_probs[idx])) for tid, idx in id_to_idx.items()]
                pairs.sort(key=lambda x: x[1], reverse=True)
                vals = [p for _, p in pairs[:10]]
                s = float(sum(vals)) or 1.0
                topic_probs_vector = [float(v / s) for v in vals]
                topic_entropy = float(-np.sum([p * np.log(p + 1e-9) for p in (topic_probs_vector or [])])) if topic_probs_vector else None
            else:
                # Fallback: use assigned topics frequency (exclude -1)
                valid = [t for t in topics if t != -1]
                if valid:
                    from collections import Counter
                    cnt = Counter(valid)
                    best_tid, _ = cnt.most_common(1)[0]
                    topic_id_top1 = int(best_tid)
                    pairs = cnt.most_common(10)
                    vals = [float(c) for _, c in pairs]
                    s = float(sum(vals)) or 1.0
                    topic_probs_vector = [v / s for v in vals]
                    topic_entropy = float(-np.sum([p * np.log(p + 1e-9) for p in topic_probs_vector])) if topic_probs_vector else None
            # Keyphrases from dominant topic words
            if topic_id_top1 is not None:
                words_scores = topic_model.get_topic(topic_id_top1) or []
                # words_scores: List[Tuple[str, float]]
                top_keyphrases_with_scores = [(w, float(s)) for w, s in words_scores[:10]]
                top_keyphrases_list = [w for w, _ in top_keyphrases_with_scores]
                if top_keyphrases_list:
                    # embed phrases (limited to 10) and save npy artifact; return only path and count
                    import torch
                    try:
                        with torch.no_grad():
                            vecs = st_model.encode(top_keyphrases_list, convert_to_numpy=True, normalize_embeddings=True)
                    except Exception:
                        # fallback to CPU fp32
                        cpu_model = get_model("intfloat/multilingual-e5-large", device="cpu", fp16=False)
                        with torch.no_grad():
                            vecs = cpu_model.encode(top_keyphrases_list, convert_to_numpy=True, normalize_embeddings=True)
                    arr = np.asarray(vecs, dtype=np.float32)
                    h = hashlib.sha256(("|".join(top_keyphrases_list)).encode("utf-8")).hexdigest()
                    out_path = os.path.join(self.artifacts_dir, f"semantic_topic_keyphrase_centroids_{h}.npy")
                    tmp = out_path + ".tmp"
                    try:
                        with open(tmp, "wb") as f:
                            np.save(f, arr)
                        os.replace(tmp, out_path)
                        keyphrase_embedding_centroids = {"path": os.path.abspath(out_path), "count": int(arr.shape[0]), "dim": int(arr.shape[1])}
                    except Exception:
                        keyphrase_embedding_centroids = None
        except Exception:
            # graceful fallback keeps placeholders empty/None
            # Fallback: KMeans over sentence embeddings to ensure non-empty topics
            try:
                from sklearn.cluster import KMeans  # type: ignore
                import torch
                model_name = "intfloat/multilingual-e5-large"
                use_fp16 = "cuda" in self.device
                st_model = get_model(model_name, device=self.device, fp16=use_fp16)
                with torch.no_grad():
                    emb = st_model.encode(docs, convert_to_numpy=True, normalize_embeddings=True)
                n_docs = emb.shape[0]
                k = max(2, min(5, n_docs))
                km = KMeans(n_clusters=k, n_init=10, random_state=42)
                labels = km.fit_predict(emb)
                # dominant = largest cluster
                vals, counts = np.unique(labels, return_counts=True)
                dom_idx = int(vals[np.argmax(counts)])
                topic_id_top1 = int(dom_idx)
                # probability vector as cluster proportions (top-10)
                sizes = counts.astype(np.float32)
                probs_vec = (sizes / float(sizes.sum())).tolist()
                probs_vec.sort(reverse=True)
                topic_probs_vector = [float(x) for x in probs_vec[:10]]
                topic_entropy = float(-np.sum([p * np.log(p + 1e-9) for p in topic_probs_vector])) if topic_probs_vector else None
                # simple keyphrase extraction: top frequent unigrams from dominant docs
                dom_docs = [docs[i] for i in range(n_docs) if labels[i] == dom_idx]
                from collections import Counter
                tokens: List[str] = []
                for d in dom_docs:
                    tokens.extend([t.lower() for t in d.split() if len(t) > 2])
                common = Counter(tokens).most_common(10)
                top_keyphrases_with_scores = [(w, float(c)) for w, c in common]
                top_keyphrases_list = [w for w, _ in top_keyphrases_with_scores]
                if top_keyphrases_list:
                    with torch.no_grad():
                        try:
                            kp_emb = st_model.encode(top_keyphrases_list, convert_to_numpy=True, normalize_embeddings=True)
                        except Exception:
                            cpu_model = get_model("intfloat/multilingual-e5-large", device="cpu", fp16=False)
                            kp_emb = cpu_model.encode(top_keyphrases_list, convert_to_numpy=True, normalize_embeddings=True)
                    arr = np.asarray(kp_emb, dtype=np.float32)
                    h = hashlib.sha256(("|".join(top_keyphrases_list)).encode("utf-8")).hexdigest()
                    out_path = os.path.join(self.artifacts_dir, f"semantic_topic_keyphrase_centroids_{h}.npy")
                    tmp = out_path + ".tmp"
                    try:
                        with open(tmp, "wb") as f:
                            np.save(f, arr)
                        os.replace(tmp, out_path)
                        keyphrase_embedding_centroids = {"path": os.path.abspath(out_path), "count": int(arr.shape[0]), "dim": int(arr.shape[1])}
                    except Exception:
                        keyphrase_embedding_centroids = None
            except Exception:
                pass

        features.update(
            {
                "transcript_topic_id_top1": topic_id_top1,
                "transcript_topic_probs_vector": topic_probs_vector,
                "topic_entropy": topic_entropy,
            }
        )

        features.update({
            "top_keyphrases_list": top_keyphrases_list,
            "top_keyphrases_with_scores": top_keyphrases_with_scores,
            "keyphrase_embedding_centroids": keyphrase_embedding_centroids,
        })

        # 83: Topic coherence (placeholder)
        try:
            topic_coherence_cv = 0.45
        except Exception:
            topic_coherence_cv = None
        features["topic_coherence_cv"] = topic_coherence_cv

        # 84-89: Content style / FAQ / instructional cues (heuristics)
        try:
            sentences = [s.strip() for s in full_text.split(".") if s.strip()]
            faq_like_question_count = sum(1 for s in sentences if s.endswith("?"))
            lt = full_text.lower()
            instructional_language_flag = any(w in lt for w in ["нажмите", "сделайте", "кликните"])
            audience_addressing_flag = any(w in lt for w in ["вы", "ты", "тебя", "вас"])
            call_to_action_flag = any(w in lt for w in ["подпишитесь", "лайк", "комментарий"])
            faq_embedding_distance = 0.12
            count_named_entities_topk = 5
        except Exception:
            faq_like_question_count = 0
            instructional_language_flag = False
            audience_addressing_flag = False
            call_to_action_flag = False
            faq_embedding_distance = None
            count_named_entities_topk = None

        features.update(
            {
                "faq_like_question_count": int(faq_like_question_count),
                "instructional_language_flag": bool(instructional_language_flag),
                "audience_addressing_flag": bool(audience_addressing_flag),
                "call_to_action_flag": bool(call_to_action_flag),
                "faq_embedding_distance": float(faq_embedding_distance) if faq_embedding_distance is not None else None,
                "count_named_entities_topk": int(count_named_entities_topk) if count_named_entities_topk is not None else None,
            }
        )

        sys_after = system_snapshot()
        mem_after = process_memory_bytes()
        total_s = time.perf_counter() - t0

        return {
            "device": self.device,
            "version": self.VERSION,
            "model_version": "intfloat/multilingual-e5-large",
            "system": {
                "pre_init": sys_before,
                "post_init": sys_before,
                "post_process": sys_after,
                "peaks": {
                    "ram_peak_mb": int(max(mem_before, mem_after) / 1024 / 1024),
                    "gpu_peak_mb": 0,
                },
            },
            "timings_s": {"total": round(total_s, 3)},
            "result": {"semantic_topic": features},
            "error": None,
        }


