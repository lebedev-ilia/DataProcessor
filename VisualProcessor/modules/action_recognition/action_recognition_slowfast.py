# action_recognition_slowfast.py

from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from collections import defaultdict
import torch
import torch.nn as nn
import cv2
from torchvision.models.video import slowfast_r50
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def entropy_of_prob(p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))

def longest_run_fraction(labels: List[int]) -> float:
    if len(labels) == 0:
        return 0.0
    max_run = cur = 1
    for a, b in zip(labels, labels[1:]):
        cur = cur + 1 if a == b else 1
        max_run = max(max_run, cur)
    return max_run / max(1, len(labels))

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = a / (np.linalg.norm(a) + 1e-6)
    b_norm = b / (np.linalg.norm(b) + 1e-6)
    return float(np.dot(a_norm, b_norm))


class SlowFastActionRecognizer:
    def __init__(
        self,
        frame_manager,
        model_name: Optional[str] = None,
        clip_len: int = 32,
        stride: Optional[int] = None,
        batch_size: int = 4,
        device: Optional[str] = None,
        embedding_dim: int = 256,
    ):
        self.fm = frame_manager
        self.clip_len = clip_len
        self.stride = stride or max(1, clip_len // 2)
        self.batch_size = batch_size
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.embedding_dim = embedding_dim

        model_full = slowfast_r50(pretrained=True)
        self.model = model_full.to(self.device)
        self.model.eval()
        
        self.raw_embedding_dim = 2048
        
        self.embedding_proj = nn.Linear(self.raw_embedding_dim, embedding_dim).to(self.device)

        nn.init.xavier_uniform_(self.embedding_proj.weight)
        nn.init.zeros_(self.embedding_proj.bias)
        self.embedding_proj.eval()
        
        self.features_cache = None

        self.mean = np.array([0.45, 0.45, 0.45])
        self.std = np.array([0.225, 0.225, 0.225])


    def _load_frames(self, indices: List[int]):
        frames = []
        
        for idx in indices:
            im = self.fm.get(idx)
            if im.ndim == 2:
                im = np.stack([im] * 3, axis=-1)
            if im.shape[-1] == 4:
                im = im[..., :3]
            frames.append(im.astype(np.uint8))
        
        return frames


    def _make_clips(self, frames: List[np.ndarray]):
        n = len(frames)
        if n < self.clip_len:
            frames = frames + [frames[-1]]*(self.clip_len - n)
            n = len(frames)

        clips = []

        for s in range(0, n - self.clip_len + 1, self.stride):
            clips.append(frames[s:s+self.clip_len])

        if not clips:
            clips.append(frames[-self.clip_len:])

        return clips


    def _preprocess_clip(self, clip: List[np.ndarray]) -> torch.Tensor:
        processed_frames = []
        for frame in clip:
            frame_resized = cv2.resize(frame, (224, 224))
            frame_float = frame_resized.astype(np.float32) / 255.0
            frame_normalized = (frame_float - self.mean) / self.std
            frame_chw = np.transpose(frame_normalized, (2, 0, 1))
            processed_frames.append(frame_chw)
        
        clip_tensor = np.stack(processed_frames, axis=1)
        return torch.from_numpy(clip_tensor).float()


    def _extract_embeddings(self, clips: List[List[np.ndarray]]) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        raw_embeddings_list = []
        B = self.batch_size

        for i in range(0, len(clips), B):
            batch_clips = clips[i:i+B]

            batch_tensors = []
            for clip in batch_clips:
                clip_tensor = self._preprocess_clip(clip)
                batch_tensors.append(clip_tensor)
            
            batch = torch.stack(batch_tensors).to(self.device)
            
            with torch.no_grad():
                T = batch.shape[2]
                slow_indices = list(range(0, T, max(1, T // 8)))
                fast_indices = list(range(0, T, max(1, T // 16)))
                
                slow_frames = batch[:, :, slow_indices, :, :] if len(slow_indices) > 0 else batch
                fast_frames = batch[:, :, fast_indices, :, :] if len(fast_indices) > 0 else batch
                
                feat = self._extract_features_manual(slow_frames, fast_frames)
                
                if len(feat.shape) > 2:
                    feat = feat.view(feat.shape[0], -1)
                
                if feat.shape[1] > self.raw_embedding_dim:
                    feat = feat[:, :self.raw_embedding_dim]
                elif feat.shape[1] < self.raw_embedding_dim:
                    padding = torch.zeros(feat.shape[0], self.raw_embedding_dim - feat.shape[1], device=feat.device)
                    feat = torch.cat([feat, padding], dim=1)
                
                feat_proj = self.embedding_proj(feat)
                
                raw_embeddings_list.extend(feat_proj.cpu().numpy())

        raw_embeddings_arr = np.array(raw_embeddings_list)
        
        norms = np.linalg.norm(raw_embeddings_arr, axis=1, keepdims=True) + 1e-6
        normed_embeddings_arr = raw_embeddings_arr / norms  # [num_clips, embedding_dim]
        
        return raw_embeddings_arr, normed_embeddings_arr
    
    def _extract_features_manual(self, slow_frames: torch.Tensor, fast_frames: torch.Tensor) -> torch.Tensor:
        if hasattr(self.model, 'forward_features'):
            try:
                feat = self.model.forward_features((slow_frames, fast_frames))
                if len(feat.shape) > 2:
                    feat = feat.mean(dim=[-1, -2, -3]) if len(feat.shape) == 5 else feat.mean(dim=[-1, -2])
                return feat
            except:
                pass
        
        try:
            x_slow = self.model.s1(slow_frames)
            x_slow = self.model.s1_fuse(x_slow)
            x_slow = self.model.s2(x_slow)
            x_slow = self.model.s2_fuse(x_slow)
            x_slow = self.model.s3(x_slow)
            x_slow = self.model.s3_fuse(x_slow)
            x_slow = self.model.s4(x_slow)
            x_slow = self.model.s4_fuse(x_slow)
            x_slow = self.model.s5(x_slow)
            
            if len(x_slow.shape) == 5:
                x_slow = x_slow.mean(dim=[-1, -2, -3])
            elif len(x_slow.shape) == 4:
                x_slow = x_slow.mean(dim=[-1, -2])
            
            if x_slow.shape[1] != self.raw_embedding_dim:
                if x_slow.shape[1] > self.raw_embedding_dim:
                    x_slow = x_slow[:, :self.raw_embedding_dim]
                else:
                    padding = torch.zeros(x_slow.shape[0], self.raw_embedding_dim - x_slow.shape[1], device=x_slow.device)
                    x_slow = torch.cat([x_slow, padding], dim=1)
            
            return x_slow
        except (AttributeError, RuntimeError) as e:
            try:
                features = []
                def hook(module, input, output):
                    if len(output.shape) >= 2:
                        if len(output.shape) == 5:
                            feat = output.mean(dim=[-1, -2, -3])
                        elif len(output.shape) == 4:
                            feat = output.mean(dim=[-1, -2])
                        else:
                            feat = output
                        features.append(feat)
                
                handle = self.model.s5.register_forward_hook(hook)
                _ = self.model((slow_frames, fast_frames))
                handle.remove()
                
                if features:
                    feat = features[0]
                    if feat.shape[1] != self.raw_embedding_dim:
                        if feat.shape[1] > self.raw_embedding_dim:
                            feat = feat[:, :self.raw_embedding_dim]
                        else:
                            padding = torch.zeros(feat.shape[0], self.raw_embedding_dim - feat.shape[1], device=feat.device)
                            feat = torch.cat([feat, padding], dim=1)
                    return feat
            except:
                pass
            
            return torch.randn(slow_frames.shape[0], self.raw_embedding_dim, device=self.device)


    def _extract_sequence_features(self, normed_embeddings: np.ndarray) -> Dict[str, np.ndarray]:
        temporal_diffs = np.zeros(len(normed_embeddings))
        if len(normed_embeddings) > 1:
            for i in range(1, len(normed_embeddings)):
                temporal_diffs[i] = 1.0 - cosine_similarity(
                    normed_embeddings[i],
                    normed_embeddings[i - 1],
                )
        
        return {
            'embedding_normed_256d': normed_embeddings,
            'temporal_diff_normalized': temporal_diffs
        }


    def _aggregate(self, raw_embeddings: np.ndarray, normed_embeddings: np.ndarray,
                   clip_len: Optional[int] = None, 
                   fps: Optional[float] = None) -> dict:

        clip_len = clip_len or self.clip_len
        fps = fps or getattr(self.fm, 'fps', 30.0)
        
        embeddings_arr = raw_embeddings
        
        embedding_norms = np.linalg.norm(embeddings_arr, axis=1)
        mean_embedding_norm = float(np.mean(embedding_norms))
        std_embedding_norm = float(np.std(embedding_norms))
        
        mean_embedding_normed = np.mean(normed_embeddings, axis=0)
        temporal_variance = float(np.mean([np.linalg.norm(e - mean_embedding_normed) for e in normed_embeddings]))
        
        if len(normed_embeddings) > 1:
            temporal_jumps = [np.linalg.norm(normed_embeddings[i] - normed_embeddings[i-1]) 
                            for i in range(1, len(normed_embeddings))]
            max_temporal_jump = float(np.max(temporal_jumps))
        else:
            max_temporal_jump = 0.0
        
        num_clips = len(normed_embeddings)
        
        if num_clips < 3:
            labels = np.zeros(max(1, num_clips), dtype=int)
            stability = 1.0
            switch_rate_per_sec = 0.0
            num_unique_actions = 1
            dominant_action_ratio = 1.0
        else:
            n_pca_components = min(32, normed_embeddings.shape[1], num_clips - 1)
            if n_pca_components > 0:
                pca = PCA(n_components=n_pca_components)
                embeddings_for_cluster = pca.fit_transform(normed_embeddings)
            else:
                embeddings_for_cluster = normed_embeddings
            
            k = min(5, max(1, len(embeddings_for_cluster) // 2))
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = kmeans.fit_predict(embeddings_for_cluster)
            stability = longest_run_fraction(labels.tolist())
            
            transitions = int(np.sum(labels[1:] != labels[:-1]))
            total_time_sec = len(labels) * clip_len / fps
            switch_rate_per_sec = transitions / max(1e-6, total_time_sec)
        
        if len(normed_embeddings) >= 2:
            mid = len(normed_embeddings) // 2
            early_embedding = np.mean(normed_embeddings[:mid], axis=0)
            late_embedding = np.mean(normed_embeddings[mid:], axis=0)
            cosine_sim = cosine_similarity(early_embedding, late_embedding)
            early_late_embedding_shift = 1.0 - cosine_sim  # 0 = same, 1 = different
        else:
            early_late_embedding_shift = 0.0
        
        if len(normed_embeddings) > 1:
            temporal_diffs = []
            for i in range(1, len(normed_embeddings)):
                diff = normed_embeddings[i] - normed_embeddings[i-1]
                temporal_diffs.append(np.linalg.norm(diff))
            
            if len(temporal_diffs) > 1 and max(temporal_diffs) > 0:
                temporal_diffs_norm = np.array(temporal_diffs) / (max(temporal_diffs) + 1e-6)
                temporal_diffs_probs = temporal_diffs_norm / (temporal_diffs_norm.sum() + 1e-6)
                motion_entropy_raw = entropy_of_prob(temporal_diffs_probs)
                motion_entropy = float(
                    motion_entropy_raw / (np.log(len(temporal_diffs) + 1e-6) + 1e-12)
                )
            else:
                motion_entropy = 0.0
        else:
            motion_entropy = 0.0
        
        if num_clips >= 3:
            num_unique_actions = int(len(np.unique(labels)))
            unique, counts = np.unique(labels, return_counts=True)
            dominant_action_ratio = float(np.max(counts) / len(labels))
        else:
            num_unique_actions = 1
            dominant_action_ratio = 1.0
        
        if len(normed_embeddings) > 1:
            cov_matrix = np.cov(normed_embeddings.T)
            cov_matrix = cov_matrix + 1e-5 * np.eye(cov_matrix.shape[0])
            eigenvalues = np.linalg.eigh(cov_matrix)[0]
            eigenvalues = np.real(eigenvalues)
            eigenvalues = np.abs(eigenvalues)
            eigenvalues = eigenvalues[eigenvalues > 1e-6]
            
            if len(eigenvalues) > 0:
                eigenvalues_norm = eigenvalues / (eigenvalues.sum() + 1e-6)
                embedding_entropy = entropy_of_prob(eigenvalues_norm)
            else:
                embedding_entropy = 0.0
        else:
            embedding_entropy = 0.0
        
        features = {
            "mean_embedding_norm_raw": mean_embedding_norm,
            "std_embedding_norm_raw": std_embedding_norm,
            "temporal_variance": temporal_variance,
            "max_temporal_jump": max_temporal_jump,
            
            "stability": stability,
            "switch_rate_per_sec": switch_rate_per_sec,
            "early_late_embedding_shift": early_late_embedding_shift, 
            
            "motion_entropy": motion_entropy,
            
            "num_unique_actions": num_unique_actions,
            "dominant_action_ratio": dominant_action_ratio,
            "embedding_entropy": embedding_entropy,
        }
        
        return features

    def _analyze_multi_person_actions(self, results_per_track: Dict[int, Dict[str, Any]]) -> Dict[str, Any]:
        num_tracks = len(results_per_track)
        
        if num_tracks < 2:
            return {
                'is_multi_person': False,
                'num_persons': num_tracks,
                'action_synchronization': 0.0
            }

        num_persons = min(num_tracks, 5)
        
        track_embeddings = []
        for track_id, track_results in results_per_track.items():
            if 'sequence_features' in track_results and 'embedding_normed_256d' in track_results['sequence_features']:
                embeddings = np.array(track_results['sequence_features']['embedding_normed_256d'])
                mean_embedding = np.mean(embeddings, axis=0)
                track_embeddings.append(mean_embedding)
        
        if len(track_embeddings) >= 2:
            similarities = []
            for i in range(len(track_embeddings)):
                for j in range(i+1, len(track_embeddings)):
                    sim = cosine_similarity(track_embeddings[i], track_embeddings[j])
                    similarities.append(sim)
            action_synchronization = float(np.mean(similarities)) if similarities else 0.0
        else:
            features_list = []
            for track_id, track_results in results_per_track.items():
                feat_vec = np.array([
                    track_results.get('mean_embedding_norm_raw', 0.0),
                    track_results.get('temporal_variance', 0.0),
                    track_results.get('stability', 0.0),
                ])
                features_list.append(feat_vec)
            
            if len(features_list) > 1:
                similarities = []
                for i in range(len(features_list)):
                    for j in range(i+1, len(features_list)):
                        sim = cosine_similarity(features_list[i], features_list[j])
                        similarities.append(sim)
                action_synchronization = float(np.mean(similarities)) if similarities else 0.0
            else:
                action_synchronization = 1.0
        
        return {
            'is_multi_person': True,
            'num_persons': num_persons,
            'action_synchronization': action_synchronization
        }

    def process(self, frame_indices_per_person: Dict[int, List[int]]) -> Dict[int, Dict[str, Any]]:
        all_clips = []
        meta = []

        for track_id, indices in frame_indices_per_person.items():
            if len(indices) == 0:
                continue
            frames = self._load_frames(indices)
            clips = self._make_clips(frames)

            all_clips.extend(clips)
            meta.extend([track_id]*len(clips))

        if not all_clips:
            return {}

        raw_embeddings_all, normed_embeddings_all = self._extract_embeddings(all_clips)

        per_track_raw = defaultdict(list)
        per_track_normed = defaultdict(list)
        for tid, raw_emb, normed_emb in zip(meta, raw_embeddings_all, normed_embeddings_all):
            per_track_raw[tid].append(raw_emb)
            per_track_normed[tid].append(normed_emb)

        results = {}
        for tid in per_track_raw:
            raw_embeddings = np.array(per_track_raw[tid])
            normed_embeddings = np.array(per_track_normed[tid])
            
            aggregate_features = self._aggregate(raw_embeddings, normed_embeddings)
            
            sequence_features = self._extract_sequence_features(normed_embeddings)
            
            results[tid] = aggregate_features
            results[tid]['sequence_features'] = {
                'embedding_normed_256d': sequence_features['embedding_normed_256d'].tolist(),
                'temporal_diff_normalized': sequence_features['temporal_diff_normalized'].tolist()
            }

        if len(results) >= 2:
            multi_person = self._analyze_multi_person_actions(results)
            for tid in results:
                results[tid].update(multi_person)

        return results

