# video_story_structure_optimized.py

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
import torch
import clip
import os
from sentence_transformers import SentenceTransformer
import mediapipe as mp
from scipy.ndimage import uniform_filter1d

def compute_optical_flow(frame_manager, frames):
    """Compute dense optical flow magnitude per frame"""
    flows = []
    prev_gray = cv2.cvtColor(frame_manager.get(frames[0]), cv2.COLOR_RGB2GRAY)
    for idx in frames[1:]:
        frame = frame_manager.get(idx)
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                            0.5, 3, 15, 3, 5, 1.2, 0)
        mag, _ = cv2.cartToPolar(flow[...,0], flow[...,1])
        flows.append(np.mean(mag))
        prev_gray = gray
    return np.array(flows)

def embedding_diff(embeddings):
    """Compute frame-to-frame embedding difference (cosine distance)"""
    diffs = []
    for i in range(1, len(embeddings)):
        sim = cosine_similarity([embeddings[i-1]], [embeddings[i]])[0][0]
        diffs.append(1 - sim)
    return np.array(diffs)

def smooth_signal(signal, window=3):
    return uniform_filter1d(signal.astype(np.float32), size=window)

# -----------------------------
# Story Structure Pipeline
# -----------------------------
class StoryStructurePipelineOptimized:
    def __init__(self, frame_manager, frame_indices, clip_model='ViT-B/32', sentence_model='all-MiniLM-L6-v2'):
        self.frame_manager = frame_manager
        self.frame_indices = frame_indices

        # Load models
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.clip_preprocess = clip.load(clip_model, device=self.device)
        m_name = sentence_model.replace("-","_")
        model_path = f"{os.path.dirname(__file__)}/models/{m_name}"
        self.sentence_model = SentenceTransformer(model_name_or_path=sentence_model, cache_folder=model_path)
        
        # Face tracking
        self.mp_face = mp.solutions.face_mesh.FaceMesh(static_image_mode=True)

        # Outputs
        self.features = {}

    # -----------------------------
    # 1. CLIP embeddings
    # -----------------------------
    def compute_clip_embeddings(self):
        embeddings = []
        for idx in tqdm(self.frame_indices, desc="CLIP embeddings"):
            frame = self.frame_manager.get(idx)
            img = Image.fromarray(frame)
            img_input = self.clip_preprocess(img).unsqueeze(0).to(self.device)
            with torch.no_grad():
                emb = self.clip_model.encode_image(img_input)
                emb = emb / emb.norm(dim=-1, keepdim=True)
                embeddings.append(emb.cpu().numpy()[0])
        self.clip_embeddings = np.array(embeddings)
        return self.clip_embeddings

    # -----------------------------
    # 2. Story Segmentation
    # -----------------------------
    def story_segmentation(self, n_segments=5):
        """Segment video based on CLIP embeddings + smoothing"""
        # Smooth embeddings
        smooth_emb = smooth_signal(self.clip_embeddings, window=3)
        # Hierarchical clustering
        clustering = AgglomerativeClustering(n_clusters=n_segments, metric='cosine', linkage='average')
        labels = clustering.fit_predict(smooth_emb)
        self.segment_labels = labels
        
        # Features
        segment_durations = [np.sum(labels==i) for i in range(n_segments)]
        self.features['number_of_story_segments'] = n_segments
        self.features['avg_story_segment_duration'] = np.mean(segment_durations)
        self.features['abrupt_story_transition_count'] = np.sum(np.diff(labels)!=0)
        # Narrative continuity score: mean similarity between consecutive segments
        cont_scores = []
        cont_std = []
        for i in range(n_segments-1):
            idx1 = np.where(labels==i)[0]
            idx2 = np.where(labels==i+1)[0]
            sim = cosine_similarity(self.clip_embeddings[idx1].mean(axis=0).reshape(1,-1),
                                    self.clip_embeddings[idx2].mean(axis=0).reshape(1,-1))[0][0]
            cont_scores.append(sim)
        self.features['narrative_continuity_score'] = np.mean(cont_scores)
        self.features['narrative_continuity_std'] = np.std(cont_scores)
        return self.features

    # -----------------------------
    # 3. Hook Features
    # -----------------------------
    def hook_features(self, hook_seconds=5):
        n_frames = min(len(self.frame_indices), hook_seconds)
        hook_frames = self.frame_indices[:n_frames]

        # Optical flow
        if len(hook_frames) > 1:
            hook_flow = compute_optical_flow(self.frame_manager, hook_frames)
            hook_flow_smooth = smooth_signal(hook_flow)
            self.features['hook_motion_intensity'] = np.mean(hook_flow_smooth)
            self.features['hook_cut_rate'] = np.sum(hook_flow_smooth > np.percentile(hook_flow_smooth,75)) / hook_seconds
            self.features['hook_motion_spikes'] = np.sum(hook_flow_smooth > np.percentile(hook_flow_smooth,90))
        else:
            self.features['hook_motion_intensity'] = 0
            self.features['hook_cut_rate'] = 0
            self.features['hook_motion_spikes'] = 0

        # Face presence
        face_count = 0
        for idx in self.frame_indices:
            frame = self.frame_manager.get(idx)
            results = self.mp_face.process(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            if results.multi_face_landmarks:
                face_count +=1
        self.features['hook_face_presence'] = face_count / n_frames

        # Visual surprise: embedding jumps
        hook_emb = self.clip_embeddings[:n_frames]
        diff = embedding_diff(hook_emb)
        self.features['hook_visual_surprise_score'] = np.mean(diff) if diff.size else 0
        self.features['hook_visual_surprise_std'] = np.std(diff) if diff.size else 0

        # Brightness / Saturation spike
        brightness = []
        saturation = []
        for idx in hook_frames:
            frame = self.frame_manager.get(idx)
            hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
            brightness.append(np.mean(hsv[:,:,2]))
            saturation.append(np.mean(hsv[:,:,1]))
        self.features['hook_brightness_spike'] = max(brightness) - np.mean(brightness)
        self.features['hook_saturation_spike'] = max(saturation) - np.mean(saturation)
        return self.features

    # -----------------------------
    # 4. Climax Detection
    # -----------------------------
    def climax_detection(self):
        # Combine signals: motion + embedding diff
        motion = compute_optical_flow(self.frame_manager, self.frame_indices)
        motion_smooth = smooth_signal(motion)
        embed_diff = embedding_diff(self.clip_embeddings)
        embed_diff_smooth = smooth_signal(embed_diff)
        combined_signal = motion_smooth[:len(embed_diff_smooth)] + embed_diff_smooth

        peak_idx = np.argmax(combined_signal)
        self.features['climax_timestamp'] = peak_idx  # in frames (1 FPS)
        self.features['climax_strength'] = combined_signal[peak_idx]
        self.features['number_of_peaks'] = np.sum(combined_signal > np.percentile(combined_signal,90))
        self.features['climax_duration'] = np.sum(combined_signal > np.percentile(combined_signal,50))
        self.features['story_energy_curve'] = combined_signal.tolist()
        return self.features

    # -----------------------------
    # 5. Character-level Features
    # -----------------------------
    def character_features(self):
        face_tracks = []
        for idx in self.frame_indices:
            frame = self.frame_manager.get(idx)
            results = self.mp_face.process(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            face_tracks.append(len(results.multi_face_landmarks) if results.multi_face_landmarks else 0)
        face_tracks = np.array(face_tracks)
        self.features['number_of_speakers'] = max(face_tracks)
        self.features['main_character_screen_time'] = np.sum(face_tracks>0)/len(face_tracks)
        self.features['speaker_switch_rate'] = np.sum(np.diff(face_tracks>0)!=0) / len(face_tracks)
        self.features['face_presence_curve'] = face_tracks.tolist()
        return self.features

    # -----------------------------
    # 6. Topic Features
    # -----------------------------
    def topic_features(self, subtitles=None):
        if subtitles is None or len(subtitles)==0:
            self.features['number_of_topics'] = 0
            self.features['avg_topic_duration'] = 0
            self.features['topic_shift_times'] = []
            self.features['topic_diversity'] = 0
            self.features['semantic_coherence_score'] = 0
            return self.features
        
        embeddings = self.sentence_model.encode(subtitles)
        clustering = AgglomerativeClustering(n_clusters=min(5,len(subtitles)), metric='cosine', linkage='average')
        labels = clustering.fit_predict(embeddings)
        self.features['number_of_topics'] = len(np.unique(labels))
        durations = [sum(np.array(labels)==i) for i in range(len(np.unique(labels)))]
        self.features['avg_topic_duration'] = np.mean(durations)
        self.features['topic_shift_times'] = np.where(np.diff(labels)!=0)[0].tolist()
        self.features['topic_diversity'] = len(np.unique(labels))/len(labels)
        # Semantic coherence
        coherences = []
        for i in range(len(np.unique(labels))):
            idx = np.where(labels==i)[0]
            if len(idx)>1:
                sim = cosine_similarity(embeddings[idx]).mean()
                coherences.append(sim)
        self.features['semantic_coherence_score'] = np.mean(coherences) if coherences else 0
        return self.features

    # -----------------------------
    # 7. Run All
    # -----------------------------
    def extract_all_features(self, subtitles=None):
        self.compute_clip_embeddings()
        self.story_segmentation()
        self.hook_features()
        self.climax_detection()
        self.character_features()
        self.topic_features(subtitles=subtitles)
        return self.features
