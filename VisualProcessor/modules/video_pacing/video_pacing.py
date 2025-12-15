import cv2
import numpy as np
import torch
from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2lab
from scipy.stats import entropy
from typing import List, Dict

import warnings
warnings.filterwarnings("ignore")

# Подключаем CLIP
import clip
device = "cuda" if torch.cuda.is_available() else "cpu"

class VideoPacingPipelineVisualOptimized:
    def __init__(self, frame_manager, frame_indices, clip_model_name="ViT-B/32", batch_size=32, downscale_factor=0.25):
        """
        batch_size: батч для CLIP
        downscale_factor: для Optical Flow и color/lighting features
        """
        self.batch_size = batch_size
        self.downscale_factor = downscale_factor

        # Загружаем кадры через FrameManager
        self.frame_manager = frame_manager
        self.frame_indices = frame_indices
        self.total_frames = len(frame_indices)
        self.fps = self.frame_manager.fps

        # CLIP модель
        self.clip_model, _ = clip.load(clip_model_name, device=device)
        self.clip_model.eval()

        # Определяем шоты и сцены
        self.shot_boundaries = self._detect_shots()
        self.scene_boundaries = self.shot_boundaries  # для упрощения

    def _get_resize_frame(self, idx):
        return cv2.resize(self.frame_manager.get(idx), (0, 0), fx=self.downscale_factor, fy=self.downscale_factor) 

    # -------------------------
    # Shot Detection with SSIM
    # -------------------------

    def _safe_ssim(self, img1, img2):
        h, w = img1.shape[:2]
        min_side = min(h, w)

        if min_side < 7:
            # fallback: считаем, что кадры сильно отличаются
            return 0.0

        win_size = min(7, min_side if min_side % 2 == 1 else min_side - 1)

        return ssim(
            img1,
            img2,
            channel_axis=-1,
            win_size=win_size
        )

    def _detect_shots(self) -> List[int]:
        shot_indices = [0]
        prev_frame = self._get_resize_frame(0)
        for idx in self.frame_indices:
            curr_frame = self._get_resize_frame(idx)
            score = self._safe_ssim(prev_frame, curr_frame)
            if score < 0.95:  # threshold for hard cut
                shot_indices.append(idx)
                prev_frame = curr_frame
        return shot_indices

    # -------------------------
    # Shot Features
    # -------------------------
    def extract_shot_features(self) -> Dict:
        durations = np.diff([0] + self.shot_boundaries + [self.total_frames])
        return {
            "shot_duration_mean": float(np.mean(durations)),
            "shot_duration_median": float(np.median(durations)),
            "shot_duration_min": float(np.min(durations)),
            "shot_duration_max": float(np.max(durations)),
            "shot_duration_std": float(np.std(durations)),
            "shot_duration_entropy": float(entropy(np.histogram(durations, bins=20)[0])),
            "cuts_per_10s": float(len(self.shot_boundaries) / (self.total_frames / self.fps / 10)),
            "cuts_variance": float(np.var(durations)),
            "longest_shot_duration": float(np.max(durations)),
            "shortest_shot_duration": float(np.min(durations))
        }

    def extract_pace_curve(self) -> Dict:
        durations = np.diff([0] + self.shot_boundaries + [self.total_frames])
        curve_slope = np.polyfit(np.arange(len(durations)), durations, 1)[0]
        peaks = ((durations[1:-1] > durations[:-2]) & (durations[1:-1] > durations[2:])).sum()
        autocorr = np.correlate(durations - np.mean(durations), durations - np.mean(durations), mode="full")
        autocorr /= autocorr.max()
        period = np.argmax(autocorr[len(autocorr)//2+1:]) + 1
        return {
            "pace_curve_mean": float(np.mean(durations)),
            "pace_curve_slope": float(curve_slope),
            "pace_curve_peaks": int(peaks),
            "pace_curve_periodicity": int(period)
        }

    def extract_scene_pacing(self) -> Dict:
        durations = np.diff([0] + self.scene_boundaries + [self.total_frames])
        return {
            "scene_changes_per_minute": float(len(self.scene_boundaries) / ((self.total_frames/self.fps)/60)),
            "average_scene_duration": float(np.mean(durations)),
            "scene_duration_variance": float(np.var(durations))
        }

    # -------------------------
    # Motion / Optical Flow
    # -------------------------
    def extract_motion_features(self) -> Dict:
        flow_mags = []
        dir_changes = []
        prev_gray = cv2.cvtColor(self._get_resize_frame(0), cv2.COLOR_RGB2GRAY)
        for idx in self.frame_indices[1:]:
            frame = self._get_resize_frame(idx)
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            flow = cv2.calcOpticalFlowFarneback(prev_gray, gray, None,
                                                pyr_scale=0.5, levels=3, winsize=15,
                                                iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            flow_mags.append(np.mean(mag))
            dir_changes.append(np.std(ang))
            prev_gray = gray
        flow_mags = np.array(flow_mags)
        dir_changes = np.array(dir_changes)
        return {
            "mean_motion_speed_per_shot": float(np.mean(flow_mags)),
            "motion_speed_median": float(np.median(flow_mags)),
            "motion_speed_variance": float(np.var(flow_mags)),
            "motion_speed_90perc": float(np.percentile(flow_mags, 90)),
            "share_of_high_motion_frames": float(np.mean(flow_mags > np.percentile(flow_mags, 75))),
            "optical_flow_direction_changes_per_second": float(np.mean(dir_changes)*self.fps)
        }

    def _get_clip_frame(self, idx):
        frame = self.frame_manager.get(idx)
        frame = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_AREA)
        return frame

    # -------------------------
    # CLIP embeddings (batched)
    # -------------------------
    def extract_content_change_rate(self) -> Dict:
        embeddings = []
        for i in range(0, self.total_frames, self.batch_size):
            batch_frames = [self._get_clip_frame(idx) for idx in self.frame_indices[i:i+self.batch_size]]
            batch_tensor = torch.tensor(np.stack(batch_frames)/255.0).permute(0,3,1,2).float().to(device)
            with torch.no_grad():
                emb = self.clip_model.encode_image(batch_tensor.half() if device=="cuda" else batch_tensor)
                embeddings.append(emb.cpu().numpy())
        embeddings = np.vstack(embeddings)
        diff = np.linalg.norm(np.diff(embeddings, axis=0), axis=1)
        diff_smooth = np.convolve(diff, np.ones(5)/5, mode='same')
        return {
            "frame_embedding_diff_mean": float(np.mean(diff_smooth)),
            "frame_embedding_diff_std": float(np.std(diff_smooth)),
            "high_change_frames_ratio": float(np.mean(diff_smooth > np.percentile(diff_smooth, 75))),
            "scene_embedding_jumps": int(np.sum(diff_smooth > 2*np.std(diff_smooth)))
        }

    # -------------------------
    # Color & Lighting Pacing
    # -------------------------
    def extract_color_pacing(self) -> Dict:
        hist_diffs = []
        prev_frame = self._get_resize_frame(0)
        for idx in self.frame_indices[1:]:
            frame = self._get_resize_frame(idx)
            lab1 = rgb2lab(prev_frame)
            lab2 = rgb2lab(frame)
            deltaE = np.sqrt(np.sum((lab1-lab2)**2, axis=2))
            hist_diffs.append(np.mean(deltaE))
            prev_frame = frame
        hist_diffs = np.array(hist_diffs)
        saturation = [np.mean(cv2.cvtColor(self._get_resize_frame(idx), cv2.COLOR_RGB2HSV)[:,:,1]) for idx in self.frame_indices]
        brightness = [np.mean(cv2.cvtColor(self._get_resize_frame(idx), cv2.COLOR_RGB2HSV)[:,:,2]) for idx in self.frame_indices]
        return {
            "color_histogram_diff_mean": float(np.mean(hist_diffs)),
            "color_histogram_diff_std": float(np.std(hist_diffs)),
            "saturation_change_rate": float(np.std(saturation)),
            "brightness_change_rate": float(np.std(brightness))
        }

    def extract_lighting_pacing(self) -> Dict:
        lum = [np.mean(cv2.cvtColor(self._get_resize_frame(idx), cv2.COLOR_RGB2GRAY)) for idx in self.frame_indices]
        lum_diff = np.diff(lum)
        lum_fft = np.fft.fft(lum_diff)
        hf_ratio = np.sum(np.abs(lum_fft[len(lum_fft)//4:len(lum_fft)//2])) / (np.sum(np.abs(lum_fft))+1e-9)
        return {
            "luminance_spikes_per_minute": float(np.sum(np.abs(lum_diff) > np.std(lum_diff)) / (len(self.frame_indices)/self.fps*60)),
            "high_frequency_flash_ratio": float(hf_ratio)
        }

    # -------------------------
    # Structural Pacing
    # -------------------------
    def extract_structural_pacing(self) -> Dict:
        durations = np.diff([0] + self.shot_boundaries + [self.total_frames])
        n = len(durations)
        quarter = max(n//4,1)
        return {
            "intro_speed": float(np.median(durations[:quarter])),
            "main_speed": float(np.median(durations[quarter:3*quarter])),
            "climax_speed": float(np.median(durations[3*quarter:])),
            "pacing_symmetry": float(np.mean(np.diff(durations)))
        }

    # -------------------------
    # Full Pipeline
    # -------------------------
    def extract_all_features(self) -> Dict:
        features = {}
        # Основные визуальные метрики
        features.update(self.extract_shot_features())
        features.update(self.extract_pace_curve())
        features.update(self.extract_scene_pacing())
        features.update(self.extract_motion_features())
        features.update(self.extract_content_change_rate())
        features.update(self.extract_color_pacing())
        features.update(self.extract_lighting_pacing())
        features.update(self.extract_structural_pacing())

        # TODO: сюда позже подключаем AV sync, per-person и object pacing
        # чтобы main.py мог передавать JSON словари или аудио путь

        return features