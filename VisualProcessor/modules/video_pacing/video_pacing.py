"""
video_pacing_visual_optimized.py
Оптимизированный pipeline для извлечения визуальных метрик темпа видео.
- Shot/Cut pacing
- Motion pacing (optical flow)
- Content pacing (CLIP embeddings)
- Color & lighting pacing
- Structural pacing
- Audio-visual pacing sync (получает аудио данные на вход)
- Per-person motion pace (получает данные о треках людей)
- Object change pacing (получает данные о детекции объектов)
Улучшения:
- SSIM для более точной детекции шотов
- Дополнительные статистики motion и content change
- deltaE и FFT для визуальных изменений цвета/яркости
- Нормализация по FPS
"""

import cv2
import numpy as np
import torch
from scipy.stats import entropy
from skimage.metrics import structural_similarity as ssim
from skimage.color import rgb2lab
from typing import List, Dict, Optional, Union
import clip
from cv2 import calcOpticalFlowFarneback
from scenedetect import VideoManager, SceneManager
from scenedetect.detectors import ContentDetector
import os
from scipy import signal
from scipy.stats import pearsonr

device = "cuda" if torch.cuda.is_available() else "cpu"

class VideoPacingPipelineVisualOptimized:
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        self.clip_model, _ = clip.load("ViT-B/32", device=device)
        self.frames = self._load_frames()
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
        self.num_frames = len(self.frames)

        # Scene / Shot segmentation
        self.shot_boundaries = self._detect_shots()
        self.scene_boundaries = self._detect_scenes()

    def _load_frames(self) -> List[np.ndarray]:
        frames = []
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        self.cap.release()
        return frames

    # -------------------------
    # Shot Detection with SSIM refinement
    # -------------------------
    def _detect_shots(self) -> List[int]:
        video_manager = VideoManager([self.video_path])
        scene_manager = SceneManager()
        scene_manager.add_detector(ContentDetector(threshold=27.0))  # default threshold
        video_manager.set_downscale_factor(1)
        video_manager.start()
        scene_manager.detect_scenes(frame_source=video_manager)
        shot_list = [start.get_frames() for start, end in scene_manager.get_scene_list()]
        video_manager.release()

        # Refine using SSIM for hard cuts
        refined_shots = []
        prev_frame = self.frames[0]
        for f in shot_list:
            frame_idx = f
            curr_frame = self.frames[min(frame_idx, self.num_frames-1)]
            score = ssim(prev_frame, curr_frame, multichannel=True)
            if score < 0.95:  # threshold for hard cut
                refined_shots.append(frame_idx)
            prev_frame = curr_frame
        return refined_shots

    def _detect_scenes(self) -> List[int]:
        # For now, scene boundaries = shot boundaries
        return self.shot_boundaries

    # -------------------------
    # Shot Features
    # -------------------------
    def extract_shot_features(self) -> Dict:
        durations = np.diff([0] + self.shot_boundaries + [self.num_frames])
        return {
            "shot_duration_mean": np.mean(durations),
            "shot_duration_median": np.median(durations),
            "shot_duration_min": np.min(durations),
            "shot_duration_max": np.max(durations),
            "shot_duration_std": np.std(durations),
            "shot_duration_entropy": entropy(np.histogram(durations, bins=20)[0]),
            "cuts_per_10s": len(self.shot_boundaries)/(self.num_frames/self.fps/10),
            "cuts_variance": np.var(durations),
            "longest_shot_duration": np.max(durations),
            "shortest_shot_duration": np.min(durations)
        }

    def extract_pace_curve(self) -> Dict:
        durations = np.diff([0] + self.shot_boundaries + [self.num_frames])
        curve_slope = np.polyfit(np.arange(len(durations)), durations, 1)[0]
        peaks = ((durations[1:-1] > durations[:-2]) & (durations[1:-1] > durations[2:])).sum()
        autocorr = np.correlate(durations - np.mean(durations), durations - np.mean(durations), mode="full")
        autocorr /= autocorr.max()
        period = np.argmax(autocorr[len(autocorr)//2+1:]) + 1
        return {
            "pace_curve_mean": np.mean(durations),
            "pace_curve_slope": curve_slope,
            "pace_curve_peaks": peaks,
            "pace_curve_periodicity": period
        }

    def extract_scene_pacing(self) -> Dict:
        durations = np.diff([0] + self.scene_boundaries + [self.num_frames])
        return {
            "scene_changes_per_minute": len(self.scene_boundaries)/((self.num_frames/self.fps)/60),
            "average_scene_duration": np.mean(durations),
            "scene_duration_variance": np.var(durations)
        }

    # -------------------------
    # Motion / Optical Flow
    # -------------------------
    def extract_motion_features(self) -> Dict:
        flow_mags = []
        dir_changes = []
        prev_gray = cv2.cvtColor(self.frames[0], cv2.COLOR_RGB2GRAY)
        for frame in self.frames[1:]:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            flow = calcOpticalFlowFarneback(prev_gray, gray, None,
                                            pyr_scale=0.5, levels=3, winsize=15,
                                            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
            mag, ang = cv2.cartToPolar(flow[...,0], flow[...,1])
            flow_mags.append(np.mean(mag))
            dir_changes.append(np.std(ang))
            prev_gray = gray
        flow_mags = np.array(flow_mags)
        dir_changes = np.array(dir_changes)
        return {
            "mean_motion_speed_per_shot": np.mean(flow_mags),
            "motion_speed_median": np.median(flow_mags),
            "motion_speed_variance": np.var(flow_mags),
            "motion_speed_90perc": np.percentile(flow_mags, 90),
            "share_of_high_motion_frames": np.mean(flow_mags > np.percentile(flow_mags, 75)),
            "optical_flow_direction_changes_per_second": np.mean(dir_changes)*self.fps
        }

    # -------------------------
    # Content Change Rate (CLIP embeddings)
    # -------------------------
    def extract_content_change_rate(self) -> Dict:
        embeddings = []
        for frame in self.frames:
            img_tensor = torch.tensor(frame/255.0).permute(2,0,1).unsqueeze(0).float().to(device)
            with torch.no_grad():
                emb = self.clip_model.encode_image(img_tensor)
                embeddings.append(emb.cpu().numpy())
        embeddings = np.vstack(embeddings)
        diff = np.linalg.norm(np.diff(embeddings, axis=0), axis=1)
        diff_smooth = np.convolve(diff, np.ones(5)/5, mode='same')
        return {
            "frame_embedding_diff_mean": np.mean(diff_smooth),
            "frame_embedding_diff_std": np.std(diff_smooth),
            "high_change_frames_ratio": np.mean(diff_smooth > np.percentile(diff_smooth, 75)),
            "scene_embedding_jumps": np.sum(diff_smooth > 2*np.std(diff_smooth))
        }

    # -------------------------
    # Color & Lighting Pacing
    # -------------------------
    def extract_color_pacing(self) -> Dict:
        hist_diffs = []
        prev_frame = self.frames[0]
        for frame in self.frames[1:]:
            # Lab deltaE
            lab1 = rgb2lab(prev_frame)
            lab2 = rgb2lab(frame)
            deltaE = np.sqrt(np.sum((lab1-lab2)**2, axis=2))
            hist_diffs.append(np.mean(deltaE))
            prev_frame = frame
        hist_diffs = np.array(hist_diffs)
        saturation = [np.mean(cv2.cvtColor(f, cv2.COLOR_RGB2HSV)[:,:,1]) for f in self.frames]
        brightness = [np.mean(cv2.cvtColor(f, cv2.COLOR_RGB2HSV)[:,:,2]) for f in self.frames]
        return {
            "color_histogram_diff_mean": np.mean(hist_diffs),
            "color_histogram_diff_std": np.std(hist_diffs),
            "saturation_change_rate": np.std(saturation),
            "brightness_change_rate": np.std(brightness)
        }

    def extract_lighting_pacing(self) -> Dict:
        lum = [np.mean(cv2.cvtColor(f, cv2.COLOR_RGB2GRAY)) for f in self.frames]
        lum_diff = np.diff(lum)
        # FFT-based high-frequency component
        lum_fft = np.fft.fft(lum_diff)
        hf_ratio = np.sum(np.abs(lum_fft[len(lum_fft)//4:len(lum_fft)//2])) / np.sum(np.abs(lum_fft))
        return {
            "luminance_spikes_per_minute": np.sum(np.abs(lum_diff) > np.std(lum_diff)) / (len(lum)/self.fps*60),
            "high_frequency_flash_ratio": hf_ratio
        }

    # -------------------------
    # Structural Pacing
    # -------------------------
    def extract_structural_pacing(self) -> Dict:
        durations = np.diff([0] + self.shot_boundaries + [self.num_frames])
        n = len(durations)
        quarter = max(n//4, 1)
        return {
            "intro_speed": np.median(durations[:quarter]),
            "main_speed": np.median(durations[quarter:3*quarter]),
            "climax_speed": np.median(durations[3*quarter:]),
            "pacing_symmetry": np.mean(np.diff(durations))
        }

    # -------------------------
    # Audio-Visual Pacing Sync
    # -------------------------
    def extract_audio_visual_sync(self, audio_data: Optional[Union[str, Dict]] = None) -> Dict:
        """
        Извлекает метрики синхронизации аудио и визуального темпа.
        
        Args:
            audio_data: Путь к аудио файлу (str) или словарь с предобработанными данными:
                {
                    'energy_curve': np.ndarray,  # Энергия аудио по времени (per frame)
                    'beats': List[float],  # Временные метки битов в секундах
                    'tempo': float  # BPM (опционально)
                }
        
        Returns:
            Dict с метриками синхронизации
        """
        if audio_data is None:
            return {
                "av_sync_score": 0.0,
                "av_energy_alignment": 0.0,
                "beats_per_cut_ratio": 0.0
            }
        
        # Загружаем или используем предобработанные аудио данные
        if isinstance(audio_data, str):
            # Если передан путь к файлу, пытаемся загрузить через librosa (если доступна)
            try:
                import librosa
                y, sr = librosa.load(audio_data, sr=None)
                # Вычисляем энергию аудио
                hop_length = int(sr / self.fps)  # Синхронизация с FPS видео
                rms = librosa.feature.rms(y=y, hop_length=hop_length)[0]
                # Нормализуем до количества кадров
                frame_times = np.linspace(0, len(self.frames) / self.fps, len(rms))
                target_times = np.linspace(0, len(self.frames) / self.fps, len(self.frames))
                energy_curve = np.interp(target_times, frame_times, rms)
                
                # Детекция битов
                tempo, beats = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
                beat_times = librosa.frames_to_time(beats, sr=sr, hop_length=hop_length)
            except ImportError:
                # Если librosa недоступна, возвращаем нули
                return {
                    "av_sync_score": 0.0,
                    "av_energy_alignment": 0.0,
                    "beats_per_cut_ratio": 0.0
                }
        else:
            # Используем предобработанные данные
            energy_curve = audio_data.get('energy_curve', np.zeros(self.num_frames))
            beat_times = audio_data.get('beats', [])
            tempo = audio_data.get('tempo', 0.0)
        
        # Получаем визуальную динамику (optical flow magnitude)
        flow_mags = []
        prev_gray = cv2.cvtColor(self.frames[0], cv2.COLOR_RGB2GRAY)
        for frame in self.frames[1:]:
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            flow = calcOpticalFlowFarneback(prev_gray, gray, None,
                                            pyr_scale=0.5, levels=3, winsize=15,
                                            iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
            mag, _ = cv2.cartToPolar(flow[...,0], flow[...,1])
            flow_mags.append(np.mean(mag))
            prev_gray = gray
        flow_mags = np.array([flow_mags[0]] + flow_mags)  # Добавляем первый кадр
        
        # Нормализуем массивы до одинаковой длины
        min_len = min(len(energy_curve), len(flow_mags))
        energy_curve = energy_curve[:min_len]
        flow_mags = flow_mags[:min_len]
        
        # Нормализуем для корреляции
        energy_norm = (energy_curve - np.mean(energy_curve)) / (np.std(energy_curve) + 1e-9)
        flow_norm = (flow_mags - np.mean(flow_mags)) / (np.std(flow_mags) + 1e-9)
        
        # Вычисляем корреляцию (AV sync score)
        if len(energy_norm) > 1 and len(flow_norm) > 1:
            correlation, _ = pearsonr(energy_norm, flow_norm)
            av_sync_score = max(0.0, min(1.0, (correlation + 1) / 2))  # Нормализуем к [0, 1]
        else:
            av_sync_score = 0.0
        
        # AV energy alignment (скользящая корреляция)
        window_size = min(30, len(energy_norm) // 4)  # Окно ~1 секунда при 30 FPS
        if window_size > 1:
            alignments = []
            for i in range(len(energy_norm) - window_size + 1):
                e_win = energy_norm[i:i+window_size]
                f_win = flow_norm[i:i+window_size]
                if len(e_win) > 1 and len(f_win) > 1:
                    corr, _ = pearsonr(e_win, f_win)
                    alignments.append(max(0.0, (corr + 1) / 2))
            av_energy_alignment = np.mean(alignments) if alignments else 0.0
        else:
            av_energy_alignment = av_sync_score
        
        # Beats per cut ratio
        if beat_times and self.shot_boundaries:
            cut_times = [idx / self.fps for idx in self.shot_boundaries]
            # Считаем сколько битов близко к катам (окно 0.2 секунды)
            beats_near_cuts = 0
            for beat_time in beat_times:
                for cut_time in cut_times:
                    if abs(beat_time - cut_time) < 0.2:
                        beats_near_cuts += 1
                        break
            beats_per_cut_ratio = beats_near_cuts / len(self.shot_boundaries) if self.shot_boundaries else 0.0
        else:
            beats_per_cut_ratio = 0.0
        
        return {
            "av_sync_score": float(av_sync_score),
            "av_energy_alignment": float(av_energy_alignment),
            "beats_per_cut_ratio": float(beats_per_cut_ratio)
        }

    # -------------------------
    # Per-Person Motion Pace
    # -------------------------
    def extract_per_person_motion_pace(self, person_tracks: Optional[Dict[int, List[int]]] = None,
                                       person_keypoints: Optional[Dict[int, Dict[int, np.ndarray]]] = None) -> Dict:
        """
        Извлекает метрики темпа движения для каждого человека.
        
        Args:
            person_tracks: Словарь {person_id: [frame_indices]} - треки людей по кадрам
            person_keypoints: Словарь {person_id: {frame_idx: keypoints_array}} - ключевые точки для каждого человека
        
        Returns:
            Dict с метриками темпа движения людей
        """
        if person_tracks is None or len(person_tracks) == 0:
            return {
                "avg_pose_speed": 0.0,
                "pose_speed_variance": 0.0,
                "bursts_of_activity_per_minute": 0.0,
                "freeze_moments_count": 0,
                "per_person_motion_pace": {}
            }
        
        all_pose_speeds = []
        per_person_features = {}
        total_bursts = 0
        total_freezes = 0
        
        for person_id, frame_indices in person_tracks.items():
            if len(frame_indices) < 2:
                continue
            
            # Вычисляем скорость движения для этого человека
            pose_speeds = []
            
            if person_keypoints and person_id in person_keypoints:
                # Используем ключевые точки для более точного расчета
                keypoints_dict = person_keypoints[person_id]
                sorted_frames = sorted([f for f in frame_indices if f in keypoints_dict])
                
                for i in range(1, len(sorted_frames)):
                    prev_frame = sorted_frames[i-1]
                    curr_frame = sorted_frames[i]
                    
                    if prev_frame in keypoints_dict and curr_frame in keypoints_dict:
                        kp_prev = keypoints_dict[prev_frame]
                        kp_curr = keypoints_dict[curr_frame]
                        
                        # Вычисляем среднее смещение ключевых точек
                        if len(kp_prev.shape) >= 2 and len(kp_curr.shape) >= 2:
                            # Предполагаем формат [N, 2] или [N, 3] (x, y, confidence)
                            kp_prev_2d = kp_prev[:, :2] if kp_prev.shape[1] >= 2 else kp_prev
                            kp_curr_2d = kp_curr[:, :2] if kp_curr.shape[1] >= 2 else kp_curr
                            
                            if len(kp_prev_2d) == len(kp_curr_2d):
                                displacement = np.mean(np.linalg.norm(kp_curr_2d - kp_prev_2d, axis=1))
                                # Нормализуем по FPS
                                time_diff = (curr_frame - prev_frame) / self.fps
                                speed = displacement / (time_diff + 1e-9)
                                pose_speeds.append(speed)
            else:
                # Используем optical flow в области человека (упрощенный подход)
                # Для этого нужно было бы иметь bbox, но если их нет, используем общий flow
                sorted_frames = sorted(frame_indices)
                flow_mags = []
                prev_gray = cv2.cvtColor(self.frames[sorted_frames[0]], cv2.COLOR_RGB2GRAY)
                
                for frame_idx in sorted_frames[1:]:
                    if frame_idx >= len(self.frames):
                        continue
                    gray = cv2.cvtColor(self.frames[frame_idx], cv2.COLOR_RGB2GRAY)
                    flow = calcOpticalFlowFarneback(prev_gray, gray, None,
                                                    pyr_scale=0.5, levels=3, winsize=15,
                                                    iterations=3, poly_n=5, poly_sigma=1.2, flags=0)
                    mag, _ = cv2.cartToPolar(flow[...,0], flow[...,1])
                    flow_mags.append(np.mean(mag))
                    prev_gray = gray
                
                if flow_mags:
                    pose_speeds = flow_mags
            
            if not pose_speeds:
                continue
            
            pose_speeds = np.array(pose_speeds)
            all_pose_speeds.extend(pose_speeds)
            
            # Вычисляем метрики для этого человека
            avg_speed = np.mean(pose_speeds)
            speed_variance = np.var(pose_speeds)
            
            # Bursts of activity (резкие всплески скорости)
            speed_threshold = np.percentile(pose_speeds, 75)
            bursts = np.sum(pose_speeds > speed_threshold)
            # Нормализуем на минуту
            duration_minutes = len(frame_indices) / (self.fps * 60)
            bursts_per_min = bursts / (duration_minutes + 1e-9)
            
            # Freeze moments (моменты без движения)
            freeze_threshold = np.percentile(pose_speeds, 25)
            freezes = np.sum(pose_speeds < freeze_threshold)
            
            per_person_features[person_id] = {
                "avg_pose_speed": float(avg_speed),
                "pose_speed_variance": float(speed_variance),
                "bursts_of_activity_per_minute": float(bursts_per_min),
                "freeze_moments_count": int(freezes)
            }
            
            total_bursts += bursts
            total_freezes += freezes
        
        # Агрегированные метрики
        if all_pose_speeds:
            all_pose_speeds = np.array(all_pose_speeds)
            avg_pose_speed = np.mean(all_pose_speeds)
            pose_speed_variance = np.var(all_pose_speeds)
            
            # Общее количество всплесков активности на минуту
            total_duration_minutes = self.num_frames / (self.fps * 60)
            bursts_per_minute = total_bursts / (total_duration_minutes + 1e-9)
        else:
            avg_pose_speed = 0.0
            pose_speed_variance = 0.0
            bursts_per_minute = 0.0
        
        return {
            "avg_pose_speed": float(avg_pose_speed),
            "pose_speed_variance": float(pose_speed_variance),
            "bursts_of_activity_per_minute": float(bursts_per_minute),
            "freeze_moments_count": int(total_freezes),
            "per_person_motion_pace": per_person_features
        }

    # -------------------------
    # Object Change Pacing
    # -------------------------
    def extract_object_change_pacing(self, object_detections: Optional[Dict[int, List[Dict]]] = None) -> Dict:
        """
        Извлекает метрики темпа изменения объектов.
        
        Args:
            object_detections: Словарь {frame_idx: [detections]} где каждое detection:
                {
                    'label': str,
                    'bbox': [x1, y1, x2, y2],
                    'score': float,
                    'track_id': int (опционально)
                }
        
        Returns:
            Dict с метриками темпа изменения объектов
        """
        if object_detections is None or len(object_detections) == 0:
            return {
                "new_objects_per_10s": 0.0,
                "object_entry_exit_rate": 0.0,
                "main_object_switching_rate": 0.0
            }
        
        # Сортируем кадры
        sorted_frames = sorted(object_detections.keys())
        if len(sorted_frames) < 2:
            return {
                "new_objects_per_10s": 0.0,
                "object_entry_exit_rate": 0.0,
                "main_object_switching_rate": 0.0
            }
        
        # Трекинг объектов по track_id или label+bbox overlap
        object_tracks = {}  # {track_id: [frame_indices]}
        track_id_counter = 0
        
        # Простой трекинг на основе IoU и label
        for frame_idx in sorted_frames:
            detections = object_detections[frame_idx]
            
            for det in detections:
                track_id = det.get('track_id', None)
                label = det.get('label', 'unknown')
                bbox = det.get('bbox', [0, 0, 0, 0])
                
                if track_id is None:
                    # Пытаемся найти существующий трек по IoU
                    matched_track = None
                    if frame_idx > sorted_frames[0]:
                        prev_frame_idx = sorted_frames[sorted_frames.index(frame_idx) - 1]
                        if prev_frame_idx in object_detections:
                            prev_detections = object_detections[prev_frame_idx]
                            best_iou = 0.3  # Порог для матчинга
                            
                            for prev_det in prev_detections:
                                if prev_det.get('label') == label:
                                    prev_bbox = prev_det.get('bbox', [0, 0, 0, 0])
                                    iou = self._calculate_iou(bbox, prev_bbox)
                                    if iou > best_iou:
                                        best_iou = iou
                                        matched_track = prev_det.get('track_id')
                    
                    if matched_track is not None:
                        track_id = matched_track
                    else:
                        track_id = track_id_counter
                        track_id_counter += 1
                
                if track_id not in object_tracks:
                    object_tracks[track_id] = []
                object_tracks[track_id].append(frame_idx)
        
        # Вычисляем метрики
        # 1. New objects per 10s
        new_objects = 0
        for track_id, frame_indices in object_tracks.items():
            if len(frame_indices) > 0:
                first_frame = min(frame_indices)
                # Считаем объект новым, если он появился в этом окне
                new_objects += 1  # Упрощенно: каждый трек = новый объект
        
        duration_10s_units = (self.num_frames / self.fps) / 10.0
        new_objects_per_10s = new_objects / (duration_10s_units + 1e-9)
        
        # 2. Object entry/exit rate
        entries = 0
        exits = 0
        for track_id, frame_indices in object_tracks.items():
            if len(frame_indices) > 0:
                first_frame = min(frame_indices)
                last_frame = max(frame_indices)
                
                # Entry: объект появился не в первом кадре
                if first_frame > sorted_frames[0]:
                    entries += 1
                
                # Exit: объект исчез не в последнем кадре
                if last_frame < sorted_frames[-1]:
                    exits += 1
        
        total_entries_exits = entries + exits
        duration_minutes = (self.num_frames / self.fps) / 60.0
        object_entry_exit_rate = total_entries_exits / (duration_minutes + 1e-9)
        
        # 3. Main object switching rate (смена главного объекта)
        # Главный объект = объект с наибольшей площадью bbox или наибольшим score
        main_object_per_frame = []
        for frame_idx in sorted_frames:
            if frame_idx not in object_detections:
                continue
            detections = object_detections[frame_idx]
            if not detections:
                continue
            
            # Находим главный объект (наибольшая площадь или score)
            main_det = max(detections, key=lambda d: (
                (d.get('bbox', [0,0,0,0])[2] - d.get('bbox', [0,0,0,0])[0]) *
                (d.get('bbox', [0,0,0,0])[3] - d.get('bbox', [0,0,0,0])[1]) *
                d.get('score', 0.0)
            ))
            main_track_id = main_det.get('track_id', None)
            if main_track_id is None:
                # Пытаемся найти по label и bbox
                for tid, frames in object_tracks.items():
                    if frame_idx in frames:
                        main_track_id = tid
                        break
            main_object_per_frame.append(main_track_id)
        
        # Считаем переключения главного объекта
        switches = 0
        for i in range(1, len(main_object_per_frame)):
            if main_object_per_frame[i] != main_object_per_frame[i-1]:
                switches += 1
        
        main_object_switching_rate = switches / (duration_minutes + 1e-9)
        
        return {
            "new_objects_per_10s": float(new_objects_per_10s),
            "object_entry_exit_rate": float(object_entry_exit_rate),
            "main_object_switching_rate": float(main_object_switching_rate)
        }
    
    def _calculate_iou(self, bbox1: List[float], bbox2: List[float]) -> float:
        """Вычисляет IoU между двумя bbox."""
        if len(bbox1) < 4 or len(bbox2) < 4:
            return 0.0
        
        x1_1, y1_1, x2_1, y2_1 = bbox1[0], bbox1[1], bbox1[2], bbox1[3]
        x1_2, y1_2, x2_2, y2_2 = bbox2[0], bbox2[1], bbox2[2], bbox2[3]
        
        # Вычисляем пересечение
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Вычисляем объединение
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union

    # -------------------------
    # Full Pipeline
    # -------------------------
    def extract_all_features(self, audio_data: Optional[Union[str, Dict]] = None,
                           person_tracks: Optional[Dict[int, List[int]]] = None,
                           person_keypoints: Optional[Dict[int, Dict[int, np.ndarray]]] = None,
                           object_detections: Optional[Dict[int, List[Dict]]] = None) -> Dict:
        """
        Извлекает все метрики темпа видео.
        
        Args:
            audio_data: Путь к аудио файлу или словарь с предобработанными данными
            person_tracks: Словарь {person_id: [frame_indices]} - треки людей
            person_keypoints: Словарь {person_id: {frame_idx: keypoints}} - ключевые точки людей
            object_detections: Словарь {frame_idx: [detections]} - детекции объектов
        
        Returns:
            Dict со всеми метриками темпа
        """
        features = {}
        features.update(self.extract_shot_features())
        features.update(self.extract_pace_curve())
        features.update(self.extract_scene_pacing())
        features.update(self.extract_motion_features())
        features.update(self.extract_content_change_rate())
        features.update(self.extract_color_pacing())
        features.update(self.extract_lighting_pacing())
        features.update(self.extract_structural_pacing())
        
        # Новые фичи
        features.update(self.extract_audio_visual_sync(audio_data))
        features.update(self.extract_per_person_motion_pace(person_tracks, person_keypoints))
        features.update(self.extract_object_change_pacing(object_detections))
        
        return features

if __name__ == "__main__":
    import argparse
    import json
    
    parser = argparse.ArgumentParser(
        description='Video Pacing Module - Extracts pacing metrics from video (visual + optional audio/person/object data)'
    )
    parser.add_argument('--video', type=str, required=True, help='Input video file path')
    parser.add_argument('--audio', type=str, default=None, 
                       help='Optional audio file path for audio-visual sync analysis')
    parser.add_argument('--person-tracks', type=str, default=None,
                       help='Optional JSON file with person tracks: {"person_id": [frame_indices]}')
    parser.add_argument('--person-keypoints', type=str, default=None,
                       help='Optional JSON file with person keypoints (complex structure)')
    parser.add_argument('--object-detections', type=str, default=None,
                       help='Optional JSON file with object detections: {"frame_idx": [{"label": str, "bbox": [...], "score": float}]}')
    parser.add_argument('--output', type=str, default=None,
                       help='Output JSON file path (if not specified, prints to stdout)')
    parser.add_argument('--summary', action='store_true',
                       help='Print summary statistics instead of all features')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"Error: Video file not found: {args.video}")
        exit(1)
    
    print(f"Loading video: {args.video}")
    print(f"Using device: {device}")
    
    # Инициализация pipeline
    try:
        pipeline = VideoPacingPipelineVisualOptimized(args.video)
        print(f"Loaded {pipeline.num_frames} frames at {pipeline.fps} FPS")
    except Exception as e:
        print(f"Error initializing pipeline: {e}")
        exit(1)
    
    # Загрузка опциональных данных
    audio_data = None
    if args.audio:
        if os.path.exists(args.audio):
            audio_data = args.audio
            print(f"Audio file provided: {args.audio}")
        else:
            print(f"Warning: Audio file not found: {args.audio}, continuing without audio sync")
    
    person_tracks = None
    if args.person_tracks:
        if os.path.exists(args.person_tracks):
            with open(args.person_tracks, 'r') as f:
                person_tracks = json.load(f)
                # Конвертируем ключи в int
                person_tracks = {int(k): v for k, v in person_tracks.items()}
            print(f"Loaded person tracks: {len(person_tracks)} persons")
        else:
            print(f"Warning: Person tracks file not found: {args.person_tracks}")
    
    person_keypoints = None
    if args.person_keypoints:
        if os.path.exists(args.person_keypoints):
            with open(args.person_keypoints, 'r') as f:
                person_keypoints = json.load(f)
                # Конвертируем структуру (может быть сложной)
                # Предполагаем формат: {"person_id": {"frame_idx": [[x, y], ...]}}
                converted = {}
                for pid_str, frames_dict in person_keypoints.items():
                    pid = int(pid_str)
                    converted[pid] = {}
                    for fid_str, kp_list in frames_dict.items():
                        fid = int(fid_str)
                        converted[pid][fid] = np.array(kp_list)
                person_keypoints = converted
            print(f"Loaded person keypoints")
        else:
            print(f"Warning: Person keypoints file not found: {args.person_keypoints}")
    
    object_detections = None
    if args.object_detections:
        if os.path.exists(args.object_detections):
            with open(args.object_detections, 'r') as f:
                object_detections = json.load(f)
                # Конвертируем ключи frame_idx в int
                object_detections = {int(k): v for k, v in object_detections.items()}
            print(f"Loaded object detections: {len(object_detections)} frames")
        else:
            print(f"Warning: Object detections file not found: {args.object_detections}")
    
    # Обработка
    print("\nProcessing video...")
    try:
        features = pipeline.extract_all_features(
            audio_data=audio_data,
            person_tracks=person_tracks,
            person_keypoints=person_keypoints,
            object_detections=object_detections
        )
    except Exception as e:
        print(f"Error processing video: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
    
    # Вывод результатов
    if args.summary:
        print("\n=== Video Pacing Summary ===")
        print(f"\nShot/Cut Features:")
        print(f"  Cuts per 10s: {features.get('cuts_per_10s', 0):.2f}")
        print(f"  Average shot duration: {features.get('shot_duration_mean', 0):.2f} frames")
        print(f"  Shot duration variance: {features.get('shot_duration_variance', 0):.2f}")
        
        print(f"\nMotion Features:")
        print(f"  Mean motion speed: {features.get('mean_motion_speed_per_shot', 0):.4f}")
        print(f"  High motion frames: {features.get('share_of_high_motion_frames', 0):.2%}")
        
        print(f"\nContent Change:")
        print(f"  Frame embedding diff mean: {features.get('frame_embedding_diff_mean', 0):.4f}")
        print(f"  High change frames ratio: {features.get('high_change_frames_ratio', 0):.2%}")
        
        print(f"\nColor & Lighting:")
        print(f"  Color histogram diff mean: {features.get('color_histogram_diff_mean', 0):.4f}")
        print(f"  Brightness change rate: {features.get('brightness_change_rate', 0):.4f}")
        
        if audio_data:
            print(f"\nAudio-Visual Sync:")
            print(f"  AV sync score: {features.get('av_sync_score', 0):.3f}")
            print(f"  AV energy alignment: {features.get('av_energy_alignment', 0):.3f}")
            print(f"  Beats per cut ratio: {features.get('beats_per_cut_ratio', 0):.3f}")
        
        if person_tracks:
            print(f"\nPer-Person Motion:")
            print(f"  Avg pose speed: {features.get('avg_pose_speed', 0):.4f}")
            print(f"  Bursts per minute: {features.get('bursts_of_activity_per_minute', 0):.2f}")
            print(f"  Freeze moments: {features.get('freeze_moments_count', 0)}")
        
        if object_detections:
            print(f"\nObject Change Pacing:")
            print(f"  New objects per 10s: {features.get('new_objects_per_10s', 0):.2f}")
            print(f"  Object entry/exit rate: {features.get('object_entry_exit_rate', 0):.2f}")
            print(f"  Main object switching rate: {features.get('main_object_switching_rate', 0):.2f}")
    else:
        print("\n=== All Video Pacing Features ===")
        for k, v in sorted(features.items()):
            if isinstance(v, dict):
                print(f"\n{k}:")
                for sub_k, sub_v in v.items():
                    print(f"  {sub_k}: {sub_v}")
            else:
                print(f"{k}: {v}")
    
    # Сохранение в файл
    if args.output:
        # Конвертируем numpy типы для JSON
        def convert_to_json_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_json_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_json_serializable(item) for item in obj]
            return obj
        
        output_data = {
            'video_path': args.video,
            'fps': pipeline.fps,
            'num_frames': pipeline.num_frames,
            'features': convert_to_json_serializable(features)
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"\nResults saved to: {args.output}")
