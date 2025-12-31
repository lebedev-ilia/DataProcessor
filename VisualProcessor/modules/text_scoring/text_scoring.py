# text_video_pipeline.py

"""
Содержит:
- библиотечный класс `TextVideoInteractionPipeline` (feature extraction given OCR detections)
- production wrapper `TextScoringModule(BaseModule)`:
  consumer OCR-артефакта (NPZ) и выдача NPZ результатов по стандарту.
"""

from __future__ import annotations

import os
import math
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional

from modules.base_module import BaseModule
from utils.frame_manager import FrameManager

class TextVideoInteractionPipeline:
    """
    Пайплайн для извлечения фичей взаимодействия текста и видео.
    Вход: кадры, OCR с bbox, motion/face/audio пики
    Выход: словарь фичей на видео
    """

    def __init__(
        self,
        video_fps: int = 30,
        frame_width: int | None = None,
        frame_height: int | None = None,
        alignment_window_seconds: float = 0.5,
        motion_weight: float = 0.4,
        face_weight: float = 0.3,
        audio_weight: float = 0.3,
        min_ocr_confidence: float = 0.4,
    ):
        self.video_fps = float(video_fps)
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.alignment_window_seconds = float(alignment_window_seconds)
        # нормализуем веса, чтобы сумма была 1.0
        w_sum = motion_weight + face_weight + audio_weight
        if w_sum <= 0:
            self.motion_weight = 1.0
            self.face_weight = 0.0
            self.audio_weight = 0.0
        else:
            self.motion_weight = motion_weight / w_sum
            self.face_weight = face_weight / w_sum
            self.audio_weight = audio_weight / w_sum
        self.min_ocr_confidence = float(min_ocr_confidence)

    @staticmethod
    def _iou(boxA: Tuple[int, int, int, int], boxB: Tuple[int, int, int, int]) -> float:
        """Intersection over union для bbox"""
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])

        iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
        return iou

    @staticmethod
    def _normalize_signal(signal: np.ndarray) -> np.ndarray:
        """Нормализация и сглаживание сигнала для уменьшения шумов"""
        if len(signal) == 0:
            return np.array([])
        signal = np.array(signal, dtype=np.float32)
        signal = gaussian_filter1d(signal, sigma=1)
        max_val = float(signal.max())
        if max_val <= 0:
            return np.zeros_like(signal, dtype=np.float32)
        return signal / max_val

    @staticmethod
    def _zscore(signal: np.ndarray) -> np.ndarray:
        """Перевод сигнала в z-score по видео."""
        if len(signal) == 0:
            return np.array([], dtype=np.float32)
        signal = np.array(signal, dtype=np.float32)
        mean = float(signal.mean())
        std = float(signal.std()) + 1e-6
        return (signal - mean) / std

    @staticmethod
    def _trimmed_mean(values: List[float], proportion_to_cut: float = 0.1) -> float:
        """Робастное среднее: усреднение по центральной части распределения."""
        if not values:
            return 0.0
        arr = np.sort(np.asarray(values, dtype=np.float32))
        n = len(arr)
        k = int(n * proportion_to_cut)
        if k * 2 >= n:
            return float(arr.mean())
        return float(arr[k : n - k].mean())

    @staticmethod
    def _normalize_text(s: str) -> str:
        """Простая нормализация текста: lower + обрезка пробелов и пунктуации по краям."""
        import re

        s = (s or "").lower()
        s = s.strip()
        # Удаляем лишнюю пунктуацию по краям
        s = re.sub(r"^[\W_]+|[\W_]+$", "", s)
        return s

    @staticmethod
    def _normalized_text_similarity(a: str, b: str) -> float:
        """
        Нормализованная похожесть строк (0..1) через Levenshtein-подобное расстояние.
        Реализуем простую динамику, чтобы не тянуть внешние зависимости.
        """
        a = a or ""
        b = b or ""
        if a == b:
            return 1.0
        if not a or not b:
            return 0.0
        la, lb = len(a), len(b)
        # DP по двум строкам
        dp = list(range(lb + 1))
        for i in range(1, la + 1):
            prev = dp[0]
            dp[0] = i
            for j in range(1, lb + 1):
                cur = dp[j]
                cost = 0 if a[i - 1] == b[j - 1] else 1
                dp[j] = min(
                    dp[j] + 1,       # удаление
                    dp[j - 1] + 1,   # вставка
                    prev + cost,     # замена
                )
                prev = cur
        dist = dp[lb]
        max_len = max(la, lb)
        return 1.0 - float(dist) / float(max_len)

    @staticmethod
    def _shannon_entropy(counts: Dict[str, int]) -> float:
        """Энтропия распределения языков."""
        total = sum(counts.values())
        if total <= 0:
            return 0.0
        probs = [c / total for c in counts.values() if c > 0]
        return float(-sum(p * math.log(p + 1e-12) for p in probs))

    def _group_ocr_elements(self, ocr_data: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
        """
        Группировка OCR-детекций в уникальные текстовые элементы по IoU + текстовой похожести.

        Возвращает:
        - raw_detections: отфильтрованный по confidence список исходных детекций (+ time_s при необходимости)
        - unique_elements: список агрегированных элементов с полями:
            - text_raw, text_norm, language
            - frames, times, first_frame, last_frame, first_time, last_time
            - bbox_median, aggregated_confidence
        """
        if not ocr_data:
            return [], []

        # фильтрация по confidence
        raw_filtered = []
        for d in ocr_data:
            conf = float(d.get("confidence", 1.0))
            if conf < self.min_ocr_confidence:
                continue
            # гарантируем наличие time_s
            if "time_s" not in d:
                d = dict(d)
                d["time_s"] = d["frame"] / self.video_fps
            raw_filtered.append(d)

        if not raw_filtered:
            return [], []

        elements: List[Dict[str, Any]] = []
        for det in raw_filtered:
            frame_idx = det["frame"]
            time_s = det.get("time_s", frame_idx / self.video_fps)
            bbox = det["bbox"]
            text_raw = det.get("text_raw", det.get("text", ""))
            text_norm = det.get("text_norm", self._normalize_text(text_raw))
            language = det.get("language", None)

            matched_idx = None
            best_score = 0.0
            for i, elem in enumerate(elements):
                iou = self._iou(bbox, elem["bbox_median"])
                if iou < 0.6:
                    continue
                sim = self._normalized_text_similarity(text_norm, elem["text_norm"])
                score = 0.5 * iou + 0.5 * sim
                if score > 0.8 and score > best_score:
                    best_score = score
                    matched_idx = i

            if matched_idx is None:
                elements.append(
                    {
                        "text_raw": text_raw,
                        "text_norm": text_norm,
                        "language": language,
                        "frames": [frame_idx],
                        "times": [time_s],
                        "bboxes": [bbox],
                        "first_frame": frame_idx,
                        "last_frame": frame_idx,
                        "first_time": time_s,
                        "last_time": time_s,
                        "confidences": [float(det.get("confidence", 1.0))],
                        "is_cta_candidate": bool(det.get("is_cta_candidate", False)),
                    }
                )
            else:
                elem = elements[matched_idx]
                elem["frames"].append(frame_idx)
                elem["times"].append(time_s)
                elem["bboxes"].append(bbox)
                elem["last_frame"] = frame_idx
                elem["last_time"] = time_s
                elem["confidences"].append(float(det.get("confidence", 1.0)))
                elem["is_cta_candidate"] = elem["is_cta_candidate"] or bool(det.get("is_cta_candidate", False))

        # агрегируем bbox и confidence
        for elem in elements:
            xs1, ys1, xs2, ys2 = [], [], [], []
            for (x1, y1, x2, y2) in elem["bboxes"]:
                xs1.append(x1)
                ys1.append(y1)
                xs2.append(x2)
                ys2.append(y2)
            elem["bbox_median"] = (
                float(np.median(xs1)),
                float(np.median(ys1)),
                float(np.median(xs2)),
                float(np.median(ys2)),
            )
            confs = np.asarray(elem["confidences"], dtype=np.float32)
            elem["aggregated_confidence"] = float(confs.mean()) if confs.size else 0.0

        return raw_filtered, elements

    def _compute_text_area_fraction(self, elements: List[Dict[str, Any]]) -> Tuple[float, List[float]]:
        """
        Оценка доли площади кадра, занятой текстом.
        Возвращает:
        - средняя доля площади текста по уникальным элементам
        - список долей по элементам
        """
        if not elements or not self.frame_width or not self.frame_height:
            return 0.0, []
        frame_area = float(self.frame_width * self.frame_height)
        fractions = []
        for elem in elements:
            x1, y1, x2, y2 = elem["bbox_median"]
            w = max(0.0, float(x2 - x1))
            h = max(0.0, float(y2 - y1))
            area = w * h
            fractions.append(float(area / (frame_area + 1e-6)))
        if not fractions:
            return 0.0, []
        arr = np.asarray(fractions, dtype=np.float32)
        return float(arr.mean()), fractions

    @staticmethod
    def _readability_score(text_norm: str) -> float:
        """
        Простейший скор читаемости: короткие, хорошо структурированные CTA/заголовки получают больший скор.
        """
        if not text_norm:
            return 0.0
        import re

        # убираем лишние пробелы
        text = re.sub(r"\s+", " ", text_norm.strip())
        words = text.split(" ")
        num_words = len(words)
        num_chars = len(text)
        num_punct = len(re.findall(r"[^\w\s]", text))
        avg_word_len = num_chars / max(num_words, 1)
        punct_ratio = num_punct / max(num_chars, 1)
        # эвристика: 1.0 для коротких заголовков с малым количеством пунктуации
        score = 1.0
        score *= 1.0 / (1.0 + max(0.0, (num_words - 6) / 10.0))
        score *= 1.0 / (1.0 + max(0.0, (avg_word_len - 6) / 10.0))
        score *= 1.0 / (1.0 + 5.0 * punct_ratio)
        return float(max(0.0, min(1.0, score)))

    def extract_features(
        self,
        ocr_data: List[Dict[str, Any]],
        motion_peaks: List[float],
        face_peaks: List[float],
        audio_peaks: List[float] = None
    ) -> Dict[str, Any]:
        """
        ocr_data: List[Dict] = [
            {"frame": 10, "bbox": (x1,y1,x2,y2), "text": "...", "confidence": 0.95, "is_cta": False},
            ...
        ]
        motion_peaks: List[float] = motion intensity per frame
        face_peaks: List[float] = face detection peaks per frame
        audio_peaks: List[float] = audio energy per frame (optional)
        """
        features: Dict[str, Any] = defaultdict(float)

        # --- Preprocess & group OCR detections ---
        raw_ocr, unique_elements = self._group_ocr_elements(ocr_data)

        num_frames = max([d["frame"] for d in raw_ocr], default=0) + 1
        motion_signal = np.array(motion_peaks[:num_frames])
        face_signal = np.array(face_peaks[:num_frames])
        audio_signal = np.array(audio_peaks[:num_frames]) if audio_peaks is not None else np.zeros(num_frames)

        # нормализованные сигналы (0..1) для alignment
        motion_norm = self._normalize_signal(motion_signal)
        face_norm = self._normalize_signal(face_signal)
        audio_norm = self._normalize_signal(audio_signal)

        # z-score для energy-based оконных метрик
        motion_z = self._zscore(motion_signal)

        # ---------- 1. Text → Action / Motion ----------
        text_action_scores_windowed: List[float] = []
        text_motion_align_scores: List[float] = []
        text_motion_align_scores_windowed: List[float] = []
        multimodal_per_text: List[Tuple[float, float]] = []  # (score, time_s)

        # CTA промежуточные агрегаты
        cta_elements_indices: List[int] = []
        cta_multimodal_scores: List[float] = []

        half_window_frames = int(self.alignment_window_seconds * self.video_fps)
        total_video_seconds = num_frames / self.video_fps if self.video_fps > 0 else 0.0

        for idx, elem in enumerate(unique_elements):
            if not elem["frames"]:
                continue
            center_frame = int(elem["first_frame"])
            center_time = float(elem["first_time"])

            # окно вокруг появления текста
            start = max(0, center_frame - half_window_frames)
            end = min(num_frames, center_frame + half_window_frames + 1)

            # motion z-score в окне
            if end > start and len(motion_z) > 0:
                window_motion = motion_z[start:end]
                # учитываем максимум/среднее в окне
                max_energy = float(np.max(window_motion))
                mean_energy = float(np.mean(window_motion))
                # комбинируем как среднее max и mean
                window_score = 0.5 * max_energy + 0.5 * mean_energy
                text_action_scores_windowed.append(window_score)

            # Alignment score: motion + face + audio в точке
            if center_frame < len(motion_norm):
                m_val = float(motion_norm[center_frame])
                f_val = float(face_norm[center_frame]) if center_frame < len(face_norm) else 0.0
                a_val = float(audio_norm[center_frame]) if center_frame < len(audio_norm) else 0.0
            else:
                m_val = f_val = a_val = 0.0

            multimodal_score = (
                self.motion_weight * m_val
                + self.face_weight * f_val
                + self.audio_weight * a_val
            )
            text_motion_align_scores.append(multimodal_score)

            # окно для alignment
            if end > start:
                m_win = motion_norm[start:end]
                f_win = face_norm[start:end]
                a_win = audio_norm[start:end]
                win_len = len(m_win)
                if win_len > 0:
                    # нормализуем длину массивов
                    f_win = f_win if len(f_win) == win_len else np.resize(f_win, win_len)
                    a_win = a_win if len(a_win) == win_len else np.resize(a_win, win_len)
                    window_multimodal = (
                        self.motion_weight * m_win
                        + self.face_weight * f_win
                        + self.audio_weight * a_win
                    )
                    text_motion_align_scores_windowed.append(float(np.max(window_multimodal)))
                    multimodal_per_text.append((float(np.max(window_multimodal)), center_time))
                else:
                    text_motion_align_scores_windowed.append(0.0)
                    multimodal_per_text.append((multimodal_score, center_time))
            else:
                text_motion_align_scores_windowed.append(multimodal_score)
                multimodal_per_text.append((multimodal_score, center_time))

            # CTA candidate (по флагу из OCR или по тексту)
            text_norm = elem["text_norm"]
            is_cta_flag = bool(elem.get("is_cta_candidate", False))
            is_cta_lexical = False
            if text_norm:
                cta_keywords = [
                    "subscribe",
                    "follow",
                    "like",
                    "link in bio",
                    "click",
                    "watch",
                    "подпишись",
                    "подписаться",
                    "ставь лайк",
                    "ссылка в описании",
                ]
                for kw in cta_keywords:
                    sim = self._normalized_text_similarity(text_norm, self._normalize_text(kw))
                    if sim >= 0.75 or kw in text_norm:
                        is_cta_lexical = True
                        break

            is_cta = is_cta_flag or is_cta_lexical
            if is_cta:
                cta_elements_indices.append(idx)
                cta_multimodal_scores.append(multimodal_score)

        # ---------- Aggregate features ----------
        # Text → Action / Motion: робастное среднее по окнам (z-score motion)
        features["text_action_sync_score"] = self._trimmed_mean(text_action_scores_windowed)

        # Alignment: среднее и "оконное" (максимум в окне)
        features["text_motion_alignment"] = float(
            np.mean(text_motion_align_scores) if text_motion_align_scores else 0.0
        )
        features["text_motion_alignment_windowed"] = float(
            np.mean(text_motion_align_scores_windowed) if text_motion_align_scores_windowed else 0.0
        )

        # Multimodal attention boost: максимум + относительная позиция
        if multimodal_per_text:
            scores_arr = np.asarray([s for s, _ in multimodal_per_text], dtype=np.float32)
            times_arr = np.asarray([t for _, t in multimodal_per_text], dtype=np.float32)
            max_idx = int(np.argmax(scores_arr))
            features["multimodal_attention_boost_score"] = float(scores_arr[max_idx])
            rel_pos = (
                float(times_arr[max_idx] / max(total_video_seconds, 1e-6))
                if total_video_seconds > 0
                else 0.0
            )
            features["multimodal_attention_boost_position"] = rel_pos
        else:
            features["multimodal_attention_boost_score"] = 0.0
            features["multimodal_attention_boost_position"] = 0.0

        # ---------- 2. Text Duration and Continuity ----------
        durations_sec: List[float] = []
        for elem in unique_elements:
            if not elem["frames"]:
                continue
            dur_frames = elem["last_frame"] - elem["first_frame"] + 1
            durations_sec.append(dur_frames / self.video_fps)

        if durations_sec:
            d_arr = np.asarray(durations_sec, dtype=np.float32)
            mean_dur = float(d_arr.mean())
            features["text_on_screen_continuity"] = mean_dur
            features["text_on_screen_continuity_median"] = float(np.median(d_arr))
            features["text_on_screen_continuity_max"] = float(d_arr.max())
            features["text_on_screen_continuity_std"] = float(d_arr.std())
            features["text_on_screen_continuity_normalized"] = float(
                mean_dur / max(total_video_seconds, 1e-6)
            ) if total_video_seconds > 0 else 0.0
        else:
            features["text_on_screen_continuity"] = 0.0
            features["text_on_screen_continuity_median"] = 0.0
            features["text_on_screen_continuity_max"] = 0.0
            features["text_on_screen_continuity_std"] = 0.0
            features["text_on_screen_continuity_normalized"] = 0.0

        # text_switch_rate: число уникальных элементов / длительность видео
        num_unique_texts = len(unique_elements)
        features["num_unique_texts"] = int(num_unique_texts)
        features["text_switch_rate"] = (
            float(num_unique_texts) / max(total_video_seconds, 1e-6)
            if total_video_seconds > 0
            else 0.0
        )

        # time_to_first_text
        if unique_elements:
            first_time = min(elem["first_time"] for elem in unique_elements)
            features["time_to_first_text_sec"] = float(first_time)
            features["time_to_first_text_position"] = float(
                first_time / max(total_video_seconds, 1e-6)
            ) if total_video_seconds > 0 else 0.0
        else:
            features["time_to_first_text_sec"] = None
            features["time_to_first_text_position"] = None

        # text_area_fraction
        mean_text_area_fraction, per_elem_area_frac = self._compute_text_area_fraction(unique_elements)
        features["text_area_fraction"] = mean_text_area_fraction

        # ---------- 3. Call-to-Action (CTA) Detection ----------
        cta_times_sec: List[float] = []
        cta_durations_sec: List[float] = []
        cta_readability_scores: List[float] = []
        cta_confidences: List[float] = []

        for idx in cta_elements_indices:
            elem = unique_elements[idx]
            cta_times_sec.append(elem["first_time"])
            dur_frames = elem["last_frame"] - elem["first_frame"] + 1
            cta_durations_sec.append(dur_frames / self.video_fps)
            cta_readability_scores.append(self._readability_score(elem["text_norm"]))
            cta_confidences.append(elem.get("aggregated_confidence", 0.0))

        # cta_presence как вероятность (0..1) на основе числа CTA-элементов и их уверенности
        if cta_elements_indices:
            base_prob = min(1.0, len(cta_elements_indices) / max(num_unique_texts, 1) * 1.5)
            conf_mean = float(np.mean(cta_confidences)) if cta_confidences else 0.5
            features["cta_presence"] = float(max(0.0, min(1.0, 0.5 * base_prob + 0.5 * conf_mean)))
        else:
            features["cta_presence"] = 0.0

        if cta_times_sec:
            times_arr = np.asarray(cta_times_sec, dtype=np.float32)
            first_t = float(times_arr.min())
            mean_t = float(times_arr.mean())
            last_t = float(times_arr.max())
            features["cta_first_timestamp"] = first_t
            features["cta_mean_timestamp"] = mean_t
            features["cta_last_timestamp"] = last_t
            features["cta_first_position"] = float(
                first_t / max(total_video_seconds, 1e-6)
            ) if total_video_seconds > 0 else 0.0
            features["cta_mean_position"] = float(
                mean_t / max(total_video_seconds, 1e-6)
            ) if total_video_seconds > 0 else 0.0
            features["cta_last_position"] = float(
                last_t / max(total_video_seconds, 1e-6)
            ) if total_video_seconds > 0 else 0.0
            # оставляем cta_timestamp для обратной совместимости (mean)
            features["cta_timestamp"] = mean_t
        else:
            features["cta_first_timestamp"] = None
            features["cta_mean_timestamp"] = None
            features["cta_last_timestamp"] = None
            features["cta_first_position"] = None
            features["cta_mean_position"] = None
            features["cta_last_position"] = None
            features["cta_timestamp"] = None

        # cta_strength как нормализованный мультимодальный скор в CTA-элементах
        if cta_multimodal_scores:
            c_arr = np.asarray(cta_multimodal_scores, dtype=np.float32)
            features["cta_strength"] = float(np.clip(c_arr.mean(), 0.0, 1.0))
        else:
            features["cta_strength"] = 0.0

        # persistent_cta_flag: CTA, который держится дольше 3 секунд
        persistent = any(dur > 3.0 for dur in cta_durations_sec)
        features["persistent_cta_flag"] = bool(persistent)

        # ---------- 4. Text Emphasis Peaks ----------
        # используем последовательность мультимодальных скорингов по текстовым элементам
        if text_motion_align_scores:
            scores_arr = np.asarray(text_motion_align_scores, dtype=np.float32)
            peaks, props = find_peaks(
                scores_arr,
                prominence=0.1,
                distance=1,
            )
            features["text_emphasis_peak_flags"] = peaks.tolist()
            features["text_emphasis_peak_prominence"] = (
                props.get("prominences", np.zeros_like(peaks, dtype=np.float32)).tolist()
                if "prominences" in props
                else []
            )
            # относительное положение пиков в видео
            peak_times = []
            for pi in peaks:
                if pi < len(unique_elements):
                    peak_times.append(unique_elements[pi]["first_time"])
            if peak_times and total_video_seconds > 0:
                peak_positions = [float(t / max(total_video_seconds, 1e-6)) for t in peak_times]
            else:
                peak_positions = []
            features["text_emphasis_peak_positions"] = peak_positions
        else:
            features["text_emphasis_peak_flags"] = []
            features["text_emphasis_peak_prominence"] = []
            features["text_emphasis_peak_positions"] = []

        # ---------- 5. Дополнительные агрегаты ----------
        # text_readability_score (средний по уникальным элементам)
        readability_scores = [
            self._readability_score(elem["text_norm"]) for elem in unique_elements
        ]
        features["text_readability_score"] = float(
            np.mean(readability_scores) if readability_scores else 0.0
        )

        # ocr_language_entropy
        lang_counts: Dict[str, int] = defaultdict(int)
        for elem in unique_elements:
            lang = elem.get("language")
            if lang:
                lang_counts[lang] += 1
        features["ocr_language_entropy"] = self._shannon_entropy(lang_counts)

        # text_movement_speed: средняя скорость движения bbox (в долях кадра в секунду), если есть треки
        movement_speeds = []
        if self.frame_width and self.frame_height:
            diag = math.sqrt(self.frame_width ** 2 + self.frame_height ** 2)
            for elem in unique_elements:
                frames = elem["frames"]
                bboxes = elem["bboxes"]
                if len(frames) < 2:
                    continue
                dist_sum = 0.0
                time_sum = 0.0
                for i in range(1, len(frames)):
                    (x1a, y1a, x2a, y2a) = bboxes[i - 1]
                    (x1b, y1b, x2b, y2b) = bboxes[i]
                    cxa = (x1a + x2a) / 2.0
                    cya = (y1a + y2a) / 2.0
                    cxb = (x1b + x2b) / 2.0
                    cyb = (y1b + y2b) / 2.0
                    dist = math.sqrt((cxb - cxa) ** 2 + (cyb - cya) ** 2) / (diag + 1e-6)
                    dt = (frames[i] - frames[i - 1]) / self.video_fps
                    if dt > 0:
                        dist_sum += dist
                        time_sum += dt
                if time_sum > 0:
                    movement_speeds.append(dist_sum / time_sum)
        features["text_movement_speed"] = float(
            np.mean(movement_speeds) if movement_speeds else 0.0
        )

        # --- Raw / grouped OCR data for explainability ---
        features["ocr_raw"] = raw_ocr
        features["ocr_unique_elements"] = [
            {
                "text_raw": elem["text_raw"],
                "text_norm": elem["text_norm"],
                "language": elem.get("language"),
                "first_frame": elem["first_frame"],
                "last_frame": elem["last_frame"],
                "first_time": elem["first_time"],
                "last_time": elem["last_time"],
                "bbox_median": elem["bbox_median"],
                "aggregated_confidence": elem["aggregated_confidence"],
            }
            for elem in unique_elements
        ]

        return dict(features)


def _find_ocr_npz(rs_path: str) -> Optional[str]:
    """
    Canonical location (proposed):
    - `<rs_path>/text_ocr/ocr.npz`

    Compatibility:
    - `<rs_path>/ocr/ocr.npz`
    - `<rs_path>/text_scoring/ocr.npz` (legacy custom runs)
    """
    candidates = [
        os.path.join(rs_path, "text_ocr", "ocr.npz"),
        os.path.join(rs_path, "ocr", "ocr.npz"),
        os.path.join(rs_path, "text_scoring", "ocr.npz"),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    return None


def _load_ocr_npz(path: str) -> List[Dict[str, Any]]:
    """
    Minimal supported schema:
    - key `ocr_raw` -> object array holding list[dict]
    - or key `ocr_data` -> object array holding list[dict]
    Each dict should contain at least: `frame`, `bbox`, `text` (or `text_raw`), `confidence`.
    """
    data = np.load(path, allow_pickle=True)
    raw = data.get("ocr_raw")
    if raw is None:
        raw = data.get("ocr_data")
    if raw is None:
        return []
    if isinstance(raw, np.ndarray) and raw.dtype == object:
        raw_item = raw.item() if raw.ndim == 0 else raw.tolist()
    else:
        raw_item = raw
    if isinstance(raw_item, list):
        # ensure dict-like
        out: List[Dict[str, Any]] = []
        for d in raw_item:
            if isinstance(d, dict):
                out.append(d)
        return out
    return []


def _face_presence_signal_from_core_face_landmarks(core_npz: Dict[str, Any]) -> np.ndarray:
    """
    Returns float32 signal in [0,1] with length = number of frames in core_face_landmarks sample.
    """
    face_present = core_npz.get("face_present")
    if face_present is None:
        return np.asarray([], dtype=np.float32)
    fp = np.asarray(face_present)
    if fp.ndim == 1:
        present_any = fp.astype(bool)
    else:
        present_any = np.any(fp.astype(bool), axis=1)
    return present_any.astype(np.float32)


class TextScoringModule(BaseModule):
    @property
    def module_name(self) -> str:
        return "text_scoring"

    def process(self, frame_manager: FrameManager, frame_indices: List[int], config: Dict[str, Any]) -> Dict[str, Any]:
        if self.rs_path is None:
            raise ValueError("text_scoring | rs_path is required")
        fi = np.asarray([int(i) for i in frame_indices], dtype=np.int32)

        ocr_npz = config.get("ocr_npz")
        if ocr_npz is None:
            ocr_npz = _find_ocr_npz(self.rs_path)

        # Optional face signal
        use_face_data = bool(config.get("use_face_data", False))
        face_signal: Optional[np.ndarray] = None
        if use_face_data:
            core = self.load_core_provider("core_face_landmarks")
            if core is None:
                raise FileNotFoundError("text_scoring | core_face_landmarks requested but not found")
            # Align by frame_indices (strict)
            core_idx = core.get("frame_indices")
            if core_idx is None:
                raise RuntimeError("text_scoring | core_face_landmarks missing frame_indices")
            core_idx = np.asarray(core_idx, dtype=np.int32)
            mapping = {int(x): i for i, x in enumerate(core_idx.tolist())}
            pos = [mapping.get(int(x), -1) for x in fi.tolist()]
            if any(p < 0 for p in pos):
                raise RuntimeError(
                    "text_scoring | core_face_landmarks does not cover requested frame_indices. "
                    "Segmenter must provide consistent sampling if you enable --use-face-data."
                )
            face_all = _face_presence_signal_from_core_face_landmarks(core)
            face_signal = face_all[np.asarray(pos, dtype=np.int64)]

        if not ocr_npz:
            # Valid empty output (OCR not available)
            features = {
                "text_present": False,
                "empty_reason": "ocr_not_available",
            }
            return {
                "frame_indices": fi,
                "text_present": np.asarray(False),
                "features": np.asarray(features, dtype=object),
                "ocr_raw": np.asarray([], dtype=object),
                "ocr_unique_elements": np.asarray([], dtype=object),
            }

        ocr_data = _load_ocr_npz(str(ocr_npz))
        if not ocr_data:
            features = {
                "text_present": False,
                "empty_reason": "ocr_empty",
                "ocr_npz": str(ocr_npz),
            }
            return {
                "frame_indices": fi,
                "text_present": np.asarray(False),
                "features": np.asarray(features, dtype=object),
                "ocr_raw": np.asarray([], dtype=object),
                "ocr_unique_elements": np.asarray([], dtype=object),
            }

        # Filter OCR detections to this module's frame_indices (union-domain)
        allowed = set(int(x) for x in fi.tolist())
        ocr_filtered = [d for d in ocr_data if int(d.get("frame", -1)) in allowed]

        # If after filtering there is nothing — still a valid empty result
        if not ocr_filtered:
            features = {
                "text_present": False,
                "empty_reason": "ocr_outside_sampling",
                "ocr_npz": str(ocr_npz),
            }
            return {
                "frame_indices": fi,
                "text_present": np.asarray(False),
                "features": np.asarray(features, dtype=object),
                "ocr_raw": np.asarray([], dtype=object),
                "ocr_unique_elements": np.asarray([], dtype=object),
            }

        # Build motion/face/audio signals for the pipeline.
        # Baseline: no motion/audio integration (zeros). Face signal optional.
        max_frame = max(int(d.get("frame", 0)) for d in ocr_filtered)
        num_frames = max_frame + 1

        motion_peaks = np.zeros((num_frames,), dtype=np.float32)
        face_peaks = np.zeros((num_frames,), dtype=np.float32)
        if face_signal is not None:
            # face_signal is aligned to fi; scatter it into union-indexed buffer
            for idx_in_list, frame_idx in enumerate(fi.tolist()):
                if int(frame_idx) < num_frames:
                    face_peaks[int(frame_idx)] = float(face_signal[idx_in_list])

        fps = float(getattr(frame_manager, "fps", 30.0) or 30.0)
        pipeline = TextVideoInteractionPipeline(
            video_fps=int(round(fps)),
            frame_width=int(getattr(frame_manager, "width", 0) or 0) or None,
            frame_height=int(getattr(frame_manager, "height", 0) or 0) or None,
        )

        feats = pipeline.extract_features(
            ocr_data=ocr_filtered,
            motion_peaks=motion_peaks.tolist(),
            face_peaks=face_peaks.tolist(),
            audio_peaks=None,
        )

        # Pipeline already returns `ocr_raw` and `ocr_unique_elements`.
        ocr_raw = feats.pop("ocr_raw", [])
        ocr_unique = feats.pop("ocr_unique_elements", [])
        feats["ocr_npz"] = str(ocr_npz)
        feats["text_present"] = True

        return {
            "frame_indices": fi,
            "text_present": np.asarray(True),
            "features": np.asarray(dict(feats), dtype=object),
            "ocr_raw": np.asarray(ocr_raw, dtype=object),
            "ocr_unique_elements": np.asarray(ocr_unique, dtype=object),
        }
