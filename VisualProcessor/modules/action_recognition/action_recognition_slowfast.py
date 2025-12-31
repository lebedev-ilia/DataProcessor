# action_recognition_slowfast.py
"""
Production-ready action recognition and analytics based on SlowFast (ResNet-50).

Основные отличия от "простого" варианта:
- интеграция с BaseModule для единообразия
- более безопасная обработка ошибок и памяти
- нормализация эмбеддингов выполняется на устройстве (GPU если доступен)
- результаты сохраняются в детерминированном формате, сопровождаемые metadata
- комментарии на русском
"""

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict
import random
import json

import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models.video import slowfast_r50
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

from modules.base_module import BaseModule
from utils.frame_manager import FrameManager


def entropy_of_prob(p: np.ndarray) -> float:
    """Энтропия распределения вероятностей"""
    p = np.clip(p, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Косинусное сходство"""
    a_norm = a / (np.linalg.norm(a) + 1e-6)
    b_norm = b / (np.linalg.norm(b) + 1e-6)
    return float(np.dot(a_norm, b_norm))


def longest_run_fraction(labels: List[int]) -> float:
    """Доля самой длинной непрерывной серии одинаковых кластеров"""
    if not labels:
        return 0.0
    max_run = cur = 1
    for a, b in zip(labels, labels[1:]):
        cur = cur + 1 if a == b else 1
        max_run = max(max_run, cur)
    return max_run / len(labels)


class SlowFastActionRecognizer(BaseModule):
    """
    Production-реализация анализатора действий на основе SlowFast.
    
    Наследуется от BaseModule для единообразия с другими модулями.
    """

    def __init__(
        self,
        rs_path: Optional[str] = None,
        clip_len: int = 16,
        stride: Optional[int] = None,
        batch_size: int = 8,
        embedding_dim: int = 256,
        device: Optional[str] = None,
        seed: Optional[int] = 42,
        **kwargs: Any
    ):
        """
        Инициализация SlowFastActionRecognizer.
        
        Args:
            rs_path: Путь к хранилищу результатов
            clip_len: Длина клипа в кадрах
            stride: Шаг для создания клипов (по умолчанию clip_len // 2)
            batch_size: Размер батча для inference
            embedding_dim: Размерность эмбеддингов
            device: Устройство для обработки (cuda/cpu)
            seed: Seed для детерминированности
            **kwargs: Дополнительные параметры для BaseModule
        """
        super().__init__(rs_path=rs_path, **kwargs)
        
        # параметры
        self.clip_len = int(clip_len)
        self.stride = stride or max(1, self.clip_len // 2)
        self.batch_size = int(batch_size)
        self.embedding_dim = int(embedding_dim)
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.seed = seed

        np.random.seed(seed)
        random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

        # Модель будет загружена в _do_initialize()
        self.model: Optional[torch.nn.Module] = None
        self.embedding_proj: Optional[torch.nn.Module] = None
        self.raw_embedding_dim = 2048

        # нормализация входа (ImageNet-like)
        self.mean = np.array([0.45, 0.45, 0.45], dtype=np.float32)
        self.std = np.array([0.225, 0.225, 0.225], dtype=np.float32)
    
    def _do_initialize(self) -> None:
        """Инициализация модели SlowFast."""
        self.logger.info("Инициализация SlowFast R50 (pretrained=True)")

        # загружаем модель
        model = slowfast_r50(pretrained=True)
        self.model = model.to(self.device).eval()

        # проекция в компактное пространство (инициализация)
        self.embedding_proj = nn.Linear(self.raw_embedding_dim, self.embedding_dim).to(self.device)
        nn.init.xavier_uniform_(self.embedding_proj.weight)
        nn.init.zeros_(self.embedding_proj.bias)
        self.embedding_proj.eval()

        self.logger.info(
            f"SlowFastActionRecognizer готов | clip_len={self.clip_len} stride={self.stride} "
            f"batch_size={self.batch_size} embedding_dim={self.embedding_dim} device={self.device}"
        )

    def _load_frames(self, frame_manager: FrameManager, indices: List[int]) -> List[np.ndarray]:
        """Загружает и нормализует кадры как RGB uint8 HxWx3"""
        frames: List[np.ndarray] = []
        for idx in indices:
            im = frame_manager.get(idx)
            if im is None:
                raise ValueError(f"FrameManager.get({idx}) вернул None")
            if im.ndim == 2:
                im = np.stack([im] * 3, axis=-1)
            if im.shape[-1] == 4:
                im = im[..., :3]
            frames.append(im.astype(np.uint8))
        return frames

    def _make_clips(self, frames: List[np.ndarray]) -> List[List[np.ndarray]]:
        """Разбивает последовательность кадров на перекрывающиеся клипы length=clip_len, stride=self.stride"""
        if len(frames) == 0:
            return []

        if len(frames) < self.clip_len:
            frames = frames + [frames[-1]] * (self.clip_len - len(frames))

        clips: List[List[np.ndarray]] = []
        for start in range(0, len(frames) - self.clip_len + 1, self.stride):
            clips.append(frames[start:start + self.clip_len])

        if not clips:
            clips.append(frames[-self.clip_len:])

        return clips

    def _preprocess_clip(self, clip: List[np.ndarray]) -> torch.Tensor:
        """
        Преобразует clip (List[HxWx3]) в Tensor [C, T, H, W] float32 на CPU.
        Приводит кадры к 224x224, выполняет нормализацию.
        """
        processed = []
        for frame in clip:
            # resize -> float32 -> normalize -> C,H,W
            frame_resized = cv2.resize(frame, (224, 224), interpolation=cv2.INTER_LINEAR)
            frame_float = frame_resized.astype(np.float32) / 255.0
            frame_norm = (frame_float - self.mean) / self.std
            frame_chw = np.transpose(frame_norm, (2, 0, 1))
            processed.append(frame_chw)
        clip_arr = np.stack(processed, axis=1)  # C,T,H,W
        return torch.from_numpy(clip_arr).float()

    @staticmethod
    def _prepare_slow_fast(batch: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        batch: [B, C, T, H, W] -> возвращает slow и fast пути
        Замечание: выбор индексов стабильный и детерминированный.
        """
        if batch.dim() != 5:
            raise ValueError(f"Ожидался batch.dim()==5, получено {batch.dim()}")
        B, C, T, H, W = batch.shape
        # детерминированные индексы: берем примерно 8 и 16 сэмплов
        slow_idx = list(range(0, T, max(1, T // 8)))
        fast_idx = list(range(0, T, max(1, T // 16)))
        slow = batch[:, :, slow_idx, :, :]
        fast = batch[:, :, fast_idx, :, :]
        return slow, fast

    def _extract_features(self, slow: torch.Tensor, fast: torch.Tensor) -> torch.Tensor:
        """
        Прогоняем через модель и приводим к [B, raw_embedding_dim].
        В случае ошибки возвращаем нулевой тензор (предсказуемый fallback).
        """
        try:
            with torch.no_grad():
                out = self.model((slow, fast))

            # модель может вернуть tensor или tuple/list; пытаемся извлечь tensor
            if torch.is_tensor(out):
                feat = out
            elif isinstance(out, (list, tuple)) and len(out) > 0 and torch.is_tensor(out[0]):
                feat = out[0]
            else:
                raise RuntimeError("Неожиданный тип выхода модели: %s" % type(out))

            # уменьшаем пространственно-временные размерности
            if feat.dim() == 5:
                # B, C, T, H, W -> усредняем по T,H,W -> B,C
                feat = feat.mean(dim=[2, 3, 4])
            elif feat.dim() == 4:
                # B,C,H,W -> усредняем по H,W -> B,C
                feat = feat.mean(dim=[2, 3])
            elif feat.dim() == 3:
                # B,C,T -> усредняем по T -> B,C
                feat = feat.mean(dim=2)

            feat = feat.view(feat.size(0), -1)  # B, D

        except Exception:
            self.logger.exception("Ошибка при извлечении признаков через SlowFast. Возвращаем нули как fallback.")
            feat = torch.zeros((slow.size(0), self.raw_embedding_dim), device=self.device)

        # выравнивание по целевому raw_embedding_dim
        if feat.shape[1] > self.raw_embedding_dim:
            feat = feat[:, : self.raw_embedding_dim]
        elif feat.shape[1] < self.raw_embedding_dim:
            pad = torch.zeros((feat.shape[0], self.raw_embedding_dim - feat.shape[1]), device=feat.device)
            feat = torch.cat([feat, pad], dim=1)

        return feat

    def _extract_embeddings(self, clips: List[List[np.ndarray]]) -> np.ndarray:
        """
        Прогоняет все клипы батчами и возвращает np.ndarray эмбеддингов формы [N_clips, embedding_dim].
        Если клипов нет — возвращает пустой массив shape (0, embedding_dim).
        """
        if not clips:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        projected_all = []
        total_clips = len(clips)
        self.logger.info("Начинаем извлечение эмбеддингов: clips=%d batch_size=%d", total_clips, self.batch_size)

        for start in range(0, total_clips, self.batch_size):
            batch_clips = clips[start: start + self.batch_size]
            # преобразуем каждый клип в тензор C,T,H,W на CPU
            tensors_cpu = [self._preprocess_clip(c) for c in batch_clips]  # list of [C,T,H,W]
            batch = torch.stack(tensors_cpu, dim=0).to(self.device)  # [B,C,T,H,W]

            try:
                slow, fast = self._prepare_slow_fast(batch)
            except Exception:
                self.logger.exception("Некорректный batch shape при подготовке slow/fast. Пропускаем этот батч.")
                # очищаем память и продолжаем
                del batch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue

            feat = self._extract_features(slow, fast)  # [B, raw_dim]

            # Проекция + нормализация на device
            with torch.no_grad():
                proj = self.embedding_proj(feat)  # [B, embedding_dim]
                proj = F.normalize(proj, p=2, dim=1)

            # переносим на CPU и сохраняем
            projected_all.append(proj.cpu().numpy().astype(np.float32))

            # освобождение
            del batch, slow, fast, feat, proj
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            if (start // self.batch_size) % 10 == 0:
                self.logger.debug("Processed %d/%d clips", min(start + self.batch_size, total_clips), total_clips)

        if not projected_all:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)

        embeddings = np.concatenate(projected_all, axis=0)
        # safety: если размер не совпадает — обрезаем/паддим
        if embeddings.shape[1] != self.embedding_dim:
            if embeddings.shape[1] > self.embedding_dim:
                embeddings = embeddings[:, : self.embedding_dim]
            else:
                pad = np.zeros((embeddings.shape[0], self.embedding_dim - embeddings.shape[1]), dtype=np.float32)
                embeddings = np.concatenate([embeddings, pad], axis=1)

        return embeddings

    def _aggregate(self, embeddings: np.ndarray) -> Dict[str, Any]:
        """
        Вычисляет метрики для последовательности эмбеддингов трека.
        Возвращает словарь метрик (числа и статистика).
        """
        n = len(embeddings)
        if n == 0:
            # пустой трек — возвращаем нейтральные значения
            return {
                "mean_embedding_norm": 0.0,
                "std_embedding_norm": 0.0,
                "max_temporal_jump": 0.0,
                "stability": 1.0,
                "num_switches": 0,
                "embedding_dim": self.embedding_dim,
            }

        norms = np.linalg.norm(embeddings, axis=1)
        mean_norm = float(np.mean(norms))
        std_norm = float(np.std(norms))

        if n > 1:
            diffs = [np.linalg.norm(embeddings[i] - embeddings[i - 1]) for i in range(1, n)]
            max_jump = float(np.max(diffs))
        else:
            max_jump = 0.0

        # кластеризация только при достаточном числе клипов
        if n >= 3:
            pca_dim = min(32, embeddings.shape[1], n - 1)
            try:
                emb_pca = PCA(n_components=pca_dim).fit_transform(embeddings)
                k = min(5, max(1, n // 2))
                labels = KMeans(n_clusters=k, n_init=10, random_state=42).fit_predict(emb_pca)
                stability = longest_run_fraction(labels.tolist())
                switches = int(np.sum(labels[1:] != labels[:-1]))
            except Exception:
                self.logger.exception("Ошибка в PCA/KMeans на треке — возвращаем безопасные метрики.")
                stability = 1.0
                switches = 0
        else:
            stability = 1.0
            switches = 0

        return {
            "mean_embedding_norm": mean_norm,
            "std_embedding_norm": std_norm,
            "max_temporal_jump": max_jump,
            "stability": stability,
            "num_switches": switches,
            "embedding_dim": self.embedding_dim,
        }

    def process(
        self,
        frame_manager: FrameManager,
        frame_indices: List[int],
    ) -> Dict[int, Dict[str, Any]]:
        """
        Основной метод обработки видео (интерфейс BaseModule).
        
        Args:
            frame_manager: Менеджер кадров
            frame_indices: Список индексов кадров для обработки
            config: Конфигурация модуля
                
        Returns:
            Dict[track_id, Dict] - результаты по трекам с эмбеддингами
        """
        self.initialize()  # Гарантируем инициализацию модели
        
        frame_indices_per_person = self._prepare_tracks(
            frame_indices=frame_indices
        )
        
        if not frame_indices_per_person:
            self.logger.warning("Нет треков для обработки")
            return {}
        
        all_clips: List[List[np.ndarray]] = []
        clip_owner: List[int] = []

        # сбор всех клипов
        for tid, indices in frame_indices_per_person.items():
            if not indices:
                self.logger.debug("Пропускаем трек %s (пустой список индексов)", tid)
                continue
            frames = self._load_frames(frame_manager, indices)
            clips = self._make_clips(frames)
            if not clips:
                self.logger.debug("Трек %s не дал клипов после _make_clips()", tid)
                continue
            all_clips.extend(clips)
            clip_owner.extend([tid] * len(clips))

        if not all_clips:
            self.logger.warning("Нет клипов для обработки")
            return {}

        embeddings = self._extract_embeddings(all_clips)  # numpy [N_clips, embedding_dim]

        # агрегируем по трекам
        per_track_embeddings: Dict[int, List[np.ndarray]] = defaultdict(list)
        for owner_tid, emb in zip(clip_owner, embeddings):
            per_track_embeddings[owner_tid].append(emb)

        results: Dict[int, Dict[str, Any]] = {}
        for tid, embs in per_track_embeddings.items():
            embs_arr = np.stack(embs, axis=0) if len(embs) else np.zeros((0, self.embedding_dim), dtype=np.float32)
            metrics = self._aggregate(embs_arr)
            metrics["embedding_normed_256d"] = embs_arr
            results[tid] = metrics

        self.logger.info("Обработано треков: %d", len(results))

        return results
    
    def _prepare_tracks(
        self,
        frame_indices: List[int]
    ) -> Dict[int, List[int]]:
        """
        Подготавливает треки для обработки из detections.npz.
        
        Формат из detections.npz:
        - tracks_list: object array, где tracks_list[i] - numpy массив frame indices для трека
        - tracks_list_ids: массив track_ids, где tracks_list_ids[i] - track_id для tracks_list[i]
        - frame_indices: массив индексов кадров, которые были обработаны
        
        Args:
            frame_indices: Список индексов кадров для обработки
            config: Конфигурация с max_tracks (опционально)
            
        Returns:
            Dict[track_id, List[frame_indices]] - словарь треков
        """
        
        detections_data = self.load_core_provider("core_object_detections", "detections.npz")
        
        if detections_data is None:
            raise ValueError(
                "Не найдены результаты core_object_detections. "
                "Убедитесь, что модуль object_detections был запущен."
            )
        
        tracks_list = detections_data.get("tracks_list")
        tracks_list_ids = detections_data.get("tracks_list_ids")
        
        if tracks_list is None or tracks_list_ids is None:
            raise ValueError(
                "В detections.npz отсутствуют tracks_list или tracks_list_ids. "
                "Убедитесь, что трекинг был выполнен."
            )
        
        if not isinstance(tracks_list, np.ndarray) or not isinstance(tracks_list_ids, np.ndarray):
            raise ValueError(
                "tracks_list и tracks_list_ids должны быть numpy массивами. "
                f"Получено: tracks_list={type(tracks_list)}, tracks_list_ids={type(tracks_list_ids)}"
            )
        
        if len(tracks_list) != len(tracks_list_ids):
            raise ValueError(
                f"Длины tracks_list и tracks_list_ids не совпадают: "
                f"{len(tracks_list)} != {len(tracks_list_ids)}"
            )
        
        if len(tracks_list) == 0:
            raise ValueError(
                "tracks_list пуст. Убедитесь, что трекинг был выполнен и дал результаты."
            )
        
        frame_indices_per_person: Dict[int, List[int]] = {}
        frame_indices_set = set(frame_indices)
        
        for i in range(len(tracks_list_ids)):
            track_id = int(tracks_list_ids[i])
            track_frames = tracks_list[i]
            
            # Преобразуем numpy массив в список
            if isinstance(track_frames, np.ndarray):
                frame_indices_for_track = track_frames.tolist()
            elif isinstance(track_frames, (list, tuple)):
                frame_indices_for_track = list(track_frames)
            else:
                self.logger.warning(
                    "Неожиданный тип для tracks_list[%d]: %s. Пропускаем трек %d.",
                    i, type(track_frames), track_id
                )
                continue
            
            # Фильтруем только те индексы, которые есть в frame_indices
            filtered_indices = [int(idx) for idx in frame_indices_for_track if int(idx) in frame_indices_set]
            if filtered_indices:
                frame_indices_per_person[track_id] = sorted(filtered_indices)
        
        if not frame_indices_per_person:
            raise ValueError(
                "Не найдено ни одного трека в detections.npz, который пересекается с frame_indices."
            )
        
        self.logger.info(
            "Загружены треки из detections.npz (tracks=%d)", 
            len(frame_indices_per_person)
        )
        
        return frame_indices_per_person