"""
Экстрактор диаризации спикеров на основе Resemblyzer с поддержкой GPU.
"""
import time
import logging
import numpy as np
import torch
import librosa
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path
from sklearn.cluster import AgglomerativeClustering
from resemblyzer import VoiceEncoder, preprocess_wav
from contextlib import redirect_stdout, redirect_stderr
import os

from src.core.base_extractor import BaseExtractor, ExtractorResult
from src.core.audio_utils import AudioUtils

logger = logging.getLogger(__name__)


class SpeakerDiarizationExtractor(BaseExtractor):
    """Экстрактор диаризации спикеров на основе Resemblyzer."""
    
    name = "speaker_diarization_extractor"
    version = "1.0.0"
    description = "Диаризация спикеров с помощью Resemblyzer"
    category = "speech"
    dependencies = ["torch", "resemblyzer", "librosa", "scikit-learn"]
    estimated_duration = 3.0
    
    # Предпочитает GPU для Resemblyzer
    gpu_required = False
    gpu_preferred = True
    gpu_memory_required = 0.5  # 500MB для Resemblyzer
    
    def __init__(
        self, 
        device: str = "auto",
        segment_duration: float = 2.0,
        min_speakers: int = 1,
        max_speakers: int = 10,
        sample_rate: int = 16000,
        clustering_method: str = "agglomerative"
    ):
        """
        Инициализация диаризационного экстрактора.
        
        Args:
            device: Устройство для обработки
            segment_duration: Длительность сегмента для анализа (секунды)
            min_speakers: Минимальное количество спикеров
            max_speakers: Максимальное количество спикеров
            sample_rate: Частота дискретизации
            clustering_method: Метод кластеризации ('agglomerative', 'kmeans')
        """
        super().__init__(device=device)
        
        self.segment_duration = segment_duration
        self.min_speakers = min_speakers
        self.max_speakers = max_speakers
        self.sample_rate = sample_rate
        self.clustering_method = clustering_method
        
        self.audio_utils = AudioUtils(device=device, sample_rate=sample_rate)
        
        # Инициализируем энкодер голоса
        self._setup_voice_encoder()
    
    def _setup_voice_encoder(self):
        """Настройка энкодера голоса Resemblyzer."""
        try:
            # Скрываем инфо-логи инициализации энкодера
            # self.logger.debug("Инициализация VoiceEncoder...")
            
            # Приглушаем stdout/stderr во время инициализации энкодера (lib печатает в stdout)
            with open(os.devnull, "w") as _devnull:
                with redirect_stdout(_devnull), redirect_stderr(_devnull):
                    # Инициализируем энкодер, принудительно на выбранном устройстве
                    encoder_device = "cuda" if (self.device == "cuda" and torch.cuda.is_available()) else "cpu"
                    self.encoder = VoiceEncoder(device=encoder_device)
            
            # Если используем GPU, перемещаем модель на GPU
            # Перенос на GPU уже учтён при создании; повторно не переносим
                # self.logger.debug("VoiceEncoder перемещен на GPU")
            
            # self.logger.debug("VoiceEncoder инициализирован")
            
        except Exception as e:
            self.logger.error(f"Ошибка инициализации VoiceEncoder: {e}")
            raise
    
    def run(self, input_uri: str, tmp_path: str) -> ExtractorResult:
        """
        Извлечение диаризации спикеров.
        
        Args:
            input_uri: Путь к аудио файлу
            tmp_path: Временная директория
            
        Returns:
            ExtractorResult с диаризацией спикеров
        """
        start_time = time.time()
        
        try:
            # Валидация входного файла
            if not self._validate_input(input_uri):
                return self._create_result(
                    success=False,
                    error="Некорректный входной файл",
                    processing_time=time.time() - start_time
                )
            
            self._log_extraction_start(input_uri)
            
            # Загружаем аудио
            waveform, sample_rate = self.audio_utils.load_audio(input_uri, self.sample_rate)
            
            # Нормализуем аудио
            waveform = self.audio_utils.normalize_audio(waveform)
            
            # Перемещаем на нужное устройство
            waveform = self.audio_utils._move_to_device(waveform)
            
            # Извлекаем диаризацию
            diarization_result = self._extract_diarization(waveform)
            
            # Обрабатываем результат
            processed_result = self._process_diarization_result(diarization_result)
            
            processing_time = time.time() - start_time
            
            # Сохраняем спикерные эмбеддинги в .npy, в JSON только путь и краткая статистика
            from pathlib import Path
            out_dir = Path(tmp_path)
            out_dir.mkdir(parents=True, exist_ok=True)
            mean_embeds_dict = processed_result.get("speaker_embeddings", {})
            speaker_ids_sorted = sorted(mean_embeds_dict.keys()) if mean_embeds_dict else []
            embeds_array = None
            if speaker_ids_sorted:
                import numpy as np
                embeds_array = np.stack([np.asarray(mean_embeds_dict[sid], dtype=np.float32) for sid in speaker_ids_sorted], axis=0)
                emb_path = out_dir / f"{Path(input_uri).stem}_speaker_embeddings_mean.npy"
                np.save(emb_path, embeds_array)
                speaker_embeddings_npy = str(emb_path)
                speaker_embeddings_shape = list(embeds_array.shape)
            else:
                speaker_embeddings_npy = None
                speaker_embeddings_shape = [0, 0]

            # Создаем результат
            payload = {
                "speaker_segments": processed_result["speaker_segments"],
                "speaker_embeddings_npy": speaker_embeddings_npy,
                "speaker_embeddings_shape": speaker_embeddings_shape,
                "speaker_ids_order": speaker_ids_sorted,
                "speaker_count": processed_result["speaker_count"],
                "segment_duration": self.segment_duration,
                "clustering_method": self.clustering_method,
                "duration": processed_result["duration"],
                "device_used": self.device,
                "sample_rate": sample_rate
            }
            
            self._log_extraction_success(input_uri, processing_time)
            
            return self._create_result(
                success=True,
                payload=payload,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Ошибка извлечения диаризации: {str(e)}"
            self._log_extraction_error(input_uri, error_msg, processing_time)
            
            return self._create_result(
                success=False,
                error=error_msg,
                processing_time=processing_time
            )
    
    def _extract_diarization(self, waveform: torch.Tensor) -> Dict[str, Any]:
        """Извлечение диаризации спикеров."""
        try:
            # Конвертируем в numpy для Resemblyzer
            audio_np = self.audio_utils.to_numpy(waveform)
            
            # Убираем batch dimension если есть
            if audio_np.ndim > 1:
                audio_np = audio_np[0] if audio_np.shape[0] == 1 else audio_np
            
            # Разделяем аудио на сегменты
            segments = self._split_audio_into_segments(audio_np)
            
            # Получаем эмбеддинги для каждого сегмента
            embeddings = []
            for segment in segments:
                if len(segment) > 0:
                    # Предобработка для Resemblyzer
                    processed_segment = preprocess_wav(segment)
                    embedding = self.encoder.embed_utterance(processed_segment)
                    embeddings.append(embedding)
                else:
                    # Пустой сегмент - добавляем нулевой эмбеддинг
                    embeddings.append(np.zeros(256))  # Resemblyzer возвращает 256-мерные эмбеддинги
            
            # Кластеризация спикеров
            speaker_labels = self._cluster_speakers(embeddings)
            
            return {
                "segments": segments,
                "embeddings": embeddings,
                "speaker_labels": speaker_labels
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка извлечения диаризации: {e}")
            raise
    
    def _split_audio_into_segments(self, audio: np.ndarray) -> List[np.ndarray]:
        """Разделение аудио на сегменты."""
        try:
            segments = []
            step = int(self.segment_duration * self.sample_rate)
            
            for start in range(0, len(audio), step):
                end = min(start + step, len(audio))
                segment = audio[start:end]
                segments.append(segment)
            
            return segments
            
        except Exception as e:
            self.logger.error(f"Ошибка разделения аудио на сегменты: {e}")
            return []
    
    def _cluster_speakers(self, embeddings: List[np.ndarray]) -> List[int]:
        """Кластеризация спикеров по эмбеддингам."""
        try:
            if len(embeddings) == 0:
                return []
            
            # Конвертируем в numpy array
            embeddings_array = np.array(embeddings)
            
            # Определяем оптимальное количество кластеров
            n_speakers = self._estimate_speaker_count(embeddings_array)
            
            if n_speakers <= 1:
                return [0] * len(embeddings)
            
            # Применяем кластеризацию
            if self.clustering_method == "agglomerative":
                clustering = AgglomerativeClustering(n_clusters=n_speakers)
                labels = clustering.fit_predict(embeddings_array)
            else:
                # Fallback к AgglomerativeClustering
                clustering = AgglomerativeClustering(n_clusters=n_speakers)
                labels = clustering.fit_predict(embeddings_array)
            
            return labels.tolist()
            
        except Exception as e:
            self.logger.error(f"Ошибка кластеризации спикеров: {e}")
            return [0] * len(embeddings)
    
    def _estimate_speaker_count(self, embeddings: np.ndarray) -> int:
        """Оценка количества спикеров."""
        try:
            if len(embeddings) < 2:
                return 1
            
            # Простая эвристика: используем расстояние между эмбеддингами
            from sklearn.metrics.pairwise import cosine_distances
            
            # Вычисляем попарные расстояния
            distances = cosine_distances(embeddings)
            
            # Находим среднее расстояние
            mean_distance = np.mean(distances[np.triu_indices_from(distances, k=1)])
            
            # Эвристика: если среднее расстояние большое, то спикеров много
            if mean_distance > 0.3:
                return min(3, len(embeddings) // 2)
            elif mean_distance > 0.2:
                return min(2, len(embeddings) // 3)
            else:
                return 1
                
        except Exception as e:
            self.logger.warning(f"Ошибка оценки количества спикеров: {e}")
            return 2  # Fallback к 2 спикерам
    
    def _process_diarization_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Обработка результата диаризации."""
        try:
            segments = result.get("segments", [])
            speaker_labels = result.get("speaker_labels", [])
            embeddings = result.get("embeddings", [])
            
            # Создаем сегменты спикеров
            speaker_segments = []
            for i, (segment, label) in enumerate(zip(segments, speaker_labels)):
                start_time = i * self.segment_duration
                end_time = start_time + self.segment_duration
                
                speaker_segment = {
                    "start": start_time,
                    "end": end_time,
                    "speaker_id": int(label),
                    "duration": self.segment_duration,
                    "segment_index": i
                }
                speaker_segments.append(speaker_segment)
            
            # Вычисляем статистики
            unique_speakers = list(set(speaker_labels))
            speaker_count = len(unique_speakers)
            
            # Группируем эмбеддинги по спикерам
            speaker_embeddings = {}
            for i, (embedding, label) in enumerate(zip(embeddings, speaker_labels)):
                if label not in speaker_embeddings:
                    speaker_embeddings[label] = []
                speaker_embeddings[label].append(embedding.tolist())
            
            # Вычисляем средние эмбеддинги для каждого спикера
            speaker_mean_embeddings = {}
            for speaker_id, speaker_emb_list in speaker_embeddings.items():
                if speaker_emb_list:
                    mean_embedding = np.mean(speaker_emb_list, axis=0)
                    speaker_mean_embeddings[speaker_id] = mean_embedding.tolist()
            
            # Общая длительность
            duration = len(segments) * self.segment_duration

            # Пер-спикерные агрегаты (кол-во сегментов и длительность)
            per_speaker_stats = {}
            for seg in speaker_segments:
                sid = int(seg.get("speaker_id", 0))
                per_speaker_stats.setdefault(sid, {"segments_count": 0, "total_duration": 0.0})
                per_speaker_stats[sid]["segments_count"] += 1
                per_speaker_stats[sid]["total_duration"] += float(seg.get("duration", 0.0) or 0.0)
            
            return {
                "speaker_segments": speaker_segments,
                "speaker_embeddings": speaker_mean_embeddings,
                "speaker_count": speaker_count,
                "duration": duration,
                "speaker_stats": per_speaker_stats,
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка обработки результата диаризации: {e}")
            return {
                "speaker_segments": [],
                "speaker_embeddings": {},
                "speaker_count": 0,
                "duration": 0.0
            }
    
    def _validate_input(self, input_uri: str) -> bool:
        """Валидация входного файла."""
        if not super()._validate_input(input_uri):
            return False
        
        # Проверяем, что это аудио файл
        audio_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.mp4', '.avi', '.mov'}
        if not any(input_uri.lower().endswith(ext) for ext in audio_extensions):
            self.logger.error(f"Файл не является поддерживаемым аудио/видео форматом: {input_uri}")
            return False
        
        return True
    
    def get_encoder_info(self) -> Dict[str, Any]:
        """Получение информации об энкодере."""
        return {
            "segment_duration": self.segment_duration,
            "min_speakers": self.min_speakers,
            "max_speakers": self.max_speakers,
            "sample_rate": self.sample_rate,
            "clustering_method": self.clustering_method,
            "device": self.device
        }
