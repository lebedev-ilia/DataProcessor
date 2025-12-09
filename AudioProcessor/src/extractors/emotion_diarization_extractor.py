"""
Экстрактор эмоциональной диаризации на основе SpeechBrain.
"""
import time
import logging
import numpy as np
import torch
from typing import Dict, Any, Optional, List
from pathlib import Path
from contextlib import redirect_stdout, redirect_stderr
import os

from src.core.base_extractor import BaseExtractor, ExtractorResult
from src.core.audio_utils import AudioUtils

logger = logging.getLogger(__name__)


class EmotionDiarizationExtractor(BaseExtractor):
    """Экстрактор эмоциональной диаризации на основе SpeechBrain."""
    
    name = "emotion_diarization_extractor"
    version = "1.0.0"
    description = "Эмоциональная диаризация с помощью SpeechBrain"
    category = "speech"
    dependencies = ["torch", "speechbrain", "librosa"]
    estimated_duration = 5.0
    
    # Предпочитает GPU для SpeechBrain
    gpu_required = False
    gpu_preferred = True
    gpu_memory_required = 1.0  # 1GB для SpeechBrain модели
    
    def __init__(
        self, 
        device: str = "auto",
        model_path: str = "/home/ilya/Рабочий стол/DataProcessor/AudioProcessor/pretrained_models/emotion_diarization",
        sample_rate: int = 16000,
        batch_size: int = 1
    ):
        """
        Инициализация экстрактора эмоциональной диаризации.
        
        Args:
            device: Устройство для обработки
            model_path: Путь к предобученной модели
            sample_rate: Частота дискретизации
            batch_size: Размер батча для обработки
        """
        super().__init__(device=device)
        
        self.model_path = model_path
        self.sample_rate = sample_rate
        self.batch_size = batch_size
        
        self.audio_utils = AudioUtils(device=device, sample_rate=sample_rate)
        
        # Инициализируем модель
        self._setup_model()
    
    def _setup_model(self):
        """Настройка модели эмоциональной диаризации."""
        try:
            # Приглушаем шумные логи SpeechBrain
            try:
                for _name in [
                    "speechbrain",
                    "speechbrain.utils.fetching",
                    "speechbrain.utils.parameter_transfer",
                    "speechbrain.dataio.encoder",
                ]:
                    logging.getLogger(_name).setLevel(logging.ERROR)
            except Exception:
                pass

            # Скрываем инфо-логи инициализации модели
            with open(os.devnull, "w") as _devnull:
                with redirect_stdout(_devnull), redirect_stderr(_devnull):
                    # Импортируем SpeechBrain только здесь, чтобы избежать ошибок при отсутствии библиотеки
                    from speechbrain.inference.diarization import Speech_Emotion_Diarization
                    
                    # Инициализируем модель
                    self.sed_model = Speech_Emotion_Diarization.from_hparams(
                        source=self.model_path,
                        savedir=self.model_path,
                    )
            
            # self.logger.debug("Модель эмоциональной диаризации инициализирована")
            
        except ImportError as e:
            self.logger.error(f"SpeechBrain не установлен: {e}")
            raise RuntimeError("SpeechBrain не установлен. Установите: pip install speechbrain")
        except Exception as e:
            self.logger.error(f"Ошибка инициализации модели эмоциональной диаризации: {e}")
            raise
    
    def run(self, input_uri: str, tmp_path: str) -> ExtractorResult:
        """
        Извлечение эмоциональной диаризации.
        
        Args:
            input_uri: Путь к аудио файлу
            tmp_path: Временная директория
            
        Returns:
            ExtractorResult с эмоциональной диаризацией
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
            
            # Извлекаем эмоциональную диаризацию
            diarization_result = self._extract_emotion_diarization(input_uri)
            
            # Обрабатываем результат
            processed_result = self._process_diarization_result(diarization_result)
            
            processing_time = time.time() - start_time
            
            # Создаем результат
            payload = {
                "emotion_segments": processed_result["emotion_segments"],
                "emotion_labels": processed_result["emotion_labels"],
                "speaker_segments": processed_result["speaker_segments"],
                "emotion_speaker_mapping": processed_result["emotion_speaker_mapping"],
                "emotion_statistics": processed_result["emotion_statistics"],
                "speaker_count": processed_result["speaker_count"],
                "duration": processed_result["duration"],
                "device_used": self.device,
                "sample_rate": self.sample_rate,
                "model_path": self.model_path
            }
            
            self._log_extraction_success(input_uri, processing_time)
            
            return self._create_result(
                success=True,
                payload=payload,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Ошибка извлечения эмоциональной диаризации: {str(e)}"
            self._log_extraction_error(input_uri, error_msg, processing_time)
            
            return self._create_result(
                success=False,
                error=error_msg,
                processing_time=processing_time
            )
    
    def _extract_emotion_diarization(self, audio_path: str) -> Dict[str, Any]:
        """Извлечение эмоциональной диаризации."""
        try:
            # SpeechBrain сам управляет устройством внутри diarize_file; приглушаем stdout/stderr
            with open(os.devnull, "w") as _devnull:
                with redirect_stdout(_devnull), redirect_stderr(_devnull):
                    result = self.sed_model.diarize_file(audio_path)
            return result
        except Exception as e:
            self.logger.error(f"Ошибка извлечения эмоциональной диаризации: {e}")
            raise
    
    def _process_diarization_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Обработка результата эмоциональной диаризации."""
        try:
            # Извлекаем сегменты из результата SpeechBrain
            # Структура результата зависит от конкретной модели
            emotion_segments = []
            speaker_segments = []
            emotion_labels = []
            emotion_speaker_mapping = {}
            
            # Обрабатываем результат в зависимости от его структуры
            if isinstance(result, dict):
                # Если результат - словарь, извлекаем сегменты
                segments = result.get("segments", [])
                emotions = result.get("emotions", [])
                speakers = result.get("speakers", [])
                
                for i, segment in enumerate(segments):
                    start_time = segment.get("start", 0.0)
                    end_time = segment.get("end", 0.0)
                    duration = end_time - start_time
                    
                    # Эмоциональный сегмент
                    emotion = emotions[i] if i < len(emotions) else "neutral"
                    emotion_segment = {
                        "start": start_time,
                        "end": end_time,
                        "duration": duration,
                        "emotion": emotion,
                        "confidence": segment.get("confidence", 0.0),
                        "segment_index": i
                    }
                    emotion_segments.append(emotion_segment)
                    
                    # Спикерский сегмент
                    speaker = speakers[i] if i < len(speakers) else 0
                    speaker_segment = {
                        "start": start_time,
                        "end": end_time,
                        "duration": duration,
                        "speaker_id": int(speaker),
                        "segment_index": i
                    }
                    speaker_segments.append(speaker_segment)
                    
                    # Маппинг эмоций к спикерам
                    emotion_speaker_mapping[f"speaker_{speaker}"] = emotion
                    
                    emotion_labels.append(emotion)
            
            elif isinstance(result, list):
                # Если результат - список сегментов
                for i, segment in enumerate(result):
                    if isinstance(segment, dict):
                        start_time = segment.get("start", 0.0)
                        end_time = segment.get("end", 0.0)
                        duration = end_time - start_time
                        
                        emotion = segment.get("emotion", "neutral")
                        speaker = segment.get("speaker", 0)
                        
                        # Эмоциональный сегмент
                        emotion_segment = {
                            "start": start_time,
                            "end": end_time,
                            "duration": duration,
                            "emotion": emotion,
                            "confidence": segment.get("confidence", 0.0),
                            "segment_index": i
                        }
                        emotion_segments.append(emotion_segment)
                        
                        # Спикерский сегмент
                        speaker_segment = {
                            "start": start_time,
                            "end": end_time,
                            "duration": duration,
                            "speaker_id": int(speaker),
                            "segment_index": i
                        }
                        speaker_segments.append(speaker_segment)
                        
                        # Маппинг эмоций к спикерам
                        emotion_speaker_mapping[f"speaker_{speaker}"] = emotion
                        
                        emotion_labels.append(emotion)
            
            # Вычисляем статистики
            unique_emotions = list(set(emotion_labels))
            unique_speakers = list(set([seg["speaker_id"] for seg in speaker_segments]))
            speaker_count = len(unique_speakers)
            
            # Статистики по эмоциям
            emotion_stats = {}
            for emotion in unique_emotions:
                emotion_segs = [seg for seg in emotion_segments if seg["emotion"] == emotion]
                total_duration = sum(seg["duration"] for seg in emotion_segs)
                emotion_stats[emotion] = {
                    "count": len(emotion_segs),
                    "total_duration": total_duration,
                    "percentage": (total_duration / sum(seg["duration"] for seg in emotion_segments)) * 100 if emotion_segments else 0
                }
            
            # Общая длительность
            duration = sum(seg["duration"] for seg in emotion_segments) if emotion_segments else 0.0
            
            return {
                "emotion_segments": emotion_segments,
                "speaker_segments": speaker_segments,
                "emotion_labels": emotion_labels,
                "emotion_speaker_mapping": emotion_speaker_mapping,
                "emotion_statistics": emotion_stats,
                "speaker_count": speaker_count,
                "duration": duration
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка обработки результата эмоциональной диаризации: {e}")
            return {
                "emotion_segments": [],
                "speaker_segments": [],
                "emotion_labels": [],
                "emotion_speaker_mapping": {},
                "emotion_statistics": {},
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
    
    def get_model_info(self) -> Dict[str, Any]:
        """Получение информации о модели."""
        return {
            "model_path": self.model_path,
            "sample_rate": self.sample_rate,
            "batch_size": self.batch_size,
            "device": self.device,
            "gpu_available": self.gpu_available
        }
