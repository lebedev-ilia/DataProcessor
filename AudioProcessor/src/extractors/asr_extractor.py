"""
ASR экстрактор на основе Whisper с поддержкой GPU.
"""
import time
import logging
import numpy as np
import torch
import whisper
from typing import Dict, Any, Optional, List
from pathlib import Path

from src.core.base_extractor import BaseExtractor, ExtractorResult
from src.core.audio_utils import AudioUtils
from src.utils.prof import timeit

logger = logging.getLogger(__name__)


class ASRExtractor(BaseExtractor):
    """Экстрактор автоматического распознавания речи на основе Whisper."""
    
    name = "asr_extractor"
    version = "1.0.0"
    description = "Автоматическое распознавание речи с помощью Whisper"
    category = "speech"
    dependencies = ["torch", "whisper", "librosa"]
    estimated_duration = 5.0
    
    # Предпочитает GPU для Whisper
    gpu_required = False
    gpu_preferred = True
    gpu_memory_required = 1.0  # 1GB для Whisper
    
    def __init__(
        self, 
        device: str = "auto",
        model_size: str = "small",
        language: Optional[str] = None,
        task: str = "transcribe",
        sample_rate: int = 16000
    ):
        """
        Инициализация ASR экстрактора.
        
        Args:
            device: Устройство для обработки
            model_size: Размер модели Whisper ('tiny', 'base', 'small', 'medium', 'large')
            language: Язык для распознавания (None для автоопределения)
            task: Тип задачи ('transcribe' или 'translate')
            sample_rate: Частота дискретизации
        """
        super().__init__(device=device)
        
        self.model_size = model_size
        self.language = language
        self.task = task
        self.sample_rate = sample_rate
        
        self.audio_utils = AudioUtils(device=device, sample_rate=sample_rate)
        
        # Инициализируем модель Whisper
        self._setup_whisper_model()
    
    def _setup_whisper_model(self):
        """Настройка модели Whisper."""
        try:
            # Убираем шумные инфо-логи инициализации
            # self.logger.debug(f"Загрузка модели Whisper {self.model_size}...")
            
            # Загружаем модель на нужное устройство
            with timeit("asr: load_model"):
                self.model = whisper.load_model(self.model_size, device=self.device)
            try:
                print(f"[ASR] model={self.model_size} device={self.device} fp16={self.device=='cuda'} cuda_available={torch.cuda.is_available()}")
            except Exception:
                pass
            
            # Настраиваем опции декодирования
            self.decoding_options = whisper.DecodingOptions(
                language=self.language,
                task=self.task,
                fp16=self.device == "cuda"  # Используем fp16 на GPU для экономии памяти
            )
            
            # self.logger.debug(f"Модель Whisper {self.model_size} загружена на {self.device}")
            
        except Exception as e:
            self.logger.error(f"Ошибка загрузки модели Whisper: {e}")
            raise
    
    def run(self, input_uri: str, tmp_path: str) -> ExtractorResult:
        """
        Извлечение транскрипции речи.
        
        Args:
            input_uri: Путь к аудио файлу
            tmp_path: Временная директория
            
        Returns:
            ExtractorResult с транскрипцией и сегментами
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
            with timeit("asr: load_audio"):
                waveform, sample_rate = self.audio_utils.load_audio(input_uri, self.sample_rate)
            # Полная длительность аудио (сек)
            try:
                if hasattr(waveform, "shape"):
                    total_samples = int(waveform.shape[-1])
                else:
                    # numpy-like
                    total_samples = int(getattr(waveform, "size", 0))
                audio_duration = float(total_samples) / float(sample_rate or self.sample_rate or 1)
            except Exception:
                audio_duration = 0.0
            
            # Нормализуем аудио
            with timeit("asr: normalize_audio"):
                waveform = self.audio_utils.normalize_audio(waveform)
            
            # Перемещаем на нужное устройство
            with timeit("asr: move_to_device"):
                waveform = self.audio_utils._move_to_device(waveform)
            
            # Извлекаем транскрипцию
            with timeit("asr: whisper.transcribe"):
                transcription_result = self._extract_transcription(waveform)
            
            # Обрабатываем результат
            with timeit("asr: process_transcription_result"):
                processed_result = self._process_transcription_result(transcription_result)
            
            processing_time = time.time() - start_time
            
            # Создаем результат
            payload = {
                "transcription": processed_result["text"],
                "segments": processed_result["segments"],
                "language": processed_result["language"],
                "language_probability": processed_result["language_probability"],
                # speech_duration — суммарная длительность речи (по сегментам Whisper)
                "speech_duration": processed_result["duration"],
                # audio_duration — полная длительность входного аудио
                "audio_duration": audio_duration,
                "model_size": self.model_size,
                "task": self.task,
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
            error_msg = f"Ошибка извлечения транскрипции: {str(e)}"
            self._log_extraction_error(input_uri, error_msg, processing_time)
            
            return self._create_result(
                success=False,
                error=error_msg,
                processing_time=processing_time
            )
    
    def _extract_transcription(self, waveform: torch.Tensor) -> Dict[str, Any]:
        """Извлечение транскрипции с помощью Whisper."""
        try:
            # Конвертируем в numpy для Whisper
            audio_np = self.audio_utils.to_numpy(waveform)
            
            # Убираем batch dimension если есть
            if audio_np.ndim > 1:
                audio_np = audio_np[0] if audio_np.shape[0] == 1 else audio_np
            
            # Применяем Whisper
            result = self.model.transcribe(
                audio_np,
                language=self.language,
                task=self.task,
                fp16=self.device == "cuda"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Ошибка извлечения транскрипции: {e}")
            raise
    
    def _process_transcription_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Обработка результата транскрипции."""
        try:
            # Извлекаем основную информацию
            text = result.get("text", "")
            language = result.get("language", "unknown")
            language_probability = result.get("language_probability", 0.0)
            
            # Обрабатываем сегменты
            segments = []
            for segment in result.get("segments", []):
                processed_segment = {
                    "start": float(segment.get("start", 0.0)),
                    "end": float(segment.get("end", 0.0)),
                    "text": segment.get("text", "").strip(),
                    "tokens": segment.get("tokens", []),
                    "temperature": segment.get("temperature", 0.0),
                    "avg_logprob": segment.get("avg_logprob", 0.0),
                    "compression_ratio": segment.get("compression_ratio", 0.0),
                    "no_speech_prob": segment.get("no_speech_prob", 0.0)
                }
                segments.append(processed_segment)
            
            # Вычисляем общую длительность
            duration = max([seg["end"] for seg in segments], default=0.0)
            
            return {
                "text": text,
                "segments": segments,
                "language": language,
                "language_probability": language_probability,
                "duration": duration
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка обработки результата транскрипции: {e}")
            return {
                "text": "",
                "segments": [],
                "language": "unknown",
                "language_probability": 0.0,
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
        """Получение информации о модели Whisper."""
        return {
            "model_size": self.model_size,
            "language": self.language,
            "task": self.task,
            "sample_rate": self.sample_rate,
            "device": self.device,
            "fp16_enabled": self.device == "cuda"
        }