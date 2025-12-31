"""
Комбинированный экстрактор для анализа речи с сопоставлением ASR и диаризации.
"""
import time
import logging
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

from src.core.base_extractor import BaseExtractor, ExtractorResult
from src.core.audio_utils import AudioUtils
from src.extractors.asr_extractor import ASRExtractor
from src.extractors.speaker_diarization_extractor import SpeakerDiarizationExtractor
from src.extractors.pitch_extractor import PitchExtractor
from src.utils.prof import timeit

logger = logging.getLogger(__name__)


class SpeechAnalysisExtractor(BaseExtractor):
    """Комбинированный экстрактор для анализа речи с сопоставлением ASR и диаризации."""
    
    name = "speech_analysis_extractor"
    version = "1.1.0"
    description = "Комбинированный анализ речи: ASR + диаризация + сопоставление"
    category = "speech"
    dependencies = ["torch", "whisper", "resemblyzer", "librosa", "scikit-learn"]
    estimated_duration = 8.0  # ASR + диаризация + сопоставление
    
    # Предпочитает GPU для обеих моделей
    gpu_required = False
    gpu_preferred = True
    gpu_memory_required = 1.5  # 1.5GB для обеих моделей
    
    def __init__(
        self, 
        device: str = "auto",
        asr_model_size: str = "small",
        asr_language: Optional[str] = None,
        asr_task: str = "transcribe",
        diarization_segment_duration: float = 2.0,
        diarization_min_speakers: int = 1,
        diarization_max_speakers: int = 10,
        pitch_fmin: float = 50.0,
        pitch_fmax: float = 2000.0,
        pitch_backend: str = "classic",
        pitch_enabled: bool = True,
        sample_rate: int = 16000,
        average_channels_for_asr: bool = True
    ):
        """
        Инициализация комбинированного экстрактора.
        
        Args:
            device: Устройство для обработки
            asr_model_size: Размер модели Whisper
            asr_language: Язык для ASR
            asr_task: Тип задачи ASR
            diarization_segment_duration: Длительность сегмента для диаризации
            diarization_min_speakers: Минимальное количество спикеров
            diarization_max_speakers: Максимальное количество спикеров
            pitch_fmin: Минимальная частота для pitch анализа
            pitch_fmax: Максимальная частота для pitch анализа
            pitch_backend: Backend для pitch анализа
            sample_rate: Частота дискретизации
        """
        super().__init__(device=device)
        
        self.sample_rate = sample_rate
        
        # Инициализируем подэкстракторы
        self.asr_extractor = ASRExtractor(
            device=device,
            model_size=asr_model_size,
            language=asr_language,
            task=asr_task,
            sample_rate=sample_rate
        )
        
        self.diarization_extractor = SpeakerDiarizationExtractor(
            device=device,
            segment_duration=diarization_segment_duration,
            min_speakers=diarization_min_speakers,
            max_speakers=diarization_max_speakers,
            sample_rate=sample_rate
        )
        
        self.pitch_enabled = bool(pitch_enabled)
        self.pitch_extractor = PitchExtractor(
            device=device,
            sample_rate=sample_rate,
            fmin=pitch_fmin,
            fmax=pitch_fmax,
            backend=pitch_backend
        )
        
        self.audio_utils = AudioUtils(device=device, sample_rate=sample_rate)
        self.average_channels_for_asr = bool(average_channels_for_asr)
    
    def run(self, input_uri: str, tmp_path: str) -> ExtractorResult:
        """
        Комбинированный анализ речи.
        
        Args:
            input_uri: Путь к аудио файлу
            tmp_path: Временная директория
            
        Returns:
            ExtractorResult с полным анализом речи
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
            
            # Запускаем ASR с опциональным усреднением стерео каналов
            # self.logger.info("Запуск ASR анализа...")
            asr_start = time.time()
            if self.average_channels_for_asr:
                try:
                    with timeit("speech_analysis: load_audio for ASR"):
                        wav_t, sr = self.audio_utils.load_audio(input_uri, self.sample_rate)
                    y = self.audio_utils.to_numpy(wav_t)
                    if y.ndim == 2:
                        with timeit("speech_analysis: avg stereo->mono"):
                            y = np.mean(y, axis=0)
                    # сохраняем во временный WAV для корректной работы текущего ASRExtractor
                    import soundfile as sf
                    tmp_wav = str(Path(tmp_path) / f"{Path(input_uri).stem}_mono_avg.wav")
                    Path(tmp_path).mkdir(parents=True, exist_ok=True)
                    with timeit("speech_analysis: write tmp wav"):
                        sf.write(tmp_wav, y.astype(np.float32), sr)
                    with timeit("speech_analysis: ASR run"):
                        asr_result = self.asr_extractor.run(tmp_wav, tmp_path)
                except Exception:
                    # fallback на исходный файл
                    with timeit("speech_analysis: ASR run (fallback)"):
                        asr_result = self.asr_extractor.run(input_uri, tmp_path)
            else:
                with timeit("speech_analysis: ASR run"):
                    asr_result = self.asr_extractor.run(input_uri, tmp_path)
            asr_time = time.time() - asr_start
            
            if not asr_result.success:
                return self._create_result(
                    success=False,
                    error=f"ASR анализ не удался: {asr_result.error}",
                    processing_time=time.time() - start_time
                )
            
            # Запускаем диаризацию и pitch анализ параллельно
            # self.logger.info("Запуск диаризации спикеров и pitch анализа...")
            
            # Запускаем диаризацию
            diar_start = time.time()
            with timeit("speech_analysis: diarization run"):
                diarization_result = self.diarization_extractor.run(input_uri, tmp_path)
            diar_time = time.time() - diar_start
            
            if not diarization_result.success:
                return self._create_result(
                    success=False,
                    error=f"Диаризация не удалась: {diarization_result.error}",
                    processing_time=time.time() - start_time
                )
            
            # Запускаем pitch анализ (опционально)
            pitch_result = None
            pitch_start = time.time()
            if self.pitch_enabled:
                with timeit("speech_analysis: pitch run"):
                    pitch_result = self.pitch_extractor.run(input_uri, tmp_path)
                if not pitch_result.success:
                    self.logger.warning(f"Pitch анализ не удался: {pitch_result.error}, продолжаем без него")
                    pitch_result = None
            pitch_time = time.time() - pitch_start if self.pitch_enabled else 0.0
            
            # Сопоставляем результаты
            # self.logger.info("Сопоставление ASR и диаризации...")
            aligned_result = self._align_asr_and_diarization(
                asr_result.payload,
                diarization_result.payload
            )

            processing_time = time.time() - start_time

            payload = {
                "asr_result": asr_result.payload,
                "diarization_result": diarization_result.payload,
                "pitch_result": pitch_result.payload if pitch_result else None,
                "aligned_speech": aligned_result,
                "total_processing_time": processing_time,
                "asr_processing_time": asr_result.processing_time or asr_time,
                "diarization_processing_time": diarization_result.processing_time or diar_time,
                "pitch_processing_time": pitch_time,
                "device_used": self.device
            }
            
            self._log_extraction_success(input_uri, processing_time)
            
            return self._create_result(
                success=True,
                payload=payload,
                processing_time=processing_time
            )
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Ошибка комбинированного анализа речи: {str(e)}"
            self._log_extraction_error(input_uri, error_msg, processing_time)
            
            return self._create_result(
                success=False,
                error=error_msg,
                processing_time=processing_time
            )
    
    def _align_asr_and_diarization(
        self, 
        asr_payload: Dict[str, Any], 
        diarization_payload: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Сопоставление результатов ASR и диаризации."""
        try:
            asr_segments = asr_payload.get("segments", [])
            speaker_segments = diarization_payload.get("speaker_segments", [])
            segment_duration = diarization_payload.get("segment_duration", 2.0)
            
            # Создаем результирующие сегменты с присвоенными спикерами
            aligned_segments = []
            
            for asr_segment in asr_segments:
                seg_start = asr_segment.get("start", 0.0)
                seg_end = asr_segment.get("end", 0.0)
                seg_text = asr_segment.get("text", "").strip()
                
                # Находим среднее время сегмента
                seg_mid = (seg_start + seg_end) / 2
                
                # Находим соответствующий сегмент диаризации
                speaker_id = self._find_speaker_for_time(
                    seg_mid, 
                    speaker_segments, 
                    segment_duration
                )
                # Попробуем достать уверенность диаризации, если доступна
                speaker_confidence = None
                try:
                    if 0 <= int(seg_mid // segment_duration) < len(speaker_segments):
                        speaker_confidence = speaker_segments[int(seg_mid // segment_duration)].get("confidence")
                except Exception:
                    speaker_confidence = None
                
                # Создаем выровненный сегмент
                aligned_segment = {
                    "start": seg_start,
                    "end": seg_end,
                    "text": seg_text,
                    "speaker_id": speaker_id,
                    "duration": seg_end - seg_start,
                    "confidence": asr_segment.get("avg_logprob", 0.0),
                    "no_speech_prob": asr_segment.get("no_speech_prob", 0.0),
                    "tokens": asr_segment.get("tokens", []),
                    "speaker_confidence": speaker_confidence
                }
                
                aligned_segments.append(aligned_segment)
            
            # Группируем по спикерам
            speaker_groups = self._group_segments_by_speaker(aligned_segments)
            
            # Вычисляем статистики
            stats = self._compute_speech_statistics(aligned_segments, speaker_groups)
            
            return {
                "aligned_segments": aligned_segments,
                "speaker_groups": speaker_groups,
                "statistics": stats,
                "total_speakers": len(speaker_groups),
                "total_segments": len(aligned_segments)
            }
            
        except Exception as e:
            self.logger.error(f"Ошибка сопоставления ASR и диаризации: {e}")
            return {
                "aligned_segments": [],
                "speaker_groups": {},
                "statistics": {},
                "total_speakers": 0,
                "total_segments": 0
            }
    
    def _find_speaker_for_time(
        self, 
        time: float, 
        speaker_segments: List[Dict[str, Any]], 
        segment_duration: float
    ) -> int:
        """Поиск спикера для заданного времени."""
        try:
            # Вычисляем индекс сегмента диаризации
            segment_index = int(time // segment_duration)
            
            # Проверяем границы
            if 0 <= segment_index < len(speaker_segments):
                return speaker_segments[segment_index].get("speaker_id", 0)
            else:
                # Fallback к ближайшему сегменту
                if segment_index < 0:
                    return speaker_segments[0].get("speaker_id", 0) if speaker_segments else 0
                else:
                    return speaker_segments[-1].get("speaker_id", 0) if speaker_segments else 0
                    
        except Exception as e:
            self.logger.warning(f"Ошибка поиска спикера для времени {time}: {e}")
            return 0
    
    def _group_segments_by_speaker(
        self, 
        aligned_segments: List[Dict[str, Any]]
    ) -> Dict[int, List[Dict[str, Any]]]:
        """Группировка сегментов по спикерам."""
        try:
            speaker_groups = {}
            
            for segment in aligned_segments:
                speaker_id = segment.get("speaker_id", 0)
                
                if speaker_id not in speaker_groups:
                    speaker_groups[speaker_id] = []
                
                speaker_groups[speaker_id].append(segment)
            
            return speaker_groups
            
        except Exception as e:
            self.logger.error(f"Ошибка группировки сегментов по спикерам: {e}")
            return {}
    
    def _compute_speech_statistics(
        self, 
        aligned_segments: List[Dict[str, Any]], 
        speaker_groups: Dict[int, List[Dict[str, Any]]]
    ) -> Dict[str, Any]:
        """Вычисление статистик речи."""
        try:
            stats = {
                "total_duration": 0.0,
                "total_words": 0,
                "speaker_stats": {},
                "speech_activity": {},
                "confidence_stats": {}
            }
            
            # Общие статистики
            total_duration = 0.0
            total_words = 0
            confidences = []
            
            for segment in aligned_segments:
                duration = segment.get("duration", 0.0)
                text = segment.get("text", "")
                confidence = segment.get("confidence", 0.0)
                
                total_duration += duration
                total_words += len(text.split())
                confidences.append(confidence)
            
            stats["total_duration"] = total_duration
            stats["total_words"] = total_words
            stats["confidence_stats"] = {
                "mean": np.mean(confidences) if confidences else 0.0,
                "std": np.std(confidences) if confidences else 0.0,
                "min": np.min(confidences) if confidences else 0.0,
                "max": np.max(confidences) if confidences else 0.0
            }
            
            # Статистики по спикерам
            for speaker_id, segments in speaker_groups.items():
                speaker_duration = sum(seg.get("duration", 0.0) for seg in segments)
                speaker_words = sum(len(seg.get("text", "").split()) for seg in segments)
                speaker_confidences = [seg.get("confidence", 0.0) for seg in segments]
                
                stats["speaker_stats"][speaker_id] = {
                    "duration": speaker_duration,
                    "words": speaker_words,
                    "segments_count": len(segments),
                    "speech_percentage": (speaker_duration / total_duration * 100) if total_duration > 0 else 0.0,
                    "average_confidence": np.mean(speaker_confidences) if speaker_confidences else 0.0
                }
            
            # Активность речи (временные интервалы)
            speech_intervals = []
            for segment in aligned_segments:
                if segment.get("text", "").strip():
                    speech_intervals.append({
                        "start": segment.get("start", 0.0),
                        "end": segment.get("end", 0.0),
                        "speaker_id": segment.get("speaker_id", 0)
                    })
            
            stats["speech_activity"] = {
                "intervals": speech_intervals,
                "total_intervals": len(speech_intervals)
            }
            
            return stats
            
        except Exception as e:
            self.logger.error(f"Ошибка вычисления статистик речи: {e}")
            return {
                "total_duration": 0.0,
                "total_words": 0,
                "speaker_stats": {},
                "speech_activity": {},
                "confidence_stats": {}
            }
    
    def _validate_input(self, input_uri: str) -> bool:
        """Валидация входного файла."""
        if not super()._validate_input(input_uri):
            return False
        
        # Проверяем, что это аудио/видео файл
        media_extensions = {'.wav', '.mp3', '.flac', '.m4a', '.mp4', '.avi', '.mov'}
        if not any(input_uri.lower().endswith(ext) for ext in media_extensions):
            self.logger.error(f"Файл не является поддерживаемым медиа форматом: {input_uri}")
            return False
        
        return True
    
    def get_extractors_info(self) -> Dict[str, Any]:
        """Получение информации о подэкстракторах."""
        return {
            "asr_extractor": self.asr_extractor.get_model_info(),
            "diarization_extractor": self.diarization_extractor.get_encoder_info(),
            "pitch_extractor": {
                "fmin": self.pitch_extractor.fmin,
                "fmax": self.pitch_extractor.fmax,
                "backend": self.pitch_extractor.backend,
                "sample_rate": self.pitch_extractor.sample_rate,
                "device": self.pitch_extractor.device
            },
            "device": self.device,
            "sample_rate": self.sample_rate
        }
