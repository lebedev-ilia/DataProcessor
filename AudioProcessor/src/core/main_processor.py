"""
Основной процессор для координации работы экстракторов.
"""
import os
import time
import logging
import warnings
from typing import Dict, Any, List, Optional, Union
import threading
import psutil
from pathlib import Path
import asyncio
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

from .base_extractor import BaseExtractor, ExtractorResult
from .audio_utils import AudioUtils

logger = logging.getLogger(__name__)

# Suppress noisy torch meshgrid warning about upcoming indexing arg requirement
try:
    warnings.filterwarnings(
        "ignore",
        message=r".*torch\.meshgrid:.*indexing.*",
        category=UserWarning,
        module=r"torch\.functional"
    )
except Exception:
    pass


class MainProcessor:
    """Основной процессор для обработки видео и извлечения аудио признаков."""
    
    def __init__(
        self,
        device: str = "auto",
        max_workers: int = 4,
        gpu_memory_limit: float = 0.8,
        sample_rate: int = 22050,
        asr_model_size: str = "small",
        diarization_model_size: str = "small",
        emotion_model_size: str = "small",
        source_separation_model_size: str = "small",
        speech_analysis_pitch_enabled: bool = False,
        save_debug_results: bool = False,
        enabled_extractors: Optional[List[str]] = None,
        write_legacy_manifest: bool = True,
    ):
        """
        Инициализация основного процессора.
        
        Args:
            device: Устройство для обработки ('cuda', 'cpu', 'auto')
            max_workers: Максимальное количество воркеров
            gpu_memory_limit: Лимит памяти GPU (0.0-1.0)
            sample_rate: Частота дискретизации
        """
        self.device = device
        self.max_workers = max_workers
        self.gpu_memory_limit = gpu_memory_limit
        self.sample_rate = sample_rate
        self.asr_model_size = str(asr_model_size or "small")
        self.diarization_model_size = str(diarization_model_size or "small")
        self.emotion_model_size = str(emotion_model_size or "small")
        self.source_separation_model_size = str(source_separation_model_size or "small")
        self.speech_analysis_pitch_enabled = bool(speech_analysis_pitch_enabled)
        self.save_debug_results = save_debug_results
        self.write_legacy_manifest = bool(write_legacy_manifest)
        
        self.logger = logging.getLogger(f"{__name__}.MainProcessor")
        self.audio_utils = AudioUtils(device=device, sample_rate=sample_rate)
        
        # Реестр экстракторов
        self.extractors: Dict[str, BaseExtractor] = {}
        
        # Инициализируем только запрошенные экстракторы (или все по умолчанию)
        self._initialize_extractors(enabled_extractors)

    # -------------------- System Monitor --------------------
    class _SystemMonitor:
        def __init__(self, logger: logging.Logger, sample_interval: float = 5,
                     cpu_threshold: float = 95.0, ram_threshold: float = 95.0,
                     gpu_threshold: float = 95.0):
            self.logger = logger
            self.sample_interval = sample_interval
            self.cpu_thr = cpu_threshold
            self.ram_thr = ram_threshold
            self.gpu_thr = gpu_threshold
            self._stop_event = threading.Event()
            self._thread: Optional[threading.Thread] = None
            self.exceeded: Optional[str] = None

            # maxima
            self.max_cpu: float = 0.0
            self.max_ram: float = 0.0
            self.max_gpu_util: float = 0.0
            self.max_gpu_mem_pct: float = 0.0

        def _get_gpu_stats(self) -> Dict[str, float]:
            util = 0.0
            mem_pct = 0.0
            try:
                import torch
                if torch.cuda.is_available():
                    try:
                        import pynvml  # type: ignore
                        pynvml.nvmlInit()
                        h = pynvml.nvmlDeviceGetHandleByIndex(0)
                        util_s = pynvml.nvmlDeviceGetUtilizationRates(h)
                        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
                        util = float(util_s.gpu)
                        mem_pct = float(mem.used) / float(mem.total) * 100.0 if mem.total else 0.0
                        pynvml.nvmlShutdown()
                    except Exception:
                        # Fallback by memory percent only
                        total = torch.cuda.get_device_properties(0).total_memory
                        used = torch.cuda.memory_allocated(0)
                        mem_pct = float(used) / float(total) * 100.0 if total else 0.0
                        util = 0.0
            except Exception:
                pass
            return {"gpu_util": util, "gpu_mem_pct": mem_pct}

        def _loop(self):
            # Prime cpu_percent
            try:
                psutil.cpu_percent(interval=None)
            except Exception:
                pass
            while not self._stop_event.wait(self.sample_interval):
                try:
                    cpu = float(psutil.cpu_percent(interval=None))
                    ram = float(psutil.virtual_memory().percent)
                    gpu_stats = self._get_gpu_stats()
                    gpu_util = gpu_stats["gpu_util"]
                    gpu_mem_pct = gpu_stats["gpu_mem_pct"]

                    # update maxima
                    if cpu > self.max_cpu:
                        self.max_cpu = cpu
                    if ram > self.max_ram:
                        self.max_ram = ram
                    if gpu_util > self.max_gpu_util:
                        self.max_gpu_util = gpu_util
                    if gpu_mem_pct > self.max_gpu_mem_pct:
                        self.max_gpu_mem_pct = gpu_mem_pct

                    # periodic info
                    self.logger.info(
                        f"SYS cpu={cpu:.1f}% ram={ram:.1f}% gpu_util={gpu_util:.1f}% gpu_mem={gpu_mem_pct:.1f}%"
                    )

                    # threshold check
                    if self.exceeded is None:
                        if cpu >= self.cpu_thr:
                            self.exceeded = f"CPU usage {cpu:.1f}% >= {self.cpu_thr:.1f}%"
                        elif ram >= self.ram_thr:
                            self.exceeded = f"RAM usage {ram:.1f}% >= {self.ram_thr:.1f}%"
                        elif gpu_util >= self.gpu_thr and gpu_util > 0:
                            self.exceeded = f"GPU util {gpu_util:.1f}% >= {self.gpu_thr:.1f}%"
                        elif gpu_mem_pct >= self.gpu_thr and gpu_mem_pct > 0:
                            self.exceeded = f"GPU mem {gpu_mem_pct:.1f}% >= {self.gpu_thr:.1f}%"

                except Exception:
                    # do not break monitoring on transient errors
                    pass

        def start(self):
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._loop, name="SystemMonitor", daemon=True)
            self._thread.start()

        def stop(self):
            self._stop_event.set()
            if self._thread is not None:
                try:
                    self._thread.join(timeout=1.0)
                except Exception:
                    pass
    
    def _initialize_extractors(self, enabled_extractors: Optional[List[str]] = None):
        """Инициализация только выбранных экстракторов (ленивая загрузка модулей)."""
        try:
            # Фабрики экстракторов: импорт внутри, чтобы не грузить лишние зависимости
            extractor_factories: Dict[str, Any] = {
                "video_audio": lambda: __import__(
                    "src.extractors.video_audio_extractor", fromlist=["VideoAudioExtractor"]
                ).VideoAudioExtractor(device=self.device),
                "mfcc": lambda: __import__(
                    "src.extractors.mfcc_extractor", fromlist=["MFCCExtractor"]
                ).MFCCExtractor(device=self.device, sample_rate=self.sample_rate),
                "mel": lambda: __import__(
                    "src.extractors.mel_extractor", fromlist=["MelExtractor"]
                ).MelExtractor(device=self.device, sample_rate=self.sample_rate),
                "clap": lambda: __import__(
                    "src.extractors.clap_extractor", fromlist=["CLAPExtractor"]
                ).CLAPExtractor(device=self.device, sample_rate=48000),
                "asr": lambda: __import__(
                    "src.extractors.asr_extractor", fromlist=["ASRExtractor"]
                ).ASRExtractor(device=self.device, model_size=self.asr_model_size, sample_rate=16000),
                "speaker_diarization": lambda: __import__(
                    "src.extractors.speaker_diarization_extractor", fromlist=["SpeakerDiarizationExtractor"]
                ).SpeakerDiarizationExtractor(
                    device=self.device,
                    model_size=self.diarization_model_size,
                    sample_rate=16000,
                ),
                "tempo": lambda: __import__(
                    "src.extractors.tempo_extractor", fromlist=["TempoExtractor"]
                ).TempoExtractor(device=self.device, sample_rate=self.sample_rate),
                "loudness": lambda: __import__(
                    "src.extractors.loudness_extractor", fromlist=["LoudnessExtractor"]
                ).LoudnessExtractor(device=self.device, sample_rate=self.sample_rate),
                "onset": lambda: __import__(
                    "src.extractors.onset_extractor", fromlist=["OnsetExtractor"]
                ).OnsetExtractor(device=self.device, sample_rate=self.sample_rate),
                "chroma": lambda: __import__(
                    "src.extractors.chroma_extractor", fromlist=["ChromaExtractor"]
                ).ChromaExtractor(device=self.device, sample_rate=self.sample_rate),
                "spectral": lambda: __import__(
                    "src.extractors.spectral_extractor", fromlist=["SpectralExtractor"]
                ).SpectralExtractor(device=self.device, sample_rate=self.sample_rate),
                "quality": lambda: __import__(
                    "src.extractors.quality_extractor", fromlist=["QualityExtractor"]
                ).QualityExtractor(device=self.device, sample_rate=self.sample_rate),
                "rhythmic": lambda: __import__(
                    "src.extractors.rhythmic_extractor", fromlist=["RhythmicExtractor"]
                ).RhythmicExtractor(device=self.device, sample_rate=self.sample_rate),
                "voice_quality": lambda: __import__(
                    "src.extractors.voice_quality_extractor", fromlist=["VoiceQualityExtractor"]
                ).VoiceQualityExtractor(device=self.device, sample_rate=self.sample_rate),
                "hpss": lambda: __import__(
                    "src.extractors.hpss_extractor", fromlist=["HPSSExtractor"]
                ).HPSSExtractor(device=self.device, sample_rate=self.sample_rate),
                "key": lambda: __import__(
                    "src.extractors.key_extractor", fromlist=["KeyExtractor"]
                ).KeyExtractor(device=self.device, sample_rate=self.sample_rate),
                "band_energy": lambda: __import__(
                    "src.extractors.band_energy_extractor", fromlist=["BandEnergyExtractor"]
                ).BandEnergyExtractor(device=self.device, sample_rate=self.sample_rate),
                "spectral_entropy": lambda: __import__(
                    "src.extractors.spectral_entropy_extractor", fromlist=["SpectralEntropyExtractor"]
                ).SpectralEntropyExtractor(device=self.device, sample_rate=self.sample_rate),
                "source_separation": lambda: __import__(
                    "src.extractors.source_separation_extractor", fromlist=["SourceSeparationExtractor"]
                ).SourceSeparationExtractor(device=self.device, model_size=self.source_separation_model_size, batch_size=8),
                "emotion_diarization": lambda: __import__(
                    "src.extractors.emotion_diarization_extractor", fromlist=["EmotionDiarizationExtractor"]
                ).EmotionDiarizationExtractor(device=self.device, model_size=self.emotion_model_size, sample_rate=16000, batch_size=16),
                "speech_analysis": lambda: __import__(
                    "src.extractors.speech_analysis_extractor", fromlist=["SpeechAnalysisExtractor"]
                ).SpeechAnalysisExtractor(
                    device=self.device,
                    asr_model_size=self.asr_model_size,
                    diarization_model_size=self.diarization_model_size,
                    sample_rate=16000,
                    pitch_enabled=bool(self.speech_analysis_pitch_enabled),
                    pitch_backend=("torchcrepe" if str(self.device).lower() == "cuda" else "classic"),
                ),
            }

            requested = list(extractor_factories.keys()) if enabled_extractors is None else list(enabled_extractors)
            # Обрабатываем виртуальные экстракторы: 'pitch' реализуется внутри 'speech_analysis'
            if enabled_extractors is not None:
                # Виртуальные экстракторы реализуются внутри speech_analysis
                virtuals = ["pitch", "asr", "speaker_diarization"]
                ensure_sa = any(v in requested for v in virtuals)
                if ensure_sa and "speech_analysis" not in requested:
                    requested.append("speech_analysis")
                # Убираем виртуальные из инициализации, они будут опубликованы после run(speech_analysis)
                # If an extractor exists as a real factory (e.g., asr / speaker_diarization), do NOT treat it as virtual.
                requested = [n for n in requested if not (n in virtuals and n not in extractor_factories)]

            self.extractors = {}
            for name in requested:
                factory = extractor_factories.get(name)
                if factory is None:
                    # Тихо пропускаем отсутствующие фабрики (например, виртуальные, уже обработанные)
                    continue
                try:
                    t0 = time.time()
                    self.extractors[name] = factory()
                    t1 = time.time()
                    try:
                        print(f"[TIMER] init extractor {name}: {t1 - t0:.3f}s")
                    except Exception:
                        pass
                except Exception as err:
                    self.logger.error(f"Не удалось инициализировать экстрактор {name}: {err}")
            
        except Exception as e:
            self.logger.error(f"Ошибка инициализации экстракторов: {e}")
            raise
    
    def process_video(
        self,
        video_path: str,
        output_dir: str,
        extractor_names: Optional[List[str]] = None,
        extract_audio: bool = True
    ) -> Dict[str, Any]:
        """
        Обработка видео файла с извлечением аудио и признаков.
        
        Args:
            video_path: Путь к видео файлу
            output_dir: Директория для сохранения результатов
            extractor_names: Список экстракторов для запуска
            extract_audio: Извлекать ли аудио из видео
            
        Returns:
            Словарь с результатами обработки
        """
        start_time = time.time()
        
        try:
            # Старт системного мониторинга
            # monitor = MainProcessor._SystemMonitor(self.logger, sample_interval=10,
            #                                        cpu_threshold=95.0, ram_threshold=95.0, gpu_threshold=95.0)
            # monitor.start()
            # Создаем выходную директорию
            t_mkdir0 = time.time()
            os.makedirs(output_dir, exist_ok=True)
            t_mkdir1 = time.time()
            
            # self.logger.debug(f"Начинаем обработку видео: {video_path}")
            
            results = {
                "video_path": video_path,
                "output_dir": output_dir,
                "success": False,
                "extracted_audio_path": None,
                "extractor_results": {},
                "processing_time": 0.0,
                "errors": [],
                "timings": {
                    "wall_clock": {
                        "start_ts": start_time,
                        "start_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(start_time)),
                        "end_ts": None,
                        "end_iso": None,
                        "elapsed_s": None,
                    },
                    "mkdir_ms": float((t_mkdir1 - t_mkdir0) * 1000.0),
                    "audio_extract_ms": 0.0,
                    "per_extractor_wall_ms": {},
                    "per_extractor_reported_ms": {},
                    "save_manifest_ms": 0.0,
                }
            }
            
            # Шаг 1: Извлечение аудио из видео (если нужно)
            audio_path = video_path
            if extract_audio and self._is_video_file(video_path):
                # self.logger.debug("Извлекаем аудио из видео...")
                
                audio_filename = f"{Path(video_path).stem}_extracted_audio.wav"
                audio_output_path = os.path.join(output_dir, audio_filename)
                
                try:
                    # Извлекаем сразу в 48kHz, чтобы избежать последующего ресемплинга для CLAP
                    target_sr = 48000
                    t_a0 = time.time()
                    audio_path = self.audio_utils.extract_audio_from_video(
                        video_path=video_path,
                        output_path=audio_output_path,
                        target_sr=target_sr,
                    )
                    t_a1 = time.time()
                    results["timings"]["audio_extract_ms"] = float((t_a1 - t_a0) * 1000.0)
                    results["extracted_audio_path"] = audio_path
                    # self.logger.debug(f"Аудио извлечено: {audio_path}")
                    
                except Exception as e:
                    error_msg = f"Ошибка извлечения аудио: {e}"
                    self.logger.error(error_msg)
                    results["errors"].append(error_msg)
                    return results
            
            # Шаг 2: Запуск экстракторов
            if extractor_names is None:
                # Запускаем все доступные экстракторы
                extractor_names = list(self.extractors.keys())
            
            # Поддержка виртуального экстрактора 'pitch', который считается внутри 'speech_analysis'
            requested_names = list(extractor_names)
            wants_virtual_pitch = "pitch" in requested_names and "pitch" not in self.extractors
            wants_virtual_asr = "asr" in requested_names and "asr" not in self.extractors
            wants_virtual_speaker = "speaker_diarization" in requested_names and "speaker_diarization" not in self.extractors
            if wants_virtual_pitch:
                # Убираем 'pitch', чтобы не было предупреждения, и гарантируем запуск speech_analysis
                requested_names = [n for n in requested_names if n != "pitch"]
                if "speech_analysis" not in requested_names:
                    requested_names.append("speech_analysis")
            
            # Скрываем подробный список экстракторов в логах запуска
            # self.logger.debug(f"Запускаем экстракторы: {extractor_names}")
            
            # Запускаем экстракторы последовательно (можно распараллелить)
            extractor_times = {}
            for extractor_name in requested_names:
                if extractor_name not in self.extractors:
                    self.logger.warning(f"Экстрактор {extractor_name} пропущен (не доступен)")
                    continue
                
                try:
                    extractor = self.extractors[extractor_name]
                    
                    # Для video_audio экстрактора используем оригинальный видео файл
                    # Для остальных - извлеченный аудио файл
                    input_path = video_path if extractor_name == "video_audio" else audio_path
                    
                    t_e0 = time.time()
                    extractor_result = extractor.run(input_path, output_dir)
                    t_e1 = time.time()
                    
                    results["extractor_results"][extractor_name] = {
                        "success": extractor_result.success,
                        "payload": extractor_result.payload,
                        "error": extractor_result.error,
                        "processing_time": extractor_result.processing_time,
                        "device_used": extractor_result.device_used
                    }
                    
                    # Сохраняем время выполнения
                    extractor_times[extractor_name] = extractor_result.processing_time
                    results["timings"]["per_extractor_wall_ms"][extractor_name] = float((t_e1 - t_e0) * 1000.0)
                    results["timings"]["per_extractor_reported_ms"][extractor_name] = float((extractor_result.processing_time or 0.0) * 1000.0)
                    
                    if not extractor_result.success:
                        error_msg = f"❌ {extractor_name} не удался: {extractor_result.error}"
                        self.logger.error(error_msg)
                        results["errors"].append(error_msg)
                    
                    # Виртуальные результаты из speech_analysis
                    if extractor_name == "speech_analysis" and extractor_result.success:
                        payload = extractor_result.payload or {}
                        # pitch
                        if wants_virtual_pitch:
                            try:
                                pitch_payload = payload.get("pitch_result") or {}
                                pitch_time = float(payload.get("pitch_processing_time") or 0.0)
                                pitch_success = bool(pitch_payload) and (
                                    (pitch_payload.get("f0_count_pyin") or 0) > 0
                                    or (pitch_payload.get("f0_count_yin") or 0) > 0
                                    or (pitch_payload.get("f0_mean") or 0.0) > 0.0
                                )
                                results["extractor_results"]["pitch"] = {
                                    "success": pitch_success,
                                    "payload": pitch_payload if pitch_success else None,
                                    "error": None if pitch_success else "pitch empty/zero values",
                                    "processing_time": pitch_time,
                                    "device_used": payload.get("device_used", "unknown"),
                                }
                                extractor_times["pitch"] = pitch_time
                            except Exception:
                                pass
                        # asr (всегда добавляем, если есть в payload)
                        if wants_virtual_asr or (payload.get("asr_result") is not None):
                            try:
                                asr_time = float(payload.get("asr_processing_time") or 0.0)
                                # Забираем исходный результат ASR из speech_analysis
                                asr_src = payload.get("asr_result") or {}
                                asr_payload = {
                                    "transcription": asr_src.get("transcription", ""),
                                    "language": asr_src.get("language", "unknown"),
                                    "language_probability": asr_src.get("language_probability", 0.0),
                                    # Совместимость: если старое поле duration есть, трактуем как speech_duration
                                    "speech_duration": float(asr_src.get("speech_duration", asr_src.get("duration", 0.0)) or 0.0),
                                    "audio_duration": float(asr_src.get("audio_duration", 0.0) or 0.0),
                                    "model_size": asr_src.get("model_size", "unknown"),
                                    "task": asr_src.get("task", "transcribe"),
                                    "segments": asr_src.get("segments", []) or [],
                                    "sample_rate": asr_src.get("sample_rate", 0) or 0,
                                    # device_used может отсутствовать в asr_src; пробрасываем из speech_analysis
                                    "device_used": asr_src.get("device_used") or payload.get("device_used", "unknown"),
                                }
                                # Подстрахуемся статистикой из aligned_speech
                                aligned = payload.get("aligned_speech") or {}
                                stats = aligned.get("statistics") or {}
                                if not asr_payload["duration"]:
                                    asr_payload["duration"] = float(stats.get("total_duration", 0.0) or 0.0)
                                asr_success = (
                                    bool(asr_payload.get("transcription"))
                                    or len(asr_payload.get("segments", []) or []) > 0
                                    or (asr_payload.get("speech_duration", 0.0) or 0.0) > 0.0
                                )
                                results["extractor_results"]["asr"] = {
                                    "success": asr_success,
                                    "payload": asr_payload if asr_success else None,
                                    "error": None if asr_success else "asr data unavailable in speech_analysis payload",
                                    "processing_time": asr_time,
                                    "device_used": asr_payload.get("device_used", payload.get("device_used", "unknown")),
                                }
                                extractor_times["asr"] = asr_time
                            except Exception:
                                pass
                        # speaker_diarization
                        if wants_virtual_speaker:
                            try:
                                diar_time = float(payload.get("diarization_processing_time") or 0.0)
                                aligned = payload.get("aligned_speech") or {}
                                stats = aligned.get("statistics") or {}
                                diar_src = payload.get("diarization_result") or {}
                                sp_payload = {
                                    "speaker_count": int(aligned.get("total_speakers", 0) or diar_src.get("speaker_count", 0) or 0),
                                    "segment_duration": float(diar_src.get("segment_duration", 0.0) or 0.0),
                                    "clustering_method": diar_src.get("clustering_method", "unknown") or "unknown",
                                    # duration в speaker_diarization — используем длительность из самого диаризационного экстрактора (полная длительность аудио, если так задано там)
                                    "duration": float(diar_src.get("duration", stats.get("total_duration", 0.0)) or 0.0),
                                    "speaker_segments": diar_src.get("speaker_segments", []) or [],
                                    "speaker_stats": (stats.get("speaker_stats") if isinstance(stats.get("speaker_stats"), dict) else diar_src.get("speaker_stats")),
                                    "device_used": payload.get("device_used", "unknown"),
                                    "sample_rate": payload.get("sample_rate", 0) or 0,
                                }
                                sp_success = (sp_payload["speaker_count"] > 0) or (len(sp_payload.get("speaker_segments", []) or []) > 0) or (sp_payload["duration"] > 0.0)
                                results["extractor_results"]["speaker_diarization"] = {
                                    "success": sp_success,
                                    "payload": sp_payload if sp_success else None,
                                    "error": None if sp_success else "speaker diarization data unavailable in speech_analysis payload",
                                    "processing_time": diar_time,
                                    "device_used": payload.get("device_used", "unknown"),
                                }
                                extractor_times["speaker_diarization"] = diar_time
                            except Exception:
                                pass
                        
                except Exception as e:
                    error_msg = f"Ошибка в экстракторе {extractor_name}: {e}"
                    self.logger.error(error_msg)
                    results["errors"].append(error_msg)
                    
                    results["extractor_results"][extractor_name] = {
                        "success": False,
                        "payload": None,
                        "error": str(e),
                        "processing_time": 0.0,
                        "device_used": "unknown"
                    }
                    extractor_times[extractor_name] = 0.0

            # Fallback: гарантируем публикацию виртуального ASR, если он присутствует в speech_analysis
            try:
                if (
                    "asr" not in results["extractor_results"]
                    and "speech_analysis" in results["extractor_results"]
                    and results["extractor_results"]["speech_analysis"].get("success")
                ):
                    sa_payload = results["extractor_results"]["speech_analysis"].get("payload") or {}
                    asr_src = sa_payload.get("asr_result") or {}
                    if asr_src:
                        asr_time = float(sa_payload.get("asr_processing_time") or 0.0)
                        asr_payload = {
                            "transcription": asr_src.get("transcription", ""),
                            "language": asr_src.get("language", "unknown"),
                            "language_probability": asr_src.get("language_probability", 0.0),
                            "speech_duration": float(asr_src.get("speech_duration", asr_src.get("duration", 0.0)) or 0.0),
                            "audio_duration": float(asr_src.get("audio_duration", 0.0) or 0.0),
                            "model_size": asr_src.get("model_size", "unknown"),
                            "task": asr_src.get("task", "transcribe"),
                            "segments": asr_src.get("segments", []) or [],
                            "sample_rate": asr_src.get("sample_rate", 0) or 0,
                            "device_used": asr_src.get("device_used") or sa_payload.get("device_used", "unknown"),
                        }
                        asr_success = (
                            bool(asr_payload.get("transcription"))
                            or len(asr_payload.get("segments", []) or []) > 0
                            or (asr_payload.get("speech_duration", 0.0) or 0.0) > 0.0
                        )
                        results["extractor_results"]["asr"] = {
                            "success": asr_success,
                            "payload": asr_payload if asr_success else None,
                            "error": None if asr_success else "asr data unavailable in speech_analysis payload",
                            "processing_time": asr_time,
                            "device_used": asr_payload.get("device_used", sa_payload.get("device_used", "unknown")),
                        }
                        extractor_times["asr"] = asr_time
            except Exception:
                pass
            
            # Остановка монитора и добавление макс. значений в лог
            # Монитор сейчас отключен; оставляем заглушку
            try:
                monitor  # type: ignore[name-defined]
            except Exception:
                pass
            
            # Вычисляем время обработки перед сохранением
            end_time_total = time.time()
            results["processing_time"] = end_time_total - start_time
            results["timings"]["wall_clock"]["end_ts"] = end_time_total
            results["timings"]["wall_clock"]["end_iso"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(end_time_total))
            results["timings"]["wall_clock"]["elapsed_s"] = float(end_time_total - start_time)
            
            # Определяем общий успех
            successful_extractors = sum(
                1 for result in results["extractor_results"].values() 
                if result["success"]
            )
            
            # Считаем только доступные экстракторы (те, которые реально запускались)
            available_extractors = len(results["extractor_results"])
            
            results["success"] = len(results["errors"]) == 0 and successful_extractors > 0
            
            # Шаг 3: Сохранение результатов
            if self.write_legacy_manifest:
                t_s0 = time.time()
                self._save_results(results, output_dir)
                t_s1 = time.time()
                results["timings"]["save_manifest_ms"] = float((t_s1 - t_s0) * 1000.0)
            
            # Выводим время выполнения по экстракторам
            if extractor_times:
                # self.logger.info("⏱️ Время выполнения экстракторов:")
                for name, time_taken in sorted(extractor_times.items(), key=lambda x: x[1], reverse=True):
                    status = "✅" if results["extractor_results"][name]["success"] else "❌"
                    # self.logger.info(f"  {status} {name}: {time_taken:.2f}s")
                
                # Добавляем время ASR из speech_analysis если оно есть
                if "speech_analysis" in results["extractor_results"] and results["extractor_results"]["speech_analysis"]["success"]:
                    speech_payload = results["extractor_results"]["speech_analysis"].get("payload", {})
                    asr_time = speech_payload.get("asr_processing_time", 0.0)
                    if asr_time > 0:
                        pass
                        # self.logger.info(f"  📝 asr (в speech_analysis): {asr_time:.2f}s")
                
                total_extractor_time = sum(extractor_times.values())
                # self.logger.info(f"📊 Суммарное время экстракторов: {total_extractor_time:.2f}s")
            
            # Максимальные значения ресурсов за время обработки
            # Логи пиков ресурсов оставляем включаемыми позже (монитор отключен)
            try:
                monitor  # type: ignore[name-defined]
            except Exception:
                pass
            
            # self.logger.info(f"🎯 Общее время обработки: {results['processing_time']:.2f}s")
            # self.logger.info(f"✅ Успешных экстракторов: {successful_extractors}/{available_extractors}")
            
            return results
            
        except Exception as e:
            processing_time = time.time() - start_time
            error_msg = f"Критическая ошибка обработки: {e}"
            self.logger.error(error_msg)
            # Пытаемся остановить монитор
            try:
                monitor  # type: ignore[name-defined]
            except Exception:
                pass
            
            return {
                "video_path": video_path,
                "output_dir": output_dir,
                "success": False,
                "extracted_audio_path": None,
                "extractor_results": {},
                "processing_time": processing_time,
                "errors": [error_msg]
            }
    
    def _is_video_file(self, file_path: str) -> bool:
        """Проверка, является ли файл видео."""
        video_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v', '.3gp', '.ogv'}
        return any(file_path.lower().endswith(ext) for ext in video_extensions)
    
    def _save_results(self, results: Dict[str, Any], output_dir: str):
        """Сохранение результатов в формате manifest."""
        try:
            import json
            import numpy as np
            from datetime import datetime
            
            # Функция для конвертации numpy arrays в списки
            def convert_numpy(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, dict):
                    return {key: convert_numpy(value) for key, value in obj.items()}
                elif isinstance(obj, list):
                    return [convert_numpy(item) for item in obj]
                return obj
            
            # Создаем manifest в формате старой версии
            video_id = os.path.basename(results["video_path"]).split('.')[0]
            
            # Конвертируем результаты экстракторов в формат manifest
            extractor_results = []
            for extractor_name, result in results["extractor_results"].items():
                if result["success"] and result["payload"]:
                    # Создаем плоские признаки из payload
                    flat_payload = self._flatten_payload(result["payload"], extractor_name)
                    
                    extractor_result = {
                        "name": extractor_name,
                        "version": "1.0.0",
                        "success": True,
                        "payload": convert_numpy(flat_payload),
                        "error": None,
                        "processing_time": result["processing_time"]
                    }
                else:
                    extractor_result = {
                        "name": extractor_name,
                        "version": "1.0.0",
                        "success": False,
                        "payload": None,
                        "error": result["error"],
                        "processing_time": result["processing_time"]
                    }
                extractor_results.append(extractor_result)
            
            # Создаем manifest
            manifest = {
                "video_id": video_id,
                "task_id": f"audio_processor_{video_id}",
                "dataset": "audio_processor",
                "timestamp": datetime.utcnow().isoformat() + "Z",
                "extractors": extractor_results,
                "schema_version": "audio_manifest_v1",
                "total_processing_time": results["processing_time"],
                "manifest_uri": None
            }
            
            # Сохраняем manifest
            manifest_file = os.path.join(output_dir, f"{video_id}_manifest.json")
            with open(manifest_file, 'w', encoding='utf-8') as f:
                json.dump(manifest, f, indent=2, ensure_ascii=False)
            
            # self.logger.info(f"Manifest сохранен: {manifest_file}")
            
            # По умолчанию не сохраняем подробные отладочные результаты
            if getattr(self, "save_debug_results", False):
                results_file = os.path.join(output_dir, "processing_results.json")
                with open(results_file, 'w', encoding='utf-8') as f:
                    json.dump(convert_numpy(results), f, indent=2, ensure_ascii=False)
            
        except Exception as e:
            self.logger.error(f"Ошибка сохранения результатов: {e}")
    
    def _flatten_payload(self, payload: Dict[str, Any], extractor_name: str) -> Dict[str, Any]:
        """Преобразование payload в плоские признаки для manifest."""
        flat_payload: Dict[str, Any] = {}
        
        if extractor_name == "video_audio":
            # Для video_audio экстрактора
            if "audio_info" in payload:
                audio_info = payload["audio_info"]
                flat_payload.update({
                    "audio_duration": audio_info.get("duration", 0.0),
                    "audio_sample_rate": audio_info.get("sample_rate", 0),
                    "audio_channels": audio_info.get("channels", 0),
                    "audio_samples": audio_info.get("samples", 0)
                })
            
            if "video_info" in payload:
                video_info = payload["video_info"]
                flat_payload.update({
                    "video_duration": video_info.get("duration", 0.0),
                    "video_width": video_info.get("width", 0),
                    "video_height": video_info.get("height", 0),
                    "video_fps": video_info.get("fps", 0.0),
                    "video_codec": video_info.get("codec", "unknown")
                })
        
        elif extractor_name == "mfcc":
            # Для MFCC экстрактора
            if "mfcc_statistics" in payload:
                stats = payload["mfcc_statistics"]
                flat_payload.update({
                    "mfcc_mean": stats.get("mfcc_mean", []) if stats.get("mfcc_mean", None) is not None else [],
                    "mfcc_std": stats.get("mfcc_std", []) if stats.get("mfcc_std", None) is not None else [],
                    "mfcc_min": stats.get("mfcc_min", []) if stats.get("mfcc_min", None) is not None else [],
                    "mfcc_max": stats.get("mfcc_max", []) if stats.get("mfcc_max", None) is not None else [],
                    "delta_mean": stats.get("delta_mean", []) if stats.get("delta_mean", None) is not None else [],
                    "delta_std": stats.get("delta_std", []) if stats.get("delta_std", None) is not None else [],
                    "delta_delta_mean": stats.get("delta_delta_mean", []) if stats.get("delta_delta_mean", None) is not None else [],
                    "delta_delta_std": stats.get("delta_delta_std", []) if stats.get("delta_delta_std", None) is not None else [],
                    "total_features": stats.get("total_features", 0) if stats.get("total_features", None) is not None else 0
                })
        
        elif extractor_name == "mel":
            # Для Mel экстрактора - массивы в .npy файлах, в JSON только статистики и пути
            flat_payload.update({
                "mel_spectrogram_npy": payload.get("mel_spectrogram_npy", None),
                "mel_shape": payload.get("mel_shape", []) if payload.get("mel_shape", None) is not None else [],
                "mel_elements": payload.get("mel_elements", 0) if payload.get("mel_elements", None) is not None else 0,
                "sample_rate": payload.get("sample_rate", 0) if payload.get("sample_rate", None) is not None else 0,
                "n_fft": payload.get("n_fft", 0) if payload.get("n_fft", None) is not None else 0,
                "hop_length": payload.get("hop_length", 0) if payload.get("hop_length", None) is not None else 0,
                "n_mels": payload.get("n_mels", 0) if payload.get("n_mels", None) is not None else 0,
                "fmin": payload.get("fmin", 0.0) if payload.get("fmin", None) is not None else 0.0,
                "fmax": payload.get("fmax", 0.0) if payload.get("fmax", None) is not None else 0.0,
                "power": payload.get("power", 0.0) if payload.get("power", None) is not None else 0.0,
                "duration": payload.get("duration", 0.0) if payload.get("duration", None) is not None else 0.0,
                "device_used": payload.get("device_used", "unknown") if payload.get("device_used", None) is not None else "unknown",
                "total_features": payload.get("mel_elements", 0) if payload.get("mel_elements", None) is not None else 0,
                # Пути к .npy файлам с массивами
                "mel_mean_npy": payload.get("mel_mean_npy", None),
                "mel_std_npy": payload.get("mel_std_npy", None),
                "mel_min_npy": payload.get("mel_min_npy", None),
                "mel_max_npy": payload.get("mel_max_npy", None),
                "freq_mean_npy": payload.get("freq_mean_npy", None),
                "freq_std_npy": payload.get("freq_std_npy", None),
                "spectral_centroid_npy": payload.get("spectral_centroid_npy", None),
                "spectral_bandwidth_npy": payload.get("spectral_bandwidth_npy", None),
                "mel_stats_vector_npy": payload.get("mel_stats_vector_npy", None),
                # Размеры массивов
                "mel_mean_shape": payload.get("mel_mean_shape", []) if payload.get("mel_mean_shape", None) is not None else [],
                "mel_std_shape": payload.get("mel_std_shape", []) if payload.get("mel_std_shape", None) is not None else [],
                "mel_min_shape": payload.get("mel_min_shape", []) if payload.get("mel_min_shape", None) is not None else [],
                "mel_max_shape": payload.get("mel_max_shape", []) if payload.get("mel_max_shape", None) is not None else [],
                "freq_mean_shape": payload.get("freq_mean_shape", []) if payload.get("freq_mean_shape", None) is not None else [],
                "freq_std_shape": payload.get("freq_std_shape", []) if payload.get("freq_std_shape", None) is not None else [],
                "spectral_centroid_shape": payload.get("spectral_centroid_shape", []) if payload.get("spectral_centroid_shape", None) is not None else [],
                "spectral_bandwidth_shape": payload.get("spectral_bandwidth_shape", []) if payload.get("spectral_bandwidth_shape", None) is not None else [],
                "mel_stats_vector_shape": payload.get("mel_stats_vector_shape", []) if payload.get("mel_stats_vector_shape", None) is not None else [],
                # Статистики по массивам
                "mel_mean_stats": payload.get("mel_mean_stats", {}) if payload.get("mel_mean_stats", None) is not None else {},
                "mel_std_stats": payload.get("mel_std_stats", {}) if payload.get("mel_std_stats", None) is not None else {},
                "freq_mean_stats": payload.get("freq_mean_stats", {}) if payload.get("freq_mean_stats", None) is not None else {},
                "spectral_centroid_stats": payload.get("spectral_centroid_stats", {}) if payload.get("spectral_centroid_stats", None) is not None else {},
                "mel_stats_vector_stats": payload.get("mel_stats_vector_stats", {}) if payload.get("mel_stats_vector_stats", None) is not None else {}
            })
        
        elif extractor_name == "clap":
            # Для CLAP экстрактора - массивы статистик в .npy файлах, в JSON только скалярные статистики и пути
            flat_payload.update({
                "clap_embeddings_npy": payload.get("clap_embeddings_npy", None),
                "embeddings_shape": payload.get("embeddings_shape", []) if payload.get("embeddings_shape", None) is not None else [],
                "embeddings_dtype": payload.get("embeddings_dtype", "unknown") if payload.get("embeddings_dtype", None) is not None else "unknown",
                "embedding_dim": payload.get("embedding_dim", 0) if payload.get("embedding_dim", None) is not None else 0,
                "sample_rate": payload.get("sample_rate", 0) if payload.get("sample_rate", None) is not None else 0,
                "model_available": payload.get("model_available", False) if payload.get("model_available", None) is not None else False,
                # Пути к .npy файлам с массивами статистик
                "clap_mean_npy": payload.get("clap_mean_npy", None),
                "clap_std_npy": payload.get("clap_std_npy", None),
                "clap_min_npy": payload.get("clap_min_npy", None),
                "clap_max_npy": payload.get("clap_max_npy", None),
                # Размеры массивов
                "clap_mean_shape": payload.get("clap_mean_shape", []) if payload.get("clap_mean_shape", None) is not None else [],
                "clap_std_shape": payload.get("clap_std_shape", []) if payload.get("clap_std_shape", None) is not None else [],
                "clap_min_shape": payload.get("clap_min_shape", []) if payload.get("clap_min_shape", None) is not None else [],
                "clap_max_shape": payload.get("clap_max_shape", []) if payload.get("clap_max_shape", None) is not None else [],
                # Скалярные статистики
                "clap_norm": payload.get("clap_norm", 0.0) if payload.get("clap_norm", None) is not None else 0.0,
                "clap_non_zero_count": payload.get("clap_non_zero_count", 0) if payload.get("clap_non_zero_count", None) is not None else 0,
                "clap_magnitude_mean": payload.get("clap_magnitude_mean", 0.0) if payload.get("clap_magnitude_mean", None) is not None else 0.0,
                "clap_magnitude_std": payload.get("clap_magnitude_std", 0.0) if payload.get("clap_magnitude_std", None) is not None else 0.0,
                "total_features": payload.get("total_features", 0) if payload.get("total_features", None) is not None else 0
            })
        
        elif extractor_name == "tempo":
            # Для Tempo экстрактора
            flat_payload.update({
                "tempo_bpm": float(payload.get("tempo_bpm", 0.0) or 0.0),
                "tempo_bpm_mean": float(payload.get("tempo_bpm_mean", 0.0) or 0.0),
                "tempo_bpm_median": float(payload.get("tempo_bpm_median", 0.0) or 0.0),
                "tempo_bpm_std": float(payload.get("tempo_bpm_std", 0.0) or 0.0),
                "tempo_confidence": float(payload.get("confidence", 0.0) or 0.0),
                "tempo_estimates_count": (lambda v: int(len(v)) if v is not None else 0)(payload.get("tempo_estimates", None)),
            })

        elif extractor_name == "loudness":
            # Для Loudness экстрактора
            flat_payload.update({
                "loudness_rms": payload.get("rms", 0.0) or 0.0,
                "loudness_peak": payload.get("peak", 0.0) or 0.0,
                "loudness_dbfs": payload.get("dbfs", 0.0) or 0.0,
                # Исключаем None — используем 0.0 как безопасный дефолт
                "loudness_lufs": (payload.get("lufs") if isinstance(payload.get("lufs"), (int, float)) else 0.0),
            })

        elif extractor_name == "onset":
            # Для Onset экстрактора
            flat_payload.update({
                "onset_count": payload.get("onset_count", 0) or 0,
                "onset_avg_interval_sec": payload.get("avg_interval_sec", 0.0) or 0.0,
                "onset_density_per_sec": payload.get("onset_density_per_sec", 0.0) or 0.0,
                "onset_interval_std": payload.get("interval_std", 0.0) or 0.0,
                "onset_interval_min": payload.get("interval_min", 0.0) or 0.0,
                "onset_interval_max": payload.get("interval_max", 0.0) or 0.0,
                "onset_interval_median": payload.get("interval_median", 0.0) or 0.0,
                "onset_insufficient_onsets": payload.get("insufficient_onsets", True),
            })

        elif extractor_name == "chroma":
            # Для Chroma экстрактора
            flat_payload.update({
                "chroma_mean": payload.get("chroma_mean", []) if payload.get("chroma_mean", None) is not None else [],
                "chroma_std": payload.get("chroma_std", []) if payload.get("chroma_std", None) is not None else [],
                "chroma_min": payload.get("chroma_min", []) if payload.get("chroma_min", None) is not None else [],
                "chroma_max": payload.get("chroma_max", []) if payload.get("chroma_max", None) is not None else [],
            })

        elif extractor_name == "spectral":
            # Для Spectral экстрактора
            def _s(key: str) -> Dict[str, float]:
                data = payload.get(key) or {}
                if not isinstance(data, dict):
                    return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
                return {
                    "mean": float(data.get("mean", 0.0) or 0.0),
                    "std": float(data.get("std", 0.0) or 0.0),
                    "min": float(data.get("min", 0.0) or 0.0),
                    "max": float(data.get("max", 0.0) or 0.0),
                }
            flat_payload.update({
                "spectral_centroid_mean": _s("spectral_centroid_stats").get("mean", 0.0),
                "spectral_centroid_std": _s("spectral_centroid_stats").get("std", 0.0),
                "spectral_bandwidth_mean": _s("spectral_bandwidth_stats").get("mean", 0.0),
                "spectral_bandwidth_std": _s("spectral_bandwidth_stats").get("std", 0.0),
                "spectral_flatness_mean": _s("spectral_flatness_stats").get("mean", 0.0),
                "spectral_flatness_std": _s("spectral_flatness_stats").get("std", 0.0),
                "spectral_rolloff_mean": _s("spectral_rolloff_stats").get("mean", 0.0),
                "spectral_rolloff_std": _s("spectral_rolloff_stats").get("std", 0.0),
                "zcr_mean": _s("zcr_stats").get("mean", 0.0),
                "zcr_std": _s("zcr_stats").get("std", 0.0),
            })


        elif extractor_name == "quality":
            # Для Quality экстрактора
            flat_payload.update({
                "quality_dc_offset": float(payload.get("dc_offset", 0.0) or 0.0),
                "quality_clipping_ratio": float(payload.get("clipping_ratio", 0.0) or 0.0),
                "quality_crest_factor_db": float(payload.get("crest_factor_db", 0.0) or 0.0),
                "quality_dynamic_range_db": float(payload.get("dynamic_range_db", 0.0) or 0.0),
                "quality_snr_db": float(payload.get("snr_db", 0.0) or 0.0),
            })

        elif extractor_name == "rhythmic":
            # Для Rhythmic экстрактора
            flat_payload.update({
                "rhythm_tempo_bpm": float(payload.get("rhythm_tempo_bpm", 0.0) or 0.0),
                "rhythm_beats_count": int(payload.get("rhythm_beats_count", 0) or 0),
                "rhythm_avg_period_sec": float(payload.get("rhythm_avg_period_sec", 0.0) or 0.0),
                "rhythm_period_std_sec": float(payload.get("rhythm_period_std_sec", 0.0) or 0.0),
                "rhythm_regularity": float(payload.get("rhythm_regularity", 0.0) or 0.0),
                "rhythm_beat_density": float(payload.get("rhythm_beat_density", 0.0) or 0.0),
            })

        elif extractor_name == "voice_quality":
            # Для VoiceQuality экстрактора
            flat_payload.update({
                "vq_jitter": float(payload.get("vq_jitter", 0.0) or 0.0),
                "vq_shimmer": float(payload.get("vq_shimmer", 0.0) or 0.0),
                "vq_hnr_like_db": float(payload.get("vq_hnr_like_db", 0.0) or 0.0),
            })

        elif extractor_name == "hpss":
            # Для HPSS экстрактора
            flat_payload.update({
                "hpss_harmonic_share": float(payload.get("hpss_harmonic_share", 0.0) or 0.0),
                "hpss_percussive_share": float(payload.get("hpss_percussive_share", 0.0) or 0.0),
                "hpss_energy_total": float(payload.get("hpss_energy_total", 0.0) or 0.0),
            })

        elif extractor_name == "key":
            # Для Key экстрактора
            flat_payload.update({
                "music_key": payload.get("key_name", "unknown") or "unknown",
                "music_mode": payload.get("key_mode", "unknown") or "unknown",
                "music_key_confidence": float(payload.get("key_confidence", 0.0) or 0.0),
            })

        elif extractor_name == "band_energy":
            # Для BandEnergy экстрактора
            flat_payload.update({
                "band_energy_total": float(payload.get("total_energy", 0.0) or 0.0),
                "band_energy_shares": payload.get("band_energy_shares", []) if payload.get("band_energy_shares", None) is not None else [],
            })

        elif extractor_name == "spectral_entropy":
            # Для SpectralEntropy экстрактора
            flat_payload.update({
                "spectral_entropy_mean": float(payload.get("spectral_entropy_mean", 0.0) or 0.0),
                "spectral_entropy_std": float(payload.get("spectral_entropy_std", 0.0) or 0.0),
            })

        elif extractor_name == "source_separation":
            # Для Open-Unmix экстрактора
            flat_payload.update({
                "sep_energy_total": float(payload.get("energy_total", 0.0) or 0.0),
                "sep_share_vocals": float(payload.get("share_vocals", 0.0) or 0.0),
                "sep_share_drums": float(payload.get("share_drums", 0.0) or 0.0),
                "sep_share_bass": float(payload.get("share_bass", 0.0) or 0.0),
                "sep_share_other": float(payload.get("share_other", 0.0) or 0.0),
            })

        elif extractor_name == "emotion_diarization":
            # Для эмоциональной диаризации SpeechBrain
            flat_payload.update({
                "speaker_count": int(payload.get("speaker_count", 0) or 0),
                "duration": float(payload.get("duration", 0.0) or 0.0),
            })
            # Краткие агрегаты по эмоциям
            if isinstance(payload.get("emotion_statistics"), dict):
                flat_payload["emotion_statistics"] = payload.get("emotion_statistics")
            # Сегменты (оставим в полном виде — обычно их не слишком много)
            if payload.get("emotion_segments") is not None:
                flat_payload["emotion_segments"] = payload.get("emotion_segments")
            if payload.get("speaker_segments") is not None:
                flat_payload["speaker_segments"] = payload.get("speaker_segments")
            # Маппинг эмоций к спикерам
            if payload.get("emotion_speaker_mapping") is not None:
                flat_payload["emotion_speaker_mapping"] = payload.get("emotion_speaker_mapping")

        elif extractor_name == "pitch":
            # Для Pitch экстрактора (без CREPE)
            flat_payload.update({
                "f0_mean": float(payload.get("f0_mean", 0.0) or 0.0),
                "f0_std": float(payload.get("f0_std", 0.0) or 0.0),
                "f0_min": float(payload.get("f0_min", 0.0) or 0.0),
                "f0_max": float(payload.get("f0_max", 0.0) or 0.0),
                "f0_median": float(payload.get("f0_median", 0.0) or 0.0),
                "f0_method": payload.get("f0_method", "none") or "none",
                "pitch_variation": float(payload.get("pitch_variation", 0.0) or 0.0),
                "pitch_stability": float(payload.get("pitch_stability", 0.0) or 0.0),
                "pitch_range": float(payload.get("pitch_range", 0.0) or 0.0),
                "f0_mean_pyin": float(payload.get("f0_mean_pyin", 0.0) or 0.0),
                "f0_std_pyin": float(payload.get("f0_std_pyin", 0.0) or 0.0),
                "f0_min_pyin": float(payload.get("f0_min_pyin", 0.0) or 0.0),
                "f0_max_pyin": float(payload.get("f0_max_pyin", 0.0) or 0.0),
                "f0_median_pyin": float(payload.get("f0_median_pyin", 0.0) or 0.0),
                "f0_count_pyin": int(payload.get("f0_count_pyin", 0) or 0),
                "voiced_fraction_pyin": float(payload.get("voiced_fraction_pyin", 0.0) or 0.0),
                "voiced_probability_mean_pyin": float(payload.get("voiced_probability_mean_pyin", 0.0) or 0.0),
                "f0_mean_yin": float(payload.get("f0_mean_yin", 0.0) or 0.0),
                "f0_std_yin": float(payload.get("f0_std_yin", 0.0) or 0.0),
                "f0_min_yin": float(payload.get("f0_min_yin", 0.0) or 0.0),
                "f0_max_yin": float(payload.get("f0_max_yin", 0.0) or 0.0),
                "f0_median_yin": float(payload.get("f0_median_yin", 0.0) or 0.0),
                "f0_count_yin": int(payload.get("f0_count_yin", 0) or 0),
                "device_used": payload.get("device_used", "unknown") or "unknown",
                "sample_rate": payload.get("sample_rate", 0) or 0,
                "total_features": int(payload.get("f0_count_pyin", 0) or 0) + int(payload.get("f0_count_yin", 0) or 0)
            })
            # Пути к npy для torchcrepe (если есть) и счетчики
            if payload.get("f0_series_torchcrepe_npy"):
                flat_payload["f0_series_torchcrepe_npy"] = payload.get("f0_series_torchcrepe_npy")
                flat_payload["f0_count_torchcrepe"] = int(payload.get("f0_count_torchcrepe", 0) or 0)

        elif extractor_name == "asr":
            # Для ASR экстрактора
            flat_payload.update({
                "transcription": payload.get("transcription", "") or "",
                "language": payload.get("language", "unknown") or "unknown",
                "language_probability": float(payload.get("language_probability", 0.0) or 0.0),
                # Разделяем полную длительность аудио и длительность речи
                "audio_duration": float(payload.get("audio_duration", 0.0) or 0.0),
                "speech_duration": float(payload.get("speech_duration", payload.get("duration", 0.0)) or 0.0),
                "model_size": payload.get("model_size", "unknown") or "unknown",
                "task": payload.get("task", "transcribe") or "transcribe",
                "segments_count": len(payload.get("segments", []) or []),
                "device_used": payload.get("device_used", "unknown") or "unknown",
                "sample_rate": payload.get("sample_rate", 0) or 0
            })

        elif extractor_name == "speaker_diarization":
            # Для диаризационного экстрактора
            flat_payload.update({
                "speaker_count": payload.get("speaker_count", 0) or 0,
                "segment_duration": float(payload.get("segment_duration", 0.0) or 0.0),
                "clustering_method": payload.get("clustering_method", "unknown") or "unknown",
                "duration": float(payload.get("duration", 0.0) or 0.0),
                "segments_count": len(payload.get("speaker_segments", []) or []),
                "device_used": payload.get("device_used", "unknown") or "unknown",
                "sample_rate": payload.get("sample_rate", 0) or 0
            })
            # Таймкоды сегментов спикеров
            if payload.get("speaker_segments"):
                flat_payload["speaker_segments"] = payload.get("speaker_segments")
            # Пер-спикерные агрегаты
            if payload.get("speaker_stats"):
                flat_payload["speaker_stats"] = payload.get("speaker_stats")
            # Добавим путь к npy и форму, если сохранены средние эмбеддинги спикеров
            if payload.get("speaker_embeddings_npy"):
                flat_payload["speaker_embeddings_npy"] = payload.get("speaker_embeddings_npy")
                flat_payload["speaker_embeddings_shape"] = payload.get("speaker_embeddings_shape", []) or []
                flat_payload["speaker_ids_order"] = payload.get("speaker_ids_order", []) or []

        elif extractor_name == "speech_analysis":
            # Для комбинированного экстрактора речи
            aligned_speech = payload.get("aligned_speech", {}) or {}
            pitch_result = payload.get("pitch_result", {}) or {}
            
            flat_payload.update({
                "total_speakers": aligned_speech.get("total_speakers", 0) or 0,
                "total_segments": aligned_speech.get("total_segments", 0) or 0,
                "total_duration": aligned_speech.get("statistics", {}).get("total_duration", 0.0) or 0.0,
                "total_words": aligned_speech.get("statistics", {}).get("total_words", 0) or 0,
                "confidence_mean": aligned_speech.get("statistics", {}).get("confidence_stats", {}).get("mean", 0.0) or 0.0,
                "asr_processing_time": payload.get("asr_processing_time", 0.0) or 0.0,
                "diarization_processing_time": payload.get("diarization_processing_time", 0.0) or 0.0,
                "pitch_processing_time": payload.get("pitch_processing_time", 0.0) or 0.0,
                # Pitch данные
                "pitch_mean": pitch_result.get("f0_mean", 0.0) or 0.0,
                "pitch_std": pitch_result.get("f0_std", 0.0) or 0.0,
                "pitch_min": pitch_result.get("f0_min", 0.0) or 0.0,
                "pitch_max": pitch_result.get("f0_max", 0.0) or 0.0,
                "pitch_median": pitch_result.get("f0_median", 0.0) or 0.0,
                "pitch_method": pitch_result.get("f0_method", "none") or "none",
                "pitch_stability": pitch_result.get("pitch_stability", 0.0) or 0.0,
                "pitch_variation": pitch_result.get("pitch_variation", 0.0) or 0.0,
                "device_used": payload.get("device_used", "unknown") or "unknown"
            })
            # Таймкоды выровненных сегментов доступны по пути к JSON
            if payload.get("aligned_segments_json"):
                flat_payload["aligned_segments_json"] = payload.get("aligned_segments_json")
            # Пер-спикерная статистика из анализа речи
            sp_stats = aligned_speech.get("statistics", {}).get("speaker_stats")
            if sp_stats:
                flat_payload["speaker_stats"] = sp_stats
            # Пробрасываем путь к torchcrepe npy и счетчик, если присутствуют внутри pitch_result
            if pitch_result.get("f0_series_torchcrepe_npy"):
                flat_payload["f0_series_torchcrepe_npy"] = pitch_result.get("f0_series_torchcrepe_npy")
                flat_payload["f0_count_torchcrepe"] = int(pitch_result.get("f0_count_torchcrepe", 0) or 0)
        
        return flat_payload
    
    def get_available_extractors(self) -> Dict[str, Dict[str, Any]]:
        """Получение списка доступных экстракторов."""
        return {
            name: extractor.get_info() 
            for name, extractor in self.extractors.items()
        }
    
    def get_processor_info(self) -> Dict[str, Any]:
        """Получение информации о процессоре."""
        return {
            "device": self.device,
            "max_workers": self.max_workers,
            "gpu_memory_limit": self.gpu_memory_limit,
            "sample_rate": self.sample_rate,
            "available_extractors": list(self.extractors.keys()),
            "total_extractors": len(self.extractors)
        }
