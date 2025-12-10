"""
Основной класс для обработки видео и анализа эмоций.
"""
import sys
import torch
from typing import Dict, Any, List, Optional, Tuple
from collections import OrderedDict

from core.processing_config import (
    ProcessingParams, ProcessingMetrics
)
from core.memory_manager import memory_context, cleanup_memory
from core.retry_strategy import RetryStrategy, QualityMetrics
from core.validation import ValidationLogic, ValidationCriteria
from core.logger import StructuredLogger
from core.exceptions import (
    VideoProcessingError, FrameSelectionError,
    EmotionAnalysisError, ValidationError
)
from core.validators import (
    validate_target_length
)
from core.edge_cases import validate_edge_cases
from core.cache_with_ttl import FaceScanCacheWithTTL

from utils import (
    segmentation, select_from_segments, uniform_time_coverage,
    build_emotion_curve, detect_keyframes, compress_sequence,
    expand_sequence, temporal_smoothing, validate_sequence_quality,
    save_for_user, save_for_model, get_video_type,
    analyze_emotion_profile, sample_for_static_face,
    analyze_emotion_changes, print_memory_usage,
    get_available_memory_mb,
    compute_steps, scan_for_faces, process_frames_in_batches
)
from core.advanced_emotion_features import (
    detect_micro_expressions,
    compute_physiological_signals,
    compute_face_asymmetry,
    compute_emotional_individuality
)

EMOTION_CLASSES = {
    0: "Neutral", 1: "Happy", 2: "Sad", 3: "Surprise",
    4: "Fear", 5: "Disgust", 6: "Anger", 7: "Contempt"
}


def log(*a, **kw):
    """Функция логирования."""
    print(*a, file=sys.stderr, **kw)


class FaceScanCache:
    """
    Кэш для результатов сканирования лиц с LRU стратегией.
    
    Использует OrderedDict для эффективной реализации LRU с O(1) операциями.
    """
    
    def __init__(self, max_size: int = 10):
        """
        Инициализация кэша.
        
        Args:
            max_size: Максимальное количество кэшированных результатов.
        
        Raises:
            ValueError: Если max_size <= 0.
        """
        if max_size <= 0:
            raise ValueError(f"max_size must be positive, got {max_size}")
        
        self.cache: OrderedDict[Tuple[int, float], Tuple[List[int], int]] = OrderedDict()
        self.max_size = max_size
    
    def get(self, scan_stride: int, detect_thr: float) -> Optional[Tuple[List[int], int]]:
        """
        Получает результат из кэша.
        
        Args:
            scan_stride: Шаг сканирования.
            detect_thr: Порог детекции.
        
        Returns:
            Кортеж (timeline, scanned_count) или None, если не найдено.
        """
        key = (scan_stride, detect_thr)
        if key in self.cache:
            # Перемещаем в конец (MRU) - O(1) операция
            self.cache.move_to_end(key)
            return self.cache[key]
        return None
    
    def put(self, scan_stride: int, detect_thr: float, timeline: List[int], scanned_count: int):
        """
        Сохраняет результат в кэш.
        
        Args:
            scan_stride: Шаг сканирования.
            detect_thr: Порог детекции.
            timeline: Список индексов кадров с лицами.
            scanned_count: Количество просканированных кадров.
        """
        key = (scan_stride, detect_thr)
        
        # Если ключ уже существует, обновляем и перемещаем в конец
        if key in self.cache:
            self.cache.move_to_end(key)
        else:
            # Если кэш переполнен, удаляем самый старый элемент (первый)
            if len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)  # O(1) операция
        
        self.cache[key] = (timeline, scanned_count)
    
    def clear(self):
        """Очищает кэш."""
        self.cache.clear()
    
    def __len__(self) -> int:
        """Возвращает количество элементов в кэше."""
        return len(self.cache)
    
    def __contains__(self, key: Tuple[int, float]) -> bool:
        """Проверяет наличие ключа в кэше."""
        return key in self.cache


class VideoEmotionProcessor:
    """
    Основной класс для обработки видео и анализа эмоций.
    Реализует все этапы обработки с четким разделением ответственности.
    """
    
    def __init__(
        self,
        # cache
        ttl_enabled: bool = True,
        ttl_seconds: int = 1800,
        cache_size_limit: int = 10,
        # validate
        min_frames_ratio: float = 0.8,
        min_keyframes: int = 3,
        min_transitions: int = 2,
        min_diversity_threshold: float = 0.2,
        quality_threshold: float = 0.4,
        # perfomance
        memory_threshold_low: int = 2000,
        batch_load_low: int = 20,
        batch_process_low: int = 8,
        memory_threshold_medium: int = 4000,
        batch_load_medium: int = 30,
        batch_process_medium: int = 12,
        memory_threshold_high: int = 8000,
        batch_load_high: int = 50,
        batch_process_high: int = 15,
        batch_load_very_high: int = 80,
        batch_process_very_high: int = 24,
        # logging
        enable_structured_metrics: bool = True,
        log_memory_usage: bool = True,
        # processing
        min_faces_threshold: int = 20,
        target_length: int = 256,
        max_retries: int = 2,
        # face detection
        default_threshold: float = 0.5,
        # keyframes
        transition_threshold: float = 0.3,
        # segmentation
        max_gap_seconds: float = 0.5,
        max_samples_per_segment: int = 10,
        # face app
        det_size: Tuple[int] = (640, 640),
        # emonet
        emo_path: str = None,
        # other
        device: str = "cuda",
    ):
        """
        Инициализация процессора.
        
        Args:
            config_path: Путь к файлу конфигурации. Если None, используется config.yaml.
        """
        self.device = device
        
        self.metrics = ProcessingMetrics()
        
        self.target_length = validate_target_length(target_length)
        self.max_retries = max_retries
        self.default_threshold = default_threshold
        self.transition_threshold = transition_threshold
        self.quality_threshold = quality_threshold
        self.min_diversity_threshold = min_diversity_threshold
        self.max_gap_seconds = max_gap_seconds
        self.max_samples_per_segment = max_samples_per_segment
        self.min_faces_threshold = min_faces_threshold
        
        self.memory_threshold_low = memory_threshold_low
        self.batch_load_low = batch_load_low
        self.batch_process_low = batch_process_low
        self.memory_threshold_medium = memory_threshold_medium
        self.batch_load_medium = batch_load_medium
        self.batch_process_medium = batch_process_medium
        self.memory_threshold_high = memory_threshold_high
        self.batch_load_high = batch_load_high
        self.batch_process_high = batch_process_high
        self.batch_load_very_high = batch_load_very_high
        self.batch_process_very_high = batch_process_very_high
        
        self.face_app = self.init_face_app(det_size)
        self.model = self.load_emonet(path=emo_path)
        
        if ttl_enabled:
            self.face_cache = FaceScanCacheWithTTL(
                max_size=cache_size_limit,
                ttl_seconds=ttl_seconds
            )
        else:
            self.face_cache = FaceScanCache(
                max_size=cache_size_limit
            )
        self.validation_logic = ValidationLogic(
            ValidationCriteria(
                min_frames_ratio=min_frames_ratio,
                min_keyframes=min_keyframes,
                min_transitions=min_transitions,
                min_diversity=self.min_diversity_threshold
            )
        )

        self.logger = StructuredLogger(
            enable_metrics=enable_structured_metrics
        )
        
    def load_emonet(self, path: str, n_expression: int = 8):
        from models.emonet.emonet.models.emonet import EmoNet
        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict):
            state = {k.replace("module.", ""): v for k, v in state.items()}
        model = EmoNet(n_expression=n_expression).to(self.device)
        model.load_state_dict(state, strict=False)
        model.eval()
        return model
        
    def init_face_app(self, det_size=(640,640)):
        try:
            from insightface.app import FaceAnalysis
        except Exception as e:
            raise RuntimeError("insightface not installed or failed to import") from e
        # try GPU, fallback CPU
        try:
            app = FaceAnalysis(providers=["CUDAExecutionProvider"])
            app.prepare(ctx_id=0, det_size=det_size)
        except Exception:
            app = FaceAnalysis(providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=-1, det_size=det_size)
        return app
    
    def process(
        self,
        frame_manager,
    ) -> Dict[str, Any]:
        """
        Основной метод обработки видео.
        
        Args:
            video_path: Путь к видео файлу.
            model: Загруженная модель EmoNet.
            self.face_app: Инициализированное приложение для детекции лиц.
            target_length: Целевая длина последовательности. Если None, берется из конфига.
            chunk_size: Размер чанка для обработки. Если None, берется из конфига.
        
        Returns:
            Словарь с результатами обработки.
        
        Raises:
            VideoFileError: Если видео файл некорректен.
            ConfigurationValidationError: Если параметры некорректны.
            VideoProcessingError: При ошибках обработки.
        """
        retry_strategy = RetryStrategy(max_retries=self.max_retries)
        
        # Инициализация параметров
        base_params = ProcessingParams(
            face_detection_threshold=self.default_threshold,
            scan_stride_multiplier=1.0,
            keyframe_threshold=self.transition_threshold,
            quality_threshold=self.quality_threshold,
            min_diversity=self.min_diversity_threshold,
            segment_max_gap=self.max_gap_seconds,
            samples_per_segment=self.max_samples_per_segment
        )
        
        current_params = base_params.copy()
        
        try:
            with memory_context():
                
                if self.log_memory_usage:
                    print_memory_usage(label="After frame_writer", log=log)
                    
                total_frames = frame_manager.total_frames
                fps = frame_manager.fps
                meta = frame_manager.meta
                video_path = meta["video_path"]
                
                # Валидация граничных случаев
                try:
                    edge_cases_info = validate_edge_cases(
                        total_frames,
                        fps,
                        [],  # timeline будет заполнен позже
                        min_faces_ratio=self.min_faces_threshold / total_frames if total_frames > 0 else 0.01
                    )
                    self.logger.log(f"Edge cases validation: {edge_cases_info.get('warnings', [])}")
                except VideoProcessingError as e:
                    self.logger.log(f"Edge case validation failed: {e}", level="WARNING")
                    # Продолжаем обработку, но с предупреждением
                
                # Основной цикл обработки с повторными попытками
                while retry_strategy.attempts <= self.max_retries:
                    try:
                        cleanup_memory()
                        
                        self.logger.log(f"Попытка {retry_strategy.attempts + 1}/{self.max_retries + 1}")
                        self.logger.log_metrics(current_params.to_dict(), "Параметры обработки")
                        
                        # Этап 1: Адаптивный сбор кадров
                        self.logger.start_stage("adaptive_frame_selection")
                        result = self._adaptive_frame_selection(
                            frame_manager, self.face_app, total_frames, fps, current_params
                        )
                        self.logger.end_stage("adaptive_frame_selection")
                        
                        if not result["success"]:
                            if retry_strategy.next_attempt():
                                current_params = retry_strategy.adjust_parameters(
                                    current_params,
                                    QualityMetrics(is_valid=False, is_acceptable=False),
                                    result.get("video_type", "UNKNOWN"),
                                    result.get("segments_count", 0),
                                    result.get("faces_found", 0),
                                    log
                                )
                                continue
                            else:
                                return self._build_failure_result(current_params, retry_strategy.attempts)
                        
                        selected_indices = result["selected_indices"]
                        timeline = result["timeline"]
                        segments = result["segments"]
                        video_type = result["video_type"]
                        
                        # Этап 2: Анализ эмоций
                        self.logger.start_stage("emotion_analysis")
                        emotion_result = self._emotion_analysis_pipeline(
                            frame_manager, selected_indices, self.model, total_frames, fps, current_params, self.face_app
                        )
                        self.logger.end_stage("emotion_analysis")
                        
                        if not emotion_result["success"]:
                            if retry_strategy.next_attempt():
                                current_params = retry_strategy.adjust_parameters(
                                    current_params,
                                    QualityMetrics(is_valid=False, is_acceptable=False),
                                    video_type,
                                    len(segments),
                                    len(timeline),
                                    log
                                )
                                continue
                            else:
                                return self._build_failure_result(current_params, retry_strategy.attempts)
                        
                        # Этап 3: Валидация и нормализация
                        self.logger.start_stage("validation_and_normalization")
                        validation_result = self._validation_and_retry_logic(
                            emotion_result,
                            selected_indices,
                            self.target_length,
                            video_type,
                            current_params,
                            retry_strategy
                        )
                        self.logger.end_stage("validation_and_normalization")
                        
                        # Логируем метрики качества
                        if validation_result.get("quality_metrics"):
                            quality_dict = validation_result["quality_metrics"].to_dict()
                            self.logger.log_quality_metrics(quality_dict)
                        
                        if validation_result["should_retry"]:
                            if retry_strategy.should_retry(validation_result["quality_metrics"]):
                                current_params = retry_strategy.adjust_parameters(
                                    current_params,
                                    validation_result["quality_metrics"],
                                    video_type,
                                    len(segments),
                                    len(timeline),
                                    log
                                )
                                retry_strategy.next_attempt()
                                continue
                        
                        # Успешная обработка
                        self.logger.start_stage("save_results")
                        result = self._save_results(
                            validation_result,
                            emotion_result,
                            selected_indices,
                            video_path,
                            meta,
                            current_params,
                            retry_strategy.attempts + 1,
                            video_type,
                            len(segments),
                            len(timeline)
                        )
                        self.logger.end_stage("save_results")
                        
                        # Добавляем метрики логирования в результат
                        result["logging_metrics"] = self.logger.get_summary()
                        return result
                        
                    except Exception as e:
                        self.logger.log(f"Ошибка в попытке {retry_strategy.attempts}: {e}", level="ERROR")
                        import traceback
                        traceback.print_exc()
                        
                        if retry_strategy.next_attempt():
                            current_params = retry_strategy.get_safe_params()
                            self.logger.log("Переход к безопасным параметрам", level="WARNING")
                            continue
                        else:
                            return self._build_failure_result(current_params, retry_strategy.attempts, str(e))
                
                # Все попытки неудачны
                return self._build_failure_result(current_params, retry_strategy.attempts)
        
        finally:
            torch.cuda.empty_cache()
            self.face_cache.clear()
    
    def _adaptive_frame_selection(
        self,
        fm,
        face_app,
        total_frames: int,
        fps: float,
        params: ProcessingParams
    ) -> Dict[str, Any]:
        """
        Этап 1: Адаптивный сбор кадров с лицами.
        
        Returns:
            Словарь с результатами: success, selected_indices, timeline, segments, video_type
        """
        try:
            # Динамическое сканирование
            scan_stride, target_scans = compute_steps(total_frames)
            adjusted_scan_stride = int(scan_stride * params.scan_stride_multiplier)
            adjusted_scan_stride = max(1, adjusted_scan_stride)
            
            self.logger.log(f"Scan stride: {scan_stride} -> {adjusted_scan_stride}")
            
            # Проверяем кэш (с поддержкой TTL, если используется FaceScanCacheWithTTL)
            if isinstance(self.face_cache, FaceScanCacheWithTTL):
                cached_result = self.face_cache.get_scan_result(adjusted_scan_stride, params.face_detection_threshold)
                if cached_result:
                    self.logger.log("Используется кэшированный результат сканирования (с TTL)")
                    timeline, scanned_count = cached_result
                else:
                    # Сканирование лиц
                    timeline, scanned_count = scan_for_faces(
                        fm,
                        face_app,
                        scan_stride=adjusted_scan_stride,
                        detect_thr=params.face_detection_threshold
                    )
                    self.face_cache.put_scan_result(adjusted_scan_stride, params.face_detection_threshold, timeline, scanned_count)
            else:
                # Старый кэш без TTL
                cache_key = (adjusted_scan_stride, params.face_detection_threshold)
                cached_result = self.face_cache.get(*cache_key)
                
                if cached_result:
                    self.logger.log("Используется кэшированный результат сканирования")
                    timeline, scanned_count = cached_result
                else:
                    # Сканирование лиц
                    timeline, scanned_count = scan_for_faces(
                        fm,
                        face_app,
                        scan_stride=adjusted_scan_stride,
                        detect_thr=params.face_detection_threshold
                    )
                    self.face_cache.put(adjusted_scan_stride, params.face_detection_threshold, timeline, scanned_count)
            
            # Проверка граничных случаев после сканирования
            try:
                edge_cases_info = validate_edge_cases(
                    total_frames,
                    fps,
                    timeline,
                    min_faces_ratio=self.min_faces_threshold / total_frames if total_frames > 0 else 0.01
                )
                if edge_cases_info.get("warnings"):
                    for warning in edge_cases_info["warnings"]:
                        self.logger.log(f"Warning: {warning}", level="WARNING")
            except FrameSelectionError as e:
                # Если нет лиц - это критическая ошибка
                raise
            
            self.logger.log(f"Найдено лиц: {len(timeline)} (порог: {params.face_detection_threshold})")
            
            # Сегментация
            segments = segmentation(
                timeline,
                fps=fps,
                max_gap_seconds=params.segment_max_gap
            )
            self.logger.log(f"Создано {len(segments)} сегментов")
            
            # Определение типа видео
            video_type = get_video_type(timeline, total_frames, segments)
            self.logger.log(f"Тип видео: {video_type}")
            
            # Выборка кадров
            if video_type == "STATIC_FACE":
                selected_indices = sample_for_static_face(
                    segments,
                    total_frames,
                    fps,
                    target_samples=min(150, total_frames)
                )
            else:
                selected_indices = select_from_segments(
                    segments,
                    total_frames,
                    fps=fps,
                    max_samples_per_segment=params.samples_per_segment
                )
            
            n_frames = len(selected_indices)
            self.logger.log(f"Выбрано индексов: {n_frames}")
            
            if n_frames < 10:
                self.logger.log(f"Слишком мало кадров ({n_frames}), переключаюсь на равномерное покрытие", level="WARNING")
                selected_indices = uniform_time_coverage(
                    total_frames,
                    min(256 * 3, total_frames)
                )
            elif n_frames < self.min_faces_threshold:
                self.logger.log("Комбинирую лица и равномерное покрытие")
                uniform_indices = uniform_time_coverage(
                    total_frames,
                    min(256 * 2, total_frames - len(selected_indices))
                )
                selected_indices = sorted(list(set(selected_indices + uniform_indices)))
            

            self.logger.log(f"Кол-во кадров на выходе 1 Этапа: {len(selected_indices)}")

            return {
                "success": len(selected_indices) > 0,
                "selected_indices": selected_indices,
                "timeline": timeline,
                "segments": segments,
                "video_type": video_type,
                "faces_found": len(timeline),
                "segments_count": len(segments)
            }
        
        except Exception as e:
            self.logger.log(f"Ошибка в _adaptive_frame_selection: {e}", level="ERROR")
            if isinstance(e, VideoProcessingError):
                raise
            raise FrameSelectionError(
                f"Failed to select frames: {e}",
                details={"error": str(e), "total_frames": total_frames}
            ) from e
    
    def _emotion_analysis_pipeline(
        self,
        fm,
        selected_indices: List[int],
        model,
        total_frames: int,
        fps: float,
        params: ProcessingParams,
        face_app=None
    ) -> Dict[str, Any]:
        """
        Этап 2: Анализ эмоций в выбранных кадрах.
        
        Returns:
            Словарь с результатами: success, emo_results, emotion_profile, change_analysis
        """
        try:
            # Определение размера батча
            available_memory = get_available_memory_mb()
            
            if available_memory < self.memory_threshold_low:
                batch_load = self.batch_load_low
                batch_process = self.batch_process_low
            elif available_memory < self.memory_threshold_medium:
                batch_load = self.batch_load_medium
                batch_process = self.batch_process_medium
            elif available_memory < self.memory_threshold_high:
                batch_load = self.batch_load_high
                batch_process = self.batch_process_high
            else:
                batch_load = self.batch_load_very_high
                batch_process = self.batch_process_very_high
            
            self.logger.log(f"Размеры батчей: загрузка={batch_load}, обработка={batch_process}")
            
            # Обработка кадров батчами
            emo_results = process_frames_in_batches(
                fm,
                selected_indices,
                model,
                log,
                batch_size_load=batch_load,
                batch_size_process=batch_process
            )
            
            # Анализ эмоционального профиля
            emotion_profile = analyze_emotion_profile(emo_results)
            self.logger.log(f"Доминирующая эмоция: {emotion_profile['dominant_emotion']}")
            
            # Анализ изменений эмоций
            change_analysis = analyze_emotion_changes(emo_results)
            self.logger.log(f"Тип изменений: {change_analysis['change_type']}")
            
            # Расширенные фичи: микроэмоции
            self.logger.log("Вычисление микроэмоций...")
            microexpressions = detect_micro_expressions(emo_results, fps=fps)
            self.logger.log(f"Найдено микроэмоций: {microexpressions['microexpressions_count']}")
            
            # Расширенные фичи: физиологические сигналы
            self.logger.log("Вычисление физиологических сигналов...")
            physiological_signals = compute_physiological_signals(
                emo_results, 
                microexpressions=microexpressions,
                fps=fps
            )
            self.logger.log(f"Стресс: {physiological_signals['stress_level_score']:.2f}, "
                          f"Уверенность: {physiological_signals['confidence_face_score']:.2f}")
            
            # Расширенные фичи: индивидуальность выражения эмоций
            self.logger.log("Анализ индивидуальности выражения эмоций...")
            emotional_individuality = compute_emotional_individuality(emo_results, fps=fps)
            self.logger.log(f"Индекс выразительности: {emotional_individuality['expressivity_index']:.2f}")
            
            # Расширенные фичи: асимметрия лица (упрощенная версия без landmarks)
            # Для полной версии нужны landmarks из face_app, что требует дополнительной обработки
            # Здесь используем упрощенную версию на основе эмоциональных паттернов
            self.logger.log("Анализ асимметрии лица (упрощенная версия)...")
            face_asymmetry = compute_face_asymmetry(landmarks=None, face_data=None)
            # Примечание: полная версия требует landmarks, которые можно получить из face_app
            
            return {
                "success": True,
                "emo_results": emo_results,
                "emotion_profile": emotion_profile,
                "change_analysis": change_analysis,
                "microexpressions": microexpressions,
                "physiological_signals": physiological_signals,
                "emotional_individuality": emotional_individuality,
                "face_asymmetry": face_asymmetry
            }
        
        except Exception as e:
            self.logger.log(f"Ошибка в _emotion_analysis_pipeline: {e}", level="ERROR")
            if isinstance(e, VideoProcessingError):
                raise
            raise EmotionAnalysisError(
                f"Failed to analyze emotions: {e}",
                details={"error": str(e), "frames_count": len(selected_indices)}
            ) from e
    
    def _validation_and_retry_logic(
        self,
        emotion_result: Dict[str, Any],
        selected_indices: List[int],
        target_length: int,
        video_type: str,
        params: ProcessingParams,
        retry_strategy: RetryStrategy
    ) -> Dict[str, Any]:
        """
        Этап 3: Валидация и нормализация последовательности.
        
        Returns:
            Словарь с результатами: should_retry, quality_metrics, final_indices, smoothed_emotions, keyframes
        """
        try:
            emo_results = emotion_result["emo_results"]
            emotion_profile = emotion_result["emotion_profile"]
            neutral_percentage = emotion_profile["neutral_percentage"]
            
            # Построение кривой эмоций
            emotion_curve = build_emotion_curve(emo_results)
            
            # Детекция ключевых кадров
            keyframes_indices = detect_keyframes(
                emotion_curve,
                EMOTION_CLASSES,
                threshold=params.keyframe_threshold
            )
            
            # Нормализация до target_length
            n_frames = len(selected_indices)
            if n_frames == target_length:
                final_indices = selected_indices
                final_emotions = emo_results
            elif n_frames > target_length:
                final_indices, final_emotions = compress_sequence(
                    selected_indices,
                    emo_results,
                    keyframes_indices,
                    target_length
                )
            else:
                final_indices, final_emotions = expand_sequence(
                    selected_indices,
                    emo_results,
                    keyframes_indices,
                    target_length
                )
            
            # Сглаживание
            smoothed_emotions = temporal_smoothing(final_emotions, window=3)
            
            # Валидация качества
            quality_metrics_raw = validate_sequence_quality(
                smoothed_emotions,
                min_diversity_threshold=params.min_diversity,
                is_static_face=(video_type == "STATIC_FACE"),
                neutral_percentage=neutral_percentage,
                logger=self.logger
            )
            
            # Единая логика валидации
            quality_metrics = self.validation_logic.validate_quality(
                smoothed_emotions,
                quality_metrics_raw,
                target_length,
                len(keyframes_indices),
                is_monotonic=quality_metrics_raw.get("is_monotonic", False),
                neutral_percentage=neutral_percentage,
                logger=self.logger
            )
            
            self.logger.log(f"Валидация: acceptable={quality_metrics.is_acceptable}")
            
            return {
                "should_retry": not quality_metrics.is_acceptable,
                "quality_metrics": quality_metrics,
                "quality_metrics_raw": quality_metrics_raw,
                "final_indices": final_indices,
                "smoothed_emotions": smoothed_emotions,
                "keyframes_indices": keyframes_indices,
                "emotion_curve": emotion_curve
            }
        
        except Exception as e:
            self.logger.log(f"Ошибка в _validation_and_retry_logic: {e}", level="ERROR")
            if isinstance(e, VideoProcessingError):
                raise
            raise ValidationError(
                f"Failed to validate results: {e}",
                details={"error": str(e), "target_length": target_length}
            ) from e
    
    def _save_results(
        self,
        validation_result: Dict[str, Any],
        emotion_result: Dict[str, Any],
        selected_indices: List[int],
        video_path: str,
        meta: Dict[str, Any],
        params: ProcessingParams,
        attempt_number: int,
        video_type: str,
        segments_count: int,
        faces_found: int
    ) -> Dict[str, Any]:
        """
        Сохранение результатов обработки.
        
        Returns:
            Словарь с результатами обработки.
        """
        quality_metrics = validation_result["quality_metrics"]
        quality_metrics_raw = validation_result["quality_metrics_raw"]
        final_indices = validation_result["final_indices"]
        smoothed_emotions = validation_result["smoothed_emotions"]
        keyframes_indices = validation_result["keyframes_indices"]
        emotion_curve = validation_result["emotion_curve"]
        emo_results = emotion_result["emo_results"]
        emotion_profile = emotion_result["emotion_profile"]
        
        # Расширенные фичи
        microexpressions = emotion_result.get("microexpressions", {})
        physiological_signals = emotion_result.get("physiological_signals", {})
        emotional_individuality = emotion_result.get("emotional_individuality", {})
        face_asymmetry = emotion_result.get("face_asymmetry", {})
        
        # Подготовка данных для пользователя
        user_data = {
            "original_emotions": emo_results,
            "keyframes": [],
            "emotion_curve": emotion_curve,
            "quality_metrics": quality_metrics_raw,
            "processing_params": params.to_dict(),
            "processing_stats": {
                "total_frames": meta["total_frames"],
                "faces_found": faces_found,
                "segments": segments_count,
                "selected_frames": len(selected_indices),
                "final_length": len(smoothed_emotions),
                "keyframes_count": len(keyframes_indices),
                "attempt_number": attempt_number,
                "success": True,
                "video_type": video_type
            },
            # Расширенные фичи
            "advanced_features": {
                "microexpressions": microexpressions,
                "physiological_signals": physiological_signals,
                "emotional_individuality": emotional_individuality,
                "face_asymmetry": face_asymmetry
            }
        }
        
        # Добавляем ключевые кадры
        for idx in keyframes_indices.keys():
            if idx < len(selected_indices):
                user_data["keyframes"].append({
                    "global_index": int(selected_indices[idx]),
                    "local_index": int(idx),
                    "type": keyframes_indices[idx]["type"],
                    "emotion": emo_results[idx] if idx < len(emo_results) else {}
                })
        
        # Сохранение для пользователя
        user_file = save_for_user(user_data, video_path)
        
        # Подготовка данных для модели
        model_data = {
            "indices": [int(idx) for idx in final_indices],
            "emotions": smoothed_emotions,
            "valence": [e["valence"] for e in smoothed_emotions],
            "arousal": [e["arousal"] for e in smoothed_emotions],
            "sequence_length": len(smoothed_emotions),
            "video_metadata": meta,
            "quality_score": quality_metrics_raw.get("overall_score", 0),
            "processing_attempt": attempt_number,
            # Расширенные фичи для модели
            "microexpressions_count": microexpressions.get("microexpressions_count", 0),
            "microexpression_rate": microexpressions.get("microexpression_rate", 0.0),
            "stress_level_score": physiological_signals.get("stress_level_score", 0.0),
            "confidence_face_score": physiological_signals.get("confidence_face_score", 0.0),
            "tension_face_index": physiological_signals.get("tension_face_index", 0.0),
            "nervousness_score": physiological_signals.get("nervousness_score", 0.0),
            "emotional_intensity_baseline": emotional_individuality.get("emotional_intensity_baseline", 0.0),
            "expressivity_index": emotional_individuality.get("expressivity_index", 0.0),
            "emotional_range": emotional_individuality.get("emotional_range", 0.0),
            "dominant_style": emotional_individuality.get("dominant_style", "neutral"),
            "asymmetry_score": face_asymmetry.get("asymmetry_score", 0.0),
            "sincerity_score": face_asymmetry.get("sincerity_score", 0.5)
        }
        
        # Сохранение для модели
        model_files = save_for_model(model_data, video_path)
        
        return {
            "success": True,
            "user_data": user_data,
            "model_data": model_data,
            "phase1_stats": user_data["processing_stats"],
            "phase2_stats": {
                "final_length": len(smoothed_emotions),
                "keyframes_count": len(keyframes_indices),
                "quality": quality_metrics_raw
            },
            "files": {
                "user": user_file,
                "model": model_files
            },
            "attempts": attempt_number,
            "final_params": params.to_dict(),
            "metrics": self.metrics.to_dict()
        }
    
    def _build_failure_result(
        self,
        params: ProcessingParams,
        attempts: int,
        error: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Создает результат при неудачной обработке.
        
        Returns:
            Словарь с информацией об ошибке.
        """
        return {
            "success": False,
            "error": error or "Failed to process video after retries",
            "attempts": attempts + 1,
            "final_params": params.to_dict(),
            "metrics": self.metrics.to_dict()
        }

