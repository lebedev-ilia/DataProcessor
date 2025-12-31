"""
Основной класс для обработки видео и анализа эмоций.

Все TODO выполнены:
    1. ✅ Интеграция с внешними зависимостями через BaseModule (face_detection)
    2. ✅ Использование результатов face_detection вместо прямых вызовов
    3. ✅ Интеграция с BaseModule через класс EmotionFaceModule
    4. ✅ Единый формат вывода для сохранения в npz
"""
import os
import sys
import torch
import json
from pathlib import Path
from typing import Dict, Any, List, Optional

# Добавляем путь для импорта BaseModule
_MODULE_PATH = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if _MODULE_PATH not in sys.path:
    sys.path.append(_MODULE_PATH)

from modules.base_module import BaseModule
from utils.frame_manager import FrameManager

from core.processing_config import (
    ProcessingParams, ProcessingMetrics
)
from core.memory_manager import memory_context, cleanup_memory
from core.retry_strategy import RetryStrategy, QualityMetrics
from core.validation import ValidationLogic, ValidationCriteria
from core.exceptions import (
    VideoProcessingError, FrameSelectionError,
    EmotionAnalysisError, ValidationError
)
from core.validators import (
    validate_target_length
)

from _utils import (
    segmentation, select_from_segments, uniform_time_coverage,
    build_emotion_curve, detect_keyframes, compress_sequence,
    expand_sequence, temporal_smoothing, validate_sequence_quality,
    save_for_user, save_for_model, get_video_type,
    analyze_emotion_profile, sample_for_static_face,
    analyze_emotion_changes,
    get_available_memory_mb,
    compute_steps, process_frames_in_batches
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

from utils.logger import get_logger
logger = get_logger("VideoEmotionProcessor")

class VideoEmotionProcessor:
    """
    Основной класс для обработки видео и анализа эмоций.
    Реализует все этапы обработки с четким разделением ответственности.
    """
    
    def __init__(
        self,
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
        # processing
        min_faces_threshold: int = 20,
        target_length: int = 256,
        max_retries: int = 2,
        # keyframes
        transition_threshold: float = 0.3,
        # segmentation
        max_gap_seconds: float = 0.5,
        max_samples_per_segment: int = 10,
        # emonet
        emo_path: str = None,
        # other
        device: str = "cuda",
        # BaseModule integration
        rs_path: Optional[str] = None,
        load_dependency_func: Optional[callable] = None,
    ):
        """
        Инициализация процессора.
        
        Args:
            config_path: Путь к файлу конфигурации. Если None, используется config.yaml.
            rs_path: Путь к хранилищу результатов (для загрузки зависимостей)
            load_dependency_func: Функция для загрузки зависимостей через BaseModule
        """
        self.device = device
        self.rs_path = rs_path
        
        self.metrics = ProcessingMetrics()
        
        self.target_length = validate_target_length(target_length)
        self.max_retries = max_retries
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

        # Загружаем frames_with_face с поддержкой BaseModule
        self.frames_with_face = self.frames_with_face_load("auto", rs_path=rs_path, load_func=load_dependency_func)

        if emo_path == "None" or emo_path is None:
            import os
            p = os.path.dirname(os.path.dirname(__file__))
            emo_path = f"{p}/models/emonet/pretrained/emonet_8.pth"
        
        self.model = self.load_emonet(path=emo_path)
        
        self.validation_logic = ValidationLogic(
            ValidationCriteria(
                min_frames_ratio=min_frames_ratio,
                min_keyframes=min_keyframes,
                min_transitions=min_transitions,
                min_diversity=self.min_diversity_threshold
            )
        )
        
    def frames_with_face_load(self, filename, rs_path: Optional[str] = None, load_func: Optional[callable] = None):
        """
        Загружает список кадров с лицами из результатов face_detection.
        
        Args:
            filename: Имя файла или "auto" для автоматического поиска
            rs_path: Путь к хранилищу результатов (если None, использует старый метод)
            load_func: Функция для загрузки через BaseModule (load_dependency_results)
            
        Returns:
            Список индексов кадров с лицами
        """
        # Вариант 1: через BaseModule (если доступна функция загрузки)
        if load_func is not None and rs_path:
            try:
                face_data = load_func("face_detection", format="json")
                if face_data and isinstance(face_data, dict):
                    frames_with_face = face_data.get("frames_with_face", [])
                    if not frames_with_face:
                        # Альтернативный формат: frames -> keys
                        frames = face_data.get("frames", {})
                        if frames:
                            frames_with_face = [int(k) for k, v in frames.items() if v and len(v) > 0]
                    
                    logger.info(f"VideoEmotionProcessor | frames_with_face_load | Загружено {len(frames_with_face)} кадров через BaseModule")
                    return sorted(frames_with_face)
            except Exception as e:
                logger.warning(f"VideoEmotionProcessor | frames_with_face_load | Ошибка загрузки через BaseModule: {e}, используем fallback")
        
        # Вариант 2: прямой доступ к файлу (fallback для обратной совместимости)
        if rs_path:
            face_detection_dir = os.path.join(rs_path, "face_detection")
            if os.path.exists(face_detection_dir):
                json_files = list(Path(face_detection_dir).glob("*.json"))
                if json_files:
                    if filename == "auto":
                        json_file = max(json_files, key=lambda p: p.stat().st_mtime)
                    else:
                        json_file = Path(face_detection_dir) / filename
                        if not json_file.exists():
                            logger.warning(f"VideoEmotionProcessor | frames_with_face_load | Файл {json_file} не найден")
                            return []
                    
                    try:
                        with open(json_file, "r", encoding="utf-8") as f:
                            face_data = json.load(f)
                        frames_with_face = face_data.get("frames_with_face", [])
                        logger.info(f"VideoEmotionProcessor | frames_with_face_load | Загружено {len(frames_with_face)} кадров (fallback)")
                        return sorted(frames_with_face)
                    except Exception as e:
                        logger.error(f"VideoEmotionProcessor | frames_with_face_load | Ошибка чтения файла: {e}")
                        return []
        
        # Вариант 3: старый метод (для обратной совместимости)
        p = f"{os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))}/result_store/face_detection"
        try:
            if os.path.exists(p):
                if filename == "auto":
                    files = os.listdir(p)
                    if files:
                        filename = files[-1]
                    else:
                        logger.warning("VideoEmotionProcessor | frames_with_face_load | Нет файлов в директории")
                        return []
                file_path = os.path.join(p, filename)
                if os.path.exists(file_path):
                    with open(file_path, "r", encoding="utf-8") as f:
                        face_data = json.load(f)
                    frames_with_face = face_data.get("frames_with_face", [])
                    logger.info(f"VideoEmotionProcessor | frames_with_face_load | Загружено {len(frames_with_face)} кадров (legacy)")
                    return sorted(frames_with_face)
        except Exception as e:
            logger.error(f"VideoEmotionProcessor | frames_with_face_load | Error: {e}")
        
        logger.warning("VideoEmotionProcessor | frames_with_face_load | Не удалось загрузить frames_with_face, возвращаем пустой список")
        return []

    def load_emonet(self, path: str, n_expression: int = 8):
        from models.emonet.emonet.models.emonet import EmoNet
        state = torch.load(path, map_location="cpu")
        if isinstance(state, dict):
            state = {k.replace("module.", ""): v for k, v in state.items()}
        model = EmoNet(n_expression=n_expression).to(self.device)
        model.load_state_dict(state, strict=False)
        model.eval()
        return model
    
    def process(
        self,
        frame_manager,
        save_path
    ) -> Dict[str, Any]:
        """
        Основной метод обработки видео.
        
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
                    
                total_frames = frame_manager.total_frames
                fps = frame_manager.fps
                meta = frame_manager.meta
                
                # Основной цикл обработки с повторными попытками
                while retry_strategy.attempts <= self.max_retries:
                    try:
                        cleanup_memory()
                        
                        logger.info(f"Попытка {retry_strategy.attempts + 1}/{self.max_retries + 1}")
                        
                        result = self._adaptive_frame_selection(
                            frame_manager, self.frames_with_face, total_frames, fps, current_params
                        )
                        
                        if not result["success"]:
                            if retry_strategy.next_attempt():
                                current_params = retry_strategy.adjust_parameters(
                                    current_params,
                                    QualityMetrics(is_valid=False, is_acceptable=False),
                                    result.get("video_type", "UNKNOWN"),
                                    result.get("segments_count", 0),
                                    result.get("faces_found", 0),
                                    log_func=logger.info
                                )
                                continue
                            else:
                                return self._build_failure_result(current_params, retry_strategy.attempts)
                        
                        selected_indices = result["selected_indices"]
                        timeline = result["timeline"]
                        segments = result["segments"]
                        video_type = result["video_type"]
                        
                        emotion_result = self._emotion_analysis_pipeline(
                            frame_manager, selected_indices, self.model, fps
                        )
                        
                        if not emotion_result["success"]:
                            if retry_strategy.next_attempt():
                                current_params = retry_strategy.adjust_parameters(
                                    current_params,
                                    QualityMetrics(is_valid=False, is_acceptable=False),
                                    video_type,
                                    len(segments),
                                    len(timeline),
                                    logger.info
                                )
                                continue
                            else:
                                return self._build_failure_result(current_params, retry_strategy.attempts)
                        
                        # Этап 3: Валидация и нормализация
                        validation_result = self._validation_and_retry_logic(
                            emotion_result,
                            selected_indices,
                            self.target_length,
                            video_type,
                            current_params,
                            retry_strategy
                        )
                        
                        # Логируем метрики качества
                        if validation_result.get("quality_metrics"):
                            quality_dict = validation_result["quality_metrics"].to_dict()
                        
                        if validation_result["should_retry"]:
                            if retry_strategy.should_retry(validation_result["quality_metrics"]):
                                current_params = retry_strategy.adjust_parameters(
                                    current_params,
                                    validation_result["quality_metrics"],
                                    video_type,
                                    len(segments),
                                    len(timeline),
                                    logger.info
                                )
                                retry_strategy.next_attempt()
                                continue
                        
                        # Успешная обработка
                        result = self._save_results(
                            validation_result,
                            emotion_result,
                            selected_indices,
                            save_path,
                            meta,
                            current_params,
                            retry_strategy.attempts + 1,
                            video_type,
                            len(segments),
                            len(timeline)
                        )
                        
                        return result
                        
                    except Exception as e:
                        logger.error(f"Ошибка в попытке {retry_strategy.attempts}: {e}")
                        import traceback
                        traceback.print_exc()
                        
                        if retry_strategy.next_attempt():
                            current_params = retry_strategy.get_safe_params()
                            logger.info("Переход к безопасным параметрам")
                            continue
                        else:
                            return self._build_failure_result(current_params, retry_strategy.attempts, str(e))
                
                # Все попытки неудачны
                return self._build_failure_result(current_params, retry_strategy.attempts)
        
        finally:
            torch.cuda.empty_cache()
    
    def _adaptive_frame_selection(
        self,
        fm,
        timeline,
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
            
            logger.info(f"Scan stride: {scan_stride} -> {adjusted_scan_stride}")
            
            logger.info(f"Найдено лиц: {len(timeline)} (порог: {params.face_detection_threshold})")
            
            # Сегментация
            segments = segmentation(
                timeline,
                fps=fps,
                max_gap_seconds=params.segment_max_gap
            )
            logger.info(f"Создано {len(segments)} сегментов")
            
            # Определение типа видео
            video_type = get_video_type(timeline, total_frames, segments)
            logger.info(f"Тип видео: {video_type}")
            
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
            logger.info(f"Выбрано индексов: {n_frames}")
            
            if n_frames < 10:
                logger.info(f"Слишком мало кадров ({n_frames}), переключаюсь на равномерное покрытие")
                selected_indices = uniform_time_coverage(
                    total_frames,
                    min(256 * 3, total_frames)
                )
            elif n_frames < self.min_faces_threshold:
                logger.info("Комбинирую лица и равномерное покрытие")
                uniform_indices = uniform_time_coverage(
                    total_frames,
                    min(256 * 2, total_frames - len(selected_indices))
                )
                selected_indices = sorted(list(set(selected_indices + uniform_indices)))
            

            logger.info(f"Кол-во кадров на выходе 1 Этапа: {len(selected_indices)}")

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
            logger.error(f"Ошибка в _adaptive_frame_selection: {e}")
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
        fps: float,
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
            
            logger.info(f"Размеры батчей: загрузка={batch_load}, обработка={batch_process}")
            
            # Обработка кадров батчами
            emo_results = process_frames_in_batches(
                fm,
                selected_indices,
                model,
                logger.info,
                batch_size_load=batch_load,
                batch_size_process=batch_process
            )
            
            # Анализ эмоционального профиля
            emotion_profile = analyze_emotion_profile(emo_results)
            logger.info(f"Доминирующая эмоция: {emotion_profile['dominant_emotion']}")
            
            # Анализ изменений эмоций
            change_analysis = analyze_emotion_changes(emo_results)
            logger.info(f"Тип изменений: {change_analysis['change_type']}")
            
            # Расширенные фичи: микроэмоции (с улучшенными параметрами)
            logger.info("Вычисление микроэмоций...")
            microexpressions = detect_micro_expressions(
                emo_results, 
                fps=fps,
                min_frames=2,  # Require at least 2 frames (0.03s at 30fps ≈ 1 frame is too short)
                change_threshold=None  # Use adaptive threshold (85th percentile)
            )
            logger.info(f"Найдено микроэмоций: {microexpressions['microexpressions_count']}")
            
            # Расширенные фичи: физиологические сигналы
            logger.info("Вычисление физиологических сигналов...")
            physiological_signals = compute_physiological_signals(
                emo_results, 
                microexpressions=microexpressions,
                fps=fps
            )
            logger.info(f"Стресс: {physiological_signals['stress_level_score']:.2f}, "
                          f"Уверенность: {physiological_signals['confidence_face_score']:.2f}")
            
            # Расширенные фичи: индивидуальность выражения эмоций
            logger.info("Анализ индивидуальности выражения эмоций...")
            emotional_individuality = compute_emotional_individuality(emo_results, fps=fps)
            logger.info(f"Индекс выразительности: {emotional_individuality['expressivity_index']:.2f}")
            
            # Расширенные фичи: асимметрия лица (упрощенная версия без landmarks)
            # Для полной версии нужны landmarks из face_app, что требует дополнительной обработки
            # Здесь используем упрощенную версию на основе эмоциональных паттернов
            logger.info("Анализ асимметрии лица (упрощенная версия)...")
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
            logger.error(f"Ошибка в _emotion_analysis_pipeline: {e}")
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
            
            # Детекция ключевых кадров с улучшенными параметрами
            keyframes_indices = detect_keyframes(
                emotion_curve,
                EMOTION_CLASSES,
                threshold=params.keyframe_threshold,
                prominence=0.1,  # Normalized prominence threshold
                min_distance=8  # Minimum distance between peaks (8-12 frames recommended)
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
                logger=logger
            )
            
            # Единая логика валидации
            quality_metrics = self.validation_logic.validate_quality(
                smoothed_emotions,
                quality_metrics_raw,
                target_length,
                len(keyframes_indices),
                is_monotonic=quality_metrics_raw.get("is_monotonic", False),
                neutral_percentage=neutral_percentage,
                logger=logger
            )
            
            logger.info(f"Валидация: acceptable={quality_metrics.is_acceptable}")
            
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
            logger.error(f"Ошибка в _validation_and_retry_logic: {e}")
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
        save_path: str,
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
            "emotion_profile": emotion_profile,
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
        user_file = save_for_user(user_data, save_path)
        
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
        model_files = save_for_model(model_data, save_path)
        
        return {
            "success": True,
            "user_data": user_data,
            "model_data": model_data,
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


class EmotionFaceModule(BaseModule):
    """
    Модуль для анализа эмоций на лицах в видео.
    
    Наследуется от BaseModule для интеграции с системой зависимостей и единым форматом вывода.
    Использует VideoEmotionProcessor для обработки видео.
    
    Зависимости:
    - face_detection (обязательная) - для получения списка кадров с лицами
    """
    
    def __init__(
        self,
        rs_path: Optional[str] = None,
        min_frames_ratio: float = 0.8,
        min_keyframes: int = 3,
        min_transitions: int = 2,
        min_diversity_threshold: float = 0.2,
        quality_threshold: float = 0.4,
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
        enable_structured_metrics: bool = True,
        min_faces_threshold: int = 20,
        target_length: int = 256,
        max_retries: int = 2,
        transition_threshold: float = 0.3,
        max_gap_seconds: float = 0.5,
        max_samples_per_segment: int = 10,
        emo_path: Optional[str] = None,
        device: str = "cuda",
        **kwargs: Any
    ):
        """
        Инициализация EmotionFaceModule.
        
        Args:
            rs_path: Путь к хранилищу результатов
            min_frames_ratio: Минимальное соотношение кадров
            min_keyframes: Минимальное количество ключевых кадров
            min_transitions: Минимальное количество переходов
            min_diversity_threshold: Минимальный порог разнообразия
            quality_threshold: Порог качества
            memory_threshold_low: Порог памяти (низкий)
            batch_load_low: Размер батча загрузки (низкий)
            batch_process_low: Размер батча обработки (низкий)
            memory_threshold_medium: Порог памяти (средний)
            batch_load_medium: Размер батча загрузки (средний)
            batch_process_medium: Размер батча обработки (средний)
            memory_threshold_high: Порог памяти (высокий)
            batch_load_high: Размер батча загрузки (высокий)
            batch_process_high: Размер батча обработки (высокий)
            batch_load_very_high: Размер батча загрузки (очень высокий)
            batch_process_very_high: Размер батча обработки (очень высокий)
            enable_structured_metrics: Включить структурированные метрики
            min_faces_threshold: Минимальный порог лиц
            target_length: Целевая длина последовательности
            max_retries: Максимальное количество повторных попыток
            transition_threshold: Порог перехода
            max_gap_seconds: Максимальный разрыв в секундах
            max_samples_per_segment: Максимальное количество сэмплов на сегмент
            emo_path: Путь к модели EmoNet
            device: Устройство для обработки (cuda/cpu)
            **kwargs: Дополнительные параметры для BaseModule
        """
        super().__init__(rs_path=rs_path, **kwargs)
        
        # Создаем обёртку для load_dependency_results
        def load_dep_wrapper(module_name: str, format: str = "auto"):
            return self.load_dependency_results(module_name, format=format)
        
        # Инициализируем процессор
        self.processor = VideoEmotionProcessor(
            min_frames_ratio=min_frames_ratio,
            min_keyframes=min_keyframes,
            min_transitions=min_transitions,
            min_diversity_threshold=min_diversity_threshold,
            quality_threshold=quality_threshold,
            memory_threshold_low=memory_threshold_low,
            batch_load_low=batch_load_low,
            batch_process_low=batch_process_low,
            memory_threshold_medium=memory_threshold_medium,
            batch_load_medium=batch_load_medium,
            batch_process_medium=batch_process_medium,
            memory_threshold_high=memory_threshold_high,
            batch_load_high=batch_load_high,
            batch_process_high=batch_process_high,
            batch_load_very_high=batch_load_very_high,
            batch_process_very_high=batch_process_very_high,
            enable_structured_metrics=enable_structured_metrics,
            min_faces_threshold=min_faces_threshold,
            target_length=target_length,
            max_retries=max_retries,
            transition_threshold=transition_threshold,
            max_gap_seconds=max_gap_seconds,
            max_samples_per_segment=max_samples_per_segment,
            emo_path=emo_path,
            device=device,
            rs_path=rs_path,
            load_dependency_func=load_dep_wrapper,
        )
    
    def required_dependencies(self) -> List[str]:
        """
        Возвращает список зависимостей модуля.
        
        Обязательные:
        - face_detection: для получения списка кадров с лицами
        """
        return ["face_detection"]
    
    def process(
        self,
        frame_manager: FrameManager,
        frame_indices: List[int],
        config: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Основной метод обработки (интерфейс BaseModule).
        
        Args:
            frame_manager: Менеджер кадров
            frame_indices: Список индексов кадров для обработки (не используется напрямую,
                          VideoEmotionProcessor использует frames_with_face из face_detection)
            config: Конфигурация модуля (не используется, параметры заданы в __init__)
            
        Returns:
            Словарь с результатами в формате для сохранения в npz:
            - features: агрегированные фичи эмоций
            - sequence_features: последовательности эмоций для VisualTransformer
            - summary: метаданные обработки
        """
        # VideoEmotionProcessor использует frames_with_face из face_detection,
        # поэтому frame_indices игнорируется (процессор сам выбирает кадры)
        
        # Создаем временный путь для сохранения (VideoEmotionProcessor требует save_path)
        import tempfile
        temp_save_path = os.path.join(self.rs_path, self.module_name, "temp")
        os.makedirs(temp_save_path, exist_ok=True)
        
        try:
            # Обрабатываем через VideoEmotionProcessor
            result = self.processor.process(frame_manager, temp_save_path)
            
            if not result.get("success", False):
                self.logger.warning("VideoEmotionProcessor завершился с ошибкой")
                return self._empty_result()
            
            # Преобразуем результаты в единый формат для npz
            formatted_result = self._format_results_for_npz(result)
            
            self.logger.info(
                f"EmotionFaceModule | Обработка завершена: "
                f"success={result.get('success', False)}, "
                f"attempts={result.get('attempts', 0)}"
            )
            
            return formatted_result
            
        except Exception as e:
            self.logger.exception(f"EmotionFaceModule | Ошибка обработки: {e}")
            return self._empty_result()
        finally:
            # Очищаем временные файлы
            try:
                import shutil
                if os.path.exists(temp_save_path):
                    shutil.rmtree(temp_save_path)
            except Exception:
                pass
    
    def _format_results_for_npz(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Преобразует результаты VideoEmotionProcessor в формат для сохранения в npz.
        
        Args:
            result: Результаты из processor.process()
            
        Returns:
            Словарь в формате для npz
        """
        import numpy as np
        
        user_data = result.get("user_data", {})
        model_data = result.get("model_data", {})
        
        # Извлекаем основные данные
        emotions = model_data.get("emotions", [])
        indices = model_data.get("indices", [])
        valence = model_data.get("valence", [])
        arousal = model_data.get("arousal", [])
        emotion_profile = user_data.get("emotion_profile", {})
        quality_metrics = user_data.get("quality_metrics", {})
        processing_stats = user_data.get("processing_stats", {})
        advanced_features = user_data.get("advanced_features", {})
        
        # Подготавливаем sequence_features для VisualTransformer
        sequence_features = {
            "emotion_sequence": np.array(emotions, dtype=object) if emotions else np.array([], dtype=object),
            "valence_sequence": np.array(valence, dtype=np.float32) if valence else np.array([], dtype=np.float32),
            "arousal_sequence": np.array(arousal, dtype=np.float32) if arousal else np.array([], dtype=np.float32),
            "frame_indices": np.array(indices, dtype=np.int32) if indices else np.array([], dtype=np.int32),
        }
        
        # Подготавливаем агрегированные фичи
        features = {}
        
        # Emotion profile features
        if emotion_profile:
            for key, value in emotion_profile.items():
                if isinstance(value, (int, float, bool)):
                    features[f"emotion_profile_{key}"] = float(value) if isinstance(value, bool) else value
                elif isinstance(value, (list, tuple)):
                    try:
                        features[f"emotion_profile_{key}"] = np.asarray(value, dtype=np.float32)
                    except Exception:
                        features[f"emotion_profile_{key}"] = np.asarray(value, dtype=object)
        
        # Quality metrics
        if quality_metrics:
            for key, value in quality_metrics.items():
                if isinstance(value, (int, float, bool)):
                    features[f"quality_{key}"] = float(value) if isinstance(value, bool) else value
        
        # Processing stats
        if processing_stats:
            for key, value in processing_stats.items():
                if isinstance(value, (int, float, bool)):
                    features[f"processing_{key}"] = float(value) if isinstance(value, bool) else value
        
        # Advanced features
        if advanced_features:
            for category, category_data in advanced_features.items():
                if isinstance(category_data, dict):
                    for key, value in category_data.items():
                        if isinstance(value, (int, float, bool)):
                            features[f"{category}_{key}"] = float(value) if isinstance(value, bool) else value
                        elif isinstance(value, (list, tuple)):
                            try:
                                features[f"{category}_{key}"] = np.asarray(value, dtype=np.float32)
                            except Exception:
                                features[f"{category}_{key}"] = np.asarray(value, dtype=object)
        
        # Model data features
        for key in ["microexpressions_count", "microexpression_rate", "stress_level_score",
                   "confidence_face_score", "tension_face_index", "nervousness_score",
                   "emotional_intensity_baseline", "expressivity_index", "emotional_range",
                   "asymmetry_score", "sincerity_score"]:
            if key in model_data:
                value = model_data[key]
                if isinstance(value, (int, float, bool)):
                    features[key] = float(value) if isinstance(value, bool) else value
        
        if "dominant_style" in model_data:
            features["dominant_style"] = str(model_data["dominant_style"])
        
        # Подготавливаем summary
        summary = {
            "success": result.get("success", False),
            "attempts": result.get("attempts", 0),
            "sequence_length": len(emotions),
            "total_frames": processing_stats.get("total_frames", 0),
            "faces_found": processing_stats.get("faces_found", 0),
            "selected_frames": processing_stats.get("selected_frames", 0),
            "keyframes_count": processing_stats.get("keyframes_count", 0),
            "video_type": processing_stats.get("video_type", "UNKNOWN"),
        }
        
        # Формируем итоговый результат
        formatted_result = {
            "features": features,
            "sequence_features": sequence_features,
            "summary": summary,
        }
        
        return formatted_result
    
    def _empty_result(self) -> Dict[str, Any]:
        """Возвращает пустой результат в правильном формате."""
        import numpy as np
        return {
            "features": {},
            "sequence_features": {
                "emotion_sequence": np.array([], dtype=object),
                "valence_sequence": np.array([], dtype=np.float32),
                "arousal_sequence": np.array([], dtype=np.float32),
                "frame_indices": np.array([], dtype=np.int32),
            },
            "summary": {
                "success": False,
                "attempts": 0,
                "sequence_length": 0,
                "total_frames": 0,
                "faces_found": 0,
                "selected_frames": 0,
                "keyframes_count": 0,
                "video_type": "UNKNOWN",
            },
        }

