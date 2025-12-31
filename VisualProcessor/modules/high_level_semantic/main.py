#!/usr/bin/env python3
"""
CLI интерфейс для модуля высокоуровневой семантики видео.

Использует HighLevelSemanticsOptimized для извлечения семантических фичей из видео.
"""

from __future__ import annotations

import os
import sys
import argparse
import json
import warnings
from typing import Optional, List, Dict, Any
from PIL import Image
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from modules.high_level_semantic.hl_semantic import HighLevelSemanticsOptimized
from utils.frame_manager import FrameManager
from utils.results_store import ResultsStore
from utils.logger import get_logger

MODULE_NAME = "high_level_semantic"
logger = get_logger(MODULE_NAME)

warnings.filterwarnings("ignore", category=UserWarning)


def _load_json(path: str) -> Dict[str, Any]:
    """Загружает JSON файл."""
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise Exception(f"{MODULE_NAME} | main | load_json | Ошибка при открытии файла {path}: {e}")


def run_pipeline(
    frames_dir: str,
    rs_path: str,
    device: str = "cuda",
    clip_model_name: str = "ViT-B/32",
    clip_batch_size: int = 64,
    use_face_data: bool = False,
    use_audio_data: bool = False,
    use_cut_data: bool = False,
    class_prompts: Optional[str] = None,
) -> str:
    """
    Основная логика обработки high_level_semantic.
    
    Возвращает путь к сохраненному файлу результатов.
    """
    if not rs_path:
        raise ValueError(f"{MODULE_NAME} | rs_path не указан")

    rs = ResultsStore(rs_path)

    meta_path = os.path.join(frames_dir, "metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"{MODULE_NAME} | metadata.json не найден в {frames_dir}")

    metadata = _load_json(meta_path)

    frame_manager = FrameManager(
        frames_dir,
        chunk_size=int(metadata.get("chunk_size", 32)),
        cache_size=int(metadata.get("cache_size", 2)),
    )

    fps = metadata.get("fps", 30)
    total_frames = metadata.get("total_frames", 0)

    logger.info(f"VisualProcessor | {MODULE_NAME} | main | Initializing HighLevelSemanticsOptimized")

    processor = HighLevelSemanticsOptimized(
        device=device,
        clip_model_name=clip_model_name,
        clip_batch_size=clip_batch_size,
        fps=fps,
    )

    # Загружаем дополнительные данные из других модулей, если доступны
    face_emotion_curve = None
    audio_energy_curve = None
    scene_boundary_frames = None

    if use_face_data:
        try:
            face_results_path = os.path.join(rs_path, "emotion_face")
            if os.path.exists(face_results_path):
                face_files = [f for f in os.listdir(face_results_path) if f.endswith(".json")]
                if face_files:
                    face_data = _load_json(os.path.join(face_results_path, sorted(face_files)[-1]))
                    if "emotion_curve" in face_data:
                        face_emotion_curve = np.array(face_data["emotion_curve"])
                    logger.info(f"VisualProcessor | {MODULE_NAME} | main | Loaded face emotion data")
        except Exception as e:
            logger.warning(f"VisualProcessor | {MODULE_NAME} | main | Could not load face data: {e}")

    if use_audio_data:
        try:
            # Audio processing would go here
            # For now, we'll skip it
            logger.info(f"VisualProcessor | {MODULE_NAME} | main | Audio data path found but not processed yet")
        except Exception as e:
            logger.warning(f"VisualProcessor | {MODULE_NAME} | main | Could not load audio data: {e}")

    if use_cut_data:
        try:
            cut_results_path = os.path.join(rs_path, "cut_detection")
            if os.path.exists(cut_results_path):
                cut_files = [f for f in os.listdir(cut_results_path) if f.endswith(".json")]
                if cut_files:
                    cut_data = _load_json(os.path.join(cut_results_path, sorted(cut_files)[-1]))
                    # Extract scene boundaries if available
                    if "scene_boundaries" in cut_data:
                        scene_boundary_frames = cut_data["scene_boundaries"]
                    elif "cuts" in cut_data:
                        # Extract frame indices from cuts
                        scene_boundary_frames = [cut.get("frame", 0) for cut in cut_data["cuts"]]
                    logger.info(f"VisualProcessor | {MODULE_NAME} | main | Loaded cut detection data")
        except Exception as e:
            logger.warning(f"VisualProcessor | {MODULE_NAME} | main | Could not load cut data: {e}")

    # Получаем scene frames - используем cut boundaries если доступны, иначе равномерная выборка
    logger.info(f"VisualProcessor | {MODULE_NAME} | main | Extracting scene frames")

    try:
        if scene_boundary_frames and len(scene_boundary_frames) > 0:
            # Используем cut boundaries для определения сцен
            scene_boundary_frames = sorted(set([0] + scene_boundary_frames + [total_frames - 1]))
            scene_frames = []
            for i in range(len(scene_boundary_frames) - 1):
                start_frame = scene_boundary_frames[i]
                end_frame = scene_boundary_frames[i + 1]
                # Берем средний кадр каждой сцены
                mid_frame = (start_frame + end_frame) // 2
                frame = frame_manager.get(mid_frame)
                scene_frames.append(Image.fromarray(frame))
        else:
            # Равномерная выборка - один кадр каждые ~2 секунды
            sample_rate = max(1, int(fps * 2))
            scene_frames = []
            for frame_idx in range(0, total_frames, sample_rate):
                frame = frame_manager.get(frame_idx)
                scene_frames.append(Image.fromarray(frame))

        logger.info(f"VisualProcessor | {MODULE_NAME} | main | Extracted {len(scene_frames)} scene frames")

        # Парсим class prompts
        parsed_class_prompts = None
        if class_prompts:
            parsed_class_prompts = [p.strip() for p in class_prompts.split(",")]

        # Извлекаем фичи
        logger.info(f"VisualProcessor | {MODULE_NAME} | main | Extracting high-level semantic features")

        result = processor.extract_all(
            scene_frames=scene_frames,
            scene_embeddings=None,
            face_emotion_curve=face_emotion_curve,
            audio_energy_curve=audio_energy_curve,
            pose_activity_curve=None,
            text_features=None,
            topic_vectors=None,
            class_prompts=parsed_class_prompts,
            scene_boundary_frames=scene_boundary_frames,
        )

        # Добавляем метаданные
        result["metadata"] = {
            "total_frames": total_frames,
            "fps": fps,
            "n_scenes": len(scene_frames),
            "device": device,
            "clip_model": clip_model_name,
        }

        # Сохраняем результаты через ResultsStore
        rs.store(result, name=MODULE_NAME)

        logger.info(
            f"VisualProcessor | {MODULE_NAME} | main | Обработка завершена. "
            f"Результаты сохранены в {rs_path}/{MODULE_NAME}"
        )

        return os.path.join(rs_path, MODULE_NAME)

    finally:
        try:
            frame_manager.close()
        except Exception as e:  # noqa: BLE001
            logger.exception(
                f"VisualProcessor | {MODULE_NAME} | main | Ошибка при закрытии FrameManager: {e}"
            )


def main(argv: Optional[List[str]] = None) -> int:
    """Главная функция CLI."""
    parser = argparse.ArgumentParser(
        prog=f"run_{MODULE_NAME}",
        description="Высокоуровневая семантика видео — CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--frames-dir",
        type=str,
        required=True,
        help="Директория с кадрами (должна содержать metadata.json)",
    )
    parser.add_argument(
        "--rs-path",
        type=str,
        required=True,
        help="Путь к директории ResultsStore для сохранения результатов",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Устройство для обработки (cuda/cpu)",
    )
    parser.add_argument(
        "--clip-model-name",
        type=str,
        default="ViT-B/32",
        help="Имя модели CLIP",
    )
    parser.add_argument(
        "--clip-batch-size",
        type=int,
        default=64,
        help="Размер батча для обработки CLIP",
    )
    parser.add_argument(
        "--use-face-data",
        action="store_true",
        help="Использовать данные эмоций лиц из модуля emotion_face",
    )
    parser.add_argument(
        "--use-audio-data",
        action="store_true",
        help="Использовать аудио данные из аудио процессора",
    )
    parser.add_argument(
        "--use-cut-data",
        action="store_true",
        help="Использовать данные детекции склеек для границ сцен",
    )
    parser.add_argument(
        "--class-prompts",
        type=str,
        default=None,
        help="Список промптов классов для zero-shot классификации (через запятую)",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Уровень логирования (DEBUG/INFO/WARN/ERROR)",
    )

    args = parser.parse_args(argv)

    # Настройка уровня логирования
    try:
        import logging as _logging

        _logging.getLogger().setLevel(getattr(_logging, args.log_level.upper(), _logging.INFO))
    except Exception:
        logger.warning("Не удалось установить log-level: %s", args.log_level)

    try:
        saved_path = run_pipeline(
            frames_dir=args.frames_dir,
            rs_path=args.rs_path,
            device=args.device,
            clip_model_name=args.clip_model_name,
            clip_batch_size=args.clip_batch_size,
            use_face_data=args.use_face_data,
            use_audio_data=args.use_audio_data,
            use_cut_data=args.use_cut_data,
            class_prompts=args.class_prompts,
        )

        logger.info(f"Обработка завершена. Результаты сохранены: {saved_path}")
        return 0

    except FileNotFoundError as e:
        logger.error("Файл не найден: %s", e)
        return 2
    except ValueError as e:
        logger.error("Некорректные данные: %s", e)
        return 3
    except Exception as e:
        logger.exception("Fatal error в %s: %s", MODULE_NAME, e)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
