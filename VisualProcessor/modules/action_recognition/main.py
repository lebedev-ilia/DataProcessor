"""
CLI интерфейс для модуля распознавания действий в видео.
"""

import os
import sys
_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

if _path not in sys.path:
    sys.path.append(_path)

import json
import argparse
from typing import Optional

import numpy as np

from action_recognition_slowfast import SlowFastActionRecognizer
from utils.frame_manager import FrameManager
from utils.results_store import ResultsStore

name = "action_recognition"

from utils.logger import get_logger
logger = get_logger(name)

def load_metadata(meta_path):
    try:
        with open(meta_path, "r") as f:
            return json.load(f)
    except:
        raise f"{name} | main | load_metadata | Ошибка при открытии метадаты"

def process_video(
    frame_manager,
    total_frames,
    model_dir: str = None,
    clip_len: int = 16,
    batch_size: int = 8,
    max_tracks: Optional[int] = None
):
    """
    Обрабатывает видео и распознает действия.
    
    Args:
        model_dir: Путь к директории с моделью VideoMAE
        clip_len: Длина клипа в кадрах
        batch_size: Размер батча для inference
        max_tracks: Максимальное количество треков для обработки
    """
    if not model_dir:
        model_dir = f"{os.path.dirname(__file__)}/models"
            
    recognizer = SlowFastActionRecognizer(
        frame_manager=frame_manager,
        model_name=str(model_dir) if model_dir else None,
        clip_len=clip_len,
        batch_size=batch_size
    )
    
    # Подготавливаем треки.
    # В реальном сценарии треки должны приходить из модуля трекинга (YOLO + ByteTrack и т.п.).
    # CLI-обвязка по умолчанию обрабатывает всё видео как один трек (single-person).
    logger.info(f"[INFO] Подготавливаю треки для обработки (single-track режим)...")
    
    frame_indices_per_person = {
        1: list(range(0, total_frames))
    }
    
    logger.info(f"[INFO] Обрабатываю {len(frame_indices_per_person)} трек(ов)...")
    
    # Обрабатываем треки
    results = recognizer.process(frame_indices_per_person)
    
    # Добавляем метаданные
    output_data = {
        "total_frames": total_frames,
        "num_tracks": len(results),
        "processing_params": {
            "clip_len": clip_len,
            "batch_size": batch_size,
        },
        "results": {}
    }
    
    # Преобразуем результаты в JSON-совместимый формат
    for track_id, track_results in results.items():
        # Конвертируем numpy типы в нативные Python типы
        json_results = {}
        for key, value in track_results.items():
            if isinstance(value, (np.integer, np.floating)):
                json_results[key] = float(value) if isinstance(value, np.floating) else int(value)
            elif isinstance(value, np.ndarray):
                json_results[key] = value.tolist()
            elif isinstance(value, (list, dict)):
                # Рекурсивно обрабатываем вложенные структуры
                json_results[key] = _convert_to_json(value)
            else:
                json_results[key] = value
        output_data["results"][str(track_id)] = json_results
    
    logger.info(f"[INFO] Обработано треков: {len(results)}")
    
    # Выводим краткую статистику
    for track_id, track_results in results.items():
        logger.info(f"\n  Трек {track_id}:")
        logger.info(f"    Средняя норма embedding: {track_results.get('mean_embedding_norm', 0.0):.2f}")
        logger.info(f"    Временная вариация: {track_results.get('temporal_variance', 0.0):.2f}")
        logger.info(f"    Уникальных действий: {track_results.get('num_unique_actions', 0)}")
        logger.info(f"    Стабильность: {track_results.get('stability', 0.0):.2f}")
        logger.info(f"    Частота смены действий: {track_results.get('switch_rate_per_sec', 0.0):.2f} сек^-1")
        logger.info(f"    Многолюдная сцена: {track_results.get('is_multi_person', False)}")
    
    return output_data


def _convert_to_json(obj):
    """Рекурсивно конвертирует объекты в JSON-совместимый формат."""
    import numpy as np
    
    if isinstance(obj, (np.integer, np.floating)):
        return float(obj) if isinstance(obj, np.floating) else int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: _convert_to_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_json(item) for item in obj]
    else:
        return obj


def main():
    parser = argparse.ArgumentParser(
        description='Распознавание действий в видео с использованием SlowFast',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--frames-dir',    type=str, required=True, help='Путь к входному видео файлу')
    parser.add_argument('--rs-path',       type=str, required=True, help='Путь к входному видео файлу')
    parser.add_argument('--clip-len',      type=int, default=16, help='Длина клипа в кадрах для обработки')
    parser.add_argument('--batch-size',    type=int, default=8, help='Размер батча для inference')
    parser.add_argument('--max-tracks',    type=int, default=None, help='Максимальное количество треков для обработки')
    
    args = parser.parse_args()
    
    metadata = load_metadata(f"{args.frames_dir}/metadata.json")
    
    frame_manager = FrameManager(frames_dir=args.frames_dir, chunk_size=metadata["chunk_size"], cache_size=metadata["cache_size"])
    
    rs = ResultsStore(args.rs_path)
    
    results = process_video(
        frame_manager=frame_manager,
        total_frames=metadata[name]["num_indices"],
        clip_len=args.clip_len,
        batch_size=args.batch_size,
        max_tracks=args.max_tracks
    )
    
    rs.store(results, name=name)
        

if __name__ == "__main__":
    main()
