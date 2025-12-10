"""
CLI интерфейс для модуля распознавания действий в видео.
"""

import json
import argparse
from typing import Optional

import numpy as np

from action_recognition_videomae import VideoMAEActionRecognizer
from utils.frame_manager import FrameManager
from utils.results_store import ResultsStore

name = "action_recognition"

def load_metadata(meta_path):
    try:
        with open(meta_path, "r") as f:
            return json.load(f)
    except:
        raise f"{name} | main | load_metadata | Ошибка при открытии метадаты"

def process_video(
    frame_manager,
    frame_skip,
    model_dir: str,
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
        frame_skip: Пропуск кадров (обрабатывать каждый N-й кадр)
        max_tracks: Максимальное количество треков для обработки
    """
    
    fps = frame_manager.fps
    total_frames = frame_manager.total_frames
            
    recognizer = VideoMAEActionRecognizer(
        frame_manager=frame_manager,
        model_name=str(model_dir),
        clip_len=clip_len,
        batch_size=batch_size
    )
    
    # Подготавливаем треки (для простоты используем равномерную выборку)
    # В реальном сценарии треки должны приходить из модуля трекинга
    print(f"[INFO] Подготавливаю треки для обработки...")
    
    # Создаем треки на основе равномерной выборки кадров
    # Для демонстрации создаем несколько треков
    num_tracks = min(5, max_tracks or 5)  # По умолчанию 5 треков
    track_length = total_frames // num_tracks
    
    frame_indices_per_person = {}
    for track_id in range(1, num_tracks + 1):
        start_frame = (track_id - 1) * track_length
        end_frame = min(track_id * track_length, total_frames - 1)
        
        # Выбираем кадры с учетом frame_skip
        indices = list(range(start_frame, end_frame + 1, frame_skip))
        if indices:
            frame_indices_per_person[track_id] = indices
    
    print(f"[INFO] Обрабатываю {len(frame_indices_per_person)} треков...")
    
    # Обрабатываем треки
    results = recognizer.process(frame_indices_per_person)
    
    # Добавляем метаданные
    output_data = {
        "model_dir": str(model_dir),
        "total_frames": total_frames,
        "fps": fps,
        "num_tracks": len(results),
        "processing_params": {
            "clip_len": clip_len,
            "batch_size": batch_size,
            "frame_skip": frame_skip
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
    
    print(f"[INFO] Обработано треков: {len(results)}")
    
    # Выводим краткую статистику
    for track_id, track_results in results.items():
        print(f"\n  Трек {track_id}:")
        print(f"    Доминирующее действие: {track_results.get('dominant_action_label', 'unknown')}")
        print(f"    Уверенность: {track_results.get('dominant_confidence', 0.0):.2f}")
        print(f"    Сложность действия: {track_results.get('complexity_score', 0.0):.2f}")
        print(f"    Тип активности сцены: {track_results.get('scene_activity_type', 'unknown')}")
    
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
        description='Распознавание действий в видео с использованием VideoMAE',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument('--frames-dir',    type=str, required=True, help='Путь к входному видео файлу')
    parser.add_argument('--rs-path',       type=str, required=True, help='Путь к входному видео файлу')
    
    parser.add_argument('--model',         type=str, required=True, help='Путь к директории с моделью VideoMAE')
    parser.add_argument('--clip-len',      type=int, default=16, help='Длина клипа в кадрах для обработки')
    parser.add_argument('--batch-size',    type=int, default=8, help='Размер батча для inference')
    parser.add_argument('--frame-skip',    type=int, default=1, help='Обрабатывать каждый N-й кадр (для ускорения)')
    parser.add_argument('--max-tracks',    type=int, default=None, help='Максимальное количество треков для обработки')
    
    args = parser.parse_args()
    
    metadata = load_metadata(f"{args.frames_dir}/metadata.json")
    
    frame_manager = FrameManager(frames_dir=args.frames_dir, chunk_size=metadata["chunk_size"], cache_size=metadata["cache_size"])
    
    rs = ResultsStore(args.rs_path)
    
    results = process_video(
        frame_manager=frame_manager,
        frame_skip=args.frame_skip,
        model_dir=args.model_path,
        clip_len=args.clip_len,
        batch_size=args.batch_size,
        max_tracks=args.max_tracks
    )
    
    rs.store(results, name=name)
        

if __name__ == "__main__":
    main()
