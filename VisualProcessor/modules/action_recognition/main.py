"""
CLI интерфейс для модуля распознавания действий в видео.
"""

import argparse
import sys
import json
import tempfile
import shutil
from pathlib import Path
from typing import Optional

# Добавляем путь к emotion_face для импорта FrameManager
emotion_face_path = Path(__file__).parent.parent / "emotion_face"
if str(emotion_face_path) not in sys.path:
    sys.path.insert(0, str(emotion_face_path))

from action_recognition_videomae import VideoMAEActionRecognizer
try:
    from utils import FrameManager, frame_writer  # type: ignore
except ImportError:
    # Альтернативный путь импорта
    import importlib.util
    spec = importlib.util.spec_from_file_location("utils", emotion_face_path / "utils.py")
    utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils)
    FrameManager = utils.FrameManager
    frame_writer = utils.frame_writer


def process_video(
    video_path: str,
    model_dir: str,
    output_path: str,
    frames_dir: Optional[str] = None,
    clip_len: int = 16,
    batch_size: int = 8,
    frame_skip: int = 1,
    max_tracks: Optional[int] = None
):
    """
    Обрабатывает видео и распознает действия.
    
    Args:
        video_path: Путь к входному видео
        model_dir: Путь к директории с моделью VideoMAE
        output_path: Путь для сохранения результатов JSON
        frames_dir: Опциональная директория с уже извлеченными кадрами
        clip_len: Длина клипа в кадрах
        batch_size: Размер батча для inference
        frame_skip: Пропуск кадров (обрабатывать каждый N-й кадр)
        max_tracks: Максимальное количество треков для обработки
    """
    video_path = Path(video_path)
    if not video_path.exists():
        raise FileNotFoundError(f"Видео файл не найден: {video_path}")
    
    model_dir = Path(model_dir)
    if not model_dir.exists():
        raise FileNotFoundError(f"Директория с моделью не найдена: {model_dir}")
    
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Определяем, нужно ли извлекать кадры
    tmp_dir = None
    use_tmp = frames_dir is None
    
    try:
        if use_tmp:
            # Создаем временную директорию для кадров
            tmp_dir = tempfile.mkdtemp(prefix="action_recognition_frames_")
            frames_dir = tmp_dir
            print(f"[INFO] Извлекаю кадры из видео в: {frames_dir}")
            
            # Извлекаем кадры
            meta = frame_writer(str(video_path), frames_dir, batch_size=64)
            total_frames = meta["total_frames"]
            fps = meta.get("fps", 30.0)
            print(f"[INFO] Извлечено кадров: {total_frames}, FPS: {fps}")
        else:
            # Используем существующую директорию с кадрами
            frames_dir = Path(frames_dir)
            meta_path = frames_dir / "metadata.json"
            if not meta_path.exists():
                raise FileNotFoundError(f"metadata.json не найден в {frames_dir}")
            meta = json.load(open(meta_path, "r", encoding="utf-8"))
            total_frames = meta["total_frames"]
            fps = meta.get("fps", 30.0)
            print(f"[INFO] Использую существующие кадры: {total_frames} кадров, FPS: {fps}")
        
        # Инициализируем FrameManager
        fm = FrameManager(frames_dir, chunk_size=64)
        fm.fps = fps  # Устанавливаем FPS для правильных вычислений
        
        # Инициализируем распознаватель действий
        print(f"[INFO] Загружаю модель из: {model_dir}")
        recognizer = VideoMAEActionRecognizer(
            frame_manager=fm,
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
            "video_path": str(video_path),
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
        
        # Сохраняем результаты
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)
        
        print(f"[SUCCESS] Результаты сохранены в: {output_path}")
        print(f"[INFO] Обработано треков: {len(results)}")
        
        # Выводим краткую статистику
        for track_id, track_results in results.items():
            print(f"\n  Трек {track_id}:")
            print(f"    Доминирующее действие: {track_results.get('dominant_action_label', 'unknown')}")
            print(f"    Уверенность: {track_results.get('dominant_confidence', 0.0):.2f}")
            print(f"    Сложность действия: {track_results.get('complexity_score', 0.0):.2f}")
            print(f"    Тип активности сцены: {track_results.get('scene_activity_type', 'unknown')}")
        
        return output_data
        
    finally:
        # Закрываем FrameManager
        if 'fm' in locals():
            fm.close()
        
        # Удаляем временную директорию
        if use_tmp and tmp_dir and Path(tmp_dir).exists():
            print(f"[INFO] Удаляю временные файлы: {tmp_dir}")
            shutil.rmtree(tmp_dir)


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
    
    # Входные данные
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Путь к входному видео файлу'
    )
    
    parser.add_argument(
        '--model', '-m',
        type=str,
        required=True,
        help='Путь к директории с моделью VideoMAE'
    )
    
    # Выходные данные
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Путь к выходному JSON файлу с результатами (по умолчанию: input_name_actions.json)'
    )
    
    # Опциональные параметры
    parser.add_argument(
        '--frames-dir',
        type=str,
        default=None,
        help='Директория с уже извлеченными кадрами (если не указана, кадры будут извлечены автоматически)'
    )
    
    parser.add_argument(
        '--clip-len',
        type=int,
        default=16,
        help='Длина клипа в кадрах для обработки'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=8,
        help='Размер батча для inference'
    )
    
    parser.add_argument(
        '--frame-skip',
        type=int,
        default=1,
        help='Обрабатывать каждый N-й кадр (для ускорения)'
    )
    
    parser.add_argument(
        '--max-tracks',
        type=int,
        default=None,
        help='Максимальное количество треков для обработки'
    )
    
    args = parser.parse_args()
    
    # Проверка входного файла
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Ошибка: файл не найден: {args.input}")
        sys.exit(1)
    
    # Проверка модели
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"Ошибка: директория с моделью не найдена: {args.model}")
        sys.exit(1)
    
    # Определение выходного файла
    if args.output is None:
        output_path = input_path.parent / f"{input_path.stem}_actions.json"
    else:
        output_path = Path(args.output)
    
    print(f"[INFO] Начинаю распознавание действий: {args.input}")
    print(f"[INFO] Выходной файл: {output_path}")
    print(f"[INFO] Модель: {args.model}")
    print(f"[INFO] Параметры: clip_len={args.clip_len}, batch_size={args.batch_size}, frame_skip={args.frame_skip}")
    
    # Обработка видео
    try:
        import numpy as np  # Для конвертации типов
        
        results = process_video(
            video_path=str(input_path),
            model_dir=str(model_path),
            output_path=str(output_path),
            frames_dir=args.frames_dir,
            clip_len=args.clip_len,
            batch_size=args.batch_size,
            frame_skip=args.frame_skip,
            max_tracks=args.max_tracks
        )
        
        print(f"\n[SUCCESS] Обработка завершена успешно!")
        
    except Exception as e:
        print(f"\n[ERROR] Ошибка при обработке: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
