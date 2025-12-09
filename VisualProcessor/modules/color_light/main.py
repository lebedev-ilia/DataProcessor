"""
CLI интерфейс для модуля анализа цвета и освещения видео.
"""

import argparse
import sys
import json
import tempfile
import shutil
from pathlib import Path
import cv2
import numpy as np
from processor import FrameManager, ColorLightProcessor


def extract_frames(video_path: str, out_dir: str, batch_size: int = 64):
    """
    Извлекает кадры из видео и создает metadata.json для FrameManager.
    
    Args:
        video_path: Путь к видео файлу
        out_dir: Директория для сохранения кадров
        batch_size: Размер батча кадров
        
    Returns:
        dict: Метаданные с информацией о кадрах
    """
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {video_path}")
    
    # Получаем FPS из видео
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30.0  # значение по умолчанию
    
    meta_path = Path(out_dir) / "metadata.json"
    
    # Если metadata.json уже существует, возвращаем его
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        cap.release()
        return meta
    
    meta = {
        "total_frames": 0,
        "batch_size": batch_size,
        "fps": fps,
        "batches": []
    }
    
    H = W = C = None
    batch_id = 0
    buf_count = 0
    arr = None
    current_path = None
    
    print(f"[INFO] Извлекаю кадры из видео...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Конвертируем BGR в RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        if H is None:
            H, W, C = frame.shape
            meta["height"] = H
            meta["width"] = W
            meta["channels"] = C
        
        # Фиксируем возможные битые размеры
        if frame.shape != (H, W, C):
            frame = cv2.resize(frame, (W, H), interpolation=cv2.INTER_AREA)
        
        # Создаём memmap под батч
        if buf_count == 0:
            fname = f"batch_{batch_id:05d}.raw"
            current_path = Path(out_dir) / fname
            
            arr = np.memmap(
                str(current_path),
                dtype=np.uint8,
                mode="w+",
                shape=(batch_size, H, W, C)
            )
        
        arr[buf_count] = frame
        buf_count += 1
        meta["total_frames"] += 1
        
        # Батч заполнен → записываем мету, закрываем
        if buf_count == batch_size:
            arr.flush()
            del arr
            arr = None
            
            if batch_id % 10 == 0:
                print(f"[INFO] Обработано кадров: {meta['total_frames']}")
            
            meta["batches"].append({
                "batch_index": batch_id,
                "path": fname,
                "start_frame": batch_id * batch_size,
                "end_frame": batch_id * batch_size + buf_count - 1
            })
            
            buf_count = 0
            batch_id += 1
    
    # Последний неполный батч
    if buf_count > 0:
        arr.flush()
        # Обрезаем файл до реального размера
        frame_bytes = H * W * C
        actual_bytes = buf_count * frame_bytes
        with open(current_path, "r+b") as f:
            f.truncate(actual_bytes)
        del arr
        arr = None
        
        meta["batches"].append({
            "batch_index": batch_id,
            "path": fname,
            "start_frame": batch_id * batch_size,
            "end_frame": batch_id * batch_size + buf_count - 1
        })
    
    # Сохраняем metadata.json
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    
    cap.release()
    print(f"[INFO] Извлечено кадров: {meta['total_frames']}, FPS: {fps}")
    
    return meta


def create_scenes_from_video(total_frames: int, fps: float, scene_duration: float = 5.0):
    """
    Создает сцены из видео с заданной длительностью.
    
    Args:
        total_frames: Общее количество кадров
        fps: Частота кадров
        scene_duration: Длительность сцены в секундах
        
    Returns:
        dict: Словарь сцен {scene_id: [start_frame, end_frame]}
    """
    frames_per_scene = int(scene_duration * fps)
    scenes = {}
    scene_id = 1
    
    for start_frame in range(0, total_frames, frames_per_scene):
        end_frame = min(start_frame + frames_per_scene - 1, total_frames - 1)
        scenes[scene_id] = [start_frame, end_frame]
        scene_id += 1
    
    return scenes


def _convert_to_json(obj):
    """Рекурсивно конвертирует numpy типы в JSON-совместимые."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: _convert_to_json(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [_convert_to_json(item) for item in obj]
    else:
        return obj


def main():
    parser = argparse.ArgumentParser(
        description='Анализ цвета и освещения видео',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Входные данные
    parser.add_argument(
        '--input', '-i',
        type=str,
        required=True,
        help='Путь к входному видео файлу'
    )
    
    # Выходные данные
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Путь к выходному JSON файлу с результатами (по умолчанию: input_name_color_light.json)'
    )
    
    # Параметры обработки
    parser.add_argument(
        '--frames-dir',
        type=str,
        default=None,
        help='Директория с уже извлеченными кадрами (если не указана, кадры будут извлечены автоматически)'
    )
    
    parser.add_argument(
        '--max-frames-per-scene',
        type=int,
        default=350,
        help='Максимальное количество кадров для обработки на сцену'
    )
    
    parser.add_argument(
        '--scene-duration',
        type=float,
        default=5.0,
        help='Длительность сцены в секундах (если сцены не указаны явно)'
    )
    
    parser.add_argument(
        '--scenes',
        type=str,
        default=None,
        help='JSON файл со сценами в формате {"scene_id": [start_frame, end_frame]} или путь к JSON'
    )
    
    parser.add_argument(
        '--keep-frames',
        action='store_true',
        help='Сохранять извлеченные кадры после обработки (по умолчанию удаляются)'
    )
    
    args = parser.parse_args()
    
    # Проверка входного файла
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Файл не найден: {args.input}")
        sys.exit(1)
    
    # Определение выходного файла
    if args.output is None:
        output_path = input_path.parent / f"{input_path.stem}_color_light.json"
    else:
        output_path = Path(args.output)
    
    print(f"[INFO] Начинаю анализ цвета и освещения: {args.input}")
    print(f"[INFO] Выходной файл: {output_path}")
    
    tmp_dir = None
    use_tmp = args.frames_dir is None
    
    try:
        if use_tmp:
            # Создаем временную директорию для кадров
            tmp_dir = tempfile.mkdtemp(prefix="color_light_frames_")
            frames_dir = tmp_dir
            print(f"[INFO] Извлекаю кадры из видео в: {frames_dir}")
            
            # Извлекаем кадры
            meta = extract_frames(str(input_path), frames_dir, batch_size=64)
            total_frames = meta["total_frames"]
            fps = meta.get("fps", 30.0)
        else:
            # Используем существующую директорию с кадрами
            frames_dir = Path(args.frames_dir)
            meta_path = frames_dir / "metadata.json"
            if not meta_path.exists():
                print(f"[ERROR] metadata.json не найден в {frames_dir}")
                sys.exit(1)
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            total_frames = meta["total_frames"]
            fps = meta.get("fps", 30.0)
            print(f"[INFO] Использую существующие кадры: {total_frames} кадров, FPS: {fps}")
        
        # Инициализируем FrameManager
        frame_manager = FrameManager(frames_dir, chunk_size=64, cache_size=2)
        frame_manager.fps = fps
        
        # Подготавливаем сцены
        if args.scenes:
            # Загружаем сцены из файла
            scenes_path = Path(args.scenes)
            if scenes_path.exists():
                with open(scenes_path, "r", encoding="utf-8") as f:
                    scenes = json.load(f)
            else:
                # Пытаемся распарсить как JSON строку
                scenes = json.loads(args.scenes)
        else:
            # Создаем сцены автоматически
            scenes = create_scenes_from_video(total_frames, fps, args.scene_duration)
            print(f"[INFO] Создано сцен: {len(scenes)}")
        
        # Инициализируем процессор
        processor = ColorLightProcessor(max_frames_per_scene=args.max_frames_per_scene)
        
        # Подготавливаем входные данные
        input_data = {
            "total_frames": total_frames,
            "scenes": scenes
        }
        
        # Обработка видео
        print(f"[INFO] Обрабатываю видео...")
        result = processor.process(
            frame_manager=frame_manager,
            input_data=input_data,
            video_id=input_path.stem
        )
        
        # Конвертируем numpy типы в JSON-совместимые
        result_json = _convert_to_json(result)
        
        # Сохраняем результаты
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(result_json, f, indent=2, ensure_ascii=False)
        
        print(f"\n[SUCCESS] Анализ завершен успешно!")
        print(f"  - Обработано кадров: {len(result['frames'])}")
        print(f"  - Обработано сцен: {len(result['scenes'])}")
        print(f"  - Видео фич: {len(result['video_features'])}")
        
        # Выводим некоторые ключевые метрики
        if result['video_features']:
            vf = result['video_features']
            print(f"\n[РЕЗУЛЬТАТЫ]")
            if 'cinematic_lighting_score' in vf:
                print(f"  - Cinematic Lighting Score: {vf['cinematic_lighting_score']:.3f}")
            if 'professional_look_score' in vf:
                print(f"  - Professional Look Score: {vf['professional_look_score']:.3f}")
            if 'style_teal_orange_prob' in vf:
                print(f"  - Teal & Orange Style: {vf['style_teal_orange_prob']:.3f}")
            if 'color_distribution_entropy' in vf:
                print(f"  - Color Distribution Entropy: {vf['color_distribution_entropy']:.3f}")
        
        print(f"\n[INFO] Результаты сохранены в: {output_path}")
        
        # Очистка
        frame_manager.close()
        
    except KeyboardInterrupt:
        print("\n[INFO] Прервано пользователем")
        sys.exit(0)
    except Exception as e:
        print(f"[ERROR] Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Удаляем временную директорию, если не нужно сохранять кадры
        if tmp_dir and not args.keep_frames:
            print(f"[INFO] Удаляю временные файлы: {tmp_dir}")
            shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()

