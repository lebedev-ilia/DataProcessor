"""
CLI интерфейс для модуля анализа качества кадров видео.
"""

import argparse
import sys
import json
import numpy as np
from pathlib import Path
from typing import Optional
import cv2
from shot_quality_pipline import ShotQualityPipeline


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
    elif isinstance(obj, (bool, str, int, float)) or obj is None:
        return obj
    else:
        return str(obj)


def extract_frames(video_path: str, frame_skip: int = 1, max_frames: Optional[int] = None):
    """
    Извлекает кадры из видео.
    
    Args:
        video_path: Путь к видео файлу
        frame_skip: Обрабатывать каждый N-й кадр
        max_frames: Максимальное количество кадров для обработки
        
    Returns:
        Список кадров (RGB)
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Не удалось открыть видео: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    frames = []
    frame_count = 0
    
    print(f"[INFO] Извлекаю кадры из видео...")
    print(f"[INFO] Всего кадров: {total_frames}, FPS: {fps:.2f}")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_skip == 0:
            # Конвертируем BGR в RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
            
            if len(frames) % 10 == 0:
                print(f"[INFO] Извлечено кадров: {len(frames)}")
        
        frame_count += 1
        
        if max_frames and len(frames) >= max_frames:
            break
    
    cap.release()
    print(f"[INFO] Извлечено кадров: {len(frames)}")
    
    return frames, fps, total_frames


def main():
    parser = argparse.ArgumentParser(
        description='Анализ качества кадров видео',
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
        help='Путь к выходному JSON файлу с результатами (по умолчанию: input_name_shot_quality.json)'
    )
    
    # Параметры обработки
    parser.add_argument(
        '--frame-skip',
        type=int,
        default=1,
        help='Обрабатывать каждый N-й кадр (для ускорения)'
    )
    
    parser.add_argument(
        '--max-frames',
        type=int,
        default=None,
        help='Максимальное количество кадров для обработки'
    )
    
    parser.add_argument(
        '--device',
        type=str,
        default='cuda',
        choices=['cuda', 'cpu'],
        help='Устройство для обработки (cuda или cpu)'
    )
    
    args = parser.parse_args()
    
    # Проверка входного файла
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"[ERROR] Файл не найден: {args.input}")
        sys.exit(1)
    
    # Определение выходного файла
    if args.output is None:
        output_path = input_path.parent / f"{input_path.stem}_shot_quality.json"
    else:
        output_path = Path(args.output)
    
    print(f"[INFO] Начинаю анализ качества кадров: {args.input}")
    print(f"[INFO] Выходной файл: {output_path}")
    print(f"[INFO] Пропуск кадров: {args.frame_skip}")
    print(f"[INFO] Устройство: {args.device}")
    
    # Инициализация pipeline
    try:
        print(f"[INFO] Загружаю модели...")
        pipeline = ShotQualityPipeline(device=args.device)
        print(f"[INFO] Модели загружены успешно")
    except Exception as e:
        print(f"[ERROR] Ошибка при инициализации pipeline: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    # Извлечение кадров
    try:
        frames, fps, total_frames = extract_frames(
            str(input_path),
            frame_skip=args.frame_skip,
            max_frames=args.max_frames
        )
        
        if len(frames) == 0:
            print(f"[ERROR] Не удалось извлечь кадры из видео")
            sys.exit(1)
        
        # Обработка видео
        print(f"[INFO] Обрабатываю {len(frames)} кадров...")
        results = pipeline.process(frames, frame_skip=1)
        
        # Конвертируем numpy типы в JSON-совместимые
        results_json = _convert_to_json(results)
        
        # Добавляем метаданные
        results_json["metadata"] = {
            "video_path": str(input_path),
            "fps": float(fps),
            "total_frames": int(total_frames),
            "processed_frames": len(frames),
            "frame_skip": args.frame_skip
        }
        
        # Сохраняем результаты
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(results_json, f, indent=2, ensure_ascii=False)
        
        print(f"\n[SUCCESS] Анализ завершен успешно!")
        print(f"  - Обработано кадров: {len(frames)}")
        print(f"  - Всего кадров в видео: {total_frames}")
        print(f"  - FPS: {fps:.2f}")
        
        # Выводим некоторые ключевые метрики
        if results.get("frame_features"):
            ff = results["frame_features"]
            print(f"\n[РЕЗУЛЬТАТЫ]")
            if "avg_sharpness_laplacian" in ff:
                print(f"  - Средняя резкость (Laplacian): {ff['avg_sharpness_laplacian']:.2f}")
            if "avg_noise_level_luma" in ff:
                print(f"  - Средний уровень шума (Luma): {ff['avg_noise_level_luma']:.3f}")
            if "avg_underexposure_ratio" in ff:
                print(f"  - Средний недоэкспонированный: {ff['avg_underexposure_ratio']:.3f}")
            if "avg_overexposure_ratio" in ff:
                print(f"  - Средний переэкспонированный: {ff['avg_overexposure_ratio']:.3f}")
            if "avg_quality_cinematic_prob" in ff:
                print(f"  - Вероятность кинематографического качества: {ff['avg_quality_cinematic_prob']:.3f}")
            if "avg_aesthetic_score" in ff:
                print(f"  - Средний эстетический score: {ff['avg_aesthetic_score']:.3f}")
        
        if results.get("temporal_features"):
            tf = results["temporal_features"]
            print(f"\n[ВРЕМЕННЫЕ МЕТРИКИ]")
            if "temporal_sharpness_stability" in tf:
                print(f"  - Стабильность резкости: {tf['temporal_sharpness_stability']:.3f}")
            if "temporal_exposure_stability" in tf:
                print(f"  - Стабильность экспозиции: {tf['temporal_exposure_stability']:.3f}")
            if "temporal_noise_variation" in tf:
                print(f"  - Вариация шума: {tf['temporal_noise_variation']:.3f}")
            if "rolling_shutter_artifacts_score" in results.get("frame_features", {}):
                rs_avg = results["frame_features"].get("avg_rolling_shutter_artifacts_score", 0)
                print(f"  - Rolling shutter artifacts: {rs_avg:.3f}")
        
        print(f"\n[INFO] Результаты сохранены в: {output_path}")
        
    except KeyboardInterrupt:
        print("\n[INFO] Прервано пользователем")
        sys.exit(0)
    except Exception as e:
        print(f"[ERROR] Критическая ошибка: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    finally:
        # Очистка ресурсов
        if 'pipeline' in locals():
            del pipeline


if __name__ == "__main__":
    main()

