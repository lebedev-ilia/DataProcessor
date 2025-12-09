"""
CLI интерфейс для модуля анализа поведения людей в видео.
"""

import argparse
import sys
from pathlib import Path
from behavior_analyzer import BehaviorAnalyzer


def main():
    parser = argparse.ArgumentParser(
        description='Анализ поведения людей в видео',
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
        help='Путь к выходному JSON файлу с результатами (по умолчанию: input_name_behavior.json)'
    )
    
    # Параметры обработки
    parser.add_argument(
        '--frame-skip',
        type=int,
        default=1,
        help='Обрабатывать каждый N-й кадр (для ускорения)'
    )
    
    # Визуализация
    parser.add_argument(
        '--visualize',
        action='store_true',
        help='Сохранять визуализацию результатов на кадрах'
    )
    
    parser.add_argument(
        '--visualize-dir',
        type=str,
        default='./behavior_visualizations',
        help='Директория для сохранения визуализаций'
    )
    
    args = parser.parse_args()
    
    # Проверка входного файла
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Ошибка: файл не найден: {args.input}")
        sys.exit(1)
    
    # Определение выходного файла
    if args.output is None:
        output_path = input_path.parent / f"{input_path.stem}_behavior.json"
    else:
        output_path = Path(args.output)
    
    # Создание директории для визуализации
    if args.visualize:
        visualize_dir = Path(args.visualize_dir)
        visualize_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"[INFO] Начинаю анализ поведения: {args.input}")
    print(f"[INFO] Выходной файл: {output_path}")
    print(f"[INFO] Пропуск кадров: {args.frame_skip}")
    
    # Инициализация анализатора
    try:
        analyzer = BehaviorAnalyzer()
    except Exception as e:
        print(f"Ошибка при инициализации анализатора: {e}")
        sys.exit(1)
    
    # Обработка видео
    try:
        results = analyzer.process_video(
            video_path=str(input_path),
            output_path=str(output_path),
            frame_skip=args.frame_skip
        )
        
        if results.get('success'):
            print(f"\n[SUCCESS] Анализ завершен успешно!")
            print(f"  - Обработано кадров: {results['processed_frames']}")
            print(f"  - Всего кадров: {results['total_frames']}")
            print(f"  - FPS: {results['fps']}")
            
            if 'aggregated' in results:
                agg = results['aggregated']
                print(f"\n[РЕЗУЛЬТАТЫ]")
                print(f"  - Средний Engagement Index: {agg.get('avg_engagement', 0):.3f}")
                print(f"  - Средний Confidence Index: {agg.get('avg_confidence', 0):.3f}")
                print(f"  - Средний Stress Level: {agg.get('avg_stress', 0):.3f}")
                
                if 'gesture_statistics' in agg:
                    print(f"\n[ЖЕСТЫ]")
                    for gesture, count in agg['gesture_statistics'].items():
                        print(f"  - {gesture}: {count}")
                
                if 'posture_statistics' in agg:
                    print(f"\n[ПОЗЫ]")
                    for posture, count in agg['posture_statistics'].items():
                        print(f"  - {posture}: {count}")
            
            print(f"\n[INFO] Результаты сохранены в: {output_path}")
        else:
            print(f"[ERROR] Ошибка при обработке: {results.get('error', 'Unknown error')}")
            sys.exit(1)
    
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
        if 'analyzer' in locals():
            del analyzer


if __name__ == "__main__":
    main()

