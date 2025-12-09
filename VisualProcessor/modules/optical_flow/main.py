from pathlib import Path

from core.flow_statistics import FlowStatisticsAnalyzer
from core.optical_flow import OpticalFlowProcessor
from core.utils import create_frame_metadata_csv
from core.config import FlowPipelineConfig, FlowStatisticsConfig

if __name__ == "__main__":
    import argparse
    import sys
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description='Production пайплайн обработки видео с RAFT')
    parser.add_argument('video_path', type=str, nargs='?', help='Путь к видео файлу')
    parser.add_argument('--output', type=str, default='raft_output', help='Директория для результатов')
    parser.add_argument('--model', type=str, default='small', choices=['small', 'large'], help='Модель RAFT')
    parser.add_argument('--max_dim', type=int, default=256, help='Максимальный размер стороны')
    parser.add_argument('--skip', type=int, default=15, help='Шаг выборки кадров')
    parser.add_argument('--no_overlay', action='store_true', help='Не сохранять overlay')
    parser.add_argument('--run_stats', action='store_true', help='Запустить статистический анализ')
    parser.add_argument('--run_camera_motion', action='store_true', help='Запустить анализ движения камеры')
    parser.add_argument('--no_advanced_features', action='store_true', help='Отключить продвинутые фичи (MEI, FG/BG, Clusters, Smoothness)')
    
    args = parser.parse_args()

    # Определяем путь к видео
    if args.video_path:
        video_path = args.video_path
    else:
        # Пробуем найти видео в текущей директории
        video_files = list(Path('.').glob('*.mp4'))
        if video_files:
            video_path = str(video_files[0])
            print(f"Используется видео: {video_path}")
        else:
            print("Ошибка: не указан путь к видео и не найдено видео в текущей директории")
            print("Использование: python main.py <путь_к_видео> [опции]")
            sys.exit(1)
    
    # Проверяем существование файла
    if not Path(video_path).exists():
        print(f"Ошибка: файл не найден: {video_path}")
        sys.exit(1)
    
    flow_config = FlowPipelineConfig(
        output_dir=args.output,
        model_type=args.model,
        max_dimension=args.max_dim,
        frame_skip=args.skip,
        save_overlay=not args.no_overlay,
        save_flow_tensors=True
    )

    stats_config = FlowStatisticsConfig() if args.run_stats else None
    if stats_config:
        stats_config.enable_camera_motion = args.run_camera_motion
        stats_config.enable_advanced_features = not args.no_advanced_features

    # 1. Обработка оптического потока
    flow_processor = OpticalFlowProcessor(flow_config)
    flow_results = flow_processor.process_video(video_path)
    
    # 2. Создание CSV метаданных
    csv_path = create_frame_metadata_csv(
        Path(flow_results['flow_dir']),
        flow_results['metadata']
    )
    
    # 3. Статистический анализ (опционально)
    stats_results = None
    if stats_config is not None:
        stats_analyzer = FlowStatisticsAnalyzer(stats_config)
        stats_results = stats_analyzer.analyze_video(
            flow_results['flow_dir'],
            flow_results['metadata']
        )

    print({
        'flow_processing': flow_results,
        'statistical_analysis': stats_results,
        'metadata_csv': str(csv_path) if csv_path else None
    })