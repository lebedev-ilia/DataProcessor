import torch
from typing import Dict, Any, Union, List
import json
from pathlib import Path
import pandas as pd
import logging
import warnings

# Настройка логирования
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Подавление предупреждений
warnings.filterwarnings('ignore', category=RuntimeWarning)
warnings.filterwarnings('ignore', category=UserWarning)

def create_frame_metadata_csv(flow_dir: Path, metadata: Dict[str, Any]) -> Path:
    """
    Создание CSV с метаданными кадров.
    Совместимо с flow_statistics модулем.
    """
    import pandas as pd
    import numpy as np
    
    flow_files = sorted([f for f in flow_dir.glob("*.pt")])
    frame_data = []
    
    for flow_file in flow_files:
        try:
            frame_idx = int(flow_file.stem.split('_')[1])
            
            # Базовая статистика (можно заменить на вызов flow_statistics)
            flow_tensor = torch.load(flow_file)
            dx = flow_tensor[0].numpy()
            dy = flow_tensor[1].numpy()
            
            magnitude = np.sqrt(dx**2 + dy**2)
            direction = np.arctan2(dy, dx)
            
            frame_stats = {
                'frame_id': f"{metadata['video_id']}_{frame_idx:06d}",
                'video_id': metadata['video_id'],
                'frame_index': frame_idx,
                'original_frame_idx': frame_idx * metadata['processing_parameters']['frame_skip'],
                'flow_filename': flow_file.name,
                'timestamp_seconds': frame_idx * metadata['processing_parameters']['frame_skip'] / 
                                   metadata['video_properties']['fps'],
                'mean_magnitude': float(np.mean(magnitude)),
                'std_magnitude': float(np.std(magnitude)),
                'moving_pixels_ratio': float(np.sum(magnitude > 0.5) / magnitude.size)
            }
            
            frame_data.append(frame_stats)
            
        except Exception as e:
            print(f"Ошибка обработки {flow_file}: {e}")
            continue
    
    if frame_data:
        df = pd.DataFrame(frame_data)
        csv_path = flow_dir.parent / "frame_metadata.csv"
        df.to_csv(csv_path, index=False, encoding='utf-8')
        return csv_path
    
    return None


# Утилитарные функции для удобного использования
def analyze_single_video(
        flow_dir: Union[str, Path], 
        metadata_path: Union[str, Path],
        config = None,
        analyzer = None
    ) -> Dict[str, Any]:
    """
    Анализ одного видео.
    
    Args:
        flow_dir: Директория с flow файлами
        metadata_path: Путь к metadata.json
        config: Конфигурация анализа
        
    Returns:
        Результаты анализа
    """
    # Загрузка метаданных
    try:
        with open(metadata_path, 'r', encoding='utf-8') as f:
            video_metadata = json.load(f)
    except Exception as e:
        logger.error(f"Ошибка загрузки метаданных {metadata_path}: {e}")
        raise
    
    return analyzer.analyze_video(flow_dir, video_metadata)

def batch_analysis(root_dir: Union[str, Path], 
                  config = None,
                  limit: int = None,
                  parallel: bool = False) -> List[Dict[str, Any]]:
    """
    Пакетный анализ нескольких видео.
    
    Args:
        root_dir: Корневая директория с видео
        config: Конфигурация анализа
        limit: Максимальное количество видео для анализа
        parallel: Использовать параллельную обработку
        
    Returns:
        Список результатов анализа
    """
    root_dir = Path(root_dir)
    all_results = []
    
    # Поиск папок с видео
    video_folders = []
    for item in root_dir.iterdir():
        if item.is_dir():
            metadata_file = item / 'metadata.json'
            flow_dir = item / 'flow'
            if metadata_file.exists() and flow_dir.exists():
                video_folders.append(item)
    
    logger.info(f"Найдено {len(video_folders)} видео для анализа")
    
    if limit:
        video_folders = video_folders[:limit]
        logger.info(f"Будет проанализировано {len(video_folders)} видео (лимит: {limit})")
    
    if parallel:
        # Параллельная обработка
        from concurrent.futures import ProcessPoolExecutor
        with ProcessPoolExecutor() as executor:
            futures = []
            for folder in video_folders:
                future = executor.submit(
                    analyze_single_video,
                    folder / 'flow',
                    folder / 'metadata.json',
                    config
                )
                futures.append((folder, future))
            
            for folder, future in futures:
                try:
                    result = future.result(timeout=3600)  # Таймаут 1 час
                    all_results.append(result)
                    logger.info(f"Завершен анализ: {folder.name}")
                except Exception as e:
                    logger.error(f"Ошибка анализа {folder}: {e}")
    else:
        # Последовательная обработка
        for i, folder in enumerate(video_folders, 1):
            logger.info(f"[{i}/{len(video_folders)}] Анализируем: {folder.name}")
            try:
                result = analyze_single_video(
                    folder / 'flow',
                    folder / 'metadata.json',
                    config
                )
                all_results.append(result)
            except Exception as e:
                logger.error(f"Ошибка анализа {folder}: {e}")
    
    # Создание сравнительного отчета
    if all_results:
        create_comparative_report(all_results, root_dir)
    
    return all_results

def create_comparative_report(results: List[Dict[str, Any]], 
                             output_dir: Union[str, Path]) -> None:
    """
    Создание сравнительного отчета по всем видео.
    
    Args:
        results: Список результатов анализа
        output_dir: Директория для сохранения отчета
    """
    try:
        summary_data = []
        for result in results:
            if 'error' in result:
                continue
            
            metrics = result['statistics']['summary_metrics']
            summary_data.append({
                'video_id': result['video_metadata']['video_id'],
                'video_name': result['video_metadata']['video_filename'],
                'magnitude_mean': metrics.get('overall_magnitude_mean', 0),
                'motion_intensity': metrics.get('dominant_motion_intensity', 'unknown'),
                'stability': metrics.get('temporal_stability', 0),
                'peak_count': metrics.get('peak_activity_frames', 0),
                'frames_analyzed': result['processing_info']['total_frames_analyzed']
            })
        
        if summary_data:
            df = pd.DataFrame(summary_data)
            
            # Сохраняем CSV
            csv_path = Path(output_dir) / 'comparative_summary.csv'
            df.to_csv(csv_path, index=False, encoding='utf-8')
            
            # Создаем отчет
            report = f"""# Сравнительный отчет анализа {len(summary_data)} видео

            ## Общая статистика
            - Средняя скорость: {df['magnitude_mean'].mean():.2f} ± {df['magnitude_mean'].std():.2f} px/frame
            - Средняя стабильность: {df['stability'].mean():.2f}/1.0
            - Всего проанализировано кадров: {df['frames_analyzed'].sum():,}

            ## Распределение по интенсивности
            {df['motion_intensity'].value_counts().to_string()}

            ## Топ-5 самых активных видео
            {df.nlargest(5, 'magnitude_mean')[['video_name', 'magnitude_mean']].to_string(index=False)}

            ## Топ-5 самых стабильных видео
            {df.nlargest(5, 'stability')[['video_name', 'stability']].to_string(index=False)}

            ## Детальная статистика доступна в comparative_summary.csv
            """
                        
            report_path = Path(output_dir) / 'comparative_report.md'
            report_path.write_text(report, encoding='utf-8')
            
            logger.info(f"Сравнительный отчет сохранен в {output_dir}")
    except Exception as e:
        logger.error(f"Ошибка создания сравнительного отчета: {e}")