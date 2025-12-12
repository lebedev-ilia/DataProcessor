"""
CLI интерфейс для модуля анализа цвета и освещения видео.
"""

import argparse
import sys
import json
import numpy as np
from processor import ColorLightProcessor

import os
import sys
_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

if _path not in sys.path:
    sys.path.append(_path)
from utils.frame_manager import FrameManager
from utils.results_store import ResultsStore

name = "color_light"

def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

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
    
    parser.add_argument('--rs-path', type=str, default=None)
    parser.add_argument('--frames-dir', type=str, default=None, help='Директория с уже извлеченными кадрами (если не указана, кадры будут извлечены автоматически)')

    parser.add_argument('--max-frames-per-scene', type=int, default=350, help='Максимальное количество кадров для обработки на сцену')

    args = parser.parse_args()
        
    # Инициализируем процессор
    processor = ColorLightProcessor(max_frames_per_scene=args.max_frames_per_scene)

    metadata = load_json(f"{args.frames_dir}/metadata.json")

    frame_manager = FrameManager(frames_dir=args.frames_dir, chunk_size=metadata["chunk_size"], cache_size=metadata["cache_size"])

    rs = ResultsStore(args.rs_path)
    
    p = f"{args.rs_path}/scene_classification"
    f = os.listdir(p)[-1]
    scenes = load_json(f"{p}/{f}")

    result = processor.process(frame_manager=frame_manager, scenes=scenes)
    
    # Конвертируем numpy типы в JSON-совместимые
    result_json = _convert_to_json(result)

    rs.store(result_json, nema=name)
    
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
        

if __name__ == "__main__":
    main()

