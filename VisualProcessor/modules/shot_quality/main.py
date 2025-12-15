"""
CLI интерфейс для модуля анализа качества кадров видео.
"""
import json
import argparse
from shot_quality_pipline import ShotQualityPipeline

import os
import sys
_path = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))

if _path not in sys.path:
    sys.path.append(_path)

from utils.frame_manager import FrameManager
from utils.results_store import ResultsStore

name = "shot_quality"

def load_metadata(meta_path):
    try:
        with open(meta_path, "r") as f:
            return json.load(f)
    except:
        raise f"{name} | main | load_metadata | Ошибка при открытии метадаты"

def main():
    parser = argparse.ArgumentParser(
        description='Анализ качества кадров видео',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument('--frames-dir', '-i', type=str, required=True, help='Путь к входному видео файлу')
    parser.add_argument('--rs-path', '-o', type=str, default=None, help='Путь к выходному JSON файлу с результатами (по умолчанию: input_name_shot_quality.json)')
    parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Устройство для обработки (cuda или cpu)')
    
    args = parser.parse_args()
    
    pipeline = ShotQualityPipeline(device=args.device)

    rs = ResultsStore(args.rs_path)

    metadata = load_metadata(f"{args.frames_dir}/metadata.json")

    frame_manager = FrameManager(args.frames_dir, chunk_size=metadata["chunk_size"], cache_size=metadata["cache_size"])
    frame_indices = metadata[name]["frame_indices"]

    results = pipeline.process(frame_manager, frame_indices)

    rs.store(results, name=name)

if __name__ == "__main__":
    main()

