import os
import subprocess
import sys

_path = os.path.dirname(__file__)

if _path not in sys.path:
    sys.path.append(_path)


from utils.logger import get_logger
logger = get_logger("DataProcessor")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--video-path',    type=str,   default="NSumhkOwSg.mp4", help='')
    parser.add_argument('--output',        type=str,   default=f"{_path}/Segmenter/data", help='')
    parser.add_argument('--chunk-size',    type=int,   default=64, help=f'')

    parser.add_argument('--visual-cfg-path',    type=str,   default=f"{_path}/VisualProcessor/config.yaml", help='')
    args = parser.parse_args()

    # cmd = [
    #     f"{_path}/.data_venv/bin/python",
    #     f"{_path}/Segmenter/segmenter.py",
    #     "--video-path", f"{args.video_path}",
    #     "--output", args.output,
    #     "--chunk-size", str(args.chunk_size)
    # ]

    # subprocess.run(cmd)

    logger.info("DataProcessor | main | Запуск main VisualProcessor")

    cmd = [
        f"{_path}/VisualProcessor/.venv/bin/python",
        f"{_path}/VisualProcessor/main.py",
        "--cfg-path", args.visual_cfg_path
    ]

    subprocess.run(cmd)

