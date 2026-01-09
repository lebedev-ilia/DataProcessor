"""
CLI для `optical_flow` (BaseModule, NPZ-only).

ВНИМАНИЕ:
- Этот модуль НЕ считает RAFT сам.
- Он потребляет `core_optical_flow/flow.npz` и сохраняет агрегаты/кривую в `optical_flow/*.npz`.
"""

import argparse
import os
import sys
from typing import Optional, List, Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from modules.optical_flow.optical_flow import OpticalFlowModule
from utils.logger import get_logger

MODULE_NAME = "optical_flow"
logger = get_logger(MODULE_NAME)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog=f"run_{MODULE_NAME}",
        description="Optical flow (consumer of core_optical_flow) — CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--frames-dir", required=True, help="Директория с кадрами (с metadata.json)")
    parser.add_argument("--rs-path", required=True, help="Папка result_store для артефактов")
    parser.add_argument("--log-level", type=str, default="INFO", help="DEBUG/INFO/WARN/ERROR")
    args = parser.parse_args(argv)

    try:
        import logging as _logging

        _logging.getLogger().setLevel(getattr(_logging, args.log_level.upper(), _logging.INFO))
    except Exception:
        logger.warning("Не удалось установить log-level: %s", args.log_level)

    try:
        module = OpticalFlowModule(rs_path=args.rs_path)
        config: Dict[str, Any] = {}
        saved_path = module.run(frames_dir=args.frames_dir, config=config)
        logger.info("Готово. Результаты сохранены: %s", saved_path)
        return 0
    except FileNotFoundError as e:
        logger.error("Файл не найден: %s", e)
        return 2
    except ValueError as e:
        logger.error("Некорректные данные: %s", e)
        return 3
    except Exception as e:
        logger.exception("Fatal error в %s: %s", MODULE_NAME, e)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())


