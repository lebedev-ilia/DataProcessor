"""
CLI для `story_structure` (BaseModule, NPZ output).

Baseline-версия: story/energy/coherence по `core_clip` embeddings (без локальных ML моделей).
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Optional, List, Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from modules.story_structure.story_structure import StoryStructureBaselineModule
from utils.logger import get_logger

MODULE_NAME = "story_structure"
logger = get_logger(MODULE_NAME)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog=f"run_{MODULE_NAME}",
        description="Story structure (baseline) — CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--frames-dir", required=True, help="Директория с кадрами (с metadata.json)")
    parser.add_argument("--rs-path", required=True, help="Папка result_store для артефактов")

    # Эти аргументы остаются для совместимости с config.yaml, но baseline их не использует напрямую.
    parser.add_argument("--clip-model", type=str, default=None, help="legacy/compat (unused in baseline)")
    parser.add_argument("--sentence-model", type=str, default=None, help="legacy/compat (unused in baseline)")
    parser.add_argument("--subtitles", type=str, default=None, help="Comma-separated subtitles/ASR chunks (optional)")

    parser.add_argument("--log-level", type=str, default="INFO", help="DEBUG/INFO/WARN/ERROR")

    args = parser.parse_args(argv)

    try:
        import logging as _logging
        _logging.getLogger().setLevel(getattr(_logging, args.log_level.upper(), _logging.INFO))
    except Exception:
        logger.warning("Не удалось установить log-level: %s", args.log_level)

    subtitles: Optional[List[str]] = None
    if args.subtitles:
        subtitles = [s.strip() for s in str(args.subtitles).split(",") if s.strip()]

    try:
        module = StoryStructureBaselineModule(rs_path=args.rs_path)
        config: Dict[str, Any] = {
            "subtitles": subtitles,
            "clip_model": args.clip_model,
            "sentence_model": args.sentence_model,
        }
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
