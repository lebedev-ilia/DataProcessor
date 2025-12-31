"""
CLI для `similarity_metrics` (BaseModule, NPZ output).
Baseline-версия: intra-video coherence/similarity + optional reference set (если задан).
"""

import argparse
import os
import sys
from typing import Optional, List, Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from modules.similarity_metrics.similarity_metrics import SimilarityBaselineModule
from utils.logger import get_logger

MODULE_NAME = "similarity_metrics"
logger = get_logger(MODULE_NAME)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog=f"run_{MODULE_NAME}",
        description="Similarity metrics (baseline: intra-video coherence; optional reference similarity) — CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("--frames-dir", required=True, help="Директория с кадрами (с metadata.json)")
    parser.add_argument("--rs-path", required=True, help="Папка result_store для артефактов")
    parser.add_argument("--top-n", type=int, default=10, help="Top-N для reference similarity (если reference задан)")
    parser.add_argument("--reference-embeddings-npz", type=str, default=None, help="NPZ с reference embeddings (optional)")
    parser.add_argument("--log-level", type=str, default="INFO", help="DEBUG/INFO/WARN/ERROR")

    args = parser.parse_args(argv)

    try:
        import logging as _logging
        _logging.getLogger().setLevel(getattr(_logging, args.log_level.upper(), _logging.INFO))
    except Exception:
        logger.warning("Не удалось установить log-level: %s", args.log_level)

    try:
        module = SimilarityBaselineModule(
            rs_path=args.rs_path,
            top_n=int(args.top_n),
            reference_embeddings_npz=args.reference_embeddings_npz,
        )
        config: Dict[str, Any] = {
            "top_n": int(args.top_n),
            "reference_embeddings_npz": args.reference_embeddings_npz,
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

