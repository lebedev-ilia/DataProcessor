"""
CLI для `scene_classification` (Places365), приведённый к единому формату запуска.
"""

from __future__ import annotations

import os
import sys
import argparse
from typing import Optional, List, Dict, Any

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from modules.scene_classification.scene_classification import Places365SceneClassifier
from utils.logger import get_logger

MODULE_NAME = "scene_classification"
logger = get_logger(MODULE_NAME)

def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        prog=f"run_{MODULE_NAME}",
        description="Scene classification (Places365) — CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--frames-dir",
        required=True,
        help="Директория с кадрами (FrameManager ожидает metadata.json внутри)",
    )
    parser.add_argument(
        "--rs-path",
        required=True,
        help="Папка для результирующего стора (ResultsStore и npz)",
    )

    # Model options
    parser.add_argument("--model-arch", type=str, default="resnet50", help="Архитектура модели Places365/timm")
    parser.add_argument("--use-timm", action="store_true", help="Использовать timm-модели (если установлено)")
    parser.add_argument(
        "--runtime",
        type=str,
        default="inprocess",
        help="Runtime for Places365: inprocess | triton",
    )
    parser.add_argument(
        "--triton-model-spec",
        type=str,
        default="places365_resnet50_224_triton",
        help="dp_models spec name for Triton-backed Places365 (used when --runtime=triton)",
    )
    parser.add_argument("--device", type=str, default=None, help="cuda/cpu (по умолчанию auto)")
    # NOTE: model artifacts are resolved via ModelManager (DP_MODELS_ROOT). No direct paths here (no-fallback).

    # Runtime thresholds
    parser.add_argument("--min-scene-length", type=int, default=30, help="Минимальная длина сцены в кадрах")
    parser.add_argument("--min-scene-seconds", type=float, default=None, help="Минимальная длина сцены в секундах (fps-aware)")

    # Inference options
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size для inference")
    parser.add_argument("--input-size", type=int, default=224, help="Размер входа (224/320 и т.п.)")
    parser.add_argument("--use-tta", action="store_true", help="TTA (несколько аугментаций + усреднение)")
    parser.add_argument("--use-multi-crop", action="store_true", help="Multi-crop inference (5 кропов)")
    parser.add_argument("--temporal-smoothing", action="store_true", help="Темпоральное сглаживание предсказаний")
    parser.add_argument("--smoothing-window", type=int, default=5, help="Окно сглаживания (в кадрах)")

    # Advanced features
    parser.add_argument("--enable-advanced-features", action="store_true", help="Включить advanced features")
    parser.add_argument("--use-clip-for-semantics", action="store_true", help="Включить CLIP-based семантику (или core_clip)")

    # Misc
    parser.add_argument("--log-level", type=str, default="INFO", help="DEBUG/INFO/WARN/ERROR")
    
    args = parser.parse_args(argv)

    # Log-level
    try:
        import logging as _logging
        _logging.getLogger().setLevel(getattr(_logging, args.log_level.upper(), _logging.INFO))
    except Exception:
        logger.warning("Не удалось установить log-level: %s", args.log_level)

    try:
        classifier = Places365SceneClassifier(
            runtime=args.runtime,
            triton_model_spec=args.triton_model_spec,
            model_arch=args.model_arch,
            use_timm=args.use_timm,
            min_scene_length=args.min_scene_length,
            min_scene_seconds=args.min_scene_seconds,
            batch_size=args.batch_size,
            device=args.device,
            input_size=args.input_size,
            use_tta=args.use_tta,
            use_multi_crop=args.use_multi_crop,
            temporal_smoothing=args.temporal_smoothing,
            smoothing_window=args.smoothing_window,
            enable_advanced_features=args.enable_advanced_features,
            use_clip_for_semantics=args.use_clip_for_semantics,
            rs_path=args.rs_path,
        )

        # BaseModule.run will: load metadata, create FrameManager, call process(), save results.
        config: Dict[str, Any] = {
            "min_scene_length": args.min_scene_length,
            "min_scene_seconds": args.min_scene_seconds,
            "enable_advanced_features": args.enable_advanced_features,
            "use_clip_for_semantics": args.use_clip_for_semantics,
        }

        saved_path = classifier.run(frames_dir=args.frames_dir, config=config)
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