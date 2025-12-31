"""
CLI интерфейс для модуля micro_emotion (OpenFace / MicroEmotionProcessor).

Приведён к единому формату, аналогичному `modules/action_recognition/main.py`:
- отдельная функция `main(argv)` для удобного вызова
- единая схема аргументов (`--frames-dir`, `--rs-path`, и т.п.)
- аккуратное логирование и гарантированное закрытие `FrameManager`.
"""

from __future__ import annotations

import os
import sys
import argparse
import json
from typing import Optional, List, Dict, Any

import numpy as np
import cv2


_PATH = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
if _PATH not in sys.path:
    sys.path.append(_PATH)

from utils.frame_manager import FrameManager
from utils.results_store import ResultsStore
from utils.logger import get_logger


MODULE_NAME = "micro_emotion"
logger = get_logger(MODULE_NAME)


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _get_openface_analyzer() -> Any:
    """
    Возвращает класс OpenFaceAnalyzer.
    Предпочитает отдельный файл `openface_analyzer.py`, при его отсутствии
    делает fallback на определение внутри этого же модуля (старый формат).
    """
    module_path = os.path.dirname(__file__)
    openface_file = os.path.join(module_path, "openface_analyzer.py")

    if os.path.exists(openface_file):
        from openface_analyzer import OpenFaceAnalyzer  # type: ignore

        return OpenFaceAnalyzer

    # Fallback: загружаем класс из текущего файла через importlib
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "openface_module", os.path.join(module_path, "main.py")
    )
    if spec is None or spec.loader is None:
        raise ImportError("Не удалось загрузить OpenFaceAnalyzer ни из openface_analyzer.py, ни из main.py")

    openface_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(openface_module)  # type: ignore[arg-type]
    if not hasattr(openface_module, "OpenFaceAnalyzer"):
        raise ImportError("В fallback‑модуле не найден класс OpenFaceAnalyzer")
    return openface_module.OpenFaceAnalyzer  # type: ignore[attr-defined]


def run_pipeline(
    frames_dir: str,
    rs_path: str,
    features: str = "all",
    batch_size: int = 50,
    use_face_detection: bool = False,
    docker_image: str = "openface/openface:latest",
) -> Dict[str, Any]:
    """
    Основная логика обработки micro_emotion.

    Возвращает словарь с результатами (будет сохранён через ResultsStore).
    """
    import warnings

    warnings.filterwarnings("ignore", category=UserWarning)

    if not rs_path:
        raise ValueError(f"{MODULE_NAME} | rs_path не указан")

    rs = ResultsStore(rs_path)

    meta_path = os.path.join(frames_dir, "metadata.json")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"{MODULE_NAME} | metadata.json не найден в {frames_dir}")

    metadata = _load_json(meta_path)

    frame_manager = FrameManager(
        frames_dir,
        chunk_size=int(metadata.get("chunk_size", 32)),
        cache_size=int(metadata.get("cache_size", 2)),
    )

    logger.info(f"VisualProcessor | {MODULE_NAME} | main | Initializing OpenFaceAnalyzer")

    OpenFaceAnalyzer = _get_openface_analyzer()
    analyzer = OpenFaceAnalyzer(docker_image=docker_image)

    # 1. Получаем список индексов кадров (опционально фильтруем по результатам face_detection)
    frame_indices: List[int] = list(range(int(metadata.get("total_frames", 0))))

    if use_face_detection and rs_path:
        try:
            face_results_path = os.path.join(rs_path, "face_detection")
            if os.path.exists(face_results_path):
                face_files = [f for f in os.listdir(face_results_path) if f.endswith(".json")]
                if face_files:
                    face_files.sort()
                    face_data = _load_json(os.path.join(face_results_path, face_files[-1]))
                    if "frames" in face_data:
                        frames_with_faces = [
                            int(k)
                            for k, v in face_data["frames"].items()
                            if v and len(v) > 0
                        ]
                        frame_indices = sorted(set(frame_indices) & set(frames_with_faces))
                        logger.info(
                            f"VisualProcessor | {MODULE_NAME} | main | "
                            f"Filtered to {len(frame_indices)} frames with faces"
                        )
        except Exception as e:  # noqa: BLE001
            logger.warning(
                "VisualProcessor | %s | main | Could not load face detection data: %s",
                MODULE_NAME,
                e,
            )

    logger.info(
        "VisualProcessor | %s | main | Processing %d frames in batches of %d",
        MODULE_NAME,
        len(frame_indices),
        batch_size,
    )

    all_results: List[Dict[str, Any]] = []

    try:
        for batch_start in range(0, len(frame_indices), batch_size):
            batch_end = min(batch_start + batch_size, len(frame_indices))
            batch_indices = frame_indices[batch_start:batch_end]

            logger.info(
                "VisualProcessor | %s | main | Processing batch %d/%d",
                MODULE_NAME,
                (batch_start // batch_size) + 1,
                (len(frame_indices) + batch_size - 1) // batch_size,
            )

            # Получаем кадры
            frames = []
            for idx in batch_indices:
                try:
                    frame = frame_manager.get(idx)
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                    frames.append(frame_bgr)
                except Exception as e:  # noqa: BLE001
                    logger.warning(
                        "VisualProcessor | %s | main | Error loading frame %d: %s",
                        MODULE_NAME,
                        idx,
                        e,
                    )
                    continue

            if not frames:
                continue

            # Анализируем кадры через OpenFaceAnalyzer
            try:
                batch_results = analyzer.analyze_frames(
                    frames=frames,
                    frame_indices=batch_indices[: len(frames)],
                    output_prefix=f"batch_{batch_start}",
                )

                if batch_results:
                    all_results.extend(batch_results)
                    logger.info(
                        "VisualProcessor | %s | main | Processed %d frames in batch",
                        MODULE_NAME,
                        len(batch_results),
                    )
            except Exception as e:  # noqa: BLE001
                logger.error(
                    "VisualProcessor | %s | main | Error processing batch: %s",
                    MODULE_NAME,
                    e,
                )
                continue
    finally:
        try:
            frame_manager.close()
        except Exception as e:  # noqa: BLE001
            logger.exception(
                "VisualProcessor | %s | main | Ошибка при закрытии FrameManager: %s",
                MODULE_NAME,
                e,
            )

    logger.info(
        "VisualProcessor | %s | main | Processed %d frames total",
        MODULE_NAME,
        len(all_results),
    )

    # 2. Пытаемся использовать оптимизированный MicroEmotionProcessor (по DataFrame/OpenFace CSV)
    result: Dict[str, Any] | None = None
    try:
        from micro_emotion_processor import MicroEmotionProcessor  # type: ignore
        import pandas as pd  # type: ignore

        df = None
        csv_paths: List[str] = []

        for res in all_results:
            if not isinstance(res, dict):
                continue
            if res.get("csv_path"):
                csv_paths.append(str(res["csv_path"]))
            elif res.get("dataframe") is not None:
                if df is None:
                    df = res["dataframe"]
                else:
                    df = pd.concat([df, res["dataframe"]], ignore_index=True)

        if df is None and csv_paths:
            csv_path = csv_paths[-1]
            if os.path.exists(csv_path):
                df = pd.read_csv(csv_path)
                logger.info(
                    "VisualProcessor | %s | main | Loaded DataFrame from %s, shape=%s",
                    MODULE_NAME,
                    csv_path,
                    df.shape,
                )

        if df is not None and len(df) > 0:
            logger.info(
                "VisualProcessor | %s | main | Using optimized MicroEmotionProcessor",
                MODULE_NAME,
            )
            processor = MicroEmotionProcessor(fps=int(metadata.get("fps", 30)))
            processed = processor.process_openface_dataframe(df, fit_models=True)

            frames_with_face = int(df["success"].sum()) if "success" in df.columns else len(df)

            result = {
                "success": bool(processed.get("success", False)) and frames_with_face > 0,
                "face_count": frames_with_face,
                "success_rate": float(df["success"].mean()) if "success" in df.columns else 1.0,
                "features": processed.get("features", {}),
                "per_frame_vectors": processed.get("per_frame_vectors", np.zeros((0, 1))).tolist(),
                "reliability_flags": processed.get("reliability_flags", {}),
                "microexpr_features": processed.get("microexpr_features", {}),
                "summary": {
                    "total_frames": len(frame_indices),
                    "frames_processed": len(df),
                    "frames_with_face": frames_with_face,
                    "au_count": len(
                        [k for k in processed.get("features", {}).keys() if str(k).startswith("AU")]
                    ),
                    "landmarks_2d_count": 68,
                    "landmarks_3d_count": 68,
                },
                "metadata": {
                    "features_extracted": features,
                    "batch_size": batch_size,
                    "docker_image": docker_image,
                    "processing_mode": "optimized",
                },
            }
    except Exception as e:  # noqa: BLE001
        logger.warning(
            "VisualProcessor | %s | main | Could not use optimized processor: %s, falling back to original",
            MODULE_NAME,
            e,
        )
        result = None

    # 3. Fallback: агрегируем исходные результаты, если оптимизированный путь не сработал
    if result is None:
        if not all_results:
            logger.warning(
                "VisualProcessor | %s | main | No results extracted from OpenFaceAnalyzer",
                MODULE_NAME,
            )
            result = {
                "success": False,
                "face_count": 0,
                "action_units": {},
                "pose": {},
                "gaze": {},
                "facial_landmarks_2d": [],
                "facial_landmarks_3d": [],
                "summary": {
                    "total_frames": len(frame_indices),
                    "frames_with_face": 0,
                    "au_count": 0,
                    "landmarks_2d_count": 0,
                    "landmarks_3d_count": 0,
                },
                "metadata": {
                    "features_extracted": features,
                    "batch_size": batch_size,
                    "docker_image": docker_image,
                    "processing_mode": "fallback",
                },
            }
        else:
            action_units: Dict[str, Dict[str, List[float]]] = {}
            pose_data: Dict[str, Dict[str, List[float]]] = {}
            gaze_data: Dict[str, Dict[str, List[float]]] = {}
            landmarks_2d_all: List[Any] = []
            landmarks_3d_all: List[Any] = []

            frames_with_face = 0

            for res in all_results:
                if not isinstance(res, dict) or not res.get("success", False):
                    continue
                frames_with_face += 1

                if "action_units" in res:
                    for au_name, au_data in res["action_units"].items():
                        if au_name not in action_units:
                            action_units[au_name] = {
                                "intensity_mean": [],
                                "intensity_std": [],
                                "presence_mean": [],
                                "presence_std": [],
                            }
                        action_units[au_name]["intensity_mean"].append(
                            float(au_data.get("intensity_mean", 0.0))
                        )
                        action_units[au_name]["intensity_std"].append(
                            float(au_data.get("intensity_std", 0.0))
                        )
                        action_units[au_name]["presence_mean"].append(
                            float(au_data.get("presence_mean", 0.0))
                        )
                        action_units[au_name]["presence_std"].append(
                            float(au_data.get("presence_std", 0.0))
                        )

                if "pose" in res:
                    for pose_key, pose_val in res["pose"].items():
                        if pose_key not in pose_data:
                            pose_data[pose_key] = {"mean": [], "std": [], "min": [], "max": []}
                        pose_data[pose_key]["mean"].append(float(pose_val.get("mean", 0.0)))
                        pose_data[pose_key]["std"].append(float(pose_val.get("std", 0.0)))
                        pose_data[pose_key]["min"].append(float(pose_val.get("min", 0.0)))
                        pose_data[pose_key]["max"].append(float(pose_val.get("max", 0.0)))

                if "gaze" in res:
                    for gaze_key, gaze_val in res["gaze"].items():
                        if gaze_key not in gaze_data:
                            gaze_data[gaze_key] = {"mean": [], "std": []}
                        gaze_data[gaze_key]["mean"].append(float(gaze_val.get("mean", 0.0)))
                        gaze_data[gaze_key]["std"].append(float(gaze_val.get("std", 0.0)))

                # TODO: при необходимости можно вернуть и сами landmarks (сейчас не используются)
                if "facial_landmarks_2d" in res:
                    landmarks_2d_all.extend(res["facial_landmarks_2d"])
                if "facial_landmarks_3d" in res:
                    landmarks_3d_all.extend(res["facial_landmarks_3d"])

            # Усредняем агрегированные значения
            agg_action_units: Dict[str, Dict[str, float]] = {}
            for au_name, vals in action_units.items():
                agg_action_units[au_name] = {
                    "intensity_mean": float(np.mean(vals["intensity_mean"])) if vals["intensity_mean"] else 0.0,
                    "intensity_std": float(np.mean(vals["intensity_std"])) if vals["intensity_std"] else 0.0,
                    "presence_mean": float(np.mean(vals["presence_mean"])) if vals["presence_mean"] else 0.0,
                    "presence_std": float(np.mean(vals["presence_std"])) if vals["presence_std"] else 0.0,
                }

            agg_pose: Dict[str, Dict[str, float]] = {}
            for pose_key, vals in pose_data.items():
                agg_pose[pose_key] = {
                    "mean": float(np.mean(vals["mean"])) if vals["mean"] else 0.0,
                    "std": float(np.mean(vals["std"])) if vals["std"] else 0.0,
                    "min": float(np.min(vals["min"])) if vals["min"] else 0.0,
                    "max": float(np.max(vals["max"])) if vals["max"] else 0.0,
                }

            agg_gaze: Dict[str, Dict[str, float]] = {}
            for gaze_key, vals in gaze_data.items():
                agg_gaze[gaze_key] = {
                    "mean": float(np.mean(vals["mean"])) if vals["mean"] else 0.0,
                    "std": float(np.mean(vals["std"])) if vals["std"] else 0.0,
                }

            result = {
                "success": frames_with_face > 0,
                "face_count": frames_with_face,
                "success_rate": float(frames_with_face / len(all_results)) if all_results else 0.0,
                "action_units": agg_action_units,
                "pose": agg_pose,
                "gaze": agg_gaze,
                "facial_landmarks_2d": landmarks_2d_all[:68] if landmarks_2d_all else [],
                "facial_landmarks_3d": landmarks_3d_all[:68] if landmarks_3d_all else [],
                "summary": {
                    "total_frames": len(frame_indices),
                    "frames_processed": len(all_results),
                    "frames_with_face": frames_with_face,
                    "au_count": len(agg_action_units),
                    "landmarks_2d_count": len(landmarks_2d_all),
                    "landmarks_3d_count": len(landmarks_3d_all),
                },
                "metadata": {
                    "features_extracted": features,
                    "batch_size": batch_size,
                    "docker_image": docker_image,
                    "processing_mode": "aggregated",
                },
            }

    # Сохраняем результат через ResultsStore
    rs.store(result, name=MODULE_NAME)
    logger.info("VisualProcessor | %s | main | Results stored successfully", MODULE_NAME)

    return result


def main(argv: Optional[List[str]] = None) -> int:
    """CLI‑вход для модуля micro_emotion (аналогично action_recognition.main)."""

    parser = argparse.ArgumentParser(
        prog=f"run_{MODULE_NAME}",
        description="Micro Emotion Module - Extracts micro-expressions and Action Units using OpenFace",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--frames-dir",
        required=True,
        help="Директория с кадрами (должна содержать metadata.json)",
    )
    parser.add_argument(
        "--rs-path",
        required=True,
        help="Путь к директории ResultsStore (будут сохранены результаты micro_emotion)",
    )
    parser.add_argument(
        "--features",
        type=str,
        default="all",
        choices=["all", "basic", "au", "pose", "gaze"],
        help="Какие группы фич извлекать (информационное поле в metadata)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Размер батча для обработки кадров OpenFace",
    )
    parser.add_argument(
        "--use-face-detection",
        action="store_true",
        help="Фильтровать кадры по результатам face_detection (если есть в rs_path)",
    )
    parser.add_argument(
        "--docker-image",
        type=str,
        default="openface/openface:latest",
        help="Docker‑образ для OpenFace",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        help="Уровень логирования (DEBUG/INFO/WARN/ERROR)",
    )

    args = parser.parse_args(argv)

    # Настройка уровня логирования
    try:
        import logging as _logging

        _logging.getLogger().setLevel(
            getattr(_logging, args.log_level.upper(), _logging.INFO)
        )
    except Exception:  # noqa: BLE001
        logger.warning("Не удалось установить log-level: %s", args.log_level)

    try:
        run_pipeline(
            frames_dir=args.frames_dir,
            rs_path=args.rs_path,
            features=args.features,
            batch_size=args.batch_size,
            use_face_detection=args.use_face_detection,
            docker_image=args.docker_image,
        )
        return 0
    except FileNotFoundError as e:
        logger.error("Файл не найден: %s", e)
        return 2
    except ValueError as e:
        logger.error("Некорректные данные: %s", e)
        return 3
    except Exception as e:  # noqa: BLE001
        logger.exception("Fatal error в %s: %s", MODULE_NAME, e)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())