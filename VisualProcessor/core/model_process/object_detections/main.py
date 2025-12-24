#!/VisualProcessor/core/model_process/.model_process_venv python3
"""
Production-ready object detection extractor.

Supports:
- YOLO (ultralytics)
- OWL (OWL-ViT / OWLv2 via internal ObjectDetectionModule)

Output:
- Single compressed NPZ artifact with fixed-shape numpy arrays
- No Python dicts/lists in the core data representation
"""

import argparse
import os
import sys
from datetime import datetime
from typing import List, Dict, Tuple

import numpy as np  # type: ignore


_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if _path not in sys.path:
    sys.path.append(_path)

from utils.frame_manager import FrameManager
from utils.logger import get_logger
from utils.utilites import load_metadata


NAME = "core_object_detections"
VERSION = "2.0"
LOGGER = get_logger(NAME)


MAX_DETECTIONS = 100   # hard cap per frame (production safeguard)
BBOX_DIMS = 4          # x1, y1, x2, y2


def run_yolo(
    frame_manager: FrameManager,
    frame_indices: List[int],
    model_path: str,
    box_threshold: float,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, str]]:
    """
    Run YOLO inference and return fixed-shape numpy tensors.

    Returns:
        boxes       (N, max_det, 4)
        scores      (N, max_det)
        class_ids   (N, max_det)
        valid_mask  (N, max_det)
        class_names (dict: class_id -> class_name)
    """
    from ultralytics import YOLO    # type: ignore

    LOGGER.info(f"{NAME} | YOLO | loading model: {model_path}")
    model = YOLO(model_path)

    n = len(frame_indices)

    boxes = np.zeros((n, MAX_DETECTIONS, BBOX_DIMS), dtype=np.float32)
    scores = np.zeros((n, MAX_DETECTIONS), dtype=np.float32)
    class_ids = np.zeros((n, MAX_DETECTIONS), dtype=np.int32)
    valid_mask = np.zeros((n, MAX_DETECTIONS), dtype=bool)

    class_names: Dict[int, str] = {}

    for start in range(0, n, batch_size):
        batch_idx = frame_indices[start : start + batch_size]
        batch_frames = [frame_manager.get(i) for i in batch_idx]

        try:
            results = model(batch_frames, verbose=False)
        except Exception as e:
            LOGGER.warning(f"{NAME} | YOLO | batch failed {batch_idx}: {e}")
            continue

        for i_local, res in enumerate(results):
            out_i = start + i_local
            if res.boxes is None:
                continue

            for j in range(min(len(res.boxes), MAX_DETECTIONS)):
                conf = float(res.boxes.conf[j].item())
                if conf < box_threshold:
                    continue

                xyxy = res.boxes.xyxy[j].cpu().numpy().astype(np.float32)
                cls_id = int(res.boxes.cls[j].item())

                boxes[out_i, j] = xyxy
                scores[out_i, j] = conf
                class_ids[out_i, j] = cls_id
                valid_mask[out_i, j] = True

                if cls_id not in class_names:
                    class_names[cls_id] = res.names.get(cls_id, f"class_{cls_id}")

        LOGGER.info(
            f"{NAME} | YOLO | processed {min(start + batch_size, n)}/{n}"
        )

    return boxes, scores, class_ids, valid_mask, class_names


def run_owl(
    frame_manager: FrameManager,
    frame_indices: List[int],
    model: str,
    model_family: str,
    device: str,
    default_categories: List[str],
    box_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, str]]:
    """
    Run OWL-based detector and normalize output to fixed-shape tensors.

    We assume ObjectDetectionModule returns per-frame detections compatible
    with the original implementation.
    """
    from modules.object_detection.object_detection_owl import ObjectDetectionModule

    detector = ObjectDetectionModule(
        model_name=model,
        model_family=model_family,
        device=device,
        default_categories=default_categories,
        box_threshold=box_threshold,
    )

    raw = detector.run(frame_manager=frame_manager, frame_indices=frame_indices)

    n = len(frame_indices)

    boxes = np.zeros((n, MAX_DETECTIONS, BBOX_DIMS), dtype=np.float32)
    scores = np.zeros((n, MAX_DETECTIONS), dtype=np.float32)
    class_ids = np.zeros((n, MAX_DETECTIONS), dtype=np.int32)
    valid_mask = np.zeros((n, MAX_DETECTIONS), dtype=bool)

    class_names: Dict[int, str] = {}

    # raw expected: {frame_idx: [ {bbox, class_id, class, score}, ... ]}
    for i, fi in enumerate(frame_indices):
        detections = raw.get(fi, [])
        for j, det in enumerate(detections[:MAX_DETECTIONS]):
            boxes[i, j] = np.array(det["bbox"], dtype=np.float32)
            scores[i, j] = float(det.get("confidence", det.get("score", 0.0)))
            cid = int(det["class_id"])
            class_ids[i, j] = cid
            valid_mask[i, j] = True
            if cid not in class_names:
                class_names[cid] = det.get("class", f"class_{cid}")

    return boxes, scores, class_ids, valid_mask, class_names


def main():
    parser = argparse.ArgumentParser(description="Production object detection extractor (YOLO / OWL)")
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--rs-path", required=True)
    parser.add_argument("--model", default="yolo11x.pt")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--box-threshold", type=float, default=0.6)
    parser.add_argument("--use-queries", action="store_true")
    parser.add_argument("--default-categories", type=str, default=None)
    parser.add_argument("--model-family", type=str, default="owlv2")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    meta = load_metadata(os.path.join(args.frames_dir, "metadata.json"), NAME)
    total_frames = int(meta["total_frames"])

    frame_indices = list(range(0, total_frames, 10))
    LOGGER.warning(
        f"{NAME} | sampled frames: {len(frame_indices)} / {total_frames} "
        "(Это фиксированая выборка для провайдера object_detections, но использовать ее могут разные модули с разной "
        "логикой извлечения фичей. В будующем нужно грамотно коректировать выборку для получения хорошего качества фичей на всех модулях)"
    )

    frame_manager = FrameManager(
        frames_dir=args.frames_dir,
        chunk_size=meta.get("chunk_size", 32),
        cache_size=meta.get("cache_size", 2),
    )

    if args.use_queries:
        default_categories = (
            args.default_categories.split(",")
            if args.default_categories
            else None
        )
        boxes, scores, class_ids, valid_mask, class_names = run_owl(
            frame_manager=frame_manager,
            frame_indices=frame_indices,
            model=args.model,
            model_family=args.model_family,
            device=args.device,
            default_categories=default_categories,
            box_threshold=args.box_threshold,
        )
        impl = "owl"
    else:
        boxes, scores, class_ids, valid_mask, class_names = run_yolo(
            frame_manager=frame_manager,
            frame_indices=frame_indices,
            model_path=args.model,
            box_threshold=args.box_threshold,
            batch_size=args.batch_size,
        )
        impl = "yolo"

    frame_manager.close()

    out_dir = os.path.join(args.rs_path, NAME)
    os.makedirs(out_dir, exist_ok=True)

    out_path = os.path.join(args.rs_path, "detections.npz")

    np.savez_compressed(
        out_path,
        version=VERSION,
        created_at=datetime.utcnow().isoformat(),
        impl=impl,
        model=args.model,
        model_family=args.model_family if impl == "owl" else "yolo",
        box_threshold=args.box_threshold,
        total_frames=total_frames,
        frame_indices=np.array(frame_indices, dtype=np.int32),
        boxes=boxes,
        scores=scores,
        class_ids=class_ids,
        valid_mask=valid_mask,
        class_names=np.array(
            [f"{k}:{v}" for k, v in sorted(class_names.items())],
            dtype="U",
        ),
    )

    LOGGER.info(f"{NAME} | saved NPZ artifact: {out_path}")


if __name__ == "__main__":
    main()
