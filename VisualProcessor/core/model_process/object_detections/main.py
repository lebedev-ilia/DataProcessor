#!/usr/bin/env python3
# core_object_detections.py
"""
Production-ready object detection + tracking extractor.

Поддерживает:
- YOLO (ultralytics)
- OWL (OWL-ViT via module)

Выход:
- detections.npz с фиксированными numpy-массивами:
    - boxes       (N_frames, MAX_DETECTIONS, 4)  float32 (x1,y1,x2,y2)
    - scores      (N_frames, MAX_DETECTIONS)     float32
    - class_ids   (N_frames, MAX_DETECTIONS)     int32
    - valid_mask  (N_frames, MAX_DETECTIONS)     bool
    - class_names (M,)                            unicode array "id:name"
    - tracks      (N_frames, MAX_DETECTIONS)     int32 (track_id или -1)
    - tracks_list (K,)                            object array: each entry -> list of frame indices for that track
    - metadata fields...
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import tempfile
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple, Union
from collections import defaultdict

import cv2
from PIL import Image
import numpy as np
import torch

from transformers import (
    OwlViTProcessor,
    OwlViTForObjectDetection,
    Owlv2Processor,
    Owlv2ForObjectDetection,
)

_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if _path not in sys.path:
    sys.path.append(_path)

from utils.frame_manager import FrameManager
from utils.logger import get_logger
from utils.utilites import load_metadata

NAME = "core_object_detections"
VERSION = "2.1"
LOGGER = get_logger(NAME)

MAX_DETECTIONS = 100
BBOX_DIMS = 4


class OWLModule:
    """
    Object detection module using OWL-ViT / OWLv2.

    Основные улучшения:
      - стабильная работа с PIL (processor требует PIL)
      - корректное вычисление target_sizes из numpy
      - безопасный постпроцессинг и обрезка bbox по краям изображения
      - fallback для извлечения цвета, если sklearn не установлен
    """
    def __init__(
        self,
        model_name: str = "google/owlv2-large-patch14",
        model_family: str = "owlv2",  # "owlvit" or "owlv2"
        device: Optional[str] = None,
        default_categories: Optional[List[str]] = None,
        box_threshold: float = 0.1,
        enable_tracking: bool = True,
        enable_brand_detection: bool = True,
        enable_semantic_tags: bool = True,
        enable_attributes: bool = True,
    ):
        self.model_name = model_name
        self.model_family = model_family.lower()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.box_threshold = float(box_threshold)

        # Default COCO-like categories (kept from original)
        self.default_categories = default_categories or [
            "person", "car", "truck", "bicycle", "motorcycle", "bus", "train", "airplane",
            "boat", "traffic light", "stop sign", "bench", "bird", "cat", "dog", "horse",
            "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
            "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball",
            "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
            "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
            "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
            "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
            "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
            "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush",
        ]

        # Flags
        self.enable_tracking = enable_tracking
        self.enable_brand_detection = enable_brand_detection
        self.enable_semantic_tags = enable_semantic_tags
        self.enable_attributes = enable_attributes

        # Lazy-loaded objects
        self._model = None
        self._processor = None
        self._model_loaded = False

        LOGGER.info(
            "ObjectDetectionModule initialized: model=%s family=%s device=%s threshold=%s",
            self.model_name, self.model_family, self.device, self.box_threshold
        )

    def _load_model(self) -> None:
        """Lazy load the processor and the model. Safe to call multiple times."""
        if self._model_loaded:
            return

        LOGGER.info("Loading model %s (family=%s) on device=%s", self.model_name, self.model_family, self.device)

        # Automatic model name mapping for common older name
        if self.model_family == "owlv2":
            # map old accidental name to owlv2 if necessary
            if self.model_name == "google/owlvit-base-patch16":
                self.model_name = "google/owlv2-base-patch16"
            self._processor = Owlv2Processor.from_pretrained(self.model_name)
            self._model = Owlv2ForObjectDetection.from_pretrained(self.model_name).to(self.device)
        else:
            self._processor = OwlViTProcessor.from_pretrained(self.model_name)
            self._model = OwlViTForObjectDetection.from_pretrained(self.model_name).to(self.device)

        self._model.eval()
        self._model_loaded = True
        LOGGER.info("Model loaded successfully")

    def _prepare_text_queries(self, text_queries: Optional[Union[str, List[str]]]) -> List[str]:
        """Normalize text_queries into a list of trimmed strings."""
        if text_queries is None:
            return list(self.default_categories)
        if isinstance(text_queries, str):
            # allow comma-separated string
            items = [q.strip() for q in text_queries.split(",") if q.strip()]
            return items or list(self.default_categories)
        # already a list
        return [str(q).strip() for q in text_queries if str(q).strip()]

    def _clamp_bbox(self, bbox: List[float], width: int, height: int) -> List[float]:
        """Clamp bbox coordinates to image boundaries and ensure valid box."""
        x_min, y_min, x_max, y_max = map(float, bbox)
        x_min = max(0.0, min(x_min, width - 1.0))
        x_max = max(0.0, min(x_max, width - 1.0))
        y_min = max(0.0, min(y_min, height - 1.0))
        y_max = max(0.0, min(y_max, height - 1.0))
        if x_max <= x_min or y_max <= y_min:
            return [0.0, 0.0, 0.0, 0.0]
        return [x_min, y_min, x_max, y_max]

    def _detect_objects_in_frame(
        self,
        frame: np.ndarray,
        text_queries: Optional[List[str]] = None,
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Detect objects in a single frame.

        :param frame: BGR numpy array (OpenCV format)
        :param text_queries: list of text queries (strings)
        :return: tuple of (list of detections {bbox, score, label, ...}, processing_time)
        """
        if frame is None:
            return [], 0.0

        # ensure model is loaded
        self._load_model()

        # Convert to RGB (from BGR OpenCV)
        if frame.ndim == 3:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            # grayscale? convert to 3-channel
            rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        # Keep original width/height from numpy to compute target_sizes
        h, w = rgb.shape[:2]

        # Use PIL Image for processor (this is critical)
        image = Image.fromarray(rgb)

        queries = self._prepare_text_queries(text_queries)
        if not queries:
            queries = list(self.default_categories)

        # Run processor + model on device
        tik = time.time()
        try:
            inputs = self._processor(text=queries, images=image, return_tensors="pt").to(self.device)
        except Exception as e:
            LOGGER.exception("Processor failed for the given image. Falling back to CPU processor call: %s", e)
            # fallback: build inputs on CPU and then move tensors to device
            inputs = self._processor(text=queries, images=image, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        tok = round(time.time() - tik, 2)

        with torch.no_grad():
            outputs = self._model(**inputs)

        # target_sizes must be [height, width]
        target_sizes = torch.tensor([[h, w]], dtype=torch.long).to(self.device)

        # Post-process (returns list with dict per image)
        results = self._processor.post_process_grounded_object_detection(
            outputs=outputs,
            target_sizes=target_sizes,
            threshold=self.box_threshold
        )

        detections: List[Dict[str, Any]] = []

        if not results or len(results) == 0:
            return [], tok

        result = results[0]
        boxes = result.get("boxes", torch.tensor([]))
        scores = result.get("scores", torch.tensor([]))
        labels = result.get("labels", torch.tensor([]))

        # Bring to CPU and numpy for iteration
        if isinstance(boxes, torch.Tensor):
            boxes = boxes.cpu().numpy()
        if isinstance(scores, torch.Tensor):
            scores = scores.cpu().numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()

        # Iterate detections
        for box, score, label_idx in zip(boxes, scores, labels):
            score_val = float(score)
            if score_val < float(self.box_threshold):
                continue

            # label_idx is index into queries (sometimes long)
            try:
                label_i = int(label_idx)
                if 0 <= label_i < len(queries):
                    label_name = queries[label_i]
                else:
                    label_name = f"class_{label_i}"
            except Exception:
                label_name = str(label_idx)

            # clamp bbox
            clamped = self._clamp_bbox(box.tolist(), width=w, height=h)
            if clamped[2] <= clamped[0] or clamped[3] <= clamped[1]:
                # invalid box after clamp
                continue

            x_min, y_min, x_max, y_max = clamped

            det = {
                "bbox": [float(x_min), float(y_min), float(x_max), float(y_max)],
                "score": score_val,
                "label": label_name,
            }
            detections.append(det)

        return detections, tok

    def run(self, frame_manager, frame_indices: List[int]) -> Dict[str, Any]:
        """
        Main execution method.

        :param frame_manager: object with .fps and .get(index) -> numpy frame (BGR)
        :param frame_indices: list of frame indices to process
        :return: result dictionary containing per-frame detections and summary
        """
        import time

        if frame_manager is None:
            raise ValueError("frame_manager is None")

        self._load_model()

        # Prepare text queries: base categories + brands if enabled
        queries = list(self.default_categories)
        if self.enable_brand_detection:
            queries.extend(self.brand_queries)

        all_detections: Dict[int, List[Dict[str, Any]]] = {}
        skipped = 0

        t = time.time()

        for k, frame_idx in enumerate(frame_indices):
            frame = frame_manager.get(frame_idx)

            if frame is None:
                all_detections[frame_idx] = []
                skipped += 1
                continue

            try:
                tik = time.time()
                detections, pred_time = self._detect_objects_in_frame(frame, text_queries=queries)
                det_tok = round(time.time() - tik, 2)

                LOGGER.debug(f"pred_time: {pred_time} | det_tok: {det_tok}")

                all_detections[frame_idx] = detections

                # Periodic log
                if k % 20 == 0:
                    processed_nonempty = sum(1 for v in all_detections.values() if v)
                    c_t = time.time()
                    LOGGER.info(
                        "run | processed_frames=%d | current_index=%d | skipped=%d | nonempty_frames=%d | all_time:%d",
                        k, frame_idx, skipped, processed_nonempty, round(c_t - t, 2)
                    )
                    t = c_t

            except Exception as e:
                LOGGER.exception("Error detecting objects in frame %s: %s", frame_idx, e)
                all_detections[frame_idx] = []
                skipped += 1

        # Build summary
        object_counts: Dict[str, int] = {}
        total_detections = 0

        for frame_idx, detections in all_detections.items():
            for det in detections:
                label = det.get("label", "unknown")
                object_counts[label] = object_counts.get(label, 0) + 1
                total_detections += 1

        result = {
            "frames": all_detections,
            "summary": {
                "total_detections": total_detections,
                "unique_categories": len(object_counts),
                "category_counts": object_counts,
            },
            "frame_count": len(frame_indices),
        }

        return result


def iou_xyxy(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    IOU между bbox a и b в формате [x1,y1,x2,y2]
    a: (4,), b: (N,4)  -> returns (N,)
    """
    ax1, ay1, ax2, ay2 = a
    bx1 = b[:, 0]
    by1 = b[:, 1]
    bx2 = b[:, 2]
    by2 = b[:, 3]

    inter_x1 = np.maximum(ax1, bx1)
    inter_y1 = np.maximum(ay1, by1)
    inter_x2 = np.minimum(ax2, bx2)
    inter_y2 = np.minimum(ay2, by2)

    inter_w = np.maximum(0.0, inter_x2 - inter_x1)
    inter_h = np.maximum(0.0, inter_y2 - inter_y1)
    inter_area = inter_w * inter_h

    area_a = max(0.0, (ax2 - ax1)) * max(0.0, (ay2 - ay1))
    area_b = np.maximum(0.0, (bx2 - bx1)) * np.maximum(0.0, (by2 - by1))

    union = area_a + area_b - inter_area + 1e-12
    return inter_area / union


def run_yolo(
    frame_manager: FrameManager,
    frame_indices: List[int],
    model_path: str,
    box_threshold: float,
    batch_size: int,
    device: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, str], List[np.ndarray]]:
    """
    Запускает ультраликтикс YOLO на батчах и возвращает фиксированные тензоры + raw detections per frame.

    Возвращает:
        boxes (N, MAX_DETECTIONS, 4),
        scores (N, MAX_DETECTIONS),
        class_ids (N, MAX_DETECTIONS),
        valid_mask (N, MAX_DETECTIONS),
        class_names dict,
        raw_per_frame: list length N, each element - ndarray (M,5) [x1,y1,x2,y2,score]
    """
    try:
        from ultralytics import YOLO  # type: ignore
    except Exception as e:
        LOGGER.exception("YOLO import failed: %s", e)
        raise

    LOGGER.info("%s | YOLO | loading model: %s", NAME, model_path)
    model = YOLO(model_path)

    n = len(frame_indices)
    boxes = np.zeros((n, MAX_DETECTIONS, BBOX_DIMS), dtype=np.float32)
    scores = np.zeros((n, MAX_DETECTIONS), dtype=np.float32)
    class_ids = np.zeros((n, MAX_DETECTIONS), dtype=np.int32)
    valid_mask = np.zeros((n, MAX_DETECTIONS), dtype=bool)
    class_names: Dict[int, str] = {}

    raw_per_frame: List[np.ndarray] = [np.zeros((0, 5), dtype=np.float32) for _ in frame_indices]

    for start in range(0, n, batch_size):
        batch_idx = frame_indices[start : start + batch_size]
        batch_frames = [frame_manager.get(i) for i in batch_idx]

        try:
            results = model(batch_frames, device=device, verbose=False)
        except Exception as e:
            LOGGER.warning("%s | YOLO | batch failed %s: %s", NAME, batch_idx, e)
            continue

        for i_local, res in enumerate(results):
            out_i = start + i_local
            # res.boxes may be empty
            try:
                if res.boxes is None or len(res.boxes) == 0:
                    continue
            except Exception:
                # ultralytics API sometimes differs by versions
                continue

            detections = []
            # iterate boxes
            for j in range(min(len(res.boxes), MAX_DETECTIONS)):
                try:
                    conf = float(res.boxes.conf[j].item())
                    if conf < box_threshold:
                        continue
                    xyxy = res.boxes.xyxy[j].cpu().numpy().astype(np.float32)
                    cls_id = int(res.boxes.cls[j].item())
                except Exception:
                    # fallback if API shape differs
                    try:
                        box = res.boxes.data[j].cpu().numpy()
                        xyxy = box[:4].astype(np.float32)
                        conf = float(box[4])
                        cls_id = int(box[5]) if box.shape[0] > 5 else 0
                        if conf < box_threshold:
                            continue
                    except Exception:
                        continue

                boxes[out_i, j] = xyxy
                scores[out_i, j] = conf
                class_ids[out_i, j] = cls_id
                valid_mask[out_i, j] = True
                detections.append([xyxy[0], xyxy[1], xyxy[2], xyxy[3], conf])

                if cls_id not in class_names:
                    try:
                        class_names[cls_id] = res.names.get(cls_id, f"class_{cls_id}")
                    except Exception:
                        class_names[cls_id] = f"class_{cls_id}"

            raw_per_frame[out_i] = np.array(detections, dtype=np.float32)

        LOGGER.info("%s | YOLO | processed %d/%d", NAME, min(start + batch_size, n), n)

    return boxes, scores, class_ids, valid_mask, class_names, raw_per_frame


def run_owl(
    frame_manager: FrameManager,
    frame_indices: List[int],
    model: str,
    model_family: str,
    device: str,
    default_categories: Optional[List[str]],
    box_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, str], List[np.ndarray]]:
    """
    Запускает OWL-based detector. Возвращаем те же структуры, что и run_yolo.
    Требует реализации modules.object_detection.object_detection_owl.ObjectDetectionModule с методом run.
    """

    detector = OWLModule(
        model_name=model,
        model_family=model_family,
        device=device,
        default_categories=default_categories,
        box_threshold=box_threshold,
    )

    raw = detector.run(frame_manager=frame_manager, frame_indices=frame_indices)
    # raw expected: {"frames": {frame_idx: [ {bbox, label, score, ...}, ... ]}, ...}

    n = len(frame_indices)
    boxes = np.zeros((n, MAX_DETECTIONS, BBOX_DIMS), dtype=np.float32)
    scores = np.zeros((n, MAX_DETECTIONS), dtype=np.float32)
    class_ids = np.zeros((n, MAX_DETECTIONS), dtype=np.int32)
    valid_mask = np.zeros((n, MAX_DETECTIONS), dtype=bool)
    class_names: Dict[int, str] = {}
    raw_per_frame: List[np.ndarray] = [np.zeros((0, 5), dtype=np.float32) for _ in frame_indices]

    # Create label to class_id mapping
    label_to_id: Dict[str, int] = {}
    next_id = 0

    frames_dict = raw.get("frames", {})
    for i, fi in enumerate(frame_indices):
        detections = frames_dict.get(fi, []) or []
        dets_list = []
        for j, det in enumerate(detections[:MAX_DETECTIONS]):
            bb = np.array(det["bbox"], dtype=np.float32)
            sc = float(det.get("score", 0.0))
            label = det.get("label", "unknown")
            
            # Map label to class_id
            if label not in label_to_id:
                label_to_id[label] = next_id
                class_names[next_id] = label
                next_id += 1
            cid = label_to_id[label]
            
            boxes[i, j] = bb
            scores[i, j] = sc
            class_ids[i, j] = cid
            valid_mask[i, j] = True
            dets_list.append([bb[0], bb[1], bb[2], bb[3], sc])
        
        raw_per_frame[i] = np.array(dets_list, dtype=np.float32)

    return boxes, scores, class_ids, valid_mask, class_names, raw_per_frame


def run_tracking(
    raw_per_frame: List[np.ndarray],
    frame_indices: List[int],
    frame_manager: FrameManager,
    tracker_cfg: dict,
    iou_threshold: float = 0.3,
) -> Tuple[np.ndarray, Dict[int, List[int]]]:
    """
    Запускает ByteTrack поверх raw_per_frame (list of Nx5 arrays [x1,y1,x2,y2,score])
    Возвращает:
      - tracks_arr: (N_frames, MAX_DETECTIONS) int32  (track_id or -1)
      - tracks_map: dict track_id -> sorted list of frame indices where track appears
    """
    try:
        from yolox.tracker.byte_tracker import BYTETracker
    except Exception as e:
        LOGGER.exception("ByteTrack import failed: %s", e)
        raise

    tracker = BYTETracker(
        track_thresh=tracker_cfg.get("track_thresh", 0.25),
        match_thresh=tracker_cfg.get("match_thresh", 0.8),
        frame_rate=tracker_cfg.get("frame_rate", 30),
    )

    n = len(frame_indices)
    tracks_arr = np.full((n, MAX_DETECTIONS), -1, dtype=np.int32)  # default -1
    tracks_map: Dict[int, List[int]] = {}

    LOGGER.info("%s | TRACKING | starting ByteTrack over %d frames", NAME, n)

    for i, fi in enumerate(frame_indices):
        dets = raw_per_frame[i]  # shape (M,5) or empty
        img = frame_manager.get(fi)
        img_shape = img.shape[:2] if img is not None else (0, 0)

        if dets is None or dets.size == 0:
            # still update tracker with empty input
            try:
                tracker.update(np.empty((0,5)), img_shape, img_shape)
            except Exception:
                # Some BYTETracker implementations require different args; try alternative call
                try:
                    tracker.update(np.empty((0,5)))
                except Exception:
                    LOGGER.debug("%s | TRACKING | tracker.update(empty) failed for frame %d", NAME, fi)
            continue

        # ByteTrack expects dets as ndarray Nx5 in format [x1,y1,x2,y2,score]
        dets_np = dets.astype(np.float32)

        # call tracker
        try:
            online_targets = tracker.update(dets_np, img_shape, img_shape)
        except TypeError:
            # fallback: some implementations use tracker.update(dets, img_shape)
            try:
                online_targets = tracker.update(dets_np, img_shape)
            except Exception as e:
                LOGGER.exception("%s | TRACKING | tracker.update failed at frame %d: %s", NAME, fi, e)
                online_targets = []

        # For each online target, find best-matching detection index by IoU and assign track_id
        # online_target expected attributes: track_id, tlbr or tlwh
        assigned_detection_idxs = set()
        for t in online_targets:
            # get bbox of target in xyxy
            tt = None
            try:
                if hasattr(t, "tlbr"):
                    tt = np.array(t.tlbr, dtype=np.float32)
                elif hasattr(t, "tlwh"):
                    x, y, w, h = t.tlwh
                    tt = np.array([x, y, x + w, y + h], dtype=np.float32)
                elif hasattr(t, "tlbr_tlbr"):  # some variants
                    tt = np.array(t.tlbr_tlbr, dtype=np.float32)
                else:
                    # try attribute 'bbox' or 'xyxy'
                    if hasattr(t, "bbox"):
                        tt = np.array(t.bbox, dtype=np.float32)
            except Exception:
                LOGGER.debug("%s | TRACKING | cannot read bbox from target object; skipping", NAME)
                continue

            if tt is None or tt.size != 4:
                continue

            # compute IoU with this frame detections
            if dets_np.size == 0:
                continue
            ious = iou_xyxy(tt, dets_np[:, :4])
            best_idx = int(np.argmax(ious))
            best_iou = float(ious[best_idx])
            if best_iou < iou_threshold:
                # not matched confidently -> skip (tracker may be new/old)
                continue

            # avoid double-assigning same detection to multiple targets
            if best_idx in assigned_detection_idxs:
                continue

            assigned_detection_idxs.add(best_idx)
            tid = int(getattr(t, "track_id", getattr(t, "track_id_", -1)))
            tracks_arr[i, best_idx] = tid
            tracks_map.setdefault(tid, []).append(frame_indices[i])

        # also ensure unique frame entries in tracks_map (sorted later)
    # Postprocess: deduplicate and sort track frame lists
    for tid, frames in list(tracks_map.items()):
        tracks_map[tid] = sorted(list(dict.fromkeys(frames)))

    LOGGER.info("%s | TRACKING | finished. tracks_count=%d", NAME, len(tracks_map))
    return tracks_arr, tracks_map


def atomic_save_npz(path: str, **kwargs) -> None:
    """
    Атомарно сохраняет np.savez_compressed через временный файл.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(path), dir=os.path.dirname(path))
    os.close(fd)
    try:
        np.savez_compressed(tmp, **kwargs)
        os.replace(tmp, path)
    except Exception:
        try:
            os.remove(tmp)
        except Exception:
            pass
        raise


def main():
    parser = argparse.ArgumentParser(description="Production object detection + tracking provider")
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--rs-path", required=True)
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--box-threshold", type=float, default=0.6)
    parser.add_argument("--use-queries", action="store_true")
    parser.add_argument("--default-categories", type=str, default=None)
    parser.add_argument("--model-family", type=str, default="owlv2")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    # NOTE: frame sampling is owned by Segmenter/DataProcessor.
    # We keep sample-step for backward compatibility, but production uses metadata[NAME].frame_indices.
    parser.add_argument("--sample-step", type=int, default=None)
    args = parser.parse_args()

    meta = load_metadata(os.path.join(args.frames_dir, "metadata.json"), NAME)
    total_frames = int(meta["total_frames"])

    # Strict sampling contract: prefer metadata[NAME].frame_indices, no fallback allowed.
    block = meta.get(NAME)
    if not isinstance(block, dict) or "frame_indices" not in block:
        raise RuntimeError(
            f"{NAME} | metadata missing '{NAME}.frame_indices'. "
            "Segmenter must provide per-provider frame_indices. No fallback is allowed."
        )
    frame_indices_raw = block.get("frame_indices")
    if not isinstance(frame_indices_raw, list) or not frame_indices_raw:
        raise RuntimeError(f"{NAME} | metadata '{NAME}.frame_indices' is empty/invalid.")
    frame_indices = [int(x) for x in frame_indices_raw]
    LOGGER.info("%s | sampled frames: %d / total=%d", NAME, len(frame_indices), total_frames)

    frame_manager = FrameManager(
        frames_dir=args.frames_dir,
        chunk_size=meta.get("chunk_size", 32),
        cache_size=meta.get("cache_size", 2),
    )

    try:
        if args.use_queries:
            default_categories = (
                args.default_categories.split(",") if args.default_categories else None
            )
            boxes, scores, class_ids, valid_mask, class_names, raw_per_frame = run_owl(
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
            boxes, scores, class_ids, valid_mask, class_names, raw_per_frame = run_yolo(
                frame_manager=frame_manager,
                frame_indices=frame_indices,
                model_path=args.model,
                box_threshold=args.box_threshold,
                batch_size=args.batch_size,
                device=args.device,
            )
            impl = "yolo"

        # run tracking (ByteTrack) on raw detections
        try:
            tracker_cfg = {"track_thresh": 0.25, "match_thresh": 0.8, "frame_rate": meta.fps}
            tracks_arr, tracks_map = run_tracking(
                raw_per_frame=raw_per_frame,
                frame_indices=frame_indices,
                frame_manager=frame_manager,
                tracker_cfg=tracker_cfg,
                iou_threshold=args.iou_threshold,
            )
        except Exception:
            LOGGER.exception("%s | TRACKING failed; producing outputs without tracks", NAME)
            tracks_arr = np.full((len(frame_indices), MAX_DETECTIONS), -1, dtype=np.int32)
            tracks_map = {}

        class_names_arr = np.array([f"{k}:{v}" for k, v in sorted(class_names.items())], dtype="U")

        if tracks_map:
            track_ids_sorted = sorted(tracks_map.keys())
            tracks_list = np.empty((len(track_ids_sorted),), dtype=object)
            for i, tid in enumerate(track_ids_sorted):
                tracks_list[i] = np.array(tracks_map[tid], dtype=np.int32)
            tracks_list_ids = np.asarray(track_ids_sorted, dtype=np.int32)
        else:
            tracks_list = np.empty((0,), dtype=object)
            tracks_list_ids = np.asarray([], dtype=np.int32)

        out_dir = os.path.join(args.rs_path, NAME)
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, "detections.npz")

        created_at = datetime.utcnow().isoformat()
        meta_info = {
            "producer": NAME,
            "producer_version": VERSION,
            "created_at": created_at,
            "status": "ok" if len(frame_indices) > 0 else "empty",
            "empty_reason": None if len(frame_indices) > 0 else "no_frames",
            "impl": impl,
            "model": args.model,
            "model_family": args.model_family if impl == "owl" else "yolo",
            "box_threshold": args.box_threshold,
            "total_frames": int(total_frames),
            "sample_step": None,
        }
        for k in ["platform_id", "video_id", "run_id", "sampling_policy_version", "config_hash"]:
            if k in meta:
                meta_info[k] = meta.get(k)

        atomic_save_npz(
            out_path,
            # metadata fields
            meta=np.asarray(meta_info, dtype=object),
            frame_indices=np.asarray(frame_indices, dtype=np.int32),
            # detection arrays
            boxes=boxes,
            scores=scores,
            class_ids=class_ids,
            valid_mask=valid_mask,
            class_names=class_names_arr,
            # tracking
            tracks=tracks_arr,
            tracks_list=tracks_list,
            tracks_list_ids=tracks_list_ids,
        )

        LOGGER.info("%s | saved NPZ artifact: %s", NAME, out_path)

    finally:
        try:
            frame_manager.close()
        except Exception:
            LOGGER.exception("Failed to close FrameManager")


if __name__ == "__main__":
    main()
