#!/usr/bin/env python3
"""
Production-ready object detection + tracking extractor.

Поддерживает:
- YOLO (ultralytics)

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
import numpy as np
import torch

_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if _path not in sys.path:
    sys.path.append(_path)

from utils.frame_manager import FrameManager
from utils.logger import get_logger
from utils.resource_probe import pick_device
from utils.utilites import load_metadata
from utils.meta_builder import apply_models_meta, model_used

NAME = "core_object_detections"
VERSION = "2.1"
SCHEMA_VERSION = "core_object_detections_npz_v1"
LOGGER = get_logger(NAME)

MAX_DETECTIONS = 100
BBOX_DIMS = 4


def _load_triton_spec_via_model_manager(model_spec_name: str) -> dict:
    """
    Resolve Triton model spec via dp_models.ModelManager (no-network, reproducible).
    Returns dict with keys:
      - client: TritonHttpClient
      - rp: runtime_params
      - models_used_entry: dict (model_used)
    """
    from dp_models import get_global_model_manager  # type: ignore

    mm = get_global_model_manager()
    rm = mm.get(model_name=str(model_spec_name))
    rp = rm.spec.runtime_params or {}
    handle = rm.handle or {}
    client = None
    if isinstance(handle, dict):
        client = handle.get("client")
    if client is None:
        raise RuntimeError(f"{NAME} | ModelManager returned empty Triton client handle for: {model_spec_name}")
    if not isinstance(rp, dict) or not rp:
        raise RuntimeError(f"{NAME} | ModelManager returned empty runtime_params for: {model_spec_name}")
    return {"client": client, "rp": rp, "models_used_entry": rm.models_used_entry}


def _letterbox_bgr_no_upscale(
    img_bgr: np.ndarray,
    *,
    new_size: int,
    color: Tuple[int, int, int] = (114, 114, 114),
) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    Ultralytics-style letterbox to square (new_size x new_size), but with NO UPSCALE.

    Returns:
      - img_lb: (new_size, new_size, 3) uint8 BGR
      - r: resize ratio applied (<=1.0)
      - pad: (left, top) padding applied in pixels
    """
    if img_bgr.ndim != 3 or img_bgr.shape[2] != 3:
        raise ValueError(f"{NAME} | invalid image shape: {img_bgr.shape}")
    h0, w0 = int(img_bgr.shape[0]), int(img_bgr.shape[1])
    s = int(new_size)
    if s <= 0:
        raise ValueError(f"{NAME} | invalid new_size={new_size}")

    r = min(float(s) / float(h0), float(s) / float(w0))
    r = min(r, 1.0)  # no upscale

    new_w, new_h = int(round(w0 * r)), int(round(h0 * r))
    if (new_w, new_h) != (w0, h0):
        img = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        img = img_bgr

    dw = s - new_w
    dh = s - new_h
    left = int(round(dw / 2.0 - 0.1))
    right = int(round(dw / 2.0 + 0.1))
    top = int(round(dh / 2.0 - 0.1))
    bottom = int(round(dh / 2.0 + 0.1))

    img_lb = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    if img_lb.shape[0] != s or img_lb.shape[1] != s:
        # Defensive: ensure exact shape
        img_lb = cv2.resize(img_lb, (s, s), interpolation=cv2.INTER_LINEAR)
        left, top = 0, 0
        r = float(s) / float(max(h0, w0))
        r = min(r, 1.0)

    return img_lb, float(r), (int(left), int(top))


def _prep_yolo_tensor_from_rgb_uint8(
    frame_rgb_uint8: np.ndarray,
    *,
    input_size: int,
) -> Tuple[np.ndarray, float, Tuple[int, int], Tuple[int, int]]:
    """
    Preprocess one RGB uint8 frame for Ultralytics-exported YOLO ONNX:
      RGB -> BGR -> letterbox(no-upscale) -> RGB -> FP32 /255 -> NCHW

    Returns:
      - x: (1,3,S,S) float32
      - r: resize ratio
      - pad: (left, top)
      - orig_hw: (h0, w0)
    """
    bgr = cv2.cvtColor(frame_rgb_uint8, cv2.COLOR_RGB2BGR)
    img_lb, r, pad = _letterbox_bgr_no_upscale(bgr, new_size=int(input_size))
    rgb = cv2.cvtColor(img_lb, cv2.COLOR_BGR2RGB)
    x = rgb.astype(np.float32) / 255.0
    x = np.transpose(x, (2, 0, 1))[None, ...]  # (1,3,S,S)
    return x.astype(np.float32), float(r), (int(pad[0]), int(pad[1])), (int(frame_rgb_uint8.shape[0]), int(frame_rgb_uint8.shape[1]))


def _scale_boxes_back(
    boxes_xyxy: np.ndarray,
    *,
    r: float,
    pad: Tuple[int, int],
    orig_hw: Tuple[int, int],
) -> np.ndarray:
    """
    Reverse letterbox: boxes in letterboxed image coords -> original image coords.
    """
    if boxes_xyxy.size == 0:
        return boxes_xyxy.astype(np.float32)
    left, top = int(pad[0]), int(pad[1])
    b = boxes_xyxy.astype(np.float32).copy()
    b[:, [0, 2]] -= float(left)
    b[:, [1, 3]] -= float(top)
    rr = float(max(r, 1e-9))
    b[:, :4] /= rr
    h0, w0 = int(orig_hw[0]), int(orig_hw[1])
    b[:, [0, 2]] = np.clip(b[:, [0, 2]], 0.0, float(w0 - 1))
    b[:, [1, 3]] = np.clip(b[:, [1, 3]], 0.0, float(h0 - 1))
    return b


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
        # FrameManager.get() returns RGB; ultralytics accepts numpy images, but many CV pipelines assume BGR.
        # Convert RGB->BGR for stability with OpenCV-based preprocessing.
        batch_frames = [cv2.cvtColor(frame_manager.get(i), cv2.COLOR_RGB2BGR) for i in batch_idx]

        results = model(batch_frames, device=device, verbose=False)

        for i_local, res in enumerate(results):
            out_i = start + i_local
            # res.boxes may be empty
            if res.boxes is None or len(res.boxes) == 0:
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
                        raise RuntimeError(f"{NAME} | YOLO | cannot parse detection output (ultralytics API drift?)")

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


def run_yolo_triton(
    *,
    frame_manager: FrameManager,
    frame_indices: List[int],
    triton_client,
    triton_model_name: str,
    triton_model_version: Optional[str],
    triton_input_name: str,
    triton_output_name: str,
    input_size: int,
    box_threshold: float,
    iou_threshold: float,
    class_names: Dict[int, str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[int, str], List[np.ndarray]]:
    """
    Triton-backed YOLO inference.

    Notes:
    - Fixed-shape baseline models are batch=1, so we process frames one-by-one.
    - We implement NMS locally to avoid ultralytics API drift.
    """
    def _xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
        # xywh: (N,4) with center x,y,w,h
        x = xywh[:, 0]
        y = xywh[:, 1]
        w = xywh[:, 2]
        h = xywh[:, 3]
        x1 = x - w / 2.0
        y1 = y - h / 2.0
        x2 = x + w / 2.0
        y2 = y + h / 2.0
        return np.stack([x1, y1, x2, y2], axis=1).astype(np.float32)

    def _nms_single_class(boxes_xyxy: np.ndarray, scores_: np.ndarray, iou_th: float, max_det: int) -> List[int]:
        if boxes_xyxy.size == 0:
            return []
        order = scores_.argsort()[::-1]
        keep: List[int] = []
        while order.size > 0 and len(keep) < int(max_det):
            i = int(order[0])
            keep.append(i)
            if order.size == 1:
                break
            rest = order[1:]
            ious = iou_xyxy(boxes_xyxy[i], boxes_xyxy[rest])
            order = rest[ious <= float(iou_th)]
        return keep

    def _decode_and_nms_yolo(out_b84n: np.ndarray) -> np.ndarray:
        """
        out_b84n: (1,84,N)
        returns det: (M,6) [x1,y1,x2,y2,conf,cls_id]
        """
        if out_b84n.ndim != 3 or out_b84n.shape[0] != 1 or out_b84n.shape[1] < 6:
            raise RuntimeError(f"{NAME} | unexpected YOLO output shape: {out_b84n.shape}")
        pred = out_b84n[0].T  # (N,84)
        boxes_xywh = pred[:, :4].astype(np.float32)
        cls_scores = pred[:, 4:].astype(np.float32)  # (N,nc)
        cls_id = np.argmax(cls_scores, axis=1).astype(np.int32)
        conf = np.max(cls_scores, axis=1).astype(np.float32)
        m = conf >= float(box_threshold)
        if not np.any(m):
            return np.zeros((0, 6), dtype=np.float32)
        boxes_xyxy = _xywh_to_xyxy(boxes_xywh[m])
        conf_m = conf[m]
        cls_m = cls_id[m]

        dets: List[np.ndarray] = []
        for c in np.unique(cls_m):
            idx = np.where(cls_m == c)[0]
            if idx.size == 0:
                continue
            keep = _nms_single_class(boxes_xyxy[idx], conf_m[idx], float(iou_threshold), int(MAX_DETECTIONS))
            if not keep:
                continue
            kk = idx[np.asarray(keep, dtype=np.int64)]
            cc = np.full((kk.size, 1), float(c), dtype=np.float32)
            dets.append(np.concatenate([boxes_xyxy[kk], conf_m[kk, None], cc], axis=1))

        if not dets:
            return np.zeros((0, 6), dtype=np.float32)
        det = np.concatenate(dets, axis=0)
        # global top-k by conf
        order = det[:, 4].argsort()[::-1]
        det = det[order[: int(MAX_DETECTIONS)]]
        return det.astype(np.float32)

    n = len(frame_indices)
    boxes = np.zeros((n, MAX_DETECTIONS, BBOX_DIMS), dtype=np.float32)
    scores = np.zeros((n, MAX_DETECTIONS), dtype=np.float32)
    class_ids = np.zeros((n, MAX_DETECTIONS), dtype=np.int32)
    valid_mask = np.zeros((n, MAX_DETECTIONS), dtype=bool)
    raw_per_frame: List[np.ndarray] = [np.zeros((0, 5), dtype=np.float32) for _ in frame_indices]

    for i_out, fi in enumerate(frame_indices):
        fr_rgb = frame_manager.get(int(fi))  # RGB uint8
        x, r, pad, orig_hw = _prep_yolo_tensor_from_rgb_uint8(fr_rgb, input_size=int(input_size))
        try:
            res = triton_client.infer(
                model_name=str(triton_model_name),
                model_version=str(triton_model_version) if triton_model_version else None,
                input_name=str(triton_input_name),
                input_tensor=x,
                output_name=str(triton_output_name),
                datatype="FP32",
            )
        except Exception as e:
            raise RuntimeError(f"{NAME} | Triton infer failed: {e}") from e

        out = np.asarray(res.output, dtype=np.float32)  # expected (1,84,N)
        det_np = _decode_and_nms_yolo(out)
        if det_np.size == 0:
            continue
        # scale boxes back to original
        det_np[:, :4] = _scale_boxes_back(det_np[:, :4], r=float(r), pad=pad, orig_hw=orig_hw)

        detections: List[List[float]] = []
        for j in range(min(det_np.shape[0], MAX_DETECTIONS)):
            xyxy = det_np[j, :4].astype(np.float32)
            conf = float(det_np[j, 4])
            cls_id = int(det_np[j, 5])
            if conf < float(box_threshold):
                continue
            boxes[i_out, j] = xyxy
            scores[i_out, j] = conf
            class_ids[i_out, j] = cls_id
            valid_mask[i_out, j] = True
            detections.append([float(xyxy[0]), float(xyxy[1]), float(xyxy[2]), float(xyxy[3]), float(conf)])

            if cls_id not in class_names:
                class_names[cls_id] = f"class_{cls_id}"

        raw_per_frame[i_out] = np.asarray(detections, dtype=np.float32)

        if (i_out + 1) % 25 == 0:
            LOGGER.info("%s | Triton YOLO | processed %d/%d", NAME, i_out + 1, n)

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
        # Ensure local ByteTrack/yolox package is importable (vendored in this repo).
        bt_root = os.path.join(os.path.dirname(__file__), "ByteTrack")
        if bt_root not in sys.path:
            sys.path.insert(0, bt_root)
        from yolox.tracker.byte_tracker import BYTETracker
    except Exception as e:
        LOGGER.exception("ByteTrack import failed: %s", e)
        raise

    class args:
        track_thresh=tracker_cfg.get("track_thresh", 0.25)
        match_thresh=tracker_cfg.get("match_thresh", 0.8)
        track_buffer=5
        mot20=True

    tracker = BYTETracker(
        args,
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
    # IMPORTANT: tmp must have .npz suffix, otherwise numpy will write to tmp + ".npz"
    # leaving tmp empty and corrupting the final artifact on os.replace().
    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(path) + ".", suffix=".npz", dir=os.path.dirname(path))
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
    parser.add_argument("--runtime", type=str, default="ultralytics", choices=["ultralytics", "triton"])
    # Triton (preferred via ModelManager specs)
    parser.add_argument("--triton-model-spec", type=str, default=None, help="dp_models spec name (e.g., yolo11x_640_triton)")
    parser.add_argument("--triton-http-url", type=str, default=None)
    parser.add_argument("--triton-model-name", type=str, default=None)
    parser.add_argument("--triton-model-version", type=str, default=None)
    parser.add_argument("--triton-input-name", type=str, default="images")
    parser.add_argument("--triton-output-name", type=str, default="output0")
    parser.add_argument("--triton-preprocess-preset", type=str, default="yolo11x_640", choices=["yolo11x_320", "yolo11x_640", "yolo11x_960"])
    parser.add_argument("--batch-size", type=int, required=True, help="Batch size (must be provided by scheduler/orchestrator)")
    parser.add_argument("--box-threshold", type=float, default=0.6)
    parser.add_argument("--device", type=str, default="auto", help="'auto'|'cpu'|'cuda'")
    parser.add_argument("--iou-threshold", type=float, default=0.3)
    args = parser.parse_args()
    # Expand env vars in --model (so configs can use ${DP_MODELS_ROOT}/...).
    if isinstance(args.model, str):
        args.model = os.path.expandvars(str(args.model))

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
    if len(frame_indices) <= 0:
        raise RuntimeError(f"{NAME} | empty frame_indices is not allowed (no-fallback)")

    frame_manager = FrameManager(
        frames_dir=args.frames_dir,
        chunk_size=meta.get("chunk_size", 32),
        cache_size=meta.get("cache_size", 2),
    )

    try:
        device = pick_device(args.device)
        batch_size = int(args.batch_size)
        if batch_size <= 0:
            raise RuntimeError(f"{NAME} | --batch-size must be > 0 (scheduler-controlled); got {batch_size}")

        class_names: Dict[int, str] = {}
        if str(args.runtime).lower() == "triton":
            # Load class names from local weights file (NO network).
            if not os.path.exists(str(args.model)):
                # If user passed a relative path and DP_MODELS_ROOT is set, try resolving from it.
                mr = os.environ.get("DP_MODELS_ROOT")
                if mr and not os.path.isabs(str(args.model)):
                    cand = os.path.join(str(mr), str(args.model))
                    if os.path.exists(cand):
                        args.model = cand
                if not os.path.exists(str(args.model)):
                    raise RuntimeError(
                        f"{NAME} | runtime=triton requires --model pointing to local weights (for class names). "
                        f"File not found: {args.model}"
                    )
            try:
                from ultralytics import YOLO  # type: ignore
                y = YOLO(str(args.model))
                class_names = {int(k): str(v) for k, v in getattr(y, "names", {}).items()}
            except Exception:
                class_names = {}

            # Resolve Triton client/params
            from dp_triton import TritonHttpClient, TritonError  # type: ignore

            if args.triton_model_spec:
                mm_entry = _load_triton_spec_via_model_manager(str(args.triton_model_spec))
                client = mm_entry["client"]
                rp = mm_entry["rp"]
                args.triton_http_url = str(rp.get("triton_http_url") or args.triton_http_url or "")
                args.triton_model_name = str(rp.get("triton_model_name") or args.triton_model_name or "")
                args.triton_model_version = str(rp.get("triton_model_version") or "") or None
                args.triton_input_name = str(rp.get("triton_input_name") or args.triton_input_name)
                args.triton_output_name = str(rp.get("triton_output_name") or args.triton_output_name)
            else:
                if not args.triton_http_url or not str(args.triton_http_url).strip():
                    raise RuntimeError(f"{NAME} | runtime=triton requires --triton-http-url or --triton-model-spec (no-fallback)")
                if not args.triton_model_name or not str(args.triton_model_name).strip():
                    raise RuntimeError(f"{NAME} | runtime=triton requires --triton-model-name or --triton-model-spec (no-fallback)")
                client = TritonHttpClient(base_url=str(args.triton_http_url), timeout_sec=10.0)
                if not client.ready():
                    raise TritonError(f"{NAME} | Triton is not ready at {args.triton_http_url}", error_code="triton_unavailable")

            if batch_size != 1:
                LOGGER.info("%s | runtime=triton: forcing fixed batch=1 (was %d)", NAME, batch_size)

            preset = str(args.triton_preprocess_preset).strip().lower()
            if preset == "yolo11x_320":
                input_size = 320
            elif preset == "yolo11x_640":
                input_size = 640
            elif preset == "yolo11x_960":
                input_size = 960
            else:
                raise RuntimeError(f"{NAME} | unknown triton_preprocess_preset: {preset}")

            boxes, scores, class_ids, valid_mask, class_names, raw_per_frame = run_yolo_triton(
                frame_manager=frame_manager,
                frame_indices=frame_indices,
                triton_client=client,
                triton_model_name=str(args.triton_model_name),
                triton_model_version=str(args.triton_model_version) if args.triton_model_version else None,
                triton_input_name=str(args.triton_input_name),
                triton_output_name=str(args.triton_output_name),
                input_size=int(input_size),
                box_threshold=float(args.box_threshold),
                iou_threshold=float(args.iou_threshold),
                class_names=class_names,
            )
            impl = f"triton:{args.triton_model_name}"
        else:
            boxes, scores, class_ids, valid_mask, class_names, raw_per_frame = run_yolo(
                frame_manager=frame_manager,
                frame_indices=frame_indices,
                model_path=str(args.model),
                box_threshold=float(args.box_threshold),
                batch_size=batch_size,
                device=device,
            )
            impl = "yolo"

        # Tracking is REQUIRED (per project decision).
        tracker_cfg = {
            "track_thresh": 0.25,
            "match_thresh": 0.8,
            "frame_rate": int(getattr(frame_manager, "fps", meta.get("fps", 30))),
        }
        tracks_arr, tracks_map = run_tracking(
            raw_per_frame=raw_per_frame,
            frame_indices=frame_indices,
            frame_manager=frame_manager,
            tracker_cfg=tracker_cfg,
            iou_threshold=float(args.iou_threshold),
        )

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
            "schema_version": SCHEMA_VERSION,
            "created_at": created_at,
            "status": "ok",
            "empty_reason": None,
            "impl": impl,
            "model": args.model,
            "box_threshold": args.box_threshold,
            "batch_size": int(batch_size),
            "device": str(device),
            "total_frames": int(total_frames),
        }
        required_run_keys = ["platform_id", "video_id", "run_id", "sampling_policy_version", "config_hash"]
        missing = [k for k in required_run_keys if not meta.get(k)]
        if missing:
            raise RuntimeError(f"{NAME} | frames metadata missing required run identity keys: {missing}")
        for k in required_run_keys:
            meta_info[k] = meta.get(k)

        # PR-3: model system baseline
        if str(args.runtime).lower() == "triton":
            model_name = str(args.triton_model_name or "triton")
            meta_info = apply_models_meta(
                meta_info,
                models_used=[
                    model_used(
                        model_name=model_name,
                        model_version=str(args.triton_model_version or "1"),
                        weights_digest="provided_by_deploy",
                        runtime="triton",
                        engine="triton",
                        precision="fp32",
                        device="cuda",
                    )
                ],
            )
        else:
            model_name = str(args.model)
            engine = "ultralytics"
            meta_info = apply_models_meta(
                meta_info,
                models_used=[
                    model_used(
                        model_name=model_name,
                        model_version="unknown",
                        weights_digest="unknown",
                        runtime="inprocess",
                        engine=engine,
                        precision="fp32",
                        device=str(device),
                    )
                ],
            )

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
