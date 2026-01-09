#!/VisualProcessor/core/model_process/.model_process_venv python3



import argparse
import math
import os
import sys
import hashlib
import json
from datetime import datetime
from typing import List, Any, Dict, Optional, Tuple

import cv2              # type: ignore
import numpy as np      # type: ignore
import mediapipe as mp 

_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if _path not in sys.path:
    sys.path.append(_path)

from utils.frame_manager import FrameManager
from utils.logger import get_logger
from utils.utilites import load_metadata
from utils.meta_builder import apply_models_meta, model_used


VERSION = "2.0"
NAME = "core_face_landmarks"
SCHEMA_VERSION = "core_face_landmarks_npz_v1"
LOGGER = get_logger(NAME)


POSE_LANDMARKS = 33
POSE_DIMS = 4          # x, y, z, visibility

HAND_LANDMARKS = 21
HAND_DIMS = 3          # x, y, z

FACE_LANDMARKS = 468
FACE_DIMS = 3          # x, y, z


def _pick_detection_stride(n_frames: int, *, target: int, min_frames: int, max_frames: int) -> int:
    """
    Stage-1 sampling: choose stride on the PRIMARY frame_indices list (union-domain).
    We want ~target frames to run lightweight face detection on (bounded by min/max).
    """
    n = max(0, int(n_frames))
    if n <= 0:
        return 1
    t = max(1, int(target))
    mn = max(1, int(min_frames))
    mx = max(mn, int(max_frames))
    desired = int(max(mn, min(mx, t)))
    # stride >= 1
    return max(1, int(math.ceil(float(n) / float(desired))))


def _pick_window_radius(stride: int, *, min_radius: int = 1, max_radius: int = 5) -> int:
    """
    Stage-2 policy: if stage-1 detection stride is large, expand window a bit.
    This keeps "face neighbourhood" coverage without running FaceMesh everywhere.
    """
    s = max(1, int(stride))
    r = max(min_radius, int(math.ceil(s / 5.0)))
    return int(min(max_radius, r))


def init_face_detector(cfg):
    # MediaPipe lightweight face detector (faster than FaceMesh).
    # Uses an internal tracker-like temporal consistency when frames are close enough.
    return mp.solutions.face_detection.FaceDetection(
        model_selection=int(getattr(cfg, "face_detection_model_selection", 0)),
        min_detection_confidence=float(getattr(cfg, "face_detection_min_confidence", 0.5)),
    )


def _stage1_detect_faces(
    frame_manager: FrameManager,
    frame_indices_primary: List[int],
    cfg,
) -> Tuple[List[int], List[int], int]:
    """
    Stage-1: lightweight face detection on a sparse subset of PRIMARY indices.
    Returns:
      - det_primary_pos: positions (0..N-1) in primary list where we ran detection
      - det_face_pos: subset of det_primary_pos where at least one face was detected
      - stride used
    """
    n = len(frame_indices_primary)
    stride = _pick_detection_stride(
        n,
        target=int(getattr(cfg, "face_detection_target_frames", 50)),
        min_frames=int(getattr(cfg, "face_detection_min_frames", 20)),
        max_frames=int(getattr(cfg, "face_detection_max_frames", 200)),
    )
    det_primary_pos = list(range(0, n, stride))
    det_face_pos: List[int] = []

    fd = init_face_detector(cfg)
    try:
        for j, pos in enumerate(det_primary_pos):
            idx = frame_indices_primary[pos]
            fr_rgb = frame_manager.get(idx)
            fr_bgr = cv2.cvtColor(fr_rgb, cv2.COLOR_RGB2BGR)
            res = fd.process(fr_bgr)
            if getattr(res, "detections", None):
                det_face_pos.append(int(pos))
            if j % 30 == 0:
                LOGGER.info(f"{NAME} | stage1 | processed {j + 1}/{len(det_primary_pos)} det frames")
    finally:
        try:
            fd.close()
        except Exception:
            pass

    return det_primary_pos, det_face_pos, stride


def _stage2_select_face_mesh_positions(
    n_frames: int,
    det_face_positions: List[int],
    stride: int,
    cfg,
) -> List[int]:
    """
    Stage-2: choose which PRIMARY frame positions will run FaceMesh.
    Policy: for each detected face position, include itself + a small window around it.
    """
    n = max(0, int(n_frames))
    if n <= 0:
        return []
    if not det_face_positions:
        return []
    # window radius can be explicitly overridden, otherwise derived from stride.
    if getattr(cfg, "face_mesh_window_radius", None) is not None:
        r = int(getattr(cfg, "face_mesh_window_radius"))
        r = max(0, r)
    else:
        r = _pick_window_radius(stride)

    sel = set()
    for p in det_face_positions:
        pp = int(p)
        for q in range(pp - r, pp + r + 1):
            if 0 <= q < n:
                sel.add(int(q))
    return sorted(sel)


def init_pose(cfg):
    return mp.solutions.pose.Pose(
        static_image_mode=cfg.pose_static_image_mode,
        model_complexity=cfg.pose_model_complexity,
        enable_segmentation=cfg.pose_enable_segmentation,
        min_detection_confidence=cfg.pose_min_detection_confidence,
        min_tracking_confidence=cfg.pose_min_tracking_confidence,
    )


def init_hands(cfg):
    return mp.solutions.hands.Hands(
        static_image_mode=cfg.hands_static_image_mode,
        max_num_hands=cfg.hands_max_num_hands,
        model_complexity=cfg.hands_model_complexity,
        min_detection_confidence=cfg.hands_min_detection_confidence,
        min_tracking_confidence=cfg.hands_min_tracking_confidence,
    )


def init_face(cfg):
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=cfg.face_mesh_static_image_mode,
        refine_landmarks=cfg.face_mesh_refine_landmarks,
        max_num_faces=cfg.face_mesh_max_num_faces,
        min_detection_confidence=cfg.face_mesh_min_detection_confidence,
        min_tracking_confidence=cfg.face_mesh_min_tracking_confidence,
    )


def process_video(
    frame_manager: FrameManager,
    frame_indices: List[int],
    cfg,
):

    n_frames = len(frame_indices)

    pose_data = (
        np.full((n_frames, POSE_LANDMARKS, POSE_DIMS), np.nan, dtype=np.float32)
        if cfg.use_pose else None
    )

    hands_data = (
        np.full(
            (n_frames, cfg.hands_max_num_hands, HAND_LANDMARKS, HAND_DIMS),
            np.nan,
            dtype=np.float32,
        )
        if cfg.use_hands else None
    )

    face_data = (
        np.full(
            (n_frames, cfg.face_mesh_max_num_faces, FACE_LANDMARKS, FACE_DIMS),
            np.nan,
            dtype=np.float32,
        )
        if cfg.use_face_mesh else None
    )

    # Validity masks: explicit "worked but empty" signals for downstream modules.
    pose_present = np.zeros((n_frames,), dtype=bool) if cfg.use_pose else None
    hands_present = (
        np.zeros((n_frames, cfg.hands_max_num_hands), dtype=bool) if cfg.use_hands else None
    )
    face_present = (
        np.zeros((n_frames, cfg.face_mesh_max_num_faces), dtype=bool) if cfg.use_face_mesh else None
    )

    # Stage-1/2 acceleration for face_mesh:
    # - stage1: sparse face detection on PRIMARY indices
    # - stage2: run FaceMesh only near detected faces, but OUTPUT stays aligned to PRIMARY indices.
    face_mesh_positions: Optional[set[int]] = None
    det_stride = 1
    if cfg.use_face_mesh:
        det_pos, det_face_pos, det_stride = _stage1_detect_faces(frame_manager, frame_indices, cfg)
        sel_pos = _stage2_select_face_mesh_positions(n_frames, det_face_pos, det_stride, cfg)
        face_mesh_positions = set(sel_pos)
        LOGGER.info(
            f"{NAME} | stage1/2 | primary_frames={n_frames} det_frames={len(det_pos)} "
            f"faces_found={len(det_face_pos)} face_mesh_frames={len(sel_pos)} stride={det_stride}"
        )

    mp_pose = init_pose(cfg) if cfg.use_pose else None
    mp_hands = init_hands(cfg) if cfg.use_hands else None
    mp_face = init_face(cfg) if cfg.use_face_mesh else None

    LOGGER.info(f"{NAME} | Models initialized")

    try:
        for i, frame_idx in enumerate(frame_indices):
            frame_rgb = frame_manager.get(frame_idx)
            frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            if mp_pose:
                res = mp_pose.process(frame_bgr)
                if res.pose_landmarks:
                    pose_present[i] = True
                    for j, lm in enumerate(res.pose_landmarks.landmark):
                        pose_data[i, j] = (
                            lm.x,
                            lm.y,
                            lm.z,
                            lm.visibility,
                        )

            if mp_hands:
                res = mp_hands.process(frame_bgr)
                if res.multi_hand_landmarks:
                    for h, hand in enumerate(res.multi_hand_landmarks):
                        if h >= cfg.hands_max_num_hands:
                            break
                        hands_present[i, h] = True
                        for j, lm in enumerate(hand.landmark):
                            hands_data[i, h, j] = (lm.x, lm.y, lm.z)

            if mp_face:
                # Skip expensive FaceMesh if stage2 did not select this frame.
                if face_mesh_positions is not None and i not in face_mesh_positions:
                    # leave NaNs + face_present=False
                    continue
                res = mp_face.process(frame_bgr)
                if res.multi_face_landmarks:
                    max_landmarks = face_data.shape[2]

                    for f, face in enumerate(res.multi_face_landmarks):
                        if f >= cfg.face_mesh_max_num_faces:
                            break
                        face_present[i, f] = True

                        for j, lm in enumerate(face.landmark):
                            if j >= max_landmarks:
                                break
                            face_data[i, f, j] = (lm.x, lm.y, lm.z)

            if i % 30 == 0:
                LOGGER.info(f"{NAME} | processed {i + 1}/{n_frames} frames")

    finally:
        if mp_pose:
            mp_pose.close()
        if mp_hands:
            mp_hands.close()
        if mp_face:
            mp_face.close()

    return pose_data, hands_data, face_data, pose_present, hands_present, face_present


def main():
    parser = argparse.ArgumentParser(description="Production MediaPipe landmarks extractor")
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--rs-path", required=True)
    parser.add_argument("--use-pose", action="store_true")
    parser.add_argument("--use-hands", action="store_true")
    parser.add_argument("--use-face-mesh", action="store_true")
    # Stage-1 lightweight face detection (used to decide where to run FaceMesh)
    parser.add_argument("--face-detection-target-frames", type=int, default=50)
    parser.add_argument("--face-detection-min-frames", type=int, default=20)
    parser.add_argument("--face-detection-max-frames", type=int, default=200)
    parser.add_argument("--face-detection-model-selection", type=int, default=0, choices=[0, 1])
    parser.add_argument("--face-detection-min-confidence", type=float, default=0.5)
    # Stage-2 FaceMesh window (primary index positions)
    parser.add_argument(
        "--face-mesh-window-radius",
        type=int,
        default=None,
        help="If set, run FaceMesh on detected frames plus +/- radius neighbours (in primary frame_indices positions). "
             "If not set, derived from detection stride.",
    )
    parser.add_argument("--pose-static-image-mode", action="store_true")
    parser.add_argument("--pose-model-complexity", type=int, default=2)
    parser.add_argument("--pose-enable-segmentation", action="store_true")
    parser.add_argument("--pose-min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--pose-min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--hands-static-image-mode", action="store_true")
    parser.add_argument("--hands-max-num-hands", type=int, default=2)
    parser.add_argument("--hands-model-complexity", type=int, default=1)
    parser.add_argument("--hands-min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--hands-min-tracking-confidence", type=float, default=0.5)
    parser.add_argument("--face-mesh-static-image-mode", action="store_true")
    parser.add_argument("--face-mesh-max-num-faces", type=int, default=1)
    parser.add_argument("--face-mesh-refine-landmarks", action="store_true")
    parser.add_argument("--face-mesh-min-detection-confidence", type=float, default=0.5)
    parser.add_argument("--face-mesh-min-tracking-confidence", type=float, default=0.5)
    args = parser.parse_args()

    # Baseline policy: core_face_landmarks must produce face_mesh outputs (shot_quality depends on it).
    if not bool(args.use_face_mesh):
        raise RuntimeError(f"{NAME} | baseline requires --use-face-mesh (no-fallback)")

    meta = load_metadata(os.path.join(args.frames_dir, "metadata.json"), NAME)
    total_frames = int(meta["total_frames"])

    # Strict sampling contract: Segmenter must provide per-provider indices in metadata[NAME].frame_indices.
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
    LOGGER.info(f"{NAME} | sampled frames: {len(frame_indices)} / total={total_frames}")

    frame_manager = FrameManager(
        frames_dir=args.frames_dir,
        chunk_size=meta.get("chunk_size", 32),
        cache_size=meta.get("cache_size", 2),
    )

    pose, hands, face, pose_present, hands_present, face_present = process_video(
        frame_manager=frame_manager,
        frame_indices=frame_indices,
        cfg=args,
    )

    frame_manager.close()

    out_dir = os.path.join(args.rs_path, NAME)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "landmarks.npz")

    # Global availability flags for human-friendly consumers
    has_any_face = bool(np.any(face_present)) if face_present is not None else False
    has_any_pose = bool(np.any(pose_present)) if pose_present is not None else False
    has_any_hands = bool(np.any(hands_present)) if hands_present is not None else False

    # Valid empty reasons:
    # - no_faces_in_video is a valid empty for this provider (NOT an error)
    face_empty_reason: Optional[str] = None
    pose_empty_reason: Optional[str] = None
    hands_empty_reason: Optional[str] = None
    if args.use_face_mesh and not has_any_face:
        face_empty_reason = "no_faces_in_video"
    if args.use_pose and not has_any_pose:
        pose_empty_reason = "no_pose_detected"
    if args.use_hands and not has_any_hands:
        hands_empty_reason = "no_hands_detected"

    # Provider-level status: empty only for face-mesh absence (baseline consumer logic expects this)
    status = "empty" if face_empty_reason else "ok"
    empty_reason = face_empty_reason

    meta_out = {
        "producer": NAME,
        "producer_version": VERSION,
        "schema_version": SCHEMA_VERSION,
        "created_at": datetime.utcnow().isoformat(),
        "status": status,
        "empty_reason": empty_reason,
        "model_name": "mediapipe",
        "total_frames": int(total_frames),
        # extended empty reasons (optional)
        "face_empty_reason": face_empty_reason,
        "pose_empty_reason": pose_empty_reason,
        "hands_empty_reason": hands_empty_reason,
    }
    required_run_keys = ["platform_id", "video_id", "run_id", "sampling_policy_version", "config_hash"]
    missing = [k for k in required_run_keys if not meta.get(k)]
    if missing:
        raise RuntimeError(f"{NAME} | frames metadata missing required run identity keys: {missing}")
    for k in required_run_keys:
        meta_out[k] = meta.get(k)

    # Optional but recommended for reproducibility
    if meta.get("dataprocessor_version") is not None:
        meta_out["dataprocessor_version"] = meta.get("dataprocessor_version")

    # PR-3: model system baseline (mediapipe models are internal; treat as a model component)
    meta_out["device"] = str(meta_out.get("device") or "cpu")
    # Deterministic digest: best-effort "weights token" based on mediapipe version + key config params.
    mp_ver = str(mp.__version__ if hasattr(mp, "__version__") else "unknown")
    digest_payload: Dict[str, Any] = {
        "engine": "mediapipe",
        "mediapipe_version": mp_ver,
        "use_pose": bool(args.use_pose),
        "use_hands": bool(args.use_hands),
        "use_face_mesh": True,
        "pose": {
            "static_image_mode": bool(args.pose_static_image_mode),
            "model_complexity": int(args.pose_model_complexity),
            "enable_segmentation": bool(args.pose_enable_segmentation),
            "min_detection_confidence": float(args.pose_min_detection_confidence),
            "min_tracking_confidence": float(args.pose_min_tracking_confidence),
        },
        "hands": {
            "static_image_mode": bool(args.hands_static_image_mode),
            "max_num_hands": int(args.hands_max_num_hands),
            "model_complexity": int(args.hands_model_complexity),
            "min_detection_confidence": float(args.hands_min_detection_confidence),
            "min_tracking_confidence": float(args.hands_min_tracking_confidence),
        },
        "face_mesh": {
            "static_image_mode": bool(args.face_mesh_static_image_mode),
            "max_num_faces": int(args.face_mesh_max_num_faces),
            "refine_landmarks": bool(args.face_mesh_refine_landmarks),
            "min_detection_confidence": float(args.face_mesh_min_detection_confidence),
            "min_tracking_confidence": float(args.face_mesh_min_tracking_confidence),
        },
    }
    digest_text = json.dumps(digest_payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    weights_digest = hashlib.sha256(digest_text.encode("utf-8")).hexdigest()
    meta_out = apply_models_meta(
        meta_out,
        models_used=[
            model_used(
                model_name="mediapipe",
                model_version=mp_ver,
                weights_digest=weights_digest,
                runtime="inprocess",
                engine="mediapipe",
                precision="fp32",
                device=str(meta_out.get("device") or "cpu"),
            )
        ],
    )

    np.savez_compressed(
        out_path,
        # legacy fields (kept)
        version=VERSION,
        created_at=meta_out["created_at"],
        model_name="mediapipe",
        total_frames=total_frames,
        frame_indices=np.array(frame_indices, dtype=np.int32),
        pose_landmarks=pose,
        hands_landmarks=hands,
        face_landmarks=face,
        pose_present=pose_present,
        hands_present=hands_present,
        face_present=face_present,
        has_any_face=np.asarray(has_any_face),
        has_any_pose=np.asarray(has_any_pose),
        has_any_hands=np.asarray(has_any_hands),
        empty_reason=np.asarray(empty_reason, dtype=object),
        face_empty_reason=np.asarray(face_empty_reason, dtype=object),
        pose_empty_reason=np.asarray(pose_empty_reason, dtype=object),
        hands_empty_reason=np.asarray(hands_empty_reason, dtype=object),
        # canonical meta (required by artifact_validator)
        meta=np.asarray(meta_out, dtype=object),
    )

    LOGGER.info(f"{NAME} | Saved result: {out_path}")


if __name__ == "__main__":
    main()