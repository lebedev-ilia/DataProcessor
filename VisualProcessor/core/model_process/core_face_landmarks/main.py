#!/VisualProcessor/core/model_process/.model_process_venv python3

import argparse
import os
import sys
from datetime import datetime
from typing import List

import cv2              # type: ignore
import numpy as np      # type: ignore
import mediapipe as mp 

_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
if _path not in sys.path:
    sys.path.append(_path)

from utils.frame_manager import FrameManager
from utils.logger import get_logger
from utils.utilites import load_metadata


VERSION = "2.0"
NAME = "core_face_landmarks"
LOGGER = get_logger(NAME)


POSE_LANDMARKS = 33
POSE_DIMS = 4          # x, y, z, visibility

HAND_LANDMARKS = 21
HAND_DIMS = 3          # x, y, z

FACE_LANDMARKS = 468
FACE_DIMS = 3          # x, y, z


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
                        for j, lm in enumerate(hand.landmark):
                            hands_data[i, h, j] = (lm.x, lm.y, lm.z)

            if mp_face:
                res = mp_face.process(frame_bgr)
                if res.multi_face_landmarks:
                    max_landmarks = face_data.shape[2]

                    for f, face in enumerate(res.multi_face_landmarks):
                        if f >= cfg.face_mesh_max_num_faces:
                            break

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

    return pose_data, hands_data, face_data


def main():
    parser = argparse.ArgumentParser(description="Production MediaPipe landmarks extractor")
    parser.add_argument("--frames-dir", required=True)
    parser.add_argument("--rs-path", required=True)
    parser.add_argument("--use-pose", action="store_true")
    parser.add_argument("--use-hands", action="store_true")
    parser.add_argument("--use-face-mesh", action="store_true")
    parser.add_argument("--pose-static-image-mode", action="store_true")
    parser.add_argument("--pose-model-complexity", type=int)
    parser.add_argument("--pose-enable-segmentation", action="store_true")
    parser.add_argument("--pose-min-detection-confidence", type=float)
    parser.add_argument("--pose-min-tracking-confidence", type=float)
    parser.add_argument("--hands-static-image-mode", action="store_true")
    parser.add_argument("--hands-max-num-hands", type=int)
    parser.add_argument("--hands-model-complexity", type=int)
    parser.add_argument("--hands-min-detection-confidence", type=float)
    parser.add_argument("--hands-min-tracking-confidence", type=float)
    parser.add_argument("--face-mesh-static-image-mode", action="store_true")
    parser.add_argument("--face-mesh-max-num-faces", type=int)
    parser.add_argument("--face-mesh-refine-landmarks", action="store_true")
    parser.add_argument("--face-mesh-min-detection-confidence", type=float)
    parser.add_argument("--face-mesh-min-tracking-confidence", type=float)
    args = parser.parse_args()

    meta = load_metadata(os.path.join(args.frames_dir, "metadata.json"), NAME)
    total_frames = int(meta["total_frames"])

    frame_indices = list(range(0, total_frames, 10))
    LOGGER.warning(f"{NAME} | Sampling frames: {len(frame_indices)} | "
        "(Это фиксированая выборка для провайдера core_face_landmarks, но использовать ее могут разные модули с разной "
        "логикой извлечения фичей. В будующем нужно грамотно коректировать выборку для получения хорошего качества фичей на всех модулях)"
    )

    frame_manager = FrameManager(
        frames_dir=args.frames_dir,
        chunk_size=meta.get("chunk_size", 32),
        cache_size=meta.get("cache_size", 2),
    )

    pose, hands, face = process_video(
        frame_manager=frame_manager,
        frame_indices=frame_indices,
        cfg=args,
    )

    frame_manager.close()

    out_path = os.path.join(args.rs_path, "landmarks.npz")

    np.savez_compressed(
        out_path,
        version=VERSION,
        created_at=datetime.utcnow().isoformat(),
        model_name="mediapipe",
        total_frames=total_frames,
        frame_indices=np.array(frame_indices, dtype=np.int32),
        pose_landmarks=pose,
        hands_landmarks=hands,
        face_landmarks=face,
    )

    LOGGER.info(f"{NAME} | Saved result: {out_path}")


if __name__ == "__main__":
    main()