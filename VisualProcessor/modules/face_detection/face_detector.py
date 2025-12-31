"""
Этот модуль нужно вынести в core (будет идти после object_detections(1) - если нашел person то face_detection(2) по этим кадрам ищет лица, если нашел то по этим кадрам строятся landmarks(3))
"""

import numpy as np
import time

from utils.logger import get_logger
logger = get_logger("FaceDetector")

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

class FaceDetector:
    def __init__(
        self,
        detect_thr: float = 0.3,
        det_size = (640, 640)
    ) -> None:
        self.detect_thr = detect_thr
        self.face_app = self.init_face_app(det_size)

    def detect_face(self, frame_bgr: np.ndarray, face_app, thr: float = 0.5) -> bool:
        """
        frame_bgr: OpenCV BGR ndarray
        Returns True if any face >= thr
        """
        faces = face_app.get(frame_bgr)
        if not faces:
            return False
        best = max(self.safe_det_score(f) for f in faces)
        return best >= thr


    def init_face_app(self, det_size=(640, 640)):
        try:
            from insightface.app import FaceAnalysis
        except Exception as e:
            raise RuntimeError("insightface not installed or failed to import") from e

        # force det_size to int tuple
        det_size = tuple(int(x) for x in det_size)

        try:
            app = FaceAnalysis(providers=["CUDAExecutionProvider"])
            app.prepare(ctx_id=0, det_size=det_size)
            logger.info(f"Detector | init_face_app | GPU Detector")
        except Exception as e:
            logger.error(f"Detector | init_face_app | Error: {e}")
            app = FaceAnalysis(providers=["CPUExecutionProvider"])
            app.prepare(ctx_id=-1, det_size=det_size)
        return app


    def safe_det_score(self, face) -> float:
        return float(getattr(face, "det_score", getattr(face, "score", 0.0) or 0.0))

    def run(self, frame_manager, frame_indices):
        """
        Scan video for face presence. Returns sorted list of frame indices where face detected.
        scan_stride can be >1 to speed up scanning.
        """
        timeline = {"frames_with_face": []}
        t = time.time()
        c = 0
        # iterate with stride, but we will later use windows around hits
        for idx in frame_indices:
            try:
                frame = frame_manager.get(idx)
                c += 1
            except IndexError:
                continue
            # detect expects BGR
            if self.detect_face(frame, self.face_app, thr=self.detect_thr):
                timeline["frames_with_face"].append(idx)
            if c % 20 == 0:
                ti = time.time()
                tok = round(ti - t, 2)
                t = ti
                l = len(timeline["frames_with_face"])
                logger.info(f"Detector | Обработано кадров: {c}/{len(frame_indices)} | Лиц: {l} | Время: {tok}")

        return timeline