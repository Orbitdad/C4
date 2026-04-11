"""
Face Detector using InsightFace.

Architecture Decision:
    Uses `buffalo_sc` (small, CPU-optimized) model pack for real-time perf.
    On systems with CUDA, switch ctx_id=0 for GPU acceleration.
    Returns structured DetectedFace dataclasses, not raw dicts, to enforce
    strong typing throughout the pipeline.

Performance Notes:
    - InsightFace's analysis is ~30-50ms/frame on CPU.
    - We downsample the frame to 640px width before analysis to cut this in half.
    - Landmarks are used downstream for head-pose attention estimation.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class DetectedFace:
    """
    Structured output of a single detected face in a frame.
    """
    bbox: Tuple[int, int, int, int]          # (x1, y1, x2, y2) in original frame coords
    landmarks: Optional[np.ndarray]           # 5-point facial landmarks (Nx2 float array)
    embedding: Optional[np.ndarray]           # 512-dim ArcFace embedding vector
    detection_score: float = 0.0             # Detector confidence 0-1
    pose: Optional[Tuple[float, float, float]] = None  # (yaw, pitch, roll) in degrees


class FaceDetector:
    """
    Wraps InsightFace FaceAnalysis to detect and embed faces per frame.

    Responsible ONLY for detection + embedding extraction.
    Identity matching is handled downstream by FaceRecognizer.
    """

    # Inference model pack: buffalo_sc = detection + ArcFace embedding only (fastest)
    MODEL_PACK = "buffalo_sc"
    # Downscale input frames for ~2x speedup at minor accuracy cost
    INFERENCE_WIDTH = 640

    def __init__(self, model_root: str = "~/.insightface") -> None:
        self._app = None
        self._model_root = model_root
        self._initialized = False
        self._init_error: Optional[str] = None
        self._load_model()

    def _load_model(self) -> None:
        """Lazily load the InsightFace model. Gracefully handles missing dependency."""
        try:
            import insightface
            from insightface.app import FaceAnalysis

            # ctx_id=-1 forces CPU. Set ctx_id=0 on CUDA-capable machines for GPU.
            self._app = FaceAnalysis(
                name=self.MODEL_PACK,
                root=self._model_root,
                providers=["CPUExecutionProvider"]
            )
            # det_size: detection network input size. 320x320 is fastest.
            self._app.prepare(ctx_id=-1, det_size=(320, 320))
            self._initialized = True
            logger.info("[FaceDetector] InsightFace model loaded successfully.")
        except ImportError:
            self._init_error = "insightface not installed. Run: pip install insightface onnxruntime"
            logger.error(f"[FaceDetector] {self._init_error}")
        except Exception as e:
            self._init_error = str(e)
            logger.error(f"[FaceDetector] Failed to initialize InsightFace: {e}")

    @property
    def is_ready(self) -> bool:
        return self._initialized and self._app is not None

    def detect(self, frame_bgr: np.ndarray) -> List[DetectedFace]:
        """
        Detect all faces in a BGR OpenCV frame.

        Args:
            frame_bgr: Raw camera frame (H x W x 3, uint8, BGR).

        Returns:
            List of DetectedFace objects. Empty list if no faces detected or model unavailable.
        """
        if not self.is_ready:
            return []

        # Downscale for faster inference while maintaining aspect ratio
        h, w = frame_bgr.shape[:2]
        scale = self.INFERENCE_WIDTH / w if w > self.INFERENCE_WIDTH else 1.0
        if scale < 1.0:
            small = cv2.resize(frame_bgr, (int(w * scale), int(h * scale)))
        else:
            small = frame_bgr
            scale = 1.0

        try:
            faces_raw = self._app.get(small)
        except Exception as e:
            logger.warning(f"[FaceDetector] Detection error: {e}")
            return []

        results: List[DetectedFace] = []
        for face in faces_raw:
            # Scale bbox back to original frame coordinates
            x1, y1, x2, y2 = face.bbox.astype(int)
            bbox = (
                int(x1 / scale), int(y1 / scale),
                int(x2 / scale), int(y2 / scale)
            )

            # Scale landmarks back as well
            lm = None
            if face.kps is not None:
                lm = (face.kps / scale).astype(np.float32)

            # Estimate head pose from landmarks (yaw/pitch/roll)
            pose = self._estimate_pose(lm, (h, w)) if lm is not None else None

            results.append(DetectedFace(
                bbox=bbox,
                landmarks=lm,
                embedding=face.embedding,  # 512-dim float32 ArcFace embedding
                detection_score=float(face.det_score),
                pose=pose,
            ))

        return results

    def _estimate_pose(
        self, landmarks: np.ndarray, frame_shape: Tuple[int, int]
    ) -> Optional[Tuple[float, float, float]]:
        """
        Estimate rough head pose (yaw, pitch, roll) from 5-point landmarks using
        solvePnP with a standard 3D face model.

        Returns:
            (yaw, pitch, roll) in degrees, or None on failure.
        """
        # Standard 3D model points (nose, eyes, mouth corners, chin) in mm
        model_points_3d = np.array([
            [0.0, 0.0, 0.0],          # Nose tip
            [-30.0, -30.0, -30.0],    # Left eye
            [30.0, -30.0, -30.0],     # Right eye
            [-25.0, 25.0, -30.0],     # Mouth left
            [25.0, 25.0, -30.0],      # Mouth right
        ], dtype=np.float64)

        h, w = frame_shape
        cam_matrix = np.array([
            [w, 0, w / 2],
            [0, w, h / 2],
            [0, 0, 1]
        ], dtype=np.float64)

        image_points = landmarks[:5].astype(np.float64)
        dist_coeffs = np.zeros((4, 1))

        try:
            success, rot_vec, _ = cv2.solvePnP(
                model_points_3d, image_points, cam_matrix, dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            if not success:
                return None
            rot_mat, _ = cv2.Rodrigues(rot_vec)
            # Decompose to Euler angles
            sy = np.sqrt(rot_mat[0, 0] ** 2 + rot_mat[1, 0] ** 2)
            yaw   = float(np.degrees(np.arctan2(rot_mat[2, 1], rot_mat[2, 2])))
            pitch = float(np.degrees(np.arctan2(-rot_mat[2, 0], sy)))
            roll  = float(np.degrees(np.arctan2(rot_mat[1, 0], rot_mat[0, 0])))
            return (yaw, pitch, roll)
        except Exception:
            return None
