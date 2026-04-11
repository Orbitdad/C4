"""
JARVIS Face Recognition Sub-package.
Provides real-time face detection, recognition, and identity management
via InsightFace. Separate from gesture tracking.
"""
from .detector import FaceDetector, DetectedFace
from .recognizer import FaceRecognizer
from .identity_manager import IdentityManager
from .face_pipeline import FacePipeline

__all__ = [
    "FaceDetector",
    "DetectedFace",
    "FaceRecognizer",
    "IdentityManager",
    "FacePipeline",
]
