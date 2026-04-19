"""
jarvis/vision/gesture/detector.py

Modular MediaPipe Hand Detector.
Responsible for raw landmark extraction and basic point helpers.
Supports up to 2 hands for dual-hand interaction.
"""

from typing import Optional, List, Tuple, Any


class HandDetector:
    """
    Provides helper methods that operate on raw MediaPipe landmark lists.

    Note: actual MediaPipe hands inference is performed in VisionManager.
    This class is kept for landmark-query utilities.
    """

    @staticmethod
    def get_index_tip(landmarks: List[Any]) -> Tuple[float, float, float]:
        """Extract index finger tip (landmark id 8)."""
        if not landmarks or len(landmarks) < 21:
            return (0.0, 0.0, 0.0)
        tip = landmarks[8]
        return (tip.x, tip.y, tip.z)

    @staticmethod
    def get_thumb_tip(landmarks: List[Any]) -> Tuple[float, float, float]:
        """Extract thumb tip (landmark id 4)."""
        if not landmarks or len(landmarks) < 21:
            return (0.0, 0.0, 0.0)
        tip = landmarks[4]
        return (tip.x, tip.y, tip.z)

    @staticmethod
    def get_wrist(landmarks: List[Any]) -> Tuple[float, float, float]:
        """Extract wrist position (landmark id 0)."""
        if not landmarks or len(landmarks) < 21:
            return (0.0, 0.0, 0.0)
        w = landmarks[0]
        return (w.x, w.y, w.z)
