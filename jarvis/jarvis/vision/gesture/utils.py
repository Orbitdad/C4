"""
jarvis/vision/gesture/utils.py
Small shared helpers for gesture modules.
"""

from __future__ import annotations

from typing import Any, Tuple


def index_tip_xyz(landmarks: Any) -> Tuple[float, float, float]:
    if not landmarks or len(landmarks) < 9:
        return (0.0, 0.0, 0.0)
    tip = landmarks[8]
    return (float(getattr(tip, "x", 0.0)), float(getattr(tip, "y", 0.0)), float(getattr(tip, "z", 0.0)))

