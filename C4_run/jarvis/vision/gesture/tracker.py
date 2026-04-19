"""
jarvis/vision/gesture/tracker.py

Smooths hand movement using a moving average filter to reduce jitter.
Supports per-hand tracking and full XYZ smoothing.
"""

from collections import deque
from typing import Tuple


class HandTracker:
    """Moving-average smoother for a single hand's position (x, y, z)."""

    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.history_x: deque = deque(maxlen=window_size)
        self.history_y: deque = deque(maxlen=window_size)
        self.history_z: deque = deque(maxlen=window_size)

    def smooth(self, pos: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """
        Apply moving average smoothing to the (x, y, z) coordinates.
        Returns the smoothed position.
        """
        x, y, z = pos

        self.history_x.append(x)
        self.history_y.append(y)
        self.history_z.append(z)

        avg_x = sum(self.history_x) / len(self.history_x)
        avg_y = sum(self.history_y) / len(self.history_y)
        avg_z = sum(self.history_z) / len(self.history_z)

        return (avg_x, avg_y, avg_z)

    def reset(self):
        self.history_x.clear()
        self.history_y.clear()
        self.history_z.clear()

    @property
    def is_stable(self) -> bool:
        """Check if we have enough history for a stable reading."""
        return len(self.history_x) >= self.window_size
