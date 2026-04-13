"""
jarvis/vision/gesture/tracker.py

Smooths hand movement using a moving average filter to reduce jitter.
"""

from collections import deque
from typing import Tuple, List

class HandTracker:
    def __init__(self, window_size: int = 5):
        self.window_size = window_size
        self.history_x = deque(maxlen=window_size)
        self.history_y = deque(maxlen=window_size)

    def smooth(self, pos: Tuple[float, float, float]) -> Tuple[float, float, float]:
        """
        Apply moving average smoothing to the (x, y) coordinates.
        z (depth) is returned as is.
        """
        x, y, z = pos
        
        self.history_x.append(x)
        self.history_y.append(y)
        
        avg_x = sum(self.history_x) / len(self.history_x)
        avg_y = sum(self.history_y) / len(self.history_y)
        
        return (avg_x, avg_y, z)

    def reset(self):
        self.history_x.clear()
        self.history_y.clear()

    @property
    def is_stable(self) -> bool:
        """Check if we have enough history for a stable reading."""
        return len(self.history_x) >= self.window_size
