"""
jarvis/vision/gesture/intent_engine.py

Infer *user intent* from gesture data + motion characteristics.
Pure logic — no OS calls.
"""
from __future__ import annotations

from typing import Dict, List, Optional


# ── Intent constants ──────────────────────────────────────────────────────────
class Intent:
    CLICK      = "CLICK"
    SCROLL     = "SCROLL"
    DRAG       = "DRAG"
    NAVIGATE   = "NAVIGATE"
    COMMAND    = "COMMAND"
    PAUSE      = "PAUSE"
    MOVE_CURSOR = "MOVE_CURSOR"
    NONE       = "NONE"


class Mode:
    NORMAL    = "NORMAL"
    PRECISION = "PRECISION"   # slow, deliberate movement
    FAST      = "FAST"        # quick sweep — likely navigation


# ── Gesture → baseline intent mapping ────────────────────────────────────────
_GESTURE_INTENT: Dict[str, str] = {
    "PINCH":     Intent.CLICK,
    "SCROLL":    Intent.SCROLL,
    "MOVE":      Intent.MOVE_CURSOR,
    "FIST":      Intent.PAUSE,
    "OPEN_PALM": Intent.COMMAND,
    "SWIPE_LEFT":  Intent.NAVIGATE,
    "SWIPE_RIGHT": Intent.NAVIGATE,
    "SWIPE_UP":    Intent.NAVIGATE,
    "SWIPE_DOWN":  Intent.NAVIGATE,
    "CIRCLE":    Intent.COMMAND,
}


class IntentEngine:
    """
    Maps a gesture event + velocity into a structured intent dict.

    Output of :meth:`infer`::

        {
            "intent": str,  # e.g. "CLICK", "SCROLL", "NAVIGATE"
            "mode":   str,  # "NORMAL" | "PRECISION" | "FAST"
        }
    """

    # Velocity thresholds (normalised units / second)
    PRECISION_THRESHOLD: float = 0.30   # below → PRECISION
    FAST_THRESHOLD:      float = 1.50   # above → FAST

    def __init__(self, config: Optional[Dict] = None) -> None:
        cfg = config or {}
        self.precision_threshold = cfg.get("velocity_precision_threshold", self.PRECISION_THRESHOLD)
        self.fast_threshold      = cfg.get("velocity_fast_threshold",      self.FAST_THRESHOLD)
        self.mode: str = Mode.NORMAL

    # ── Public API ────────────────────────────────────────────────────────────

    def infer(self, gesture_data: Dict) -> Dict:
        """
        Infer intent and operating mode from a gesture event.

        :param gesture_data: Output from :class:`GestureDetector`.
        :returns: ``{"intent": str, "mode": str}``
        """
        gesture  = gesture_data.get("gesture", "NONE")
        velocity = gesture_data.get("velocity", 0.0)

        intent = _GESTURE_INTENT.get(gesture, Intent.NONE)
        self.mode = self.detect_mode(velocity)

        # Refine intent with mode context
        if intent == Intent.MOVE_CURSOR and self.mode == Mode.FAST:
            intent = Intent.NAVIGATE   # fast move → navigation intent

        return {"intent": intent, "mode": self.mode}

    def calculate_velocity(self, history: List) -> float:
        """
        Compute average velocity from a list of (x, y, timestamp) tuples.

        Convenience method when the caller holds their own history buffer.
        """
        if len(history) < 2:
            return 0.0

        total_dist = 0.0
        total_time = 0.0
        for i in range(1, len(history)):
            x0, y0, t0 = history[i - 1]
            x1, y1, t1 = history[i]
            import math
            total_dist += math.hypot(x1 - x0, y1 - y0)
            total_time += max(t1 - t0, 1e-6)

        return total_dist / total_time if total_time > 0 else 0.0

    def detect_mode(self, velocity: float) -> str:
        """
        Map a scalar velocity to an operating mode.

        :returns: ``"PRECISION"`` | ``"NORMAL"`` | ``"FAST"``
        """
        if velocity < self.precision_threshold:
            return Mode.PRECISION
        if velocity > self.fast_threshold:
            return Mode.FAST
        return Mode.NORMAL
