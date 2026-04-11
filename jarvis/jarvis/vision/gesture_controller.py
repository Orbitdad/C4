# gesture_controller.py
# ──────────────────────────────────────────────────────────────────────────────
# Backward-compatibility shim.
#
# The implementation has been refactored into the modular package at:
#   jarvis/vision/gesture/
#
# All existing imports of the form:
#   from jarvis.vision.gesture_controller import GestureController
# continue to work without any changes.
# ──────────────────────────────────────────────────────────────────────────────

from jarvis.vision.gesture import GestureController  # noqa: F401

__all__ = ["GestureController"]
