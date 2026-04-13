"""
jarvis/vision/gesture/gesture_config.py

Central configuration defaults for the gesture pipeline.
"""

from __future__ import annotations

DEFAULTS = {
    "pinch_threshold": 0.05,
    "smoothing": 5.0,
    "click_cooldown": 0.5,
    "frame_margin_x": 0.20,
    "frame_margin_y": 0.20,
    "adaptive": {
        "enabled": True,
        "profile_path": "data/gesture_profile.json",
    },
}

