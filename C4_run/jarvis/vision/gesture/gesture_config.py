"""
jarvis/vision/gesture/gesture_config.py

Central configuration defaults for the universal intent-based gesture pipeline.
"""

from __future__ import annotations

DEFAULTS = {
    # ── Detection thresholds ──────────────────────────────────────────────────
    "pinch_threshold": 0.05,       # normalised distance for thumb-index contact
    "push_pull_threshold": 0.04,   # Z-axis delta for PUSH / PULL
    "swipe_threshold": 0.12,       # X-axis delta for SWIPE gestures
    "swipe_ratio_min": 1.5,        # |dx| must be > ratio * |dy| to count as swipe

    # ── Temporal thresholds ───────────────────────────────────────────────────
    "hold_duration": 0.4,          # seconds pinch must be held to promote SELECT→DRAG
    "debounce_frames": 3,          # frames a gesture must persist before accepted
    "dual_hand_entry_frames": 5,   # frames of 2 hands to switch to 3D mode
    "dual_hand_exit_frames": 10,   # frames of 1 hand to revert to UI mode

    # ── Smoothing & quality ───────────────────────────────────────────────────
    "smoothing_window": 5,         # moving-average window for position tracking
    "confidence_min": 0.6,         # minimum confidence to act on a gesture
    "pos_history_len": 15,         # frames of position history for motion detection

    # ── Executor ──────────────────────────────────────────────────────────────
    "click_cooldown": 0.4,         # seconds between consecutive clicks
    "frame_margin": 0.20,          # screen-edge margin for cursor mapping
    "cursor_smoothing": 5.0,       # exponential smoothing factor for cursor movement
}
