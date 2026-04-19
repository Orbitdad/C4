"""
jarvis/vision/gesture/classifier.py

Heuristic-based gesture classifier.
Recognizes only the 9 approved foundational gestures.
"""

import math
from typing import List, Any, Tuple, Deque
from collections import deque
from .gesture_config import DEFAULTS
from .detector import HandDetector

class GestureClassifier:
    def __init__(self):
        self.pinch_threshold = DEFAULTS["pinch_threshold"]
        self.swipe_threshold = DEFAULTS["swipe_threshold"]
        self.swipe_ratio = DEFAULTS["swipe_ratio_min"]
        self.push_pull_threshold = DEFAULTS["push_pull_threshold"]
        
        # History for dynamic gestures (X, Y, Z)
        self.pos_history: Deque[Tuple[float, float, float]] = deque(maxlen=DEFAULTS["pos_history_len"])

    def classify(self, landmarks: List[Any]) -> str:
        """
        Determines the current hand gesture from a single hand's landmarks.
        Approved outputs: POINT, PINCH, OPEN_PALM, FIST, SWIPE_LEFT, SWIPE_RIGHT, PUSH, PULL, NONE
        """
        if not landmarks:
            return "NONE"

        # 1. Update movement history (wrist position is best for overall hand motion)
        wrist = HandDetector.get_wrist(landmarks)
        self.pos_history.append(wrist)

        # 2. Extract basic static state
        fingers_up = self._get_fingers_up(landmarks)
        pinch_dist = self._get_pinch_distance(landmarks)
        
        # 3. Decision Logic - Static Priorities

        # PINCH (Index + Thumb contact)
        if pinch_dist < self.pinch_threshold:
            return "PINCH"

        # OPEN_PALM (All fingers up)
        if all(fingers_up):
            return "OPEN_PALM"

        # FIST (All fingers down)
        if not any(fingers_up):
            return "FIST"

        # POINT (Only index up - previously called INDEX_POINT)
        if fingers_up[0] and not any(fingers_up[1:]):
            return "POINT"

        # 4. Decision Logic - Dynamic Priorities (only evaluate if hand is somewhat flat/open and history is full)
        # We don't want to trigger swipes if the user is pointing or pinching
        if len(self.pos_history) == self.pos_history.maxlen and (all(fingers_up) or not any(fingers_up)):
            
            # SWIPE detection (X axis)
            swipe = self._detect_swipe()
            if swipe != "NONE":
                return swipe
                
            # PUSH / PULL detection (Z axis)
            z_motion = self._detect_z_motion()
            if z_motion != "NONE":
                return z_motion

        return "NONE"

    def _get_fingers_up(self, lm: List[Any]) -> List[bool]:
        """
        Returns list of booleans for [index, middle, ring, pinky]
        True if finger is 'up' (extended/straight).
        Uses distance from the wrist so it is invariant to hand rotation.
        """
        wrist = lm[0]
        # Landmark indices: Tip: 8, 12, 16, 20 | PIP: 6, 10, 14, 18
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        
        results = []
        for tip_idx, pip_idx in zip(tips, pips):
            tip = lm[tip_idx]
            pip = lm[pip_idx]
            dist_tip = math.hypot(tip.x - wrist.x, tip.y - wrist.y)
            dist_pip = math.hypot(pip.x - wrist.x, pip.y - wrist.y)
            # If the tip is further from the wrist than the PIP joint, the finger is extended
            results.append(dist_tip > dist_pip)
            
        return results

    def _get_pinch_distance(self, lm: List[Any]) -> float:
        """Distance between thumb tip (4) and index tip (8)."""
        idx_tip = HandDetector.get_index_tip(lm)
        thb_tip = HandDetector.get_thumb_tip(lm)
        return math.hypot(idx_tip[0] - thb_tip[0], idx_tip[1] - thb_tip[1])

    def _detect_swipe(self) -> str:
        """Detect rapid horizontal hand movement."""
        first = self.pos_history[0]
        last = self.pos_history[-1]
        dx = last[0] - first[0]
        dy = last[1] - first[1]
        
        # Must exceed threshold AND be predominantly horizontal
        if abs(dx) > self.swipe_threshold and abs(dx) > abs(dy) * self.swipe_ratio:
            return "SWIPE_RIGHT" if dx > 0 else "SWIPE_LEFT"
        return "NONE"

    def _detect_z_motion(self) -> str:
        """Detect Z-axis movement (push/pull)."""
        first = self.pos_history[0]
        last = self.pos_history[-1]
        dz = last[2] - first[2]
        
        # In MediaPipe, smaller Z (more negative) means closer to camera
        # Hand moving away from camera = Z increases = PUSH
        # Hand moving toward camera = Z decreases = PULL
        if abs(dz) > self.push_pull_threshold:
            return "PUSH" if dz > 0 else "PULL"
        return "NONE"
