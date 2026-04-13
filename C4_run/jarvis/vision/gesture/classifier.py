"""
jarvis/vision/gesture/classifier.py

Heuristic-based gesture classification from hand landmarks and velocity history.
"""

import math
from typing import List, Any, Optional, Tuple, Deque
from collections import deque

class GestureClassifier:
    def __init__(self, pinch_threshold: float = 0.05, swipe_threshold: float = 0.15):
        self.pinch_threshold = pinch_threshold
        self.swipe_threshold = swipe_threshold
        # History for dynamic gestures (X, Y, Z, Roll)
        self.pos_history: Deque[Tuple[float, float, float, float]] = deque(maxlen=20)

    def classify(self, landmarks: List[Any]) -> str:
        """
        Determines the current hand gesture from landmarks.
        Returns: OPEN_PALM, FIST, INDEX_POINT, PINCH, SWIPE_LEFT, SWIPE_RIGHT, CIRCLE, UNKNOWN
        """
        if not landmarks:
            return "UNKNOWN"

        # 1. Extract basic state
        fingers_up = self._get_fingers_up(landmarks)
        pinch_dist = self._get_pinch_distance(landmarks)
        
        # 2. Update movement history (Include Z and Roll)
        index_tip = landmarks[8]
        # Calculate wrist roll angle (rads) using Index MCP (5) and Pinky MCP (17)
        index_mcp = landmarks[5]
        pinky_mcp = landmarks[17]
        roll = math.atan2(index_mcp.y - pinky_mcp.y, index_mcp.x - pinky_mcp.x)
        self.pos_history.append((index_tip.x, index_tip.y, index_tip.z, roll))

        # 3. Decision Logic
        
        # PINCH (Index + Thumb)
        if pinch_dist < self.pinch_threshold:
            return "PINCH"

        # 3D Depth (ZOOM) and Rotation detection (Continuous Signals)
        if len(self.pos_history) == self.pos_history.maxlen:
            zoom_gesture = self._detect_zoom()
            if zoom_gesture != "NONE":
                return zoom_gesture
                
            rot_gesture = self._detect_rotation()
            if rot_gesture != "NONE":
                return rot_gesture

        # OPEN_PALM (All fingers up)
        if all(fingers_up):
            return "OPEN_PALM"

        # FIST (All fingers down)
        if not any(fingers_up):
            return "FIST"

        # INDEX_POINT (Only index up)
        if fingers_up[0] and not any(fingers_up[1:]):
            return "INDEX_POINT"

        # SWIPE detection
        if len(self.pos_history) == self.pos_history.maxlen:
            swipe = self._detect_swipe()
            if swipe != "NONE":
                return swipe
                
        # CIRCLE detection
        if self._detect_circle():
            return "CIRCLE"

        return "UNKNOWN"

    def classify_pose(self, lm: List[Any]) -> str:
        """
        Classify full body pose heuristics.
        Returns: ARMS_SPREAD, ARMS_CROSSED, or NONE
        """
        if not lm or len(lm) < 33:
            return "NONE"
            
        # Landmarks: 11 (Left Shoulder), 12 (Right Shoulder)
        # 15 (Left Wrist), 16 (Right Wrist)
        ls_x, ls_y = lm[11].x, lm[11].y
        rs_x, rs_y = lm[12].x, lm[12].y
        lw_x, lw_y = lm[15].x, lm[15].y
        rw_x, rw_y = lm[16].x, lm[16].y
        
        shoulder_dist = abs(ls_x - rs_x)
        if shoulder_dist < 0.05:
            return "NONE" # sideways or error
            
        wrist_dist_x = abs(lw_x - rw_x)
        
        # 1. ARMS_SPREAD
        # If wrists are much farther apart than shoulders
        if wrist_dist_x > shoulder_dist * 1.8:
            return "ARMS_SPREAD"
            
        # 2. ARMS_CROSSED
        # Determine if horizontally crossed:
        # If looking at a mirror, right shoulder (12) is left side (smaller x), left shoulder (11) is right side (larger x).
        # Wrists crossed means left wrist (15) is at smaller x than right wrist (16).
        is_mirrored = ls_x > rs_x
        if is_mirrored:
            is_crossed = lw_x < rw_x
        else:
            is_crossed = lw_x > rw_x
            
        # Additionally, must be somewhat elevated towards the chest
        if is_crossed and lw_y < ls_y + 0.3 and rw_y < rs_y + 0.3:
            return "ARMS_CROSSED"
            
        return "NONE"

    def _get_fingers_up(self, lm: List[Any]) -> List[bool]:
        """
        Returns list of booleans for [index, middle, ring, pinky]
        True if finger is 'up' (tip.y < pip.y)
        """
        # Landmark indices: Tip: 8, 12, 16, 20 | PIP: 6, 10, 14, 18
        tips = [8, 12, 16, 20]
        pips = [6, 10, 14, 18]
        return [lm[tip].y < lm[pip].y for tip, pip in zip(tips, pips)]

    def _get_pinch_distance(self, lm: List[Any]) -> float:
        """Distance between thumb tip (4) and index tip (8)."""
        return math.hypot(lm[8].x - lm[4].x, lm[8].y - lm[4].y)

    def _detect_swipe(self) -> str:
        first = self.pos_history[0]
        last = self.pos_history[-1]
        dx = last[0] - first[0]
        dy = last[1] - first[1]
        
        if abs(dx) > self.swipe_threshold and abs(dx) > abs(dy):
            return "SWIPE_RIGHT" if dx > 0 else "SWIPE_LEFT"
        return "NONE"

    def _detect_circle(self) -> bool:
        if len(self.pos_history) < 15:
            return False
            
        # Very simple circularity check: check if center of points is close to average 
        # and points are somewhat equidistant from center.
        # But for mid-air, a simpler "direction change count" is often more robust.
        # Let's check for X/Y direction shifts.
        x_changes = 0
        y_changes = 0
        
        last_dx = self.pos_history[1][0] - self.pos_history[0][0]
        last_dy = self.pos_history[1][1] - self.pos_history[0][1]
        
        for i in range(2, len(self.pos_history)):
            dx = self.pos_history[i][0] - self.pos_history[i-1][0]
            dy = self.pos_history[i][1] - self.pos_history[i-1][1]
            
            if (dx > 0) != (last_dx > 0) and abs(dx) > 0.005: x_changes += 1
            if (dy > 0) != (last_dy > 0) and abs(dy) > 0.005: y_changes += 1
            
            last_dx = dx if abs(dx) > 0.005 else last_dx
            last_dy = dy if abs(dy) > 0.005 else last_dy
            
        # A circle has 2 direction flips in each axis (roughly)
        return x_changes >= 2 and y_changes >= 2

    def _detect_zoom(self) -> str:
        """Detect Z-axis movement for zooming."""
        first = self.pos_history[0]
        last = self.pos_history[-1]
        dz = last[2] - first[2]
        
        # In MediaPipe, smaller Z (more negative) means closer to camera
        z_threshold = 0.05
        if abs(dz) > z_threshold:
            return "ZOOM_IN" if dz < 0 else "ZOOM_OUT"
        return "NONE"

    def _detect_rotation(self) -> str:
        """Detect wrist roll for 3D rotation."""
        first_roll = self.pos_history[0][3]
        last_roll = self.pos_history[-1][3]
        d_roll = last_roll - first_roll
        
        # Handle wraparound safely for atan2 output (-pi to pi)
        if d_roll > math.pi:
            d_roll -= 2 * math.pi
        elif d_roll < -math.pi:
            d_roll += 2 * math.pi
            
        roll_threshold = 0.4 # roughly 23 degrees
        if abs(d_roll) > roll_threshold:
            return "ROTATE_RIGHT" if d_roll > 0 else "ROTATE_LEFT"
        return "NONE"
