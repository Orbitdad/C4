"""
jarvis/vision/gesture/interpreter.py

Interprets continuous gesture telemetry, managing smoothing, states, and confidence.
"""

import math
import time
from collections import deque
from typing import Dict, List, Any, Optional

class GestureState:
    IDLE = "IDLE"
    START = "START"
    ACTIVE = "ACTIVE"
    END = "END"

class Mode:
    SINGLE_HAND = "SINGLE_HAND_MODE"
    DUAL_HAND = "DUAL_HAND_MODE"

class GestureInterpreter:
    def __init__(self, alpha: float = 0.3):
        self.alpha = alpha
        
        # State machine tracking
        self.state = GestureState.IDLE
        self.current_gesture = "NONE"
        self.gesture_frames = 0
        self.THRESHOLD_FRAMES = 3
        
        # EMA smoothed values
        self.smoothed_z = None
        self.smoothed_roll = None
        self.smoothed_dist = None
        
        # Mode tracking
        self.mode = Mode.SINGLE_HAND
        self.dual_hand_frames = 0
        
        # Previous values for delta calculation
        self.prev_z = None
        self.prev_roll = None
        self.prev_dist = None
        
        self.last_update_time = time.time()
        
    def _ema(self, current: float, previous: Optional[float]) -> float:
        if previous is None:
            return current
        return self.alpha * current + (1 - self.alpha) * previous
        
    def _calculate_confidence(self, delta: float, frames: int) -> float:
        """Calculate confidence based on delta magnitude and consistency over frames."""
        conf = 0.5 + min(0.4, (frames / 10.0)) # Build confidence over 10 frames
        # Add slight boost for significant deliberate movement
        if abs(delta) > 0.01:
            conf += 0.1
        return min(1.0, conf)

    def process(self, classification: str, hands_list: List[Any]) -> Dict:
        """
        Process the classifier output and raw hands list.
        Returns a dict with state, output, and debug metrics.
        """
        result = {
            "type": "NONE",
            "action": "NONE",
            "state": self.state,
            "mode": self.mode,
            "confidence": 0.0,
            "delta": 0.0,
            "metrics": {
                "z": 0.0,
                "roll": 0.0,
                "dist": 0.0
            }
        }
        
        if not hands_list:
            # Loss of tracking tracking
            self.state = GestureState.IDLE
            self.current_gesture = "NONE"
            self.gesture_frames = 0
            self.prev_z = None
            self.prev_roll = None
            self.prev_dist = None
            
            # Decay dual hand mode
            self.dual_hand_frames = max(0, self.dual_hand_frames - 1)
            if self.dual_hand_frames == 0:
                self.mode = Mode.SINGLE_HAND
                
            return result

        # Mode detection
        if len(hands_list) >= 2:
            self.dual_hand_frames += 1
            if self.dual_hand_frames >= 5:
                self.mode = Mode.DUAL_HAND
        else:
            self.dual_hand_frames = 0
            self.mode = Mode.SINGLE_HAND
            
        # Parse inputs
        hand1 = hands_list[0]
        
        # Dual-hand logic
        if self.mode == Mode.DUAL_HAND and len(hands_list) >= 2:
            hand2 = hands_list[1]
            # Hand 1 and Hand 2 index tips
            tip1 = hand1[8]
            tip2 = hand2[8]
            raw_dist = math.hypot(tip1.x - tip2.x, tip1.y - tip2.y)
            
            self.smoothed_dist = self._ema(raw_dist, self.prev_dist)
            
            if self.prev_dist is not None:
                delta = self.smoothed_dist - self.prev_dist
            else:
                delta = 0.0
                
            self.prev_dist = self.smoothed_dist
            result["metrics"]["dist"] = self.smoothed_dist
            
            # Ensure continuity structure
            if "ZOOM" in classification or abs(delta) > 0.005:
                self.gesture_frames += 1
                if self.gesture_frames >= self.THRESHOLD_FRAMES:
                    self.state = GestureState.ACTIVE
                    conf = self._calculate_confidence(delta, self.gesture_frames)
                    if conf >= 0.6:
                        result["type"] = "TRANSFORM"
                        result["action"] = "ZOOM"
                        result["delta"] = delta * 5.0 # Amplify slightly for dual hand (scale)
                        result["confidence"] = conf
                elif self.gesture_frames == 1:
                    self.state = GestureState.START
            else:
                self.gesture_frames = 0
                self.state = GestureState.IDLE
                
            result["state"] = self.state
            return result
            
        # Single-hand logic
        # 1. Wrist roll (Rotation)
        index_mcp = hand1[5]
        pinky_mcp = hand1[17]
        raw_roll = math.atan2(index_mcp.y - pinky_mcp.y, index_mcp.x - pinky_mcp.x)
        self.smoothed_roll = self._ema(raw_roll, self.prev_roll)
        result["metrics"]["roll"] = self.smoothed_roll
        
        delta_roll = 0.0
        if self.prev_roll is not None:
            d_roll = self.smoothed_roll - self.prev_roll
            # Handle wraparound safely
            if d_roll > math.pi:
                d_roll -= 2 * math.pi
            elif d_roll < -math.pi:
                d_roll += 2 * math.pi
            delta_roll = d_roll
        self.prev_roll = self.smoothed_roll

        # 2. Z-depth (Zoom)
        tip = hand1[8]
        raw_z = tip.z
        self.smoothed_z = self._ema(raw_z, self.prev_z)
        result["metrics"]["z"] = self.smoothed_z
        
        delta_z = 0.0
        if self.prev_z is not None:
            delta_z = self.smoothed_z - self.prev_z
        self.prev_z = self.smoothed_z
        
        # Lifecycle Logic
        is_zoom = "ZOOM" in classification
        is_rotate = "ROTATE" in classification
        is_continuous_gesture = is_zoom or is_rotate
        
        if is_continuous_gesture:
            if classification == self.current_gesture or self.current_gesture == "NONE":
                self.current_gesture = classification
                self.gesture_frames += 1
                
                if self.gesture_frames >= self.THRESHOLD_FRAMES:
                    self.state = GestureState.ACTIVE
                    
                    if is_rotate:
                        conf = self._calculate_confidence(delta_roll, self.gesture_frames)
                        if conf >= 0.6:
                            result["type"] = "TRANSFORM"
                            result["action"] = "ROTATE"
                            result["delta"] = delta_roll
                            result["confidence"] = conf
                    elif is_zoom:
                        conf = self._calculate_confidence(delta_z, self.gesture_frames)
                        if conf >= 0.6:
                            result["type"] = "TRANSFORM"
                            result["action"] = "ZOOM"
                            result["delta"] = delta_z
                            result["confidence"] = conf
                elif self.gesture_frames == 1:
                    self.state = GestureState.START
            else:
                 # Transitioned to new continuous gesture
                 self.state = GestureState.END
                 self.current_gesture = classification
                 self.gesture_frames = 1
        else:
            # Discrete or unhandled gesture
            if self.state in [GestureState.START, GestureState.ACTIVE]:
                self.state = GestureState.END
            else:
                self.state = GestureState.IDLE
            self.gesture_frames = 0
            self.current_gesture = "NONE"
            
            result["type"] = "DISCRETE"
            result["action"] = classification
            result["confidence"] = 0.9 # Default for discrete heuristcs (since classifier handles it)
            
        result["state"] = self.state
        return result
