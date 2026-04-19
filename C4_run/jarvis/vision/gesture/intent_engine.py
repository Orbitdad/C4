"""
jarvis/vision/gesture/intent_engine.py

Maps raw gesture classifications into universal semantic intents.
Manages internal state (like hold durations for drag).
"""

import time
from typing import Dict, Any, List
from .gesture_config import DEFAULTS
from .detector import HandDetector
import math

class IntentEngine:
    def __init__(self):
        self.hold_threshold = DEFAULTS["hold_duration"]
        self.confidence_min = DEFAULTS["confidence_min"]
        self.debounce_frames = DEFAULTS["debounce_frames"]
        
        # State tracking
        self.current_gesture = "NONE"
        self.gesture_frames = 0
        
        self.current_intent = "IDLE"
        self.pinch_start_time = 0.0
        self.is_dragging = False

    def process(self, gesture: str, pos: tuple, hands_list: List[Any], mode: str) -> dict:
        """
        Maps the raw gesture to an Intent and formats the required output payload.
        Handles both 1-hand semantic intent mapping and 2-hand dual interactions.
        """
        # 1. Debouncing 
        if gesture == self.current_gesture:
            self.gesture_frames += 1
        else:
            self.current_gesture = gesture
            self.gesture_frames = 1
            
        confidence = min(1.0, self.gesture_frames / max(1, self.debounce_frames))
        
        # Fallback if unconfident or lost tracking
        if confidence < self.confidence_min or gesture == "NONE":
             # Need to cleanly end a drag if we lose tracking
             if self.is_dragging:
                 self.is_dragging = False
                 self.pinch_start_time = 0.0
                 return self._build_intent("DROP", pos, confidence)
             return self._build_intent("IDLE", pos, confidence)

        # 2. Dual-hand intercepts (3D Mode features)
        if mode == "INTERACTION_3D" and len(hands_list) >= 2:
            return self._process_dual_hand(gesture, hands_list, pos, confidence)

        # 3. Universal Single-Hand Intent Mapping
        return self._process_single_hand(gesture, pos, confidence)

    def _process_single_hand(self, gesture: str, pos: tuple, confidence: float) -> dict:
        """Map a single hand gesture to a universal intent."""
        
        # PINCH FSM -> SELECT, DRAG, DROP
        if gesture == "PINCH":
            if not self.is_dragging:
                if self.pinch_start_time == 0.0:
                    self.pinch_start_time = time.time()
                    return self._build_intent("SELECT", pos, confidence) # First frame of pinch = click
                elif time.time() - self.pinch_start_time > self.hold_threshold:
                    self.is_dragging = True
                    return self._build_intent("DRAG", pos, confidence)
            else:
                return self._build_intent("DRAG", pos, confidence)
        else:
            # Not pinching
            if self.is_dragging:
                # We were dragging, now we aren't -> DROP
                self.is_dragging = False
                self.pinch_start_time = 0.0
                return self._build_intent("DROP", pos, confidence)
                
            self.pinch_start_time = 0.0

            # 1:1 Static Mappings
            if gesture == "POINT":
                return self._build_intent("TARGET", pos, confidence)
            elif gesture == "OPEN_PALM":
                return self._build_intent("IDLE", pos, confidence)
            elif gesture == "FIST":
                return self._build_intent("CANCEL", pos, confidence)
                
            # Dynamic Mappings (One-shots)
            elif "SWIPE" in gesture:
                # Note: Nav directions could be added to payload metadata if needed
                # For now, UI router handles it, we just send NAVIGATE
                nav_dir = "left" if "LEFT" in gesture else "right"
                return self._build_intent("NAVIGATE", pos, confidence, {"direction": nav_dir})
            elif gesture == "PUSH":
                return self._build_intent("CONFIRM", pos, confidence)
            elif gesture == "PULL":
                return self._build_intent("BACK", pos, confidence)

        return self._build_intent("IDLE", pos, confidence)

    def _process_dual_hand(self, primary_gesture: str, hands_list: List[Any], pos: tuple, confidence: float) -> dict:
        """Evaluate dual hand geometries for SCALE/ROTATE."""
        hand1 = hands_list[0]
        hand2 = hands_list[1]
        
        # Helper to check if BOTH hands are pinching
        def is_pinching(lm):
            idx = HandDetector.get_index_tip(lm)
            thb = HandDetector.get_thumb_tip(lm)
            dist = math.hypot(idx[0] - thb[0], idx[1] - thb[1])
            return dist < DEFAULTS["pinch_threshold"]

        h1_pinch = is_pinching(hand1)
        h2_pinch = is_pinching(hand2)

        if h1_pinch and h2_pinch:
             # Both hands pinch -> SCALE
             # Calculate distance between the two pinches
             h1_idx = HandDetector.get_index_tip(hand1)
             h2_idx = HandDetector.get_index_tip(hand2)
             dist = math.hypot(h1_idx[0] - h2_idx[0], h1_idx[1] - h2_idx[1])
             return self._build_intent("SCALE", pos, confidence, {"distance": dist})
             
        elif primary_gesture == "POINT":
             # We can infer two-hand rotation if pointing and moving
             # (Simplification: using dual wrists for rotation angle)
             w1 = HandDetector.get_wrist(hand1)
             w2 = HandDetector.get_wrist(hand2)
             angle = math.atan2(w2[1] - w1[1], w2[0] - w1[0])
             return self._build_intent("ROTATE", pos, confidence, {"angle": angle})

        # If no dual-hand specific interaction, fallback to primary hand intent
        return self._process_single_hand(primary_gesture, pos, confidence)

    def _build_intent(self, intent_name: str, pos: tuple, confidence: float, meta: dict = None) -> dict:
        self.current_intent = intent_name
        data = {
            "intent": intent_name,
            "x": pos[0],
            "y": pos[1],
            "z": pos[2],
            "confidence": confidence
        }
        if meta:
            data.update(meta)
        return data
