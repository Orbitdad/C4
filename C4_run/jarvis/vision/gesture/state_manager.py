"""
jarvis/vision/gesture/state_manager.py

Finite State Machine for Gesture Interaction.
Handles persistence, debouncing, and state transitions.
"""

import time
from typing import Dict, Optional

class GestureState:
    IDLE = "IDLE"
    ACTIVE = "ACTIVE"
    DRAGGING = "DRAGGING"
    NAVIGATING = "NAVIGATING"

class StateManager:
    def __init__(self, debounce_frames: int = 3):
        self.current_state = GestureState.IDLE
        self.debounce_frames = debounce_frames
        
        self.last_gesture = "UNKNOWN"
        self.gesture_count = 0
        
        self.pinch_start_time = 0.0
        self.PINCH_HOLD_THRESHOLD = 0.5 # seconds to trigger DRAGGING

    def update(self, gesture: str) -> str:
        """
        Processes the new gesture and returns the current state.
        """
        # 1. Debouncing logic
        if gesture == self.last_gesture:
            self.gesture_count += 1
        else:
            self.last_gesture = gesture
            self.gesture_count = 1
            
        if self.gesture_count < self.debounce_frames:
            return self.current_state

        # 2. State transition logic
        if gesture == "OPEN_PALM":
            self.current_state = GestureState.ACTIVE
            self.pinch_start_time = 0.0
            
        elif gesture == "FIST":
            self.current_state = GestureState.IDLE
            self.pinch_start_time = 0.0

        elif self.current_state == GestureState.ACTIVE:
            if gesture == "PINCH":
                if self.pinch_start_time == 0.0:
                    self.pinch_start_time = time.time()
                elif time.time() - self.pinch_start_time > self.PINCH_HOLD_THRESHOLD:
                    self.current_state = GestureState.DRAGGING
                    
            elif "SWIPE" in gesture:
                self.current_state = GestureState.NAVIGATING
                
        elif self.current_state == GestureState.DRAGGING:
            if gesture != "PINCH":
                self.current_state = GestureState.ACTIVE
                self.pinch_start_time = 0.0

        elif self.current_state == GestureState.NAVIGATING:
            # Revert to active after navigation (usually one-shot in emitter/mapper)
            # But here we let it stay until gesture changes
            if "SWIPE" not in gesture:
                self.current_state = GestureState.ACTIVE

        return self.current_state
