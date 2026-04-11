"""
jarvis/vision/gesture/mapper.py

Maps raw gestures and states to high-level system actions.
"""

from .state_manager import GestureState

class GestureMapper:
    def map_to_action(self, gesture: str, state: str) -> str:
        """
        Maps (gesture, state) to one of: 
        CLICK, DRAG, SCROLL_UP, SCROLL_DOWN, PREVIOUS, NEXT, TRIGGER, NONE
        """
        
        if state == GestureState.IDLE:
            return "NONE"

        if state == GestureState.ACTIVE:
            if gesture == "PINCH":
                return "CLICK"
            if gesture == "SWIPE_LEFT":
                return "PREVIOUS"
            if gesture == "SWIPE_RIGHT":
                return "NEXT"
            if gesture == "CIRCLE":
                return "TRIGGER"
            if gesture == "INDEX_POINT":
                return "MOVE"
            
            # Continuous 3D Signals
            if gesture == "ZOOM_IN":
                return "ZOOM_IN"
            if gesture == "ZOOM_OUT":
                return "ZOOM_OUT"
            if gesture == "ROTATE_LEFT":
                return "ROTATE_LEFT"
            if gesture == "ROTATE_RIGHT":
                return "ROTATE_RIGHT"

        if state == GestureState.DRAGGING:
            return "DRAG"
            
        if state == GestureState.NAVIGATING:
            if gesture == "SWIPE_LEFT":
                return "PREVIOUS"
            if gesture == "SWIPE_RIGHT":
                return "NEXT"

        return "NONE"
