"""
jarvis/vision/gesture/__init__.py

Entry point for the Gesture Interaction System.
Integrates the intent-based pipeline: Gesture -> Intent -> Context Router -> Action.
"""

from .detector import HandDetector
from .tracker import HandTracker
from .classifier import GestureClassifier
from .intent_engine import IntentEngine
from .context_router import ContextRouter
from .action_executor import ActionExecutor
from .gesture_config import DEFAULTS
from jarvis.core.event_bus import bus, SystemEvent
import time

class GestureController:
    """
    Main controller that coordinates the modular gesture pipeline.
    """
    def __init__(self):
        self.classifier = GestureClassifier()
        self.intent_engine = IntentEngine()
        self.router = ContextRouter()
        
        # We need a tracker per hand
        self.trackers = {
            0: HandTracker(window_size=DEFAULTS["smoothing_window"]),
            1: HandTracker(window_size=DEFAULTS["smoothing_window"])
        }
        
        # The executor is instantiated here to ensure it binds to the bus
        self.executor = ActionExecutor(
             smoothing=DEFAULTS["cursor_smoothing"], 
             frame_margin=DEFAULTS["frame_margin"]
        )
        
        self.enabled = True
        self.mode = "UI_MODE"
        self.dual_hand_frames = 0
        self.single_hand_frames = 0

    def process_all_hands(self, hands_list, pose_landmarks=None):
        """
        Process all detected hands through the intent pipeline.
        Returns debug dict for the HUI overlay.
        """
        if not self.enabled:
            return None

        if not hands_list:
            self._update_mode(0)
            # Send loss of tracking to intent engine to force DROP if dragging
            intent = self.intent_engine.process("NONE", (0,0,0), hands_list, self.mode)
            self.router.route_intent(intent, self.mode)
            return None

        self._update_mode(len(hands_list))

        primary_hand = hands_list[0]
        
        # 1. Classify raw gesture
        raw_gesture = self.classifier.classify(primary_hand)
        
        # 2. Smooth position tracking
        idx_tip = HandDetector.get_index_tip(primary_hand)
        smooth_pos = self.trackers[0].smooth(idx_tip)
        
        # Support dual hand tracking smoothing if needed
        if len(hands_list) >= 2:
            idx2 = HandDetector.get_index_tip(hands_list[1])
            self.trackers[1].smooth(idx2)

        # 3. Map to Intent (handles single and dual-hand overrides internally)
        intent = self.intent_engine.process(raw_gesture, smooth_pos, hands_list, self.mode)
        
        # 4. Route Intent to Action
        self.router.route_intent(intent, self.mode)

        # Return standard debug dictionary for HUI
        return {
            "gesture": intent["intent"], # We surface the INTENT to the UI
            "pos": smooth_pos,
            "mode": self.mode,
            "raw": raw_gesture
        }

    def _update_mode(self, hand_count: int):
        """Implement hysteresis for mode switching."""
        if hand_count >= 2:
            self.dual_hand_frames += 1
            self.single_hand_frames = 0
            if self.dual_hand_frames >= DEFAULTS["dual_hand_entry_frames"] and self.mode != "INTERACTION_3D":
                self.mode = "INTERACTION_3D"
                bus.publish(SystemEvent("gesture.mode_change", {"mode": self.mode}))
        elif hand_count == 1:
            self.single_hand_frames += 1
            self.dual_hand_frames = 0
            if self.single_hand_frames >= DEFAULTS["dual_hand_exit_frames"] and self.mode != "UI_MODE":
                self.mode = "UI_MODE"
                bus.publish(SystemEvent("gesture.mode_change", {"mode": self.mode}))
        else:
             # If 0 hands, we keep the current mode so state doesn't wipe randomly on a dropped frame
             pass

__all__ = [
    'GestureController',
    'HandDetector',
    'HandTracker',
    'GestureClassifier',
    'IntentEngine',
    'ContextRouter',
    'ActionExecutor'
]
