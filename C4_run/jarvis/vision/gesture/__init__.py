"""
jarvis/vision/gesture/__init__.py

Entry point for the Gesture Interaction System.
Integrates detector, tracker, classifier, state manager, mapper, and emitter.
"""

from .detector import HandDetector
from .tracker import HandTracker
from .classifier import GestureClassifier
from .state_manager import StateManager
from .mapper import GestureMapper
from .emitter import GestureEmitter
from .interpreter import GestureInterpreter

class GestureController:
    """
    Main controller that coordinates the modular gesture pipeline.
    """
    def __init__(self):
        self.detector = HandDetector()
        self.tracker = HandTracker(window_size=5)
        self.classifier = GestureClassifier()
        self.fsm = StateManager()
        self.interpreter = GestureInterpreter()
        self.mapper = GestureMapper()
        self.emitter = GestureEmitter()
        
        self.enabled = True

    def process_all_hands(self, hands_list, pose_landmarks=None):
        """
        Process all detected hands and full body pose if available.
        """
        if not self.enabled:
            return

        # 1. First, check full-body macro gestures if pose is available.
        if pose_landmarks:
            pose_gesture = self.classifier.classify_pose(pose_landmarks)
            if pose_gesture != "NONE":
                # If we detect a macro pose, we inject it directly
                # We skip hand processing so it doesn't conflict
                raw_gesture = pose_gesture
                return self.process_landmarks(None, raw_gesture)

        if not hands_list:
            return

        # Always feed the first hand to the classifier for backwards parity
        raw_gesture = self.classifier.classify(hands_list[0])
        
        # Intercept with our new Interpreter
        interp_res = self.interpreter.process(raw_gesture, hands_list)
        
        # We can fire TRANSFORM actions directly if confidence is high enough
        if interp_res["type"] == "TRANSFORM":
            # For continuous transforms, we want to bypass the discrete StateManager
            # But we still map it and fuse it
            action_dict = {
                "action": interp_res["action"],
                "type": "TRANSFORM",
                "delta": interp_res["delta"],
                "confidence": interp_res["confidence"]
            }
            # For HUI visualization and state updates via Emmitter
            self.emitter.emit_detection(interp_res["action"], interp_res["state"], (0,0,0))
            # Just push directly to bus for the fusion_engine to resolve via fallback
            # Our modified emitter can take a dict or string, here we fallback to emitting dict via a special event or rely on action executor directly...
            # The ActionExecutor needs an action dict. Let's rely on standard event bus `gesture.action`
            import jarvis.core.event_bus as event_bus
            event_bus.bus.publish(event_bus.SystemEvent(name="gesture.action", data=action_dict))
            
            return interp_res
            
        # If not TRANSFORM, use original process_landmarks for single hand discrete gestures
        return self.process_landmarks(hands_list[0], raw_gesture)

    def process_landmarks(self, landmarks, raw_gesture=None):
        """
        Process raw landmarks from vision manager (Primary hand).
        """
        if not self.enabled:
            return
            
        if not landmarks and raw_gesture is None:
            return

        # 1. Pose Smoothing (Skip if no hand landmarks, just use center)
        smooth_pos = (0.5, 0.5, 0.5)
        if landmarks:
            raw_pos = self.detector.get_index_tip(landmarks)
            smooth_pos = self.tracker.smooth(raw_pos)

        # 2. Gesture Recognition
        if raw_gesture is None:
            gesture = self.classifier.classify(landmarks)
        else:
            gesture = raw_gesture

        # 3. State Management
        state = self.fsm.update(gesture)

        # 4. Action Mapping
        action = self.mapper.map_to_action(gesture, state)

        # 5. Event Emission
        self.emitter.emit_detection(gesture, state, smooth_pos)
        if action != "NONE":
            self.emitter.emit_action(action)
        
        return {
            "gesture": gesture,
            "state": state,
            "action": action,
            "pos": smooth_pos
        }

# For backward compatibility if needed, but we are refactoring.
__all__ = [
    'GestureController',
    'HandDetector',
    'HandTracker',
    'GestureClassifier',
    'StateManager',
    'GestureMapper',
    'GestureEmitter'
]
