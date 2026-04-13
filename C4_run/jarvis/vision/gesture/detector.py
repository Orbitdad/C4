"""
jarvis/vision/gesture/detector.py

Modular MediaPipe Hand Detector.
Responsible for raw landmark extraction and basic point tracking.
"""

import cv2
import mediapipe as mp
from typing import Optional, List, Tuple, Any

class HandDetector:
    def __init__(self, 
                 static_mode=False, 
                 max_hands=1, 
                 min_detection_confidence=0.7, 
                 min_tracking_confidence=0.5):
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=static_mode,
            max_num_hands=max_hands,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence
        )
        self.mp_draw = mp.solutions.drawing_utils

    def process_frame(self, frame: cv2.Mat) -> Optional[List[Any]]:
        """
        Process a BGR frame and return 21 landmarks for the primary hand.
        """
        img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.hands.process(img_rgb)
        
        if results.multi_hand_landmarks:
            # We focus on the first detected hand for interaction
            return results.multi_hand_landmarks[0].landmark
        return None

    def get_index_tip(self, landmarks: List[Any]) -> Tuple[float, float, float]:
        """Extract index finger tip (landmark id 8)."""
        if not landmarks or len(landmarks) < 21:
            return (0.0, 0.0, 0.0)
        tip = landmarks[8]
        return (tip.x, tip.y, tip.z)

    def draw_landmarks(self, frame: cv2.Mat, landmarks: List[Any]):
        """Helper to draw landmarks on frame for debugging/UI."""
        # Convert landmarks back to MediaPipe's internal format for drawing
        # This is a bit tricky if landmarks is just a list of objects with .x, .y
        # But if it's the raw result from self.hands.process, it works.
        # Here we assume landmarks is what process_frame returns.
        pass # VisionManager usually handles drawing, but good to have.
