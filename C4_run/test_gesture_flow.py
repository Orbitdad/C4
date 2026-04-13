import time
from typing import NamedTuple

class MockPoint(NamedTuple):
    x: float
    y: float
    z: float

def create_mock_landmarks(fingers_up: list[bool], pinch_dist: float = 0.1, x_offset: float = 0.5, y_offset: float = 0.5):
    """
    Creates 21 mock landmarks.
    fingers_up: [index, middle, ring, pinky]
    """
    landmarks = [MockPoint(0, 0, 0) for _ in range(21)]
    
    # Thumb (4) and Index (8)
    landmarks[4] = MockPoint(x_offset - pinch_dist, y_offset, 0)
    landmarks[8] = MockPoint(x_offset, y_offset, 0)
    
    # Index PIP (6)
    landmarks[6] = MockPoint(x_offset, y_offset + (0.1 if fingers_up[0] else -0.1), 0)
    
    # Middle (12) and PIP (10)
    landmarks[12] = MockPoint(x_offset + 0.1, y_offset, 0)
    landmarks[10] = MockPoint(x_offset + 0.1, y_offset + (0.1 if fingers_up[1] else -0.1), 0)
    
    # Ring (16) and PIP (14)
    landmarks[16] = MockPoint(x_offset + 0.2, y_offset, 0)
    landmarks[14] = MockPoint(x_offset + 0.2, y_offset + (0.1 if fingers_up[2] else -0.1), 0)
    
    # Pinky (20) and PIP (18)
    landmarks[20] = MockPoint(x_offset + 0.3, y_offset, 0)
    landmarks[18] = MockPoint(x_offset + 0.3, y_offset + (0.1 if fingers_up[3] else -0.1), 0)
    
    return landmarks

if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.abspath('c:/C4/jarvis'))
    
    # Mock EventBus
    from jarvis.core.event_bus import bus
    published_events = []
    
    def mock_publish(event):
        published_events.append(event)
        print(f"[EventBus Mock] Published: {event.name} | Data: {event.data}")
        
    bus.publish = mock_publish
    
    from jarvis.vision.gesture import GestureController
    
    controller = GestureController()
    
    print("--- Testing FIST (IDLE) ---")
    lm_fist = create_mock_landmarks([False, False, False, False], pinch_dist=0.2)
    for _ in range(5):
        controller.process_landmarks(lm_fist)
        
    print("\n--- Testing OPEN_PALM (ACTIVE) ---")
    lm_palm = create_mock_landmarks([True, True, True, True], pinch_dist=0.2)
    for _ in range(5):
        controller.process_landmarks(lm_palm)
        
    print("\n--- Testing PINCH_HOLD (DRAGGING) ---")
    lm_pinch = create_mock_landmarks([True, False, False, False], pinch_dist=0.02)
    # This should transition to DRAGGING after a delay
    for i in range(15): # Need to simulate > 0.5 seconds for DRAGGING
        print(f" Frame {i}:")
        controller.process_landmarks(lm_pinch)
        time.sleep(0.05)
        
    print("\n--- Testing INDEX_POINT (MOVE in ACTIVE) ---")
    # First back to active (needs to stop pinching)
    for _ in range(3):
        controller.process_landmarks(lm_palm)
        
    lm_point = create_mock_landmarks([True, False, False, False], pinch_dist=0.2, x_offset=0.6, y_offset=0.6)
    for _ in range(3):
        controller.process_landmarks(lm_point)
