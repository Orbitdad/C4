import cv2
import mediapipe as mp
import numpy as np
import time

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Create a dummy image of a "hand" or noise that might trigger palm detection
# Just randomize, or load an actual image if possible. We can just run from the camera!
cap = cv2.VideoCapture(0)
print("Starting camera... please put your hand in view to test crash.")

start = time.time()
frame_count = 0
found_hand = False

while time.time() - start < 10:  # run for 10 seconds
    ret, frame = cap.read()
    if not ret: continue
    
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)
    
    if results.multi_hand_landmarks:
        found_hand = True
        print("DETECTED HAND! No crash yet.")
        break
        
    time.sleep(0.03)

cap.release()

if found_hand:
    print("SUCCESS: MediaPipe did not crash when processing the landmark model.")
else:
    print("Did not detect a hand. Cannot confirm fix.")
