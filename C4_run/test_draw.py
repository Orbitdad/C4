import cv2
import numpy as np
import mediapipe as mp

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

try:
    from mediapipe.solutions.drawing_styles import get_default_hand_landmarks_style, get_default_hand_connections_style
    lm_spec = get_default_hand_landmarks_style()
    conn_spec = get_default_hand_connections_style()
except ImportError:
    lm_spec = mp_draw.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=4)
    conn_spec = mp_draw.DrawingSpec(color=(0, 210, 255), thickness=2)

img = np.zeros((480, 640, 3), dtype=np.uint8)
img = cv2.flip(img, 1)

annotated = img.copy()

print("Processing...")
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)
results = hands.process(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

# Fake a landmark to test drawing
from mediapipe.framework.formats import landmark_pb2
lmlist = landmark_pb2.NormalizedLandmarkList()
for i in range(21):
    lm = lmlist.landmark.add()
    lm.x, lm.y, lm.z = 0.5, 0.5, 0.0

print("Drawing...")
try:
    mp_draw.draw_landmarks(annotated, lmlist, mp_hands.HAND_CONNECTIONS, lm_spec, conn_spec)
    print("Draw complete. Contiguous:", annotated.flags.c_contiguous)
except Exception as e:
    import traceback
    traceback.print_exc()
    print("ERROR:", e)
