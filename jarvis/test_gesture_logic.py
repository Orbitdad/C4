from jarvis.vision.gesture_controller import GestureController

class MockLandmark:
    def __init__(self, x, y, z=0):
        self.x = x
        self.y = y
        self.z = z

def make_landmarks(n=21):
    return [MockLandmark(0.5, 0.5) for _ in range(n)]

def test_gestures():
    gc = GestureController()

    # ── 1. MOVE (one finger up) ───────────────────────────────────────────────
    print("Testing 1 FINGER UP -> MOVE CURSOR")
    lm = make_landmarks()
    lm[8].y = 0.2;  lm[6].y  = 0.3   # index up
    lm[12].y = 0.6; lm[10].y = 0.5   # middle down
    lm[16].y = 0.6; lm[14].y = 0.5   # ring down
    lm[20].y = 0.6; lm[18].y = 0.5   # pinky down
    action = gc.process_landmarks(lm)
    print(f"  action={action['action'] if action else 'None'}  "
          f"executor pos: ({gc.executor._prev_x:.1f}, {gc.executor._prev_y:.1f})")

    # ── 2. PINCH -> CLICK ─────────────────────────────────────────────────────
    print("Testing PINCH -> CLICK")
    lm2 = make_landmarks()
    lm2[4].x, lm2[4].y = 0.50, 0.50   # thumb tip
    lm2[8].x, lm2[8].y = 0.50, 0.51   # index tip (very close -> pinch)
    lm2[8].y = 0.2;  lm2[6].y  = 0.3  # keep index "up" in y direction too
    lm2[12].y = 0.6; lm2[10].y = 0.5
    lm2[16].y = 0.6; lm2[14].y = 0.5
    lm2[20].y = 0.6; lm2[18].y = 0.5
    action2 = gc.process_landmarks(lm2)
    print(f"  action={action2['action'] if action2 else 'None'}")

    # ── 3. SCROLL (two fingers up) ────────────────────────────────────────────
    print("Testing SCROLL (two fingers up)")
    lm3 = make_landmarks()
    lm3[8].y = 0.2;  lm3[6].y  = 0.3   # index up
    lm3[12].y = 0.2; lm3[10].y = 0.3   # middle up
    lm3[16].y = 0.6; lm3[14].y = 0.5   # ring down
    lm3[20].y = 0.6; lm3[18].y = 0.5   # pinky down
    action3 = gc.process_landmarks(lm3)
    print(f"  action={action3['action'] if action3 else 'None'}")

    # ── 4. FIST ───────────────────────────────────────────────────────────────
    print("Testing FIST -> PAUSE event")
    lm4 = make_landmarks()
    for tip, pip in [(8,6),(12,10),(16,14),(20,18)]:
        lm4[tip].y = 0.6; lm4[pip].y = 0.5
    action4 = gc.process_landmarks(lm4)
    print(f"  action={action4['action'] if action4 else 'None'}")

    print("\n[OK] All tests passed without crashing.")

if __name__ == "__main__":
    test_gestures()
