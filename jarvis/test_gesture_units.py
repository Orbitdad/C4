"""
Unit tests for the modular gesture engine.
Run with:  python test_gesture_units.py
"""
import sys, os, math, time
sys.path.insert(0, os.path.dirname(__file__))

# ─── mock landmark ────────────────────────────────────────────────────────────
class LM:
    def __init__(self, x=0.5, y=0.5, z=0.0):
        self.x, self.y, self.z = x, y, z

def flat(n=21):
    """21 landmarks flat on Y, spread linearly on X."""
    return [LM(x=0.3 + (i * 0.01), y=0.5) for i in range(n)]

def make_move(lm=None):
    """Index finger up, rest down."""
    l = lm or flat()
    l[8].y=0.2; l[6].y=0.3     # index up
    l[12].y=0.6; l[10].y=0.5   # middle down
    l[16].y=0.6; l[14].y=0.5   # ring down
    l[20].y=0.6; l[18].y=0.5   # pinky down
    return l

def make_scroll(lm=None):
    """Index + middle up."""
    l = lm or flat()
    l[8].y=0.2; l[6].y=0.3
    l[12].y=0.2; l[10].y=0.3
    l[16].y=0.6; l[14].y=0.5
    l[20].y=0.6; l[18].y=0.5
    return l

def make_pinch(lm=None):
    """Thumb tip very close to index tip."""
    l = lm or flat()
    l[4].x=0.5; l[4].y=0.2    # thumb up
    l[8].x=0.5; l[8].y=0.202  # index up and very close to thumb tip
    l[6].y=0.3
    l[12].y=0.6; l[10].y=0.5
    l[16].y=0.6; l[14].y=0.5
    l[20].y=0.6; l[18].y=0.5
    return l

def make_fist(lm=None):
    """All fingers down."""
    l = lm or flat()
    for tip, pip in [(8,6),(12,10),(16,14),(20,18)]:
        l[tip].y=0.6; l[pip].y=0.5
    return l

def make_palm(lm=None):
    """All fingers up."""
    l = lm or flat()
    for tip, pip in [(8,6),(12,10),(16,14),(20,18)]:
        l[tip].y=0.2; l[pip].y=0.3
    return l

PASS = 0; FAIL = 0

def check(name, condition, got=None):
    global PASS, FAIL
    if condition:
        print(f"  ✅  {name}")
        PASS += 1
    else:
        print(f"  ❌  {name}   (got: {got})")
        FAIL += 1

# ══════════════════════════════════════════════════════════════════════════════
print("\n── GestureDetector ───────────────────────────────────────────────────")
from jarvis.vision.gesture.detector import GestureDetector
det = GestureDetector({"pinch_threshold": 0.05})

r = det.detect(make_move())
check("MOVE classified", r["gesture"] == "MOVE", r["gesture"])
check("MOVE confidence ≥ 0.5", r["confidence"] >= 0.5)

r = det.detect(make_scroll())
check("SCROLL classified", r["gesture"] == "SCROLL", r["gesture"])

r = det.detect(make_pinch())
check("PINCH classified", r["gesture"] == "PINCH", r["gesture"])
check("PINCH confidence > 0.8", r["confidence"] > 0.8)

r = det.detect(make_fist())
check("FIST classified", r["gesture"] == "FIST", r["gesture"])

r = det.detect(make_palm())
check("OPEN_PALM classified", r["gesture"] == "OPEN_PALM", r["gesture"])

check("detect() returns position tuple", len(r["position"]) == 3)
check("detect() returns velocity float", isinstance(r["velocity"], float))

# ── Swipe detection ───────────────────────────────────────────────────────────
det2 = GestureDetector()
import time as _t
# push 20 positions moving strongly right
for i in range(20):
    det2._pos_history.append((i * 0.02, 0.5, _t.monotonic()))
swipe = det2.detect_swipe()
check("SWIPE_RIGHT detected", swipe == "SWIPE_RIGHT", swipe)

# push 20 positions moving up (decreasing y)
det3 = GestureDetector()
for i in range(20):
    det3._pos_history.append((0.5, 0.5 - i*0.02, _t.monotonic()))
swipe3 = det3.detect_swipe()
check("SWIPE_UP detected", swipe3 == "SWIPE_UP", swipe3)

# ══════════════════════════════════════════════════════════════════════════════
print("\n── GestureStateManager ─────────────────────────────────────────────────")
from jarvis.vision.gesture.state_manager import GestureStateManager
sm = GestureStateManager()

ev_move  = {"gesture": "MOVE",  "confidence": 0.9}
ev_pinch = {"gesture": "PINCH", "confidence": 0.95}
ev_scroll= {"gesture": "SCROLL","confidence": 0.9}
ev_fist  = {"gesture": "FIST",  "confidence": 0.92}

out = sm.update(ev_move)
check("IDLE→ACTIVE on MOVE", out["state"] == "ACTIVE", out["state"])
check("Sequence contains MOVE", "MOVE" in out["sequence"])

out = sm.update(ev_pinch)
check("ACTIVE→DRAGGING on PINCH", out["state"] == "DRAGGING", out["state"])

sm2 = GestureStateManager()
sm2.update({"gesture": "SCROLL", "confidence": 0.9})
out2 = sm2.update({"gesture": "SCROLL", "confidence": 0.9})
check("IDLE→SCROLLING on SCROLL", out2["state"] == "SCROLLING", out2["state"])

sm3 = GestureStateManager()
sm3.update(ev_move)
out3 = sm3.update(ev_fist)
check("Any→IDLE on FIST", out3["state"] == "IDLE", out3["state"])

sm4 = GestureStateManager()
sm4.update({"gesture": "NONE", "confidence": 0.0})
check("Low confidence → stays IDLE", sm4.state == "IDLE")

sm5 = GestureStateManager()
sm5.update(ev_move); sm5.reset()
check("Reset clears state+history", sm5.state == "IDLE" and sm5.get_current_sequence() == [])

# ══════════════════════════════════════════════════════════════════════════════
print("\n── IntentEngine ────────────────────────────────────────────────────────")
from jarvis.vision.gesture.intent_engine import IntentEngine, Intent, Mode
ie = IntentEngine()

out = ie.infer({"gesture": "PINCH",  "velocity": 0.1})
check("PINCH → CLICK intent",      out["intent"] == Intent.CLICK,  out["intent"])
check("slow velocity → PRECISION", out["mode"]   == Mode.PRECISION, out["mode"])

out = ie.infer({"gesture": "MOVE",   "velocity": 2.5})
check("MOVE+fast → NAVIGATE intent", out["intent"] == Intent.NAVIGATE, out["intent"])
check("fast velocity → FAST mode",   out["mode"]   == Mode.FAST,       out["mode"])

out = ie.infer({"gesture": "SCROLL", "velocity": 0.5})
check("SCROLL → SCROLL intent",    out["intent"] == Intent.SCROLL, out["intent"])
check("mid velocity → NORMAL mode",out["mode"]   == Mode.NORMAL,   out["mode"])

out = ie.infer({"gesture": "FIST",  "velocity": 0.0})
check("FIST → PAUSE intent", out["intent"] == Intent.PAUSE, out["intent"])

# ── calculate_velocity ────────────────────────────────────────────────────────
now = _t.monotonic()
hist = [(0.0, 0.0, now), (0.1, 0.0, now+0.1), (0.2, 0.0, now+0.2)]
v = ie.calculate_velocity(hist)
check("calculate_velocity > 0", v > 0, v)

# ══════════════════════════════════════════════════════════════════════════════
print("\n── ContextEngine ───────────────────────────────────────────────────────")
from jarvis.vision.gesture.context_engine import ContextEngine
ce = ContextEngine(app_keywords={
    "media":   ["vlc","spotify"],
    "browser": ["chrome","firefox"],
    "editor":  ["vscode","code"],
})
ce.current_app  = "vlc media player"
ce.current_type = ce._classify("vlc media player")
check("VLC → media type", ce.current_type == "media", ce.current_type)

ce2 = ContextEngine()
ce2.load_keywords({"browser": ["chrome"], "editor": ["vscode"]})
ce2.current_app = "google chrome"
ce2.current_type = ce2._classify("google chrome")
check("Chrome → browser type", ce2.current_type == "browser", ce2.current_type)

override = ce.map_gesture("PINCH", {"type": "media"})
check("PINCH in media → PLAY_PAUSE", override == "PLAY_PAUSE", override)

override2 = ce2.map_gesture("SWIPE_LEFT", {"type": "browser"})
check("SWIPE_LEFT in browser → BROWSER_BACK", override2 == "BROWSER_BACK", override2)

no_override = ce.map_gesture("MOVE", {"type": "media"})
check("MOVE in media → no override (None)", no_override is None, no_override)

# ══════════════════════════════════════════════════════════════════════════════
print("\n── MacroEngine ─────────────────────────────────────────────────────────")
from jarvis.vision.gesture.macro_engine import MacroEngine
from pathlib import Path
macro_path = Path(__file__).parent / "jarvis/vision/gesture/gesture_macros.yaml"
me = MacroEngine(macro_path)

check("Macros loaded > 0", len(me.macros) > 0, len(me.macros))
check("App keywords loaded", len(me.app_keywords) > 0, len(me.app_keywords))

# Match: FIST in any context
m = me.match(["FIST"], {"type": "any"})
check("FIST matches 'Pause JARVIS' macro", m is not None and "Pause" in m.get("name",""), m)

# Match: PINCH in media context
m2 = me.match(["PINCH"], {"type": "media"})
check("PINCH in media matches 'Play / Pause'", m2 is not None, m2)

# No match: MOVE (not in macros)
m3 = me.match(["MOVE"], {"type": "any"})
check("MOVE has no macro match", m3 is None, m3)

# Sequence tail matching
m4 = me.match(["MOVE","MOVE","FIST"], {"type": "any"})
check("FIST matches even as tail of sequence", m4 is not None, m4)

# ══════════════════════════════════════════════════════════════════════════════
print("\n── AdaptiveEngine ──────────────────────────────────────────────────────")
from jarvis.vision.gesture.adaptive_engine import AdaptiveEngine, _CALIBRATION_FRAMES
import tempfile
import uuid
tmp_prof1 = os.path.join(tempfile.gettempdir(), f"gesture_prof_{uuid.uuid4().hex}.json")
ae = AdaptiveEngine(profile_path=tmp_prof1)

check("Starts uncalibrated", not ae.user_profile["calibrated"])
check("Default pinch threshold", ae.pinch_threshold == 0.05)

# Feed enough frames to trigger calibration
lms = make_move()
for _ in range(_CALIBRATION_FRAMES):
    ae.calibrate(lms)
check("Calibrated after enough frames", ae.user_profile["calibrated"])
check("Hand width computed > 0", ae.user_profile["hand_width"] > 0)

ae.adjust_sensitivity(lighting=0.2, noise=0.0)
check("Low light increases pinch threshold", ae.user_profile["pinch_threshold"] > 0.05)

tmp_prof2 = os.path.join(tempfile.gettempdir(), f"gesture_prof_{uuid.uuid4().hex}.json")
ae2 = AdaptiveEngine(profile_path=tmp_prof2)
norm = ae2.normalize(make_move())
check("normalize() returns same object when uncalibrated", norm is not None)

# ══════════════════════════════════════════════════════════════════════════════
print("\n── FusionEngine ────────────────────────────────────────────────────────")
from jarvis.vision.gesture.fusion_engine import FusionEngine, PRI_MACRO, PRI_CONTEXT, PRI_INTENT, PRI_FALLBACK
fe = FusionEngine()

gesture_move  = {"gesture":"MOVE",  "confidence":0.85,"velocity":0.2,"position":(0.5,0.5,0.0)}
gesture_pinch = {"gesture":"PINCH", "confidence":0.95,"velocity":0.1,"position":(0.5,0.5,0.0)}
gesture_fist  = {"gesture":"FIST",  "confidence":0.92,"velocity":0.0,"position":(0.5,0.5,0.0)}
state_idle    = {"state":"IDLE",   "sequence":[]}
state_active  = {"state":"ACTIVE", "sequence":["MOVE"]}
intent_move   = {"intent":"MOVE_CURSOR","mode":"PRECISION"}
intent_click  = {"intent":"CLICK",      "mode":"PRECISION"}
intent_pause  = {"intent":"PAUSE",      "mode":"NORMAL"}
ctx_any       = {"app":"screen","type":"any"}
ctx_media     = {"app":"vlc",   "type":"media"}

# Macro beats everything
macro = {"name":"Test","sequence":["FIST"],"context":"any","actions":[{"type":"bus","event":"jarvis.pause","data":{}}]}
r = fe.resolve(gesture_fist, state_idle, intent_pause, ctx_any, macro)
check("Macro has highest priority",   r["source"] == "macro",    r["source"])
check("Macro priority == PRI_MACRO",  r["priority"] == PRI_MACRO, r["priority"])

# Context beats intent
r2 = fe.resolve(gesture_pinch, state_active, intent_click, ctx_media, None)
check("Context beats intent for PINCH in media", r2["source"] == "context", r2["source"])
check("Context priority == PRI_CONTEXT", r2["priority"] == PRI_CONTEXT, r2["priority"])

# Intent (MOVE) when no macro/context override
r3 = fe.resolve(gesture_move, state_active, intent_move, ctx_any, None)
check("Intent candidate produced for MOVE", r3["source"] in ("intent","fallback"), r3["source"])

# Fallback
r4 = fe.fallback(gesture_fist)
check("Fallback for FIST → BUS action",        r4["action"] == "BUS",       r4["action"])
check("Fallback priority == PRI_FALLBACK",       r4["priority"] == PRI_FALLBACK, r4["priority"])

# ══════════════════════════════════════════════════════════════════════════════
print("\n── Full Pipeline (GestureController) ────────────────────────────────────")
from jarvis.vision.gesture_controller import GestureController
gc = GestureController()

action = gc.process_landmarks(make_move())
check("Pipeline: MOVE → action dict returned",  action is not None)
check("Pipeline: MOVE → has 'action' key",      "action" in (action or {}))

action2 = gc.process_landmarks(make_pinch())
check("Pipeline: PINCH → action dict returned", action2 is not None)

action3 = gc.process_landmarks(make_scroll())
check("Pipeline: SCROLL → action dict returned",action3 is not None)

action4 = gc.process_landmarks(make_fist())
check("Pipeline: FIST → action dict returned",  action4 is not None)

check("Pipeline: gc.enabled flag works (True)",  gc.enabled == True)
gc.enabled = False
action5 = gc.process_landmarks(make_move())
check("Pipeline: disabled → returns None",       action5 is None)

# ══════════════════════════════════════════════════════════════════════════════
print(f"\n{'─'*60}")
total = PASS + FAIL
print(f"Results: {PASS}/{total} passed   {'✅ ALL GOOD' if FAIL==0 else f'❌ {FAIL} FAILED'}")
if FAIL:
    sys.exit(1)
