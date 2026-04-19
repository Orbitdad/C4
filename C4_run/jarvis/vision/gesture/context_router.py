"""
jarvis/vision/gesture/context_router.py

Routes universal intents to context-aware actions based on the active mode (UI vs 3D).
Emits execution instructions.
"""

from typing import Dict
from jarvis.core.event_bus import bus, SystemEvent, EventPriority

class ContextRouter:
    def __init__(self):
        self.last_action_time = 0.0
        self.last_intent = "NONE"
        self.last_nav_time = 0.0
        
        # State tracking for continuous dual-hand values
        self.prev_scale_dist = None
        self.prev_rot_angle = None

    def route_intent(self, intent_payload: Dict, mode: str) -> None:
        """
        Receives an intent payload and fires the appropriate actions based on mode.
        """
        intent = intent_payload["intent"]
        x = intent_payload["x"]
        y = intent_payload["y"]
        
        # 1. Always broadcast the current intent to general listeners (e.g. HUI overlay)
        bus.publish(SystemEvent(
            name="gesture.intent",
            data=intent_payload,
            priority=EventPriority.NORMAL
        ))

        # Reset continuous tracking state if intent changes
        if intent != "SCALE": self.prev_scale_dist = None
        if intent != "ROTATE": self.prev_rot_angle = None

        # State transition edge case (prevent action spam)
        if intent == self.last_intent and intent not in ["TARGET", "DRAG", "SCALE", "ROTATE"]:
             return # Only continuous intents re-fire every frame internally
             
        self.last_intent = intent

        # 2. Route based on mode
        if mode == "UI_MODE":
            self._route_ui(intent_payload)
        elif mode == "INTERACTION_3D":
            self._route_3d(intent_payload)

    def _route_ui(self, payload: Dict):
        """Map intents to 2D UI interactions."""
        intent = payload["intent"]
        x, y = payload["x"], payload["y"]

        action = None
        if intent == "TARGET":
            action = {"type": "MOVE_CURSOR", "x": x, "y": y}
        elif intent == "SELECT":
            action = {"type": "CLICK", "x": x, "y": y}
        elif intent == "DRAG":
            # For UI, drag implies down + move
            action = {"type": "MOUSE_DOWN_MOVE", "x": x, "y": y}
        elif intent == "DROP":
            action = {"type": "MOUSE_UP"}
        elif intent == "CANCEL":
            action = {"type": "KEY", "key": "esc"}
        elif intent == "NAVIGATE":
            dir = payload.get("direction", "left")
            action = {"type": "HOTKEY", "keys": ["alt", dir]}
        elif intent == "CONFIRM":
            action = {"type": "KEY", "key": "enter"}
        elif intent == "BACK":
            action = {"type": "KEY", "key": "backspace"}

        if action:
            self._dispatch(action)

    def _route_3d(self, payload: Dict):
        """Map intents to 3D object manipulation commands via WS broadcast."""
        intent = payload["intent"]
        
        # In 3D we almost exclusively talk to the web-client via Websocket
        action = None
        
        if intent == "TARGET":
             # Tell 3D space where the selection raycast should aim
             action = {"type": "WS_BROADCAST", "ws_type": "POINTER_SYNC", "x": payload["x"], "y": payload["y"]}
        elif intent == "SELECT":
             action = {"type": "WS_BROADCAST", "ws_type": "ACTION", "action": "GRAB"}
        elif intent == "DRAG":
             action = {"type": "WS_BROADCAST", "ws_type": "ACTION", "action": "MOVE", "x": payload["x"], "y": payload["y"]}
        elif intent == "DROP":
             action = {"type": "WS_BROADCAST", "ws_type": "ACTION", "action": "RELEASE"}
        elif intent == "CANCEL":
             action = {"type": "WS_BROADCAST", "ws_type": "ACTION", "action": "DESELECT"}
        elif intent == "NAVIGATE":
             dir = payload.get("direction", "left")
             if dir == "left":
                 action = {"type": "KEY", "key": "numpad4"} # Blender orbit left mapping
             else:
                 action = {"type": "KEY", "key": "numpad6"}
        elif intent == "SCALE":
             dist = payload.get("distance", 0)
             if self.prev_scale_dist:
                 delta = dist - self.prev_scale_dist
                 if abs(delta) > 0.01:
                     action = {"type": "WS_BROADCAST", "ws_type": "TRANSFORM", "action": "ZOOM", "delta": delta * 5.0}
             self.prev_scale_dist = dist
        elif intent == "ROTATE":
             angle = payload.get("angle", 0)
             if self.prev_rot_angle:
                 delta = angle - self.prev_rot_angle
                 if abs(delta) > 0.05:
                     action = {"type": "WS_BROADCAST", "ws_type": "TRANSFORM", "action": "ROTATE", "delta": delta}
             self.prev_rot_angle = angle

        if action:
            self._dispatch(action)

    def _dispatch(self, action_dict: Dict):
        """Fire the final resolved action."""
        if action_dict["type"] == "NONE":
            return
            
        import time
        
        # Cooldowns for discrete navigation
        if action_dict["type"] == "HOTKEY" and "alt" in action_dict.get("keys", []):
            if time.time() - self.last_nav_time < 0.5:
                return
            self.last_nav_time = time.time()
            
        bus.publish(SystemEvent(
            name="gesture.action",
            data=action_dict,
            priority=EventPriority.HIGH
        ))
