"""
jarvis/vision/gesture/action_executor.py

Executes concrete OS actions (mouse movement, clicks, keyboard) requested by ContextRouter.
Contains no gesture parsing or intent mapping logic.
"""

import logging
import time
from typing import Dict, Optional

log = logging.getLogger(__name__)

try:
    import pyautogui
    pyautogui.FAILSAFE = False
    pyautogui.PAUSE = 0
except ImportError:
    pyautogui = None
    log.warning("ActionExecutor: pyautogui not available.")

try:
    from jarvis.core.event_bus import bus, SystemEvent
    _BUS_AVAILABLE = True
except ImportError:
    _BUS_AVAILABLE = False


class ActionExecutor:
    def __init__(self, smoothing: float = 5.0, frame_margin: float = 0.20):
        self.smoothing = smoothing
        self.frame_margin = frame_margin

        if pyautogui:
            self.screen_w, self.screen_h = pyautogui.size()
        else:
            self.screen_w, self.screen_h = 1920, 1080

        self._prev_x: float = 0.0
        self._prev_y: float = 0.0
        self._last_click_time: float = 0.0
        
        self.click_cooldown = 0.4
        
        # Map mouse state to track drag
        self.mouse_down = False

        if _BUS_AVAILABLE:
            bus.subscribe("gesture.action", self._on_action_received)

    def _on_action_received(self, event: "SystemEvent") -> None:
        """Listener for resolved actions from the ContextRouter."""
        self.execute(event.data)

    def execute(self, action: Dict) -> None:
        """Dispatch concrete action type."""
        a_type = action.get("type", "NOOP")

        if a_type == "MOVE_CURSOR":
            self._do_move(action)
        elif a_type == "CLICK":
            self._do_click()
        elif a_type == "MOUSE_DOWN_MOVE":
            self._do_mouse_down_move(action)
        elif a_type == "MOUSE_UP":
            self._do_mouse_up()
        elif a_type == "KEY":
            self._do_key(action)
        elif a_type == "HOTKEY":
            self._do_hotkey(action)
        elif a_type == "WS_BROADCAST":
            self._do_ws_broadcast(action)
        elif a_type == "BUS":
            self._do_bus(action)

    def move_cursor(self, x_norm: float, y_norm: float) -> None:
        """Smooth normalized coordinates and send to OS."""
        x_mapped = (x_norm - self.frame_margin) / (1.0 - 2 * self.frame_margin)
        y_mapped = (y_norm - self.frame_margin) / (1.0 - 2 * self.frame_margin)

        x_mapped = max(0.0, min(1.0, x_mapped))
        y_mapped = max(0.0, min(1.0, y_mapped))

        screen_x = x_mapped * self.screen_w
        screen_y = y_mapped * self.screen_h

        if self._prev_x == 0.0 and self._prev_y == 0.0:
            self._prev_x, self._prev_y = screen_x, screen_y

        curr_x = self._prev_x + (screen_x - self._prev_x) / self.smoothing
        curr_y = self._prev_y + (screen_y - self._prev_y) / self.smoothing

        if pyautogui:
            pyautogui.moveTo(int(curr_x), int(curr_y))

        self._prev_x, self._prev_y = curr_x, curr_y

    def _do_move(self, action: Dict):
        self.move_cursor(action.get("x", 0.0), action.get("y", 0.0))

    def _do_click(self):
        # We ensure mouse is up before clicking incase we were dragging
        if self.mouse_down and pyautogui:
            pyautogui.mouseUp()
            self.mouse_down = False
            
        now = time.time()
        if now - self._last_click_time < self.click_cooldown:
            return
        if pyautogui:
            pyautogui.click()
        self._last_click_time = now

    def _do_mouse_down_move(self, action: Dict):
        self.move_cursor(action.get("x", 0.0), action.get("y", 0.0))
        if not self.mouse_down and pyautogui:
             pyautogui.mouseDown()
             self.mouse_down = True

    def _do_mouse_up(self):
        if self.mouse_down and pyautogui:
            pyautogui.mouseUp()
            self.mouse_down = False

    def _do_key(self, action: Dict):
        key = action.get("key", "")
        if pyautogui and key:
            pyautogui.press(key)

    def _do_hotkey(self, action: Dict):
        keys = action.get("keys", [])
        if pyautogui and keys:
            pyautogui.hotkey(*keys)

    def _do_bus(self, action: Dict):
        if not _BUS_AVAILABLE: return
        bus.publish(SystemEvent(name=action.get("event"), data=action.get("data", {})))

    def _do_ws_broadcast(self, action: Dict):
        from jarvis.skills.websocket_manager import WebSocketManager
        payload = dict(action)
        payload["type"] = action.get("ws_type", "ACTION") # remap keys to what WS expects
        WebSocketManager.instance().broadcast(payload)
