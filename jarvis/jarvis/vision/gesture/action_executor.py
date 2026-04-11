"""
jarvis/vision/gesture/action_executor.py

Dispatches resolved actions to pyautogui, OS APIs, and the JARVIS event bus.
This is the only module that produces real side effects (mouse, keyboard, events).
"""
from __future__ import annotations

import logging
import subprocess
import time
from typing import Dict, Optional

log = logging.getLogger(__name__)

try:
    import pyautogui
    pyautogui.FAILSAFE = False
    pyautogui.PAUSE = 0  # CRITICAL: Prevent 0.1s block on every motion
except ImportError:
    pyautogui = None  # type: ignore
    log.warning("ActionExecutor: pyautogui not available — mouse/keyboard actions disabled.")

try:
    from jarvis.core.event_bus import bus, SystemEvent
    _BUS_AVAILABLE = True
except ImportError:
    _BUS_AVAILABLE = False


class ActionExecutor:
    """
    Execute atomic action dicts produced by :class:`FusionEngine` or
    :class:`MacroEngine`.

    Supported action types
    ----------------------
    ``MOVE``    – move cursor to absolute screen position
    ``CLICK``   – left click (with cooldown guard)
    ``SCROLL``  – vertical scroll by *amount* clicks
    ``HOTKEY``  – pyautogui.hotkey(*keys)
    ``KEY``     – pyautogui.press(key)
    ``OPEN``    – subprocess.Popen(app_path)
    ``BUS``     – publish a :class:`SystemEvent` on the event bus
    ``NOOP``    – do nothing (used as a safe fallback)
    """

    def __init__(
        self,
        smoothing:      float = 5.0,
        click_cooldown: float = 0.5,
        screen_w:       Optional[int] = None,
        screen_h:       Optional[int] = None,
    ) -> None:
        self.smoothing      = smoothing
        self.click_cooldown = click_cooldown

        if pyautogui:
            self.screen_w, self.screen_h = pyautogui.size()
        else:
            self.screen_w = screen_w or 1920
            self.screen_h = screen_h or 1080

        # Mode Management (Default vs 3D)
        self.interaction_mode = "HOTKEY_MODE"
        self._scene_manager = None

        # Smooth-move state
        self._prev_x: float = 0.0
        self._prev_y: float = 0.0

        # Continuous interaction state
        self._last_scroll_y:  float = 0.0
        self._last_zoom_time: float = 0.0
        self._last_rotate_time: float = 0.0
        
        # Accumulators for continuous interaction
        self._zoom_accum: float = 0.0
        self._rotate_accum: float = 0.0
        self.ZOOM_THRESHOLD = 0.05
        self.ROTATE_THRESHOLD = 0.1
        
        # Subscribe to continuous gesture actions if bus is available
        if _BUS_AVAILABLE:
            bus.subscribe("gesture.action", self._on_gesture_action)
            bus.subscribe("gesture.detected", self._on_gesture_detected)

        # Cooldown state
        self._last_click_time: float = 0.0
        self._last_focus_time: float = 0.0

        # Screen margin mapping for MOVE
        self.frame_margin_x: float = 0.20
        self.frame_margin_y: float = 0.20

    # ── Public API / dispatch ─────────────────────────────────────────────────

    def execute(self, action: Dict) -> None:
        """
        Dispatch *action* dict.  The dict must contain at least a ``type`` key.
        Keys are case-insensitive.

        Called by :class:`FusionEngine` (rich action dicts) and
        :class:`MacroEngine` (YAML action dicts).
        """
        a_type = str(action.get("type", "NOOP")).upper()

        dispatch = {
            "MOVE":   self._do_move,
            "CLICK":  self._do_click,
            "SCROLL": self._do_scroll,
            "HOTKEY": self._do_hotkey,
            "KEY":    self._do_key,
            "OPEN":   self._do_open,
            "BUS":    self._do_bus,
            "TRANSFORM": self._do_transform,
            "SWITCH_MODE": self._do_switch_mode,
            "ZOOM_IN": self._do_zoom_in,
            "ZOOM_OUT": self._do_zoom_out,
            "ROTATE_LEFT": self._do_rotate_left,
            "ROTATE_RIGHT": self._do_rotate_right,
            "NOOP":   lambda _: None,
            "PREVIOUS": lambda d: self._do_hotkey({"keys": ["alt", "left"]}),
            "NEXT": lambda d: self._do_hotkey({"keys": ["alt", "right"]}),
            "TRIGGER": lambda d: self._do_key({"key": "f5"}),
            "DRAG": lambda d: None, # Placeholder for explicit drag handling
            # Aliases used in YAML macros
            "PLAY_PAUSE":    lambda d: self._do_key({"key": "space"}),
            "VOLUME":        lambda d: None,   # handled by SWIPE_UP/DOWN macros
            "BROWSER_BACK":  lambda d: self._do_hotkey({"keys": ["alt", "left"]}),
            "BROWSER_FORWARD": lambda d: self._do_hotkey({"keys": ["alt", "right"]}),
            "BROWSER_REFRESH": lambda d: self._do_key({"key": "f5"}),
            "COMMENT_TOGGLE":  lambda d: self._do_hotkey({"keys": ["ctrl", "/"]}),
            "TOGGLE_TERMINAL": lambda d: self._do_hotkey({"keys": ["ctrl", "`"]}),
        }

        handler = dispatch.get(a_type)
        if handler:
            handler(action)
        else:
            log.warning("ActionExecutor: unknown action type '%s'", a_type)

    def move_cursor(self, x_norm: float, y_norm: float) -> None:
        """
        High-level helper: move cursor from normalised (0-1) coordinates.
        Applies margin mapping + exponential smoothing.
        """
        x_mapped = (x_norm - self.frame_margin_x) / (1.0 - 2 * self.frame_margin_x)
        y_mapped = (y_norm - self.frame_margin_y) / (1.0 - 2 * self.frame_margin_y)

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
        return curr_x, curr_y

    def scroll(self, y_current: float) -> None:
        """High-level scroll: call each frame with the current Y-normalised position."""
        if self._last_scroll_y != 0.0:
            y_diff = y_current - self._last_scroll_y
            if abs(y_diff) > 0.01 and pyautogui:
                pyautogui.scroll(int(-y_diff * 3000))
        self._last_scroll_y = y_current

    def reset_scroll(self) -> None:
        self._last_scroll_y = 0.0

    def publish_focus(self, window_title: str, x: float, y: float) -> None:
        """Emit the pointer-focus bus event (rate-limited to 0.5 s)."""
        now = time.monotonic()
        if now - self._last_focus_time > 0.5 and _BUS_AVAILABLE:
            bus.publish(SystemEvent(
                name="vision.pointer.focus",
                data={"focus": window_title, "x": x, "y": y, "confidence": 0.85},
            ))
            self._last_focus_time = now

    # ── Atomic action handlers ────────────────────────────────────────────────

    def _do_move(self, action: Dict) -> None:
        x = action.get("x", 0.0)
        y = action.get("y", 0.0)
        self.move_cursor(x, y)

    def _do_click(self, action: Dict) -> None:
        now = time.monotonic()
        if now - self._last_click_time < self.click_cooldown:
            return
        if pyautogui:
            pyautogui.click()
        self._last_click_time = now
        if _BUS_AVAILABLE:
            bus.publish(SystemEvent(
                name="vision.gesture.pinch",
                data={
                    "gesture":    "PINCH",
                    "confidence": action.get("confidence", 0.92),
                    "position":   {"x": self._prev_x, "y": self._prev_y},
                },
            ))

    def _do_scroll(self, action: Dict) -> None:
        amount = action.get("amount", 0)
        if pyautogui and amount:
            pyautogui.scroll(int(amount))

    def _do_hotkey(self, action: Dict) -> None:
        keys = action.get("keys", [])
        if pyautogui and keys:
            pyautogui.hotkey(*keys)
            log.debug("ActionExecutor: hotkey %s", keys)

    def _do_key(self, action: Dict) -> None:
        key = action.get("key", "")
        if pyautogui and key:
            pyautogui.press(key)

    def _do_open(self, action: Dict) -> None:
        app = action.get("app", "")
        if app:
            try:
                subprocess.Popen(app, shell=True)
                log.info("ActionExecutor: opened '%s'", app)
            except Exception as exc:
                log.error("ActionExecutor: failed to open '%s' — %s", app, exc)

    def _do_bus(self, action: Dict) -> None:
        if not _BUS_AVAILABLE:
            return
        event_name = action.get("event", "gesture.macro")
        data       = action.get("data", {})
        bus.publish(SystemEvent(name=event_name, data=data))
        log.debug("ActionExecutor: bus event '%s'", event_name)

    # ── 3D Interaction Handlers (Continuous) ──────────────────────────────────
    
    def _on_gesture_action(self, event: "SystemEvent") -> None:
        """Listener for actions emitted directly from the gesture pipeline."""
        action_name = event.data.get("action", "NONE")
        event_type = event.data.get("type", "NONE")
        
        # Whitelist of actions this executor handles directly from the gesture bus
        allowed_actions = [
            "CLICK", "DRAG", "PREVIOUS", "NEXT", "TRIGGER",
            "ZOOM_IN", "ZOOM_OUT", "ROTATE_LEFT", "ROTATE_RIGHT"
        ]
        
        if event_type == "TRANSFORM":
            self.execute(event.data)
        elif action_name in allowed_actions:
            self.execute({"type": action_name})

    def _on_gesture_detected(self, event: "SystemEvent") -> None:
        """High-frequency listener for raw detection/movement."""
        gesture = event.data.get("gesture")
        state   = event.data.get("state")
        
        # If we are in INDEX_POINT gesture and ACTIVE state, move the cursor
        if gesture == "INDEX_POINT" and state == "ACTIVE":
            x = event.data.get("x", 0.0)
            y = event.data.get("y", 0.0)
            self.move_cursor(x, y)

    def _do_transform(self, action: Dict) -> None:
        if getattr(self, "interaction_mode", "HOTKEY_MODE") == "HOLOGRAM_3D_MODE":
            if self._scene_manager and hasattr(self._scene_manager, 'ws_bridge'):
                # Forward entirely to the websocket layer
                self._scene_manager.ws_bridge.send_action(action)
            return

        now = time.monotonic()
        transform_action = action.get("action")
        delta = action.get("delta", 0.0)
        
        if transform_action == "ZOOM":
            self._zoom_accum += delta
            if abs(self._zoom_accum) > self.ZOOM_THRESHOLD:
                if now - self._last_zoom_time > 0.05:
                    if self._zoom_accum < 0:
                        self.execute({"type": "ZOOM_IN"})
                    else:
                        self.execute({"type": "ZOOM_OUT"})
                    self._last_zoom_time = now
                    # Decay accumulator to prevent runaway
                    self._zoom_accum = 0.0

        elif transform_action == "ROTATE":
            self._rotate_accum += delta
            if abs(self._rotate_accum) > self.ROTATE_THRESHOLD:
                if now - self._last_rotate_time > 0.1:
                    if self._rotate_accum > 0:
                        self.execute({"type": "ROTATE_RIGHT"})
                    else:
                        self.execute({"type": "ROTATE_LEFT"})
                    self._last_rotate_time = now
                    self._rotate_accum = 0.0

    def _do_zoom_in(self, action: Dict) -> None:
        now = time.monotonic()
        # Cap zooming frequency to prevent massive overload
        if pyautogui and (now - self._last_zoom_time) > 0.05:
            # typical zoom hotkey: ctrl + up or scroll
            pyautogui.hotkey('ctrl', '+')
            self._last_zoom_time = now

    def _do_zoom_out(self, action: Dict) -> None:
        now = time.monotonic()
        if pyautogui and (now - self._last_zoom_time) > 0.05:
            pyautogui.hotkey('ctrl', '-')
            self._last_zoom_time = now

    def _do_rotate_left(self, action: Dict) -> None:
        now = time.monotonic()
        if pyautogui and (now - self._last_rotate_time) > 0.1:
            pyautogui.press('left') # General mapping, configurable per context later
            self._last_rotate_time = now
            
    def _do_rotate_right(self, action: Dict) -> None:
        now = time.monotonic()
        if pyautogui and (now - self._last_rotate_time) > 0.1:
            pyautogui.press('right')
            self._last_rotate_time = now

    def _do_switch_mode(self, action: Dict) -> None:
        new_mode = action.get("mode", "HOTKEY_MODE").upper()
        if new_mode == self.interaction_mode:
            return
            
        self.interaction_mode = new_mode
        log.info("ActionExecutor: Switched mode to %s", new_mode)
        
        if new_mode == "HOLOGRAM_3D_MODE":
            import importlib
            scene_manager_module = importlib.import_module("jarvis.hui.3d.scene_manager")
            HolographicSceneManager = scene_manager_module.HolographicSceneManager
            if not self._scene_manager:
                self._scene_manager = HolographicSceneManager()
            self._scene_manager.start()
        else:
            if self._scene_manager:
                self._scene_manager.stop()
