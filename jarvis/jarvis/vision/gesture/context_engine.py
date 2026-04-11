"""
jarvis/vision/gesture/context_engine.py

Application-aware behaviour layer.
Detects the active window / app and maps gestures to context-specific actions.
"""
from __future__ import annotations

import ctypes
import time
from typing import Dict, Optional


def _get_active_window_title() -> str:
    """Return the foreground window title on Windows, else 'screen'."""
    try:
        if not hasattr(ctypes, "windll"):
            return "screen"
        hwnd   = ctypes.windll.user32.GetForegroundWindow()
        length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
        if not length:
            return "screen"
        buff = ctypes.create_unicode_buffer(length + 1)
        ctypes.windll.user32.GetWindowTextW(hwnd, buff, length + 1)
        title = buff.value.strip()
        return title.lower() if title else "screen"
    except Exception:
        return "screen"


class ContextEngine:
    """
    Detects the active application and classifies it into a broad type
    so the rest of the pipeline can apply context-specific mappings.

    Output of :meth:`get_context`::

        {
            "app":   str,   # raw window title (lowercased)
            "type":  str,   # "media" | "browser" | "editor" | "terminal" | "any"
        }
    """

    # How often (seconds) we poll the foreground window
    POLL_INTERVAL: float = 0.5

    def __init__(self, app_keywords: Optional[Dict[str, list]] = None) -> None:
        # app_keywords: {"media": [...], "browser": [...], ...}
        self._app_keywords: Dict[str, list] = app_keywords or {}
        self.current_app:   str = "screen"
        self.current_type:  str = "any"
        self._last_poll:    float = 0.0

    # ── Public API ────────────────────────────────────────────────────────────

    def update_context(self) -> None:
        """
        Poll the OS for the active window title (rate-limited).
        Updates internal ``current_app`` and ``current_type`` state.
        """
        now = time.monotonic()
        if now - self._last_poll < self.POLL_INTERVAL:
            return
        self._last_poll = now

        title = _get_active_window_title()
        self.current_app  = title
        self.current_type = self._classify(title)

    def get_context(self) -> Dict:
        """Return the most recent context snapshot."""
        return {"app": self.current_app, "type": self.current_type}

    def map_gesture(self, gesture: str, context: Optional[Dict] = None) -> Optional[str]:
        """
        Return a context-overridden action for *gesture*, or ``None`` if
        no override applies (caller should fall back to default mapping).

        Example overrides:
        - PINCH in media  → "PLAY_PAUSE"
        - SCROLL in media → "VOLUME"
        """
        ctx_type = (context or self.get_context()).get("type", "any")

        _overrides: Dict[str, Dict[str, str]] = {
            "media": {
                "PINCH":      "PLAY_PAUSE",
                "SCROLL":     "VOLUME",
                "SWIPE_LEFT": "PREV_TRACK",
                "SWIPE_RIGHT":"NEXT_TRACK",
                "SWIPE_UP":   "VOL_UP",
                "SWIPE_DOWN": "VOL_DOWN",
            },
            "browser": {
                "SWIPE_LEFT":  "BROWSER_BACK",
                "SWIPE_RIGHT": "BROWSER_FORWARD",
                "CIRCLE":      "BROWSER_REFRESH",
            },
            "editor": {
                "CIRCLE":     "COMMENT_TOGGLE",
                "SWIPE_DOWN": "TOGGLE_TERMINAL",
            },
        }
        return _overrides.get(ctx_type, {}).get(gesture)

    def load_keywords(self, keyword_map: Dict[str, list]) -> None:
        """
        Inject keyword→type mappings (loaded from gesture_macros.yaml).
        Call this after the MacroEngine has parsed the YAML file.
        """
        self._app_keywords = keyword_map

    # ── Private ───────────────────────────────────────────────────────────────

    def _classify(self, title: str) -> str:
        """Return the app type string for the given window title."""
        for app_type, keywords in self._app_keywords.items():
            if any(kw in title for kw in keywords):
                return app_type
        return "any"
