"""
ScreenReader — Visual Grounding for JARVIS (Phase 10).

Periodically captures the screen and extracts:
  - Active window title (via pygetwindow / win32gui)
  - OCR text from the active region (pytesseract, with Pillow fallback)
  - Clipboard snapshot

The result is stored in WorldState so the ReasoningEngine can inject it
into the LLM system prompt as real-time visual context.

Dependencies (soft — all gracefully degrade):
  - pygetwindow  (pip install pygetwindow)   — window title
  - pytesseract  (pip install pytesseract)   — OCR
  - Pillow  (already in requirements.txt)    — screenshots
  - pywin32  (pip install pywin32)           — fallback window title on Windows
"""

from __future__ import annotations

import logging
import threading
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── optional heavy imports ────────────────────────────────────────────────────

try:
    import pygetwindow as gw
    _HAS_GW = True
except ImportError:
    _HAS_GW = False

try:
    import pytesseract
    from PIL import ImageGrab, ImageFilter
    _HAS_OCR = True
except ImportError:
    _HAS_OCR = False
    try:
        from PIL import ImageGrab
        _HAS_PIL = True
    except ImportError:
        _HAS_PIL = False

try:
    import ctypes
    _HAS_CTYPES = True
except ImportError:
    _HAS_CTYPES = False

try:
    import win32gui
    _HAS_WIN32 = True
except ImportError:
    _HAS_WIN32 = False


# ── helpers ───────────────────────────────────────────────────────────────────

def _get_active_window_title() -> str:
    """Return the title of the foreground window."""
    if _HAS_GW:
        try:
            w = gw.getActiveWindow()
            return (w.title or "").strip() if w else ""
        except Exception:
            pass
    if _HAS_WIN32:
        try:
            hwnd = win32gui.GetForegroundWindow()
            return win32gui.GetWindowText(hwnd).strip()
        except Exception:
            pass
    return ""


def _ocr_active_region(max_chars: int = 800) -> str:
    """
    Capture the full screen and run OCR.
    Returns up to max_chars of extracted text, cleaned up.
    Degrades gracefully if pytesseract is not installed.
    """
    if not _HAS_OCR:
        return ""
    try:
        img = ImageGrab.grab()
        # Downscale to speed up OCR (half resolution)
        w, h = img.size
        img = img.resize((w // 2, h // 2))
        img = img.filter(ImageFilter.SHARPEN)
        raw = pytesseract.image_to_string(img, config="--psm 6")
        # Collapse whitespace, strip noise lines
        lines = [ln.strip() for ln in raw.splitlines() if len(ln.strip()) > 4]
        text = " | ".join(lines)
        return text[:max_chars]
    except Exception as exc:
        logger.debug(f"[ScreenReader] OCR failed: {exc}")
        return ""


def _get_clipboard_text(max_chars: int = 200) -> str:
    """Return the current clipboard contents (text only)."""
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        text = root.clipboard_get()
        root.destroy()
        return str(text)[:max_chars]
    except Exception:
        return ""


# ── Main class ─────────────────────────────────────────────────────────────────

class ScreenReader:
    """
    Background thread that builds a structured snapshot of what is
    currently visible on the user's desktop.

    Output is pushed into WorldState every `poll_interval` seconds
    so the ReasoningEngine always has fresh visual context.
    """

    DEFAULT_POLL_INTERVAL = 15.0   # seconds between full OCR passes
    WINDOW_POLL_INTERVAL = 2.0     # faster poll for just the window title

    def __init__(self, poll_interval: float = DEFAULT_POLL_INTERVAL) -> None:
        self.poll_interval = poll_interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._snapshot: dict = {
            "active_window": "",
            "ocr_text": "",
            "clipboard_snippet": "",
            "last_updated": 0.0,
        }
        self._lock = threading.Lock()

    # ── Public API ────────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="JARVISScreenReader"
        )
        self._thread.start()
        logger.info("[ScreenReader] Visual grounding started (OCR=%s).", _HAS_OCR)

    def stop(self) -> None:
        self._running = False

    def get_snapshot(self) -> dict:
        """Return the latest screen snapshot dict (thread-safe)."""
        with self._lock:
            return dict(self._snapshot)

    def get_context_string(self) -> str:
        """
        Return a compact, human-readable context string suitable for
        injection into an LLM system prompt.
        """
        snap = self.get_snapshot()
        parts = []
        if snap["active_window"]:
            parts.append(f"Active window: {snap['active_window']}")
        if snap["ocr_text"]:
            parts.append(f"Visible text (OCR): {snap['ocr_text'][:400]}")
        if snap["clipboard_snippet"]:
            parts.append(f"Clipboard: {snap['clipboard_snippet']}")
        return "\n".join(parts) if parts else "No visual context available."

    # ── Internal loop ─────────────────────────────────────────────────────────

    def _loop(self) -> None:
        last_ocr_time = 0.0
        while self._running:
            try:
                now = time.time()

                # Always refresh window title (fast)
                win_title = _get_active_window_title()

                # Rate-limit heavy OCR
                ocr_text = self._snapshot.get("ocr_text", "")
                if now - last_ocr_time >= self.poll_interval:
                    ocr_text = _ocr_active_region()
                    last_ocr_time = now

                clipboard = _get_clipboard_text()

                with self._lock:
                    self._snapshot = {
                        "active_window": win_title,
                        "ocr_text": ocr_text,
                        "clipboard_snippet": clipboard,
                        "last_updated": now,
                    }

                # Push to WorldState
                try:
                    from jarvis.core.world_state import world
                    world.set_temporal_context(
                        "screen_context",
                        {
                            "window": win_title,
                            "ocr_preview": ocr_text[:200],
                        },
                        ttl_seconds=self.poll_interval * 3,
                    )
                    # Also keep active_window in user_environment for meta-cognition
                    world.update_user_environment(active_window=win_title)
                except Exception:
                    pass

            except Exception as exc:
                logger.debug(f"[ScreenReader] Loop error: {exc}")

            time.sleep(self.WINDOW_POLL_INTERVAL)
