"""
Attention System — controls what JARVIS processes and what it ignores.

The attention system acts as a priority-based gate for all incoming events.
It models the concept of cognitive focus: when the user is actively present,
all input channels are open. In idle or background state, only high-priority
events (safety, resource alerts) break through.

Focus States:
  USER       — User is present, all modalities active
  SYSTEM     — No user, monitoring system events only
  BACKGROUND — Reduced monitoring (sleep-like)
  IDLE       — Minimum activity
"""
from __future__ import annotations

import logging
import threading
import time
from typing import Optional

logger = logging.getLogger(__name__)


# Attention priority thresholds per focus state
# Events below these EventPriority values are suppressed
_THRESHOLDS = {
    "USER":       1,   # All events pass (LOW=1)
    "SYSTEM":     2,   # Only NORMAL+ pass
    "BACKGROUND": 3,   # Only HIGH+ pass
    "IDLE":       4,   # Only CRITICAL pass
}


class AttentionSystem:
    """
    Priority-based event filter and focus state manager.
    Integrates with the EventBus and WorldState to provide context-aware
    attention control across all perception channels.
    """

    def __init__(self):
        self._lock = threading.RLock()
        self._focus: str = "IDLE"       # Current attention focus state
        self._focus_locked: bool = False # Prevent auto-switching during a task
        self._last_activity: float = time.time()
        self._idle_timeout: float = 120.0  # Seconds of no activity before downgrading focus

        # Subscribe to relevant events
        from jarvis.core.event_bus import bus
        bus.subscribe("face.user_detected", self._on_user_detected)
        bus.subscribe("face.user_left", self._on_user_left)
        bus.subscribe("voice.raw_transcript", self._on_voice_activity)
        bus.subscribe("vision.gesture.*", self._on_gesture_activity)
        bus.subscribe("task.step_completed", self._on_task_active)
        bus.subscribe("task.step_failed", self._on_task_active)

        # Start idle watchdog
        self._running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._idle_watchdog, daemon=True)
        self._thread.start()
        logger.info("[Attention] System started. Initial focus: IDLE")

    def stop(self):
        self._running = False
        if self._thread:
            self._thread.join(timeout=3.0)

    # ── Focus Control ──────────────────────────────────────────────────────

    def set_focus(self, state: str, lock: bool = False):
        """Set the current attention focus state."""
        valid = {"USER", "SYSTEM", "BACKGROUND", "IDLE"}
        if state not in valid:
            logger.warning(f"[Attention] Invalid focus state: {state}")
            return
        with self._lock:
            if self._focus_locked and not lock:
                return  # Respect task lock
            old = self._focus
            self._focus = state
            self._focus_locked = lock
        if old != state:
            logger.info(f"[Attention] Focus changed: {old} → {state}")
            from jarvis.core.world_state import world
            from jarvis.core.event_bus import bus, SystemEvent, EventPriority
            world.update_cognitive_meta(current_attention_focus=state)
            bus.publish(SystemEvent(
                name="attention.focus_changed",
                data={"from": old, "to": state},
                priority=EventPriority.NORMAL,
            ))

    def get_focus(self) -> str:
        with self._lock:
            return self._focus

    def unlock_focus(self):
        """Release task lock so auto-switching can resume."""
        with self._lock:
            self._focus_locked = False

    # ── Event Gate ────────────────────────────────────────────────────────

    def should_process(self, event_priority_value: int) -> bool:
        """
        Returns True if an event of the given priority should be processed
        given the current attention focus.

        EventPriority values: LOW=1, NORMAL=2, HIGH=3, CRITICAL=4
        """
        with self._lock:
            threshold = _THRESHOLDS.get(self._focus, 2)
            return event_priority_value >= threshold

    # ── Activity Tracking ─────────────────────────────────────────────────

    def register_activity(self):
        """Call whenever user activity is detected to reset idle timer."""
        with self._lock:
            self._last_activity = time.time()
        # Upgrade focus if user is present
        if self.get_focus() in ("IDLE", "BACKGROUND"):
            self.set_focus("SYSTEM")

    # ── EventBus Callbacks ────────────────────────────────────────────────

    def _on_user_detected(self, event):
        with self._lock:
            self._last_activity = time.time()
        self.set_focus("USER")

    def _on_user_left(self, event):
        self.set_focus("SYSTEM")

    def _on_voice_activity(self, event):
        self.register_activity()
        if self.get_focus() != "USER":
            self.set_focus("USER")

    def _on_gesture_activity(self, event):
        self.register_activity()

    def _on_task_active(self, event):
        with self._lock:
            self._last_activity = time.time()

    # ── Idle Watchdog ─────────────────────────────────────────────────────

    def _idle_watchdog(self):
        """Gradually downgrade attention focus during periods of inactivity."""
        while self._running:
            time.sleep(15)
            with self._lock:
                if self._focus_locked:
                    continue
                elapsed = time.time() - self._last_activity
                if elapsed > self._idle_timeout * 3:
                    target = "IDLE"
                elif elapsed > self._idle_timeout * 2:
                    target = "BACKGROUND"
                elif elapsed > self._idle_timeout:
                    target = "SYSTEM"
                else:
                    continue

            self.set_focus(target)


# Module-level singleton
attention = AttentionSystem()
