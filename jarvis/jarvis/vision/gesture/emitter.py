"""
jarvis/vision/gesture/emitter.py

Publishes gesture events to the JARVIS Event Bus.
"""

import time
import logging
from jarvis.core.event_bus import bus, SystemEvent, EventPriority

logger = logging.getLogger(__name__)

class GestureEmitter:
    def __init__(self, source: str = "gesture_module"):
        self.source = source
        self.last_action = "NONE"
        self.last_action_time = 0.0
        self.ACTION_COOLDOWN = 0.3 # seconds

    def emit_detection(self, gesture: str, state: str, pos: tuple):
        """
        Emits 'gesture.detected' event.
        """
        x, y, z = pos
        data = {
            "gesture": gesture,
            "state": state,
            "x": round(x, 4),
            "y": round(y, 4),
            "timestamp": time.time()
        }
        
        bus.publish(SystemEvent(
            name="gesture.detected",
            data=data,
            priority=EventPriority.NORMAL
        ))

    def emit_action(self, action: str):
        """
        Emits 'gesture.action' event with rate limiting.
        """
        if action == "NONE" or action == "MOVE":
            return

        now = time.time()
        # Rate limit clicks and one-shot actions, but allow continuous actions like DRAG, ZOOM, and ROTATE
        is_continuous = action == "DRAG" or "ZOOM" in action or "ROTATE" in action
        if not is_continuous and action == self.last_action and (now - self.last_action_time) < self.ACTION_COOLDOWN:
            return

        data = {
            "action": action,
            "source": self.source,
            "timestamp": now
        }
        
        bus.publish(SystemEvent(
            name="gesture.action",
            data=data,
            priority=EventPriority.HIGH
        ))
        
        self.last_action = action
        self.last_action_time = now
        logger.info(f"Gesture Action Emitted: {action}")
