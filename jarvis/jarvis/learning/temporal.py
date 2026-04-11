import time
import logging
import threading
from collections import defaultdict
from typing import Dict, List, Any
from jarvis.core.event_bus import bus, SystemEvent
from jarvis.memory.semantic_db import SemanticDB

logger = logging.getLogger(__name__)

class TemporalHabitEngine:
    """
    Time-Series Analytics.
    Detects behavioral patterns bucketed by time of day.
    """
    def __init__(self, semantic_db: SemanticDB):
        self.semantic_db = semantic_db
        self.habits: Dict[int, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        
        bus.subscribe("action.completed", self._on_action)
        
    def _on_action(self, event: SystemEvent):
        step = event.data.get("step", {})
        action_type = step.get("type")
        if not action_type: return
        
        # Bucket by local hour (0-23)
        current_hour = time.localtime(event.timestamp).tm_hour
        
        # Increment frequency
        action_sig = f"{action_type}:{step.get('params', {}).get('app', '')}"
        self.habits[current_hour][action_sig] += 1
        
        logger.debug(f"[Temporal] Logged {action_sig} at {current_hour}:00")

    def get_likely_habits(self) -> List[str]:
        current_hour = time.localtime().tm_hour
        hour_habits = self.habits.get(current_hour, {})
        if not hour_habits:
            return []
            
        # Sort by frequency
        sorted_habits = sorted(hour_habits.items(), key=lambda x: x[1], reverse=True)
        return [h[0] for h in sorted_habits if h[1] > 2] # Must have happened at least 3 times
