import time
import queue
import threading
import itertools
from enum import Enum
from dataclasses import dataclass
from typing import Any, Callable, Dict, List
import logging

logger = logging.getLogger(__name__)

# Thread-safe sequence counter used as a tiebreaker in PriorityQueue so that
# SystemEvent objects are never directly compared with < (which dataclass does
# not support by default and would raise a TypeError).
_seq_counter = itertools.count()

class EventPriority(Enum):
    LOW = 1
    NORMAL = 2
    HIGH = 3
    CRITICAL = 4

@dataclass
class SystemEvent:
    name: str # e.g., "sys.cpu.spike", "vision.face.detected"
    data: Any
    priority: EventPriority = EventPriority.NORMAL
    timestamp: float = 0.0

class EventBus:
    """
    Centralized Publish/Subscribe Event Bus.
    Replaces polling architecture with immediate interrupt-driven actions.
    """
    def __init__(self):
        self.subscribers: Dict[str, List[Callable]] = {}
        self.queue: queue.PriorityQueue = queue.PriorityQueue()
        self.is_running = False
        self._thread = None
        
    def subscribe(self, event_name: str, callback: Callable):
        if event_name not in self.subscribers:
            self.subscribers[event_name] = []
        self.subscribers[event_name].append(callback)
        
    def unsubscribe(self, event_name: str, callback: Callable):
        if event_name in self.subscribers and callback in self.subscribers[event_name]:
            self.subscribers[event_name].remove(callback)
        
    def publish(self, event: SystemEvent):
        event.timestamp = time.time()
        # Tuple: (-priority, seq, event)
        # Using a monotonic sequence as the second element means Python never
        # needs to compare two SystemEvent objects when priorities are equal.
        self.queue.put((-event.priority.value, next(_seq_counter), event))
        logger.debug(f"[EventBus] Published Event: {event.name} (Priority {event.priority.name})")
        
    def start(self):
        if self.is_running: return
        self.is_running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("[EventBus] Started.")
        
    def stop(self):
        self.is_running = False
        self.publish(SystemEvent(name="sys.shutdown", data=None, priority=EventPriority.CRITICAL))
        if self._thread:
            self._thread.join(timeout=2.0)
            
    def _run_loop(self):
        while self.is_running:
            try:
                # Block for 1s waiting for event
                item = self.queue.get(timeout=1.0)
                event = item[2]
                
                if event.name == "sys.shutdown":
                    break
                    
                callbacks = self.subscribers.get(event.name, []) + self.subscribers.get("*", [])
                for cb in callbacks:
                    try:
                        # Execution is sequential for now. Could spawn threads for slow callbacks if needed.
                        cb(event)
                    except Exception as e:
                        logger.error(f"[EventBus] Error executing callback for {event.name}: {e}")
                        
                self.queue.task_done()
                
            except queue.Empty:
                pass
            except Exception as e:
                logger.error(f"[EventBus] Unexpected loop error: {e}")

# Global instance
bus = EventBus()
