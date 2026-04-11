import time
import threading
import logging
from typing import List, Protocol
from .event_bus import bus, SystemEvent, EventPriority

logger = logging.getLogger(__name__)

class MonitorableService(Protocol):
    is_running: bool
    def start(self) -> None: ...
    def stop(self) -> None: ...
    
class Watchdog:
    """
    Self-Monitoring System Health Service.
    Checks child components periodically. Auto-restarts them if they fail.
    Publishes critical events to the bus on failure.
    """
    def __init__(self, services: List[tuple[str, MonitorableService]]):
        self.services = services # [(name, service_instance), ...]
        self.is_running = False
        self._thread = None
        
    def register_service(self, name: str, service: MonitorableService):
        self.services.append((name, service))
        
    def start(self):
        if self.is_running: return
        self.is_running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("[Watchdog] Started continuous self-monitoring.")
        
    def stop(self):
        self.is_running = False
        if self._thread:
            self._thread.join()
            
    def _run_loop(self):
        # Wait a bit before first check to let everything start
        time.sleep(5)
        while self.is_running:
            time.sleep(10) # check every 10s
            for name, svc in self.services:
                # Need to handle case where svc was explicitly stopped, but for now we assume all registered
                # services should be running 24/7.
                if not getattr(svc, "is_running", False):
                    # Found dead service
                    bus.publish(SystemEvent(
                        name="sys.watchdog.error",
                        data={"service": name, "status": "dead"},
                        priority=EventPriority.CRITICAL
                    ))
                    logger.warning(f"[Watchdog] Restarting dead service: {name}")
                    try:
                        svc.start()
                        bus.publish(SystemEvent(
                            name="sys.watchdog.recovered",
                            data={"service": name, "status": "restarted"},
                            priority=EventPriority.NORMAL
                        ))
                    except Exception as e:
                        logger.error(f"[Watchdog] Failed to restart {name}: {e}")
