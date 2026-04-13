import time
import logging
import threading
from collections import deque
from jarvis.core.event_bus import bus, SystemEvent, EventPriority
from jarvis.core.world_state import world
from jarvis.memory.decision_trace import decision_trace

logger = logging.getLogger(__name__)

class TrendForecaster:
    """
    Anticipates future system states using linear regression over time-series data.
    """
    def __init__(self):
        self.cpu_history = deque(maxlen=60) # Last 60 seconds
        self.is_running = False
        self._thread = None
        
    def start(self):
        if self.is_running: return
        self.is_running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info("[Forecaster] Predictive intelligence online.")
        
    def stop(self):
        self.is_running = False
        if self._thread:
            self._thread.join()
            
    def _run_loop(self):
        while self.is_running:
            time.sleep(2)
            state = world.get_snapshot()['system_health']
            cpu = state.get('cpu_percent', 0.0)
            
            if cpu > 0:
                self.cpu_history.append((time.time(), cpu))
                
            if len(self.cpu_history) > 10:
                # Basic linear regression
                x = [t[0] for t in self.cpu_history]
                y = [t[1] for t in self.cpu_history]
                
                # Normalize X to start at 0
                x_0 = x[0]
                x_norm = [xi - x_0 for xi in x]
                
                n = len(x_norm)
                sum_x = sum(x_norm)
                sum_y = sum(y)
                sum_xy = sum(xi*yi for xi, yi in zip(x_norm, y))
                sum_xx = sum(xi*xi for xi in x_norm)
                
                denominator = (n * sum_xx - sum_x * sum_x)
                if denominator != 0:
                    slope = (n * sum_xy - sum_x * sum_y) / denominator
                    
                    # If CPU is rising wildly (> 0.5% per second)
                    if slope > 0.5 and y[-1] > 60:
                        sec_to_95 = (95.0 - y[-1]) / slope if slope > 0 else 999
                        if 0 < sec_to_95 < 120:
                            msg = f"CPU trend detected. Predicted to hit 95% in {int(sec_to_95)}s."
                            logger.warning(f"[Forecaster] {msg}")
                            decision_trace.log_decision("ForecasterAgent", "Publish Warning", msg)
                            bus.publish(SystemEvent(
                                name="sys.forecaster.cpu_critical_predicted",
                                data={"seconds_to_critical": sec_to_95, "current_cpu": y[-1]},
                                priority=EventPriority.HIGH
                            ))
                            # Sleep loop dynamically to not spam
                            time.sleep(30)
                            self.cpu_history.clear()
