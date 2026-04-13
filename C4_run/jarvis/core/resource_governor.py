import logging
from jarvis.core.world_state import world

logger = logging.getLogger(__name__)

class ResourceGovernor:
    """
    Monitors system load and throttles heavy AI components 
    to prevent the 'assistant' from melting the PC.
    """
    
    def __init__(self, vision_manager=None):
        self.vision_manager = vision_manager
        
    def check_and_throttle(self):
        """Called periodically (e.g., from ContextEngine or Watchdog)"""
        state = world.get_snapshot()
        health = state.get("system_health", {})
        cpu = health.get("cpu_percent", 0.0)
        
        # Inferred activity could be 'gaming'
        env = state.get("user_environment", {})
        activity = env.get("inferred_activity", "")
        
        if cpu > 85.0 or activity == "gaming":
            if self.vision_manager and getattr(self.vision_manager, 'is_running', False):
                # If vision manager doesn't natively support pausing, we might just log
                # We assume a mechanism to pause or reduce FPS is added.
                logger.warning(f"[ResourceGovernor] High load (CPU {cpu}%, Activity: {activity}). Throttling background processes.")
                if hasattr(self.vision_manager, 'pause'):
                    self.vision_manager.pause()
        elif cpu < 50.0 and activity != "gaming":
            if self.vision_manager and hasattr(self.vision_manager, 'resume'):
                logger.info("[ResourceGovernor] System load normalized. Resuming background processes.")
                self.vision_manager.resume()
