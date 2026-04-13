import logging
import threading
import time
from typing import Any
from jarvis.core.event_bus import bus, SystemEvent
from jarvis.core.world_state import world

logger = logging.getLogger(__name__)

class ActivityInferencer:
    """
    Sub-agent that translates raw window titles and OCR into high-level Activities.
    """
    def __init__(self, llm_client: Any):
        self.llm = llm_client
        self.is_running = False
        self._thread = None
        self._last_state_hash = ""
        
        bus.subscribe("vision.screen.ocr", self._on_ocr)
        
    def _on_ocr(self, event: SystemEvent):
        # Update world state with OCR
        world.update_environment(visible_text=event.data.get("text", ""))

    def start(self):
        if self.is_running: return
        self.is_running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        
    def stop(self):
        self.is_running = False
        if self._thread:
            self._thread.join()

    def _run_loop(self):
        while self.is_running:
            time.sleep(20) # update every 20 seconds to save inference calls
            env = world.get_snapshot()["user_environment"]
            window = env.get("active_window", "")
            ocr = env.get("visible_text", "")
            
            if not window and not ocr:
                continue
                
            current_hash = f"{window}|{ocr[:20] if ocr else ''}"
            if current_hash != self._last_state_hash and self.llm:
                prompt = f"""
You are JARVIS's Activity Inferencer. Based on the following telemetry, classify the user's current high-level activity.
Active Window: {window}
Screen Text Sample: {ocr[:100] if ocr else 'None'}

Respond ONLY with a short descriptive activity phrase. Examples: "Debugging Python Code", "Watching YouTube", "Writing an Email", "Idle on Desktop". Nothing else.
"""
                try:
                    activity = self.llm.generate(prompt).strip()
                    if activity and "<<" not in activity: # basic sanity check
                        world.update_environment(inferred_activity=activity)
                        self._last_state_hash = current_hash
                        logger.info(f"[ActivityInferencer] High-Level Activity Updated: {activity}")
                except Exception as e:
                    logger.debug(f"[ActivityInferencer] LLM inference failed: {e}")
