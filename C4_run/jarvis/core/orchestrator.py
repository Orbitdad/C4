import logging
from typing import Any
from jarvis.core.event_bus import bus, SystemEvent
from jarvis.core.world_state import world
from jarvis.core.trust_model import TrustHierarchy, TrustLevel
import uuid

logger = logging.getLogger(__name__)

class CognitiveOrchestrator:
    """
    Central Brain Controller.
    Validates all actions, resolves conflicts, and applies safety rules.
    """
    def __init__(self, executor: Any):
        self.executor = executor
        bus.subscribe("action.request", self._on_action_request)
        bus.subscribe("vision.gesture.fist", self._on_fist_gesture)
        bus.subscribe("vision.gesture.pinch", self._on_pinch_gesture)
        
    def _on_fist_gesture(self, event: SystemEvent):
        """Handle FIST gesture to pause or suspend operations."""
        logger.info("[Orchestrator] Gesture received: FIST. Pausing autonomous systems.")
        world.update_environment(inferred_activity="paused")
        
    def _on_pinch_gesture(self, event: SystemEvent):
        """Handle PINCH gesture."""
        pos = event.data.get("position", {})
        logger.info(f"[Orchestrator] Gesture received: PINCH at {pos}. Interpreting as action intent.")
        # If we had a context of what's at that position, we could execute an action.
        
    def _on_action_request(self, event: SystemEvent):
        """Handle incoming requests to perform an action."""
        action_data = event.data
        step_type = action_data.get("type", getattr(action_data, "type", None))
        
        if not step_type:
            logger.error("[Orchestrator] Action request missing type.")
            return
            
        # 1. Trust Check
        trust_level = TrustHierarchy.get_action_level(step_type)
        if trust_level >= TrustLevel.DESTRUCTIVE:
            # In a full UI, we would trigger a confirmation prompt here.
            # Using dry_run flag of the executor if it's too dangerous without confirmation
            if getattr(self.executor, 'confirm_deletes', True) and step_type == "delete_file":
                logger.warning(f"[Orchestrator] High risk action requested: {step_type}. Ensuring confirmation is enforced.")
            else:
                logger.info(f"[Orchestrator] Authorized high-risk action: {step_type}.")
        
        # 2. Conflict Resolution / Throttling
        state = world.get_snapshot()
        active_tasks = state.get("active_tasks", [])
        if len(active_tasks) > 3:
            logger.warning("[Orchestrator] Multiple tasks executing. Delaying action to avoid conflicts.")
            if event.priority.value < 3: # If not HIGH priority
                return

        # 3. Safe Execution via Executor
        task_id = str(uuid.uuid4())
        world.add_active_task(task_id, f"Executing {step_type}")
        
        try:
            from jarvis.core.recovery import ErrorRecoveryLayer
            
            # Use recovery layer to execute with retries if it's reversible or safe
            if trust_level <= TrustLevel.REVERSIBLE:
                result = ErrorRecoveryLayer.execute_with_retry(self.executor.execute_step, max_retries=2, step=action_data)
            else:
                result = self.executor.execute_step(action_data)
                
            if isinstance(result, dict) and not result.get("success"):
                logger.error(f"[Orchestrator] Execution failed: {result.get('message')}")
            else:
                logger.info(f"[Orchestrator] Successfully executed {step_type}.")
                
        finally:
            world.complete_active_task(task_id)
