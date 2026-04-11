import logging
from typing import Optional
from jarvis.core.event_bus import bus, SystemEvent
from jarvis.memory.semantic_db import SemanticDB

logger = logging.getLogger(__name__)

class PolicyOptimizer:
    """
    RL Policy Optimizer.
    Scores strategies in the SemanticDB. 
    Successful execution graphs get positive reinforcement, weighting their future retrieval.
    """
    def __init__(self, semantic_db: Optional[SemanticDB] = None):
        self.semantic_db = semantic_db
        bus.subscribe("ai.learning.success", self._on_success)
        bus.subscribe("ai.learning.failure_caught", self._on_failure)

    def _update_score(self, intent: str, delta: int):
        if not self.semantic_db: return
        mem_str = f"Strategy optimization for intent '{intent}': Score shifted by {delta}"
        self.semantic_db.add_memory(mem_str, metadata={"type": "policy_score", "intent": intent, "delta": delta})
        logger.info(f"[PolicyOptimizer] {mem_str}")

    def _on_success(self, event: SystemEvent):
        intent = event.data.get("intent", "multi-step graph")
        self._update_score(intent, 1)
        
    def _on_failure(self, event: SystemEvent):
        failed_node = event.data.get("failed_node", {})
        self._update_score(str(failed_node.get("type", "unknown")), -1)
