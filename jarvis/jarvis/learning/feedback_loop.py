"""
Reinforcement Feedback Loop — connects execution outcomes to the learning system.

Subscribes to task.step_completed and task.step_failed events from the DAGRunner
and the legacy pipeline, logs experiences, and updates behavior weights.
"""
import logging
from collections import defaultdict
from typing import Optional, Dict

from jarvis.core.event_bus import bus, SystemEvent
from jarvis.memory.semantic_db import SemanticDB

logger = logging.getLogger(__name__)


class ReinforcementFeedbackLoop:
    """
    Listens to execution outcomes and feeds them back into the Learning System.

    Experience Loop:
        Step completes → Evaluate success/failure → Update action confidence
        Pattern: repeated failures on X → log warning, suggest alternate method
    """

    def __init__(self, semantic_db: Optional[SemanticDB] = None):
        self.semantic_db = semantic_db

        # Track action outcomes: action_type → {success_count, fail_count}
        self._action_stats: Dict[str, Dict[str, int]] = defaultdict(lambda: {"success": 0, "fail": 0})

        # Subscribe to both DAGRunner events and legacy failure event
        bus.subscribe("task.step_completed", self._on_step_success)
        bus.subscribe("task.step_failed", self._on_step_failed)
        bus.subscribe("ai.learning.failure_caught", self._on_failure)  # legacy

    def _on_step_success(self, event: SystemEvent):
        """Record a successful step and reinforce that action type."""
        step_type = event.data.get("type", "unknown")
        elapsed = event.data.get("elapsed_ms", 0)
        self._action_stats[step_type]["success"] += 1

        total = sum(self._action_stats[step_type].values())
        success_rate = self._action_stats[step_type]["success"] / total

        logger.debug(f"[FeedbackLoop] ✓ {step_type} succeeded ({elapsed:.0f}ms). Rate: {success_rate:.0%}")

        if self.semantic_db and total % 5 == 0:
            # Periodically log performance to semantic memory
            self.semantic_db.add_memory(
                f"Action '{step_type}' has a {success_rate:.0%} success rate over {total} uses.",
                metadata={"type": "performance_log", "action": step_type, "rate": success_rate}
            )

    def _on_step_failed(self, event: SystemEvent):
        """Record a failed step, log the error to semantic DB, suggest alternate."""
        step_type = event.data.get("type", "unknown")
        error = event.data.get("error", "Unknown error")
        retries = event.data.get("retries", 0)
        self._action_stats[step_type]["fail"] += 1

        total = sum(self._action_stats[step_type].values())
        fail_count = self._action_stats[step_type]["fail"]
        fail_rate = fail_count / total

        logger.warning(f"[FeedbackLoop] ✗ {step_type} failed after {retries} retries. Error: {error}")

        # Store failure in semantic memory for future retrieval
        if self.semantic_db:
            mem = f"Action '{step_type}' failed: {error}. (Failure #{fail_count}, rate: {fail_rate:.0%})"
            self.semantic_db.add_memory(mem, metadata={
                "type": "failure_log",
                "action": step_type,
                "error": error,
                "fail_count": fail_count,
            })

        # Alert if repeatedly failing
        if fail_count >= 3 and fail_rate > 0.5:
            logger.error(f"[FeedbackLoop] Action '{step_type}' is failing consistently. Consider alternate method.")
            bus.publish(SystemEvent(
                name="learning.action_unreliable",
                data={"action": step_type, "fail_rate": fail_rate, "fail_count": fail_count},
            ))

    def _on_failure(self, event: SystemEvent):
        """Handle legacy pipeline failure events."""
        failed_node = event.data.get("failed_node", {})
        error = event.data.get("error", "Unknown error")
        step_type = failed_node.get("type", "unknown")
        logger.warning(f"[FeedbackLoop] Legacy failure: {step_type} | {error}")
        if self.semantic_db:
            mem = f"Execution of '{step_type}' with params {failed_node.get('params', {})} failed: {error}"
            self.semantic_db.add_memory(mem, metadata={"type": "failure_log", "node": failed_node})

    def get_stats(self) -> Dict[str, Dict[str, int]]:
        """Return raw action outcome statistics."""
        return dict(self._action_stats)

    def get_success_rate(self, action_type: str) -> float:
        """Return the success rate for a specific action type (0.0-1.0)."""
        stats = self._action_stats.get(action_type, {})
        total = stats.get("success", 0) + stats.get("fail", 0)
        if total == 0:
            return 1.0  # Optimistic default for new actions
        return stats["success"] / total
