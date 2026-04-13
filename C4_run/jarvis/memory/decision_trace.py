import logging
from typing import List, Dict, Any
import time

logger = logging.getLogger(__name__)

class DecisionTraceLog:
    """
    Explainability Layer for AI Agents.
    Stores the logical "Why" behind every action for user review.
    """
    def __init__(self):
        self._traces: List[Dict[str, Any]] = []

    def log_decision(self, agent_name: str, action: str, justification: str):
        trace = {
            "timestamp": time.time(),
            "agent": agent_name,
            "action": action,
            "justification": justification
        }
        self._traces.append(trace)
        logger.debug(f"[Trace] {agent_name} -> {action} Because: {justification}")
        
    def get_recent_traces(self, limit: int = 5) -> List[Dict[str, Any]]:
        return self._traces[-limit:]

    def explain_last_action(self) -> str:
        if not self._traces:
            return "I have no recent autonomous actions to explain."
        last = self._traces[-1]
        return f"{last['agent']} executed '{last['action']}'. Reason: {last['justification']}"

# Global singleton
decision_trace = DecisionTraceLog()
