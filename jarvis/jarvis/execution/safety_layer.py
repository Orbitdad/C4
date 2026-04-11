"""
Safety System — multi-tier risk analysis and confirmation management.

Risk Levels:
  SAFE        — Auto-approved (search, read, tell time)
  LOW         — Auto-approved (open app, manage window)
  MEDIUM      — Logged with warning (create file, run command)
  HIGH        — Requires verbal confirmation (delete, modify system files)
  DESTRUCTIVE — Hard block unless explicitly authorized

Confirmation flow:
  1. JARVIS announces the action verbally
  2. Bus publishes safety.confirmation_required
  3. HUI shows a dialog (optional)
  4. User says "yes/confirm" or "no/cancel"
  5. SafetySandbox resolves the pending event
"""
import threading
from enum import Enum
from typing import Dict, Any, Optional, Set
import logging
from jarvis.core.event_bus import bus, SystemEvent, EventPriority

logger = logging.getLogger(__name__)


class RiskLevel(Enum):
    SAFE        = 1
    LOW         = 2
    MEDIUM      = 3
    HIGH        = 4
    DESTRUCTIVE = 5


# Mapping of executor step types to risk levels
_RISK_MAP: Dict[str, RiskLevel] = {
    "tell_time":       RiskLevel.SAFE,
    "tell_date":       RiskLevel.SAFE,
    "web_search":      RiskLevel.SAFE,
    "read_file":       RiskLevel.SAFE,
    "small_talk":      RiskLevel.SAFE,

    "open_app":        RiskLevel.LOW,
    "open_url":        RiskLevel.LOW,
    "manage_window":   RiskLevel.LOW,
    "play_media":      RiskLevel.LOW,

    "create_file":     RiskLevel.MEDIUM,
    "keyboard_type":   RiskLevel.MEDIUM,
    "keyboard_hotkey": RiskLevel.MEDIUM,
    "mouse_move":      RiskLevel.MEDIUM,
    "mouse_click":     RiskLevel.MEDIUM,

    "run_command":     RiskLevel.HIGH,
    "run_python":      RiskLevel.HIGH,
    "delete_file":     RiskLevel.DESTRUCTIVE,
}

# Default user-configured permission overrides (can be loaded from config)
_ALLOWED_WITHOUT_CONFIRM: Set[str] = {
    "open_app", "open_url", "web_search", "read_file",
    "tell_time", "tell_date", "play_media", "manage_window",
    "create_file",  # Creates are allowed; deletes still need confirm
}


class SafetyAnalyzer:
    """
    Analyzes action requests and returns risk assessment.
    """

    @staticmethod
    def analyze_risk(step_type: str) -> RiskLevel:
        """Return the risk level for a given step type."""
        return _RISK_MAP.get(step_type, RiskLevel.MEDIUM)

    @staticmethod
    def is_auto_approved(step_type: str, risk: Optional[RiskLevel] = None) -> bool:
        """Return True if the action can proceed without asking the user."""
        if risk is None:
            risk = SafetyAnalyzer.analyze_risk(step_type)
        if risk == RiskLevel.SAFE:
            return True
        if risk == RiskLevel.LOW:
            return True
        return step_type in _ALLOWED_WITHOUT_CONFIRM


class SafetySandbox:
    """
    Blocks critical execution steps until the user provides verbal confirmation.
    Publishes safety.confirmation_required event so the HUI can show a dialog.
    """

    def __init__(self, voice_output=None):
        self.voice_output = voice_output
        self.pending_confirmations: Dict[str, threading.Event] = {}
        self.auth_status: Dict[str, bool] = {}

        bus.subscribe("sys.security.awaiting_confirmation", self._on_awaiting_conf)
        bus.subscribe("voice.intent.confirm", self._on_user_confirm)
        bus.subscribe("voice.intent.cancel", self._on_user_cancel)

    def _on_awaiting_conf(self, event: SystemEvent):
        action = event.data.get("action", "this action")
        risk = event.data.get("risk", "high-risk")
        msg = f"This is a {risk} action: {action}. Please say 'confirm' to proceed or 'cancel' to abort."
        logger.warning(f"[Safety] {msg}")
        if self.voice_output:
            self.voice_output.speak(msg)

    def _on_user_confirm(self, event: SystemEvent):
        for node_id, ev in self.pending_confirmations.items():
            self.auth_status[node_id] = True
            ev.set()
        logger.info("[Safety] Authorization granted by user.")

    def _on_user_cancel(self, event: SystemEvent):
        for node_id, ev in self.pending_confirmations.items():
            self.auth_status[node_id] = False
            ev.set()
        logger.info("[Safety] Authorization denied by user.")

    def require_auth(self, node_id: str, action_desc: str, risk: RiskLevel = RiskLevel.HIGH) -> bool:
        """
        Block until the user confirms or cancels.
        Returns True if authorized, False if denied or timed out.
        """
        ev = threading.Event()
        self.pending_confirmations[node_id] = ev

        bus.publish(SystemEvent(
            name="safety.confirmation_required",
            data={
                "node": node_id,
                "action": action_desc,
                "risk": risk.name.lower(),
            },
            priority=EventPriority.CRITICAL
        ))
        # Publish legacy event name for backward compat
        bus.publish(SystemEvent(
            name="sys.security.awaiting_confirmation",
            data={"node": node_id, "action": action_desc, "risk": risk.name.lower()},
            priority=EventPriority.CRITICAL
        ))

        # Block until user responds (max 20s)
        ev.wait(timeout=20.0)

        status = self.auth_status.get(node_id, False)
        self.pending_confirmations.pop(node_id, None)
        self.auth_status.pop(node_id, None)
        return status

    def check_and_gate(self, step_type: str, action_desc: str) -> bool:
        """
        Combined risk check + confirmation gate.
        Returns True if the action should proceed.
        """
        risk = SafetyAnalyzer.analyze_risk(step_type)
        if SafetyAnalyzer.is_auto_approved(step_type, risk):
            return True

        if risk == RiskLevel.DESTRUCTIVE:
            logger.warning(f"[Safety] DESTRUCTIVE action requested: {step_type}")
        return self.require_auth(
            node_id=f"{step_type}_{id(action_desc)}",
            action_desc=action_desc,
            risk=risk,
        )


# Global singleton
sandbox = SafetySandbox()
