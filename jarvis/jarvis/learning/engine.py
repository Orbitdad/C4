"""
Learning engine: fact learning, command learning, feedback, and automation suggestions.
"""

from __future__ import annotations

import re
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from jarvis.logging_utils import get_logger, log_action
from jarvis.memory.manager import MemoryManager
from jarvis.memory.models import (
    ActionStep,
    CommandDefinition,
    Fact,
    FeedbackEntry,
)
from jarvis.learning.patterns import (
    build_pattern_signature,
    detect_repeated_patterns,
    suggest_automation_message,
)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


class LearningEngine:
    """
    Orchestrates fact learning, command learning, feedback recording,
    and automation suggestions. All learning goes through this engine.
    """

    def __init__(self, memory_manager: MemoryManager) -> None:
        self.memory = memory_manager
        self.logger = get_logger(__name__)
        self._execution_log: List[Dict[str, Any]] = []
        self._pending_command: Optional[Dict[str, Any]] = None
        self._learning_history: List[Dict[str, str]] = []  # Added for Undo functionality

    def learn_fact(
        self,
        key: str,
        value: Any,
        category: str = "user",
        tags: Optional[List[str]] = None,
    ) -> Fact:
        """Store a new fact and log the learning event."""
        # Validation Layer
        if not key or not str(value).strip():
            self.logger.warning("[Learning] Rejected empty fact learning.")
            raise ValueError("Fact key and value cannot be empty.")
            
        fact = self.memory.store_fact(
            category=category,
            key=key,
            value=value,
            tags=tags,
            source="user",
        )
        log_action("learn_fact", {"key": key, "value": value, "id": fact.id})
        self.logger.info("Learned fact: %s = %s (%s)", key, value, fact.id)
        self._learning_history.append({"type": "fact", "id": fact.id})
        return fact

    def learn_command(
        self,
        trigger_phrase: str,
        steps: List[ActionStep],
        description: str = "",
        confirmation_required: bool = True,
    ) -> CommandDefinition:
        """Store a new learned command with validation and versioning."""
        name = trigger_phrase.strip().lower()
        
        # Validation Layer
        if not steps:
            raise ValueError("Command must have at least one step.")
        
        # Check for conflicts / Versioning
        existing_commands = self.memory.get_all_commands()
        for cmd in existing_commands:
            if trigger_phrase in cmd.trigger_phrases:
                self.logger.warning(f"[Learning] Command conflict for '{trigger_phrase}'. Auto-versioning.")
                cmd.version = getattr(cmd, "version", 1) + 1  # Assume CommandDefinition supports version
                self.memory.update_command(cmd.id, version=cmd.version)
                # For simplicity, we just update instead of replacing in this phase, or we can overwrite it
                
        cmd_id = f"cmd-{re.sub(r'[^a-z0-9]+', '-', name)}"
        
        cmd = CommandDefinition(
            id=cmd_id,
            name=name,
            trigger_phrases=[trigger_phrase.strip()],
            description=description or f"Executes when user says '{trigger_phrase}'",
            steps=steps,
            confirmation_required=confirmation_required,
        )
        cmd.version = getattr(cmd, "version", 1) # Assign V1
        
        self.memory.add_command(cmd)
        log_action(
            "learn_command",
            {"trigger": trigger_phrase, "steps": len(steps), "id": cmd_id},
        )
        self.logger.info("Learned command: %s (%s)", trigger_phrase, cmd_id)
        self._learning_history.append({"type": "command", "id": cmd_id})
        return cmd

    def record_feedback(
        self,
        utterance: str,
        original_action: Optional[str] = None,
        correction: Optional[str] = None,
        effect: str = "ignored",
    ) -> FeedbackEntry:
        """Record user feedback/correction."""
        entry = FeedbackEntry(
            id=_make_id("fb"),
            utterance=utterance,
            original_action=original_action,
            correction=correction,
            timestamp=_now_iso(),
            effect=effect,
        )
        self.memory.record_feedback(entry)
        log_action("record_feedback", {"utterance": utterance[:50], "effect": effect})
        return entry

    def apply_feedback(
        self,
        utterance: str,
        last_action: Optional[Dict[str, Any]] = None,
        correction_text: Optional[str] = None,
    ) -> Tuple[str, str]:
        """
        Apply feedback: update or disable the last command if applicable.
        Returns (effect, user_message).
        """
        entry = FeedbackEntry(
            id=_make_id("fb"),
            utterance=utterance,
            original_action=str(last_action) if last_action else None,
            correction=correction_text,
            timestamp=_now_iso(),
        )
        effect = "ignored"
        msg = "Understood. I will adjust how I handle that in the future."

        if last_action and correction_text:
            cmd_id = last_action.get("command_id")
            if cmd_id:
                cmd = self.memory.get_command_by_id(cmd_id)
                if cmd and correction_text.lower() in [
                    "open the python project folder",
                    "open python project",
                    "wrong",
                ]:
                    # Could parse correction and update steps; for now just record
                    effect = self.memory.apply_feedback(entry)
                    if effect == "updated_command":
                        msg = "I have updated that command based on your correction."
                    entry.effect = effect
                    self.memory.record_feedback(entry)

        self.memory.record_feedback(entry)
        return effect, msg

    def suggest_automation(
        self,
        execution_log: Optional[List[Dict[str, Any]]] = None,
        min_support: int = 3,
    ) -> Optional[Tuple[str, List[ActionStep], int]]:
        """
        Detect repeated workflow and optionally suggest automation.
        Returns (proposed_name, steps, count) if a pattern found, else None.
        """
        log = execution_log or self._execution_log
        patterns = detect_repeated_patterns(log, min_support=min_support)
        if not patterns:
            return None
        sig, count, steps = patterns[0]
        proposed_name = f"custom_{count}"
        steps_objs = [ActionStep.from_dict(s) for s in steps]
        return (proposed_name, steps_objs, count)

    def explain_knowledge(
        self, query: Optional[str] = None
    ) -> Tuple[List[Fact], List[CommandDefinition]]:
        """Return facts and commands for transparency (e.g. 'What do you remember?')."""
        facts = self.memory.find_facts(query=query)
        facts = [f for f in facts if not f.deleted]
        commands = self.memory.get_all_commands()
        return facts, commands

    def forget_fact(self, key_or_id: str) -> bool:
        """Forget a fact by key or id."""
        facts = self.memory.find_facts()
        for f in facts:
            if f.id == key_or_id or f.key == key_or_id:
                self.memory.delete_fact(f.id)
                log_action("forget_fact", {"id": f.id, "key": f.key})
                return True
        return False

    def delete_command(self, name_or_id: str) -> bool:
        """Delete a learned command by name or id."""
        commands = self.memory.get_all_commands()
        for c in commands:
            if c.id == name_or_id or c.name == name_or_id.lower():
                self.memory.delete_command(c.id)
                log_action("delete_command", {"id": c.id})
                return True
        return False

    def reset_memory(self, categories: Optional[List[str]] = None) -> int:
        """Reset memory. See MemoryManager.reset_memory."""
        count = self.memory.reset_memory(categories)
        log_action("reset_memory", {"categories": categories, "cleared": count})
        return count

    def log_execution(self, action: str, steps: List[Dict[str, Any]]) -> None:
        """Append to execution log for pattern detection."""
        self._execution_log.append({"action": action, "steps": steps})
        if len(self._execution_log) > 100:
            self._execution_log = self._execution_log[-50:]

    def set_pending_command(self, trigger: str, steps: List[ActionStep], desc: str) -> None:
        """Store pending command for confirmation before saving."""
        self._pending_command = {
            "trigger": trigger,
            "steps": steps,
            "description": desc,
        }

    def confirm_pending_command(self) -> Optional[CommandDefinition]:
        """Save pending command if user confirmed."""
        if not self._pending_command:
            return None
        pc = self._pending_command
        self._pending_command = None
        return self.learn_command(
            trigger_phrase=pc["trigger"],
            steps=pc["steps"],
            description=pc["description"],
        )

    def cancel_pending_command(self) -> None:
        """Discard pending command."""
        self._pending_command = None

    def undo_last_learning(self) -> str:
        """Undo the last learned fact or command (Gap 9)."""
        if not self._learning_history:
            return "Nothing to undo."
            
        last_learned = self._learning_history.pop()
        item_type = last_learned["type"]
        item_id = last_learned["id"]
        
        if item_type == "fact":
            self.memory.delete_fact(item_id)
            return "I forgot the last fact you taught me."
        elif item_type == "command":
            self.memory.delete_command(item_id)
            return "I forgot the last command you taught me."
            
        return "Undo failed."
