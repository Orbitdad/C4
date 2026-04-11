"""
Intent schemas and planning structures.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class IntentType(str, Enum):
    COMMAND = "command"
    QUESTION = "question"
    SMALL_TALK = "small_talk"
    LEARN_FACT = "learn_fact"
    LEARN_COMMAND = "learn_command"
    RUN_LEARNED_COMMAND = "run_learned_command"
    FEEDBACK = "feedback"
    MEMORY_QUERY = "memory_query"
    CONTROL = "control"
    WRITE_CODE = "write_code"
    WRITE_CODE_ACTIVE_WINDOW = "write_code_active_window"
    EDIT_CODE_ACTIVE_WINDOW = "edit_code_active_window"
    PASTE_CODE = "paste_code"
    SENSE_GAP = "sense_gap"   # Phase 10: capability gap — trigger SkillSynthesizer
    UNKNOWN = "unknown"


@dataclass
class ActionStep:
    """A single executable step in a plan."""

    type: str
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type, **self.params}


@dataclass
class Plan:
    """A multi-step execution plan."""

    steps: List[ActionStep] = field(default_factory=list)
    confirmation_required: bool = False
    summary: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "steps": [s.to_dict() for s in self.steps],
            "confirmation_required": self.confirmation_required,
            "summary": self.summary,
        }


@dataclass
class Intent:
    """Parsed user intent with metadata."""

    type: IntentType
    raw_text: str
    parsed_action: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    needs_clarification: bool = False
    confidence: float = 1.0

    @property
    def is_learning(self) -> bool:
        return self.type in (
            IntentType.LEARN_FACT,
            IntentType.LEARN_COMMAND,
        )

    @property
    def is_feedback(self) -> bool:
        return self.type == IntentType.FEEDBACK

