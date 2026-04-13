"""
Skill base class. New skills inherit from BaseSkill which has sensible defaults.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from jarvis.context import ConversationContext
from jarvis.nlp.schemas import Intent, IntentType


@dataclass
class SkillResult:
    text: str
    success: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    output_schema: Dict[str, Any] = field(default_factory=dict)


class Skill(ABC):
    """Base class for a JARVIS skill."""

    @property
    @abstractmethod
    def name(self) -> str:
        pass

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {}

    def describe(self) -> str:
        return self.name

    def can_handle(self, intent: Intent) -> bool:
        return False

    def setup(self) -> None:
        pass

    def teardown(self) -> None:
        pass

    @abstractmethod
    def execute(
        self,
        intent: Intent,
        context: ConversationContext,
        **kwargs: Any,
    ) -> SkillResult:
        pass


class BaseSkill(Skill):
    """
    Convenience base: subclasses declare `name` as a class attribute
    and `triggers` as a list of keyword strings for `can_handle`.
    """
    name: str = ""
    triggers: List[str] = []

    def can_handle(self, intent: Intent) -> bool:
        if not self.triggers:
            return False
        return any(t in intent.raw_text.lower() for t in self.triggers)

    def execute(self, intent: Intent, context: Optional[ConversationContext] = None, **kwargs) -> SkillResult:
        return SkillResult(text="Not implemented.", success=False)


class SkillManager:
    """Registry and executor for skills."""

    # Map parsed_action → skill name for direct routing
    ACTION_SKILL_MAP: Dict[str, str] = {
        "weather": "weather",
        "reminder": "reminder",
        "volume": "volume_control",
        "screenshot": "screenshot",
        "notes": "notes",
        "clipboard": "clipboard",
        "model_viewer": "model_viewer",
    }

    def __init__(self, skills: List[Skill]) -> None:
        self.skills: Dict[str, Skill] = {}
        for s in skills:
            s.setup()
            self.skills[s.name] = s

    def execute(
        self,
        skill_name: str,
        intent: Intent,
        context: Optional[ConversationContext] = None,
        **kwargs: Any,
    ) -> SkillResult:
        skill = self.skills.get(skill_name)
        if not skill:
            return SkillResult(text="", success=False)
        return skill.execute(intent, context, **kwargs)

    def execute_by_intent(
        self,
        intent: Intent,
        context: Optional[ConversationContext] = None,
    ) -> SkillResult:
        # 1. Direct action routing (fastest path)
        action = getattr(intent, "parsed_action", "") or ""
        if action in self.ACTION_SKILL_MAP:
            skill_name = self.ACTION_SKILL_MAP[action]
            skill = self.skills.get(skill_name)
            if skill:
                try:
                    return skill.execute(intent, context)
                except Exception as e:
                    return SkillResult(text=f"Skill error: {e}", success=False)

        # 2. can_handle fallback
        for skill in self.skills.values():
            if skill.can_handle(intent):
                try:
                    return skill.execute(intent, context)
                except Exception as e:
                    return SkillResult(
                        text=f"I encountered an issue with that, sir.",
                        success=False,
                        metadata={"error": str(e)},
                    )
        return SkillResult(text="No skill could handle that.", success=False)

    def teardown_all(self):
        for skill in self.skills.values():
            skill.teardown()
