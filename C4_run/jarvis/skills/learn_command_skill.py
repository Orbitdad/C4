"""
Learn command skill: delegates to LearningEngine for command learning.
"""

from __future__ import annotations

from typing import Any
from jarvis.learning.engine import LearningEngine
from jarvis.nlp.schemas import Intent, IntentType
from jarvis.skills.base import Skill, SkillResult
from jarvis.context import ConversationContext


class LearnCommandSkill(Skill):
    def __init__(self, learning_engine: LearningEngine, llm_client: Any) -> None:
        self.learning = learning_engine
        self.llm = llm_client

    @property
    def name(self) -> str:
        return "learn_command"

    def can_handle(self, intent: Intent) -> bool:
        return intent.type == IntentType.LEARN_COMMAND

    def execute(
        self,
        intent: Intent,
        context: ConversationContext,
        **kwargs,
    ) -> SkillResult:
        # Command learning is handled directly by ReasoningEngine
        return SkillResult(text="", success=True)
