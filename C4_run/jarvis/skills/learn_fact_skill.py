"""
Learn fact skill: delegates to LearningEngine for fact learning.
"""

from __future__ import annotations

from jarvis.learning.engine import LearningEngine
from jarvis.nlp.schemas import Intent, IntentType
from jarvis.skills.base import Skill, SkillResult
from jarvis.context import ConversationContext


class LearnFactSkill(Skill):
    def __init__(self, learning_engine: LearningEngine) -> None:
        self.learning = learning_engine

    @property
    def name(self) -> str:
        return "learn_fact"

    def can_handle(self, intent: Intent) -> bool:
        return intent.type == IntentType.LEARN_FACT

    def execute(
        self,
        intent: Intent,
        context: ConversationContext,
        **kwargs,
    ) -> SkillResult:
        # Fact learning is handled directly by ReasoningEngine
        return SkillResult(text="", success=True)
