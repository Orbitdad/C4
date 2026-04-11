"""
Small talk skill: greetings, identity, capabilities.
"""

from __future__ import annotations

from jarvis.nlp.schemas import Intent, IntentType
from jarvis.skills.base import Skill, SkillResult
from jarvis.context import ConversationContext


class SmallTalkSkill(Skill):
    def __init__(self, config: dict) -> None:
        self.config = config
        self.name_override = (config.get("app") or {}).get("name", "JARVIS")

    @property
    def name(self) -> str:
        return "small_talk"

    def can_handle(self, intent: Intent) -> bool:
        return intent.type == IntentType.SMALL_TALK

    def execute(
        self,
        intent: Intent,
        context: ConversationContext,
        **kwargs,
    ) -> SkillResult:
        params = intent.params or {}
        if params.get("greeting"):
            return SkillResult(text="Hello. How may I assist you?")
        if params.get("identity"):
            return SkillResult(
                text=f"I am {self.name_override}, your voice assistant. I can execute commands, answer questions, and learn from you."
            )
        if params.get("capabilities"):
            return SkillResult(
                text="I can open applications, manage files, search the web, tell time and date, "
                "learn facts and commands from you, and answer questions. Say 'what can you do' for more."
            )
        return SkillResult(text="How may I help you?")
