"""
Web search skill: search and summarize (opens browser, optionally uses LLM for summary).
"""

from __future__ import annotations

from typing import Any
from jarvis.nlp.schemas import Intent, IntentType, ActionStep
from jarvis.skills.base import Skill, SkillResult
from jarvis.context import ConversationContext


class WebSearchSkill(Skill):
    def __init__(self, llm_client: Any, config: dict) -> None:
        self.llm = llm_client
        self.config = config

    @property
    def name(self) -> str:
        return "web_search"

    def can_handle(self, intent: Intent) -> bool:
        return intent.type == IntentType.COMMAND and intent.parsed_action == "web_search"

    def execute(
        self,
        intent: Intent,
        context: ConversationContext,
        executor: Any = None,
        **kwargs,
    ) -> SkillResult:
        params = intent.params or {}
        query = params.get("query", "")
        if executor:
            step = ActionStep(type="web_search", params={"query": query})
            r = executor.execute_step(step)
            return SkillResult(text=r["message"], success=r["success"])
        return SkillResult(text=f"Searching for {query}.", success=True)
