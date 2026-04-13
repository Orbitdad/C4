"""
System control skill: time, date, open apps, system status.
"""

from __future__ import annotations

from jarvis.execution.executor import Executor
from jarvis.nlp.schemas import Intent, IntentType
from jarvis.skills.base import Skill, SkillResult
from jarvis.context import ConversationContext
from jarvis.nlp.schemas import ActionStep


class SystemControlSkill(Skill):
    def __init__(self, executor: Executor) -> None:
        self.executor = executor

    @property
    def name(self) -> str:
        return "system_control"

    def can_handle(self, intent: Intent) -> bool:
        return intent.type == IntentType.COMMAND and intent.parsed_action in (
            "open_app",
            "tell_time",
            "tell_date",
        )

    def execute(
        self,
        intent: Intent,
        context: ConversationContext,
        **kwargs,
    ) -> SkillResult:
        action = intent.parsed_action
        params = intent.params or {}
        if action == "tell_time":
            r = self.executor._tell_time()
            return SkillResult(text=r["message"], success=r["success"])
        if action == "tell_date":
            r = self.executor._tell_date()
            return SkillResult(text=r["message"], success=r["success"])
        if action == "open_app":
            step = ActionStep(type="open_app", params={"app": params.get("query", ""), "query": params.get("query", "")})
            r = self.executor.execute_step(step)
            return SkillResult(text=r["message"], success=r["success"])
        return SkillResult(text="", success=False)
