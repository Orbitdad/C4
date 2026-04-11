"""
File operations skill: create, read, delete files.
"""

from __future__ import annotations

from jarvis.execution.executor import Executor
from jarvis.nlp.schemas import Intent, IntentType, ActionStep
from jarvis.skills.base import Skill, SkillResult
from jarvis.context import ConversationContext


class FileOpsSkill(Skill):
    def __init__(self, executor: Executor) -> None:
        self.executor = executor

    @property
    def name(self) -> str:
        return "file_ops"

    def can_handle(self, intent: Intent) -> bool:
        return intent.type == IntentType.COMMAND and intent.parsed_action in (
            "create_file",
            "read_file",
            "delete_file",
        )

    def execute(
        self,
        intent: Intent,
        context: ConversationContext,
        **kwargs,
    ) -> SkillResult:
        action = intent.parsed_action
        params = intent.params or {}
        query = params.get("query", "")
        step = ActionStep(type=action, params={"query": query})
        r = self.executor.execute_step(step)
        return SkillResult(text=r["message"], success=r["success"])
