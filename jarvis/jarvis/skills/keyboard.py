"""
Keyboard skill: Volume control, tabs, windows, etc.
"""

from __future__ import annotations
import pyautogui
from jarvis.execution.executor import Executor
from jarvis.nlp.schemas import Intent, IntentType
from jarvis.skills.base import Skill, SkillResult
from jarvis.context import ConversationContext


class KeyboardSkill(Skill):
    def __init__(self, executor: Executor) -> None:
        self.executor = executor

    @property
    def name(self) -> str:
        return "keyboard"

    def can_handle(self, intent: Intent) -> bool:
        return intent.type == IntentType.COMMAND and intent.parsed_action in (
            "volume_up",
            "volume_down",
            "mute",
            "new_tab",
            "close_tab",
            "change_tab",
            "full_screen",
            "escape",
        )

    def execute(
        self,
        intent: Intent,
        context: ConversationContext,
        **kwargs,
    ) -> SkillResult:
        action = intent.parsed_action
        if action == "volume_up":
            pyautogui.press("volumeup")
            return SkillResult(text="Increasing volume, sir.")
        if action == "volume_down":
            pyautogui.press("volumedown")
            return SkillResult(text="Decreasing volume, sir.")
        if action == "mute":
            pyautogui.press("volumemute")
            return SkillResult(text="System muted, sir.")
        if action == "new_tab":
            pyautogui.hotkey("ctrl", "t")
            return SkillResult(text="Opening new tab.")
        if action == "close_tab":
            pyautogui.hotkey("ctrl", "w")
            return SkillResult(text="Closing current tab.")
        if action == "change_tab":
            pyautogui.hotkey("ctrl", "tab")
            return SkillResult(text="Switching tab.")
        if action == "full_screen":
            pyautogui.press("f11")
            return SkillResult(text="Switching to full screen.")
        if action == "escape":
            pyautogui.press("esc")
            return SkillResult(text="Escaping.")

        return SkillResult(text="", success=False)
