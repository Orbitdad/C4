"""
Media skill: Play music, screenshots, etc.
"""

from __future__ import annotations
import os
import pyautogui
from jarvis.execution.executor import Executor
from jarvis.nlp.schemas import Intent, IntentType
from jarvis.skills.base import Skill, SkillResult
from jarvis.context import ConversationContext


class MediaSkill(Skill):
    def __init__(self, executor: Executor) -> None:
        self.executor = executor

    @property
    def name(self) -> str:
        return "media"

    def can_handle(self, intent: Intent) -> bool:
        return intent.type == IntentType.COMMAND and intent.parsed_action in (
            "play_music",
            "screenshot",
            "click_photo",
        )

    def execute(
        self,
        intent: Intent,
        context: ConversationContext,
        **kwargs,
    ) -> SkillResult:
        action = intent.parsed_action
        if action == "play_music":
            # Using the existing logic from c4.py
            music_file = r"assets\Vibe Hai.mp3"
            if os.path.exists(music_file):
                os.system(f'start "" "{music_file}"')
                return SkillResult(text="Playing music, sir.")
            return SkillResult(text="I couldn't find the music file, sir.", success=False)

        if action == "screenshot":
            im = pyautogui.screenshot()
            im.save("ss.jpg")
            return SkillResult(text="Screenshot saved as ss.jpg.")

        if action == "click_photo":
            pyautogui.press("super")
            pyautogui.typewrite("camera")
            pyautogui.press("enter")
            # In c4.py there was a separate 'Smile' speak and then enter
            return SkillResult(text="Opening camera. Smile for the photo!")

        return SkillResult(text="", success=False)
