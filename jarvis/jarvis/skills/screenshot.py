"""
JARVIS Skill: Screenshot — capture and save screen.
"""
from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Optional

from jarvis.skills.base import BaseSkill, SkillResult
from jarvis.nlp.schemas import Intent
from jarvis.context import ConversationContext

logger = logging.getLogger(__name__)


class ScreenshotSkill(BaseSkill):
    name = "screenshot"
    triggers = ["screenshot", "capture screen", "take a screenshot", "screen capture", "grab the screen"]

    def can_handle(self, intent: Intent) -> bool:
        return any(t in intent.raw_text.lower() for t in self.triggers)

    def execute(self, intent: Intent, context: Optional[ConversationContext] = None) -> SkillResult:
        try:
            from PIL import ImageGrab
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_dir = os.path.expanduser("~/Pictures/JARVIS_Screenshots")
            os.makedirs(save_dir, exist_ok=True)
            path = os.path.join(save_dir, f"jarvis_screenshot_{ts}.png")
            img = ImageGrab.grab()
            img.save(path)
            logger.info(f"[ScreenshotSkill] Saved to {path}")
            return SkillResult(text=f"Screenshot captured and saved to your Pictures folder, sir.")
        except Exception as e:
            logger.error(f"[ScreenshotSkill] Failed: {e}")
            return SkillResult(text="I was unable to capture the screenshot, sir.")
