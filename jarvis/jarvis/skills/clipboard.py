"""
JARVIS Skill: Clipboard — read/write clipboard.
"JARVIS, what's in my clipboard?"
"JARVIS, copy this to clipboard: hello world"
"""

from __future__ import annotations

import logging
from typing import Optional

from jarvis.skills.base import BaseSkill, SkillResult
from jarvis.nlp.schemas import Intent
from jarvis.context import ConversationContext

logger = logging.getLogger(__name__)


class ClipboardSkill(BaseSkill):
    name = "clipboard"
    triggers = [
        "clipboard", "what's in my clipboard", "read clipboard",
        "copy to clipboard", "paste", "what did i copy"
    ]

    def can_handle(self, intent: Intent) -> bool:
        return any(t in intent.raw_text.lower() for t in self.triggers)

    def execute(self, intent: Intent, context: Optional[ConversationContext] = None) -> SkillResult:
        raw = intent.raw_text.lower()

        if any(t in raw for t in ["what", "read", "show", "what did i"]):
            return self._read_clipboard()

        # Copy something to clipboard
        for marker in ["copy to clipboard:", "copy to clipboard", "save to clipboard:"]:
            idx = raw.find(marker)
            if idx != -1:
                text = intent.raw_text[idx + len(marker):].strip()
                return self._write_clipboard(text)

        return self._read_clipboard()

    def _read_clipboard(self) -> SkillResult:
        try:
            import pyperclip
            content = pyperclip.paste()
            if not content or not content.strip():
                return SkillResult(text="Your clipboard appears to be empty, sir.")
            truncated = content.strip()[:200]
            suffix = "..." if len(content.strip()) > 200 else ""
            return SkillResult(text=f"Your clipboard contains: {truncated}{suffix}")
        except Exception as e:
            logger.error(f"[ClipboardSkill] Read error: {e}")
            return SkillResult(text="I was unable to access the clipboard, sir.")

    def _write_clipboard(self, text: str) -> SkillResult:
        try:
            import pyperclip
            pyperclip.copy(text)
            return SkillResult(text=f"Copied to clipboard, sir.")
        except Exception as e:
            logger.error(f"[ClipboardSkill] Write error: {e}")
            return SkillResult(text="I was unable to write to the clipboard at this time, sir.")
