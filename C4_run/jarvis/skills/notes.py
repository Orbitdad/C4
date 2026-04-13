"""
JARVIS Skill: Notes — append voice notes to a markdown file.
"JARVIS, make a note: buy groceries tomorrow"
"JARVIS, what are my notes?"
"""

from __future__ import annotations

import logging
import os
import re
from datetime import datetime
from typing import Optional

from jarvis.skills.base import BaseSkill, SkillResult
from jarvis.nlp.schemas import Intent
from jarvis.context import ConversationContext

logger = logging.getLogger(__name__)

NOTES_DIR = os.path.expanduser("~/Documents/JARVIS_Notes")
NOTES_FILE = os.path.join(NOTES_DIR, "jarvis_notes.md")


class NotesSkill(BaseSkill):
    name = "notes"
    triggers = [
        "make a note", "take a note", "note that", "add a note",
        "save a note", "my notes", "show my notes", "read my notes",
        "what are my notes", "show notes"
    ]

    def can_handle(self, intent: Intent) -> bool:
        return any(t in intent.raw_text.lower() for t in self.triggers)

    def execute(self, intent: Intent, context: Optional[ConversationContext] = None) -> SkillResult:
        raw = intent.raw_text.lower()

        if any(t in raw for t in ["show", "read", "what are", "list"]):
            return self._read_notes()

        # Extract note content
        content = intent.raw_text
        for trigger in ["make a note:", "take a note:", "note that", "add a note:", "save a note:", "make a note"]:
            idx = content.lower().find(trigger)
            if idx != -1:
                content = content[idx + len(trigger):].strip().lstrip(":")
                break

        if not content.strip():
            return SkillResult(text="What would you like me to note, sir?")

        return self._add_note(content.strip())

    def _add_note(self, content: str) -> SkillResult:
        try:
            os.makedirs(NOTES_DIR, exist_ok=True)
            ts = datetime.now().strftime("%Y-%m-%d %H:%M")
            entry = f"\n## {ts}\n{content}\n"
            with open(NOTES_FILE, "a", encoding="utf-8") as f:
                f.write(entry)
            logger.info(f"[NotesSkill] Note saved: {content[:50]}")
            return SkillResult(text=f"Noted, sir. I've added that to your notes.")
        except Exception as e:
            logger.error(f"[NotesSkill] Save failed: {e}")
            return SkillResult(text="I was unable to save that note, sir.")

    def _read_notes(self) -> SkillResult:
        try:
            if not os.path.exists(NOTES_FILE):
                return SkillResult(text="You have no notes yet, sir.")
            with open(NOTES_FILE, "r", encoding="utf-8") as f:
                content = f.read().strip()
            # Count and summarize
            entries = [l.strip() for l in content.splitlines() if l.strip() and not l.startswith("#")]
            if not entries:
                return SkillResult(text="Your notes file is empty, sir.")
            count = content.count("##")
            recent = entries[-3:]
            summary = ". ".join(recent[:3])
            return SkillResult(text=f"You have {count} note{'s' if count != 1 else ''}, sir. Most recent: {summary}")
        except Exception as e:
            logger.error(f"[NotesSkill] Read failed: {e}")
            return SkillResult(text="I couldn't retrieve your notes at the moment, sir.")
