"""
JARVIS Skill: Reminder — "Remind me in X minutes/seconds to Y"
Fires a voice + HUI alert at the specified time.
"""

from __future__ import annotations

import logging
import re
import threading
import time
from typing import Any, Optional

from jarvis.skills.base import BaseSkill, SkillResult
from jarvis.nlp.schemas import Intent
from jarvis.context import ConversationContext

logger = logging.getLogger(__name__)


class ReminderSkill(BaseSkill):
    name = "reminder"
    triggers = ["remind me", "reminder", "set an alarm", "alert me", "notify me in"]

    def __init__(self, voice_output: Optional[Any] = None, hui_window: Optional[Any] = None):
        self._voice = voice_output
        self._hui = hui_window
        self._reminders: list = []

    def set_voice(self, voice_output: Any) -> None:
        self._voice = voice_output

    def set_hui(self, hui_window: Any) -> None:
        self._hui = hui_window

    def can_handle(self, intent: Intent) -> bool:
        return any(t in intent.raw_text.lower() for t in self.triggers)

    def execute(self, intent: Intent, context: Optional[ConversationContext] = None) -> SkillResult:
        raw = intent.raw_text
        seconds, message = self._parse_reminder(raw)
        if seconds is None:
            return SkillResult(text="I couldn't determine the time for your reminder, sir. Please say something like 'remind me in 5 minutes to check the build'.")

        threading.Thread(
            target=self._fire_reminder,
            args=(seconds, message),
            daemon=True
        ).start()

        time_str = self._format_time(seconds)
        return SkillResult(text=f"Noted, sir. I'll remind you in {time_str}{f' — {message}' if message else ''}.")

    def _parse_reminder(self, text: str):
        text_lower = text.lower()
        total_seconds = 0

        patterns = [
            (r"(\d+)\s*hour", 3600),
            (r"(\d+)\s*minute", 60),
            (r"(\d+)\s*second", 1),
        ]
        for pattern, mult in patterns:
            m = re.search(pattern, text_lower)
            if m:
                total_seconds += int(m.group(1)) * mult

        if total_seconds == 0:
            return None, None

        # Extract the reminder message
        msg = ""
        for trigger in ["to ", "about ", "that "]:
            idx = text_lower.rfind(trigger)
            if idx != -1:
                raw_msg = text[idx + len(trigger):].strip()
                if raw_msg and not any(u in raw_msg.lower() for u in ["minute", "second", "hour"]):
                    msg = raw_msg
                    break

        return total_seconds, msg

    def _format_time(self, seconds: int) -> str:
        if seconds >= 3600:
            h = seconds // 3600
            m = (seconds % 3600) // 60
            return f"{h} hour{'s' if h > 1 else ''}" + (f" {m} minutes" if m else "")
        elif seconds >= 60:
            m = seconds // 60
            return f"{m} minute{'s' if m > 1 else ''}"
        else:
            return f"{seconds} second{'s' if seconds > 1 else ''}"

    def _fire_reminder(self, seconds: int, message: str):
        time.sleep(seconds)
        reminder_text = f"Sir, your reminder" + (f": {message}" if message else " has activated.")
        logger.info(f"[ReminderSkill] Firing reminder: {reminder_text}")
        if self._voice:
            try:
                self._voice.stop()
                self._voice.speak(reminder_text)
            except Exception as e:
                logger.error(f"[ReminderSkill] Voice error: {e}")
        if self._hui:
            try:
                self._hui.signals.alert_critical.emit(f"REMINDER: {message or 'Your reminder has activated.'}")
                self._hui.signals.log_message.emit(f"[REMINDER] {reminder_text}")
            except Exception:
                pass
