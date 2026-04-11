"""
JARVIS Skill: Weather — fetches current weather via wttr.in (no API key needed).
"""

from __future__ import annotations

import logging
import re
from typing import Any, Optional

from jarvis.skills.base import BaseSkill, SkillResult
from jarvis.nlp.schemas import Intent
from jarvis.context import ConversationContext

logger = logging.getLogger(__name__)


class WeatherSkill(BaseSkill):
    name = "weather"
    triggers = ["weather", "temperature", "how hot", "how cold", "raining", "forecast", "climate"]

    def can_handle(self, intent: Intent) -> bool:
        return any(t in intent.raw_text.lower() for t in self.triggers)

    def execute(self, intent: Intent, context: Optional[ConversationContext] = None) -> SkillResult:
        raw = intent.raw_text.lower()
        # Try to extract city name
        city = self._extract_city(raw)
        return SkillResult(text=self._fetch_weather(city))

    def _extract_city(self, text: str) -> str:
        patterns = [
            r"weather (?:in|at|for) ([a-zA-Z\s]+)",
            r"temperature (?:in|at|for) ([a-zA-Z\s]+)",
            r"forecast (?:in|at|for) ([a-zA-Z\s]+)",
        ]
        for pat in patterns:
            m = re.search(pat, text)
            if m:
                return m.group(1).strip()
        return ""  # Empty = use auto-location

    def _fetch_weather(self, city: str = "") -> str:
        try:
            import urllib.request
            loc = city.replace(" ", "+") if city else ""
            url = f"https://wttr.in/{loc}?format=3"
            with urllib.request.urlopen(url, timeout=5) as resp:
                data = resp.read().decode("utf-8").strip()
            if data:
                return f"Current conditions, sir: {data}"
            return "I was unable to retrieve the weather at this time, sir."
        except Exception as e:
            logger.warning(f"[WeatherSkill] Failed: {e}")
            return "I'm unable to connect to the weather service at the moment, sir."
