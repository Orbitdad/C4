"""
JARVIS Self-Learning Voice Assistant.

This package wires together:
- Voice input/output (STT/TTS)
- Intent parsing & planning
- Learning & memory (facts, commands, feedback)
- Reasoning & skills
- Safe execution layer
"""

from .main import run

__all__ = ["run"]

