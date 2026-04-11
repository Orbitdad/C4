"""
Skill modules for JARVIS capabilities.
"""

from .base import Skill, SkillResult, SkillManager
from .small_talk import SmallTalkSkill
from .system_control import SystemControlSkill
from .file_ops import FileOpsSkill
from .web_search import WebSearchSkill
from .learn_fact_skill import LearnFactSkill
from .learn_command_skill import LearnCommandSkill

__all__ = [
    "Skill",
    "SkillResult",
    "SkillManager",
    "SmallTalkSkill",
    "SystemControlSkill",
    "FileOpsSkill",
    "WebSearchSkill",
    "LearnFactSkill",
    "LearnCommandSkill",
]
