"""
File indexing and fast search skill.
"""

from __future__ import annotations
import os
from pathlib import Path
from typing import List

from jarvis.nlp.schemas import Intent, IntentType
from jarvis.skills.base import Skill, SkillResult
from jarvis.context import ConversationContext
from jarvis.nlp.schemas import ActionStep

class FileSearchSkill(Skill):
    def __init__(self) -> None:
        self.common_dirs = [
            Path.home() / "Desktop",
            Path.home() / "Documents",
            Path.home() / "Downloads",
            Path.cwd()
        ]

    @property
    def name(self) -> str:
        return "file_search"

    def can_handle(self, intent: Intent) -> bool:
        return intent.type == IntentType.COMMAND and intent.parsed_action in ("file_search", "find_file", "search_file")

    def execute(self, intent: Intent, context: ConversationContext, **kwargs) -> SkillResult:
        params = intent.params or {}
        query = params.get("query", "").lower()
        if not query:
            return SkillResult(text="What file are you looking for?", success=False)
            
        found = []
        for d in self.common_dirs:
            if not d.exists(): continue
            try:
                for root, dirs, files in os.walk(d):
                    # Limit depth to avoid freezing (2 levels deep from target)
                    depth = root[len(str(d)):].count(os.sep)
                    if depth > 3:
                        dirs.clear() # Prune
                        continue
                        
                    for f in files:
                        if query in f.lower():
                            found.append(os.path.join(root, f))
                            if len(found) >= 4:
                                break
                    if len(found) >= 4:
                        break
            except PermissionError:
                pass
            if len(found) >= 4:
                break
                
        if not found:
            return SkillResult(text=f"I could not find a file matching '{query}' in your common directories.", success=False)
            
        if len(found) == 1:
            return SkillResult(text=f"I found '{query}' located at {found[0]}.", success=True)
            
        paths = "\n".join(found)
        return SkillResult(text=f"I found {len(found)} matches for '{query}':\n{paths}", success=True)
