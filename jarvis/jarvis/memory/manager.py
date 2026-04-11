"""
JSON-backed MemoryManager for persistent storage of facts, commands, and feedback.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import Fact, CommandDefinition, FeedbackEntry, ActionStep


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _make_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


class MemoryManager:
    """
    Manages persistent JSON storage for facts (JSONL), commands (JSON),
    skills/patterns (JSON), feedback (JSONL), and meta (JSON).
    """

    def __init__(self, base_dir: Path) -> None:
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self._facts_path = self.base_dir / "facts.jsonl"
        self._commands_path = self.base_dir / "commands.json"
        self._skills_path = self.base_dir / "skills.json"
        self._feedback_path = self.base_dir / "feedback.jsonl"
        self._meta_path = self.base_dir / "meta.json"
        self._episodes_path = self.base_dir / "episodes.jsonl"
        self.semantic_db = None

    def set_semantic_db(self, db: Any) -> None:
        self.semantic_db = db

    # ---------- Facts ----------

    def store_fact(
        self,
        category: str,
        key: str,
        value: Any,
        tags: Optional[List[str]] = None,
        source: str = "user",
        confidence: float = 0.95,
    ) -> Fact:
        fact = Fact(
            id=_make_id("fact"),
            category=category,
            key=key,
            value=value,
            source=source,
            confidence=confidence,
            created_at=_now_iso(),
            tags=tags or [],
            deleted=False,
        )
        with open(self._facts_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(fact.to_dict(), ensure_ascii=False) + "\n")
            
        if self.semantic_db:
             text_rep = f"Fact: {key} is {value}. Categories: {category}. Tags: {', '.join(tags or [])}"
             self.semantic_db.add_memory(text_rep, metadata={"type": "fact", "fact_id": fact.id})
             
        return fact

    def _read_facts(self) -> List[Fact]:
        if not self._facts_path.is_file():
            return []
        facts = []
        with open(self._facts_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    fct = Fact.from_dict(d)
                    if not fct.deleted:
                        facts.append(fct)
                except json.JSONDecodeError:
                    continue
        return facts

    def find_facts(
        self, query: Optional[str] = None, category: Optional[str] = None
    ) -> List[Fact]:
        facts = self._read_facts()
        if category:
            facts = [f for f in facts if f.category == category]
        
        if not query:
            return facts

        q = query.lower()
        results = []
        
        # 1. Try Semantic DB first if available
        if self.semantic_db:
             semantic_matches = self.semantic_db.search(query, top_k=10)
             matched_fact_ids = [m[1].get("metadata", {}).get("fact_id") for m in semantic_matches if m[1].get("metadata", {}).get("type") == "fact"]
             for f in facts:
                 if f.id in matched_fact_ids:
                      idx = matched_fact_ids.index(f.id)
                      results.append((100 - idx, f)) # Boost score based on semantic ranking
                      
        # 2. Fallback to keyword matching
        for f in facts:
            # Skip if already in results from semantic search
            if any(r[1].id == f.id for r in results):
                continue
                
            score = 0
            key_lower = f.key.lower()
            val_lower = str(f.value).lower()
            
            # Exact match on key
            if q == key_lower:
                score += 10
            # Substring match on key
            elif q in key_lower:
                score += 5
            # Substring match on value
            elif q in val_lower:
                score += 3
            
            if score > 0:
                results.append((score, f))
        
        # Sort by score (descending) and return facts
        results.sort(key=lambda x: x[0], reverse=True)
        return [r[1] for r in results]

    def get_fact_by_id(self, fact_id: str) -> Optional[Fact]:
        if not self._facts_path.is_file():
            return None
        with open(self._facts_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    d = json.loads(line)
                    if d.get("id") == fact_id and not d.get("deleted", False):
                        return Fact.from_dict(d)
                except json.JSONDecodeError:
                    continue
        return None

    def update_fact(self, fact_id: str, **fields: Any) -> Optional[Fact]:
        facts = self._read_facts()
        updated = None
        for f in facts:
            if f.id == fact_id:
                for k, v in fields.items():
                    if hasattr(f, k):
                        setattr(f, k, v)
                updated = f
                break
        if updated:
            self._rewrite_facts(facts)
        return updated

    def delete_fact(self, fact_id: str) -> bool:
        facts = self._read_facts()
        found = False
        for f in facts:
            if f.id == fact_id:
                f.deleted = True
                found = True
                break
        if found:
            self._rewrite_facts(facts)
        return found

    def _rewrite_facts(self, facts: List[Fact]) -> None:
        with open(self._facts_path, "w", encoding="utf-8") as f:
            for fact in facts:
                f.write(json.dumps(fact.to_dict(), ensure_ascii=False) + "\n")

    # ---------- Commands ----------

    def _read_commands_data(self) -> Dict[str, Any]:
        if not self._commands_path.is_file():
            return {"commands": []}
        with open(self._commands_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write_commands_data(self, data: Dict[str, Any]) -> None:
        with open(self._commands_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def add_command(self, command_def: CommandDefinition) -> CommandDefinition:
        if not command_def.created_at:
            command_def.created_at = _now_iso()
        data = self._read_commands_data()
        data["commands"].append(command_def.to_dict())
        self._write_commands_data(data)
        return command_def

    def get_command_by_trigger(self, phrase: str) -> Optional[CommandDefinition]:
        phrase_lower = phrase.lower().strip()
        data = self._read_commands_data()
        for cmd_d in data.get("commands", []):
            cmd = CommandDefinition.from_dict(cmd_d)
            if not cmd.enabled:
                continue
            for trigger in cmd.trigger_phrases:
                if trigger.lower() in phrase_lower or phrase_lower in trigger.lower():
                    return cmd
        return None

    def get_all_commands(self) -> List[CommandDefinition]:
        data = self._read_commands_data()
        return [
            CommandDefinition.from_dict(c)
            for c in data.get("commands", [])
            if c.get("enabled", True)
        ]

    def get_command_by_id(self, cmd_id: str) -> Optional[CommandDefinition]:
        data = self._read_commands_data()
        for c in data.get("commands", []):
            if c.get("id") == cmd_id:
                return CommandDefinition.from_dict(c)
        return None

    def update_command(self, cmd_id: str, **fields: Any) -> Optional[CommandDefinition]:
        data = self._read_commands_data()
        for c in data["commands"]:
            if c.get("id") == cmd_id:
                for k, v in fields.items():
                    c[k] = v
                self._write_commands_data(data)
                return CommandDefinition.from_dict(c)
        return None

    def delete_command(self, cmd_id: str) -> bool:
        data = self._read_commands_data()
        orig_len = len(data["commands"])
        data["commands"] = [c for c in data["commands"] if c.get("id") != cmd_id]
        if len(data["commands"]) < orig_len:
            self._write_commands_data(data)
            return True
        return False

    # ---------- Feedback ----------

    def record_feedback(self, entry: FeedbackEntry) -> FeedbackEntry:
        if not entry.id:
            entry.id = _make_id("fb")
        if not entry.timestamp:
            entry.timestamp = _now_iso()
        with open(self._feedback_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry.to_dict(), ensure_ascii=False) + "\n")
        return entry

    def apply_feedback(
        self, entry: FeedbackEntry
    ) -> str:
        """
        Hook for automatic adjustments from feedback.
        Returns effect: 'updated_command', 'adjusted_fact', 'ignored'.
        """
        # Placeholder: just record; actual application can be extended later
        self.record_feedback(entry)
        return "ignored"

    # ---------- Skills (patterns) ----------

    def _read_skills_data(self) -> Dict[str, Any]:
        if not self._skills_path.is_file():
            return {"skills": []}
        with open(self._skills_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write_skills_data(self, data: Dict[str, Any]) -> None:
        with open(self._skills_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    def add_skill_pattern(
        self,
        pattern_signature: str,
        example_commands: List[str],
        proposed_skill_name: str,
        status: str = "suggested",
    ) -> Dict[str, Any]:
        data = self._read_skills_data()
        skill = {
            "id": _make_id("skill"),
            "pattern_signature": pattern_signature,
            "example_commands": example_commands,
            "proposed_skill_name": proposed_skill_name,
            "status": status,
            "created_at": _now_iso(),
        }
        data.setdefault("skills", []).append(skill)
        self._write_skills_data(data)
        return skill

    # ---------- Meta ----------

    def _read_meta(self) -> Dict[str, Any]:
        if not self._meta_path.is_file():
            return {}
        with open(self._meta_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _write_meta(self, meta: Dict[str, Any]) -> None:
        with open(self._meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)

    def update_meta(self, **fields: Any) -> None:
        meta = self._read_meta()
        meta.update(fields)
        self._write_meta(meta)

    # ---------- Episodic Memory ----------
    def add_episode(self, description: str) -> None:
        episode = {
            "id": _make_id("ep"),
            "timestamp": _now_iso(),
            "description": description
        }
        with open(self._episodes_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(episode, ensure_ascii=False) + "\n")
            
        if self.semantic_db:
             self.semantic_db.add_memory(f"Episode at {episode['timestamp']}: {description}", metadata={"type": "episode", "ep_id": episode['id']})
             
        # Auto-prune Check
        self._episode_counter = getattr(self, "_episode_counter", 0) + 1
        if self._episode_counter % 20 == 0:
            self.prune_episodes()

    def prune_episodes(self, max_limit: int = 100, safe_limit: int = 50) -> None:
        """
        Compresses and trims episodic memory if it grows beyond max_limit, preventing token overflow.
        """
        if not self._episodes_path.is_file():
            return
            
        episodes = []
        with open(self._episodes_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        episodes.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
                        
        if len(episodes) <= max_limit:
            return
            
        to_prune = len(episodes) - safe_limit
        pruned_episodes = episodes[:to_prune]
        kept_episodes = episodes[to_prune:]
        
        with open(self._episodes_path, "w", encoding="utf-8") as f:
            for ep in kept_episodes:
                f.write(json.dumps(ep, ensure_ascii=False) + "\n")
                
        if self.semantic_db:
            text_block = " | ".join(ep["description"] for ep in pruned_episodes[-10:])
            summary = f"Archived {to_prune} interactions. Recent highlights: {text_block}"
            self.semantic_db.add_memory(summary, metadata={"type": "archived_episodes", "count": to_prune})

    def get_recent_episodes(self, limit: int = 10) -> List[str]:
        if not self._episodes_path.is_file():
            return []
        episodes = []
        with open(self._episodes_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        ep = json.loads(line)
                        episodes.append(f"[{ep.get('timestamp')}] {ep.get('description')}")
                    except json.JSONDecodeError:
                        continue
        return episodes[-limit:]

    def get_recent_work_summary(self, limit: int = 3, scan_last: int = 120) -> List[str]:
        """
        Return a short list of recently completed "work" items from episodic memory.

        This is intentionally heuristic: it looks for episodes that sound like
        execution outcomes (created/opened/executed/generated) and skips generic chat.
        """
        if not self._episodes_path.is_file():
            return []

        # Read the last N lines only (episodes can grow over time).
        lines: List[str] = []
        with open(self._episodes_path, "r", encoding="utf-8") as f:
            lines = f.readlines()[-max(1, scan_last):]

        candidates: List[str] = []
        skip_markers = (
            "User asked:",
            "C4 replied:",
            "JARVIS replied:",
            "Session started.",
            "Session ended.",
        )
        work_markers = (
            "Created ",
            "Deleted ",
            "Opened ",
            "Searching for ",
            "Executed.",
            "generated and saved",
            "Python script executed successfully",
            "Successfully performed",
        )

        for raw in reversed(lines):
            raw = raw.strip()
            if not raw:
                continue
            try:
                ep = json.loads(raw)
            except json.JSONDecodeError:
                continue

            desc = (ep.get("description") or "").strip()
            if not desc:
                continue
            if desc.startswith(skip_markers):
                continue

            if any(m in desc for m in work_markers):
                # Keep it short for speech
                cleaned = desc
                cleaned = cleaned.replace("SYSTEM: ", "")
                cleaned = cleaned.replace("C4: ", "")
                cleaned = cleaned.replace("JARVIS: ", "")
                candidates.append(cleaned)

            if len(candidates) >= limit:
                break

        # Return chronological order (oldest -> newest within the selected set)
        return list(reversed(candidates))

    def reset_memory(self, categories: Optional[List[str]] = None) -> int:
        """
        Reset memory. If categories given, only clear facts in those categories.
        Otherwise clear all facts; commands and feedback are preserved unless
        categories includes 'all'.
        Returns number of items cleared.
        """
        count = 0
        if categories is None or "all" in (categories or []):
            # Clear facts file
            if self._facts_path.is_file():
                self._facts_path.write_text("")
                count += 1
            if not categories or "all" in categories:
                # Also clear commands
                self._write_commands_data({"commands": []})
        else:
            facts = self._read_facts()
            keep = [f for f in facts if f.category not in categories]
            removed = len(facts) - len(keep)
            self._rewrite_facts(keep)
            count = removed
        return count
