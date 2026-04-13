"""
Data models for persistent memory (facts, commands, feedback).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class Fact:
    """A stored fact with category, key, value, and metadata."""

    id: str
    category: str  # user | system | preference | other
    key: str
    value: Any
    source: str = "user"
    confidence: float = 0.95
    created_at: str = ""
    last_used_at: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    deleted: bool = False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "category": self.category,
            "key": self.key,
            "value": self.value,
            "source": self.source,
            "confidence": self.confidence,
            "created_at": self.created_at,
            "last_used_at": self.last_used_at,
            "tags": self.tags,
            "deleted": self.deleted,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "Fact":
        return cls(
            id=d.get("id", ""),
            category=d.get("category", "other"),
            key=d.get("key", ""),
            value=d.get("value"),
            source=d.get("source", "user"),
            confidence=float(d.get("confidence", 0.95)),
            created_at=d.get("created_at", ""),
            last_used_at=d.get("last_used_at"),
            tags=list(d.get("tags", [])),
            deleted=bool(d.get("deleted", False)),
        )


@dataclass
class ActionStep:
    """A single step in a learned command or plan."""

    type: str  # open_app | open_url | create_file | run_command | play_media | etc.
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {"type": self.type, **self.params}

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "ActionStep":
        type_val = d.get("type", "")
        params = {k: v for k, v in d.items() if k != "type"}
        return cls(type=type_val, params=params)


@dataclass
class CommandDefinition:
    """A learned command (macro) with trigger phrases and steps."""

    id: str
    name: str
    trigger_phrases: List[str]
    description: str
    steps: List[ActionStep]
    confirmation_required: bool = True
    created_at: str = ""
    last_run_at: Optional[str] = None
    run_count: int = 0
    enabled: bool = True

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "name": self.name,
            "trigger_phrases": self.trigger_phrases,
            "description": self.description,
            "steps": [s.to_dict() for s in self.steps],
            "confirmation_required": self.confirmation_required,
            "created_at": self.created_at,
            "last_run_at": self.last_run_at,
            "run_count": self.run_count,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "CommandDefinition":
        steps = [
            ActionStep.from_dict(s) if isinstance(s, dict) else s
            for s in d.get("steps", [])
        ]
        return cls(
            id=d.get("id", ""),
            name=d.get("name", ""),
            trigger_phrases=list(d.get("trigger_phrases", [])),
            description=d.get("description", ""),
            steps=steps,
            confirmation_required=bool(d.get("confirmation_required", True)),
            created_at=d.get("created_at", ""),
            last_run_at=d.get("last_run_at"),
            run_count=int(d.get("run_count", 0)),
            enabled=bool(d.get("enabled", True)),
        )


@dataclass
class FeedbackEntry:
    """A feedback/correction record from the user."""

    id: str
    utterance: str
    original_action: Optional[str] = None
    correction: Optional[str] = None
    timestamp: str = ""
    effect: str = ""  # updated_command | ignored | adjusted_fact
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "utterance": self.utterance,
            "original_action": self.original_action,
            "correction": self.correction,
            "timestamp": self.timestamp,
            "effect": self.effect,
            "notes": self.notes,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "FeedbackEntry":
        return cls(
            id=d.get("id", ""),
            utterance=d.get("utterance", ""),
            original_action=d.get("original_action"),
            correction=d.get("correction"),
            timestamp=d.get("timestamp", ""),
            effect=d.get("effect", ""),
            notes=d.get("notes", ""),
        )
