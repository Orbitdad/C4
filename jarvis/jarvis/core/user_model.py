"""
UserModel — persistent representation of what JARVIS knows about its user.
Persisted to data/user_model.json and updated continuously at runtime.
"""
from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class UserModel:
    """
    Complete user representation used by the World Model.
    All fields are updated dynamically during a session.
    """
    # Identity
    identity: str = "unknown"              # recognized face ID / speaker ID
    name: str = "Sir"                      # display name
    face_id: Optional[str] = None         # UUID from IdentityManager
    speaker_id: Optional[str] = None      # voice profile match ID

    # Preferences (key-value, learned over time)
    preferences: Dict[str, Any] = field(default_factory=dict)

    # Behavioural habits (timestamped action sequences, last 50)
    habits: List[Dict[str, Any]] = field(default_factory=list)

    # Skill level: 0 (novice) → 10 (expert), affects verbosity of responses
    skill_level: int = 5

    # Emotional state detected from voice / context
    emotion_state: str = "neutral"        # neutral | happy | frustrated | focused | tired
    emotion_confidence: float = 0.0

    # Session tracking
    session_start: float = field(default_factory=time.time)
    total_interactions: int = 0


class UserModelManager:
    """
    Manages loading, updating, and persisting the UserModel.
    Thread-safe singleton wrapper used by WorldState.
    """

    def __init__(self, persist_path: Optional[Path] = None):
        self._lock = threading.RLock()
        self._path = persist_path
        self._model = UserModel()
        if persist_path:
            self._load()

    def _load(self):
        """Load model from disk if it exists."""
        try:
            if self._path and self._path.is_file():
                raw = json.loads(self._path.read_text(encoding="utf-8"))
                # Restore scalar fields only (habits/preferences are dicts/lists)
                for k, v in raw.items():
                    if hasattr(self._model, k) and k not in ("session_start",):
                        setattr(self._model, k, v)
                logger.info(f"[UserModel] Loaded profile for '{self._model.name}'.")
        except Exception as e:
            logger.warning(f"[UserModel] Failed to load profile: {e}. Starting fresh.")

    def save(self):
        """Persist current model to disk."""
        if not self._path:
            return
        try:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                data = asdict(self._model)
            # Don't persist session_start or transient emotion
            data.pop("session_start", None)
            self._path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        except Exception as e:
            logger.error(f"[UserModel] Save failed: {e}")

    def get(self) -> UserModel:
        with self._lock:
            return self._model

    def update(self, **kwargs):
        """Update any field(s) on the model."""
        with self._lock:
            for k, v in kwargs.items():
                if hasattr(self._model, k):
                    setattr(self._model, k, v)
            self._model.total_interactions += kwargs.get("_inc_interactions", 0)

    def set_emotion(self, emotion: str, confidence: float):
        with self._lock:
            self._model.emotion_state = emotion
            self._model.emotion_confidence = confidence

    def set_identity(self, name: str, face_id: Optional[str] = None, speaker_id: Optional[str] = None):
        with self._lock:
            self._model.name = name
            self._model.identity = name.lower()
            if face_id:
                self._model.face_id = face_id
            if speaker_id:
                self._model.speaker_id = speaker_id

    def add_habit(self, action: str, context: Optional[str] = None):
        """Record a user habit (timestamped action)."""
        with self._lock:
            self._model.habits.append({
                "action": action,
                "context": context,
                "ts": time.time()
            })
            # Keep only last 50 habits
            if len(self._model.habits) > 50:
                self._model.habits = self._model.habits[-50:]

    def set_preference(self, key: str, value: Any):
        with self._lock:
            self._model.preferences[key] = value

    def snapshot(self) -> Dict[str, Any]:
        """Return a plain dict snapshot for WorldState / LLM injection."""
        with self._lock:
            m = self._model
            return {
                "name": m.name,
                "identity": m.identity,
                "skill_level": m.skill_level,
                "emotion_state": m.emotion_state,
                "emotion_confidence": round(m.emotion_confidence, 2),
                "preferences": m.preferences,
                "recent_habits": m.habits[-5:],
                "total_interactions": m.total_interactions,
            }


# Module-level singleton (initialized properly in main.py)
_user_model_manager: Optional[UserModelManager] = None


def init_user_model(persist_path: Optional[Path] = None) -> UserModelManager:
    global _user_model_manager
    _user_model_manager = UserModelManager(persist_path)
    return _user_model_manager


def get_user_model() -> Optional[UserModelManager]:
    return _user_model_manager
