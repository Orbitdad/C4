"""
Identity Manager: Persistent local face identity database.

Architecture Decision:
    Uses a JSON flat-file at data/identities/identities.json.
    Embeddings are stored as plain Python lists (JSON-serializable) which are
    converted back to np.ndarray on load. This keeps the DB human-readable and
    portable with zero dependencies (no SQLite, no vector DB needed for <100 users).

    For production deployments with >100 users, swap the backend to SQLite +
    numpy .npy files referenced by ID.

Security:
    - Embeddings are stored locally, never transmitted.
    - No raw images are stored — only the mathematical representation.
    - delete_user() removes ALL associated embeddings immediately.
"""

from __future__ import annotations

import json
import logging
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

logger = logging.getLogger(__name__)

# ── Schema ────────────────────────────────────────────────────────────────────
# {
#   "version": 1,
#   "users": {
#     "<uuid>": {
#       "user_id": "<uuid>",
#       "display_name": "Adarsh",
#       "embeddings": [[...512 floats...], ...],   # up to MAX_EMBEDDINGS_PER_USER
#       "registered_at": "2026-03-29T00:00:00Z",
#       "last_seen_at": "2026-03-29T01:00:00Z",
#       "metadata": {}                             # extensible user metadata
#     }
#   }
# }
# ──────────────────────────────────────────────────────────────────────────────

class IdentityManager:
    """
    Manages the known-user face embedding database.
    Thread-safe for concurrent pipeline access.
    """

    DB_VERSION = 1
    MAX_EMBEDDINGS_PER_USER = 8   # Store up to 8 angles per user
    DEFAULT_DB_PATH = Path("data/identities/identities.json")

    def __init__(self, db_path: Optional[Path] = None) -> None:
        self._db_path = Path(db_path) if db_path else self.DEFAULT_DB_PATH
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db: Dict[str, Any] = {"version": self.DB_VERSION, "users": {}}
        self._load()

    # ─── Public API ────────────────────────────────────────────────────────────

    def register_user(
        self,
        display_name: str,
        embedding: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        Register a new user with their first face embedding.

        Args:
            display_name: Human-readable name (e.g., "Adarsh").
            embedding:    512-dim ArcFace embedding from FaceDetector.
            metadata:     Optional dict of extra user info (e.g., permissions).

        Returns:
            user_id: UUID string for the new user.
        """
        user_id = str(uuid.uuid4())
        now = self._now_iso()
        with self._lock:
            self._db["users"][user_id] = {
                "user_id": user_id,
                "display_name": display_name,
                "embeddings": [embedding.tolist()],
                "registered_at": now,
                "last_seen_at": now,
                "metadata": metadata or {},
            }
            self._save()
        logger.info(f"[IdentityManager] Registered new user '{display_name}' → {user_id}")
        return user_id

    def add_embedding(self, user_id: str, embedding: np.ndarray) -> bool:
        """
        Add an additional face embedding for an existing user (multi-angle support).
        Automatically drops the oldest embedding if MAX_EMBEDDINGS_PER_USER is exceeded.
        """
        with self._lock:
            user = self._db["users"].get(user_id)
            if not user:
                logger.warning(f"[IdentityManager] add_embedding: user_id '{user_id}' not found.")
                return False
            embeddings = user["embeddings"]
            if len(embeddings) >= self.MAX_EMBEDDINGS_PER_USER:
                embeddings.pop(0)  # Drop oldest
            embeddings.append(embedding.tolist())
            user["last_seen_at"] = self._now_iso()
            self._save()
        return True

    def match(
        self, probe_embedding: np.ndarray, threshold: float = 0.45
    ) -> Tuple[Optional[str], Optional[str], float]:
        """
        Find the closest matching user for a given probe embedding.

        Returns:
            (user_id, display_name, confidence) if match found.
            (None, None, best_score) if no match above threshold.
        """
        from .recognizer import FaceRecognizer
        recognizer = FaceRecognizer(threshold=threshold)

        with self._lock:
            gallery = [
                (uid, [np.array(e, dtype=np.float32) for e in data["embeddings"]])
                for uid, data in self._db["users"].items()
            ]

        matched_id, confidence = recognizer.compare(probe_embedding, gallery)

        if matched_id:
            with self._lock:
                user = self._db["users"].get(matched_id, {})
                display_name = user.get("display_name", "Unknown")
                # Update last_seen_at on every successful match
                if matched_id in self._db["users"]:
                    self._db["users"][matched_id]["last_seen_at"] = self._now_iso()
                    self._save()
            return matched_id, display_name, confidence

        return None, None, confidence

    def delete_user(self, user_id: str) -> bool:
        """Permanently delete a user and all their embeddings."""
        with self._lock:
            if user_id not in self._db["users"]:
                return False
            name = self._db["users"][user_id].get("display_name", user_id)
            del self._db["users"][user_id]
            self._save()
        logger.info(f"[IdentityManager] Deleted user '{name}' ({user_id})")
        return True

    def list_users(self) -> List[Dict[str, Any]]:
        """Return a sanitized list of all registered users (no embeddings)."""
        with self._lock:
            return [
                {
                    "user_id": uid,
                    "display_name": data["display_name"],
                    "embedding_count": len(data["embeddings"]),
                    "registered_at": data["registered_at"],
                    "last_seen_at": data["last_seen_at"],
                    "metadata": data.get("metadata", {}),
                }
                for uid, data in self._db["users"].items()
            ]

    def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve a single user's profile (without embeddings)."""
        with self._lock:
            user = self._db["users"].get(user_id)
            if not user:
                return None
            return {k: v for k, v in user.items() if k != "embeddings"}

    def update_metadata(self, user_id: str, metadata: Dict[str, Any]) -> bool:
        """Merge new key/value pairs into a user's metadata dict."""
        with self._lock:
            user = self._db["users"].get(user_id)
            if not user:
                return False
            user.setdefault("metadata", {}).update(metadata)
            self._save()
        return True

    @property
    def user_count(self) -> int:
        with self._lock:
            return len(self._db["users"])

    # ─── Persistence ───────────────────────────────────────────────────────────

    def _load(self) -> None:
        if not self._db_path.is_file():
            logger.info("[IdentityManager] No existing DB found. Starting fresh.")
            return
        try:
            with open(self._db_path, "r", encoding="utf-8") as f:
                loaded = json.load(f)
            if loaded.get("version", 0) == self.DB_VERSION:
                self._db = loaded
                count = len(self._db.get("users", {}))
                logger.info(f"[IdentityManager] Loaded {count} user(s) from {self._db_path}")
            else:
                logger.warning("[IdentityManager] DB version mismatch. Starting fresh.")
        except Exception as e:
            logger.error(f"[IdentityManager] Failed to load DB: {e}")

    def _save(self) -> None:
        """Write the in-memory DB to disk atomically via a temp rename."""
        tmp_path = self._db_path.with_suffix(".tmp")
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                json.dump(self._db, f, indent=2, ensure_ascii=False)
            tmp_path.replace(self._db_path)
        except Exception as e:
            logger.error(f"[IdentityManager] Failed to save DB: {e}")
            if tmp_path.exists():
                tmp_path.unlink()

    @staticmethod
    def _now_iso() -> str:
        return datetime.now(timezone.utc).isoformat()
