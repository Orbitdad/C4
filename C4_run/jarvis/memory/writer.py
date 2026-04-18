"""
Memory Writer for C4.

Post-execution hook that evaluates whether a conversation exchange
should be persisted as structured memory.

Write Rules:
    - System rules              → type="system", importance=1.0
    - User preferences          → type="user",   importance=0.85
    - Successful task outcomes   → type="task",    importance=0.6
    - Greetings / trivial chat  → SKIP
    - Duplicates (sim > 0.92)   → SKIP
"""

from __future__ import annotations

import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from .vector_store import MemoryEntry, VectorMemoryStore

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Patterns for classification
# ---------------------------------------------------------------------------

# Skip patterns — never store these
_SKIP_PATTERNS = [
    r"^(hi|hey|hello|good\s*(morning|afternoon|evening|night)|bye|goodbye|thanks?|thank\s*you|ok|okay|sure|yes|no|yep|nah|nope)\s*[.!?]*$",
    r"^(what'?s?\s*up|how\s*are\s*you|how'?s?\s*it\s*going)\s*[.!?]*$",
    r"^(shut\s*up|be\s*quiet|quiet|stop|cancel|never\s*mind)\s*[.!?]*$",
    r"^c4\s*[.!?]*$",
]
_SKIP_RE = [re.compile(p, re.IGNORECASE) for p in _SKIP_PATTERNS]

# User preference patterns
_USER_PREF_PATTERNS = [
    r"\b(i\s+prefer|always\s+use|my\s+(?:favorite|preferred)|i\s+like|i\s+want\s+you\s+to|remember\s+that\s+i)\b",
    r"\b(my\s+name\s+is|i\s+am\s+called|call\s+me)\b",
    r"\b(i\s+use|i\s+work\s+with|my\s+(?:laptop|phone|email|setup))\b",
]
_USER_PREF_RE = [re.compile(p, re.IGNORECASE) for p in _USER_PREF_PATTERNS]

# System rule patterns
_SYSTEM_RULE_PATTERNS = [
    r"\b(always|never|must|rule|constraint|policy|system\s+rule)\b.*\b(use|do|apply|follow|enforce)\b",
    r"\b(use\s+\w+\s+for\s+(?:all|every|coding|planning|embedding))\b",
]
_SYSTEM_RULE_RE = [re.compile(p, re.IGNORECASE) for p in _SYSTEM_RULE_PATTERNS]

# Task outcome patterns (from assistant response)
_TASK_OUTCOME_PATTERNS = [
    r"\b(created|generated|opened|executed|installed|saved|built|deployed|completed|wrote)\b",
    r"\b(file\s+\w+\.\w+|project|application|script|website)\b",
]
_TASK_OUTCOME_RE = [re.compile(p, re.IGNORECASE) for p in _TASK_OUTCOME_PATTERNS]

# Minimum content length to consider storing
_MIN_CONTENT_LENGTH = 10

# Duplicate similarity threshold
_DEDUP_THRESHOLD = 0.92


# ---------------------------------------------------------------------------
# Writer
# ---------------------------------------------------------------------------

class MemoryWriteDecider:
    """
    Decides whether an exchange should be stored as memory and with what type/importance.
    """

    @staticmethod
    def should_skip(user_input: str, assistant_response: str) -> bool:
        """Return True if this exchange is trivial and should NOT be stored."""
        text = user_input.strip()

        # Too short
        if len(text) < _MIN_CONTENT_LENGTH:
            return True

        # Matches skip patterns
        for pattern in _SKIP_RE:
            if pattern.match(text):
                return True

        return False

    @staticmethod
    def classify(
        user_input: str,
        assistant_response: str,
    ) -> Optional[Tuple[str, str, float]]:
        """
        Classify the exchange into a memory entry.

        Returns:
            (content, type, importance) or None if not worth storing.
        """
        u = user_input.strip()
        a = assistant_response.strip()

        # 1. Check for user preference
        for pattern in _USER_PREF_RE:
            if pattern.search(u):
                content = f"User stated: {u}"
                return (content, "user", 0.85)

        # 2. Check for system rule
        for pattern in _SYSTEM_RULE_RE:
            if pattern.search(u):
                content = f"System rule: {u}"
                return (content, "system", 1.0)

        # 3. Check for task outcome (from response)
        task_signals = sum(1 for p in _TASK_OUTCOME_RE if p.search(a))
        if task_signals >= 1 and len(a) > 20:
            # Summarize the task outcome
            content = f"Task: {u[:100]} → Result: {a[:150]}"
            importance = min(0.7, 0.5 + task_signals * 0.05)
            return (content, "task", importance)

        return None


class MemoryWriter:
    """
    Post-execution hook for the C4 pipeline.
    Evaluates and stores worthy memories after each reasoning cycle.

    Usage:
        writer = MemoryWriter(vector_store)
        writer.maybe_store(user_input="...", assistant_response="...")
    """

    def __init__(self, vector_store: VectorMemoryStore) -> None:
        self.store = vector_store
        self.decider = MemoryWriteDecider()
        self._write_counter = 0

    def maybe_store(
        self,
        user_input: str,
        assistant_response: str,
        force_type: Optional[str] = None,
        force_importance: Optional[float] = None,
    ) -> Optional[MemoryEntry]:
        """
        Evaluate whether the exchange warrants a memory write.

        Args:
            user_input: What the user said
            assistant_response: What C4 responded
            force_type: Override the auto-classified type
            force_importance: Override the auto-assigned importance

        Returns:
            The created MemoryEntry, or None if skipped.
        """
        if not user_input or not user_input.strip():
            return None

        # Step 1: Skip trivial exchanges
        if self.decider.should_skip(user_input, assistant_response):
            logger.debug(f"[MemoryWriter] Skipped trivial: '{user_input[:40]}...'")
            return None

        # Step 2: Classify
        classification = self.decider.classify(user_input, assistant_response)
        if classification is None and force_type is None:
            logger.debug(f"[MemoryWriter] Not worth storing: '{user_input[:40]}...'")
            return None

        if classification:
            content, mem_type, importance = classification
        else:
            content = f"Exchange: {user_input[:100]}"
            mem_type = "task"
            importance = 0.5

        # Apply overrides
        if force_type:
            mem_type = force_type
        if force_importance is not None:
            importance = force_importance

        # Step 3: Deduplication check
        duplicates = self.store.find_duplicates(content, threshold=_DEDUP_THRESHOLD)
        if duplicates:
            logger.debug(
                f"[MemoryWriter] Duplicate detected (sim>{_DEDUP_THRESHOLD}), skipping: '{content[:40]}...'"
            )
            return None

        # Step 4: Store
        entry = self.store.add(content, mem_type=mem_type, importance=importance)
        self._write_counter += 1

        logger.info(
            f"[MemoryWriter] Stored [{entry.type}] (imp={entry.importance:.2f}): "
            f"'{entry.content[:60]}...' (total writes: {self._write_counter})"
        )
        return entry

    def store_direct(
        self,
        content: str,
        mem_type: str = "task",
        importance: float = 0.5,
    ) -> Optional[MemoryEntry]:
        """
        Directly store a memory without classification logic.
        Used for programmatic writes (e.g., system seeds, explicit learning).
        """
        if not content or len(content.strip()) < 5:
            return None

        duplicates = self.store.find_duplicates(content, threshold=_DEDUP_THRESHOLD)
        if duplicates:
            return None

        return self.store.add(content, mem_type=mem_type, importance=importance)
