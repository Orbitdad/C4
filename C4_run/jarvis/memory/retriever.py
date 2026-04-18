"""
Smart Memory Retriever for C4.

On each user input:
1. Embed the query using nomic-embed-text
2. Perform FAISS similarity search (top candidates)
3. Filter by memory type based on task context
4. Rank using: score = (0.5 × similarity) + (0.3 × importance) + (0.2 × recency)
5. Return top 3–5 relevant entries
"""

from __future__ import annotations

import logging
import math
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from .vector_store import MemoryEntry, VectorMemoryStore

logger = logging.getLogger(__name__)

# Keywords that signal user-preference queries
_USER_KEYWORDS = {
    "prefer", "preference", "favorite", "always", "habit",
    "like", "dislike", "my", "mine", "personal", "style",
    "name", "birthday", "email", "laptop", "phone",
}

# Keywords that signal task/working-context queries
_TASK_KEYWORDS = {
    "last", "current", "recent", "previous", "working",
    "doing", "task", "step", "progress", "now", "today",
    "just", "earlier", "before",
}

# Scoring weights
W_SIMILARITY = 0.50
W_IMPORTANCE = 0.30
W_RECENCY    = 0.20

# Recency decay: halves every 24 hours
_RECENCY_HALFLIFE_HOURS = 24.0


def _recency_factor(timestamp_iso: str) -> float:
    """
    Exponential decay based on age of memory.
    Returns a float in [0.0, 1.0] where 1.0 = just now.
    """
    try:
        ts = datetime.fromisoformat(timestamp_iso)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        hours_ago = (datetime.now(timezone.utc) - ts).total_seconds() / 3600.0
        return math.exp(-hours_ago / _RECENCY_HALFLIFE_HOURS)
    except Exception:
        return 0.1  # Very old / unparsable → low recency


def _detect_type_filters(query: str) -> List[str]:
    """
    Heuristically determine which memory types to include based on query content.
    System is always included. User and Task are conditionally added.
    """
    q_lower = query.lower()
    words = set(q_lower.split())

    types = ["system"]  # Always included

    if words & _USER_KEYWORDS:
        types.append("user")
    if words & _TASK_KEYWORDS:
        types.append("task")

    # If nothing specific matched, include everything
    if len(types) == 1:
        types.extend(["user", "task"])

    return types


class MemoryRetriever:
    """
    Retrieves and ranks structured memory entries for prompt injection.

    Usage:
        retriever = MemoryRetriever(vector_store)
        ranked_memories = retriever.retrieve("What model do we use for coding?")
    """

    def __init__(
        self,
        vector_store: VectorMemoryStore,
        max_results: int = 5,
        min_score: float = 0.10,
    ) -> None:
        self.store = vector_store
        self.max_results = max_results
        self.min_score = min_score

    def retrieve(
        self,
        query: str,
        max_results: Optional[int] = None,
        type_filter: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Retrieve the most relevant memory entries for a given query.

        Returns a list of dicts:
            [{"content": str, "type": str, "importance": float,
              "timestamp": str, "score": float}, ...]
        sorted by combined score (descending).
        """
        if not query or not query.strip():
            return []

        cap = max_results or self.max_results
        filters = type_filter or _detect_type_filters(query)

        # Step 1: Similarity search (over-fetch for ranking headroom)
        raw_results = self.store.search(
            query,
            top_k=cap * 3,
            type_filter=filters,
        )

        if not raw_results:
            logger.debug("[MemoryRetriever] No vector matches found.")
            return []

        # Step 2: Rank using weighted score
        scored: List[Tuple[float, MemoryEntry]] = []
        for similarity, entry in raw_results:
            recency = _recency_factor(entry.timestamp)
            combined = (
                W_SIMILARITY * similarity
                + W_IMPORTANCE * entry.importance
                + W_RECENCY * recency
            )
            if combined >= self.min_score:
                scored.append((combined, entry))

        scored.sort(key=lambda x: x[0], reverse=True)

        # Step 3: Deduplicate near-identical content
        seen_content: List[str] = []
        deduplicated: List[Tuple[float, MemoryEntry]] = []
        for score, entry in scored:
            content_lower = entry.content.lower().strip()
            is_dup = False
            for seen in seen_content:
                # Simple substring check for near-duplicates
                if content_lower in seen or seen in content_lower:
                    is_dup = True
                    break
            if not is_dup:
                deduplicated.append((score, entry))
                seen_content.append(content_lower)

        # Step 4: Cap to max results
        final = deduplicated[:cap]

        results = []
        for score, entry in final:
            results.append({
                "content": entry.content,
                "type": entry.type,
                "importance": entry.importance,
                "timestamp": entry.timestamp,
                "score": round(score, 4),
            })

        logger.debug(
            f"[MemoryRetriever] Retrieved {len(results)} memories for query: '{query[:50]}...'"
        )
        return results
