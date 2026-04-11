"""
Optional wake word support. Stub for now - can be extended with Porcupine or similar.
"""

from __future__ import annotations

import re
from typing import Optional


def _normalize(text: str) -> str:
    t = (text or "").strip().lower()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def contains_wake_word(transcript: str, wake_word: str = "c4") -> bool:
    """
    Return True if the transcript contains the wake word.

    Notes:
    - STT often returns "c four" / "see four" / "sea four" for "C4".
    - We treat these as equivalent.
    """
    t = _normalize(transcript)
    w = _normalize(wake_word)
    if not t or not w:
        return False

    # Common spoken variants
    variants = {w}
    if w in {"c4", "c 4"}:
        variants.update({"c4", "c 4", "c four", "see four", "sea four", "c-4", "cplusplus"})

    return any(v in t for v in variants)


def strip_wake_word(transcript: str, wake_word: str = "c4") -> str:
    """Remove the wake word from the transcript (best-effort)."""
    t = (transcript or "").strip()
    if not t:
        return ""
    if not contains_wake_word(t, wake_word=wake_word):
        return t
    # Remove common variants case-insensitively
    pattern = r"\b(c4|c\s*4|c\s*four|see\s*four|sea\s*four|c-4|cplusplus)\b"
    cleaned = re.sub(pattern, " ", t, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s+", " ", cleaned).strip(" ,.-")
    return cleaned


def wait_for_wake_word(timeout: Optional[float] = None) -> bool:
    """
    Wait for wake word or activation. Currently a no-op placeholder.
    In a full implementation, this would listen for a wake word (e.g. "C4")
    before the main loop starts listening for commands.
    Returns True when activated (or immediately for now).
    """
    return True
