"""
Helpers for detecting repeated workflows and proposing automations.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Tuple


def build_pattern_signature(steps: List[Dict[str, Any]]) -> str:
    """Create a hash-like signature from a list of action steps."""
    parts = []
    for s in steps:
        t = s.get("type", "")
        params = sorted(
            (k, str(v)) for k, v in s.items() if k != "type"
        )
        parts.append(f"{t}:{','.join(f'{k}={v}' for k, v in params)}")
    return "|".join(parts)


def detect_repeated_patterns(
    execution_log: List[Dict[str, Any]],
    min_support: int = 3,
) -> List[Tuple[str, int, List[Dict[str, Any]]]]:
    """
    Scan execution log for repeated multi-step sequences.
    Returns list of (signature, count, example_steps).
    """
    if not execution_log or len(execution_log) < min_support:
        return []

    signatures: List[str] = []
    steps_by_sig: Dict[str, List[Dict[str, Any]]] = {}

    for entry in execution_log:
        steps = entry.get("steps", [])
        if len(steps) < 2:
            continue
        sig = build_pattern_signature(steps)
        signatures.append(sig)
        if sig not in steps_by_sig:
            steps_by_sig[sig] = steps

    counter = Counter(signatures)
    results: List[Tuple[str, int, List[Dict[str, Any]]]] = []
    for sig, count in counter.most_common():
        if count >= min_support:
            results.append((sig, count, steps_by_sig.get(sig, [])))
    return results


def suggest_automation_message(
    proposed_name: str, steps_summary: str, count: int
) -> str:
    """Generate a user-facing suggestion for automating a repeated workflow."""
    return (
        f"I notice you often {steps_summary} ({count} times recently). "
        f"Would you like me to save this as a command called '{proposed_name}'?"
    )
