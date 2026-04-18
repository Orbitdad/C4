"""
Memory Prompt Builder for C4.

Constructs the structured memory injection block for the qwen:14b system prompt.

Format:
    --- RELEVANT MEMORY ---
    • [system] qwen:14b is the primary brain model (score: 0.92)
    • [user] Tech preference: Python, clean code (score: 0.87)
    • [task] Last task: Generated overlay.py (score: 0.71)
    ---

Constraints:
    - Max 10 memory bullets
    - Max ~500 tokens for the memory block
    - Deduplication handled upstream by MemoryRetriever
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# Hard limits
MAX_MEMORY_ENTRIES = 10
MAX_MEMORY_CHARS = 2000  # Rough proxy for ~500 tokens

# Type labels for display
_TYPE_LABELS = {
    "system": "system",
    "user": "user",
    "task": "task",
}


class MemoryPromptBuilder:
    """
    Builds structured prompt blocks from retrieved memory entries.

    Usage:
        builder = MemoryPromptBuilder()
        memory_block = builder.build_memory_block(retrieved_memories)
        full_prompt = builder.build_full_prompt(base_system_prompt, memory_block, user_input)
    """

    def __init__(
        self,
        max_entries: int = MAX_MEMORY_ENTRIES,
        max_chars: int = MAX_MEMORY_CHARS,
    ) -> None:
        self.max_entries = max_entries
        self.max_chars = max_chars

    def build_memory_block(
        self,
        memories: List[Dict[str, Any]],
        include_scores: bool = False,
    ) -> str:
        """
        Format retrieved memories into a bullet-list block for prompt injection.

        Args:
            memories: List of memory dicts from MemoryRetriever.retrieve()
            include_scores: If True, append score to each line (debug mode)

        Returns:
            Formatted string block, or empty string if no memories.
        """
        if not memories:
            return ""

        lines: List[str] = []
        total_chars = 0

        for mem in memories[: self.max_entries]:
            label = _TYPE_LABELS.get(mem.get("type", "task"), "task")
            content = mem.get("content", "").strip()

            # Truncate individual entries that are too long
            if len(content) > 200:
                content = content[:197] + "..."

            if include_scores:
                score = mem.get("score", 0.0)
                line = f"• [{label}] {content} (score: {score:.2f})"
            else:
                line = f"• [{label}] {content}"

            # Check total character budget
            if total_chars + len(line) > self.max_chars:
                break

            lines.append(line)
            total_chars += len(line)

        if not lines:
            return ""

        block = "--- RELEVANT MEMORY ---\n"
        block += "\n".join(lines)
        block += "\n---"

        return block

    def build_full_prompt(
        self,
        base_system_prompt: str,
        memory_block: str,
        user_input: str,
        extra_context: Optional[str] = None,
    ) -> str:
        """
        Assemble the complete structured prompt for qwen:14b.

        Args:
            base_system_prompt: The C4 system identity prompt
            memory_block: Output of build_memory_block()
            user_input: The raw user query/command
            extra_context: Optional additional context (world state, etc.)

        Returns:
            Complete prompt string ready for LLM consumption.
        """
        sections: List[str] = []

        # System identity
        sections.append(base_system_prompt.rstrip())

        # Memory injection
        if memory_block:
            sections.append(memory_block)

        # Extra context (world state, ambient, etc.)
        if extra_context:
            sections.append(extra_context.rstrip())

        # Instructions
        sections.append(
            "INSTRUCTIONS:\n"
            "- Think step-by-step\n"
            "- If the request involves an action, output a structured plan (JSON preferred)\n"
            "- Be concise and precise"
        )

        return "\n\n".join(sections)

    def inject_into_system_prompt(
        self,
        existing_prompt: str,
        memories: List[Dict[str, Any]],
    ) -> str:
        """
        Convenience method: inject memory block into an existing dynamic system prompt.
        Replaces any previous memory block (between --- RELEVANT MEMORY --- markers).
        """
        memory_block = self.build_memory_block(memories)
        if not memory_block:
            return existing_prompt

        # Remove any existing memory block
        cleaned = existing_prompt
        start_marker = "--- RELEVANT MEMORY ---"
        end_marker = "\n---"
        start_idx = cleaned.find(start_marker)
        if start_idx >= 0:
            end_idx = cleaned.find(end_marker, start_idx + len(start_marker))
            if end_idx >= 0:
                cleaned = cleaned[:start_idx] + cleaned[end_idx + len(end_marker):]

        # Insert memory block before the first context section or at the end
        insert_markers = [
            "--- RECENT MEMORY EPISODES ---",
            "--- RELEVANT RETRIEVED KNOWLEDGE ---",
            "--- WORLD STATE ---",
            "--- CURRENT AMBIENT CONTEXT ---",
        ]

        insert_pos = len(cleaned)
        for marker in insert_markers:
            idx = cleaned.find(marker)
            if idx >= 0 and idx < insert_pos:
                insert_pos = idx

        result = cleaned[:insert_pos].rstrip() + "\n\n" + memory_block + "\n\n" + cleaned[insert_pos:].lstrip()
        return result
