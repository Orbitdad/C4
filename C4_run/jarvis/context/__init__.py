from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any


@dataclass
class Turn:
    user_text: str
    assistant_text: str | None = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationContext:
    """
    Short-term conversational context and last action metadata.
    """

    turns: List[Turn] = field(default_factory=list)
    last_action: Optional[Dict[str, Any]] = None

    def add_turn(self, user_text: str, assistant_text: str | None = None) -> None:
        self.turns.append(Turn(user_text=user_text, assistant_text=assistant_text))

    def last_user_utterance(self) -> Optional[str]:
        return self.turns[-1].user_text if self.turns else None

    def to_prompt_history(self, max_turns: int = 8) -> List[Dict[str, str]]:
        recent = self.turns[-max_turns:]
        messages: List[Dict[str, str]] = []
        for t in recent:
            messages.append({"role": "user", "content": t.user_text})
            if t.assistant_text:
                messages.append({"role": "assistant", "content": t.assistant_text})
        return messages

