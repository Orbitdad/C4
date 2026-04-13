"""
Intent parser: classifies utterances into COMMAND, QUESTION, LEARN_FACT, etc.
"""

from __future__ import annotations

import re
from typing import Optional

from jarvis.context import ConversationContext
from jarvis.memory.manager import MemoryManager
from jarvis.nlp.schemas import Intent, IntentType


# Phrases indicating feedback/correction
FEEDBACK_PHRASES = [
    "no that's not what i meant",
    "that's not what i meant",
    "do it this way instead",
    "that was wrong",
    "no i meant",
    "actually i meant",
    "correction",
    "wrong",
    "no ",
    "not that",
]

# Phrases for memory control
FORGET_PHRASES = ["forget that", "delete that fact", "forget ", "remove that fact"]
RESET_PHRASES = ["reset memory", "clear memory", "wipe memory", "delete all memories"]

# Learn fact patterns
LEARN_FACT_PATTERNS = [
    r"remember\s+that\s+(.+)\s+is\s+(.+)",
    r"remember\s+(.+)\s+is\s+(.+)",
    r"from\s+now\s+on[,\s]+call\s+me\s+(.+)",
    r"my\s+(\w+)\s+is\s+(.+)",
    r"store\s+(.+)\s+as\s+(.+)",
]

# Learn command patterns: "when I say X, do Y"
LEARN_CMD_PATTERNS = [
    r"when\s+i\s+say\s+['\"]?(.+?)['\"]?\s*[,.]?\s*(?:do|open|run|perform)\s+(.+)",
    r"from\s+now\s+on[,\s]+when\s+i\s+say\s+['\"]?(.+?)['\"]?\s*[,.]?\s*(.+)",
]


class IntentParser:
    """
    Rule-based intent classification with optional LLM fallback.
    Checks learned commands first, then patterns for learning/feedback/control.
    """

    def __init__(self, memory_manager: Optional[MemoryManager] = None) -> None:
        self.memory = memory_manager

    def _extract_target(self, text: str, triggers: list[str]) -> str:
        lower = text.lower()
        for t in triggers:
            if lower.startswith(t):
                return text[len(t):].strip()
            if t in lower:
                idx = lower.find(t)
                return text[idx + len(t):].strip()
        return text

    def parse(self, text: str, context: Optional[ConversationContext] = None) -> Intent:
        """Parse user utterance into an Intent."""
        text = (text or "").strip()
        if not text:
            return Intent(type=IntentType.UNKNOWN, raw_text=text)

        lower = text.lower()

        # 1. Check for learned command trigger (if memory available)
        if self.memory:
            cmd = self.memory.get_command_by_trigger(text)
            if cmd:
                return Intent(
                    type=IntentType.RUN_LEARNED_COMMAND,
                    raw_text=text,
                    parsed_action="run_learned_command",
                    params={"command_id": cmd.id, "command": cmd},
                )

        # 2. Feedback/correction
        if any(p in lower for p in FEEDBACK_PHRASES):
            return Intent(
                type=IntentType.FEEDBACK,
                raw_text=text,
                parsed_action="apply_feedback",
                params={"correction": text},
            )

        # 3. Memory control
        if any(p in lower for p in RESET_PHRASES):
            return Intent(
                type=IntentType.CONTROL,
                raw_text=text,
                parsed_action="reset_memory",
                params={},
            )
        if any(p in lower for p in FORGET_PHRASES):
            return Intent(
                type=IntentType.CONTROL,
                raw_text=text,
                parsed_action="forget_fact",
                params={"query": text},
            )

        # 4. Learn command
        for pat in LEARN_CMD_PATTERNS:
            m = re.search(pat, lower, re.IGNORECASE | re.DOTALL)
            if m:
                trigger = m.group(1).strip().strip("'\"")
                steps_desc = m.group(2).strip()
                return Intent(
                    type=IntentType.LEARN_COMMAND,
                    raw_text=text,
                    parsed_action="learn_command",
                    params={"trigger_phrase": trigger, "steps_description": steps_desc},
                )

        # 5. Learn fact
        for pat in LEARN_FACT_PATTERNS:
            m = re.search(pat, lower, re.IGNORECASE | re.DOTALL)
            if m:
                key_part = m.group(1).strip().replace(" ", "_").lower()
                value = m.group(2).strip()
                key = re.sub(r"[^a-z0-9_]", "", key_part) or "fact"
                return Intent(
                    type=IntentType.LEARN_FACT,
                    raw_text=text,
                    parsed_action="learn_fact",
                    params={"key": key, "value": value, "category": "user"},
                )

        # 6. Memory query
        if any(
            q in lower
            for q in [
                "what do you remember",
                "what have you learned",
                "list your commands",
                "what commands",
                "show memory",
            ]
        ):
            return Intent(
                type=IntentType.MEMORY_QUERY,
                raw_text=text,
                parsed_action="explain_knowledge",
                params={"query": text},
            )

        # 7. Small talk
        if lower in ("hello", "hi", "hey", "greetings"):
            return Intent(type=IntentType.SMALL_TALK, raw_text=text, params={"greeting": True})
        if any(
            q in lower
            for q in [
                "who are you",
                "what are you",
                "what is your name",
                "identify yourself",
            ]
        ):
            return Intent(
                type=IntentType.SMALL_TALK,
                raw_text=text,
                params={"identity": True},
            )
        if any(
            q in lower
            for q in [
                "what can you do",
                "your capabilities",
                "help",
                "what do you do",
            ]
        ):
            return Intent(
                type=IntentType.SMALL_TALK,
                raw_text=text,
                params={"capabilities": True},
            )

        # 8. 3D Model Viewer commands — must come BEFORE generic open_triggers
        # so "open model viewer" / "open 3d viewer" etc. are captured here first.
        _3d_triggers = [
            # --- Open viewer ---
            ("open model viewer",   "open_viewer",   {}),
            ("launch model viewer", "open_viewer",   {}),
            ("open 3d viewer",      "open_viewer",   {}),
            ("open 3d model",       "load_model",    {}),
            ("show 3d model",       "load_model",    {}),
            ("3d viewer",           "open_viewer",   {}),
            ("model viewer",        "open_viewer",   {}),
            # --- Load model ---
            ("load model",          "load_model",    {}),
            ("show model",          "load_model",    {}),
            ("show engine",         "load_model",    {"model": "engine.glb"}),
            # --- Explode ---
            ("explode model",       "explode_model", {}),
            ("explode the model",   "explode_model", {}),
            ("exploded view",       "explode_model", {}),
            ("explode view",        "explode_model", {}),
            # --- Reset ---
            ("reset model",         "reset_model",   {}),
            ("reset the model",     "reset_model",   {}),
            ("collapse model",      "reset_model",   {}),
            ("reassemble model",    "reset_model",   {}),
            # --- Rotate ---
            ("rotate model",        "rotate_model",  {}),
            ("spin model",          "rotate_model",  {}),
            ("turn model",          "rotate_model",  {}),
            ("spin the model",      "rotate_model",  {}),
            # --- Zoom ---
            ("zoom in model",       "zoom_model",    {"direction": "in"}),
            ("zoom out model",      "zoom_model",    {"direction": "out"}),
            ("zoom model",          "zoom_model",    {}),
        ]
        for keyword, action, extra in _3d_triggers:
            if keyword in lower:
                return Intent(
                    type=IntentType.MODEL_VIEW,
                    raw_text=text,
                    parsed_action=action,
                    params={"query": text, **extra},
                )

        # 9. Commands (simple keywords)
        open_triggers = ["open ", "launch ", "start ", "switch to "]
        if any(w in lower for w in open_triggers):
            query = self._extract_target(text, open_triggers)
            return Intent(
                type=IntentType.COMMAND,
                raw_text=text,
                parsed_action="open_app",
                params={"query": query},
            )
            
        create_triggers = ["create file ", "make file ", "new file "]
        if any(w in lower for w in create_triggers):
            query = self._extract_target(text, create_triggers)
            return Intent(
                type=IntentType.COMMAND,
                raw_text=text,
                parsed_action="create_file",
                params={"query": query},
            )
            
        read_triggers = ["read file ", "open file ", "show file "]
        if any(w in lower for w in read_triggers):
            query = self._extract_target(text, read_triggers)
            return Intent(
                type=IntentType.COMMAND,
                raw_text=text,
                parsed_action="read_file",
                params={"query": query},
            )
            
        delete_triggers = ["delete file ", "remove file "]
        if any(w in lower for w in delete_triggers):
            query = self._extract_target(text, delete_triggers)
            return Intent(
                type=IntentType.COMMAND,
                raw_text=text,
                parsed_action="delete_file",
                params={"query": query},
            )
            
        if any(re.search(rf"\b{w}\b", lower) for w in ["time", "what time", "current time"]):
            return Intent(
                type=IntentType.COMMAND,
                raw_text=text,
                parsed_action="tell_time",
                params={},
            )
            
        if any(re.search(rf"\b{w}\b", lower) for w in ["date", "what date", "today"]):
            return Intent(
                type=IntentType.COMMAND,
                raw_text=text,
                parsed_action="tell_date",
                params={},
            )
            
        file_search_triggers = ["find file ", "search file ", "locate file ", "where is "]
        if any(w in lower for w in file_search_triggers):
            query = self._extract_target(text, file_search_triggers)
            return Intent(
                type=IntentType.COMMAND,
                raw_text=text,
                parsed_action="file_search",
                params={"query": query},
            )
            
        window_triggers = ["minimize ", "maximize ", "close window ", "focus ", "restore "]
        for trig in window_triggers:
            if trig in lower:
                target = self._extract_target(text, [trig])
                action = trig.strip().split()[0]
                if action == "close": action = "close" # close window -> close
                return Intent(
                    type=IntentType.COMMAND,
                    raw_text=text,
                    parsed_action="manage_window",
                    params={"action": action, "target": target},
                )
                
        search_triggers = ["search for ", "search ", "look up ", "find "]
        if any(w in lower for w in search_triggers):
            query = self._extract_target(text, search_triggers)
            return Intent(
                type=IntentType.COMMAND,
                raw_text=text,
                parsed_action="web_search",
                params={"query": query},
            )
        if any(q in lower for q in ["what do you see", "describe the environment", "visual status", "show me what you see"]) or ("look" in lower and "screen" in lower):
             return Intent(
                 type=IntentType.COMMAND,
                 raw_text=text,
                 parsed_action="visual_status",
                 params={},
             )
        if any(re.search(rf"\b{w}\b", lower) for w in ["exit", "quit", "goodbye", "shut down", "stop"]):
            return Intent(
                type=IntentType.CONTROL,
                raw_text=text,
                parsed_action="stop",
                params={},
            )

        # 9. Yes/No (for confirmations)
        if lower in ("yes", "yeah", "yep", "correct", "confirm", "do it"):
            return Intent(
                type=IntentType.COMMAND,
                raw_text=text,
                parsed_action="confirm",
                params={},
            )
        if lower in ("no", "nope", "cancel", "don't"):
            return Intent(
                type=IntentType.COMMAND,
                raw_text=text,
                parsed_action="cancel",
                params={},
            )

        if any(w in lower for q in ["paste code", "paste it", "write it here", "put it here", "paste the code"] for w in [q]):
             return Intent(
                 type=IntentType.PASTE_CODE,
                 raw_text=text,
                 parsed_action="paste_stored_code",
             )

        active_window_code_triggers = ["write code in the current file", "write code here", "type code", "code here", "in the current file", "in the current window", "current window", "on the screen", "type this code"]
        if any(w in lower for q in active_window_code_triggers for w in [q]):
             return Intent(
                 type=IntentType.WRITE_CODE_ACTIVE_WINDOW,
                 raw_text=text,
                 parsed_action="generate_code_active",
                 params={"query": text},
             )

        active_window_edit_triggers = ["edit code in the current file", "change this code", "update the current file", "modify code", "edit the current file", "edit this code", "edit code here", "change code here"]
        if any(w in lower for q in active_window_edit_triggers for w in [q]):
             return Intent(
                 type=IntentType.EDIT_CODE_ACTIVE_WINDOW,
                 raw_text=text,
                 parsed_action="edit_code_active",
                 params={"query": text},
             )

        if any(w in lower for q in ["write code", "generate code", "python script", "programming", "code for"] for w in [q]):
             return Intent(
                 type=IntentType.WRITE_CODE,
                 raw_text=text,
                 parsed_action="generate_code",
                 params={"query": text},
             )

        # Weather
        if any(re.search(rf"\b{w}\b", lower) for w in ["weather", "temperature", "forecast", "how hot", "how cold", "raining", "climate"]):
            return Intent(
                type=IntentType.COMMAND,
                raw_text=text,
                parsed_action="weather",
                params={"query": text},
            )

        # Reminders
        if any(w in lower for w in ["remind me", "reminder", "set an alarm", "alert me in", "notify me in"]):
            return Intent(
                type=IntentType.COMMAND,
                raw_text=text,
                parsed_action="reminder",
                params={"query": text},
            )

        # Volume
        if any(re.search(rf"\b{w}\b", lower) for w in ["volume", "mute", "unmute", "louder", "quieter", "turn up", "turn down"]):
            return Intent(
                type=IntentType.COMMAND,
                raw_text=text,
                parsed_action="volume",
                params={"query": text},
            )

        # Screenshot
        if any(w in lower for w in ["screenshot", "capture screen", "take a screenshot", "screen capture"]):
            return Intent(
                type=IntentType.COMMAND,
                raw_text=text,
                parsed_action="screenshot",
                params={},
            )

        # Notes
        if any(w in lower for w in ["make a note", "take a note", "note that", "my notes", "show notes", "read notes"]):
            return Intent(
                type=IntentType.COMMAND,
                raw_text=text,
                parsed_action="notes",
                params={"query": text},
            )

        # Clipboard
        if any(w in lower for w in ["clipboard", "what's in my clipboard", "what did i copy", "copy to clipboard"]):
            return Intent(
                type=IntentType.COMMAND,
                raw_text=text,
                parsed_action="clipboard",
                params={"query": text},
            )

        # 10. Gap detection (Phase 10: Dynamic Skill Synthesis)
        gap_keywords = [
            "synthesize", "generate a skill", "write a new skill", 
            "learn a new skill", "automate this", "create a script to",
            "write a python skill", "add a capability"
        ]
        if any(w in lower for w in gap_keywords):
            return Intent(
                type=IntentType.SENSE_GAP,
                raw_text=text,
                parsed_action="synthesize_skill",
                params={"gap_description": text},
            )

        # 12. Default: question or generic command
        return Intent(
            type=IntentType.QUESTION,
            raw_text=text,
            parsed_action="answer",
            params={"query": text},
        )
