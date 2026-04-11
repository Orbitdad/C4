"""
jarvis/vision/gesture/macro_engine.py

Loads gesture macros from a YAML file and matches gesture sequences
against them, returning the prioritized and filtered macro for execution.
"""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# ── Gesture Categories ───────────────────────────────────────────────────────
DISCRETE_GESTURES = {
    "SWIPE_LEFT", "SWIPE_RIGHT", "SWIPE_UP", "SWIPE_DOWN",
    "PINCH", "OPEN_PALM", "FIST", "CIRCLE", "TRIGGER", "CLICK"
}

CONTINUOUS_GESTURES = {"ROTATE", "ZOOM", "MOVE", "INDEX_POINT"}


class MacroEngine:
    """
    Loads, matches, and (via :class:`ActionExecutor`) executes gesture macros.
    Supports mode-awareness (HOTKEY vs 3D), priorities, and cooldowns.
    """

    def __init__(self, config_path: str | Path) -> None:
        self._path = Path(config_path)
        self.macros: List[Dict] = []
        self.app_keywords: Dict[str, list] = {}
        self._last_trigger_times: Dict[str, float] = {}
        self._load()

    # ── Public API ────────────────────────────────────────────────────────────

    def load_macros(self, path: str | Path) -> List[Dict]:
        """
        Parse the YAML macros file and return the list of macro dicts.
        Also extracts ``app_keywords`` for :class:`ContextEngine`.
        """
        try:
            import yaml
        except ImportError:
            log.warning("PyYAML not installed — macros disabled. Run: pip install pyyaml")
            return []

        try:
            with open(path, "r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            
            self.app_keywords = data.get("app_keywords", {})
            macros = data.get("macros", [])

            # Apply defaults and sort by priority (High -> Low)
            # Secondarily sort by context specificity (context-specific > any)
            for m in macros:
                m.setdefault("priority", 0)
                m.setdefault("context", "any")
                m.setdefault("mode", "HOTKEY")
                m.setdefault("cooldown_ms", 500)

            # Sorting key: (priority, is_not_any_context)
            macros.sort(key=lambda x: (x.get("priority", 0), 1 if x.get("context") != "any" else 0), reverse=True)

            log.info("MacroEngine: loaded %d macros from %s", len(macros), path)
            return macros
        except FileNotFoundError:
            log.warning("MacroEngine: macros file not found at %s", path)
            return []
        except Exception as exc:
            log.error("MacroEngine: failed to parse %s — %s", path, exc)
            return []

    def match(self, sequence: List[str], context: Dict) -> Optional[Dict]:
        """
        Find the highest priority macro whose sequence matches the discrete tail 
         of *sequence* and satisfies mode, context, and cooldown constraints.

        :param sequence: Rolling gesture token list from :class:`GestureStateManager`.
        :param context:  Dict from :class:`ContextEngine.get_context`.
        :returns: The matched macro dict, or ``None``.
        """
        if not sequence or not self.macros:
            return None

        # 1. Filter sequence to only include discrete gestures (Rule #4)
        discrete_seq = [g for g in sequence if g in DISCRETE_GESTURES]
        if not discrete_seq:
            return None

        ctx_type = context.get("type", "any")
        
        # 2. Add Mode Awareness (Rule #1)
        # We look for interaction_mode in the context (passed from FusionEngine)
        # Default to HOTKEY_MODE for backward compatibility
        raw_mode = context.get("interaction_mode", "HOTKEY_MODE")
        current_mode = "3D" if "3D" in raw_mode else "HOTKEY"

        now = time.time()

        for macro in self.macros:
            macro_seq     = macro.get("sequence", [])
            macro_context = macro.get("context", "any")
            macro_mode    = macro.get("mode", "HOTKEY").upper()
            macro_name    = macro.get("name", "unnamed")

            # Mode check
            if macro_mode != "ANY" and macro_mode != current_mode:
                continue

            # Context check
            if macro_context != "any" and macro_context != ctx_type:
                continue

            # Cooldown check (Rule #2)
            cooldown_s = macro.get("cooldown_ms", 500) / 1000.0
            last_hit = self._last_trigger_times.get(macro_name, 0.0)
            if now - last_hit < cooldown_s:
                continue

            # Sequence match (Rule #6 - Ordered match on discrete tail)
            if not macro_seq:
                continue
            if len(discrete_seq) < len(macro_seq):
                continue
            
            # Simple suffix match on the filtered discrete sequence
            if list(discrete_seq[-len(macro_seq):]) == list(macro_seq):
                log.debug("MacroEngine: matched '%s' (Priority: %d)", macro_name, macro.get("priority"))
                
                # Update cooldown state
                self._last_trigger_times[macro_name] = now
                return macro

        return None

    def execute(self, macro: Dict, executor: Any = None) -> None:
        """
        Execute a matched macro's action list.
        """
        if executor is None:
            from jarvis.vision.gesture.action_executor import ActionExecutor
            executor = ActionExecutor()

        actions = macro.get("actions", [])
        name    = macro.get("name", "unnamed")
        log.info("MacroEngine: executing macro '%s' (%d action(s))", name, len(actions))

        for action in actions:
            try:
                executor.execute(action)
            except Exception as exc:
                log.error("MacroEngine: action failed in '%s' — %s", name, exc)

    def reload(self) -> None:
        """Hot-reload macros from disk."""
        self._load()

    # ── Private ───────────────────────────────────────────────────────────────

    def _load(self) -> None:
        self.macros = self.load_macros(self._path)
        # Reset cooldowns on reload
        self._last_trigger_times = {}
