"""
jarvis/vision/gesture/fusion_engine.py

The decision arbiter — combines all engine outputs into a single,
prioritised action dict that the ActionExecutor can act on.

Priority (highest → lowest)
    1. Matched macro
    2. Context-specific gesture override
    3. Intent-inferred action
    4. Raw gesture fallback
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

# Priority constants
PRI_MACRO   = 100
PRI_CONTEXT = 70
PRI_INTENT  = 40
PRI_FALLBACK = 10


# ── Intent → action type mapping ─────────────────────────────────────────────
_INTENT_ACTION: Dict[str, str] = {
    "CLICK":       "CLICK",
    "SCROLL":      "SCROLL",
    "DRAG":        "DRAG",
    "NAVIGATE":    "NAVIGATE",
    "COMMAND":     "COMMAND",
    "PAUSE":       "BUS",
    "MOVE_CURSOR": "MOVE",
    "NONE":        "NOOP",
}

# ── Context override → action type mapping ────────────────────────────────────
_CONTEXT_OVERRIDE_ACTION: Dict[str, str] = {
    "PLAY_PAUSE":       "KEY",
    "VOLUME":           "NOOP",
    "VOL_UP":           "HOTKEY",
    "VOL_DOWN":         "HOTKEY",
    "PREV_TRACK":       "HOTKEY",
    "NEXT_TRACK":       "HOTKEY",
    "BROWSER_BACK":     "HOTKEY",
    "BROWSER_FORWARD":  "HOTKEY",
    "BROWSER_REFRESH":  "KEY",
    "COMMENT_TOGGLE":   "HOTKEY",
    "TOGGLE_TERMINAL":  "HOTKEY",
}


class FusionEngine:
    """
    The central decision maker of the gesture pipeline.

    :meth:`resolve` accepts the outputs of all five upstream engines and
    returns a single action dict ready for :class:`ActionExecutor`.

    Output format::

        {
            "action":   str,   # "CLICK" | "MOVE" | "SCROLL" | "HOTKEY" | "BUS" | ...
            "priority": int,   # 10 – 100
            "source":   str,   # "macro" | "context" | "intent" | "fallback"
            # … plus action-specific keys (x, y, keys, event, …)
        }
    """

    def __init__(self) -> None:
        pass

    # ── Public API ────────────────────────────────────────────────────────────

    def resolve(
        self,
        gesture:  Dict,
        state:    Dict,
        intent:   Dict,
        context:  Dict,
        macro:    Optional[Dict],
    ) -> Dict:
        """
        Produce the single best action to execute this frame.

        :param gesture: From :class:`GestureDetector`.
        :param state:   From :class:`GestureStateManager`.
        :param intent:  From :class:`IntentEngine`.
        :param context: From :class:`ContextEngine`.
        :param macro:   From :class:`MacroEngine` (``None`` if no match).
        """
        candidates: List[Dict] = []

        # ── 1. Macro candidate ────────────────────────────────────────────────
        if macro:
            candidates.append({
                "action":   "MACRO",
                "priority": PRI_MACRO,
                "source":   "macro",
                "macro":    macro,
            })

        # ── 2. Context-override candidate ─────────────────────────────────────
        ctx_override = self._build_context_candidate(gesture, context)
        if ctx_override:
            candidates.append(ctx_override)

        # ── 3. Intent candidate ───────────────────────────────────────────────
        intent_candidate = self._build_intent_candidate(gesture, intent)
        if intent_candidate:
            candidates.append(intent_candidate)

        # ── 4. Fallback ───────────────────────────────────────────────────────
        candidates.append(self.fallback(gesture))

        result = self.prioritize(candidates)
        log.debug(
            "FusionEngine: resolved action='%s' source='%s' priority=%d",
            result.get("action"), result.get("source"), result.get("priority", 0),
        )
        return result

    def prioritize(self, candidates: List[Dict]) -> Dict:
        """Return the candidate with the highest priority score."""
        if not candidates:
            return {"action": "NOOP", "priority": 0, "source": "fallback"}
        return max(candidates, key=lambda c: c.get("priority", 0))

    def fallback(self, gesture: Dict) -> Dict:
        """
        Produce a safe default action based purely on the raw gesture.
        This is always appended as the lowest-priority candidate.
        """
        raw = gesture.get("gesture", "NONE")
        pos = gesture.get("position", (0.0, 0.0, 0.0))

        _raw_fallbacks: Dict[str, Dict] = {
            "MOVE":  {"action": "MOVE",  "x": pos[0], "y": pos[1]},
            "PINCH": {"action": "CLICK"},
            "SCROLL":{"action": "SCROLL","amount": 0},
            "FIST":  {"action": "BUS",   "event": "vision.gesture.fist",
                      "data": {"gesture": "FIST", "confidence": gesture.get("confidence", 0.9)}},
            "OPEN_PALM": {"action": "BUS", "event": "vision.gesture.open_palm", "data": {}},
        }

        base = _raw_fallbacks.get(raw, {"action": "NOOP"})
        return {**base, "priority": PRI_FALLBACK, "source": "fallback"}

    # ── Private builders ──────────────────────────────────────────────────────

    def _build_context_candidate(
        self, gesture: Dict, context: Dict
    ) -> Optional[Dict]:
        """
        If ContextEngine maps this gesture to an override, emit a context candidate.
        We import ContextEngine here to avoid a circular import at module level.
        """
        from jarvis.vision.gesture.context_engine import ContextEngine
        # Lightweight: just use the map directly since we already have context
        _CTX_MAP: Dict[str, Dict[str, Dict]] = {
            "media": {
                "PINCH":       {"action": "KEY",    "key": "space"},
                "SWIPE_UP":    {"action": "HOTKEY", "keys": ["ctrl", "up"]},
                "SWIPE_DOWN":  {"action": "HOTKEY", "keys": ["ctrl", "down"]},
                "SWIPE_LEFT":  {"action": "HOTKEY", "keys": ["ctrl", "left"]},
                "SWIPE_RIGHT": {"action": "HOTKEY", "keys": ["ctrl", "right"]},
            },
            "browser": {
                "SWIPE_LEFT":  {"action": "HOTKEY", "keys": ["alt", "left"]},
                "SWIPE_RIGHT": {"action": "HOTKEY", "keys": ["alt", "right"]},
                "CIRCLE":      {"action": "KEY",    "key": "f5"},
            },
            "editor": {
                "CIRCLE":      {"action": "HOTKEY", "keys": ["ctrl", "/"]},
                "SWIPE_DOWN":  {"action": "HOTKEY", "keys": ["ctrl", "`"]},
            },
        }

        ctx_type = context.get("type", "any")
        raw      = gesture.get("gesture", "NONE")
        mapping  = _CTX_MAP.get(ctx_type, {}).get(raw)
        if mapping:
            return {**mapping, "priority": PRI_CONTEXT, "source": "context"}
        return None

    def _build_intent_candidate(
        self, gesture: Dict, intent: Dict
    ) -> Optional[Dict]:
        """Build a candidate from the intent engine output."""
        intent_str = intent.get("intent", "NONE")
        action_str = _INTENT_ACTION.get(intent_str, "NOOP")
        pos        = gesture.get("position", (0.0, 0.0, 0.0))

        if action_str == "NOOP":
            return None

        candidate: Dict = {"action": action_str, "priority": PRI_INTENT, "source": "intent"}

        if action_str == "MOVE":
            candidate["x"] = pos[0]
            candidate["y"] = pos[1]
        elif action_str == "BUS":
            candidate["event"] = "jarvis.pause"
            candidate["data"]  = {}

        return candidate
