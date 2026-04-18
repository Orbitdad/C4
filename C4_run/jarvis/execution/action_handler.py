"""
C4 Action Handler — Controlled Step Executor
=============================================

The model NEVER directly executes actions.
This module sits between the CommandHandler (planner output) and the
Executor (OS actions), enforcing:

  1. An explicit allowlist of permitted action types
  2. Routing: coding tasks → deepseek-coder:6.7b (via LLM)
  3. Parameter normalisation before OS execution
  4. Structured result dicts for UI feedback

Usage:
    handler = ActionHandler(executor=executor, llm_client=llm)
    result = handler.execute_step(action="open_app", params={"app": "chrome"})
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# ── Strict allowlist: only these action types may be invoked ────────────────
ALLOWED_ACTIONS = frozenset({
    "open_app",
    "open_url",
    "navigate",
    "web_search",
    "create_file",
    "read_file",
    "update_file",
    "delete_file",
    "run_command",
    "mouse_move",
    "mouse_click",
    "keyboard_type",
    "keyboard_hotkey",
    "manage_window",
    "play_media",
    "tell_time",
    "tell_date",
    "tell_user",
    "code",          # routes to deepseek-coder:6.7b
})


class ActionHandler:
    """
    Controlled wrapper around the Executor.

    - Enforces ALLOWED_ACTIONS allowlist
    - Routes `code` action type to deepseek-coder:6.7b
    - Normalises action names from Qwen output to Executor step types
    - Returns structured {success, message} dicts
    """

    # Map Qwen action names → Executor step types
    _ACTION_ALIAS: Dict[str, str] = {
        "navigate":        "open_url",
        "open_browser":    "open_app",
        "launch":          "open_app",
        "search":          "web_search",
        "type":            "keyboard_type",
        "press":           "keyboard_hotkey",
        "click":           "mouse_click",
        "move_mouse":      "mouse_move",
        "create":          "create_file",
        "read":            "read_file",
        "delete":          "delete_file",
        "update":          "update_file",
        "run":             "run_command",
        "execute":         "run_command",
        "tell_user":       "tell_user",    # no-op: log only
        "say":             "tell_user",
    }

    def __init__(
        self,
        executor: Any,
        llm_client: Optional[Any] = None,
    ) -> None:
        self.executor = executor
        self.llm = llm_client

    # ── Public API ─────────────────────────────────────────────────────────────

    def execute_step(
        self,
        action: str,
        params: Dict[str, Any],
        task_type: str = "general",
    ) -> Dict[str, Any]:
        """
        Execute a single action step.

        Args:
            action    – Action name from planning output (Qwen)
            params    – Parameter dict for the action
            task_type – "general" or "coding" (used for routing)

        Returns:
            {"success": bool, "message": str, ...}
        """
        # Normalise action name
        resolved = self._ACTION_ALIAS.get(action.lower(), action.lower())

        # Special: coding task → route to deepseek-coder
        if resolved == "code" or task_type == "coding":
            return self._handle_code(action, params)

        # Special: tell_user is a log-only action
        if resolved == "tell_user":
            msg = params.get("message", params.get("text", ""))
            logger.info(f"[ActionHandler] tell_user: {msg}")
            return {"success": True, "message": msg}

        # Allowlist check
        if resolved not in ALLOWED_ACTIONS:
            logger.warning(f"[ActionHandler] Blocked disallowed action: {action!r} → {resolved!r}")
            return {
                "success": False,
                "message": f"Action '{action}' is not permitted.",
            }

        # Normalise params for executor
        norm_params = self._normalise_params(resolved, params)

        # Build a step-like object for the executor
        step = _StepProxy(step_type=resolved, params=norm_params)

        try:
            result = self.executor.execute_step(step)
            logger.info(f"[ActionHandler] ✓ {resolved}: {result.get('message','')[:80]}")
            return result
        except Exception as e:
            logger.error(f"[ActionHandler] ✗ {resolved} error: {e}")
            return {"success": False, "message": str(e)}

    # ── Coding Router ──────────────────────────────────────────────────────────

    def _handle_code(self, action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Route coding tasks to deepseek-coder:6.7b."""
        if not self.llm:
            return {"success": False, "message": "No LLM client available for code generation."}

        task = (
            params.get("task")
            or params.get("description")
            or params.get("query")
            or params.get("prompt")
            or str(params)
        )
        lang = params.get("language", "python")

        prompt = (
            f"Write {lang} code for the following task. "
            f"Return ONLY the code — no explanation, no markdown fences:\n\n{task}"
        )

        try:
            if hasattr(self.llm, "call_llm"):
                code = self.llm.call_llm(role="coder", prompt=prompt)
            else:
                code = self.llm.generate(prompt)

            if code and not code.startswith("[Error"):
                # If output path provided, write to file
                output_path = params.get("output_path") or params.get("path")
                if output_path:
                    from pathlib import Path
                    Path(output_path).write_text(code, encoding="utf-8")
                    logger.info(f"[ActionHandler] Code written to {output_path}")
                    return {
                        "success": True,
                        "message": f"Code written to {output_path} ({len(code)} chars).",
                        "code": code,
                    }
                return {
                    "success": True,
                    "message": f"Code generated ({len(code)} chars). Ready to use.",
                    "code": code,
                }
            return {"success": False, "message": "Code generation returned empty result."}
        except Exception as e:
            logger.error(f"[ActionHandler] Code generation error: {e}")
            return {"success": False, "message": f"Code generation error: {e}"}

    # ── Param Normalisation ────────────────────────────────────────────────────

    @staticmethod
    def _normalise_params(action: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Ensure params match what the Executor expects for each action type."""
        p = dict(params)

        if action == "open_app":
            # Executor expects params["app"]
            if "app" not in p:
                p["app"] = p.get("application") or p.get("name") or p.get("query", "")

        elif action == "open_url":
            # Executor expects params["url"]
            if "url" not in p:
                p["url"] = p.get("target") or p.get("address") or p.get("query", "")

        elif action == "web_search":
            if "query" not in p:
                p["query"] = p.get("q") or p.get("term") or p.get("text", "")

        elif action == "run_command":
            if "command" not in p:
                p["command"] = p.get("cmd") or p.get("shell") or p.get("query", "")

        elif action in ("create_file", "read_file", "delete_file", "update_file"):
            if "path" not in p:
                p["path"] = p.get("file") or p.get("filename") or p.get("query", "")

        return p


# ── Step proxy so Executor.execute_step() works unchanged ──────────────────

class _StepProxy:
    """Minimal duck-type shim matching the interface Executor.execute_step expects."""

    __slots__ = ("type", "params")

    def __init__(self, step_type: str, params: Dict[str, Any]) -> None:
        self.type   = step_type
        self.params = params

    # Executor also supports dict-style steps, but we use attribute access.
    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)
