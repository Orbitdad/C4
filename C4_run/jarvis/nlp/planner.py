"""
Task planner: converts intents into structured multi-step plans.
"""

from __future__ import annotations

from typing import Optional

from jarvis.memory.manager import MemoryManager
from jarvis.memory.models import ActionStep
from jarvis.nlp.schemas import Intent, IntentType, Plan


class TaskPlanner:
    """
    Converts parsed intents into executable plans.
    For learned commands, loads steps from memory.
    """

    def __init__(self, memory_manager: MemoryManager, llm_client: Optional[Any] = None) -> None:
        self.memory = memory_manager
        self.llm = llm_client

    def build_execution_graph(self, intent: Intent, context: Optional[Any] = None) -> dict:
        prompt = f"""Generate a JSON execution graph for the requested task: "{intent.raw_text}"
Return ONLY raw JSON with this specific schema:
{{
  "start_node": "step1",
  "nodes": {{
    "step1": {{
       "type": "run_command", 
       "params": {{"query": "<command>"}}, 
       "requires_confirmation": true,
       "on_success": "step2", 
       "on_failure": "fallback1"
    }},
    "step2": {{ "type": "tell_user", "params": {{"message": "Done."}} }},
    "fallback1": {{ "type": "tell_user", "params": {{"message": "Failed."}} }}
  }}
}}
CRITICAL RULES:
- `type` must be one of: run_command, open_app, create_file, delete_file, read_file, tell_time, tell_user, web_search, manage_window, run_python.
- If an action is destructive (like deleting files or modifying system) or executes arbitrary code (run_python, run_command), set `requires_confirmation: true`.
- ALWAYS provide an `on_failure` edge that handles errors gracefully via `tell_user`.
- Return ONLY the raw JSON string starting with {{ and ending with }}. No markdown blocks.
"""
        import json
        if not self.llm: return {}
        response = self.llm.generate(prompt)
        try:
            cleaned = response.strip()
            if cleaned.startswith("```json"): cleaned = cleaned[7:]
            if cleaned.startswith("```"): cleaned = cleaned[3:]
            cleaned = cleaned.strip("` \n")
            return json.loads(cleaned)
        except Exception:
            return {}

    def plan_with_qwen(self, intent: Intent, context: Optional[Any] = None) -> dict:
        prompt = f"""Analyze the user's request and create an execution plan: "{intent.raw_text}"
Return ONLY raw JSON with this specific schema:
{{
  "intent": "<main action intent>",
  "task_type": "<general or coding>",
  "steps": [
    {{
       "action": "<action type>",
       "params": {{"<key>": "<value>"}}
    }}
  ]
}}
CRITICAL RULES:
- `task_type` MUST be "coding" if it involves writing code, scripts, html, css, or software development. Otherwise "general".
- `action` must be a valid execution step like: open_app, create_file, delete_file, read_file, tell_time, tell_user, web_search, manage_window, navigate, run_command, code.
- If the user is asking a question (e.g. "who is...", "what is..."), YOU MUST use the `tell_user` action to directly answer them with the `message` parameter. Do NOT use `analyze_request`.
- Return ONLY the raw JSON string starting with {{ and ending with }}. No markdown blocks.
"""
        import json
        if not self.llm: return {"intent": "unknown", "task_type": "general", "steps": []}
        
        # Use Qwen (planner role) to generate this plan
        history = context.to_prompt_history() if hasattr(context, "to_prompt_history") else None
        
        # We assume self.llm provides call_llm (RoleBasedLLM)
        if hasattr(self.llm, "call_llm"):
            response = self.llm.call_llm(role="planner", prompt=prompt, history=history)
        else:
            response = self.llm.generate(prompt, history=history)
            
        try:
            cleaned = response.strip()
            if cleaned.startswith("```json"): cleaned = cleaned[7:]
            if cleaned.startswith("```"): cleaned = cleaned[3:]
            cleaned = cleaned.strip("` \n")
            return json.loads(cleaned)
        except Exception:
            return {"intent": "unknown", "task_type": "general", "steps": []}


    def plan(self, intent: Intent, context: Optional[Any] = None) -> Plan:
        """Produce an execution plan for the given intent."""
        if intent.type == IntentType.RUN_LEARNED_COMMAND:
            cmd = intent.params.get("command")
            if cmd:
                return Plan(
                    steps=cmd.steps,
                    confirmation_required=cmd.confirmation_required,
                    summary=cmd.description,
                )
            return Plan(steps=[], summary="Unknown command")

        if intent.type == IntentType.COMMAND:
            action = intent.parsed_action or ""
            params = intent.params or {}
            
            # Context-aware resolution for ambiguous parameters
            if self.llm and context:
                history = context.to_prompt_history() if hasattr(context, "to_prompt_history") else []
                query = params.get("query") or intent.raw_text
                
                # If query contains "it", "that", "him", "her", "them", "there"
                if any(w in query.lower().split() for w in ["it", "that", "him", "her", "them", "there"]):
                    resolution_prompt = f"The user said: \"{query}\". Based on the conversation history, what specific entity is the user referring to? Return ONLY the resolved entity name (e.g. 'chrome', 'VS Code', 'document.txt')."
                    resolved = self.llm.generate(resolution_prompt, history=history)
                    if resolved and not resolved.startswith("[Error"):
                        # Update params with resolved entity
                        if action == "open_app":
                            params["app"] = resolved
                        elif action in ["create_file", "read_file", "delete_file"]:
                            params["path"] = resolved
                        elif action == "web_search":
                            params["query"] = f"{resolved} {query}".strip()

            if action == "open_app":
                app = params.get("app") or params.get("query", "")
                return Plan(
                    steps=[ActionStep(type="open_app", params={"app": app, "query": app})],
                    summary=f"Open application: {app}",
                )
            if action == "create_file":
                path = params.get("path") or params.get("query", "")
                return Plan(
                    steps=[ActionStep(type="create_file", params={"query": path})],
                    confirmation_required=True,
                    summary=f"Create file: {path}",
                )
            if action == "read_file":
                path = params.get("path") or params.get("query", "")
                return Plan(
                    steps=[ActionStep(type="read_file", params={"query": path})],
                    summary=f"Read file: {path}",
                )
            if action == "delete_file":
                path = params.get("path") or params.get("query", "")
                return Plan(
                    steps=[ActionStep(type="delete_file", params={"query": path})],
                    confirmation_required=True,
                    summary=f"Delete file: {path}",
                )
            if action == "tell_time":
                return Plan(
                    steps=[ActionStep(type="tell_time", params={})],
                    summary="Tell current time",
                )
            if action == "tell_date":
                return Plan(
                    steps=[ActionStep(type="tell_date", params={})],
                    summary="Tell current date",
                )
            if action == "web_search":
                q = params.get("query", "")
                return Plan(
                    steps=[ActionStep(type="web_search", params={"query": q})],
                    summary=f"Search web: {q}",
                )
            if action == "manage_window":
                act = params.get("action", "restore")
                target = params.get("target", "")
                return Plan(
                    steps=[ActionStep(type="manage_window", params={"action": act, "target": target})],
                    summary=f"{act.capitalize()} window: {target}",
                )

        return Plan(steps=[], summary="")
