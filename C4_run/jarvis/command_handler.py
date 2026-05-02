"""
C4 Command Handler — Central AI Control Pipeline
================================================

Flow:
  User Input (text / voice / gesture)
    → retrieve_memory (top 3–5 hits)
    → build prompt with context
    → qwen2:7b via plan_with_qwen()  →  {intent, task_type, steps}
    → emit UI updates (ThinkingPanel)
    → ActionHandler executes steps sequentially
    → store results to memory
    → emit "Completed" / "Failed" status

This module is the ONLY entry point for executing AI actions.
The LLM decides the plan; this module controls what actually runs.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from jarvis.nlp.planner import TaskPlanner
    from jarvis.execution.action_handler import ActionHandler
    from jarvis.memory.retriever import MemoryRetriever
    from jarvis.memory.writer import MemoryWriter
    from hui import HUIDashboard

logger = logging.getLogger(__name__)


class CommandHandler:
    """
    Centralized command handler that orchestrates the full AI pipeline.

    Attributes:
        planner        – TaskPlanner (wraps qwen2:7b via plan_with_qwen)
        action_handler – ActionHandler (controlled step executor)
        memory_retriever – MemoryRetriever for context injection
        memory_writer    – MemoryWriter for post-execution storage
        hui_window       – HUIDashboard for UI feedback
        event_bus        – C4 event bus for system-wide events
    """

    def __init__(
        self,
        planner: "TaskPlanner",
        action_handler: "ActionHandler",
        memory_retriever: Optional["MemoryRetriever"] = None,
        memory_writer: Optional["MemoryWriter"] = None,
        hui_window: Optional["HUIDashboard"] = None,
        event_bus: Optional[Any] = None,
        voice_output: Optional[Any] = None,
    ) -> None:
        self.planner = planner
        self.action_handler = action_handler
        self.memory_retriever = memory_retriever
        self.memory_writer = memory_writer
        self.hui = hui_window
        self.bus = event_bus
        self.voice_output = voice_output

        # Pending gesture confirmation
        self._pending_plan: Optional[Dict[str, Any]] = None
        self._pending_input: Optional[str] = None
        self._lock = threading.Lock()

    # ── Public API ─────────────────────────────────────────────────────────────

    def handle(self, user_input: str, source: str = "ui", context: Any = None) -> None:
        """
        Entry point for all commands. Runs asynchronously in a daemon thread
        so the Qt event loop is never blocked.

        Args:
            user_input  – Raw text from user (voice transcript or typed command)
            source      – "ui", "voice", or "gesture"
            context     – Optional ConversationContext for history-aware planning
        """
        if not user_input or not user_input.strip():
            return

        logger.info(f"[CommandHandler] Received ({source}): {user_input!r}")
        thread = threading.Thread(
            target=self._pipeline,
            args=(user_input.strip(), source, context),
            daemon=True,
            name="c4-cmd-pipeline",
        )
        thread.start()

    def confirm_pending(self) -> None:
        """Called by gesture HOLD to confirm a pending plan that needs confirmation."""
        with self._lock:
            plan = self._pending_plan
            inp = self._pending_input
            self._pending_plan = None
            self._pending_input = None

        if plan and inp:
            logger.info("[CommandHandler] Pending plan confirmed via gesture.")
            thread = threading.Thread(
                target=self._execute_plan,
                args=(plan, inp),
                daemon=True,
                name="c4-cmd-confirmed",
            )
            thread.start()
        else:
            logger.debug("[CommandHandler] No pending plan to confirm.")

    # ── Internal Pipeline ──────────────────────────────────────────────────────

    def _pipeline(self, user_input: str, source: str, context: Any) -> None:
        """Full AI pipeline: memory → plan → execute → store."""
        try:
            # ── Step 1: Retrieve relevant memory ──────────────────────────────
            self._emit_status("Planning...")
            self._emit_log(f"CMD [{source.upper()}]: {user_input}")

            memories: List[Any] = []
            if self.memory_retriever:
                try:
                    memories = self.memory_retriever.retrieve(user_input, max_results=5)
                    if memories:
                        self._refresh_globe(memories)
                        logger.debug(f"[CommandHandler] Retrieved {len(memories)} memories.")
                except Exception as e:
                    logger.warning(f"[CommandHandler] Memory retrieval failed: {e}")

            # ── Step 2: Build intent + plan via qwen2:7b ───────────────────────
            self._emit_thinking_started()

            class _FakeIntent:
                """Minimal intent shim for plan_with_qwen."""
                def __init__(self, raw: str):
                    self.raw_text = raw

            intent = _FakeIntent(user_input)
            # Inject memory into context if available
            if memories and hasattr(context, "__class__"):
                mem_text = "\n".join(
                    f"- {getattr(m, 'text', str(m))}" for m in memories[:3]
                )
                intent.raw_text = (
                    f"Relevant Memory:\n{mem_text}\n\nUser request: {user_input}"
                )

            logger.info("[CommandHandler] Calling qwen2:7b via plan_with_qwen...")
            plan = self.planner.plan_with_qwen(intent, context=context)

            intent_label  = plan.get("intent", "unknown")
            task_type     = plan.get("task_type", "general")
            steps         = plan.get("steps", [])

            logger.info(
                f"[CommandHandler] Plan — intent={intent_label!r}, "
                f"task_type={task_type!r}, steps={len(steps)}"
            )

            # ── Step 3: Update ThinkingPanel ───────────────────────────────────
            self._emit_thinking_update(task_type, intent_label, steps)
            self._refresh_radar([f"{s.get('action','?')} — pending" for s in steps])

            # ── Step 4: Execute plan ───────────────────────────────────────────
            self._execute_plan(plan, user_input)

        except Exception as e:
            logger.exception(f"[CommandHandler] Pipeline error: {e}")
            self._emit_status("Error")
            self._emit_log(f"ERROR: {e}")
            self._emit_thinking_stopped()

    def _execute_plan(self, plan: Dict[str, Any], user_input: str) -> None:
        """Execute the steps from a resolved plan sequentially."""
        task_type = plan.get("task_type", "general")
        steps     = plan.get("steps", [])
        intent    = plan.get("intent", "unknown")
        results: List[str] = []

        if not steps:
            self._emit_status("Completed")
            self._emit_thinking_stopped()
            self._emit_log("No executable steps in plan.")
            return

        self._emit_status("Executing...")

        for i, step in enumerate(steps):
            action = step.get("action", "")
            params = step.get("params", {})

            self._emit_step_status(i, "running")
            self._emit_status(f"Executing step {i+1}/{len(steps)}: {action}")
            self._refresh_radar(
                [f"{s.get('action','?')} — {'done' if j < i else ('running' if j == i else 'pending')}"
                 for j, s in enumerate(steps)]
            )

            logger.info(f"[CommandHandler] Step {i+1}/{len(steps)}: {action} {params}")

            try:
                result = self.action_handler.execute_step(
                    action=action,
                    params=params,
                    task_type=task_type,
                )
                msg = result.get("message", "Done.")
                ok  = result.get("success", True)
                results.append(msg)
                self._emit_step_status(i, "done" if ok else "error")
                self._emit_log(f"  ✓ {action}: {msg}" if ok else f"  ✗ {action}: {msg}")
                
                if action.lower() in ("tell_user", "say") and ok and self.voice_output:
                    self.voice_output.speak(msg)
                    
            except Exception as e:
                logger.error(f"[CommandHandler] Step {i+1} error: {e}")
                self._emit_step_status(i, "error")
                self._emit_log(f"  ✗ {action}: {e}")
                results.append(str(e))

        # ── Step 5: Store result to memory ────────────────────────────────────
        if self.memory_writer and results:
            result_summary = "; ".join(results[:3])
            try:
                self.memory_writer.maybe_store(
                    user_input=user_input,
                    assistant_response=result_summary,
                )
            except Exception as e:
                logger.debug(f"[CommandHandler] Memory write failed: {e}")

        # ── Step 6: Final UI update ────────────────────────────────────────────
        self._emit_status("Completed")
        self._emit_thinking_stopped()
        self._refresh_radar([f"{s.get('action','?')} — done" for s in steps])
        self._emit_log(f"✓ COMPLETED: {intent}")
        logger.info(f"[CommandHandler] Execution complete. Intent={intent!r}")

    # ── UI Helpers ─────────────────────────────────────────────────────────────

    def _emit_status(self, status: str) -> None:
        if self.hui:
            try:
                self.hui.signals.update_status.emit(status)
                self.hui.signals.execution_status.emit(status)
            except Exception:
                pass

    def _emit_log(self, msg: str) -> None:
        if self.hui:
            try:
                self.hui.signals.log_message.emit(f"C4: {msg}")
            except Exception:
                pass

    def _emit_thinking_started(self) -> None:
        if self.hui:
            try:
                self.hui.signals.thinking_started.emit()
            except Exception:
                pass

    def _emit_thinking_stopped(self) -> None:
        if self.hui:
            try:
                self.hui.signals.thinking_stopped.emit()
            except Exception:
                pass

    def _emit_thinking_update(self, task_type: str, intent: str, steps: List[Dict]) -> None:
        if self.hui:
            try:
                self.hui.signals.thinking_plan.emit(task_type, intent, steps)
            except Exception:
                pass

        # Also publish to event bus for subsystem consumers
        if self.bus:
            try:
                from jarvis.core.event_bus import SystemEvent
                self.bus.publish(SystemEvent(
                    name="c4.thinking.update",
                    data={"task_type": task_type, "intent": intent, "steps": steps},
                ))
            except Exception:
                pass

    def _emit_step_status(self, step_idx: int, status: str) -> None:
        if self.hui:
            try:
                self.hui.signals.thinking_step_status.emit(step_idx, status)
            except Exception:
                pass

        if self.bus:
            try:
                from jarvis.core.event_bus import SystemEvent
                self.bus.publish(SystemEvent(
                    name="c4.thinking.step_status",
                    data={"step_idx": step_idx, "status": status},
                ))
            except Exception:
                pass

    def _refresh_globe(self, memories: List[Any]) -> None:
        """Push memory hits to the Globe (MemoryHUD) panel."""
        if self.hui:
            try:
                snippets = [
                    getattr(m, "text", str(m))[:60] for m in memories[:5]
                ]
                self.hui.globe.refresh_memory(snippets)
            except Exception:
                pass

    def _refresh_radar(self, task_labels: List[str]) -> None:
        """Push active task labels to the Radar (ActiveTasksHUD) panel."""
        if self.hui:
            try:
                self.hui.radar.refresh_tasks(task_labels)
            except Exception:
                pass
