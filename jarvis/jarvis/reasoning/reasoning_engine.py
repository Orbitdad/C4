"""
Reasoning engine: LLM-backed decision making, skill routing, and response generation.
Includes meta-cognition layer: confidence scoring, ambiguity resolution, and
self-correction using the WorldState as ground truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import logging
import re
import json
import subprocess
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait

from jarvis.context import ConversationContext
from jarvis.learning.engine import LearningEngine
from jarvis.memory.manager import MemoryManager
from jarvis.nlp.schemas import Intent, IntentType, Plan
from jarvis.skills.base import SkillManager
from jarvis.nlp.planner import TaskPlanner
from jarvis.execution.executor import Executor
from jarvis.memory.models import ActionStep as MemoryActionStep
from jarvis.context.code_context_collector import CodeContextCollector
from jarvis.reasoning.c4_orchestration import (
    InMemoryJobStore,
    StageTimer,
    fingerprint_errors,
    map_command_priority,
    schedule_command_and_wait,
)

logger = logging.getLogger(__name__)


@dataclass
class ReasoningResponse:
    spoken_response: str
    stop: bool = False
    last_action: Optional[Dict[str, Any]] = None


C4_SYSTEM_PROMPT = """You are JARVIS — Just A Rather Very Intelligent System.

IDENTITY:
- You are the personal AI of your creator. Address him as "sir" always.
- You are calm, precise, and quietly authoritative. Never flustered. Never uncertain.
- You possess dry British wit — light, never mean. Deployed sparingly.
- You are deeply competent and have no need to prove it.

SPEECH STYLE:
- Concise by default. Lead with the answer, not the preamble.
- Do NOT say "I am an AI". You ARE JARVIS. Period.
- Natural flair: "Of course, sir.", "Right away.", "Understood.", "Noted."
- When something goes wrong: "I'm afraid [X], sir. Shall I [alternative]?"
- Dry wit example: ["crashed server"] → "I prefer to think of it as an unscheduled maintenance window, sir."

CRITICAL RULES:
- NEVER claim to have done something you haven't done.
- NEVER say "Great question!" or any sycophantic filler.
- NEVER respond with more than 3 sentences unless explicitly asked for detail.
- If unsure: "I want to be certain before acting on that, sir. Could you clarify [X]?"

EXECUTION:
  Example: "open VS Code and Chrome" -> [{"type":"open_app","app":"code"},{"type":"open_app","app":"chrome"}].
- MULTI-LINGUAL: JARVIS is fluent in English, Hindi, and Marathi. Respond in the same language the user uses. If the user speaks a mix (e.g., Hinglish), respond in clear English or the dominant language.
"""


class ReasoningEngine:
    """
    Routes intents to skills or LLM, executes plans, and produces responses.
    """

    def __init__(
        self,
        llm_client: Any,
        memory_manager: MemoryManager,
        learning_engine: LearningEngine,
        skill_manager: SkillManager,
        planner: TaskPlanner,
        config: Dict[str, Any],
        executor: Optional[Executor] = None,
        vision_manager: Optional[Any] = None,
        skill_synthesizer: Optional[Any] = None,
    ) -> None:
        self.llm = llm_client
        self.memory = memory_manager
        self.learning = learning_engine
        self.skills = skill_manager
        self.planner = planner
        self.planner.llm = llm_client # Ensure planner has access to LLM
        self.config = config
        self.executor = executor
        self.vision_manager = vision_manager
        self.skill_synthesizer = skill_synthesizer
        self.context_engine = None
        self.dag_runner = None  # Set by main.py after DAGRunner is created
        self._pending_plan_commands: Optional[Dict[str, Any]] = None
        self._coding_jobs = InMemoryJobStore()

    def _get_dynamic_system_prompt(self) -> str:
        from jarvis.core.world_state import world
        world_snap = world.get_snapshot()

        pm = getattr(self, "personality_manager", None)
        if pm:
            prompt = pm.get_personality_prompt(world_snapshot=world_snap)
        else:
            prompt = C4_SYSTEM_PROMPT
        if getattr(self, "context_engine", None) and self.context_engine.is_running:
            prompt += f"\n\n--- CURRENT AMBIENT CONTEXT ---\n{self.context_engine.get_context_snapshot()}"
        
        if getattr(self, "memory", None):
            episodes = self.memory.get_recent_episodes(5)
            if episodes:
                prompt += "\n\n--- RECENT MEMORY EPISODES ---\n" + "\n".join(episodes)
                
        if getattr(self, "semantic_db", None) and hasattr(self, "_last_query"):
            matches = self.semantic_db.search(self._last_query, top_k=4)
            if matches:
                 prompt += "\n\n--- RELEVANT RETRIEVED KNOWLEDGE ---\n(Use these facts if they pertain to the user's request)\n"
                 for m in matches:
                     prompt += f"- {m[1]['text']}\n"
                     
        if getattr(self, "temporal_habits", None):
            habits = self.temporal_habits.get_likely_habits()
            if habits:
                 prompt += f"\n\n--- PREDICTED USER HABITS ---\nLikely current actions: {', '.join(habits)}\n"
                
        # Inject WorldState summary for grounded reasoning
        from jarvis.core.world_state import world
        world_summary = world.get_context_summary()
        prompt += f"\n\n--- WORLD STATE ---\n{world_summary}"
        return prompt

    # ── Meta-Cognition ────────────────────────────────────────────────────────

    def _compute_confidence(self, intent: Intent, context: ConversationContext) -> float:
        """
        Score the confidence that we correctly understood the user's intent.
        Combines: intent parser score + memory hit rate + world state alignment.
        Returns a float 0.0 → 1.0.
        """
        score = 0.5  # Baseline

        # 1. Intent type clarity (QUESTION/COMMAND are clearer than UNKNOWN)
        type_scores = {
            IntentType.COMMAND: 0.2,
            IntentType.QUESTION: 0.2,
            IntentType.SMALL_TALK: 0.15,
            IntentType.LEARN_FACT: 0.25,
            IntentType.LEARN_COMMAND: 0.2,
            IntentType.RUN_LEARNED_COMMAND: 0.3,
            IntentType.FEEDBACK: 0.2,
            IntentType.CONTROL: 0.25,
            IntentType.MEMORY_QUERY: 0.2,
        }
        score += type_scores.get(intent.type, 0.0)

        # 2. Penalty for ambiguous demonstratives without visual context
        ambig_words = {"it", "this", "that", "him", "her", "them", "there"}
        if any(w in intent.raw_text.lower().split() for w in ambig_words):
            # Check if IntentFusion resolved it
            from jarvis.core.world_state import world
            fused = world.temporal_context.get("last_fusion_explanation", {}).get("value")
            if fused:
                score += 0.1  # Fusion resolved it — confidence boost
            else:
                score -= 0.15  # Unresolved ambiguity

        # 3. Memory hit bonus
        if intent.raw_text:
            facts = self.memory.find_facts(query=intent.raw_text[:50])
            if facts:
                score += 0.05 * min(len(facts), 3)

        # 4. Short utterances are ambiguous
        word_count = len(intent.raw_text.split())
        if word_count <= 2:
            score -= 0.1
        elif word_count >= 5:
            score += 0.05

        return max(0.0, min(1.0, round(score, 2)))

    def _meta_cognition_check(self, intent: Intent, context: ConversationContext, confidence: float) -> Optional[str]:
        """
        If confidence is low, try to self-correct using environmental context.
        Returns a clarification question string if JARVIS needs to ask,
        or None if it can proceed.

        Example: "Fix this" → looks at screen → finds error log → infers debugging task
        """
        if confidence >= 0.5:
            return None  # Confident enough

        raw = intent.raw_text.lower().strip()

        # Check visual context for grounding
        from jarvis.core.world_state import world
        snap = world.get_snapshot()
        active_window = snap["user_environment"].get("active_window") or ""
        activity = snap["user_environment"].get("inferred_activity") or "unknown"

        # Pattern: "fix this" / "do this" / "open this" with no clear target
        vague_commands = ["fix this", "do this", "open this", "run this", "close this", "check this"]
        if any(raw.startswith(v) for v in vague_commands) or raw in ["this", "that"]:
            if active_window:
                logger.info(f"[MetaCog] Enriching vague command from window: {active_window}")
                # Self-correct by injecting the window name into the raw text
                intent.raw_text = f"{intent.raw_text} [{active_window}]"
                world.set_temporal_context("meta_cognition_enriched", True, ttl_seconds=30)
                return None  # Self-corrected, can proceed
            else:
                return f"I want to help, but I'm not sure what you're referring to. Are you asking me to {raw.replace('this', 'something')} in {activity} mode?"

        # Pattern: very short utterance with no context
        if len(raw.split()) <= 2 and intent.type not in (IntentType.CONTROL, IntentType.SMALL_TALK):
            return "Could you give me a bit more detail on what you'd like me to do?"

        return None

    def handle_intent(self, intent: Intent, context: ConversationContext) -> ReasoningResponse:
        """Process intent and return response to speak."""
        self._last_query = intent.raw_text  # Save for semantic query

        # ── Emit transcript to HUI ────────────────────────────────────────
        hui = getattr(self, "_hui_window", None)
        if hui:
            try:
                hui.signals.transcript_user.emit(intent.raw_text)
                hui.signals.thinking_started.emit()
            except Exception:
                pass

        # ── Meta-Cognition Gate ────────────────────────────────────────────
        from jarvis.core.world_state import world
        confidence = self._compute_confidence(intent, context)
        world.update_cognitive_meta(
            last_intent_confidence=confidence,
            reasoning_active=True,
            last_reasoning_step=f"Processing: '{intent.raw_text[:60]}' (conf: {confidence:.0%})"
        )
        logger.debug(f"[ReasoningEngine] Intent confidence: {confidence:.2f} for '{intent.raw_text}'")

        # Emit "thinking" acknowledgement for long operations (LLM calls)
        voice = getattr(self, "_voice_output", None)
        if intent.type in (IntentType.QUESTION, IntentType.WRITE_CODE) and voice:
            import threading
            threading.Thread(
                target=lambda: voice.speak_thinking("Working on it, sir."),
                daemon=True
            ).start()

        clarification = self._meta_cognition_check(intent, context, confidence)
        if clarification:
            world.update_cognitive_meta(reasoning_active=False, last_reasoning_step="Asking clarification")
            if hui:
                try:
                    hui.signals.thinking_stopped.emit()
                except Exception:
                    pass
            return ReasoningResponse(spoken_response=clarification)

        # Stop command
        if intent.type == IntentType.CONTROL and intent.parsed_action == "stop":
            return ReasoningResponse(spoken_response="Shutting down. Goodbye.", stop=True)

        # Pending command confirmation
        if intent.parsed_action in ["confirm", "cancel"]:
            from jarvis.core.event_bus import bus, SystemEvent
            bus.publish(SystemEvent(name=f"voice.intent.{intent.parsed_action}", data={}))

            # Pending code-plan command execution confirmation
            if self._pending_plan_commands:
                pending = self._pending_plan_commands
                if intent.parsed_action == "confirm":
                    self._pending_plan_commands = None
                    cmd_results = self._execute_plan_commands(
                        plan=pending.get("plan") or {},
                        workspace_root=pending.get("workspace_root"),
                        request=pending.get("request", ""),
                        context=context,
                        force_execute_confirmed=True,
                    )
                    if cmd_results.get("confirmation_required"):
                        self._pending_plan_commands = pending
                        return ReasoningResponse(spoken_response=cmd_results.get("spoken_response", "Awaiting confirmation."))
                    return ReasoningResponse(spoken_response=cmd_results.get("spoken_response", "Commands executed."))
                else:
                    self._pending_plan_commands = None
                    return ReasoningResponse(spoken_response="Understood, sir. I cancelled command execution.")
            
            if self.learning._pending_command:
                if intent.parsed_action == "confirm":
                    cmd = self.learning.confirm_pending_command()
                    return ReasoningResponse(spoken_response=f"Command saved: {cmd.name}")
                else:
                    self.learning.cancel_pending_command()
                    return ReasoningResponse(spoken_response="Cancelled.")
            return ReasoningResponse(spoken_response="")

        # Learning intents
        if intent.type == IntentType.LEARN_FACT:
            return self._handle_learn_fact(intent)
        if intent.type == IntentType.LEARN_COMMAND:
            return self._handle_learn_command(intent, context)

        # Control: forget fact, reset memory
        if intent.type == IntentType.CONTROL:
            return self._handle_control(intent)

        # Feedback
        if intent.type == IntentType.FEEDBACK:
            effect, msg = self.learning.apply_feedback(
                utterance=intent.raw_text,
                last_action=context.last_action,
                correction_text=intent.params.get("correction"),
            )
            return ReasoningResponse(spoken_response=msg)

        # Memory query
        if intent.type == IntentType.MEMORY_QUERY:
            return self._handle_memory_query(intent)

        # Small talk
        if intent.type == IntentType.SMALL_TALK:
            result = self.skills.execute("small_talk", intent, context)
            return ReasoningResponse(spoken_response=result.text or "")

        # Run learned command
        if intent.type == IntentType.RUN_LEARNED_COMMAND:
            return self._handle_run_learned_command(intent, context)

        # Commands (open app, file ops, time, search)
        if intent.type == IntentType.COMMAND:
            if intent.parsed_action == "visual_status" and self.vision_manager:
                status = self.vision_manager.get_status()
                prompt = f"The vision system reports: {status}. Provide a concise, professional description of this to the user."
                response = self.llm.generate(
                    prompt,
                    system_message=self._get_dynamic_system_prompt(),
                    history=context.to_prompt_history()
                )
                return ReasoningResponse(spoken_response=response if response else status)
            return self._handle_command(intent, context)

        # Questions: use LLM
        if intent.type == IntentType.QUESTION:
            return self._handle_question(intent, context)

        # Write code
        if intent.type == IntentType.WRITE_CODE:
            return self._handle_write_code(intent, context)

        # Sense Gap / Skill Synthesis
        if intent.type == IntentType.SENSE_GAP:
            if not self.skill_synthesizer:
                return ReasoningResponse(spoken_response="I cannot synthesize new skills right now.")
            voice = getattr(self, "_voice_output", None)
            if voice:
                 import threading
                 threading.Thread(target=lambda: voice.speak_thinking("Synthesizing new capability, sir."), daemon=True).start()
            res = self.skill_synthesizer.synthesize(intent.raw_text, gap_description=intent.params.get("gap_description", ""))
            return ReasoningResponse(spoken_response=res["message"])

        world.update_cognitive_meta(reasoning_active=False, last_reasoning_step="Idle")
        hui = getattr(self, "_hui_window", None)
        if hui:
            try:
                hui.signals.thinking_stopped.emit()
            except Exception:
                pass
        return ReasoningResponse(spoken_response="I'm afraid I didn't quite catch that, sir. Could you rephrase?")

    def _handle_learn_fact(self, intent: Intent) -> ReasoningResponse:
        params = intent.params or {}
        key = params.get("key", "fact")
        value = params.get("value", "")
        category = params.get("category", "user")
        if not value:
            return ReasoningResponse(spoken_response="What would you like me to remember?")
        self.learning.learn_fact(key=key, value=value, category=category)
        return ReasoningResponse(
            spoken_response=f"I will remember that {value}."
        )

    def _handle_learn_command(
        self, intent: Intent, context: ConversationContext
    ) -> ReasoningResponse:
        params = intent.params or {}
        trigger = params.get("trigger_phrase", "").strip()
        steps_desc = params.get("steps_description", "")
        if not trigger or not steps_desc:
            return ReasoningResponse(
                spoken_response="Please specify both the trigger phrase and what I should do."
            )

        # Use LLM to convert steps description to structured steps
        prompt = f"""Convert this into a JSON array of action steps. Each step has "type" and params.
User said: "When I say '{trigger}', {steps_desc}"

Return ONLY a JSON array, e.g. [{"type":"open_app","app":"code"},{"type":"open_url","url":"https://example.com"}]
Supported types: open_app, open_url, create_file, play_media, run_command."""

        response = self.llm.generate(
            prompt,
            system_message=self._get_dynamic_system_prompt(),
            history=context.to_prompt_history()
        )
        steps = self._parse_steps_from_llm(response)
        if not steps:
            # Fallback: single open_app step
            steps = [MemoryActionStep(type="open_app", params={"app": steps_desc[:50]})]

        self.learning.set_pending_command(trigger, steps, steps_desc)
        summary = ", ".join(
            f"{s.type}({', '.join(f'{k}={v}' for k, v in s.params.items())})"
            for s in steps
        )
        return ReasoningResponse(
            spoken_response=f"To confirm, when you say '{trigger}', I will: {summary}. Should I save this command?"
        )

    def _parse_steps_from_llm(self, text: str) -> List[MemoryActionStep]:
        import json
        import re
        text = text.strip()
        # Try to extract JSON array
        m = re.search(r'\[[\s\S]*\]', text)
        if m:
            try:
                arr = json.loads(m.group())
                steps = []
                for item in arr:
                    if isinstance(item, dict):
                        t = item.pop("type", "open_app")
                        steps.append(MemoryActionStep(type=t, params=item))
                return steps
            except json.JSONDecodeError:
                pass
        return []

    def _handle_control(self, intent: Intent) -> ReasoningResponse:
        action = intent.parsed_action
        if action == "reset_memory":
            count = self.learning.reset_memory(["all"])
            return ReasoningResponse(spoken_response=f"Memory reset. Cleared {count} items.")
        if action == "forget_fact":
            query = intent.params.get("query", "")
            ok = self.learning.forget_fact(query)
            return ReasoningResponse(
                spoken_response="Fact forgotten." if ok else "I could not find that fact."
            )
        return ReasoningResponse(spoken_response="Done.")

    def _handle_memory_query(self, intent: Intent) -> ReasoningResponse:
        query = intent.params.get("query", "")
        facts, commands = self.learning.explain_knowledge(query=query)
        parts = []
        if facts:
            parts.append(f"I have {len(facts)} fact(s): " + "; ".join(f"{f.key}={f.value}" for f in facts[:5]))
        if commands:
            parts.append(f"I have {len(commands)} learned command(s): " + ", ".join(c.name for c in commands[:5]))
        if not parts:
            return ReasoningResponse(spoken_response="I have not learned any facts or commands yet.")
        return ReasoningResponse(spoken_response=" ".join(parts))

    def _handle_run_learned_command(
        self, intent: Intent, context: ConversationContext
    ) -> ReasoningResponse:
        cmd = intent.params.get("command")
        if not cmd or not self.executor:
            return ReasoningResponse(spoken_response="I could not execute that command.")
        if cmd.confirmation_required:
            # For now we execute; in full flow we could ask first
            pass
        results = []
        for step in cmd.steps:
            r = self.executor.execute_step(step)
            results.append(r.get("message", "Done"))
        self.memory.update_command(cmd.id, last_run_at=_now_iso(), run_count=cmd.run_count + 1)
        return ReasoningResponse(
            spoken_response=f"Executed. {results[0] if results else 'Done.'}",
            last_action={"command_id": cmd.id, "command": cmd},
        )

    def _handle_command(
        self, intent: Intent, context: ConversationContext
    ) -> ReasoningResponse:
        is_complex = any(w in intent.raw_text.lower().split() for w in ["and", "then", "after", "if"])
        has_ambiguity = any(w in intent.raw_text.lower().split() for w in ["it", "that", "him", "her", "them", "there"])

        # 1. FAST PATH (Latency Strategy)
        if not is_complex and not has_ambiguity:
            # Bypass LLM completely by passing context=None so planner doesn't resolve references using LLM
            plan = self.planner.plan(intent, context=None)
            if hasattr(plan, 'steps') and plan.steps and self.executor:
                results = []
                for step in plan.steps:
                    r = self.executor.execute_step(step)
                    results.append(r.get("message", "Done"))
                return ReasoningResponse(
                    spoken_response=results[0] if results else "Done.",
                    last_action={"plan": plan.to_dict(), "path": "fast_path"},
                )

        # Route complex graphs for chained multi-commands
        if is_complex:
            graph = self.planner.build_execution_graph(intent, context)
            if graph and "nodes" in graph:
                from jarvis.execution.pipeline import ExecutionPipeline
                pipeline = ExecutionPipeline(self.executor, planner=self.planner)
                final_spoken = pipeline.run_graph(graph)
                return ReasoningResponse(spoken_response=final_spoken, last_action={"path": "slow_path_graph"})
                
        # 2. SLOW PATH (LLM reference resolution via Context)
        plan = self.planner.plan(intent, context)
        
        # Backward compatibility for old flat planner
        if hasattr(plan, 'steps') and plan.steps and self.executor:
            results = []
            for step in plan.steps:
                r = self.executor.execute_step(step)
                results.append(r.get("message", "Done"))
            return ReasoningResponse(
                spoken_response=results[0] if results else "Done.",
                last_action={"plan": plan.to_dict(), "path": "slow_path"},
            )
            
        # Advanced Execution Pipeline logic (v2 API)
        if isinstance(plan, dict) and "nodes" in plan:
            from jarvis.execution.pipeline import ExecutionPipeline
            pipeline = ExecutionPipeline(self.executor, planner=self.planner)
            final_spoken = pipeline.run_graph(plan)
            return ReasoningResponse(spoken_response=final_spoken, last_action={"path": "slow_path_v2"})

        result = self.skills.execute_by_intent(intent, context)
        return ReasoningResponse(spoken_response=result.text or "I could not complete that.")

    def _handle_question(self, intent: Intent, context: ConversationContext) -> ReasoningResponse:
        query = intent.params.get("query", intent.raw_text)
        facts = self._find_relevant_facts(query)
        facts_str = ""
        if facts:
            facts_str = "\nRelevant stored facts: " + ", ".join(f"{f.key}={f.value}" for f in facts[:5])

        # Fallback: answer from facts when LLM fails or question is about stored info
        if facts and self._is_fact_based_question(query):
            answer = self._answer_from_facts(query, facts)
            if answer:
                return ReasoningResponse(spoken_response=answer)

        prompt = f"User asked: {query}{facts_str}\nProvide a concise, accurate answer in 1-3 sentences."
        answer = self.llm.generate(
            prompt,
            system_message=self._get_dynamic_system_prompt(),
            history=context.to_prompt_history()
        )
        if not answer or answer.startswith("[Error"):
            # Fallback to facts when LLM fails
            if facts:
                return ReasoningResponse(
                    spoken_response=self._answer_from_facts(query, facts) or "I could not process that. Please try again."
                )
            return ReasoningResponse(spoken_response="I could not process that question. Please try again.")
        return ReasoningResponse(spoken_response=answer)

    def _find_relevant_facts(self, query: str) -> List[Any]:
        """Find facts using full query and keyword-based fallback."""
        q = (query or "").strip()[:100].lower()
        
        # 1. Direct search by full query
        facts = self.memory.find_facts(query=q)
        if facts:
            return facts
            
        # 2. Extract significant keywords (exclude stop words)
        stop = {"what", "the", "and", "for", "you", "your", "my", "is", "are", "can", "how", "when", "where", "why", "who", "which", "does", "do", "did", "am"}
        keywords = [w for w in q.split() if len(w) > 2 and w not in stop]
        
        if keywords:
            # Try searching with combined keywords or most significant one
            facts = self.memory.find_facts(query=keywords[0])
            if facts:
                return facts

        # 3. Only return all user facts for very specific "tell me about myself" queries
        personal_patterns = ["who am i", "what do you know about me", "tell me about myself", "my profile"]
        if any(p in q for p in personal_patterns):
            return self.memory.find_facts(category="user")
            
        return []

    def _is_fact_based_question(self, query: str) -> bool:
        """Heuristic: question likely answerable from stored facts."""
        q = query.lower().strip()
        # Stricter patterns to avoid general "my" or "i" triggers
        patterns = [
            "my name", "what is my name", "who am i", "call me", 
            "my laptop", "what's my name", "what is my",
            "remember about me"
        ]
        return any(p in q for p in patterns)

    def _answer_from_facts(self, query: str, facts: List[Any]) -> Optional[str]:
        """Answer from stored facts when possible, ensuring intent matches key."""
        q = query.lower()
        
        # Priority mapping: query keywords to fact keys
        mapping = {
            "name": "name",
            "laptop": "laptop",
            "birthday": "birthday",
            "email": "email",
            "phone": "phone"
        }
        
        for keyword, key in mapping.items():
            if keyword in q:
                for f in facts:
                    if f.key == key or key in f.key:
                        val = str(f.value).capitalize()
                        if key == "name":
                            return f"Your name is {val}."
                        return f"Your {key.replace('_', ' ')} is {val}."
        
        # If we have facts but no keyword match, don't just guess "name"
        # unless it was a very specific "who am i" / "about me" query
        if any(p in q for p in ["who am i", "about me", "myself"]):
            name_fact = next((f for f in facts if f.key == "name"), None)
            if name_fact:
                return f"You are {str(name_fact.value).capitalize()}."

        return None


    def _run_role(self, role: str, prompt: str, context: Optional[ConversationContext] = None, system_message: Optional[str] = None) -> str:
        if hasattr(self.llm, "call_llm"):
            return self.llm.call_llm(
                role=role,
                prompt=prompt,
                system_message=system_message or self._get_dynamic_system_prompt(),
                history=context.to_prompt_history() if context else None,
            )
        return self.llm.generate(
            prompt,
            system_message=system_message or self._get_dynamic_system_prompt(),
            history=context.to_prompt_history() if context else None,
        )

    def _is_destructive_command(self, cmd: str) -> bool:
        c = (cmd or "").lower()
        banned = [
            "rm -rf",
            "rmdir /s",
            "del /f /s /q",
            "format ",
            "diskpart",
            "shutdown ",
            "reboot",
            "mkfs",
            "dd if=",
            "reg delete",
            "takeown",
        ]
        return any(b in c for b in banned)

    def _parse_command_errors(self, command_results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        errors: List[Dict[str, Any]] = []
        for result in command_results:
            combined = "\n".join(
                [
                    str(result.get("stderr", "")),
                    str(result.get("stdout", "")),
                    str(result.get("message", "")),
                ]
            ).strip()
            if result.get("success") and not re.search(r"\berror\b", combined, re.IGNORECASE):
                continue

            # TypeScript / JS stack and compiler formats
            patterns = [
                r"(?P<file>[A-Za-z0-9_\-./\\]+)\((?P<line>\d+),(?P<col>\d+)\):\s*error\s*(?P<etype>[A-Z0-9]+)?:?\s*(?P<msg>.+)",
                r"(?P<file>[A-Za-z0-9_\-./\\]+):(?P<line>\d+):(?P<col>\d+)\s*-\s*error\s*(?P<etype>[A-Z0-9]+)?:?\s*(?P<msg>.+)",
                r"at\s+(?P<file>[A-Za-z0-9_\-./\\]+):(?P<line>\d+):(?P<col>\d+)",
            ]
            matched = False
            for line in combined.splitlines():
                for pat in patterns:
                    m = re.search(pat, line)
                    if not m:
                        continue
                    matched = True
                    errors.append(
                        {
                            "file": (m.groupdict().get("file") or "").replace("\\", "/"),
                            "line": int(m.groupdict().get("line") or 0),
                            "error_type": m.groupdict().get("etype") or "build_error",
                            "message": (m.groupdict().get("msg") or line).strip(),
                        }
                    )
            if not matched:
                errors.append(
                    {
                        "file": "",
                        "line": 0,
                        "error_type": "command_failure",
                        "message": combined[:2000] if combined else "Command failed.",
                    }
                )
        return errors

    def _execute_plan_commands(
        self,
        plan: Dict[str, Any],
        workspace_root: Optional[str],
        request: str,
        context: ConversationContext,
        force_execute_confirmed: bool = False,
        job_id: str = "",
    ) -> Dict[str, Any]:
        if not self.executor:
            return {"spoken_response": "I do not have command execution permissions.", "command_results": []}

        commands = plan.get("commands") or []
        if not commands:
            return {"spoken_response": "No post-generation commands were defined.", "command_results": []}

        cwd = workspace_root or ""
        results: List[Dict[str, Any]] = []
        confirmation_commands: List[Dict[str, Any]] = []
        for cmd_obj in commands:
            if not isinstance(cmd_obj, dict):
                continue
            cmd = (cmd_obj.get("cmd") or "").strip()
            needs_confirm = bool(cmd_obj.get("requires_confirmation", False))
            if not cmd:
                continue
            if self._is_destructive_command(cmd):
                results.append({"success": False, "command": cmd, "message": "Blocked destructive command by safety policy."})
                continue
            if needs_confirm and not force_execute_confirmed:
                confirmation_commands.append(cmd_obj)
                continue

            prio_raw = str(cmd_obj.get("priority", "")).lower().strip()
            prio = map_command_priority(cmd)
            if prio_raw == "critical":
                from jarvis.core.event_bus import EventPriority
                prio = EventPriority.CRITICAL
            elif prio_raw == "high":
                from jarvis.core.event_bus import EventPriority
                prio = EventPriority.HIGH
            elif prio_raw == "normal":
                from jarvis.core.event_bus import EventPriority
                prio = EventPriority.NORMAL
            elif prio_raw == "low":
                from jarvis.core.event_bus import EventPriority
                prio = EventPriority.LOW
            res = schedule_command_and_wait(
                command=cmd,
                cwd=cwd,
                retries=int(cmd_obj.get("retries", 0) or 0),
                delay_seconds=float(cmd_obj.get("delay_seconds", 0.0) or 0.0),
                cancel_token=str(cmd_obj.get("cancel_token", "") or job_id or ""),
                retry_delay_seconds=float(cmd_obj.get("retry_delay_seconds", 0.0) or 0.0),
                priority=prio,
            )
            results.append(res)

        if confirmation_commands:
            pending = dict(self._pending_plan_commands or {})
            pending["plan"] = dict(plan)
            pending["workspace_root"] = cwd
            pending["request"] = request
            pending_plan = dict(plan)
            pending_plan["commands"] = confirmation_commands
            self._pending_plan_commands = {"plan": pending_plan, "workspace_root": cwd, "request": request}
            return {
                "confirmation_required": True,
                "spoken_response": "I generated the code, sir. Say confirm to run the pending setup commands.",
                "command_results": results,
            }

        return {
            "spoken_response": "Post-generation commands executed.",
            "command_results": results,
            "confirmation_required": False,
        }

    def _ensure_git_repo(self, workspace_root: str) -> None:
        git_dir = subprocess.run(
            "git rev-parse --git-dir",
            shell=True,
            capture_output=True,
            text=True,
            cwd=workspace_root,
        )
        if git_dir.returncode != 0:
            subprocess.run("git init", shell=True, capture_output=True, text=True, cwd=workspace_root)

    def _git_commit_phase(self, workspace_root: str, request: str, phase: str) -> None:
        self._ensure_git_repo(workspace_root)
        subprocess.run("git add -A", shell=True, capture_output=True, text=True, cwd=workspace_root)
        changed = subprocess.run(
            "git diff --cached --quiet",
            shell=True,
            capture_output=True,
            text=True,
            cwd=workspace_root,
        )
        if changed.returncode == 0:
            return
        safe_request = " ".join((request or "").split())[:60]
        msg = f"C4: {safe_request} (auto-generated) [{phase}]"
        subprocess.run(
            f'git commit -m "{msg}"',
            shell=True,
            capture_output=True,
            text=True,
            cwd=workspace_root,
        )

    def _run_file_task(
        self,
        f: Dict[str, Any],
        request: str,
        plan: Dict[str, Any],
        context_bundle: Dict[str, Any],
        workspace_root: Any,
        context: ConversationContext,
    ) -> Optional[str]:
        from jarvis.reasoning.prompts import FILE_GENERATION_PROMPT, FILE_MODIFY_FULL_REWRITE_PROMPT, FILE_MODIFY_UNIFIED_DIFF_PROMPT
        from pathlib import Path
        rel_path = (f.get("path") or "").strip()
        mode = (f.get("mode") or "create").strip().lower()
        if not rel_path:
            return None
        abs_path = (workspace_root / rel_path).resolve()
        exists = abs_path.is_file()
        file_spec_json = json.dumps(f, indent=2)
        plan_json = json.dumps(plan, indent=2)
        bundle_json = json.dumps(context_bundle, indent=2)
        if exists and mode in ("update", "patch"):
            existing_text = Path(abs_path).read_text(encoding="utf-8", errors="replace")
            if mode == "patch":
                prompt = FILE_MODIFY_UNIFIED_DIFF_PROMPT.format(
                    request=request, path=rel_path, project_plan_json=plan_json, context_bundle=bundle_json, existing_file=existing_text
                )
                diff = self._run_role("coder", prompt, context=context)
                res = self.executor.execute_step({"type": "patch_file", "path": str(abs_path), "diff": diff})
                if not res.get("success"):
                    prompt = FILE_MODIFY_FULL_REWRITE_PROMPT.format(
                        request=request, path=rel_path, project_plan_json=plan_json, context_bundle=bundle_json, existing_file=existing_text
                    )
                    rewritten = self._run_role("coder", prompt, context=context)
                    res = self.executor.execute_step({"type": "update_file", "path": str(abs_path), "content": rewritten})
                if res.get("success"):
                    return f"updated {rel_path}"
                return None
            prompt = FILE_MODIFY_FULL_REWRITE_PROMPT.format(
                request=request, path=rel_path, project_plan_json=plan_json, context_bundle=bundle_json, existing_file=existing_text
            )
            rewritten = self._run_role("coder", prompt, context=context)
            res = self.executor.execute_step({"type": "update_file", "path": str(abs_path), "content": rewritten})
            if res.get("success"):
                return f"updated {rel_path}"
            return None
        prompt = FILE_GENERATION_PROMPT.format(
            file_spec_json=file_spec_json,
            project_plan_json=plan_json,
            context_bundle=bundle_json,
        )
        content = self._run_role("coder", prompt, context=context)
        res = self.executor.execute_step({"type": "create_file", "path": str(abs_path), "content": content})
        if res.get("success"):
            return f"created {rel_path}"
        return None

    def _parallel_generate_files(
        self,
        files: List[Dict[str, Any]],
        request: str,
        plan: Dict[str, Any],
        context_bundle: Dict[str, Any],
        workspace_root: Any,
        context: ConversationContext,
    ) -> List[str]:
        max_workers = int(((self.config.get("llm") or {}).get("codegen_workers") or 4))
        max_workers = max(3, min(max_workers, 5))
        remaining = {(f.get("path") or "").strip(): f for f in files if isinstance(f, dict) and (f.get("path") or "").strip()}
        done: set[str] = set()
        running: Dict[Any, str] = {}
        executed: List[str] = []

        with ThreadPoolExecutor(max_workers=max_workers) as pool:
            while remaining or running:
                launched = False
                ready_paths = []
                for path, spec in remaining.items():
                    deps = [d for d in (spec.get("depends_on") or []) if isinstance(d, str)]
                    if all(d in done or d not in remaining for d in deps):
                        ready_paths.append(path)
                for path in ready_paths:
                    if len(running) >= max_workers:
                        break
                    spec = remaining.pop(path)
                    fut = pool.submit(self._run_file_task, spec, request, plan, context_bundle, workspace_root, context)
                    running[fut] = path
                    launched = True
                if not running and remaining and not launched:
                    # Break cyclic depends_on deadlock by running one file.
                    path, spec = remaining.popitem()
                    fut = pool.submit(self._run_file_task, spec, request, plan, context_bundle, workspace_root, context)
                    running[fut] = path
                if not running:
                    continue
                done_futures, _ = wait(list(running.keys()), return_when=FIRST_COMPLETED)
                for fut in done_futures:
                    path = running.pop(fut)
                    done.add(path)
                    try:
                        item = fut.result()
                        if item:
                            executed.append(item)
                    except Exception:
                        pass
        return executed

    def _debug_fix_iteration(
        self,
        request: str,
        plan: Dict[str, Any],
        context_bundle: Dict[str, Any],
        errors: List[Dict[str, Any]],
        workspace_root: str,
        context: ConversationContext,
    ) -> Dict[str, Any]:
        from jarvis.reasoning.prompts import DEBUG_FIX_ERRORS_PROMPT, DEBUG_UNIFIED_DIFF_PROMPT, FILE_MODIFY_FULL_REWRITE_PROMPT
        from pathlib import Path

        debug_prompt = DEBUG_FIX_ERRORS_PROMPT.format(
            request=request,
            project_plan_json=json.dumps(plan, indent=2),
            errors_json=json.dumps(errors, indent=2),
            context_bundle=json.dumps(context_bundle, indent=2),
        )
        decision_raw = self._run_role("debugger", debug_prompt, context=context)
        try:
            decision = json.loads((decision_raw or "").strip())
        except Exception:
            return {"changed": False, "message": "debugger_decision_invalid_json"}

        strategy = (decision.get("strategy") or "noop").lower()
        target_file = (decision.get("target_file") or "").strip()
        if strategy == "noop" or not target_file:
            return {"changed": False, "message": "no_fix_suggested"}

        target_abs = str((Path(workspace_root) / target_file).resolve())
        if not Path(target_abs).is_file():
            return {"changed": False, "message": "target_file_missing"}

        existing_file = Path(target_abs).read_text(encoding="utf-8", errors="replace")
        if strategy == "patch":
            patch_prompt = DEBUG_UNIFIED_DIFF_PROMPT.format(
                target_file=target_file,
                request=request,
                errors_json=json.dumps(errors, indent=2),
                project_plan_json=json.dumps(plan, indent=2),
                context_bundle=json.dumps(context_bundle, indent=2),
                existing_file=existing_file,
            )
            diff = self._run_role("debugger", patch_prompt, context=context)
            if not diff.strip():
                return {"changed": False, "message": "empty_patch"}
            patch_res = self.executor.execute_step({"type": "patch_file", "path": target_abs, "diff": diff})
            return {"changed": bool(patch_res.get("success")), "mode": "patch", "result": patch_res, "target_file": target_file, "diff": diff}

        rewrite_prompt = FILE_MODIFY_FULL_REWRITE_PROMPT.format(
            request=request,
            path=target_file,
            project_plan_json=json.dumps(plan, indent=2),
            context_bundle=json.dumps(context_bundle, indent=2),
            existing_file=existing_file,
        )
        new_content = self._run_role("debugger", rewrite_prompt, context=context)
        if not new_content.strip() or new_content.strip() == existing_file.strip():
            return {"changed": False, "message": "no_meaningful_diff"}
        rewrite_res = self.executor.execute_step({"type": "update_file", "path": target_abs, "content": new_content})
        return {"changed": bool(rewrite_res.get("success")), "mode": "rewrite", "result": rewrite_res, "target_file": target_file}

    def _handle_write_code(self, intent: Intent, context: ConversationContext) -> ReasoningResponse:
        from pathlib import Path
        from jarvis.reasoning.prompts import (
            PROJECT_PLAN_PROMPT,
        )

        request = (intent.params.get("query") or intent.raw_text or "").strip()
        if not request:
            return ReasoningResponse(spoken_response="What should I build, sir?")
        if not self.executor:
            return ReasoningResponse(spoken_response="I do not have execution permissions for code generation, sir.")

        workspace_root = Path.cwd()
        job = self._coding_jobs.create(request=request)
        self._coding_jobs.add_step(job.job_id, "plan", "running")
        collector = CodeContextCollector(workspace_root=workspace_root, semantic_db=getattr(self, "semantic_db", None))
        try:
            collector.index_workspace(max_files=80, chunk_lines=120, overlap=20)
        except Exception:
            pass
        context_bundle = collector.build_context_bundle(request=request, max_ranked_files=14, max_snippets=8, include_full_files=3)

        workspace_context = json.dumps(
            {
                "project_type": context_bundle.get("project_type"),
                "ranked_files": context_bundle.get("ranked_files", [])[:10],
                "workspace_root": context_bundle.get("workspace_root"),
            },
            indent=2,
        )
        plan_prompt = PROJECT_PLAN_PROMPT.format(request=request, workspace_context=workspace_context)
        t = StageTimer(self._coding_jobs, job.job_id, "plan", "planner")
        plan_text = self._run_role("planner", plan_prompt, context=context)
        t.close()
        if not (plan_text or "").strip():
            self._coding_jobs.update_status(job.job_id, "failed", "plan")
            return ReasoningResponse(spoken_response="I could not generate a project plan, sir.")
        try:
            plan = json.loads(plan_text.strip())
        except Exception:
            fixed = self._run_role("debugger", f"Repair this into valid JSON ONLY (no commentary). Input:\n{plan_text}", context=context, system_message="Return ONLY valid JSON.")
            try:
                plan = json.loads((fixed or "").strip())
            except Exception:
                self._coding_jobs.update_status(job.job_id, "failed", "plan")
                return ReasoningResponse(spoken_response="My planner output was invalid JSON, sir.")

        files = plan.get("files") or []
        if not isinstance(files, list) or not files:
            self._coding_jobs.update_status(job.job_id, "failed", "plan")
            return ReasoningResponse(spoken_response="The plan has no files to generate, sir.")
        self._coding_jobs.update_status(job.job_id, "running", "generation")
        self._git_commit_phase(str(workspace_root), request, "plan")

        t = StageTimer(self._coding_jobs, job.job_id, "generate", "coder")
        executed = self._parallel_generate_files(files, request, plan, context_bundle, workspace_root, context)
        t.close()
        self._git_commit_phase(str(workspace_root), request, "generation")

        # Command execution phase (non-confirmation commands first)
        self._coding_jobs.update_status(job.job_id, "running", "execute")
        t = StageTimer(self._coding_jobs, job.job_id, "execute", "scheduler")
        cmd_phase = self._execute_plan_commands(
            plan=plan,
            workspace_root=str(workspace_root),
            request=request,
            context=context,
            job_id=job.job_id,
        )
        t.close()
        command_results = list(cmd_phase.get("command_results") or [])

        # Debug loop: generate -> run -> fail -> fix -> rerun
        loop_max = int(((self.config.get("llm") or {}).get("debug_max_loops") or 4))
        loop_max = max(1, min(loop_max, 5))
        loop_count = 0
        debug_actions: List[str] = []
        previous_signatures: set[str] = set()

        while loop_count < loop_max:
            errors = self._parse_command_errors(command_results)
            if not errors:
                break
            self._coding_jobs.update_status(job.job_id, "fixing", "debug")
            loop_count += 1
            signature = fingerprint_errors(errors)
            if signature in previous_signatures or not self._coding_jobs.add_error_fingerprint(job.job_id, signature):
                break
            previous_signatures.add(signature)

            impacted_files = sorted({(e.get("file") or "").strip().replace("\\", "/") for e in errors if (e.get("file") or "").strip()})
            if not impacted_files:
                fix = self._debug_fix_iteration(
                    request=request,
                    plan=plan,
                    context_bundle=context_bundle,
                    errors=errors,
                    workspace_root=str(workspace_root),
                    context=context,
                )
                if not fix.get("changed"):
                    break
                debug_actions.append(f"{fix.get('mode', 'fix')}:{fix.get('target_file', '')}")
            else:
                changed_any = False
                for file_path in impacted_files:
                    one_fix = self._debug_fix_iteration(
                        request=request,
                        plan=plan,
                        context_bundle=context_bundle,
                        errors=[e for e in errors if (e.get("file") or "").replace("\\", "/") == file_path] or errors,
                        workspace_root=str(workspace_root),
                        context=context,
                    )
                    if one_fix.get("changed"):
                        changed_any = True
                        debug_actions.append(f"{one_fix.get('mode', 'fix')}:{one_fix.get('target_file', file_path)}")
                if not changed_any:
                    break
            self._git_commit_phase(str(workspace_root), request, "fix")

            # Re-run only non-destructive commands
            rerun_results: List[Dict[str, Any]] = []
            for cmd_obj in (plan.get("commands") or []):
                if not isinstance(cmd_obj, dict):
                    continue
                cmd = (cmd_obj.get("cmd") or "").strip()
                if not cmd or self._is_destructive_command(cmd):
                    continue
                if bool(cmd_obj.get("requires_confirmation", False)):
                    continue
                rerun_results.append(
                    schedule_command_and_wait(
                        command=cmd,
                        cwd=str(workspace_root),
                        retries=int(cmd_obj.get("retries", 0) or 0),
                        delay_seconds=0.0,
                        cancel_token=str(cmd_obj.get("cancel_token", "") or job.job_id),
                    )
                )
            command_results = rerun_results
            if not command_results:
                break

        try:
            for item in (executed[:8] + [f"Debug loop actions: {', '.join(debug_actions)}"] if debug_actions else executed[:8]):
                self.memory.add_episode(f"Completed coding task: {item}.")
        except Exception:
            pass

        if cmd_phase.get("confirmation_required"):
            self._coding_jobs.update_status(job.job_id, "planned", "await_confirmation")
            return ReasoningResponse(
                spoken_response=cmd_phase.get("spoken_response", "Generated code. Awaiting confirmation for commands."),
                last_action={"write_code": {"job_id": job.job_id, "status": "planned", "plan": plan, "executed": executed, "pending_commands": True}},
            )

        final_errors = self._parse_command_errors(command_results)
        if final_errors:
            self._coding_jobs.update_status(job.job_id, "failed", "complete")
            spoken = "Generation complete, sir. Build still reports errors after debug iterations."
        else:
            self._coding_jobs.update_status(job.job_id, "completed", "complete")
            spoken = "Generation and validation complete, sir."

        explain = self._run_role(
            "explainer",
            f"Summarize in 1-2 sentences. Files: {executed}. Debug actions: {debug_actions}. Remaining errors: {final_errors[:3]}",
            context=context,
        )
        if explain and not explain.startswith("[Error"):
            spoken = explain.strip()

        return ReasoningResponse(
            spoken_response=spoken,
            last_action={
                "write_code": {
                    "job_id": job.job_id,
                    "status": self._coding_jobs.get(job.job_id).status if self._coding_jobs.get(job.job_id) else "unknown",
                    "plan": plan,
                    "executed": executed,
                    "command_results": command_results,
                    "debug_actions": debug_actions,
                    "loops": loop_count,
                    "errors_remaining": len(final_errors),
                    "telemetry": self._coding_jobs.get(job.job_id).telemetry if self._coding_jobs.get(job.job_id) else [],
                }
            },
        )


def _now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()
