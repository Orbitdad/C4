from __future__ import annotations

from pathlib import Path

from .config_loader import load_config
from .logging_utils import configure_logging, get_logger
from .context import ConversationContext
from .voice.input import SpeechRecognitionInput
from .voice.output import create_voice_output
from .nlp.intent_parser import IntentParser
from .nlp.planner import TaskPlanner
from .memory.manager import MemoryManager
from .learning.engine import LearningEngine
from .reasoning.llm_router import RoleBasedLLM
from .reasoning.reasoning_engine import ReasoningEngine
from .skills.base import SkillManager
from .skills.small_talk import SmallTalkSkill
from .skills.system_control import SystemControlSkill
from .skills.file_ops import FileOpsSkill
from .skills.web_search import WebSearchSkill
from .skills.learn_fact_skill import LearnFactSkill
from .skills.learn_command_skill import LearnCommandSkill
from .skills.media import MediaSkill
from .skills.keyboard import KeyboardSkill
from .skills.file_search import FileSearchSkill
from .skills.weather import WeatherSkill
from .skills.reminder import ReminderSkill
from .skills.volume_control import VolumeControlSkill
from .skills.screenshot import ScreenshotSkill
from .skills.notes import NotesSkill
from .skills.clipboard import ClipboardSkill
from .skills.os_god_mode import OSGodModeSkill
from .skills.smart_home import SmartHomeSkill
from .skills.model_viewer import ModelViewerSkill
from .vision.manager import VisionManager
from .execution.executor import Executor
from .vision.gesture.action_executor import ActionExecutor

# Core subsystem imports
from .core.orchestrator import CognitiveOrchestrator
from .core.resource_governor import ResourceGovernor
from .core.event_bus import bus
from .core.watchdog import Watchdog
from .core.world_state import world
from .core.attention import attention
from .core.user_model import init_user_model
from .execution.scheduler import ActionScheduler
from .execution.dag_runner import DAGRunner
from .nlp.personality import PersonalityManager
from .memory.semantic_db import SemanticDB
from .memory.vector_store import VectorMemoryStore
from .memory.retriever import MemoryRetriever
from .memory.prompt_builder import MemoryPromptBuilder
from .memory.writer import MemoryWriter
from .learning.feedback_loop import ReinforcementFeedbackLoop
from .vision.semantic_vision import SemanticVisionEngine
from .nlp.intent_fusion import UnifiedIntentEngine
from .context.forecaster import TrendForecaster
from .learning.temporal import TemporalHabitEngine
from .agents.vision_agent import ActivityInferencer
from .agents.coordinator import CoordinatorAgent
from .context.context_engine import ContextEngine
from .execution.autonomous_agent import AutonomousAgent
from .vision.face.identity_manager import IdentityManager
from .core.event_bus import SystemEvent, EventPriority
from .execution.safety_layer import sandbox as safety_sandbox
from .perception.system_monitor import SystemMonitor
from .perception.screen_reader import ScreenReader
from .core.skill_synthesizer import SkillSynthesizer
import threading
import time
from hui import HUIDashboard
import sys
from PyQt5.QtWidgets import QApplication
from .voice.wake_word import contains_wake_word, strip_wake_word


def run() -> None:
    """
    Entry point for the JARVIS assistant.

    This wires together configuration, logging, voice I/O, memory,
    learning, reasoning, skills, and the execution layer into a
    continuous voice interaction loop.
    """
    project_root = Path(__file__).resolve().parents[1]
    config_path = project_root / "config" / "config.yaml"
    if not config_path.is_file():
        config_path = project_root / "config" / "config.example.yaml"
    config = load_config(config_path)

    log_file = project_root / "jarvis.log"
    configure_logging(log_file)
    logger = get_logger(__name__)

    logger.info("Starting C4 intelligence system")

    # ── User Model (persistent identity/preferences) ───────────────────────
    user_model_path = project_root / "data" / "user_model.json"
    user_model_mgr = init_user_model(user_model_path)
    logger.info(f"[UserModel] Loaded profile for '{user_model_mgr.get().name}'.")

    # Core components
    memory_root = project_root / "data" / "memory"
    memory_root.mkdir(parents=True, exist_ok=True)
    memory_manager = MemoryManager(memory_root)

    learning_engine = LearningEngine(memory_manager)
    llm_client = RoleBasedLLM.from_config(config)

    executor = Executor(
        dry_run=config.get("safety", {}).get("dry_run", False),
        confirm_deletes=True,
    )
    
    orchestrator = CognitiveOrchestrator(executor)

    # ── DAG Runner (graph-based task execution) ────────────────────────────
    dag_runner = DAGRunner(executor=executor, max_parallel=3)
    logger.info("[DAGRunner] Initialized.")

    # GUI Setup
    app = QApplication(sys.argv)
    hui_window = HUIDashboard()
    hui_window.show()

    from jarvis.hui.hands_overlay import HandsOverlay
    hands_overlay = HandsOverlay(signals=hui_window.signals)
    hands_overlay.show()

    # Skills and skill manager
    reminder_skill = ReminderSkill(voice_output=None, hui_window=hui_window)  # voice set after init
    skills = [
        SmallTalkSkill(config),
        SystemControlSkill(executor),
        FileOpsSkill(executor),
        WebSearchSkill(llm_client, config),
        LearnFactSkill(learning_engine),
        LearnCommandSkill(learning_engine, llm_client),
        MediaSkill(executor),
        KeyboardSkill(executor),
        FileSearchSkill(),
        WeatherSkill(),
        reminder_skill,
        VolumeControlSkill(),
        ScreenshotSkill(),
        NotesSkill(),
        ClipboardSkill(),
        OSGodModeSkill(),
        SmartHomeSkill(),
        ModelViewerSkill(),
    ]
    skill_manager = SkillManager(skills)

    skill_synthesizer = SkillSynthesizer(
        llm_client=llm_client,
        skill_manager=skill_manager,
        memory_manager=memory_manager
    )
    skill_synthesizer.load_persisted_skills()

    # NLP components
    intent_parser = IntentParser(memory_manager)
    planner = TaskPlanner(memory_manager)

    # Vision Setup
    vision_manager = VisionManager(hui_window=hui_window)
    vision_manager.start()
    
    # Face Recognition Identity DB (initialized early; FacePipeline started after bus)
    identity_manager = IdentityManager(
        db_path=project_root / "data" / "identities" / "identities.json"
    )
    logger.info(f"[FaceID] Identity DB loaded: {identity_manager.user_count} registered user(s).")
    
    governor = ResourceGovernor(vision_manager=vision_manager)

    # Voice I/O — auto-select best TTS (Edge-TTS → pyttsx3 fallback)
    voice_input = SpeechRecognitionInput(
        hui_window=hui_window,
        language=config.get("voice", {}).get("input", {}).get("language", "en-US")
    )
    voice_output = create_voice_output(config=config, hui_window=hui_window)

    # Reasoning engine
    reasoning_engine = ReasoningEngine(
        llm_client=llm_client,
        memory_manager=memory_manager,
        learning_engine=learning_engine,
        skill_manager=skill_manager,
        planner=planner,
        config=config,
        executor=executor,
        vision_manager=vision_manager,
        skill_synthesizer=skill_synthesizer,
    )
    reasoning_engine.dag_runner = dag_runner
    reasoning_engine._hui_window = hui_window    # NEW: for transcript + thinking signals
    # Voice output injected after creation so reminder skill also gets it
    reminder_skill.set_voice(voice_output)
    reminder_skill.set_hui(hui_window)

    context = ConversationContext()
    last_interaction_time = time.time()
    session_timeout = 300 # Seconds of activity gap before returning to wake-word mode

    def _compose_return_greeting(name: str = "") -> str:
        # Required catchphrase + a short recent work recap from episodic memory.
        who = "sir"
        if name and name.lower() not in {"there", "user", "unknown"}:
            who = name
        recent = memory_manager.get_recent_work_summary(limit=2)
        if recent:
            recap = " ".join(f"I have completed: {item}." for item in recent[-2:])
            return f"Welcome back {who}!! {recap}"
        return f"Welcome back {who}!!"

    def process_input(text_input: str, is_voice: bool = True):
        """Unified processing for voice and text commands."""
        nonlocal last_interaction_time
        last_interaction_time = time.time()
        
        # Phase 3: Fused Intent (only for voice/gesture fusion)
        if is_voice:
            text_input = intent_fusion.fuse_context(text_input)

        logger.info(f"Processing ({'Voice' if is_voice else 'UI'}): {text_input}")
        intent = intent_parser.parse(text_input, context)

        # Notify HUI of thinking start
        if hui_window:
            hui_window.signals.thinking_started.emit()

        response = reasoning_engine.handle_intent(intent, context)

        # Update context
        context.add_turn(
            user_text=text_input,
            assistant_text=response.spoken_response,
        )
        if response.last_action:
            context.last_action = response.last_action

        # UI Updates
        if hui_window:
            hui_window.signals.thinking_stopped.emit()
            if response.spoken_response:
                hui_window.signals.transcript_jarvis.emit(response.spoken_response)

        # Speak
        if response.spoken_response:
            voice_output.speak(response.spoken_response)

    # ── HUI Signal Connections ───────────────────────────────────────────
    if hui_window:
        hui_window.signals.command_submitted.connect(lambda cmd: process_input(cmd, is_voice=False))

    # ── EventBus Subscriptions for HUI ──────────────────────────────────
    def _on_thinking_update(event: SystemEvent):
        if hui_window:
            hui_window.thinking_panel.set_thinking(
                event.data.get("task_type", "general"),
                event.data.get("intent", "unknown"),
                event.data.get("steps", [])
            )
            
    def _on_thinking_step_status(event: SystemEvent):
        if hui_window:
            hui_window.thinking_panel.update_step_status(
                event.data.get("step_idx", 0),
                event.data.get("status", "")
            )

    def _on_hui_gesture(event: SystemEvent):
        if not hui_window: return
        action = event.data.get("action")
        # Logic to cycle focus based on swipe
        panels = ["vision", "globe", "metrics", "network", "system", "log"]
        current = hui_window.focused_panel or "vision"
        try:
            idx = panels.index(current)
            if action == "SWIPE_RIGHT":
                new_idx = (idx + 1) % len(panels)
                hui_window.set_focused_panel(panels[new_idx])
            elif action == "SWIPE_LEFT":
                new_idx = (idx - 1) % len(panels)
                hui_window.set_focused_panel(panels[new_idx])
            elif action == "PINCH":
                hui_window.trigger_glitch()
                hui_window._log_message(f"HUI: Selection confirmed on {current.upper()}")
        except ValueError:
            hui_window.focused_panel = "vision"

    bus.subscribe("c4.thinking.update", _on_thinking_update)
    bus.subscribe("c4.thinking.step_status", _on_thinking_step_status)
    bus.subscribe("hui.gesture_action", _on_hui_gesture)

    def main_loop():
        wake_word = (config.get("voice", {}) or {}).get("wake_word", "jarvis").lower()
        wake_enabled = bool((config.get("voice", {}) or {}).get("wake_word_enabled", True))
        armed = not wake_enabled
        armed = not wake_enabled
        if wake_enabled:
            try:
                hui_window.signals.update_status.emit(f"Say '{wake_word.upper()}' to activate")
            except Exception:
                pass

        # C4 boot greeting
        pm = getattr(reasoning_engine, "personality_manager", None)
        boot_msg = pm.get_greeting() if pm else "C4 intelligence online. Standing by, sir."
        voice_output.speak(boot_msg)
        if hui_window:
            try:
                hui_window.signals.transcript_jarvis.emit(boot_msg)
            except Exception:
                pass
        while True:
            try:
                # Check for session timeout if armed via wake word
                if armed and wake_enabled:
                    if time.time() - last_interaction_time > session_timeout:
                        armed = False
                        logger.info("[Main] Session timed out after period of inactivity. Disarming.")
                        try:
                            hui_window.signals.update_status.emit(f"Say '{wake_word.upper()}' to activate")
                        except Exception: pass

                transcript = voice_input.listen_once()
                if not transcript:
                    continue

                # Any heard speech resets the session timer
                last_interaction_time = time.time()

                # Wake word gate: only respond after hearing "C4".
                if wake_enabled and not armed:
                    if contains_wake_word(transcript, wake_word=wake_word):
                        armed = True
                        remainder = strip_wake_word(transcript, wake_word=wake_word)
                        # If user said only the wake word, greet and wait for the next command.
                        if not remainder:
                            voice_output.speak(_compose_return_greeting())
                            continue
                        transcript = remainder
                    else:
                        try:
                            hui_window.signals.update_status.emit("Waiting for wake word")
                        except Exception:
                            pass
                        continue
                    
                # Check for immediate interrupt
                if any(w in transcript.lower() for w in ["stop", "quiet", "be quiet", "shut up", "don't teach me"]):
                    voice_output.stop()
                    logger.info("Interrupt received. Stopping speech and disarming.")
                    armed = False # Return to waiting for wake word
                    if "don't teach me" not in transcript.lower():
                        try:
                            hui_window.signals.update_status.emit(f"Say '{wake_word.upper()}' to activate")
                        except Exception: pass
                        continue # Skip further processing for simple stop commands

                # Process the command via unified pipeline
                process_input(transcript, is_voice=True)

            except Exception as e:
                logger.error(f"[MainLoop] Error: {e}")
                time.sleep(0.5)

    # Phase 1: Event-Driven Autonomy Setup
    bus.start()

    # ── Face Recognition ──────────────────────────────────────────────
    face_pipeline = vision_manager.start_face_recognition(
        identity_manager=identity_manager,
        threshold=config.get("vision", {}).get("face_recognition_threshold", 0.45),
    )
    reasoning_engine.face_pipeline = face_pipeline

    def _on_user_detected(event: SystemEvent):
        """
        Memory hook: when a registered user is detected, JARVIS:
        1. Greets them once per session.
        2. Pre-loads their personalized facts into temporal context.
        3. Logs the session-start episode.
        """
        uid = event.data.get("user_id")
        name = event.data.get("display_name", "there")

        # Avoid re-greeting on every subsequent detection cycle
        from jarvis.core.world_state import world
        snap = world.get_snapshot()
        if snap["temporal_context"].get(f"greeted_{uid}"):
            return

        world.set_temporal_context(f"greeted_{uid}", True, ttl_seconds=3600)

        # Load user-tagged facts from memory for downstream LLM injection
        user_facts = memory_manager.find_facts(query=name, category="user")
        if user_facts:
            for fact in user_facts[:5]:
                world.set_temporal_context(
                    f"user_fact_{fact.key}", f"{fact.key}: {fact.value}", ttl_seconds=3600
                )

        memory_manager.add_episode(f"Session started. User identified: {name}.")
        greeting = _compose_return_greeting(name=name)
        logger.info(f"[FaceID] Greeting detected user: {name}")
        try:
            voice_output.speak(greeting)
        except Exception:
            pass

    def _on_user_left(event: SystemEvent):
        name = event.data.get("display_name", "User")
        memory_manager.add_episode(f"Session ended. {name} left the camera view.")
        logger.info(f"[FaceID] {name} left. Session logged.")

    def _on_unknown_user(event: SystemEvent):
        logger.info("[FaceID] Unknown person detected.")

    bus.subscribe("face.user_detected", _on_user_detected)
    bus.subscribe("face.user_left", _on_user_left)
    bus.subscribe("face.unknown_user", _on_unknown_user)
    # ──────────────────────────────────────────────────────────────
    
    watchdog = Watchdog([])
    watchdog.start()
    
    scheduler = ActionScheduler(executor)
    scheduler.start()
    
    personality_manager = PersonalityManager()
    reasoning_engine.personality_manager = personality_manager

    # ── Safety Sandbox (wire voice output for announcements) ──────────────
    safety_sandbox.voice_output = voice_output
    logger.info("[Safety] Sandbox ready with voice output.")

    # Phase 2: Deep Cognition Setup
    semantic_db = SemanticDB(Path("data/semantic_db.json"), embedder_func=llm_client.embed)
    memory_manager.set_semantic_db(semantic_db)
    reasoning_engine.semantic_db = semantic_db
    auto_corrector = ReinforcementFeedbackLoop(semantic_db)

    # ── Structured Memory System ──────────────────────────────────────────
    vector_store = VectorMemoryStore(
        storage_dir=memory_root,
        embed_fn=llm_client.embed,
    )
    vector_store.seed_system_memories()  # No-op if already seeded
    memory_retriever = MemoryRetriever(vector_store, max_results=5)
    memory_prompt_builder = MemoryPromptBuilder(max_entries=10, max_chars=2000)
    memory_writer = MemoryWriter(vector_store)

    reasoning_engine.memory_retriever = memory_retriever
    reasoning_engine.memory_writer = memory_writer
    reasoning_engine.memory_prompt_builder = memory_prompt_builder
    logger.info(
        f"[StructuredMemory] Initialized. Vector store: {vector_store.count()} entries. "
        f"FAISS index at: {vector_store._index_path}"
    )
    
    # Phase 3: Perception Fusion Setup
    semantic_vision = SemanticVisionEngine()
    semantic_vision.start()
    
    intent_fusion = UnifiedIntentEngine()

    # ── SYSTEM MONITOR (OS Perception) ────────────────────────────────────
    system_monitor = SystemMonitor(poll_interval=5.0)
    system_monitor.start()
    logger.info("[SystemMonitor] Started — tracking OS state.")

    screen_reader = ScreenReader(poll_interval=15.0)
    screen_reader.start()
    logger.info("[ScreenReader] Started — visual grounding via OCR/Windows.")

    # ── ATTENTION SYSTEM ──────────────────────────────────────────────────
    attention.start()
    logger.info("[Attention] System started.")

    # Wire voice activity to reset system monitor idle counter
    def _on_voice_activity(event: SystemEvent):
        system_monitor.reset_idle()
        attention.register_activity()
    bus.subscribe("voice.raw_transcript", _on_voice_activity)

    # Wire VAD interrupt to instantly silence TTS
    def _on_voice_interrupted(event: SystemEvent):
        # We only stop if JARVIS is actively speaking to prevent constant logs
        if getattr(voice_output, "_is_speaking", False):
            logger.info("[Main] VAD energy spike detected. Interrupting TTS instantly.")
            voice_output.stop()
            if hui_window:
                try:
                    hui_window.signals.log_message.emit("C4: [INTERRUPTED]")
                except Exception:
                    pass
    bus.subscribe("voice.interrupted", _on_voice_interrupted)

    # Wire emotion detection to WorldState user model
    def _on_emotion(event: SystemEvent):
        emotion = event.data.get("emotion", "neutral")
        conf = event.data.get("confidence", 0.5)
        world.set_user_emotion(emotion, conf)
        logger.debug(f"[Emotion] {emotion} ({conf:.0%})")
    bus.subscribe("voice.emotion_detected", _on_emotion)

    # Wire speaker identification to UserModel and WorldState
    def _on_speaker_id(event: SystemEvent):
        name = event.data.get("name", "")
        if name:
            user_model_mgr.set_identity(name)
            world.update_user_model(name=name.capitalize())
            logger.info(f"[SpeakerID] Identity set: {name}")
    bus.subscribe("voice.speaker_identified", _on_speaker_id)

    # Wire learning.action_unreliable to voice feedback
    def _on_unreliable(event: SystemEvent):
        action = event.data.get("action", "unknown")
        rate = event.data.get("fail_rate", 0)
        msg = f"Sir, I've been having trouble with {action}. Failure rate is {rate:.0%}. I may need an alternate method."
        try:
            voice_output.speak(msg)
        except Exception:
            pass
    bus.subscribe("learning.action_unreliable", _on_unreliable)

    # Phase 4/5/6: AGI Upgrades
    forecaster = TrendForecaster()
    forecaster.start()
    
    temporal_habits = TemporalHabitEngine(semantic_db)
    reasoning_engine.temporal_habits = temporal_habits
    
    activity_inferencer = ActivityInferencer(reasoning_engine.llm)
    activity_inferencer.start()
    
    coordinator = CoordinatorAgent(reasoning_engine.llm)

    # Start Context Engine
    context_engine = ContextEngine(vision_manager=vision_manager, memory_manager=memory_manager)
    context_engine.governor = governor
    context_engine.start()
    reasoning_engine.context_engine = context_engine

    # Inject voice output into reasoning engine for "thinking" acknowledgements
    reasoning_engine._voice_output = voice_output

    # Start Autonomous Agent
    autonomous_agent = AutonomousAgent(
        context_engine=context_engine,
        reasoning_engine=reasoning_engine,
        hui_window=hui_window,
        voice_output=voice_output
    )
    autonomous_agent.start()

    # Create and keep reference to the Gesture Action Executor
    gesture_executor = ActionExecutor()

    # Start AI loop in background
    ai_thread = threading.Thread(target=main_loop, daemon=True)
    ai_thread.start()

    # Run GUI loop
    sys.exit(app.exec_())


def run_safe() -> None:
    try:
        run()
    except SystemExit:
        # Normal exit (e.g., window closed)
        pass
    except BaseException:
        import traceback
        with open("crash.log", "w") as f:
            traceback.print_exc(file=f)
        print("\n--- CRITICAL STARTUP ERROR (See crash.log) ---")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    run_safe()

