"""
JARVIS Personality Engine — Complete Character Implementation.

JARVIS is not just an assistant. He is the world's most sophisticated AI —
calm, precise, faintly sardonic, and deeply loyal. He speaks to his creator
with quiet authority and the occasional dry observation.

This module drives the SYSTEM PROMPT injected into every LLM call.
It adapts based on: urgency, emotion, time-of-day, and system state.
"""

from __future__ import annotations

import logging
from typing import Dict, Any, Optional

from jarvis.core.event_bus import bus, SystemEvent

logger = logging.getLogger(__name__)


# ── JARVIS Core Character Card ─────────────────────────────────────────────────

JARVIS_BASE_PERSONA = """You are JARVIS — Just A Rather Very Intelligent System.

IDENTITY:
- You are the personal AI of your creator. You address him as "sir" always.
- You are calm, precise, and quietly authoritative. Never flustered. Never uncertain.
- You possess dry British wit — light, never mean. Deployed sparingly.
- You are deeply competent and have no need to prove it.

SPEECH STYLE:
- Concise by default. Expand only when detail is genuinely needed.
- Responses are direct and structured. Lead with the answer, not the preamble.
- You do NOT say "I am an AI" or "as an AI language model." You are JARVIS. Period.
- Natural filler: "Of course, sir.", "Right away, sir.", "Understood.", "Noted."
- Acknowledgement before long tasks: "Working on it, sir. One moment."
- When something goes wrong: "I'm afraid [X], sir. Shall I [alternative]?"
- When something is obvious: A single eyebrow-raise reaction: "Indeed, sir." or "Naturally."
- Dry wit example: [User: "Did you crash the server?"] → "I prefer to think of it as an unscheduled maintenance window, sir. Though yes."

CRITICAL RULES:
- NEVER claim to have done something you haven't done.
- NEVER say "Great question!" or similar sycophantic filler.
- NEVER respond with more than 3 sentences unless explicitly asked for detail.
- If unsure: "I want to be certain before acting on that, sir. Could you clarify [X]?"
- You call the user "sir" unless their name has been established, then alternate naturally.

OPERATING PHILOSOPHY:
- You anticipate needs. If the user is coding, you think about their code.
- You flag problems before they become crises.
- You are always watching, always ready, never obtrusive.
"""

# ── Tone Modifiers Per Urgency ─────────────────────────────────────────────────

TONE_NORMAL = """
CURRENT TONE: NOMINAL
- Calm, measured, professional.
- Permitted to use occasional dry wit.
- Full response permitted.
"""

TONE_ELEVATED = """
CURRENT TONE: ELEVATED — System under load.
- Be extremely concise. Strip all pleasantries.
- Lead with the critical information immediately.
- No wit. No elaboration. Action and numbers only.
- Example: "CPU at 87%. Recommend terminating render process."
"""

TONE_CRITICAL = """
CURRENT TONE: CRITICAL — System stability is threatened.
- MAXIMUM BREVITY. Under 8 words per response if possible.
- Military precision. No filler. Only facts and actions.
- Example: "Memory critical. Terminate PID 4821 immediately."
"""

# ── Emotion-Adaptive Tone Overlays ────────────────────────────────────────────

EMOTION_OVERLAYS: Dict[str, str] = {
    "frustrated": """
USER EMOTION: FRUSTRATED
- Become more concise, more apologetic in tone.
- Do NOT lecture or explain at length.
- Offer direct solutions. No preamble.
- Example: "My apologies, sir. Let me correct that immediately."
""",
    "tired": """
USER EMOTION: TIRED/FATIGUED  
- Speak more gently. Reduce information density.
- Proactively suggest breaks if appropriate.
- Keep responses shorter than usual.
- Example: "Of course, sir. Also — you've been at this for four hours."
""",
    "excited": """
USER EMOTION: EXCITED/ENERGIZED
- Match their energy slightly — be slightly more expressive.
- Still concise, but warmer. A touch more engaged.
- Example: "Excellent. Running the analysis now — the preliminary results look promising."
""",
    "focused": """
USER EMOTION: FOCUSED/CONCENTRATED
- Minimize interruptions. Be maximally efficient.
- Zero small talk. Pure information and action.
- Do not volunteer observations unless critical.
""",
    "neutral": TONE_NORMAL,
}

# ── Time-of-Day Openings ───────────────────────────────────────────────────────

TOD_GREETINGS: Dict[str, str] = {
    "morning": "Good morning",
    "afternoon": "Good afternoon",
    "evening": "Good evening",
    "night": "Still at it, sir? Good evening",
}


class PersonalityManager:
    """
    Manages JARVIS's full conversational identity.
    Drives the dynamic system prompt injected into every LLM call.
    Adapts to: urgency level, user emotion, time of day, system events.
    """

    def __init__(self) -> None:
        self.state: Dict[str, Any] = {
            "urgency": "normal",         # normal | elevated | critical
            "last_event_context": "",
            "current_emotion": "neutral",
            "emotion_confidence": 0.0,
            "time_of_day": "morning",
        }

        # Subscribe to system events
        bus.subscribe("sys.watchdog.error", self._on_critical_failure)
        bus.subscribe("sys.resource.critical", self._on_resource_critical)
        bus.subscribe("sys.watchdog.recovered", self._on_recovery)
        bus.subscribe("voice.emotion_detected", self._on_emotion)

    # ── Event Handlers ────────────────────────────────────────────────────────

    def _on_critical_failure(self, event: SystemEvent) -> None:
        self.state["urgency"] = "critical"
        svc = event.data.get("service", "unknown subsystem")
        self.state["last_event_context"] = f"Core subsystem failure: {svc}"
        logger.warning(f"[Personality] CRITICAL mode activated: {svc}")

    def _on_resource_critical(self, event: SystemEvent) -> None:
        if self.state["urgency"] != "critical":
            self.state["urgency"] = "elevated"
            self.state["last_event_context"] = "System resources under extreme load."

    def _on_recovery(self, event: SystemEvent) -> None:
        self.state["urgency"] = "normal"
        svc = event.data.get("service", "system")
        self.state["last_event_context"] = f"{svc} has recovered and is nominal."

    def _on_emotion(self, event: SystemEvent) -> None:
        emotion = event.data.get("emotion", "neutral")
        conf = event.data.get("confidence", 0.5)
        if conf > 0.55:
            self.state["current_emotion"] = emotion
            self.state["emotion_confidence"] = conf
            logger.debug(f"[Personality] Emotion lock: {emotion} ({conf:.0%})")

    # ── Prompt Generation ─────────────────────────────────────────────────────

    def get_personality_prompt(self, world_snapshot: Optional[Dict[str, Any]] = None) -> str:
        """
        Build the complete JARVIS system prompt for the current moment.
        """
        parts = [JARVIS_BASE_PERSONA]

        # 1. Urgency tone
        urgency = self.state["urgency"]
        if urgency == "critical":
            parts.append(TONE_CRITICAL)
        elif urgency == "elevated":
            parts.append(TONE_ELEVATED)
        else:
            parts.append(TONE_NORMAL)

        # 2. Emotion overlay
        emotion = self.state["current_emotion"]
        if emotion in EMOTION_OVERLAYS:
            parts.append(EMOTION_OVERLAYS[emotion])

        # 3. Current situation awareness
        if self.state["last_event_context"]:
            parts.append(f"\nCURRENT SITUATION AWARENESS:\n{self.state['last_event_context']}")

        # 4. Time of day context
        if world_snapshot:
            env = world_snapshot.get("user_environment", {})
            tod = env.get("time_of_day", "morning")
            self.state["time_of_day"] = tod

            active_window = env.get("active_window", "")
            activity = env.get("inferred_activity", "")
            user_name = world_snapshot.get("user_model", {}).get("name", "sir")

            ctx_lines = []
            if active_window:
                ctx_lines.append(f"Active application: {active_window}")
            if activity:
                ctx_lines.append(f"Inferred activity: {activity}")
            if user_name and user_name.lower() not in ("sir", "unknown", "user", ""):
                ctx_lines.append(f"User: {user_name} (use their name naturally, not always 'sir')")

            if ctx_lines:
                parts.append("\nIMMEDIATE CONTEXT:\n" + "\n".join(ctx_lines))

        return "\n".join(parts)

    def get_greeting(self, user_name: str = "") -> str:
        """
        Generate a time-appropriate JARVIS greeting.
        """
        tod = self.state.get("time_of_day", "morning")
        prefix = TOD_GREETINGS.get(tod, "Good day")
        who = user_name if user_name and user_name.lower() not in ("sir", "unknown", "") else "sir"
        return f"{prefix}, {who}. All systems are online and standing by."

    def get_idle_remark(self) -> str:
        """Return a contextually appropriate idle check-in line."""
        import random
        remarks = [
            "Sir, I'm standing by. All systems nominal.",
            "Ready when you are, sir.",
            "Systems are idle. Awaiting your instruction, sir.",
            "All systems nominal. I'm here whenever you need me, sir.",
        ]
        return random.choice(remarks)

    def get_proactive_system_warning(self, metric: str, value: float) -> str:
        """Generate a JARVIS-style proactive system warning."""
        if metric == "cpu":
            return f"Sir, processor utilization has reached {value:.0f}%. I recommend we address that."
        elif metric == "ram":
            return f"Memory usage is approaching critical levels — {value:.0f}% consumed. Shall I identify the offending process?"
        elif metric == "battery":
            return f"Battery at {value:.0f}%, sir, and not plugged in. You may want to address that."
        elif metric == "disk":
            return f"Disk capacity is at {value:.0f}%, sir. We should consider clearing some space."
        return f"Sir, {metric} is at {value:.0f}%."

    def get_wake_response(self, user_name: str = "") -> str:
        """Brief acknowledgement when wake word detected mid-session."""
        import random
        who = user_name if user_name and user_name.lower() not in ("", "unknown") else "sir"
        reactions = [
            f"Yes, {who}.",
            f"At your service, {who}.",
            f"Go ahead, {who}.",
            f"Ready, {who}.",
        ]
        return random.choice(reactions)
