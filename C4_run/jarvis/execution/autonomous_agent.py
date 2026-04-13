"""
JARVIS Proactive Narrator — Zero-latency event-driven situational awareness.

This replaces the old 60-second polling AutonomousAgent with an
event-driven system that reacts instantly to system events,
combined with a 10-second deep-check LLM loop for contextual proactivity.

JARVIS doesn't wait. JARVIS watches.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Optional

from jarvis.core.event_bus import bus, SystemEvent, EventPriority

logger = logging.getLogger(__name__)


class AutonomousAgent:
    """
    Zero-latency proactive JARVIS agent.

    Architecture:
    - Event-driven reactions: Instant response to system events (CPU, battery, face, etc.)
    - Deep-check loop: Every 10s, LLM evaluates full context for unprompted insights
    - Personality-driven narration: All outputs routed through PersonalityManager
    """

    # Thresholds for immediate proactive alerts
    CPU_WARN_THRESHOLD = 85.0
    RAM_WARN_THRESHOLD = 88.0
    BATTERY_WARN_THRESHOLD = 15.0
    DISK_WARN_THRESHOLD = 90.0

    # Cooldown between same-type alerts (seconds)
    ALERT_COOLDOWN = 60.0

    DEEP_CHECK_INTERVAL = 10  # seconds between LLM deep checks

    def __init__(
        self,
        context_engine: Any,
        reasoning_engine: Any,
        hui_window: Optional[Any] = None,
        voice_output: Optional[Any] = None,
    ) -> None:
        self.context_engine = context_engine
        self.reasoning_engine = reasoning_engine
        self.hui_window = hui_window
        self.voice_output = voice_output

        self.is_running = False
        self._thread: Optional[threading.Thread] = None
        self._alert_timestamps: dict = {}  # metric -> last_alert_time

        # Wire event-driven instant reactions
        bus.subscribe("context.user_idle", self._on_user_idle)
        bus.subscribe("context.window_changed", self._on_window_changed)
        bus.subscribe("system.health.critical", self._on_health_critical)
        bus.subscribe("system.battery.low", self._on_battery_low)
        bus.subscribe("face.user_detected", self._on_face_detected)
        bus.subscribe("sys.resource.critical", self._on_resource_critical)

        # System monitor events
        bus.subscribe("monitor.cpu_critical", self._on_cpu_critical)
        bus.subscribe("monitor.ram_critical", self._on_ram_critical)
        bus.subscribe("monitor.battery_critical", self._on_battery_critical)

        logger.info("[ProactiveAgent] Event-driven JARVIS agent initialized.")

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        if self.is_running:
            return
        self.is_running = True
        self._thread = threading.Thread(
            target=self._deep_check_loop,
            daemon=True,
            name="JARVISDeepCheck",
        )
        self._thread.start()
        logger.info("[ProactiveAgent] Deep-check loop started.")

    def stop(self) -> None:
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=3.0)

    # ── Utility ───────────────────────────────────────────────────────────────

    def _speak(self, message: str) -> None:
        """Emit voice log."""
        if message:
            if self.voice_output:
                try:
                    self.voice_output.speak(message)
                except Exception as e:
                    logger.error(f"[ProactiveAgent] Voice output error: {e}")

    def _can_alert(self, key: str, cooldown: float = None) -> bool:
        """Rate-limit alerts to avoid harassing the user."""
        cd = cooldown or self.ALERT_COOLDOWN
        last = self._alert_timestamps.get(key, 0)
        if time.time() - last > cd:
            self._alert_timestamps[key] = time.time()
            return True
        return False

    def _get_personality(self) -> Any:
        return getattr(self.reasoning_engine, "personality_manager", None)

    # ── Instant Event Reactions ───────────────────────────────────────────────

    def _on_user_idle(self, event: SystemEvent) -> None:
        # User requested no proactive idle check-ins unless explicitly called.
        pass
        # seconds = event.data.get("seconds", 0)
        # if seconds == 180 and self._can_alert("idle_check", 300):
        #     self._speak("Sir, you've been inactive for three minutes. Systems are standing by.")
        # elif seconds == 600 and self._can_alert("idle_long", 600):
        #     self._speak("Just checking in, sir. I'm here whenever you're ready.")

    def _on_window_changed(self, event: SystemEvent) -> None:
        win = event.data.get("window", "")
        if not win or not self._can_alert("window_change", 30):
            return
        win_lower = win.lower()
        if "visual studio code" in win_lower or "code" in win_lower:
            if self._can_alert("dev_env", 120):
                self._speak("Development environment active, sir. I'm monitoring your workspace.")
        elif "chrome" in win_lower or "firefox" in win_lower or "edge" in win_lower:
            pass   # Silent — browsing is normal
        elif "task manager" in win_lower:
            self._speak("Task Manager opened, sir. The top resource consumers are in the system panel.")

    def _on_health_critical(self, event: SystemEvent) -> None:
        resource = event.data.get("resource", "system")
        value = event.data.get("value", 0)
        p = self._get_personality()
        msg = p.get_proactive_system_warning(resource, value) if p else f"Warning: {resource} critical at {value:.0f}%."
        if self._can_alert(f"health_{resource}"):
            self._speak(msg)

    def _on_battery_low(self, event: SystemEvent) -> None:
        pct = event.data.get("percent", 0)
        if self._can_alert("battery", 120):
            self._speak(f"Battery at {pct:.0f}%, sir, and running on battery power. You may want to plug in.")

    def _on_face_detected(self, event: SystemEvent) -> None:
        # Face greeting is handled in main.py — skip here to avoid double-greeting
        pass

    def _on_resource_critical(self, event: SystemEvent) -> None:
        resource = event.data.get("resource", "system")
        value = event.data.get("value", 0)
        if self._can_alert(f"resource_{resource}", 90):
            p = self._get_personality()
            msg = p.get_proactive_system_warning(resource, value) if p else f"{resource} critical: {value:.0f}%"
            self._speak(msg)

    def _on_cpu_critical(self, event: SystemEvent) -> None:
        value = event.data.get("value", 0)
        if value >= self.CPU_WARN_THRESHOLD and self._can_alert("cpu", 90):
            p = self._get_personality()
            msg = p.get_proactive_system_warning("cpu", value) if p else f"Sir, CPU at {value:.0f}%."
            self._speak(msg)

    def _on_ram_critical(self, event: SystemEvent) -> None:
        value = event.data.get("value", 0)
        if value >= self.RAM_WARN_THRESHOLD and self._can_alert("ram", 90):
            p = self._get_personality()
            msg = p.get_proactive_system_warning("ram", value) if p else f"Sir, memory at {value:.0f}%."
            self._speak(msg)

    def _on_battery_critical(self, event: SystemEvent) -> None:
        value = event.data.get("value", 0)
        if value <= self.BATTERY_WARN_THRESHOLD and self._can_alert("battery_critical", 120):
            p = self._get_personality()
            msg = p.get_proactive_system_warning("battery", value) if p else f"Sir, battery critically low at {value:.0f}%."
            self._speak(msg)

    # ── Deep-Check LLM Loop ───────────────────────────────────────────────────

    def _deep_check_loop(self) -> None:
        """
        Every 10 seconds: feed full context to LLM and check if any
        proactive action or spoken comment is warranted.
        """
        while self.is_running:
            time.sleep(self.DEEP_CHECK_INTERVAL)
            try:
                if not self.context_engine.is_running:
                    continue

                context_snapshot = self.context_engine.get_context_snapshot()

                # Quick heuristic guard: don't bother LLM if nothing significant
                if not self._is_context_interesting(context_snapshot):
                    continue

                # Check if voice is idle before inserting autonomous speech
                if getattr(self.voice_output, "_is_speaking", False):
                    continue

                prompt = f"""You are C4, proactively monitoring the user's system.
Current system context:
{context_snapshot}

Is there anything important you should proactively SPEAK to the user right now?
Examples of valid proactive actions:
- A resource is critically high
- The user appears to be stuck (idle too long in an error state)
- Context switched to a new task type  

If YES: Return ONLY a JSON object: {{"action": "speak", "message": "Your message here"}}
If NO: Return exactly: NONE

Rules: Be brief. Use sir. Max 20 words in message. No markdown."""

                try:
                    response = self.reasoning_engine.llm.generate(prompt)
                    resp_text = (response or "").strip()

                    if resp_text and "NONE" not in resp_text.upper() and len(resp_text) > 5:
                        import json, re
                        m = re.search(r'\{.*\}', resp_text, re.DOTALL)
                        if m:
                            data = json.loads(m.group())
                            if data.get("action") == "speak":
                                msg = data.get("message", "")
                                # Rate-limit LLM-driven speech
                                if msg and self._can_alert("llm_proactive", 45):
                                    self._speak(msg)
                except Exception as e:
                    logger.debug(f"[ProactiveAgent] Deep-check LLM error: {e}")

            except Exception as e:
                logger.error(f"[ProactiveAgent] Deep-check loop error: {e}")

    def _is_context_interesting(self, snapshot: str) -> bool:
        """Quick pre-filter: only invoke LLM if something notable is in the snapshot."""
        keywords = [
            "critical", "high", "error", "low battery", "full",
            "warning", "failed", "unstable", "offline"
        ]
        snap_lower = snapshot.lower()
        return any(kw in snap_lower for kw in keywords)
