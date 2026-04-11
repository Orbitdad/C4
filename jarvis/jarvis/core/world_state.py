import time
import threading
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class WorldState:
    """
    Persistent internal representation of reality.
    Holds the complete state of the user, environment, tasks, and predictions
    continuously — the cognitive core's memory of the present moment.
    """
    def __init__(self):
        self._lock = threading.RLock()

        # ── USER MODEL ────────────────────────────────────────────────────────
        # Live snapshot merged from UserModelManager every cycle
        self.user_model: Dict[str, Any] = {
            "name": "Sir",
            "identity": "unknown",
            "skill_level": 5,                 # 0 (novice) → 10 (expert)
            "emotion_state": "neutral",        # neutral|happy|frustrated|focused|tired
            "emotion_confidence": 0.0,
            "preferences": {},
            "recent_habits": [],
            "total_interactions": 0,
        }

        # ── ENVIRONMENT MODEL ─────────────────────────────────────────────────
        self.user_environment: Dict[str, Any] = {
            # Window / OS
            "active_window": None,
            "active_process": None,
            "visible_text": None,
            "running_processes": [],
            "network_connected": True,
            "network_type": "unknown",         # wifi|ethernet|offline

            # Inferred activity from window + vision
            "inferred_activity": None,         # coding|browsing|idle|gaming|writing
            "people_present": 0,
            "time_of_day": None,               # morning|afternoon|evening|night
            
            # Background Media Content
            "media_playing": "None",           # Currently playing song/media

            # Face Recognition
            "current_user": None,
            "current_user_id": None,
            "user_presence": False,
            "attention_state": "unknown",      # looking_at_screen|looking_away|unknown
            "face_count": 0,
            "face_recognition_active": False,
        }

        # ── SYSTEM HEALTH ─────────────────────────────────────────────────────
        self.system_health: Dict[str, Any] = {
            "cpu_percent": 0.0,
            "ram_percent": 0.0,
            "battery_percent": 100,
            "is_plugged_in": True,
            "watchdog_status": "OK",
            "disk_free_gb": 0.0,
        }

        # ── TASK MODEL ────────────────────────────────────────────────────────
        self.active_tasks: List[Dict[str, Any]] = []    # Currently executing DAG nodes
        self.pending_goals: List[Dict[str, Any]] = []   # Queued / deferred tasks
        self.recent_history: List[str] = []             # Last 20 actions taken

        # ── TEMPORAL CONTEXT (TTL key-value store) ────────────────────────────
        self.temporal_context: Dict[str, Dict[str, Any]] = {}

        # ── PREDICTIONS ───────────────────────────────────────────────────────
        self.predictions: Dict[str, Any] = {
            "next_likely_action": None,
            "predicted_at": 0.0,
            "confidence": 0.0,
        }

        # ── COGNITIVE META ────────────────────────────────────────────────────
        self.cognitive_meta: Dict[str, Any] = {
            "last_intent_confidence": 0.0,
            "current_attention_focus": "IDLE",   # USER|SYSTEM|BACKGROUND|IDLE
            "reasoning_active": False,
            "last_reasoning_step": "",
        }

        self.last_updated: float = time.time()

    # ── User Model ────────────────────────────────────────────────────────────

    def update_user_model(self, **kwargs):
        with self._lock:
            self.user_model.update(kwargs)
            self.last_updated = time.time()

    def set_user_emotion(self, emotion: str, confidence: float):
        with self._lock:
            self.user_model["emotion_state"] = emotion
            self.user_model["emotion_confidence"] = round(confidence, 2)
            self.last_updated = time.time()

    # ── Environment ──────────────────────────────────────────────────────────

    def update_environment(self, **kwargs):
        with self._lock:
            self.user_environment.update(kwargs)
            # Auto-derive time_of_day
            hour = time.localtime().tm_hour
            if 5 <= hour < 12:
                tod = "morning"
            elif 12 <= hour < 17:
                tod = "afternoon"
            elif 17 <= hour < 21:
                tod = "evening"
            else:
                tod = "night"
            self.user_environment["time_of_day"] = tod
            self.last_updated = time.time()

    # ── System Health ─────────────────────────────────────────────────────────

    def update_health(self, **kwargs):
        with self._lock:
            self.system_health.update(kwargs)
            self.last_updated = time.time()

    # ── Tasks ────────────────────────────────────────────────────────────────

    def add_active_task(self, task_id: str, description: str, priority: int = 2):
        with self._lock:
            self.active_tasks.append({
                "id": task_id,
                "desc": description,
                "priority": priority,
                "started": time.time()
            })
            self.last_updated = time.time()

    def complete_active_task(self, task_id: str):
        with self._lock:
            self.active_tasks = [t for t in self.active_tasks if t["id"] != task_id]
            self.last_updated = time.time()

    def add_pending_goal(self, goal: str, trigger: Optional[str] = None):
        """Queue a deferred goal (e.g. triggered at 9AM)."""
        with self._lock:
            self.pending_goals.append({
                "goal": goal,
                "trigger": trigger,
                "added_at": time.time()
            })
            self.last_updated = time.time()

    def remove_pending_goal(self, goal: str):
        with self._lock:
            self.pending_goals = [g for g in self.pending_goals if g["goal"] != goal]
            self.last_updated = time.time()

    def add_history(self, action: str):
        with self._lock:
            self.recent_history.append(action)
            if len(self.recent_history) > 20:
                self.recent_history.pop(0)
            self.last_updated = time.time()

    # ── Temporal Context ─────────────────────────────────────────────────────

    def set_temporal_context(self, key: str, value: Any, ttl_seconds: float = 60.0):
        with self._lock:
            self.temporal_context[key] = {
                "value": value,
                "expires_at": time.time() + ttl_seconds
            }
            self.last_updated = time.time()

    def _cleanup_expired_context(self):
        now = time.time()
        expired_keys = [k for k, v in self.temporal_context.items() if v["expires_at"] < now]
        for k in expired_keys:
            del self.temporal_context[k]

    # ── Predictions ──────────────────────────────────────────────────────────

    def add_prediction(self, next_action: str, confidence: float):
        with self._lock:
            self.predictions = {
                "next_likely_action": next_action,
                "predicted_at": time.time(),
                "confidence": round(confidence, 2),
            }
            self.last_updated = time.time()

    # ── Cognitive Meta ───────────────────────────────────────────────────────

    def update_cognitive_meta(self, **kwargs):
        with self._lock:
            self.cognitive_meta.update(kwargs)
            self.last_updated = time.time()

    # ── Master Snapshot ──────────────────────────────────────────────────────

    def get_snapshot(self) -> Dict[str, Any]:
        """Return a thread-safe deep copy of the full world state for agent consumption."""
        with self._lock:
            self._cleanup_expired_context()
            import copy
            return copy.deepcopy({
                "user_model": self.user_model,
                "user_environment": self.user_environment,
                "system_health": self.system_health,
                "active_tasks": self.active_tasks,
                "pending_goals": self.pending_goals,
                "recent_history": self.recent_history,
                "temporal_context": {k: v["value"] for k, v in self.temporal_context.items()},
                "predictions": self.predictions,
                "cognitive_meta": self.cognitive_meta,
                "last_updated": self.last_updated,
            })

    def get_context_summary(self) -> str:
        """
        Returns a compact natural-language summary of the world state
        for injection into LLM prompts.
        """
        snap = self.get_snapshot()
        env = snap["user_environment"]
        usr = snap["user_model"]
        health = snap["system_health"]
        tasks = snap["active_tasks"]
        pred = snap["predictions"]
        meta = snap["cognitive_meta"]

        lines = [
            f"User: {usr['name']} | Emotion: {usr['emotion_state']} | Skill: {usr['skill_level']}/10",
            f"Active Window: {env.get('active_window', 'unknown')} | Activity: {env.get('inferred_activity', 'unknown')}",
            f"Current Media / Music Playing: {env.get('media_playing', 'None')}",
            f"Time: {env.get('time_of_day', 'unknown')} | Network: {'online' if env.get('network_connected') else 'offline'}",
            f"CPU: {health['cpu_percent']:.0f}% | RAM: {health['ram_percent']:.0f}% | Battery: {health['battery_percent']}%",
            f"Active Tasks: {len(tasks)} | Attention: {meta.get('current_attention_focus', 'IDLE')}",
            f"Confidence: {meta.get('last_intent_confidence', 0):.0%}",
        ]
        if pred.get("next_likely_action"):
            lines.append(f"Predicted Next: {pred['next_likely_action']} (conf: {pred['confidence']:.0%})")
        return "\n".join(lines)


# Global singleton representing reality
world = WorldState()
