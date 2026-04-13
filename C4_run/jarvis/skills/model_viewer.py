"""
model_viewer.py — C4 3D Model Viewer Skill
==========================================
Plugs into C4's SkillManager to handle MODEL_VIEW intents.

Responsibilities:
  1. Start the Three.js viewer HTTP server (viewer/serve.py) on first use
  2. Start the WebSocket broadcast server (websocket_manager.py) on first use
  3. Open the browser to http://localhost:8080 when needed
  4. Translate C4 intent actions into WebSocket commands sent to the frontend

Supported parsed_actions (set by intent_parser.py):
  - open_viewer      → launch browser to the viewer page
  - load_model       → tell frontend to load a specific model
  - explode_model    → trigger exploded-view animation
  - reset_model      → return model to assembled position
  - rotate_model     → begin auto-rotate animation
  - zoom_model       → zoom in or out
"""

from __future__ import annotations

import logging
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Optional

from jarvis.context import ConversationContext
from jarvis.nlp.schemas import Intent, IntentType
from jarvis.skills.base import BaseSkill, SkillResult
from jarvis.skills.websocket_manager import WebSocketManager

logger = logging.getLogger(__name__)

# ── Configuration ────────────────────────────────────────────────────────────

VIEWER_PORT = 8080
VIEWER_URL = f"http://localhost:{VIEWER_PORT}"
WS_PORT = 8765

# Path: <project_root>/viewer/serve.py  (sibling of C4_Run/)
_SERVE_PY = Path(__file__).resolve().parents[3] / "viewer" / "serve.py"
# Alternatively, if viewer/ sits inside C4_Run/:
_SERVE_PY_ALT = Path(__file__).resolve().parents[2] / "viewer" / "serve.py"


def _resolve_serve_path() -> Optional[Path]:
    """Find the serve.py script regardless of repo layout."""
    for candidate in [_SERVE_PY, _SERVE_PY_ALT]:
        if candidate.is_file():
            return candidate
    return None


# ── Skill ────────────────────────────────────────────────────────────────────

class ModelViewerSkill(BaseSkill):
    """
    C4 skill that bridges voice commands to the Three.js 3D model viewer
    via an internal WebSocket server.

    Architecture:
        Voice → IntentParser → ReasoningEngine → ModelViewerSkill
                                                ↓
                                         WebSocketManager.broadcast()
                                                ↓ ws://localhost:8765
                                         Browser (Three.js viewer)
                                                ↓
                                         ExplodedView / SceneManager
    """

    name: str = "model_viewer"

    # Trigger keywords for BaseSkill.can_handle() fallback
    triggers = [
        "explode", "3d model", "model viewer", "engine model",
        "reset model", "rotate model", "zoom model",
    ]

    # Map intent parsed_action → WebSocket command string
    _ACTION_COMMAND_MAP: dict[str, str] = {
        "explode_model": "explode",
        "reset_model":   "reset",
        "rotate_model":  "rotate",
        "zoom_model":    "zoom",
        "load_model":    "load_model",
        "open_viewer":   "open_viewer",
    }

    def __init__(self) -> None:
        self._http_proc: Optional[subprocess.Popen] = None  # viewer serve.py process
        self._ws_manager: Optional[WebSocketManager] = None
        self._viewer_open = False  # track if browser has been launched this session

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def setup(self) -> None:
        """Boot WebSocket server eagerly so it is ready when the viewer opens."""
        self._ws_manager = WebSocketManager.instance()
        self._ws_manager.start()
        logger.info("[ModelViewerSkill] WebSocket server started.")

    def teardown(self) -> None:
        """Gracefully stop background servers."""
        if self._ws_manager:
            self._ws_manager.stop()
        if self._http_proc and self._http_proc.poll() is None:
            self._http_proc.terminate()
            logger.info("[ModelViewerSkill] HTTP server stopped.")

    # ── Skill execution ──────────────────────────────────────────────────────

    def can_handle(self, intent: Intent) -> bool:
        return intent.type == IntentType.MODEL_VIEW

    def execute(
        self,
        intent: Intent,
        context: Optional[ConversationContext] = None,
        **kwargs: Any,
    ) -> SkillResult:
        """
        Dispatch the intent to the appropriate WebSocket command.
        Returns a SkillResult with a human-readable spoken response.
        """
        action = intent.parsed_action or ""

        # Ensure backend services are running
        self._ensure_http_server()
        self._ensure_ws_server()

        # ── Open viewer ──────────────────────────────────────────────────────
        if action == "open_viewer":
            self._open_browser()
            return SkillResult(text="Opening the 3D model viewer, sir.")

        # ── Load specific model ──────────────────────────────────────────────
        if action == "load_model":
            model_name = intent.params.get("model", "arc-reactor.glb")
            if not model_name.endswith(".glb"):
                model_name += ".glb"
            # Ensure browser is open first
            self._open_browser()
            # Small delay so the page can connect to the WS server
            time.sleep(1.5)
            self._broadcast({"type": "MODEL_ACTION", "action": "load_model", "params": {"model": model_name}})
            return SkillResult(text=f"Loading model {model_name} in the viewer, sir.")

        # ── Explode / Reset / Rotate / Zoom ─────────────────────────────────
        ws_cmd = self._ACTION_COMMAND_MAP.get(action)
        if ws_cmd:
            self._open_browser()                      # open viewer if not already open
            self._broadcast({"type": "MODEL_ACTION", "action": ws_cmd, "params": intent.params})
            return SkillResult(text=self._spoken_response(action))

        # ── Unknown action fallback ──────────────────────────────────────────
        logger.warning(f"[ModelViewerSkill] No handler for action: '{action}'")
        return SkillResult(
            text="I'm not sure what you'd like me to do with the model, sir.",
            success=False,
        )

    # ── Helpers ──────────────────────────────────────────────────────────────

    def _broadcast(self, payload: dict) -> None:
        """Thread-safe broadcast via WebSocketManager."""
        if self._ws_manager:
            self._ws_manager.broadcast(payload)
            logger.debug(f"[ModelViewerSkill] Broadcast: {payload}")

    def _open_browser(self) -> None:
        """Open the viewer URL in the default browser (once per session)."""
        if self._viewer_open:
            return
        self._viewer_open = True
        url = VIEWER_URL
        try:
            if platform.system() == "Windows":
                os.startfile(url)
            else:
                subprocess.Popen(["xdg-open", url], start_new_session=True)
            logger.info(f"[ModelViewerSkill] Opened browser: {url}")
        except Exception as exc:
            logger.error(f"[ModelViewerSkill] Could not open browser: {exc}")

    def _ensure_http_server(self) -> None:
        """Start the static HTTP server for the viewer if not already running."""
        if self._http_proc and self._http_proc.poll() is None:
            return  # Already running

        serve_path = _resolve_serve_path()
        if not serve_path:
            logger.warning(
                "[ModelViewerSkill] viewer/serve.py not found. "
                "Run it manually: python viewer/serve.py"
            )
            return

        try:
            self._http_proc = subprocess.Popen(
                ["python", str(serve_path), "--port", str(VIEWER_PORT)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            # Give the server a moment to bind the port
            time.sleep(0.5)
            logger.info(f"[ModelViewerSkill] HTTP server started on port {VIEWER_PORT}")
        except Exception as exc:
            logger.error(f"[ModelViewerSkill] Failed to start HTTP server: {exc}")

    def _ensure_ws_server(self) -> None:
        """Ensure the WebSocket manager is started (idempotent)."""
        if self._ws_manager is None:
            self._ws_manager = WebSocketManager.instance()
        self._ws_manager.start()

    @staticmethod
    def _spoken_response(action: str) -> str:
        """Generate a brief spoken response for each action."""
        responses = {
            "explode_model": "Exploding components for visual inspection, sir.",
            "reset_model":   "Reassembling schematic. Model reset.",
            "rotate_model":  "Initiating model rotation.",
            "zoom_model":    "Adjusting focal depth.",
            "open_viewer":   "Opening visualization interface. Please check your browser window, sir.",
            "load_model":    "Loading 3D asset into projection matrix.",
        }
        return responses.get(action, "Command sequenced. Executing on visualization interface.")
