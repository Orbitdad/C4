"""
websocket_manager.py — C4 WebSocket Server Subsystem
=====================================================
Singleton async WebSocket server that:
  - Runs in a background daemon thread (asyncio event loop)
  - Accepts connections from browser clients (e.g. the Three.js model viewer)
  - Broadcasts JSON command payloads to every connected client safely
  - Exposes a dead-simple thread-safe API: WebSocketManager.instance().broadcast(payload)

Usage (from any C4 skill or module):
    from jarvis.skills.websocket_manager import WebSocketManager
    ws = WebSocketManager.instance()
    ws.start()                            # idempotent — call once
    ws.broadcast({"command": "explode"})  # thread-safe
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Any, Dict, Optional, Set

logger = logging.getLogger(__name__)

# ── Constants ──────────────────────────────────────────────────────────────────
DEFAULT_HOST = "localhost"
DEFAULT_PORT = 8765


class WebSocketManager:
    """
    Thread-safe singleton WebSocket server manager.

    Design:
      - One asyncio event loop runs in a dedicated background daemon thread.
      - All async operations are scheduled onto that loop via
        asyncio.run_coroutine_threadsafe, making the public API fully
        thread-safe for callers on any thread (e.g. C4's voice loop).
      - Connected clients are stored in a set; the server adds/removes them
        automatically on connect/disconnect.
    """

    _instance: Optional["WebSocketManager"] = None
    _lock: threading.Lock = threading.Lock()

    # ── Singleton access ───────────────────────────────────────────────────────

    @classmethod
    def instance(cls) -> "WebSocketManager":
        """Return (and lazily create) the global WebSocketManager singleton."""
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    # ── Lifecycle ──────────────────────────────────────────────────────────────

    def __init__(self, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
        self.host = host
        self.port = port

        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

        # All websocket connection objects currently open
        self._clients: Set[Any] = set()
        self._clients_lock = asyncio.Lock()  # async lock, only used inside the event loop

    def start(self) -> None:
        """
        Start the WebSocket server in a background daemon thread.
        Idempotent — safe to call multiple times.
        """
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(
            target=self._run_event_loop,
            name="C4-WebSocket-Server",
            daemon=True,
        )
        self._thread.start()
        logger.info(f"[WebSocketManager] Server starting on ws://{self.host}:{self.port}")

    def stop(self) -> None:
        """
        Gracefully stop the WebSocket server.
        Broadcasts a shutdown message to all clients first.
        """
        if not self._running or self._loop is None:
            return
        future = asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop)
        try:
            future.result(timeout=5)
        except Exception as exc:
            logger.warning(f"[WebSocketManager] Shutdown warning: {exc}")
        self._running = False
        logger.info("[WebSocketManager] Server stopped.")

    # ── Public API (thread-safe) ───────────────────────────────────────────────

    def broadcast(self, payload: Dict[str, Any]) -> None:
        """
        Send a JSON-encoded payload to all connected browser clients.

        Thread-safe: can be called from any thread.

        Args:
            payload: dict that will be JSON-encoded before sending.
                     e.g. {"command": "explode"} or {"command": "load_model", "model": "engine.glb"}
        """
        if not self._running or self._loop is None:
            logger.warning("[WebSocketManager] broadcast() called before server started.")
            return
        asyncio.run_coroutine_threadsafe(self._broadcast_async(payload), self._loop)

    def get_connected_count(self) -> int:
        """Return the number of currently connected clients (approximate)."""
        return len(self._clients)

    # ── Internal async implementation ──────────────────────────────────────────

    def _run_event_loop(self) -> None:
        """Thread entry point: create a new event loop and run it forever."""
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        except Exception as exc:
            logger.error(f"[WebSocketManager] Event loop crashed: {exc}")
        finally:
            self._loop.close()

    async def _serve(self) -> None:
        """Start the websockets server and run until cancelled."""
        try:
            import websockets  # imported lazily to avoid hard dep at module load
        except ImportError:
            logger.error(
                "[WebSocketManager] 'websockets' package not installed. "
                "Run: pip install websockets"
            )
            return

        async with websockets.serve(
            self._handle_client,
            self.host,
            self.port,
            ping_interval=20,       # keep connections alive
            ping_timeout=10,
        ):
            logger.info(f"[WebSocketManager] Listening on ws://{self.host}:{self.port}")
            # Run forever until the loop is stopped externally
            await asyncio.Future()  # block indefinitely

    async def _handle_client(self, websocket: Any) -> None:
        """
        Register a new client connection and listen until it disconnects.
        Sends an initial handshake so the frontend knows the server is alive.
        """
        async with self._clients_lock:
            self._clients.add(websocket)

        client_addr = getattr(websocket, "remote_address", "unknown")
        logger.info(f"[WebSocketManager] Client connected: {client_addr}  "
                    f"(total: {len(self._clients)})")

        # Greet the newly connected client
        try:
            await websocket.send(json.dumps({"command": "connected", "server": "C4"}))
        except Exception:
            pass

        try:
            # Keep connection alive; handle incoming messages (future extensibility)
            async for message in websocket:
                await self._handle_incoming(websocket, message)
        except Exception:
            pass  # Connection closed or error — just remove client
        finally:
            async with self._clients_lock:
                self._clients.discard(websocket)
            logger.info(f"[WebSocketManager] Client disconnected: {client_addr}  "
                        f"(total: {len(self._clients)})")

    async def _handle_incoming(self, websocket: Any, message: str) -> None:
        """
        Handle messages arriving FROM the browser (future: telemetry, selections).
        Currently logs and ignores unknown messages.
        """
        try:
            data = json.loads(message)
            event = data.get("event", "")
            logger.debug(f"[WebSocketManager] Received from browser: {event} — {data}")
            # Future: route browser events back to the C4 event bus here
        except json.JSONDecodeError:
            logger.debug(f"[WebSocketManager] Non-JSON message from browser: {message[:100]}")

    async def _broadcast_async(self, payload: Dict[str, Any]) -> None:
        """Async implementation — sends payload JSON to every connected client."""
        if not self._clients:
            return
        message = json.dumps(payload)
        # Snapshot to avoid mutation during iteration
        async with self._clients_lock:
            clients = set(self._clients)

        dead: Set[Any] = set()
        for ws in clients:
            try:
                await ws.send(message)
            except Exception:
                dead.add(ws)  # connection died — remove after iteration

        if dead:
            async with self._clients_lock:
                self._clients -= dead

        logger.debug(f"[WebSocketManager] Broadcast → {payload.get('command', '?')}  "
                     f"recipients: {len(clients) - len(dead)}")

    async def _shutdown(self) -> None:
        """Send a shutdown notice then close all client connections."""
        await self._broadcast_async({"command": "server_shutdown"})
        async with self._clients_lock:
            clients = set(self._clients)
        for ws in clients:
            try:
                await ws.close()
            except Exception:
                pass
