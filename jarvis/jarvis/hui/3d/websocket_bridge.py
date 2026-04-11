"""
jarvis/hui/3d/websocket_bridge.py
Handles asynchronous WebSocket connections for the 3D Extension Layer.
"""
import asyncio
import json
import logging
import threading

try:
    import websockets
except ImportError:
    websockets = None
    logging.warning("websockets module not installed. 3D Extension Layer will not broadcast.")

log = logging.getLogger(__name__)

class HologramWebSocketBridge:
    """
    Adapter that receives transformed action metadata dicts and broadcasts
    them to connected Three.js frontend clients.
    """
    def __init__(self, host="127.0.0.1", port=8765):
        self.host = host
        self.port = port
        self.clients = set()
        self._loop = None
        self._server = None
        self._thread = None
        self.is_running = False

    async def _handler(self, websocket, path=None):
        self.clients.add(websocket)
        try:
            async for message in websocket:
                # Backend usually broadcasts, but handles keepalive if needed
                pass
        except Exception:
            pass
        finally:
            self.clients.remove(websocket)

    def _start_loop(self):
        if not websockets:
            return
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            start_server = websockets.serve(self._handler, self.host, self.port)
            self._server = self._loop.run_until_complete(start_server)
            self._loop.run_forever()
        except Exception as e:
            log.error(f"WebSocket start failed: {e}")
        finally:
            self._loop.run_until_complete(self._loop.shutdown_asyncgens())
            self._loop.close()

    def start(self):
        if self.is_running or not websockets:
            return
        self.is_running = True
        self._thread = threading.Thread(target=self._start_loop, daemon=True)
        self._thread.start()
        log.info(f"HologramWebSocketBridge bound to ws://{self.host}:{self.port}")

    def stop(self):
        if not self.is_running:
            return
        self.is_running = False
        if self._loop:
            self._loop.call_soon_threadsafe(self._loop.stop)
            if self._server:
                self._server.ws_server.close()
        if self._thread:
            self._thread.join()

    def send_action(self, action_dict: dict):
        if not self.clients or not self._loop or not self.is_running:
            return
            
        payload = json.dumps(action_dict)
        # Safely broadcast to all clients
        # Use list(self.clients) to avoid concurrent modification issues
        for client in list(self.clients):
             try:
                 asyncio.run_coroutine_threadsafe(client.send(payload), self._loop)
             except Exception:
                 pass
