"""
jarvis/hui/3d/scene_manager.py
Orchestrates the 3D Extension Layer by running the lightweight HTTP server
for the frontend and bootstrapping the WebSocket bridge.
"""
import http.server
import socketserver
import threading
import logging
import traceback
from .object_controller import generate_frontend
from .websocket_bridge import HologramWebSocketBridge

log = logging.getLogger(__name__)

class ContentHandler(http.server.SimpleHTTPRequestHandler):
    def log_message(self, format, *args):
        # Suppress standard HTTP logs to keep console clean
        pass

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            self.send_response(200)
            self.send_header("Content-type", "text/html")
            self.end_headers()
            # Dynamically insert our HTML/JS payload
            payload = generate_frontend()
            self.wfile.write(payload.encode("utf-8"))
        else:
            self.send_response(404)
            self.end_headers()

class HolographicSceneManager:
    def __init__(self, http_port=8000, ws_port=8765):
        self.http_port = http_port
        self.ws_bridge = HologramWebSocketBridge(port=ws_port)
        self.is_running = False
        self._httpd = None
        self._http_thread = None

    def _run_http(self):
        socketserver.TCPServer.allow_reuse_address = True
        try:
            # We initialize locally here so it can be repeatedly started/stopped if required
            with socketserver.TCPServer(("", self.http_port), ContentHandler) as httpd:
                self._httpd = httpd
                httpd.serve_forever()
        except Exception as e:
            log.error(f"HTTP Server start failed: {e}")
            traceback.print_exc()

    def start(self):
        """Invoke exclusively when switching into HOLOGRAM_3D_MODE."""
        if self.is_running:
            return
        self.is_running = True
        
        log.info("Starting up 3D Holographic Scene Manager...")
        
        # Fire up websocket backend bridge
        self.ws_bridge.start()
        
        # Serve frontend bundle payload
        self._http_thread = threading.Thread(target=self._run_http, daemon=True)
        self._http_thread.start()
        log.info(f"Holographic HUI Layer Active. Access scene at http://127.0.0.1:{self.http_port}")

    def stop(self):
        """Gracefully release HTTP/WS resources when out of mode."""
        if not self.is_running:
            return
        self.is_running = False
        
        log.info("Shutting down 3D Holographic Scene Manager...")
        self.ws_bridge.stop()
        
        if self._httpd:
            self._httpd.shutdown()
            self._httpd.server_close()
        
        if self._http_thread:
            self._http_thread.join()
