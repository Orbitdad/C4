"""
serve.py — Static HTTP server for the C4 3D Model Viewer
=========================================================
Serves the viewer/ directory over HTTP on port 8080 so the browser
can load index.html, main.js, and models/*.glb via local URLs.

Usage:
    python serve.py            # start server (blocking)
    python serve.py --port 9090  # custom port

The ModelViewerSkill starts this automatically in a subprocess; you can
also run it manually for development.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Constants ───────────────────────────────────────────────────────────────

DEFAULT_PORT = 8080

# Serve files from the directory containing this script (viewer/)
SERVE_ROOT = Path(__file__).resolve().parent


# ── Custom Request Handler ──────────────────────────────────────────────────

class SilentHandler(SimpleHTTPRequestHandler):
    """
    SimpleHTTPRequestHandler with:
      - CORS headers so fetch()/WebSocket work from any origin
      - Silenced access logs (errors still surface)
      - Correct MIME types for .glb / .gltf
    """

    # GLB/GLTF MIME types not in Python's default mimetypes table
    _EXTRA_MIME: dict[str, str] = {
        ".glb":  "model/gltf-binary",
        ".gltf": "model/gltf+json",
        ".wasm": "application/wasm",
        ".mjs":  "text/javascript",
    }

    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(SERVE_ROOT), **kwargs)

    def end_headers(self) -> None:
        # CORS — allow the WebSocket + fetch from localhost
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, OPTIONS")
        self.send_header("Cache-Control", "no-cache")
        super().end_headers()

    def guess_type(self, path):
        suffix = Path(path).suffix.lower()
        if suffix in self._EXTRA_MIME:
            return self._EXTRA_MIME[suffix]
        return super().guess_type(path)

    def log_message(self, fmt, *args) -> None:  # type: ignore[override]
        # Suppress per-request access logs to keep C4 console clean
        pass

    def log_error(self, fmt, *args) -> None:  # type: ignore[override]
        logger.warning(f"[Viewer HTTP] " + fmt % args)


# ── Server bootstrap ────────────────────────────────────────────────────────

def start_server(port: int = DEFAULT_PORT) -> None:
    """
    Block forever serving viewer/ at http://localhost:<port>.
    Kill with Ctrl-C or terminate from the parent process.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    os.chdir(SERVE_ROOT)

    server = HTTPServer(("localhost", port), SilentHandler)
    logger.info(f"[Viewer HTTP] Serving on http://localhost:{port}  (root: {SERVE_ROOT})")

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("[Viewer HTTP] Stopped.")
    finally:
        server.server_close()


# ── CLI entry point ─────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="C4 3D Model Viewer static server")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT,
                        help=f"HTTP port (default: {DEFAULT_PORT})")
    args = parser.parse_args()
    start_server(args.port)
