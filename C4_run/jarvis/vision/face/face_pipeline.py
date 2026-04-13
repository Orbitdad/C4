"""
Face Pipeline: Orchestrates real-time face detection, recognition, and event emission.

Architecture Decision:
    Runs in its own daemon thread to avoid blocking:
    1. The UI thread (PyQt5)
    2. The gesture processing thread in VisionManager

    Shares the camera frame via a thread-safe `latest_frame` attribute on
    VisionManager rather than opening a second VideoCapture device (only one
    process can own the camera at a time on Windows).

    Frame skipping strategy:
    - Processes every Nth frame (default N=5) to target ~15+ FPS on CPU.
    - The identity result is cached between skipped frames so that downstream
      consumers always have a "last known" state rather than gaps.

Event Bus Schema:
    face.user_detected  → {user_id, display_name, confidence, bbox, attention}
    face.user_left      → {user_id, display_name}
    face.unknown_user   → {confidence, bbox, attention}
    face.pipeline_error → {error}

World State Keys:
    user_environment.current_user       → display_name or None
    user_environment.current_user_id    → uuid string or None
    user_environment.user_presence      → bool
    user_environment.attention_state    → "looking_at_screen" | "looking_away" | "unknown"
    user_environment.face_count         → int
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any, Dict, List, Optional

import numpy as np

from jarvis.core.event_bus import bus, SystemEvent, EventPriority
from jarvis.core.world_state import world
from .detector import FaceDetector, DetectedFace
from .identity_manager import IdentityManager

logger = logging.getLogger(__name__)


def _estimate_attention(pose: Optional[tuple]) -> str:
    """
    Map (yaw, pitch, roll) pose angles to a human-readable attention state.
    Thresholds empirically calibrated for typical desktop usage:
      - |yaw| < 20°  → looking at screen
      - |yaw| >= 20° → looking away
    """
    if pose is None:
        return "unknown"
    yaw, pitch, _ = pose
    if abs(yaw) < 20 and abs(pitch) < 25:
        return "looking_at_screen"
    return "looking_away"


class FacePipeline:
    """
    Background face recognition pipeline.

    Lifecycle:
        pipeline = FacePipeline(vision_manager, identity_manager)
        pipeline.start()   # begins daemon thread
        pipeline.stop()    # gracefully shuts down
    """

    # Process every Nth frame to avoid overloading CPU
    FRAME_SKIP = 5
    # Seconds a known user must be absent before USER_LEFT fires
    PRESENCE_TIMEOUT = 5.0
    # Default recognition threshold (kept configurable)
    RECOGNITION_THRESHOLD = 0.45

    def __init__(
        self,
        vision_manager: Any,             # VisionManager instance
        identity_manager: IdentityManager,
        threshold: float = RECOGNITION_THRESHOLD,
        db_path: Optional[str] = None,
    ) -> None:
        self.vision_manager = vision_manager
        self.identity_manager = identity_manager
        self.threshold = threshold

        self._detector = FaceDetector()
        self._is_running = False
        self._thread: Optional[threading.Thread] = None

        # State tracking for presence/leaving detection
        self._last_seen_user_id: Optional[str] = None
        self._last_seen_user_name: Optional[str] = None
        self._last_face_ts: float = 0.0

        # Cached last result for consumers that poll instead of subscribing
        self._last_result: Dict[str, Any] = {"faces": []}
        self._result_lock = threading.Lock()

    # ─── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self) -> None:
        if self._is_running:
            return
        if not self._detector.is_ready:
            logger.warning(
                "[FacePipeline] Detector not ready (InsightFace unavailable). "
                "Face recognition disabled."
            )
            return
        self._is_running = True
        self._thread = threading.Thread(
            target=self._run_loop, name="FacePipeline", daemon=True
        )
        self._thread.start()
        logger.info("[FacePipeline] Started.")

    def stop(self) -> None:
        self._is_running = False
        if self._thread:
            self._thread.join(timeout=3.0)
        logger.info("[FacePipeline] Stopped.")

    @property
    def last_result(self) -> Dict[str, Any]:
        """Thread-safe access to the most recent pipeline output."""
        with self._result_lock:
            return dict(self._last_result)

    # ─── Core Loop ─────────────────────────────────────────────────────────────

    def _run_loop(self) -> None:
        frame_idx = 0

        while self._is_running:
            frame = getattr(self.vision_manager, "latest_frame", None)

            if frame is None:
                time.sleep(0.05)  # Wait for VisionManager to produce a frame
                continue

            frame_idx += 1
            if frame_idx % self.FRAME_SKIP != 0:
                time.sleep(0.01)
                # Check presence timeout even on skipped frames
                self._check_presence_timeout()
                continue

            try:
                inf_lock = getattr(self.vision_manager, "inference_lock", None)
                if inf_lock:
                    with inf_lock:
                        structured = self._process_frame(frame)
                else:
                    structured = self._process_frame(frame)
                    
                with self._result_lock:
                    self._last_result = structured
            except Exception as e:
                logger.error(f"[FacePipeline] Frame processing error: {e}")
                bus.publish(SystemEvent(
                    name="face.pipeline_error",
                    data={"error": str(e)},
                    priority=EventPriority.LOW,
                ))

            time.sleep(0.01)

    def _process_frame(self, frame: np.ndarray) -> Dict[str, Any]:
        """
        Run full detection + recognition on a single frame.
        Returns a structured dict suitable for UI display and downstream agents.
        """
        detected_faces: List[DetectedFace] = self._detector.detect(frame)

        faces_output = []
        recognized_this_frame: Optional[str] = None
        recognized_name_this_frame: Optional[str] = None

        for face in detected_faces:
            face_entry = self._recognize_face(face)
            faces_output.append(face_entry)

            # Track the highest-confidence recognized user in this frame
            if face_entry["id"] and (
                recognized_this_frame is None
                or face_entry["confidence"] > (faces_output[0]["confidence"] if faces_output else 0)
            ):
                recognized_this_frame = face_entry["user_id"]
                recognized_name_this_frame = face_entry["id"]

        # Update presence state and fire events
        self._update_presence(
            face_count=len(detected_faces),
            recognized_user_id=recognized_this_frame,
            recognized_user_name=recognized_name_this_frame,
            faces_output=faces_output,
        )

        return {"faces": faces_output, "timestamp": time.time()}

    def _recognize_face(self, face: DetectedFace) -> Dict[str, Any]:
        """Run identity matching on a single DetectedFace and return a structured dict."""
        attention = _estimate_attention(face.pose)

        # Skip match if embedding is missing (detection score too low)
        if face.embedding is None or face.detection_score < 0.5:
            return {
                "id": None,
                "user_id": None,
                "display_name": "Unknown",
                "confidence": 0.0,
                "bbox": list(face.bbox),
                "attention": attention,
            }

        user_id, display_name, confidence = self.identity_manager.match(
            face.embedding, threshold=self.threshold
        )

        return {
            "id": display_name,            # Human-readable label for UI
            "user_id": user_id,            # UUID for downstream systems
            "display_name": display_name or "Unknown",
            "confidence": round(confidence, 4),
            "bbox": list(face.bbox),
            "attention": attention,
        }

    # ─── Presence & Event Emission ─────────────────────────────────────────────

    def _update_presence(
        self,
        face_count: int,
        recognized_user_id: Optional[str],
        recognized_user_name: Optional[str],
        faces_output: List[Dict],
    ) -> None:
        """Update WorldState and fire EventBus events based on current frame results."""

        now = time.time()

        if face_count > 0:
            self._last_face_ts = now

        # Update world state
        attention_states = [f.get("attention", "unknown") for f in faces_output]
        dominant_attention = (
            "looking_at_screen"
            if "looking_at_screen" in attention_states
            else ("looking_away" if attention_states else "unknown")
        )

        world.update_environment(
            user_presence=face_count > 0,
            face_count=face_count,
            attention_state=dominant_attention,
            current_user=recognized_user_name,
            current_user_id=recognized_user_id,
        )

        # ── Event: Known user appeared ──
        if recognized_user_id and recognized_user_id != self._last_seen_user_id:
            confidence = next(
                (f["confidence"] for f in faces_output if f.get("user_id") == recognized_user_id),
                0.0,
            )
            bus.publish(SystemEvent(
                name="face.user_detected",
                data={
                    "user_id": recognized_user_id,
                    "display_name": recognized_user_name,
                    "confidence": confidence,
                    "faces": faces_output,
                },
                priority=EventPriority.HIGH,
            ))
            logger.info(
                f"[FacePipeline] USER DETECTED: {recognized_user_name} "
                f"(id={recognized_user_id}, conf={confidence:.2f})"
            )
            self._last_seen_user_id = recognized_user_id
            self._last_seen_user_name = recognized_user_name

        # ── Event: Unknown user appeared ──
        elif face_count > 0 and not recognized_user_id:
            # Fire sparingly — only when transitioning from no-face to unknown-face
            if self._last_seen_user_id is not None or (now - self._last_face_ts > 3.0):
                unknown_entries = [f for f in faces_output if f.get("user_id") is None]
                if unknown_entries:
                    bus.publish(SystemEvent(
                        name="face.unknown_user",
                        data={"faces": unknown_entries},
                        priority=EventPriority.NORMAL,
                    ))
                    logger.info("[FacePipeline] UNKNOWN USER detected.")

        # ── Presence timeout: trigger USER_LEFT ──
        self._check_presence_timeout()

    def _check_presence_timeout(self) -> None:
        """
        If we had a known user and they haven't appeared in PRESENCE_TIMEOUT seconds,
        fire USER_LEFT and clear world state.
        """
        if self._last_seen_user_id is None:
            return
        if time.time() - self._last_face_ts > self.PRESENCE_TIMEOUT:
            bus.publish(SystemEvent(
                name="face.user_left",
                data={
                    "user_id": self._last_seen_user_id,
                    "display_name": self._last_seen_user_name,
                },
                priority=EventPriority.HIGH,
            ))
            logger.info(f"[FacePipeline] USER LEFT: {self._last_seen_user_name}")
            self._last_seen_user_id = None
            self._last_seen_user_name = None
            world.update_environment(
                user_presence=False,
                current_user=None,
                current_user_id=None,
                attention_state="unknown",
            )

    # ─── Registration Helper ───────────────────────────────────────────────────

    def register_current_user(self, display_name: str) -> Optional[str]:
        """
        Convenience method: capture the best face in the latest frame and register it.
        Call this when the user says "JARVIS, remember my face as Adarsh".

        Returns:
            user_id on success, None if no face detected.
        """
        frame = getattr(self.vision_manager, "latest_frame", None)
        if frame is None:
            logger.warning("[FacePipeline] register_current_user: no frame available.")
            return None

        faces = self._detector.detect(frame)
        if not faces:
            logger.warning("[FacePipeline] register_current_user: no face detected.")
            return None

        # Pick highest-confidence face
        best = max(faces, key=lambda f: f.detection_score)
        if best.embedding is None:
            return None

        return self.identity_manager.register_user(display_name, best.embedding)
