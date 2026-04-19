"""
Vision AI module: Hand tracking and object detection.
"""

from __future__ import annotations
import cv2
import mediapipe as mp
import threading
from typing import Optional, Dict, Any

from jarvis.vision.gesture import GestureController


class VisionManager:
    """
    Background vision processor using OpenCV and MediaPipe.
    Tracks hands and detects simple visual states.
    """

    def __init__(self, hui_window: Optional[Any] = None) -> None:
        self.mp_hands = None
        self.hands = None
        self.mp_draw = None
        try:
            # Prefer stable public API first; older/newer wheels may not expose
            # the internal mediapipe.python package layout consistently.
            self.mp_hands = mp.solutions.hands
            self.mp_pose = mp.solutions.pose
            self.mp_draw = mp.solutions.drawing_utils
            
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.5
            )
        except Exception as e:
            try:
                import mediapipe.python.solutions.hands as mp_hands
                import mediapipe.python.solutions.drawing_utils as mp_draw
                self.mp_hands = mp_hands
                self.mp_draw = mp_draw
                self.hands = self.mp_hands.Hands(
                    static_image_mode=False,
                    max_num_hands=2,
                    min_detection_confidence=0.7,
                    min_tracking_confidence=0.5
                )
            except Exception as e2:
                print(f"Warning: Vision initialization failed: {e2}")
        self.cap: Optional[cv2.VideoCapture] = None
        self.is_running = False
        self._thread: Optional[threading.Thread] = None
        self.last_detection: Dict[str, Any] = {"hands": 0, "gestures": [], "motion_level": 0.0, "faces": 0}
        self.hui_window = hui_window
        
        self.gesture_controller = GestureController()
        
        # Shared camera frame — written by _run_loop, read by FacePipeline (zero-copy)
        self.latest_frame = None
        self._latest_frame_lock = threading.Lock()
        self.inference_lock = threading.Lock()
        self._face_pipeline = None
        
        try:
             self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        except Exception:
             self.face_cascade = None
        
        # Fallback tracking assets
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=50, detectShadows=True)
        self.fallback_mode = False
        if not self.hands:
             self.fallback_mode = True
             print("HUI: Initializing Computer Vision Fallback (Kinetic Mode)...")

    def start(self) -> None:
        """Start the vision processing thread."""
        if self.is_running:
            return
        self.is_running = True
        self.cap = cv2.VideoCapture(0)
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop the vision processing thread."""
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=2.0)
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

    # ── MediaPipe drawing spec (created once) ──────────────────────────────
    _LANDMARK_SPEC  = None
    _CONNECTION_SPEC = None

    def _get_draw_specs(self):
        """Lazily create drawing specs so we don't import at module level."""
        if VisionManager._LANDMARK_SPEC is None:
            try:
                from mediapipe.solutions.drawing_styles import (
                    get_default_hand_landmarks_style,
                    get_default_hand_connections_style,
                )
                VisionManager._LANDMARK_SPEC   = get_default_hand_landmarks_style()
                VisionManager._CONNECTION_SPEC = get_default_hand_connections_style()
            except Exception:
                # Fallback: plain white dots / cyan lines
                _du = getattr(self, "mp_draw", None)
                if _du is not None and hasattr(_du, 'DrawingSpec'):
                    VisionManager._LANDMARK_SPEC   = _du.DrawingSpec(color=(255, 255, 255), thickness=2, circle_radius=4)
                    VisionManager._CONNECTION_SPEC = _du.DrawingSpec(color=(0, 210, 255),   thickness=2)
                else:
                    VisionManager._LANDMARK_SPEC   = None
                    VisionManager._CONNECTION_SPEC = None
        return VisionManager._LANDMARK_SPEC, VisionManager._CONNECTION_SPEC



    def _run_loop(self) -> None:
        frame_idx = 0
        while self.is_running and self.cap.isOpened():
            success, frame = self.cap.read()
            if not success:
                continue

            frame_idx += 1
            frame = cv2.flip(frame, 1)

            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # ── Share frame for FacePipeline consumption ──
            with self._latest_frame_lock:
                self.latest_frame = frame.copy()
            hand_count = 0
            landmarks = []
            all_hands = []
            annotated = frame.copy()   # We'll draw on this copy

            if self.hands:
                with self.inference_lock:
                    results = self.hands.process(img_rgb)
                if results.multi_hand_landmarks:
                    hand_count = len(results.multi_hand_landmarks)
                    lm_spec, conn_spec = self._get_draw_specs()
                    for hand_lms in results.multi_hand_landmarks:
                        # ── Draw skeleton + landmark dots ──────────────────
                        self.mp_draw.draw_landmarks(
                            annotated,
                            hand_lms,
                            self.mp_hands.HAND_CONNECTIONS,
                            lm_spec,
                            conn_spec,
                        )

                    all_hands = [h.landmark for h in results.multi_hand_landmarks] if results.multi_hand_landmarks else []
                    # Extract index finger tip (landmark id 8) for UI updates of all hands
                    for h in all_hands:
                        idx_tip = h[8]
                        landmarks.append((idx_tip.x, idx_tip.y))
                        
                if self.pose:
                    with self.inference_lock:
                        pose_results = self.pose.process(img_rgb)
                    if pose_results and pose_results.pose_landmarks:
                        self.mp_draw.draw_landmarks(
                            annotated,
                            pose_results.pose_landmarks,
                            self.mp_pose.POSE_CONNECTIONS,
                            self.mp_draw.DrawingSpec(color=(245,117,66), thickness=2, circle_radius=2),
                            self.mp_draw.DrawingSpec(color=(245,66,230), thickness=2, circle_radius=2)
                        )
                    
                    
                    # Pass ALL hands to the gesture controller (pose is no longer used for gestures)
                    debug_dict = self.gesture_controller.process_all_hands(all_hands)
                    
                    if self.hui_window and debug_dict:
                        if hasattr(self.hui_window.signals, 'update_gesture_debug'):
                            self.hui_window.signals.update_gesture_debug.emit(debug_dict)
                    
                    # Extract index finger tip (landmark id 8) for UI updates of all hands
                    for h in all_hands:
                        idx_tip = h[8]
                        landmarks.append((idx_tip.x, idx_tip.y))
            elif self.fallback_mode:
                # Use motion + contour fallback
                fg_mask = self.bg_subtractor.apply(frame)
                _, thresh = cv2.threshold(fg_mask, 200, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    # Find largest moving contour (assumed to be hand)
                    largest = max(contours, key=cv2.contourArea)
                    area = cv2.contourArea(largest)
                    if area > 1000:
                        M = cv2.moments(largest)
                        if M["m00"] != 0:
                            cx = int(M["m10"] / M["m00"])
                            cy = int(M["m01"] / M["m00"])
                            cv2.circle(annotated, (cx, cy), 12, (0, 210, 255), 2)
                            cv2.putText(annotated, "KINETIC", (cx - 30, cy - 20),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 210, 255), 2, cv2.LINE_AA)
                            landmarks.append((cx / self.cap.get(3), cy / self.cap.get(4)))
                            hand_count = 1

                # Calculate general motion level
                self.last_detection["motion_level"] = cv2.countNonZero(thresh) / (frame.shape[0] * frame.shape[1])

            # ── Emit annotated frame (skeleton/fallback overlay) to the VISION MATRIX ──
            if self.hui_window:
                self.hui_window.signals.update_vision.emit(annotated)

            if self.hui_window and landmarks:
                self.hui_window.signals.update_landmarks.emit(landmarks)

            # Face Detection is very heavy, do it every 15 frames
            face_count = self.last_detection.get("faces", 0)
            if self.face_cascade and frame_idx % 15 == 0:
                # Downscale for faster detection
                small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                gray = cv2.cvtColor(small_frame, cv2.COLOR_BGR2GRAY)
                faces = self.face_cascade.detectMultiScale(gray, 1.1, 4)
                face_count = len(faces)
                
            self.last_detection = {
                "hands": hand_count,
                "faces": face_count,
                "timestamp": cv2.getTickCount(),
                "motion_level": self.last_detection.get("motion_level", 0.0)
            }

            # Optional: Show a debug window if needed (usually headless for Jarvis)
            # cv2.imshow("JARVIS Vision", frame)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break

    def get_status(self) -> str:
        """Return a natural language description of what is seen."""
        hands = self.last_detection.get("hands", 0)
        
        faces = self.last_detection.get("faces", 0)
        
        # Analyze light levels
        if self.cap and self.cap.isOpened():
             ret, frame = self.cap.read()
             if ret:
                  gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                  avg_bright = gray.mean()
                  light_lvl = "low-light" if avg_bright < 50 else "standard"
                  motion = self.last_detection.get("motion_level", 0.0)
                  motion_desc = "significant activity" if motion > 0.1 else "minimal movement"
                  
                  person_desc = f"{faces} person(s)" if faces > 0 else "no one"
                  return f"I see {person_desc} in a {light_lvl} environment. Pointer tracking active with {motion_desc}."

        return "Optical sensors are currently re-calibrating."

    def start_face_recognition(
        self,
        identity_manager=None,
        threshold: float = 0.45,
        db_path: str = None,
    ):
        """
        Initialize and start the FacePipeline in the background.
        Safe to call after start(). No-op if already running.
        """
        from jarvis.vision.face.face_pipeline import FacePipeline
        from jarvis.vision.face.identity_manager import IdentityManager
        from jarvis.core.world_state import world
        from pathlib import Path

        if self._face_pipeline and self._face_pipeline._is_running:
            return self._face_pipeline

        _idm = identity_manager or IdentityManager(
            db_path=Path(db_path) if db_path else None
        )
        self._face_pipeline = FacePipeline(
            vision_manager=self,
            identity_manager=_idm,
            threshold=threshold,
        )
        self._face_pipeline.start()
        world.update_environment(face_recognition_active=True)
        return self._face_pipeline

    def stop_face_recognition(self):
        """Gracefully stop the FacePipeline."""
        if self._face_pipeline:
            self._face_pipeline.stop()
            self._face_pipeline = None
        from jarvis.core.world_state import world
        world.update_environment(face_recognition_active=False)
