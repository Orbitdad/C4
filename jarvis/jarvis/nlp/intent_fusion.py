import time
import logging
from typing import Optional, Dict, Any
from jarvis.core.event_bus import bus, SystemEvent, EventPriority
from jarvis.nlp.schemas import Intent

logger = logging.getLogger(__name__)

class UnifiedIntentEngine:
    """
    Fuses multi-modal inputs.
    e.g., Combines Voice ("close this") + Gesture (pointing at "Chrome") 
    -> Fused Intent ("close Chrome").
    """
    def __init__(self):
        self._last_voice_query: str = ""
        self._last_voice_ts: float = 0.0
        
        self._last_pointer_focus: str = ""
        self._last_pointer_ts: float = 0.0

        self._last_ocr_text: str = ""
        self._last_ocr_ts: float = 0.0
        
        bus.subscribe("voice.raw_transcript", self._on_voice)
        bus.subscribe("vision.pointer.focus", self._on_pointer_focus)
        bus.subscribe("vision.screen.ocr", self._on_screen_ocr)

    def _on_voice(self, event: SystemEvent):
        transcript = event.data.get("transcript", "")
        self._last_voice_query = transcript
        self._last_voice_confidence = event.data.get("confidence", 0.9)
        self._last_voice_ts = event.timestamp
        logger.debug(f"[IntentFusion] Received voice: {transcript} (conf: {self._last_voice_confidence})")
        
    def _on_pointer_focus(self, event: SystemEvent):
        focus = event.data.get("focus", "")
        self._last_pointer_focus = focus
        self._last_pointer_confidence = event.data.get("confidence", 0.8)
        self._last_pointer_ts = event.timestamp
        logger.debug(f"[IntentFusion] Pointer focused on: {focus} (conf: {self._last_pointer_confidence})")

    def _on_screen_ocr(self, event: SystemEvent):
        text = event.data.get("text", "")
        if text:
            self._last_ocr_text = text
            self._last_ocr_ts = event.timestamp
            logger.debug(f"[IntentFusion] Received OCR Text.")

    def fuse_context(self, raw_transcript: str) -> str:
        """
        Replaces demonstrative pronouns with the currently focused visual target.
        Calculates IntentScore: w1*VoiceConf + w2*GestureConf (Time Delta)
        """
        now = time.time()
        
        # Calculate gesture weight (decays over 3 seconds - tighter temporal alignment)
        time_delta = now - self._last_pointer_ts
        w2_gesture = max(0.0, 1.0 - (time_delta / 3.0))
        
        # Calculate voice weight (default 1.0)
        w1_voice = 1.0
        
        # Confidence merging algorithm
        voice_conf = getattr(self, '_last_voice_confidence', 0.9)
        pointer_conf = getattr(self, '_last_pointer_confidence', 0.8)
        
        intent_score = (w1_voice * voice_conf * 0.6) + (w2_gesture * pointer_conf * 0.4)
        
        # Explainability metadata
        explanation = ""
        
        # 1. Fuse Screen OCR
        ocr_delta = now - getattr(self, "_last_ocr_ts", 0)
        if ocr_delta < 30.0 and self._last_ocr_text:  # Screen text is valid for longer
            ocr_triggers = ["the screen", "this text", "what it says", "on screen"]
            if any(t in raw_transcript.lower() for t in ocr_triggers):
                fused = raw_transcript
                for t in ocr_triggers:
                    fused = fused.replace(t, f"the text '{self._last_ocr_text[:100]}...'")
                explanation = "Injected OCR screen content into request."
                logger.info(f"[IntentFusion] Fused OCR Intent: '{fused}'.")
                return fused
        
        # 2. Fuse Pointing Gestures
        if intent_score > 0.6 and self._last_pointer_focus:
            words = raw_transcript.lower().split()
            target_words = ["this", "that", "it", "here"]
            
            if any(w in words for w in target_words):
                fused = raw_transcript
                for w in target_words:
                    fused = fused.replace(f" {w}", f" {w} [{self._last_pointer_focus}]")
                    if fused.startswith(f"{w} "):
                         fused = fused.replace(f"{w} ", f"{self._last_pointer_focus} ", 1)
                    
                explanation = f"Using '{self._last_pointer_focus}' because you looked/pointed at it {time_delta:.1f}s ago."
                logger.info(f"[IntentFusion] Fused Intent (Score: {intent_score:.2f}): '{fused}'. {explanation}")
                
                # We could attach this explanation to the world_state or context for the UI
                from jarvis.core.world_state import world
                world.set_temporal_context("last_fusion_explanation", explanation, ttl_seconds=10)
                
                return fused
                
        return raw_transcript
