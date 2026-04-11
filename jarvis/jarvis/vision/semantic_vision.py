import logging
import time
import cv2
import numpy as np
import threading
from jarvis.core.event_bus import bus, SystemEvent, EventPriority

# Optional dependencies for Advanced Vision
try:
    import pytesseract
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'
    HAS_TESSERACT = True
except Exception:
    HAS_TESSERACT = False

try:
    import mss
    HAS_MSS = True
except Exception:
    HAS_MSS = False

logger = logging.getLogger(__name__)

class SemanticVisionEngine:
    """
    Advanced Perception module extending normal webcams with Scene Analysis and Screen OCR.
    """
    def __init__(self):
        self.is_running = False
        self._thread = None
        self.last_ocr_text = ""
        
    def start(self):
        if self.is_running: return
        self.is_running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info(f"[SemanticVision] Online. OCR Enabled: {HAS_TESSERACT and HAS_MSS}")
        
    def stop(self):
        self.is_running = False
        if self._thread:
            self._thread.join()
            
    def _run_loop(self):
        while self.is_running:
            time.sleep(10) # Poll every 10s
            if HAS_MSS and HAS_TESSERACT:
                try:
                    with mss.mss() as sct:
                        monitor = sct.monitors[1]
                        
                        w, h = monitor["width"], monitor["height"]
                        # Grab middle 50% of screen
                        bbox = {"top": int(h*0.25), "left": int(w*0.25), "width": int(w*0.5), "height": int(h*0.5)}
                        img = np.array(sct.grab(bbox))
                        
                        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
                        # Basic thresholding to help OCR
                        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
                        
                        text = pytesseract.image_to_string(thresh).strip()
                        
                        # Only publish if substantial new text is found on screen
                        if text and text != self.last_ocr_text and len(text) > 5:
                            self.last_ocr_text = text
                            logger.debug(f"[SemanticVision] Found Screen Text.")
                            bus.publish(SystemEvent(
                                name="vision.screen.ocr",
                                data={"text": text},
                                priority=EventPriority.LOW
                            ))
                except Exception as e:
                    logger.debug(f"[SemanticVision] Error during OCR: {e}")
