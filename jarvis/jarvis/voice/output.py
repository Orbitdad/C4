"""
JARVIS Voice Output — Neural TTS with multi-backend support.

Priority order:
1. ElevenLabs  (if API key in config — highest quality)
2. Edge-TTS    (free Microsoft neural voices — excellent quality, en-GB-RyanNeural)
3. pyttsx3     (offline fallback)

All backends share the same thread-safe queue interface.
"""

from __future__ import annotations

import asyncio
import io
import logging
import queue
import sys
import threading
import time
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


# ── Abstract Interface ─────────────────────────────────────────────────────────

class VoiceOutput(ABC):
    """Abstract voice output interface."""

    @abstractmethod
    def speak(self, text: str) -> None:
        """Speak the given text (non-blocking queue submission)."""
        pass

    def speak_thinking(self, text: str = "One moment, sir.") -> None:
        """Speak a brief acknowledgement while processing (non-blocking)."""
        self.speak(text)

    def stop(self) -> None:
        """Interrupt current speech."""
        pass

    def set_voice(self, voice_id_or_name: Optional[str] = None) -> None:
        pass


# ── Edge-TTS Backend (Primary — Free Microsoft Neural Voices) ──────────────────

class EdgeTTSOutput(VoiceOutput):
    """
    Neural TTS via Microsoft Edge-TTS (free, no API key required).
    Uses en-GB-RyanNeural by default — deep, authoritative British male voice.
    Falls back to pyttsx3 on any failure.
    """

    RATE = "+10%"
    PITCH = "-8Hz"

    # Language to Voice Mapping
    VOICE_MAP = {
        "en": "en-GB-RyanNeural",
        "hi": "hi-IN-MadhurNeural",
        "mr": "mr-IN-ManoharNeural",
    }

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        hui_window: Optional[Any] = None,
    ) -> None:
        config = config or {}
        voice_cfg = config.get("voice", {}) or {}
        out_cfg = voice_cfg.get("output", {}) or {}

        self.voice_map = out_cfg.get("voices", self.VOICE_MAP)
        self.default_voice = out_cfg.get("edge_voice", self.voice_map.get("en", "en-GB-RyanNeural"))
        self.rate = out_cfg.get("edge_rate", self.RATE)
        self.pitch = out_cfg.get("edge_pitch", self.PITCH)
        self.hui_window = hui_window
        self._config = config

        self._queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._is_speaking = False
        self._fallback: Optional[Pyttsx3Output] = None

        # Dedicated worker thread
        self._thread = threading.Thread(target=self._worker, daemon=True, name="EdgeTTSWorker")
        self._thread.start()
        logger.info(f"[EdgeTTS] Initialized. Voice: {self.default_voice}")

    def _emit_hui(self, status: str, message: str = "") -> None:
        if not self.hui_window:
            return
        try:
            if message:
                self.hui_window.signals.log_message.emit(f"C4: {message}")
            self.hui_window.signals.update_status.emit(status)
        except Exception:
            pass

    def speak(self, text: str) -> None:
        if not text or not text.strip():
            return
        self.stop()
        self._queue.put(("speak", text))

    def speak_thinking(self, text: str = "One moment, sir.") -> None:
        """Queue a thinking acknowledgement ahead of other pending items."""
        if not text or not text.strip():
            return
        self._queue.put(("speak", text))

    def stop(self) -> None:
        self._stop_event.set()
        # Drain queue
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._queue.put(("stop", ""))
        time.sleep(0.05)
        self._stop_event.clear()

    def _play_audio_bytes(self, audio_data: bytes) -> None:
        """Play MP3 bytes using pygame.mixer (handles decoding properly)."""
        try:
            import pygame
            import io
            
            # Initialize mixer if not already done
            if not pygame.mixer.get_init():
                pygame.mixer.init(frequency=24000)
            
            sound_file = io.BytesIO(audio_data)
            pygame.mixer.music.load(sound_file)
            pygame.mixer.music.play()
            
            # Wait for playback to finish or be interrupted
            while pygame.mixer.music.get_busy():
                if self._stop_event.is_set():
                    pygame.mixer.music.stop()
                    break
                time.sleep(0.05)
                
        except Exception as e:
            logger.debug(f"[EdgeTTS] pygame playback failed: {e}. Falling back to pyttsx3.")
            self._fallback_speak_text(self._current_text)

    def _worker(self) -> None:
        """Dedicated TTS worker — runs Edge-TTS synthesis and playback."""
        # Setup asyncio event loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        while True:
            try:
                item = self._queue.get(block=True, timeout=0.2)
                if item is None:
                    break
                action, text = item
                if action == "stop":
                    continue
                if not text:
                    continue

                self._is_speaking = True
                self._current_text = text
                self._emit_hui("Speaking...", text)

                # Select voice based on text content
                selected_voice = self._get_voice_for_text(text)

                try:
                    audio_bytes = loop.run_until_complete(self._synthesize(text, selected_voice))
                    if audio_bytes and not self._stop_event.is_set():
                        self._play_audio_bytes(audio_bytes)
                except Exception as e:
                    logger.warning(f"[EdgeTTS] Synthesis failed: {e}. Falling back to pyttsx3.")
                    self._fallback_speak_text(text)

            except queue.Empty:
                pass
            except Exception as e:
                logger.error(f"[EdgeTTS] Worker error: {e}")
            finally:
                self._is_speaking = False
                self._emit_hui("ONLINE")

    def _get_voice_for_text(self, text: str) -> str:
        """Heuristic to select voice: Devanagari characters -> Hindi/Marathi voice."""
        # Check for Devanagari range (U+0900 to U+097F)
        has_devanagari = any('\u0900' <= char <= '\u097f' for char in text)
        if has_devanagari:
            # Default to Hindi for Devanagari unless Marathi specifically detected (harder without LLM)
            # We use Madhur (Hindi) as a good general Devanagari voice, 
            # or try to differentiate if possible.
            # Simplified: Use Hindi for now as it handles Marathi reasonably well, 
            # but we can look for Marathi-specific markers if needed.
            return self.voice_map.get("hi", "hi-IN-MadhurNeural")
        return self.default_voice

    async def _synthesize(self, text: str, voice: Optional[str] = None) -> Optional[bytes]:
        """Async Edge-TTS synthesis → returns PCM/MP3 bytes."""
        import edge_tts
        communicate = edge_tts.Communicate(
            text=text,
            voice=voice or self.default_voice,
            rate=self.rate,
            pitch=self.pitch,
        )
        audio_chunks = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_chunks.append(chunk["data"])
        return b"".join(audio_chunks) if audio_chunks else None

    def _fallback_speak_text(self, text: str) -> None:
        """Use pyttsx3 as emergency fallback."""
        if self._fallback is None:
            self._fallback = Pyttsx3Output(config=self._config, hui_window=self.hui_window)
        self._fallback.speak(text)


# ── ElevenLabs Backend (Premium optional) ─────────────────────────────────────

class ElevenLabsOutput(VoiceOutput):
    """
    Premium TTS via ElevenLabs API.
    Only activates if 'elevenlabs_api_key' is in config.
    Voice: Adam (deep, authoritative) or custom voice ID.
    """

    DEFAULT_VOICE_ID = "pNInz6obpgDQGcFmaJgB"  # Adam voice

    def __init__(self, api_key: str, config: Optional[Dict[str, Any]] = None, hui_window: Optional[Any] = None) -> None:
        self.api_key = api_key
        self.hui_window = hui_window
        self._config = config or {}
        out_cfg = (config or {}).get("voice", {}).get("output", {}) or {}
        self.voice_id = out_cfg.get("elevenlabs_voice_id", self.DEFAULT_VOICE_ID)

        self._queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._is_speaking = False

        self._thread = threading.Thread(target=self._worker, daemon=True, name="ElevenLabsWorker")
        self._thread.start()
        logger.info("[ElevenLabs] TTS initialized.")

    def speak(self, text: str) -> None:
        if not text or not text.strip():
            return
        self.stop()
        self._queue.put(text)

    def stop(self) -> None:
        self._stop_event.set()
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        time.sleep(0.05)
        self._stop_event.clear()

    def _emit_hui(self, status: str, message: str = "") -> None:
        if not self.hui_window:
            return
        try:
            if message:
                self.hui_window.signals.log_message.emit(f"C4: {message}")
            self.hui_window.signals.update_status.emit(status)
        except Exception:
            pass

    def _worker(self) -> None:
        while True:
            try:
                text = self._queue.get(block=True, timeout=0.2)
                if not text:
                    continue
                self._is_speaking = True
                self._emit_hui("Speaking...", text)
                try:
                    import requests
                    resp = requests.post(
                        f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}/stream",
                        headers={"xi-api-key": self.api_key, "Content-Type": "application/json"},
                        json={"text": text, "model_id": "eleven_monolingual_v1",
                              "voice_settings": {"stability": 0.75, "similarity_boost": 0.85}},
                        stream=True,
                        timeout=15,
                    )
                    if resp.ok:
                        import pyaudio
                        pa = pyaudio.PyAudio()
                        stream = pa.open(format=pyaudio.paInt16, channels=1, rate=22050, output=True)
                        for chunk in resp.iter_content(chunk_size=1024):
                            if self._stop_event.is_set():
                                break
                            if chunk:
                                stream.write(chunk)
                        stream.stop_stream()
                        stream.close()
                        pa.terminate()
                except Exception as e:
                    logger.error(f"[ElevenLabs] Playback error: {e}")
            except queue.Empty:
                pass
            except Exception as e:
                logger.error(f"[ElevenLabs] Worker error: {e}")
            finally:
                self._is_speaking = False
                self._emit_hui("ONLINE")


# ── pyttsx3 Backend (Offline Fallback) ────────────────────────────────────────

class Pyttsx3Output(VoiceOutput):
    """
    TTS via pyttsx3 — offline-capable fallback.
    Thread-safe implementation with dedicated worker loop.
    """

    def __init__(
        self,
        config: Optional[Dict[str, Any]] = None,
        rate: int = 200,
        volume: float = 0.95,
        hui_window: Optional[Any] = None,
    ) -> None:
        config = config or {}
        voice_cfg = config.get("voice", {}) or {}
        out_cfg = voice_cfg.get("output", {}) or {}

        self.rate = out_cfg.get("rate", rate)
        self.volume = min(1.0, max(0, out_cfg.get("volume", volume)))
        self.hui_window = hui_window
        self.voice_name = out_cfg.get("voice_name")

        self._queue: queue.Queue = queue.Queue()
        self._stop_request = False
        self._is_speaking = False

        self._thread = threading.Thread(target=self._tts_worker, daemon=True, name="Pyttsx3Worker")
        self._thread.start()

    def _tts_worker(self) -> None:
        if sys.platform == "win32":
            import pythoncom
            pythoncom.CoInitialize()

        try:
            import pyttsx3
            engine = pyttsx3.init()
            engine.setProperty("rate", self.rate)
            engine.setProperty("volume", self.volume)

            if self.voice_name:
                name_lower = self.voice_name.lower()
                for v in engine.getProperty("voices"):
                    if name_lower in (v.name or "").lower() or name_lower in (v.id or "").lower():
                        engine.setProperty("voice", v.id)
                        break

            def onWord(name, location, length):
                if self._stop_request:
                    engine.stop()

            engine.connect('started-word', onWord)
        except Exception as e:
            logger.error(f"[Pyttsx3] Engine init failed: {e}")
            return

        while True:
            try:
                text = self._queue.get(block=True, timeout=0.1)
                if text == "<STOP_TTS>":
                    continue
                self._stop_request = False
                self._is_speaking = True
                if self.hui_window:
                    try:
                        self.hui_window.signals.log_message.emit(f"C4: {text}")
                        self.hui_window.signals.update_status.emit("Speaking...")
                    except Exception:
                        pass
                try:
                    engine.say(text)
                    engine.runAndWait()
                except RuntimeError:
                    pass
            except queue.Empty:
                pass
            except Exception as e:
                logger.exception(f"[Pyttsx3] Worker error: {e}")
            finally:
                self._is_speaking = False
                if self.hui_window:
                    try:
                        self.hui_window.signals.update_status.emit("ONLINE")
                    except Exception:
                        pass

    def speak(self, text: str) -> None:
        if not text or not text.strip():
            return
        self.stop()
        self._queue.put(text)

    def stop(self) -> None:
        self._stop_request = True
        while not self._queue.empty():
            try:
                self._queue.get_nowait()
            except queue.Empty:
                break
        self._queue.put("<STOP_TTS>")

    def set_voice(self, voice_id_or_name: Optional[str] = None) -> None:
        if voice_id_or_name:
            self.voice_name = voice_id_or_name


# ── Factory Function (Auto-selects best available backend) ─────────────────────

def create_voice_output(config: Optional[Dict[str, Any]] = None, hui_window: Optional[Any] = None) -> VoiceOutput:
    """
    Auto-select the best TTS backend available:
    1. ElevenLabs (if API key configured)
    2. Edge-TTS   (always available with internet)
    3. pyttsx3    (offline fallback)
    """
    config = config or {}
    out_cfg = config.get("voice", {}).get("output", {}) or {}

    # 1. Try ElevenLabs
    el_key = out_cfg.get("elevenlabs_api_key", "")
    if el_key and el_key.strip() and el_key.strip() != "YOUR_KEY_HERE":
        logger.info("[VoiceFactory] Using ElevenLabs TTS (premium).")
        return ElevenLabsOutput(api_key=el_key, config=config, hui_window=hui_window)

    # 2. Try Edge-TTS (check network availability)
    backend = out_cfg.get("backend", "edge").lower()
    if backend != "pyttsx3":
        try:
            import edge_tts  # noqa: F401
            logger.info("[VoiceFactory] Using Edge-TTS (Microsoft Neural — en-GB-RyanNeural).")
            return EdgeTTSOutput(config=config, hui_window=hui_window)
        except ImportError:
            logger.warning("[VoiceFactory] edge-tts not installed. Falling back to pyttsx3.")

    # 3. pyttsx3 fallback
    logger.info("[VoiceFactory] Using pyttsx3 (offline fallback).")
    return Pyttsx3Output(config=config, hui_window=hui_window)
