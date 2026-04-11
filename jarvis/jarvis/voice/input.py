"""
Voice input layer: Speech-to-Text, Emotion Detection, Speaker Identification.
"""

from __future__ import annotations

import os
import json
import hashlib
import logging
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Any, Dict, Tuple

import speech_recognition as sr

logger = logging.getLogger(__name__)


# ── Emotion Detection (heuristic, no heavy ML model) ──────────────────────────

def _detect_emotion_heuristic(audio: sr.AudioData) -> Tuple[str, float]:
    """
    Lightweight emotion detection using raw audio energy and duration.
    Returns (emotion_label, confidence).

    This is a heuristic approach that works offline without any ML model:
    - High energy + short phrase  → frustrated/angry
    - High energy + long phrase   → excited/happy
    - Low energy + slow speech    → tired/sad
    - Normal energy               → neutral/focused
    """
    try:
        raw = audio.get_raw_data(convert_rate=16000, convert_width=2)
        import struct
        samples = struct.unpack(f"{len(raw)//2}h", raw)
        if not samples:
            return "neutral", 0.5

        # RMS energy
        rms = (sum(s * s for s in samples) / len(samples)) ** 0.5
        duration = len(samples) / 16000  # seconds

        # Normalize to 0-1 range (typical speech RMS: 500-8000)
        energy_norm = min(rms / 6000.0, 1.0)

        if energy_norm > 0.7 and duration < 2.0:
            return "frustrated", round(0.5 + energy_norm * 0.3, 2)
        elif energy_norm > 0.65 and duration >= 2.0:
            return "excited", round(0.5 + energy_norm * 0.25, 2)
        elif energy_norm < 0.2 and duration > 3.0:
            return "tired", round(0.55, 2)
        elif 0.3 < energy_norm < 0.65:
            return "focused", round(0.6, 2)
        else:
            return "neutral", round(0.5 + energy_norm * 0.1, 2)
    except Exception as e:
        logger.debug(f"[VoiceInput] Emotion detection failed: {e}")
        return "neutral", 0.5


# ── Speaker Identification ─────────────────────────────────────────────────────

class SpeakerIdentifier:
    """
    Lightweight speaker identification using audio fingerprinting.
    Compares a simple energy-band signature against stored profiles.
    Not biometric-grade, but useful for distinguishing registered users
    from strangers without any cloud dependency.

    Profiles are stored at: data/voice_profiles/<name>.json
    """

    def __init__(self, profiles_dir: Optional[Path] = None):
        self._profiles_dir = profiles_dir or Path("data/voice_profiles")
        self._profiles: Dict[str, Dict] = {}
        self._lock = threading.Lock()
        self._load_profiles()

    def _load_profiles(self):
        """Load all voice profiles from disk."""
        if not self._profiles_dir.exists():
            return
        for f in self._profiles_dir.glob("*.json"):
            try:
                profile = json.loads(f.read_text(encoding="utf-8"))
                self._profiles[f.stem] = profile
            except Exception:
                pass
        if self._profiles:
            logger.info(f"[SpeakerID] Loaded {len(self._profiles)} voice profile(s): {list(self._profiles.keys())}")

    def _extract_signature(self, audio: sr.AudioData) -> Optional[str]:
        """Extract a simple audio fingerprint (energy bands hash)."""
        try:
            raw = audio.get_raw_data(convert_rate=16000, convert_width=2)
            import struct, math
            samples = struct.unpack(f"{len(raw)//2}h", raw)
            if len(samples) < 1600:
                return None
            # Divide into 10 bands and compute RMS per band
            band_size = len(samples) // 10
            bands = []
            for i in range(10):
                band = samples[i * band_size:(i + 1) * band_size]
                rms = math.sqrt(sum(s * s for s in band) / len(band))
                # Quantize to reduce sensitivity to recording conditions
                bands.append(int(rms / 200))
            return json.dumps(bands)
        except Exception:
            return None

    def identify(self, audio: sr.AudioData) -> Tuple[Optional[str], float]:
        """
        Match audio against stored profiles.
        Returns (speaker_name, confidence) or (None, 0.0) if no match.
        """
        sig = self._extract_signature(audio)
        if not sig or not self._profiles:
            return None, 0.0

        try:
            candidate_bands = json.loads(sig)
            best_name = None
            best_score = 0.0

            for name, profile in self._profiles.items():
                stored_bands = profile.get("signature_bands")
                if not stored_bands or len(stored_bands) != len(candidate_bands):
                    continue
                # Similarity = 1 - normalized mean absolute difference
                diffs = [abs(a - b) for a, b in zip(candidate_bands, stored_bands)]
                max_val = max(max(candidate_bands), max(stored_bands), 1)
                similarity = 1.0 - (sum(diffs) / (len(diffs) * max_val))
                if similarity > best_score:
                    best_score = similarity
                    best_name = name

            # Only accept if similarity is reasonably high
            if best_score > 0.6 and best_name:
                return best_name, round(best_score, 2)
        except Exception as e:
            logger.debug(f"[SpeakerID] Matching failed: {e}")

        return None, 0.0

    def register(self, name: str, audio: sr.AudioData) -> bool:
        """Register a new voice profile for person 'name'."""
        sig = self._extract_signature(audio)
        if not sig:
            logger.warning(f"[SpeakerID] Could not extract signature for {name}.")
            return False
        try:
            self._profiles_dir.mkdir(parents=True, exist_ok=True)
            profile = {"name": name, "signature_bands": json.loads(sig), "registered_at": time.time()}
            path = self._profiles_dir / f"{name.lower()}.json"
            path.write_text(json.dumps(profile, indent=2), encoding="utf-8")
            with self._lock:
                self._profiles[name.lower()] = profile
            logger.info(f"[SpeakerID] Registered voice profile for '{name}'.")
            return True
        except Exception as e:
            logger.error(f"[SpeakerID] Registration failed: {e}")
            return False


# ── Voice Input Base ──────────────────────────────────────────────────────────

class VoiceInput(ABC):
    """Abstract voice input interface."""

    @abstractmethod
    def listen_once(self, timeout: Optional[float] = None) -> Optional[str]:
        """Listen for a single utterance and return transcribed text, or None on failure."""
        pass


# ── Main Speech Recognition Input ────────────────────────────────────────────

class SpeechRecognitionInput(VoiceInput):
    """
    Microphone-based STT using the speech_recognition library.
    Uses Google Speech Recognition by default (free, requires internet).
    Augmented with:
      - Emotion detection (heuristic, offline)
      - Speaker identification (energy-band fingerprinting, offline)
      - EventBus publishing for all detections
    """

    def __init__(
        self,
        energy_threshold: int = 300,
        pause_threshold: float = 0.8,
        dynamic_energy: bool = True,
        hui_window: Optional[Any] = None,
        profiles_dir: Optional[Path] = None,
        language: str = "en-US",
    ) -> None:
        self.recognizer = sr.Recognizer()
        self.energy_threshold = energy_threshold
        self.pause_threshold = pause_threshold
        self.dynamic_energy = dynamic_energy
        self.hui_window = hui_window
        self._speaker_id = SpeakerIdentifier(profiles_dir)
        self.default_language = language

    def listen_once(self, timeout: Optional[float] = None, language: Optional[str] = None) -> Optional[str]:
        """Capture audio, transcribe, detect emotion and speaker, publish events."""
        from jarvis.core.event_bus import bus, SystemEvent, EventPriority
        from jarvis.core.world_state import world

        lang = language or self.default_language

        timeout = timeout or 5
        audio: Optional[sr.AudioData] = None

        with sr.Microphone() as source:
            if self.hui_window:
                self.hui_window.signals.update_status.emit("Listening...")
            self.recognizer.energy_threshold = self.energy_threshold
            self.recognizer.pause_threshold = self.pause_threshold
            self.recognizer.dynamic_energy_threshold = self.dynamic_energy
            try:
                audio = self.recognizer.listen(source, timeout=timeout, phrase_time_limit=15)
            except sr.WaitTimeoutError:
                if self.hui_window:
                    self.hui_window.signals.update_status.emit("Idle")
                return None
            except OSError as e:
                if self.hui_window:
                    self.hui_window.signals.update_status.emit("Idle")
                if "No Default Input Device" in str(e) or "invalid" in str(e).lower():
                    return None
                raise

        if audio is None:
            return None

        # ── Emotion Detection ──────────────────────────────────────────────
        emotion, emotion_conf = _detect_emotion_heuristic(audio)
        world.set_user_emotion(emotion, emotion_conf)
        bus.publish(SystemEvent(
            name="voice.emotion_detected",
            data={"emotion": emotion, "confidence": emotion_conf},
            priority=EventPriority.LOW,
        ))
        logger.debug(f"[VoiceInput] Emotion: {emotion} ({emotion_conf:.0%})")

        # ── Speaker Identification ─────────────────────────────────────────
        speaker_name, speaker_conf = self._speaker_id.identify(audio)
        if speaker_name and speaker_conf > 0.6:
            world.update_user_model(name=speaker_name.capitalize())
            bus.publish(SystemEvent(
                name="voice.speaker_identified",
                data={"name": speaker_name, "confidence": speaker_conf},
                priority=EventPriority.NORMAL,
            ))
            logger.info(f"[VoiceInput] Speaker identified: {speaker_name} ({speaker_conf:.0%})")

        # ── Transcription ──────────────────────────────────────────────────
        try:
            if self.hui_window:
                self.hui_window.signals.update_status.emit("Thinking...")
            text = self.recognizer.recognize_google(audio, language=lang)
            if text:
                if self.hui_window:
                    self.hui_window.signals.log_message.emit(f"User: {text}")
                # Publish raw transcript for IntentFusion
                bus.publish(SystemEvent(
                    name="voice.raw_transcript",
                    data={"transcript": text, "confidence": 0.9, "speaker": speaker_name, "emotion": emotion},
                    priority=EventPriority.HIGH,
                ))
            if self.hui_window:
                self.hui_window.signals.update_status.emit("Idle")
            return text.strip() if text else None
        except sr.UnknownValueError:
            if self.hui_window:
                self.hui_window.signals.update_status.emit("Idle")
            return None
        except sr.RequestError:
            logger.warning("[VoiceInput] Google STT unavailable. No transcript.")
            if self.hui_window:
                self.hui_window.signals.update_status.emit("Idle")
            return None
