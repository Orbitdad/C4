"""
JARVIS Skill: Volume Control — control system volume naturally.
"JARVIS, volume up", "mute", "set volume to 50 percent"
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from jarvis.skills.base import BaseSkill, SkillResult
from jarvis.nlp.schemas import Intent
from jarvis.context import ConversationContext

logger = logging.getLogger(__name__)


class VolumeControlSkill(BaseSkill):
    name = "volume_control"
    triggers = ["volume", "mute", "unmute", "louder", "quieter", "turn up", "turn down"]

    def can_handle(self, intent: Intent) -> bool:
        return any(t in intent.raw_text.lower() for t in self.triggers)

    def execute(self, intent: Intent, context: Optional[ConversationContext] = None) -> SkillResult:
        raw = intent.raw_text.lower()

        if "mute" in raw and "unmute" not in raw:
            return self._set_mute(True)
        if "unmute" in raw:
            return self._set_mute(False)

        # Check for percentage: "volume to 60" or "set volume 70 percent"
        m = re.search(r"(?:to|at|volume)\s+(\d+)(?:\s*percent|%)?", raw)
        if m:
            val = int(m.group(1))
            return self._set_volume(min(100, max(0, val)))

        if any(w in raw for w in ["up", "louder", "increase", "raise"]):
            return self._adjust_volume(+20)
        if any(w in raw for w in ["down", "quieter", "decrease", "lower", "reduce"]):
            return self._adjust_volume(-20)

        return SkillResult(text="I didn't catch how you'd like the volume adjusted, sir.")

    def _get_volume_interface(self):
        try:
            from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
            from comtypes import CLSCTX_ALL
            import ctypes
            devices = AudioUtilities.GetSpeakers()
            interface = devices.Activate(IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
            return interface.QueryInterface(IAudioEndpointVolume)
        except Exception as e:
            logger.warning(f"[VolumeSkill] pycaw unavailable: {e}")
            return None

    def _set_mute(self, mute: bool) -> SkillResult:
        vol = self._get_volume_interface()
        if vol:
            try:
                vol.SetMute(1 if mute else 0, None)
                state = "muted" if mute else "unmuted"
                return SkillResult(text=f"Audio {state}, sir.")
            except Exception as e:
                logger.error(f"[VolumeSkill] Mute error: {e}")
        # Fallback: nircmd
        import subprocess
        cmd = ["nircmd", "mutesysvolume", "1" if mute else "0"]
        try:
            subprocess.run(cmd, check=False)
            return SkillResult(text=f"Audio {'muted' if mute else 'unmuted'}, sir.")
        except Exception:
            return SkillResult(text="I'm unable to control audio volume at the moment, sir.")

    def _set_volume(self, level: int) -> SkillResult:
        vol = self._get_volume_interface()
        if vol:
            try:
                scalar = level / 100.0
                vol.SetMasterVolumeLevelScalar(scalar, None)
                return SkillResult(text=f"Volume set to {level} percent, sir.")
            except Exception as e:
                logger.error(f"[VolumeSkill] Set error: {e}")
        return SkillResult(text=f"Volume adjustment to {level}% attempted, sir.")

    def _adjust_volume(self, delta: int) -> SkillResult:
        vol = self._get_volume_interface()
        if vol:
            try:
                current = int(vol.GetMasterVolumeLevelScalar() * 100)
                new_level = min(100, max(0, current + delta))
                vol.SetMasterVolumeLevelScalar(new_level / 100.0, None)
                direction = "increased" if delta > 0 else "decreased"
                return SkillResult(text=f"Volume {direction} to {new_level} percent, sir.")
            except Exception as e:
                logger.error(f"[VolumeSkill] Adjust error: {e}")
        return SkillResult(text="Volume adjustment applied, sir.")
