"""
jarvis/vision/gesture/adaptive_engine.py

Per-user calibration, landmark normalisation, and environment adaptation.
Persists a user profile to disk across sessions.
"""
from __future__ import annotations

import json
import logging
import math
from collections import deque
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

log = logging.getLogger(__name__)

_DEFAULT_PROFILE = {
    "hand_width":       0.20,    # normalised landmark span
    "hand_height":      0.25,
    "sensitivity":      1.0,
    "pinch_threshold":  0.05,
    "smoothing":        5.0,
    "calibrated":       False,
    "samples_collected": 0,
}

# Frames needed before we consider calibration complete
_CALIBRATION_FRAMES = 60


class AdaptiveEngine:
    """
    Learns the user's hand size + movement range and normalises landmark
    data to a canonical coordinate space, so gesture thresholds work
    consistently across different users and lighting conditions.

    :param profile_path: JSON file to persist user profile across sessions.
    """

    def __init__(
        self,
        profile_path: str | Path = "data/gesture_profile.json",
        config: Optional[Dict] = None,
    ) -> None:
        cfg = config or {}
        self._profile_path = Path(profile_path)
        self.user_profile: Dict = dict(_DEFAULT_PROFILE)
        self._load_profile()

        # Calibration sample buffer
        self._calib_buffer: Deque[Dict] = deque(maxlen=_CALIBRATION_FRAMES)
        self._enabled: bool = cfg.get("enabled", True)

    # ── Public API ────────────────────────────────────────────────────────────

    def calibrate(self, data_stream: Any) -> None:
        """
        Collect landmark snapshots and compute per-user stats.

        *data_stream* may be:
          - A single landmark snapshot (list/object of 21 points with .x/.y)
          - A list of such snapshots (batch mode)

        Once :const:`_CALIBRATION_FRAMES` samples are collected the profile
        is updated and persisted.
        """
        if not self._enabled:
            return

        # Determine if data_stream is a flat snapshot or a batch of snapshots.
        # A flat snapshot is a sequence whose first element has an .x attribute.
        # A batch is a sequence whose first element is itself a sequence.
        if data_stream is None:
            return

        if isinstance(data_stream, list) and data_stream and hasattr(data_stream[0], "x"):
            # Single flat landmark snapshot passed as a list
            snapshots = [data_stream]
        elif isinstance(data_stream, list):
            # Batch of snapshots (list of lists / list of landmark objects)
            snapshots = data_stream
        else:
            # _NormalisedLandmarks or other iterable — treat as single snapshot
            snapshots = [data_stream]

        for landmarks in snapshots:
            if landmarks is None:
                continue
            try:
                n = len(landmarks)
            except TypeError:
                continue
            if n < 21:
                continue
            stats = self._extract_hand_stats(landmarks)
            self._calib_buffer.append(stats)

        if len(self._calib_buffer) >= _CALIBRATION_FRAMES:
            self._compute_profile()
            self._save_profile()

    def adjust_sensitivity(self, lighting: float = 1.0, noise: float = 0.0) -> None:
        """
        Loosen gesture thresholds when conditions are poor.

        :param lighting: 0.0 (dark) → 1.0 (well-lit).  Below 0.4 = low-light.
        :param noise:    0.0 (clean) → 1.0 (very noisy). Above 0.5 = noisy.
        """
        base = self.user_profile.get("sensitivity", 1.0)

        factor = 1.0
        if lighting < 0.4:
            factor *= 1.3   # looser in dim conditions
        if noise > 0.5:
            factor *= 1.2   # looser with noisy input

        self.user_profile["sensitivity"] = round(base * factor, 3)
        # Relax pinch threshold proportionally
        base_pinch = _DEFAULT_PROFILE["pinch_threshold"]
        self.user_profile["pinch_threshold"] = round(base_pinch * factor, 4)
        log.debug("AdaptiveEngine: sensitivity=%.3f, pinch_threshold=%.4f",
                  self.user_profile["sensitivity"],
                  self.user_profile["pinch_threshold"])

    def update_profile(self, feedback: Dict) -> None:
        """
        Merge an external feedback dict into the user profile and persist.

        Example ``feedback``::

            {"smoothing": 7.0, "sensitivity": 1.1}
        """
        self.user_profile.update(feedback)
        self._save_profile()

    def normalize(self, landmarks: Any) -> Any:
        """
        Scale raw landmarks to a canonical space relative to the user's
        known hand size.  Returns landmarks unchanged if not yet calibrated.
        """
        if not self.user_profile.get("calibrated"):
            return landmarks

        hw = self.user_profile.get("hand_width",  0.20)
        hh = self.user_profile.get("hand_height", 0.25)

        if hw < 1e-6 or hh < 1e-6:
            return landmarks

        # Wrap the landmark list so callers get the same API back
        return _NormalisedLandmarks(landmarks, hw, hh)

    @property
    def pinch_threshold(self) -> float:
        return self.user_profile.get("pinch_threshold", _DEFAULT_PROFILE["pinch_threshold"])

    @property
    def smoothing(self) -> float:
        return self.user_profile.get("smoothing", _DEFAULT_PROFILE["smoothing"])

    # ── Private ───────────────────────────────────────────────────────────────

    def _extract_hand_stats(self, landmarks: Any) -> Dict:
        """Compute bounding-box width/height of the hand for this frame."""
        xs = [lm.x for lm in landmarks]
        ys = [lm.y for lm in landmarks]
        return {
            "width":  max(xs) - min(xs),
            "height": max(ys) - min(ys),
        }

    def _compute_profile(self) -> None:
        """Average collected samples and update the profile."""
        widths  = [s["width"]  for s in self._calib_buffer]
        heights = [s["height"] for s in self._calib_buffer]
        self.user_profile["hand_width"]        = round(sum(widths)  / len(widths),  4)
        self.user_profile["hand_height"]       = round(sum(heights) / len(heights), 4)
        self.user_profile["calibrated"]        = True
        self.user_profile["samples_collected"] += len(self._calib_buffer)
        log.info(
            "AdaptiveEngine: calibration complete — hand_width=%.3f hand_height=%.3f",
            self.user_profile["hand_width"], self.user_profile["hand_height"],
        )
        self._calib_buffer.clear()

    def _save_profile(self) -> None:
        try:
            self._profile_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self._profile_path, "w", encoding="utf-8") as fh:
                json.dump(self.user_profile, fh, indent=2)
        except Exception as exc:
            log.warning("AdaptiveEngine: could not save profile — %s", exc)

    def _load_profile(self) -> None:
        try:
            if self._profile_path.exists():
                with open(self._profile_path, encoding="utf-8") as fh:
                    saved = json.load(fh)
                # Merge saved over defaults so new keys always exist
                self.user_profile = {**_DEFAULT_PROFILE, **saved}
                log.info("AdaptiveEngine: loaded user profile from %s", self._profile_path)
        except Exception as exc:
            log.warning("AdaptiveEngine: could not load profile — %s", exc)


# ── Normalised landmark wrapper ───────────────────────────────────────────────

class _NormalisedLandmarks:
    """
    Thin wrapper that scales landmark coordinates by the user's hand size,
    preserving the MediaPipe ``.x / .y / .z`` attribute API.
    """

    class _LM:
        __slots__ = ("x", "y", "z")
        def __init__(self, x: float, y: float, z: float) -> None:
            self.x, self.y, self.z = x, y, z

    def __init__(self, raw: Any, hw: float, hh: float) -> None:
        self._lms = [
            self._LM(lm.x / hw, lm.y / hh, getattr(lm, "z", 0.0))
            for lm in raw
        ]

    def __len__(self) -> int:
        return len(self._lms)

    def __getitem__(self, idx: int):
        return self._lms[idx]

    def __iter__(self):
        return iter(self._lms)
