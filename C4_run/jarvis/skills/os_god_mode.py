"""
OS God Mode Skill: Allows JARVIS to resize, move, minimize, maximize, and interact with OS windows.
"""

from __future__ import annotations

import logging
import time

try:
    import pygetwindow as gw
except ImportError:
    gw = None

try:
    import pyautogui
except ImportError:
    pyautogui = None

from jarvis.nlp.schemas import Intent, IntentType
from jarvis.skills.base import Skill, SkillResult
from jarvis.context import ConversationContext

logger = logging.getLogger(__name__)

class OSGodModeSkill(Skill):
    """
    Grants C4 omnipotent control over foreground and background OS windows.
    Requires pygetwindow.
    """
    @property
    def name(self) -> str:
        return "os_god_mode"

    def can_handle(self, intent: Intent) -> bool:
        # Accept commands targeting window management
        actions = ["minimize_window", "maximize_window", "close_window", "hide_window", "move_window", "arrange_windows"]
        return intent.type == IntentType.COMMAND and intent.parsed_action in actions

    def execute(
        self,
        intent: Intent,
        context: ConversationContext,
        **kwargs,
    ) -> SkillResult:
        if not gw or not pyautogui:
            return SkillResult(text="Sir, I lack the pygetwindow or pyautogui libraries to interface with the OS at this level.", success=False)

        action = intent.parsed_action
        params = intent.params or {}
        window_query = params.get("window_name", "").lower()

        # If no specific window is mentioned, attempt to target the active window
        target_windows = []
        if window_query:
            target_windows = [w for w in gw.getAllWindows() if window_query in w.title.lower()]
        else:
            active = gw.getActiveWindow()
            if active:
                target_windows = [active]

        if not target_windows and action != "arrange_windows":
            return SkillResult(text=f"I couldn't find a window matching '{window_query}', sir.", success=False)

        win = target_windows[0] if target_windows else None

        try:
            if action == "minimize_window" or action == "hide_window":
                win.minimize()
                return SkillResult(text=f"Minimized {win.title}.", success=True)
            
            elif action == "maximize_window":
                win.maximize()
                return SkillResult(text=f"Maximized {win.title}.", success=True)

            elif action == "close_window":
                win.close()
                return SkillResult(text=f"Closed {win.title}.", success=True)

            elif action == "move_window":
                # Example: move to x, y
                x = int(params.get("x", 0))
                y = int(params.get("y", 0))
                win.moveTo(x, y)
                return SkillResult(text=f"Moved {win.title} to {x}, {y}.", success=True)

            elif action == "arrange_windows":
                # Naive arrange: minimize all, then restore/cascade a few
                for w in gw.getAllWindows():
                    if w.title and w.visible:
                        w.minimize()
                return SkillResult(text="I've cleared your workspace by minimizing all windows, sir.", success=True)

            return SkillResult(text="I am not sure how to perform that window operation.", success=False)
            
        except Exception as e:
            logger.error(f"[OSGodMode] Error managing window: {e}")
            return SkillResult(text="I encountered a system error while trying to manage the window.", success=False)
