"""
System Perception Module — monitors the OS environment in real time.

Upgraded for JARVIS: Immediate threshold-crossing events for CPU, RAM,
battery. Faster poll interval with tiered event priorities.
"""

from __future__ import annotations

import threading
import time
import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    import psutil
    _PSUTIL = True
except ImportError:
    psutil = None  # type: ignore
    _PSUTIL = False
    logger.warning("[SystemMonitor] psutil not installed.")

try:
    import ctypes
    _WIN_CTRL = True
except ImportError:
    _WIN_CTRL = False


def _get_active_window_title() -> Optional[str]:
    try:
        import ctypes
        hwnd = ctypes.windll.user32.GetForegroundWindow()
        length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
        if length == 0:
            return None
        buf = ctypes.create_unicode_buffer(length + 1)
        ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
        return buf.value
    except Exception:
        return None


def _get_active_process_name() -> Optional[str]:
    if not _PSUTIL:
        return None
    try:
        import ctypes
        hwnd = ctypes.windll.user32.GetForegroundWindow()
        pid = ctypes.c_ulong()
        ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        proc = psutil.Process(pid.value)
        return proc.name()
    except Exception:
        return None


def _get_spotify_track() -> Optional[str]:
    if not _PSUTIL or not _WIN_CTRL:
        return None
    try:
        import ctypes
        import psutil
        hwnds = []
        def cb(hwnd, ctx): 
            hwnds.append(hwnd)
            return True
            
        EnumWindowsProc = ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int))
        ctypes.windll.user32.EnumWindows(EnumWindowsProc(cb), 0)
        
        for hwnd in hwnds:
            if ctypes.windll.user32.IsWindowVisible(hwnd):
                pid = ctypes.c_ulong()
                ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
                try:
                    proc = psutil.Process(pid.value)
                    if proc.name().lower() == "spotify.exe":
                        length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
                        if length:
                            buf = ctypes.create_unicode_buffer(length + 1)
                            ctypes.windll.user32.GetWindowTextW(hwnd, buf, length + 1)
                            title = buf.value
                            if title and title not in ("Spotify Premium", "Spotify Free", "Spotify"):
                                return title
                except psutil.NoSuchProcess:
                    continue
    except Exception:
        pass
    return None

def _infer_activity(window_title: Optional[str], process: Optional[str]) -> str:
    if not window_title and not process:
        return "idle"
    combined = f"{(window_title or '')} {(process or '')}".lower()
    if any(k in combined for k in ["code", "pycharm", "vim", "nvim", "jupyter", "intellij"]):
        return "coding"
    if any(k in combined for k in ["chrome", "firefox", "edge", "opera", "safari", "browser"]):
        return "browsing"
    if any(k in combined for k in ["word", "excel", "notepad", "docs", "sheets", "notion"]):
        return "writing"
    if any(k in combined for k in ["discord", "slack", "teams", "zoom", "skype", "meet"]):
        return "communicating"
    if any(k in combined for k in ["vlc", "netflix", "youtube", "spotify", "music"]):
        return "consuming_media"
    if any(k in combined for k in ["steam", "epic", "game", "valorant", "minecraft"]):
        return "gaming"
    return "working"


class SystemMonitor:
    """
    Background monitor that polls OS every `poll_interval` seconds.
    Publishes IMMEDIATE threshold events for CPU/RAM/battery to trigger
    instant JARVIS voice alerts.
    """

    # Thresholds for immediate alert events
    CPU_CRITICAL = 85.0
    RAM_CRITICAL = 88.0
    BATTERY_LOW = 20.0
    DISK_FULL = 90.0

    def __init__(self, poll_interval: float = 3.0):
        self.poll_interval = poll_interval
        self.is_running = False
        self._thread: Optional[threading.Thread] = None

        self._prev_window: Optional[str] = None
        self._prev_network: Optional[bool] = None
        self._idle_seconds: int = 0

        # Alert state to prevent flooding
        self._alerted: dict = {
            "cpu": False, "ram": False, "battery": False, "disk": False
        }

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True, name="SystemMonitor")
        self._thread.start()
        logger.info("[SystemMonitor] Started (3s poll interval with immediate threshold alerts).")

    def stop(self):
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=3.0)

    def _run_loop(self):
        from jarvis.core.world_state import world
        from jarvis.core.event_bus import bus, SystemEvent, EventPriority

        while self.is_running:
            try:
                # ── Active Window ──────────────────────────────────────────────
                window = _get_active_window_title()
                process = _get_active_process_name()
                activity = _infer_activity(window, process)
                
                music_track = _get_spotify_track() or "None"

                world.update_environment(
                    active_window=window,
                    active_process=process,
                    inferred_activity=activity,
                    media_playing=music_track,
                )

                if window != self._prev_window:
                    bus.publish(SystemEvent(
                        name="context.window_changed",
                        data={"window": window, "process": process, "activity": activity},
                        priority=EventPriority.NORMAL,
                    ))
                    self._prev_window = window

                # ── System Health ──────────────────────────────────────────────
                if _PSUTIL:
                    cpu = psutil.cpu_percent(interval=None)
                    ram = psutil.virtual_memory().percent
                    disk_pct = psutil.disk_usage("/").percent
                    disk_free_gb = psutil.disk_usage("/").free / (1024 ** 3)

                    battery_info = psutil.sensors_battery()
                    battery = int(battery_info.percent) if battery_info else 100
                    plugged = battery_info.power_plugged if battery_info else True

                    world.update_health(
                        cpu_percent=cpu,
                        ram_percent=ram,
                        disk_free_gb=round(disk_free_gb, 1),
                        battery_percent=battery,
                        is_plugged_in=plugged,
                    )

                    # ── IMMEDIATE THRESHOLD EVENTS ─────────────────────────────
                    # CPU
                    if cpu >= self.CPU_CRITICAL:
                        if not self._alerted["cpu"]:
                            self._alerted["cpu"] = True
                            bus.publish(SystemEvent(
                                name="monitor.cpu_critical",
                                data={"value": cpu},
                                priority=EventPriority.HIGH,
                            ))
                    elif cpu < self.CPU_CRITICAL - 10:
                        self._alerted["cpu"] = False  # Reset when stable

                    # RAM
                    if ram >= self.RAM_CRITICAL:
                        if not self._alerted["ram"]:
                            self._alerted["ram"] = True
                            bus.publish(SystemEvent(
                                name="monitor.ram_critical",
                                data={"value": ram},
                                priority=EventPriority.HIGH,
                            ))
                    elif ram < self.RAM_CRITICAL - 10:
                        self._alerted["ram"] = False

                    # Battery
                    if not plugged and battery <= self.BATTERY_LOW:
                        if not self._alerted["battery"]:
                            self._alerted["battery"] = True
                            bus.publish(SystemEvent(
                                name="monitor.battery_critical",
                                data={"value": battery, "plugged": plugged},
                                priority=EventPriority.HIGH,
                            ))
                    elif plugged or battery > self.BATTERY_LOW + 5:
                        self._alerted["battery"] = False

                    # Disk
                    if disk_pct >= self.DISK_FULL:
                        if not self._alerted["disk"]:
                            self._alerted["disk"] = True
                            bus.publish(SystemEvent(
                                name="monitor.disk_full",
                                data={"value": disk_pct, "free_gb": disk_free_gb},
                                priority=EventPriority.NORMAL,
                            ))

                    # Running processes
                    try:
                        procs = sorted(
                            [p.info for p in psutil.process_iter(["name", "cpu_percent"])],
                            key=lambda x: x.get("cpu_percent", 0),
                            reverse=True
                        )[:10]
                        world.update_environment(running_processes=[p["name"] for p in procs])
                    except Exception:
                        pass

                # ── Network ────────────────────────────────────────────────────
                try:
                    net = psutil.net_if_stats() if _PSUTIL else {}
                    connected = any(s.isup for s in net.values()) if net else True
                    world.update_environment(network_connected=connected)

                    if connected != self._prev_network:
                        bus.publish(SystemEvent(
                            name="system.network_changed",
                            data={"connected": connected},
                            priority=EventPriority.NORMAL,
                        ))
                        self._prev_network = connected
                except Exception:
                    pass

                # ── Idle Detection ─────────────────────────────────────────────
                self._idle_seconds += int(self.poll_interval)
                # Fire at 3min, 5min for JARVIS check-ins
                if self._idle_seconds in (180, 300, 600):
                    bus.publish(SystemEvent(
                        name="context.user_idle",
                        data={"seconds": self._idle_seconds},
                        priority=EventPriority.LOW,
                    ))

            except Exception as e:
                logger.error(f"[SystemMonitor] Loop error: {e}")

            time.sleep(self.poll_interval)

    def reset_idle(self):
        """Call whenever user activity is detected."""
        self._idle_seconds = 0
