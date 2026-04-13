import time
import psutil
import threading
import ctypes
from typing import Dict, Any, Optional

class ContextEngine:
    """"""
    def __init__(self, vision_manager: Optional[Any] = None, memory_manager: Optional[Any] = None) -> None:
        self.vision_manager = vision_manager
        self.memory_manager = memory_manager
        
        self.active_window = "Unknown"
        self.window_history = []
        self.last_clipboard = ""
        self.system_status = {}
        self.idle_time_seconds = 0
        
        self.is_running = False
        self._thread: Optional[threading.Thread] = None

    def start(self):
        if self.is_running:
            return
        self.is_running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.is_running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def _get_active_window(self) -> str:
        try:
            hwnd = ctypes.windll.user32.GetForegroundWindow()
            length = ctypes.windll.user32.GetWindowTextLengthW(hwnd)
            buff = ctypes.create_unicode_buffer(length + 1)
            ctypes.windll.user32.GetWindowTextW(hwnd, buff, length + 1)
            return buff.value if buff.value else "Desktop"
        except Exception as e:
            return "Unknown"

    def _get_idle_time(self) -> int:
        """Returns the number of seconds the user has been idle (no mouse/keyboard input)."""
        try:
            class LASTINPUTINFO(ctypes.Structure):
                _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]
            lii = LASTINPUTINFO()
            lii.cbSize = ctypes.sizeof(LASTINPUTINFO)
            ctypes.windll.user32.GetLastInputInfo(ctypes.byref(lii))
            millis = ctypes.windll.kernel32.GetTickCount() - lii.dwTime
            return millis // 1000
        except Exception:
            return 0

    def _run_loop(self):
        """Continuously polls low-cost sensors to build the context snapshot."""
        while self.is_running:
            # 1. Update Active Window & Idle Time
            new_window = self._get_active_window()
            if new_window != self.active_window and new_window != "Unknown":
                self.active_window = new_window
                self.window_history.append(new_window)
                if len(self.window_history) > 5:
                    self.window_history.pop(0)
                
                from jarvis.core.event_bus import bus, SystemEvent
                bus.publish(SystemEvent("context.window_changed", {"window": new_window}))
                    
            new_idle = self._get_idle_time()
            if new_idle >= 300 and self.idle_time_seconds < 300: # Crossed 5 mins thresholds
                from jarvis.core.event_bus import bus, SystemEvent
                bus.publish(SystemEvent("context.user_idle", {"seconds": new_idle}))
            elif new_idle < 60 and self.idle_time_seconds >= 300: # Woke up from idle
                from jarvis.core.event_bus import bus, SystemEvent
                bus.publish(SystemEvent("context.user_active", {}))
                
            self.idle_time_seconds = new_idle
            
            # 2. Update System Load
            self.system_status = {
                "cpu": psutil.cpu_percent(),
                "ram": psutil.virtual_memory().percent,
                "disk": psutil.disk_usage('/').percent,
                "battery": psutil.sensors_battery().percent if hasattr(psutil, "sensors_battery") and psutil.sensors_battery() else "N/A"
            }
            
            # (Clipboard could be added here if we install pyperclip, skipping for now to prevent block or OS lock issues)
            
            if getattr(self, "governor", None):
                self.governor.check_and_throttle()
                
            time.sleep(2)


    def get_context_snapshot(self) -> str:
        """Returns a natural language summary of the current user context."""
        snapshot = []
        snapshot.append(f"Current System Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        status = "Active" if self.idle_time_seconds < 1200 else f"Idle for {self.idle_time_seconds} seconds"
        snapshot.append(f"User Status: {status}")
        snapshot.append(f"User is currently looking at Windows Application: '{self.active_window}'")
        if self.window_history:
             snapshot.append(f"Recent Applications: {', '.join(self.window_history[-3:])}")
        
        sys_warns = []
        if self.system_status.get("cpu", 0) > 85: sys_warns.append("CPU is critically high.")
        if self.system_status.get("ram", 0) > 90: sys_warns.append("RAM is almost full.")
        if sys_warns:
            snapshot.append("System Warnings: " + " ".join(sys_warns))
        else:
            snapshot.append("System Performance: Normal")
            
        if self.vision_manager:
            vision_state = self.vision_manager.get_status()
            snapshot.append(f"Vision Environment: {vision_state}")
            
        return "\n".join(snapshot)
