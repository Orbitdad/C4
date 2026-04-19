"""
C4 HUI — Cinematic Holographic User Interface

Major upgrades over original:
- JARVIS speaking waveform visualizer
- Real-time transcript panel (what JARVIS hears & says)
- Alert cascade system with dramatic visual effects
- Boot sequence animation (panel-by-panel reveal)
- Face recognition overlay
- Speaking/thinking/idle status with animated indicators
- Sound synthesis for startup sequence
"""

import sys
import os
import psutil
import math
import random
import time
import threading
import struct
from collections import deque

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget,
    QTextEdit, QHBoxLayout, QGridLayout, QFrame, QProgressBar,
    QSizePolicy, QSizeGrip, QLineEdit
)
from PyQt5.QtCore import (
    QTimer, Qt, pyqtSignal, QObject, QRectF, QPointF,
    QPropertyAnimation, QEasingCurve, pyqtProperty
)
from PyQt5.QtGui import (
    QFont, QColor, QPalette, QPainter, QPen, QBrush,
    QLinearGradient, QRadialGradient, QPolygonF
)


# ── Signal Bus ─────────────────────────────────────────────────────────────────

class HUISignals(QObject):
    update_status = pyqtSignal(str)
    log_message = pyqtSignal(str)
    update_vision = pyqtSignal(object)           # opencv frame
    update_landmarks = pyqtSignal(list)          # hand landmarks
    speaking_started = pyqtSignal(str)           # text being spoken
    speaking_stopped = pyqtSignal()
    thinking_started = pyqtSignal()
    thinking_stopped = pyqtSignal()
    face_detected = pyqtSignal(str, float)       # name, confidence
    alert_critical = pyqtSignal(str)             # critical message
    transcript_user = pyqtSignal(str)            # what user said
    transcript_jarvis = pyqtSignal(str)          # what jarvis said
    boot_step = pyqtSignal(int)                  # boot sequence step 0-7
    update_gesture_debug = pyqtSignal(dict)      # real-time continuous gesture info
    command_submitted = pyqtSignal(str)          # Custom UI text command
    # ── Command-pipeline signals ────────────────────────────────────────────
    execution_status = pyqtSignal(str)           # "Planning...", "Executing...", "Completed"
    thinking_plan = pyqtSignal(str, str, list)   # task_type, intent, steps
    thinking_step_status = pyqtSignal(int, str)  # step_idx, status ("running"/"done"/"error")


# ── JARVIS Waveform Widget ─────────────────────────────────────────────────────

class JARVISWaveform(QWidget):
    """
    Animated waveform that visualizes JARVIS speaking.
    Shows smooth sine-wave bars that animate when active.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(280, 60)
        self.setMaximumHeight(70)
        self._active = False
        self._bars = 28
        self._phases = [random.uniform(0, math.pi * 2) for _ in range(self._bars)]
        self._amplitudes = [random.uniform(0.3, 1.0) for _ in range(self._bars)]
        self._time = 0.0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(30)

    def set_active(self, active: bool):
        self._active = active
        if active:
            self._amplitudes = [random.uniform(0.5, 1.0) for _ in range(self._bars)]

    def _tick(self):
        if self._active:
            self._time += 0.15
            # Gradually randomize amplitudes for organic wave feel
            for i in range(self._bars):
                target = random.uniform(0.4, 1.0)
                self._amplitudes[i] += (target - self._amplitudes[i]) * 0.3
        else:
            self._time += 0.03
            # Dampen to flat
            for i in range(self._bars):
                self._amplitudes[i] *= 0.85
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w = self.width()
        h = self.height()
        bar_w = w / self._bars
        center_y = h / 2

        for i in range(self._bars):
            phase = self._phases[i] + self._time
            amp = self._amplitudes[i]
            bar_h = max(3, abs(math.sin(phase) * amp * (h * 0.42)))

            cx = i * bar_w + bar_w / 2
            hue_shift = (i / self._bars) * 40
            alpha = int(180 * amp) + 60 if self._active else int(50 * amp) + 20

            if self._active:
                color = QColor(0, int(200 + hue_shift), 255, alpha)
            else:
                color = QColor(0, 100, 150, alpha)

            pen = QPen(color, max(1.5, bar_w * 0.5))
            pen.setCapStyle(Qt.RoundCap)
            painter.setPen(pen)
            painter.drawLine(int(cx), int(center_y - bar_h), int(cx), int(center_y + bar_h))


# ── Transcript Panel ───────────────────────────────────────────────────────────

class TranscriptPanel(QFrame):
    """
    Clean display panel showing the conversation transcript.
    JARVIS speech is shown in cyan. User speech in amber.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Panel")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)

        header = QLabel("◤ INTERACTION LOG ◥")
        header.setFont(QFont("Consolas", 9, QFont.Bold))
        header.setStyleSheet("color: #00ffcc; letter-spacing: 2px;")
        layout.addWidget(header)

        self._text = QTextEdit()
        self._text.setReadOnly(True)
        self._text.setMaximumHeight(140)
        self._text.setStyleSheet("""
            QTextEdit {
                background: transparent;
                border: none;
                color: #00ffcc;
                font-family: 'Consolas';
                font-size: 11px;
            }
        """)
        layout.addWidget(self._text)

    def add_user(self, text: str):
        import datetime
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self._text.append(
            f"<span style='color:#888888'>[{ts}]</span> "
            f"<span style='color:#ffcc44; font-weight:bold'>YOU:</span> "
            f"<span style='color:#ffdd88'>{text}</span>"
        )
        self._text.verticalScrollBar().setValue(self._text.verticalScrollBar().maximum())

    def add_jarvis(self, text: str):
        import datetime
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self._text.append(
            f"<span style='color:#888888'>[{ts}]</span> "
            f"<span style='color:#00d2ff; font-weight:bold'>C4:</span> "
            f"<span style='color:#aaeeff'>{text}</span>"
        )
        self._text.verticalScrollBar().setValue(self._text.verticalScrollBar().maximum())


# ── Thinking Panel ─────────────────────────────────────────────────────────────

class ThinkingPanel(QFrame):
    """
    Displays the AI's real-time reasoning, intent, execution plan and step status.
    Shows:
      - Task type badge (GENERAL / CODING)
      - Intent label
      - Step-by-step plan with color-coded status indicators
      - Live execution status (Planning/Executing/Completed)
    """

    # Step status colors
    _STATUS_COLORS = {
        "pending": "#556677",
        "running": "#ffcc00",
        "done":    "#00ff88",
        "error":   "#ff4455",
    }
    _STATUS_ICONS = {
        "pending": "○",
        "running": "⟳",
        "done":    "✓",
        "error":   "✗",
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("Panel")
        self._steps: list = []
        self._step_statuses: list = []

        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(4)

        # Header row
        header_row = QHBoxLayout()
        header = QLabel("◤ C4 THINKING ◥")
        header.setFont(QFont("Consolas", 9, QFont.Bold))
        header.setStyleSheet("color: #ff00cc; letter-spacing: 2px;")
        header_row.addWidget(header)

        self._task_badge = QLabel("")
        self._task_badge.setFont(QFont("Consolas", 8, QFont.Bold))
        self._task_badge.setStyleSheet(
            "color: #000; background: #00d2ff; border-radius: 3px; padding: 1px 5px;"
        )
        self._task_badge.setFixedHeight(16)
        header_row.addStretch()
        header_row.addWidget(self._task_badge)
        layout.addLayout(header_row)

        # Status line
        self._status_lbl = QLabel("Awaiting command...")
        self._status_lbl.setFont(QFont("Consolas", 9))
        self._status_lbl.setStyleSheet("color: #88aacc; font-style: italic;")
        layout.addWidget(self._status_lbl)

        # Main content text
        self._text = QTextEdit()
        self._text.setReadOnly(True)
        self._text.setMaximumHeight(180)
        self._text.setStyleSheet("""
            QTextEdit {
                background: transparent;
                border: none;
                color: #ffaaee;
                font-family: 'Consolas';
                font-size: 11px;
            }
        """)
        layout.addWidget(self._text)

        # Animated dots timer for "Planning..." indicator
        self._dot_count = 0
        self._dot_timer = QTimer(self)
        self._dot_timer.timeout.connect(self._tick_dots)
        self._is_planning = False

    def set_status(self, status: str) -> None:
        """Update the live status line (Planning... / Executing... / Completed / Error)."""
        if status.lower().startswith("plan"):
            self._is_planning = True
            self._status_lbl.setStyleSheet("color: #ffcc00; font-style: italic;")
            if not self._dot_timer.isActive():
                self._dot_timer.start(400)
        else:
            self._is_planning = False
            self._dot_timer.stop()

        if "complet" in status.lower():
            self._status_lbl.setStyleSheet("color: #00ff88; font-weight: bold;")
            self._status_lbl.setText(f"✓  {status.upper()}")
        elif "error" in status.lower() or "fail" in status.lower():
            self._status_lbl.setStyleSheet("color: #ff4455; font-weight: bold;")
            self._status_lbl.setText(f"✗  {status.upper()}")
        elif "execut" in status.lower():
            self._status_lbl.setStyleSheet("color: #00d2ff; font-weight: bold;")
            self._status_lbl.setText(f"⚡ {status}")
        elif not self._is_planning:
            self._status_lbl.setStyleSheet("color: #88aacc; font-style: italic;")
            self._status_lbl.setText(status)

    def _tick_dots(self) -> None:
        self._dot_count = (self._dot_count + 1) % 4
        dots = "." * self._dot_count
        self._status_lbl.setText(f"⟳ PLANNING{dots}")

    def set_thinking(self, task_type: str, intent: str, steps: list) -> None:
        """Populate plan from EventBus c4.thinking.update events (legacy path)."""
        self._populate(task_type, intent, steps)

    def set_thinking_plan(self, task_type: str, intent: str, steps: list) -> None:
        """Populate plan from new thinking_plan signal (CommandHandler path)."""
        self._populate(task_type, intent, steps)

    def _populate(self, task_type: str, intent: str, steps: list) -> None:
        self._steps = steps
        self._step_statuses = ["pending"] * len(steps)

        # Badge
        tt = (task_type or "general").upper()
        badge_color = "#ff4455" if tt == "CODING" else "#00d2ff"
        self._task_badge.setText(tt)
        self._task_badge.setStyleSheet(
            f"color: #000; background: {badge_color}; border-radius: 3px; padding: 1px 5px;"
        )

        self._redraw(intent)

    def _redraw(self, intent: str = "") -> None:
        self._text.clear()
        if intent:
            self._text.append(
                f"<span style='color:#aaaaaa'>INTENT:</span> "
                f"<span style='color:#ffffff; font-weight:bold'>{intent}</span>"
            )
        if self._steps:
            self._text.append("<span style='color:#aaaaaa'>PLAN:</span>")
            for i, step in enumerate(self._steps):
                action = step.get("action", "") if isinstance(step, dict) else str(step)
                status = self._step_statuses[i] if i < len(self._step_statuses) else "pending"
                color  = self._STATUS_COLORS.get(status, "#556677")
                icon   = self._STATUS_ICONS.get(status, "○")
                params_str = ""
                if isinstance(step, dict):
                    p = step.get("params", {})
                    if p:
                        params_str = " " + " ".join(f"{k}={v}" for k, v in list(p.items())[:2])
                self._text.append(
                    f"  <span style='color:{color}'>{icon}</span> "
                    f"<span style='color:#ccddee'>[{i+1}] {action}{params_str[:40]}</span>"
                )
        self._text.verticalScrollBar().setValue(self._text.verticalScrollBar().maximum())

    def update_step_status(self, step_idx: int, status: str) -> None:
        """Update a single step's status and redraw."""
        if 0 <= step_idx < len(self._step_statuses):
            self._step_statuses[step_idx] = status
        # Re-render to reflect updated status
        self._text.clear()
        for i, step in enumerate(self._steps):
            action = step.get("action", "") if isinstance(step, dict) else str(step)
            st     = self._step_statuses[i] if i < len(self._step_statuses) else "pending"
            color  = self._STATUS_COLORS.get(st, "#556677")
            icon   = self._STATUS_ICONS.get(st, "○")
            self._text.append(
                f"  <span style='color:{color}'>{icon}</span> "
                f"<span style='color:#ccddee'>[{i+1}] {action}</span>"
            )
        self._text.verticalScrollBar().setValue(self._text.verticalScrollBar().maximum())


# ── Alert Banner ───────────────────────────────────────────────────────────────

class AlertBanner(QLabel):
    """
    Full-width alert banner that appears, holds, then fades.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAlignment(Qt.AlignCenter)
        self.setFont(QFont("Consolas", 13, QFont.Bold))
        self.setMaximumHeight(0)
        self._timer = QTimer(self)
        self._timer.setSingleShot(True)
        self._timer.timeout.connect(self._hide)
        self._active = False

    def show_alert(self, message: str, level: str = "warning"):
        colors = {
            "critical": ("rgba(200, 0, 0, 200)", "#ff4444"),
            "warning": ("rgba(180, 120, 0, 200)", "#ffcc00"),
            "info": ("rgba(0, 80, 150, 200)", "#00d2ff"),
        }
        bg, fg = colors.get(level, colors["info"])
        self.setStyleSheet(f"""
            background: {bg};
            color: {fg};
            border: 1px solid {fg};
            padding: 4px;
        """)
        self.setText(f"⚠ {message.upper()} ⚠" if level == "critical" else f"▶ {message}")
        self.setMaximumHeight(40)
        self._active = True
        self._timer.start(4000)

    def _hide(self):
        self.setMaximumHeight(0)
        self._active = False


# ── Status Orb ────────────────────────────────────────────────────────────────

class StatusOrb(QWidget):
    """Pulsing status indicator orb."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(14, 14)
        self._color = QColor(0, 255, 100)
        self._pulse = 0.0
        self._dir = 1
        self._mode = "idle"
        t = QTimer(self)
        t.timeout.connect(self._tick)
        t.start(40)

    def set_mode(self, mode: str):
        self._mode = mode
        modes = {
            "idle": QColor(0, 200, 100),
            "listening": QColor(0, 255, 80),
            "thinking": QColor(255, 200, 0),
            "speaking": QColor(0, 180, 255),
            "error": QColor(255, 60, 60),
            "boot": QColor(0, 120, 255),
        }
        self._color = modes.get(mode, QColor(0, 200, 100))

    def _tick(self):
        speed = 0.08 if self._mode in ("speaking", "thinking") else 0.04
        self._pulse += speed * self._dir
        if self._pulse > 1.0 or self._pulse < 0.0:
            self._dir *= -1
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        alpha = int(140 + 115 * self._pulse)
        glow = QColor(self._color.red(), self._color.green(), self._color.blue(), max(0, alpha - 100))
        painter.setBrush(QBrush(glow))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(0, 0, 14, 14)
        core = QColor(self._color.red(), self._color.green(), self._color.blue(), alpha)
        painter.setBrush(QBrush(core))
        painter.drawEllipse(3, 3, 8, 8)


# ── Circular Gauge ─────────────────────────────────────────────────────────────

class CircularGauge(QWidget):
    def __init__(self, label="METRIC", color="#00d2ff", parent=None):
        super().__init__(parent)
        self.value = 0
        self.label = label
        self.color = QColor(color)
        self.setMinimumSize(120, 120)

    def set_value(self, val):
        self.value = val
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        width = self.width()
        height = self.height()
        side = min(width, height)
        painter.translate(width / 2, height / 2)
        painter.scale(side / 200.0, side / 200.0)

        # Color adapts to value
        if self.value > 85:
            color = QColor(255, 60, 60)
        elif self.value > 70:
            color = QColor(255, 180, 0)
        else:
            color = self.color

        pen = QPen(color, 2)
        pen.setCapStyle(Qt.RoundCap)
        painter.setPen(pen)
        painter.drawEllipse(-90, -90, 180, 180)
        pen.setWidth(8)
        painter.setPen(pen)
        span_angle = -int(self.value * 3.6 * 16)
        painter.drawArc(-80, -80, 160, 160, 90 * 16, span_angle)
        painter.setPen(color)
        painter.setFont(QFont("Consolas", 20, QFont.Bold))
        painter.drawText(QRectF(-50, -30, 100, 40), Qt.AlignCenter, f"{int(self.value)}%")
        painter.setFont(QFont("Consolas", 10))
        painter.drawText(QRectF(-60, 10, 120, 20), Qt.AlignCenter, self.label)


# ── Active Tasks HUD ────────────────────────────────────────────────────────────

class ActiveTasksHUD(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(200, 200)
        self.tasks = []
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update)
        self.timer.start(50)
        self.angle = 0

    def refresh_tasks(self, tasks: list):
        self.tasks = tasks
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Border and background
        painter.setBrush(QBrush(QColor(0, 20, 40, 150)))
        painter.setPen(QPen(QColor(0, 210, 255, 80), 1))
        painter.drawRect(0, 0, self.width()-1, self.height()-1)
        
        # Draw tasks
        painter.setFont(QFont("Consolas", 10))
        y = 20
        painter.setPen(QColor(0, 210, 255))
        painter.drawText(10, y, "◈ ACTIVE EXECUTIONS")
        y += 20
        
        if not self.tasks:
            painter.setPen(QColor(100, 100, 100))
            painter.drawText(20, y, "Idle. Awaiting tasks...")
        else:
            self.angle += 5
            for task in self.tasks:
                painter.setPen(QColor(0, 255, 150))
                # Draw a spinning indicator
                painter.save()
                painter.translate(15, y - 4)
                painter.rotate(self.angle)
                painter.drawLine(-3, 0, 3, 0)
                painter.drawLine(0, -3, 0, 3)
                painter.restore()
                
                painter.drawText(30, y, str(task)[:30])
                y += 20


# ── Network Bar Chart ──────────────────────────────────────────────────────────

class BarChartHUD(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.data_tx = deque(maxlen=20)
        self.data_rx = deque(maxlen=20)
        for _ in range(20):
            self.data_tx.append(0)
            self.data_rx.append(0)
        self.setMinimumSize(200, 80)
        self.last_io = psutil.net_io_counters()

    def update_data(self):
        curr_io = psutil.net_io_counters()
        tx = curr_io.bytes_sent - self.last_io.bytes_sent
        rx = curr_io.bytes_recv - self.last_io.bytes_recv
        self.last_io = curr_io
        self.data_tx.append(tx)
        self.data_rx.append(rx)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w = self.width()
        h = self.height()
        bar_w = w / 20.0
        max_tx = max(max(self.data_tx), 1)
        max_rx = max(max(self.data_rx), 1)
        for i in range(20):
            bar_h_tx = (self.data_tx[i] / max_tx) * (h / 2.0)
            bar_h_rx = (self.data_rx[i] / max_rx) * (h / 2.0)
            painter.setBrush(QBrush(QColor(0, 210, 255, 150)))
            painter.setPen(QPen(QColor(0, 210, 255), 1))
            painter.drawRect(int(i * bar_w + 1), int((h / 2) - bar_h_tx), int(bar_w - 2), int(bar_h_tx))
            painter.setBrush(QBrush(QColor(0, 255, 150, 150)))
            painter.setPen(QPen(QColor(0, 255, 150), 1))
            painter.drawRect(int(i * bar_w + 1), int(h / 2), int(bar_w - 2), int(bar_h_rx))


# ── Memory HUD ────────────────────────────────────────────────────────────

class MemoryHUD(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(200, 200)
        self.memories = []

    def refresh_memory(self, memories: list):
        self.memories = memories
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Border
        painter.setBrush(QBrush(QColor(0, 30, 60, 150)))
        painter.setPen(QPen(QColor(0, 255, 255, 80), 1))
        painter.drawRect(0, 0, self.width()-1, self.height()-1)
        
        painter.setFont(QFont("Consolas", 10))
        y = 20
        painter.setPen(QColor(0, 255, 255))
        painter.drawText(10, y, "◈ CORTEX MEMORY HITS")
        y += 20
        
        if not self.memories:
            painter.setPen(QColor(100, 100, 100))
            painter.drawText(20, y, "No recent context.")
        else:
            for mem in self.memories:
                painter.setPen(QColor(255, 200, 0))
                # Text wrap logic placeholder
                painter.drawText(10, y, str(mem)[:35] + "...")
                y += 20


# ── Process Monitor ────────────────────────────────────────────────────────────

class ProcessMonitorHUD(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(200, 120)
        self.processes = []

    def update_processes(self):
        procs = []
        for p in psutil.process_iter(['name', 'cpu_percent', 'memory_percent']):
            try:
                if p.info['name'] and p.info['cpu_percent'] is not None:
                    procs.append(p.info)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        procs.sort(key=lambda x: x.get('cpu_percent', 0), reverse=True)
        self.processes = procs[:5]
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setFont(QFont("Consolas", 9))
        y = 20
        painter.setPen(QPen(QColor(0, 255, 255), 1))
        painter.drawText(10, y, "NAME           CPU%   MEM%")
        y += 20
        painter.setPen(QPen(QColor(0, 255, 200, 180), 1))
        for p in self.processes:
            name = (p['name'][:10] + '..') if len(p['name']) > 12 else p['name']
            cpu = f"{p['cpu_percent']:>5.1f}"
            mem = f"{p['memory_percent']:>5.1f}"
            line = f"{name.ljust(12)} {cpu}%  {mem}%"
            painter.drawText(10, y, line)
            bar_w = int((p['cpu_percent'] / 100.0) * (self.width() - 20))
            bar_w = min(bar_w, self.width() - 20)
            painter.fillRect(10, y + 4, bar_w, 2, QColor(0, 255, 255, 100))
            y += 25


# ── Arc Reactor HUD ────────────────────────────────────────────────────────────

class ArcReactorHUD(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(150, 150)
        self.angle_inner = 0
        self.angle_outer = 0
        self.pulse = 0
        self.pulse_dir = 1
        
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._animate)
        self.timer.start(30)

    def _animate(self):
        self.angle_inner = (self.angle_inner + 4) % 360
        self.angle_outer = (self.angle_outer - 2) % 360
        self.pulse += 0.05 * self.pulse_dir
        if self.pulse > 1.0 or self.pulse < 0.0:
            self.pulse_dir *= -1
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w = self.width()
        h = self.height()
        side = min(w, h)
        painter.translate(w / 2, h / 2)
        painter.scale(side / 250.0, side / 250.0)
        
        # Glow
        alpha = int(100 + 80 * self.pulse)
        painter.setBrush(QBrush(QColor(0, 220, 255, alpha)))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(-30, -30, 60, 60)
        
        # Inner Ring
        painter.setBrush(Qt.NoBrush)
        pen = QPen(QColor(0, 255, 255, 200), 4)
        painter.setPen(pen)
        painter.drawEllipse(-50, -50, 100, 100)
        
        # Inner Rotating dashes
        painter.save()
        painter.rotate(self.angle_inner)
        pen.setStyle(Qt.DashLine)
        pen.setDashPattern([2, 5])
        pen.setWidth(8)
        painter.setPen(pen)
        painter.drawEllipse(-70, -70, 140, 140)
        painter.restore()
        
        # Outer Rotating triangles
        painter.save()
        painter.rotate(self.angle_outer)
        painter.setPen(QPen(QColor(0, 200, 255, 150), 2))
        for i in range(12):
            painter.rotate(30)
            poly = QPolygonF([QPointF(-5, -95), QPointF(5, -95), QPointF(0, -110)])
            painter.setBrush(QBrush(QColor(0, 200, 255, 100)))
            painter.drawPolygon(poly)
            painter.drawLine(0, -75, 0, -90)
        painter.restore()


# ── Hardware Sensors ───────────────────────────────────────────────────────────

class HardwareSensorsHUD(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFont(QFont("Consolas", 10))
        self.setStyleSheet("color: #00ffcc;")
        self.setAlignment(Qt.AlignTop | Qt.AlignLeft)

    def update_sensors(self):
        import platform
        boot_time = psutil.boot_time()
        uptime = int(time.time() - boot_time)
        hrs = uptime // 3600
        mins = (uptime % 3600) // 60
        bat_str = "N/A"
        if hasattr(psutil, "sensors_battery"):
            bat = psutil.sensors_battery()
            if bat:
                bat_str = f"{bat.percent:.0f}% ({'⚡ CHARGING' if bat.power_plugged else '🔋 BATTERY'})"
        temp_str = "SYS_TEMP : OPTIMAL"
        if hasattr(psutil, "sensors_temperatures"):
            try:
                temps = psutil.sensors_temperatures()
                if temps:
                    all_temps = [t.current for v in temps.values() for t in v]
                    if all_temps:
                        temp_str = f"MAX_TEMP : {max(all_temps):.0f}°C"
            except Exception:
                pass
        lines = [
            f"OS_PLATFORM : {platform.system()} {platform.release()}",
            f"SYS_UPTIME  : {hrs}H {mins}M",
            f"BATTERY_PWR : {bat_str}",
            temp_str,
            f"C4 CORE     : TACTICAL AI ONLINE",
        ]
        self.setText("\n".join(lines))


# ── Main HUI Dashboard ─────────────────────────────────────────────────────────

class HUIDashboard(QMainWindow):

    def __init__(self):
        print("HUI: Initializing C4 HUI...")
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowMaximizeButtonHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

        self.signals = HUISignals()
        self.signals.update_status.connect(self._update_status, Qt.QueuedConnection)
        self.signals.log_message.connect(self._log_message, Qt.QueuedConnection)
        self.signals.update_vision.connect(self._update_vision, Qt.QueuedConnection)
        self.signals.update_landmarks.connect(self._update_landmarks, Qt.QueuedConnection)
        self.signals.speaking_started.connect(self._on_speaking_started, Qt.QueuedConnection)
        self.signals.speaking_stopped.connect(self._on_speaking_stopped, Qt.QueuedConnection)
        self.signals.thinking_started.connect(self._on_thinking_started, Qt.QueuedConnection)
        self.signals.thinking_stopped.connect(self._on_thinking_stopped, Qt.QueuedConnection)
        self.signals.face_detected.connect(self._on_face_detected, Qt.QueuedConnection)
        self.signals.alert_critical.connect(self._on_critical_alert, Qt.QueuedConnection)
        # ── New pipeline signals ──────────────────────────────────────────────
        self.signals.execution_status.connect(self._on_execution_status, Qt.QueuedConnection)
        self.signals.thinking_plan.connect(self._on_thinking_plan, Qt.QueuedConnection)
        self.signals.thinking_step_status.connect(self._on_thinking_step_status, Qt.QueuedConnection)
        self.signals.transcript_user.connect(self._on_transcript_user, Qt.QueuedConnection)
        self.signals.transcript_jarvis.connect(self._on_transcript_jarvis, Qt.QueuedConnection)
        self.signals.boot_step.connect(self._on_boot_step, Qt.QueuedConnection)
        self.signals.update_gesture_debug.connect(self._on_gesture_debug, Qt.QueuedConnection)

        self.landmarks = []
        self.ripples = []
        self.focused_panel = None
        self.dwell_count = 0
        self.setMouseTracking(True)
        self._face_overlay_text = ""
        self._face_overlay_timer = 0

        # Boot state
        self._boot_complete = False
        self._boot_step = 0
        self._panels_visible = []

        # Reactive props
        self.pulse_timer = QTimer()
        self.pulse_timer.timeout.connect(self._update_pulse)
        self.pulse_timer.start(40)
        self.pulse_val = 0
        self.pulse_dir = 1

        self.glitch_timer = QTimer()
        self.glitch_timer.setSingleShot(True)
        self.glitch_timer.timeout.connect(self._stop_glitch)
        self.glitch_active = False

        self._current_jarvis_text = ""

        self.init_ui()
        self.showMaximized()
        print("HUI: C4 Core Ready.")

    def init_ui(self):
        self.setWindowTitle("C4 INTELLIGENCE SYSTEM")

        central_widget = QWidget()
        central_widget.setMouseTracking(True)
        self.setCentralWidget(central_widget)
        central_widget.setStyleSheet("""
            QWidget {
                background-color: rgba(2, 6, 15, 252);
                color: #00d2ff;
                font-family: 'Consolas', 'Courier New';
            }
            QFrame#Panel {
                border: 1px solid rgba(0, 210, 255, 55);
                background-color: rgba(3, 14, 30, 200);
                border-radius: 8px;
            }
            QFrame#TopBar {
                background-color: rgba(0, 8, 20, 230);
                border-bottom: 1px solid rgba(0, 210, 255, 80);
            }
            QFrame#PanelActive {
                border: 2px solid rgba(0, 255, 255, 200);
                background-color: rgba(0, 40, 80, 60);
                border-radius: 8px;
            }
            QPushButton#WinCtrl {
                color: #00d2ff;
                background: transparent;
                border: 1px solid rgba(0,210,255,50);
                border-radius: 4px;
                font-family: 'Consolas';
                font-size: 13px;
                padding: 2px 8px;
            }
            QPushButton#WinCtrl:hover {
                background: rgba(0, 210, 255, 30);
                border-color: rgba(0, 255, 255, 180);
            }
            QPushButton#WinCtrlClose {
                color: #ff4455;
                background: transparent;
                border: 1px solid rgba(255, 68, 85, 60);
                border-radius: 4px;
                font-family: 'Consolas';
                font-size: 13px;
                padding: 2px 8px;
            }
            QPushButton#WinCtrlClose:hover {
                background: rgba(255, 68, 85, 40);
                border-color: #ff4455;
            }
            QLabel#Title {
                font-weight: bold;
                letter-spacing: 4px;
                color: #00ffff;
                font-size: 12px;
            }
            QLabel#PanelHeader {
                font-weight: bold;
                letter-spacing: 3px;
                color: rgba(0, 220, 255, 180);
                font-size: 10px;
                padding: 2px 0px 6px 0px;
                border-bottom: 1px solid rgba(0, 210, 255, 40);
            }
            QTextEdit {
                background: transparent;
                border: none;
                color: #00ffcc;
                font-size: 11px;
            }
            QScrollBar:vertical {
                background: rgba(0, 20, 40, 100);
                width: 6px;
                border-radius: 3px;
            }
            QScrollBar::handle:vertical {
                background: rgba(0, 210, 255, 120);
                border-radius: 3px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
        """)

        root_layout = QVBoxLayout(central_widget)
        root_layout.setSpacing(0)
        root_layout.setContentsMargins(0, 0, 0, 0)

        # ── CUSTOM TOP BAR ─────────────────────────────────────────────────────
        from PyQt5.QtWidgets import QPushButton
        topbar = QFrame()
        topbar.setObjectName("TopBar")
        topbar.setFixedHeight(52)
        topbar_layout = QHBoxLayout(topbar)
        topbar_layout.setContentsMargins(20, 0, 12, 0)
        topbar_layout.setSpacing(14)

        self.status_orb = StatusOrb()
        topbar_layout.addWidget(self.status_orb)

        title_lbl = QLabel("C4  ◈  INTELLIGENCE  SYSTEM  ◈  GLOBAL  COMMAND")
        title_lbl.setObjectName("Title")
        title_lbl.setFont(QFont("Consolas", 15, QFont.Bold))
        topbar_layout.addWidget(title_lbl)
        topbar_layout.addStretch()

        self.waveform = JARVISWaveform()
        self.waveform.setMaximumWidth(320)
        topbar_layout.addWidget(self.waveform)
        topbar_layout.addStretch()

        self.clock_label = QLabel("00:00:00")
        self.clock_label.setFont(QFont("Consolas", 18, QFont.Bold))
        self.clock_label.setStyleSheet("color: #00ffcc;")
        topbar_layout.addWidget(self.clock_label)

        self.ai_status_label = QLabel("  INITIALIZING")
        self.ai_status_label.setFont(QFont("Consolas", 9))
        self.ai_status_label.setStyleSheet("color: #4499aa; letter-spacing: 2px;")
        topbar_layout.addWidget(self.ai_status_label)

        topbar_layout.addSpacing(18)
        
        btn_min = QPushButton("─")
        btn_min.setObjectName("WinCtrl")
        btn_min.setFixedSize(30, 24)
        btn_min.clicked.connect(self.showMinimized)
        topbar_layout.addWidget(btn_min)

        btn_max = QPushButton("□")
        btn_max.setObjectName("WinCtrl")
        btn_max.setFixedSize(30, 24)
        btn_max.clicked.connect(self._toggle_maximize)
        topbar_layout.addWidget(btn_max)

        btn_close = QPushButton("✕")
        btn_close.setObjectName("WinCtrlClose")
        btn_close.setFixedSize(30, 24)
        btn_close.clicked.connect(self.close)
        topbar_layout.addWidget(btn_close)

        root_layout.addWidget(topbar)

        # ── TOP ALERT BANNER ──────────────────────────────────────────────────
        self.alert_banner = AlertBanner()
        root_layout.addWidget(self.alert_banner)

        # ── MAIN CONTENT AREA ─────────────────────────────────────────────────
        content_widget = QWidget()
        content_widget.setMouseTracking(True)
        root_layout.addWidget(content_widget, 1)

        layout = QGridLayout(content_widget)
        layout.setSpacing(14)
        layout.setContentsMargins(18, 12, 18, 14)
        self.panels = {}
        self._drag_topbar = topbar  # used for dragging detect

        # Column stretch: vision | center-left | center-right | system
        layout.setColumnStretch(0, 28)   # vision feed — wide
        layout.setColumnStretch(1, 22)   # globe/metrics
        layout.setColumnStretch(2, 22)   # radar/network
        layout.setColumnStretch(3, 28)   # system diagnostics — wide

        # Row stretch
        layout.setRowStretch(0, 42)   # main panels
        layout.setRowStretch(1, 28)   # gauges row  
        layout.setRowStretch(2, 30)   # interaction log

        # ── VISION PANEL ──────────────────────────────────────────────────────
        vision_panel = QFrame()
        vision_panel.setObjectName("Panel")
        self.panels["vision"] = vision_panel
        vision_layout = QVBoxLayout(vision_panel)
        vision_layout.setContentsMargins(12, 10, 12, 10)
        vision_layout.setSpacing(6)
        lbl_v = QLabel("◈  VISION MATRIX")
        lbl_v.setObjectName("PanelHeader")
        vision_layout.addWidget(lbl_v)
        self.vision_label = QLabel("INITIALIZING SENSORS...")
        self.vision_label.setAlignment(Qt.AlignCenter)
        self.vision_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.vision_label.setStyleSheet(
            "background: rgba(0,4,10,220); "
            "border: 1px solid rgba(0,210,255,35); "
            "border-radius: 6px; color: rgba(0,210,255,120); font-size:11px;"
        )
        vision_layout.addWidget(self.vision_label, 1)
        self._face_lbl = QLabel("")
        self._face_lbl.setFont(QFont("Consolas", 10))
        self._face_lbl.setStyleSheet("color: #00ff88;")
        self._face_lbl.setAlignment(Qt.AlignCenter)
        vision_layout.addWidget(self._face_lbl)
        self._gesture_lbl = QLabel("GESTURE: IDLE  |  MODE: SINGLE_HAND")
        self._gesture_lbl.setFont(QFont("Consolas", 9))
        self._gesture_lbl.setStyleSheet("color: #00d2ff;")
        self._gesture_lbl.setAlignment(Qt.AlignCenter)
        vision_layout.addWidget(self._gesture_lbl)
        self._metrics_lbl = QLabel("Z: --  |  ROLL: --  |  CONF: --")
        self._metrics_lbl.setFont(QFont("Consolas", 9))
        self._metrics_lbl.setStyleSheet("color: #0088ff;")
        self._metrics_lbl.setAlignment(Qt.AlignCenter)
        vision_layout.addWidget(self._metrics_lbl)
        layout.addWidget(vision_panel, 0, 0, 2, 1)

        # ── GLOBE + RADAR ──────────────────────────────────────────────────────
        globe_panel = QFrame()
        globe_panel.setObjectName("Panel")
        self.panels["globe"] = globe_panel
        globe_layout = QHBoxLayout(globe_panel)
        globe_layout.setContentsMargins(12, 10, 12, 10)
        globe_layout.setSpacing(10)
        globe_vbox = QVBoxLayout()
        lbl_g = QLabel("◈  GLOBAL ASSETS")
        lbl_g.setObjectName("PanelHeader")
        globe_vbox.addWidget(lbl_g)
        self.globe = MemoryHUD()
        self.globe.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        globe_vbox.addWidget(self.globe, 1)
        globe_layout.addLayout(globe_vbox)
        radar_vbox = QVBoxLayout()
        lbl_r = QLabel("◈  ACTIVE TASKS")
        lbl_r.setObjectName("PanelHeader")
        radar_vbox.addWidget(lbl_r)
        self.radar = ActiveTasksHUD()
        self.radar.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        radar_vbox.addWidget(self.radar, 1)
        globe_layout.addLayout(radar_vbox)
        layout.addWidget(globe_panel, 0, 1, 1, 2)

        # ── SYSTEM DIAGNOSTICS ────────────────────────────────────────────────
        sys_panel = QFrame()
        sys_panel.setObjectName("Panel")
        self.panels["system"] = sys_panel
        sys_layout = QVBoxLayout(sys_panel)
        sys_layout.setContentsMargins(12, 10, 12, 10)
        sys_layout.setSpacing(8)
        lbl_s = QLabel("◈  SYSTEM DIAGNOSTICS")
        lbl_s.setObjectName("PanelHeader")
        sys_layout.addWidget(lbl_s)
        self.hardware_info = HardwareSensorsHUD()
        sys_layout.addWidget(self.hardware_info)
        sep = QFrame()
        sep.setFixedHeight(1)
        sep.setStyleSheet("background: rgba(0,210,255,30); margin: 4px 0px;")
        sys_layout.addWidget(sep)
        lbl_p = QLabel("◈  ACTIVE PROCESSES")
        lbl_p.setObjectName("PanelHeader")
        sys_layout.addWidget(lbl_p)
        self.process_monitor = ProcessMonitorHUD()
        self.process_monitor.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        sys_layout.addWidget(self.process_monitor, 1)
        layout.addWidget(sys_panel, 0, 3, 2, 1)

        # ── METRICS GAUGES ────────────────────────────────────────────────────
        metrics_panel = QFrame()
        metrics_panel.setObjectName("Panel")
        self.panels["metrics"] = metrics_panel
        metrics_layout = QVBoxLayout(metrics_panel)
        metrics_layout.setContentsMargins(12, 10, 12, 10)
        metrics_layout.setSpacing(8)
        lbl_m = QLabel("◈  CORE VITALS")
        lbl_m.setObjectName("PanelHeader")
        metrics_layout.addWidget(lbl_m)
        gauges_row = QHBoxLayout()
        self.cpu_gauge = CircularGauge("CPU CORE")
        self.ram_gauge = CircularGauge("SYS RAM")
        self.disk_gauge = CircularGauge("SSD VOL", color="#ff00cc")
        self.arc_reactor = ArcReactorHUD()
        self.arc_reactor.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        gauges_row.addWidget(self.cpu_gauge)
        gauges_row.addWidget(self.arc_reactor)
        gauges_row.addWidget(self.ram_gauge)
        gauges_row.addWidget(self.disk_gauge)
        metrics_layout.addLayout(gauges_row, 1)
        layout.addWidget(metrics_panel, 1, 1, 1, 1)

        # ── NETWORK CHART ─────────────────────────────────────────────────────
        chart_panel = QFrame()
        chart_panel.setObjectName("Panel")
        self.panels["network"] = chart_panel
        chart_layout = QVBoxLayout(chart_panel)
        chart_layout.setContentsMargins(12, 10, 12, 10)
        chart_layout.setSpacing(8)
        lbl_n = QLabel("◈  NETWORK BANDWIDTH")
        lbl_n.setObjectName("PanelHeader")
        chart_layout.addWidget(lbl_n)
        self.net_chart = BarChartHUD()
        self.net_chart.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        chart_layout.addWidget(self.net_chart, 1)
        layout.addWidget(chart_panel, 1, 2, 1, 1)

        # ── TRANSCRIPT + COMMAND LOG ─────────────────────────────────────────
        log_panel = QFrame()
        log_panel.setObjectName("Panel")
        self.panels["log"] = log_panel
        log_layout = QHBoxLayout(log_panel)
        log_layout.setContentsMargins(14, 10, 14, 10)
        log_layout.setSpacing(14)

        # Left: Interaction transcript
        transcript_col = QVBoxLayout()
        transcript_col.setSpacing(6)
        lbl_t = QLabel("◈  INTERACTION LOG")
        lbl_t.setObjectName("PanelHeader")
        transcript_col.addWidget(lbl_t)
        self.transcript_panel = TranscriptPanel()
        self.transcript_panel.setMaximumHeight(9999)  # allow expand
        self.transcript_panel.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.transcript_panel._text.setMaximumHeight(9999)
        transcript_col.addWidget(self.transcript_panel, 1)

        # Thinking Panel
        self.thinking_panel = ThinkingPanel()
        transcript_col.addWidget(self.thinking_panel, 1)

        # Command Input Field
        self.command_input = QLineEdit()
        self.command_input.setPlaceholderText("Enter command...")
        self.command_input.setStyleSheet("background: rgba(0, 20, 40, 200); color: #00ffcc; border: 1px solid #00ffcc; font-family: 'Consolas'; font-size: 12px; padding: 4px;")
        self.command_input.returnPressed.connect(self._on_command_entered)
        transcript_col.addWidget(self.command_input)

        log_layout.addLayout(transcript_col, 2)

        # Vertical separator
        vsep = QFrame()
        vsep.setFixedWidth(1)
        vsep.setStyleSheet("background: rgba(0,210,255,30);")
        log_layout.addWidget(vsep)

        # Right: Explain label + raw log
        log_col = QVBoxLayout()
        log_col.setSpacing(6)
        lbl_l = QLabel("◈  SYSTEM LOG")
        lbl_l.setObjectName("PanelHeader")
        log_col.addWidget(lbl_l)
        self.explain_label = QLabel("C4 :: Awaiting instruction...")
        self.explain_label.setStyleSheet(
            "color: #ffcc00; font-size: 11px; font-style: italic; "
            "padding: 4px 8px; background: rgba(40,30,0,60); "
            "border: 1px solid rgba(255,200,0,30); border-radius: 4px;"
        )
        self.explain_label.setWordWrap(True)
        log_col.addWidget(self.explain_label)
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        log_col.addWidget(self.log_text, 1)
        log_layout.addLayout(log_col, 3)

        layout.addWidget(log_panel, 2, 0, 1, 4)
        
        # Resize grip
        layout.addWidget(QSizeGrip(central_widget), 2, 3, Qt.AlignBottom | Qt.AlignRight)

        # ── TIMERS ────────────────────────────────────────────────────────────
        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(self.update_stats)
        self.stats_timer.start(1000)

        # Kick off boot sequence
        QTimer.singleShot(200, self._start_boot_sequence)
    def _on_command_entered(self):
        cmd = self.command_input.text().strip()
        if cmd:
            self.command_input.clear()
            self.signals.transcript_user.emit(cmd)
            self.signals.command_submitted.emit(cmd)

    def set_focused_panel(self, name: str):
        """Update visual focus state of panels."""
        if name not in self.panels:
            return

        self.focused_panel = name
        for p_name, panel in self.panels.items():
            if p_name == name:
                panel.setObjectName("PanelActive")
                panel.setStyleSheet("")
            else:
                panel.setObjectName("Panel")
                panel.setStyleSheet("")

        # Force style re-polish
        self.style().unpolish(self)
        self.style().polish(self)
        self._log_message(f"HUI: Focus shifted to {name.upper()}")

    def set_command_active(self, panel_id: str, active: bool) -> None:
        """
        Dim all panels except the active command panel.
        When active=False, restore all panels to full opacity.
        """
        from PyQt5.QtWidgets import QGraphicsOpacityEffect
        for name, panel in self.panels.items():
            effect = QGraphicsOpacityEffect(panel)
            if active:
                effect.setOpacity(1.0 if name == panel_id else 0.30)
            else:
                effect.setOpacity(1.0)
            panel.setGraphicsEffect(effect)

    def _toggle_maximize(self):
        if self.isMaximized():
            self.showNormal()
        else:
            self.showMaximized()

    # ── Boot Sequence ─────────────────────────────────────────────────────────

    def _start_boot_sequence(self):
        """Animate panels revealing one-by-one on startup."""
        self._boot_step = 0
        # Hide all panels initially
        for panel in self.panels.values():
            panel.setVisible(False)

        boot_order = ["system", "vision", "globe", "metrics", "network", "log"]
        delay = 0
        for i, name in enumerate(boot_order):
            delay += 220
            step = i
            QTimer.singleShot(delay, lambda n=name, s=step: self._reveal_panel(n, s))
        QTimer.singleShot(delay + 500, self._boot_complete_event)

    def _reveal_panel(self, name: str, step: int):
        if name in self.panels:
            self.panels[name].setVisible(True)
            self.trigger_glitch()
            self._log_message(f"SYSTEM: Module '{name.upper()}' online.")

    def _boot_complete_event(self):
        self._boot_complete = True
        self.alert_banner.show_alert("C4 INTELLIGENCE SYSTEM FULLY OPERATIONAL", "info")
        self._update_status("ONLINE")

    def _on_gesture_debug(self, data: dict):
        state = data.get("state", "IDLE")
        mode = data.get("mode", "SINGLE_HAND_MODE")
        conf = data.get("confidence", 0.0)
        metrics = data.get("metrics", {})
        z = metrics.get("z", 0.0)
        roll = metrics.get("roll", 0.0)
        
        # Format the text
        mode_short = "DUAL" if "DUAL" in mode else "SINGLE"
        self._gesture_lbl.setText(f"GESTURE: {state.ljust(8)} | MODE: {mode_short}")
        
        # Color coding confidence
        conf_color = "#ff4444" if conf < 0.6 else ("#ffcc00" if conf < 0.8 else "#00ff88")
        self._metrics_lbl.setStyleSheet(f"color: {conf_color};")
        
        z_fmt = f"{z:.3f}" if z is not None else "--"
        roll_fmt = f"{roll:.2f}" if roll is not None else "--"
        self._metrics_lbl.setText(f"Z: {z_fmt} | ROLL: {roll_fmt} | CONF: {conf:.2f}")

    # ── Glitch & Draw Effects ───────────────────────────────────────────────────────

    def _update_status(self, status: str):
        self.ai_status_label.setText(f"STATUS: {status.upper()}")
        mode_map = {
            "LISTENING": "listening",
            "THINKING": "thinking",
            "SPEAKING": "speaking",
            "ONLINE": "idle",
            "IDLE": "idle",
            "ERROR": "error",
            "BOOT": "boot",
        }
        for key, mode in mode_map.items():
            if key in status.upper():
                self.status_orb.set_mode(mode)
                break

    def _log_message(self, message: str):
        import datetime
        timestamp = datetime.datetime.now().strftime("%H:%M:%S")
        self.log_text.append(
            f"<span style='color: #336688;'>[{timestamp}]</span> {message}"
        )
        self.log_text.verticalScrollBar().setValue(self.log_text.verticalScrollBar().maximum())
        if any(x in message.upper() for x in ["ERROR", "CRITICAL", "FAILED"]):
            self.trigger_glitch()
            self.alert_banner.show_alert(message[:80], "critical")

    def _on_speaking_started(self, text: str):
        self.waveform.set_active(True)
        self.status_orb.set_mode("speaking")
        self.explain_label.setText(f"C4: {text[:120]}")
        self.ai_status_label.setStyleSheet("color: #00d2ff; font-size: 10px;")

    def _on_speaking_stopped(self):
        self.waveform.set_active(False)
        self.status_orb.set_mode("idle")
        self.ai_status_label.setStyleSheet("color: #4499aa; font-size: 10px;")

    def _on_thinking_started(self):
        self.status_orb.set_mode("thinking")
        self.explain_label.setText("C4: Processing...")
        self.ai_status_label.setText("STATUS: THINKING...")
        self.ai_status_label.setStyleSheet("color: #ffcc00; font-size: 10px;")
        self.thinking_panel.set_status("Planning...")
        self.set_command_active("log", True)

    def _on_thinking_stopped(self):
        self.status_orb.set_mode("idle")
        self.ai_status_label.setStyleSheet("color: #4499aa; font-size: 10px;")
        self.ai_status_label.setText("STATUS: ONLINE")
        self.set_command_active("log", False)

    def _on_execution_status(self, status: str) -> None:
        """Handle live execution status updates from CommandHandler."""
        self.thinking_panel.set_status(status)
        if "plan" in status.lower():
            self.ai_status_label.setText("STATUS: PLANNING")
            self.ai_status_label.setStyleSheet("color: #ffcc00; font-size: 10px;")
            self.status_orb.set_mode("thinking")
        elif "execut" in status.lower():
            self.ai_status_label.setText("STATUS: EXECUTING")
            self.ai_status_label.setStyleSheet("color: #00d2ff; font-size: 10px;")
            self.status_orb.set_mode("speaking")
        elif "complet" in status.lower():
            self.ai_status_label.setText("STATUS: ONLINE")
            self.ai_status_label.setStyleSheet("color: #00ff88; font-size: 10px;")
            self.status_orb.set_mode("idle")
        elif "error" in status.lower():
            self.ai_status_label.setText("STATUS: ERROR")
            self.ai_status_label.setStyleSheet("color: #ff4455; font-size: 10px;")
            self.status_orb.set_mode("error")

    def _on_thinking_plan(self, task_type: str, intent: str, steps: list) -> None:
        """Populate ThinkingPanel from CommandHandler thinking_plan signal."""
        self.thinking_panel.set_thinking_plan(task_type, intent, steps)
        self.explain_label.setText(f"C4: {intent}")

    def _on_thinking_step_status(self, step_idx: int, status: str) -> None:
        """Update a single step's status indicator in the ThinkingPanel."""
        self.thinking_panel.update_step_status(step_idx, status)

    def _on_face_detected(self, name: str, confidence: float):
        self._face_lbl.setText(f"◉ IDENTIFIED: {name.upper()}  [{confidence:.0%}]")
        self._face_overlay_text = name
        self._face_overlay_timer = 5  # seconds
        self.alert_banner.show_alert(f"User identified: {name}", "info")

    def _on_critical_alert(self, message: str):
        self.alert_banner.show_alert(message, "critical")
        self.trigger_glitch()

    def _on_transcript_user(self, text: str):
        self.transcript_panel.add_user(text)

    def _on_transcript_jarvis(self, text: str):
        self.transcript_panel.add_jarvis(text)
        self.signals.speaking_started.emit(text)

    def _on_boot_step(self, step: int):
        pass  # Handled by _start_boot_sequence

    # ── Reactive Pulse Glow ───────────────────────────────────────────────────

    def _update_pulse(self):
        self.pulse_val += 0.08 * self.pulse_dir
        if self.pulse_val > 1.0 or self.pulse_val < 0.0:
            self.pulse_dir *= -1
        self.update()

    def trigger_glitch(self):
        self.glitch_active = True
        self.glitch_timer.start(200)

    def _stop_glitch(self):
        self.glitch_active = False
        self.update()

    # ── Mouse / Gesture Interaction ───────────────────────────────────────────

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            # Only initiate drag when clicking directly on topbar
            topbar = getattr(self, '_drag_topbar', None)
            if topbar and topbar.geometry().contains(event.pos()):
                self._drag_pos = event.globalPos()
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            if hasattr(self, '_drag_pos'):
                del self._drag_pos
            event.accept()

    def mouseMoveEvent(self, event):
        # 1) Window dragging logic
        if event.buttons() == Qt.LeftButton and hasattr(self, '_drag_pos'):
            curr_pos = event.globalPos()
            diff = curr_pos - self._drag_pos
            self.move(self.pos() + diff)
            self._drag_pos = curr_pos
            event.accept()
            return

        super().mouseMoveEvent(event)
        
        # 2) Landmarks & Ripples update
        lx = event.x() / self.width()
        ly = event.y() / self.height()
        self._update_landmarks([(lx, ly)])

        # 3) Parallax depth effect based on mouse pos
        cx, cy = self.width() / 2, self.height() / 2
        dx, dy = event.x() - cx, event.y() - cy
        
        from PyQt5.QtWidgets import QGraphicsDropShadowEffect
        for name, panel in self.panels.items():
            if not panel: continue
            effect = panel.graphicsEffect()
            if not isinstance(effect, QGraphicsDropShadowEffect):
                effect = QGraphicsDropShadowEffect()
                effect.setBlurRadius(15)
                effect.setColor(QColor(0, 210, 255, 60))
                panel.setGraphicsEffect(effect)
            
            depth_multiplier = -0.05 if name in ['globe', 'vision'] else -0.02
            effect.setOffset(dx * depth_multiplier, dy * depth_multiplier)

    def _update_landmarks(self, lms):
        self.landmarks = lms
        new_focus = None

        for x, y in lms:
            self.ripples.insert(0, [x, y, 1.0])
            px_abs = int(x * self.width())
            py_abs = int(y * self.height())
            for name, panel in self.panels.items():
                if panel.isVisible() and panel.geometry().contains(px_abs, py_abs):
                    new_focus = name
                    break

        if new_focus != self.focused_panel:
            if new_focus:
                self.set_focused_panel(new_focus)
            else:
                self.focused_panel = None
                for panel in self.panels.values():
                    panel.setObjectName("Panel")
            self.dwell_count = 0

        if new_focus and new_focus == self.focused_panel:
            self.dwell_count += 1
            if self.dwell_count == 25:
                self._log_message(f"HUI: GESTURE COMMAND → {new_focus.upper()}")
                self.trigger_glitch()
                if new_focus == "log":
                    self.log_text.clear()
                elif new_focus == "globe":
                    self._log_message("MEMORY: Cortex context refreshed.")
                elif new_focus == "system":
                    self.process_monitor.update_processes()
        else:
            self.dwell_count = 0

        for rip in self.ripples:
            rip[2] -= 0.08
        self.ripples = [r for r in self.ripples if r[2] > 0]
        if len(self.ripples) > 30:
            self.ripples = self.ripples[:30]
        self.update()

    # ── Paint ─────────────────────────────────────────────────────────────────

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        if self.glitch_active:
            painter.translate(random.randint(-3, 3), random.randint(-2, 2))

        # Ripple trails
        for rx, ry, op in self.ripples:
            px = int(rx * self.width())
            py = int(ry * self.height())
            size = int(20 * (1 - op)) + 10
            painter.setPen(QPen(QColor(0, 255, 255, int(130 * op)), 1))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(px - size // 2, py - size // 2, size, size)

        # Status pulse bar
        status_text = self.ai_status_label.text().upper()
        if "THINKING" in status_text:
            pulse_color = QColor(255, 215, 0, int(70 * self.pulse_val))
        elif "LISTENING" in status_text:
            pulse_color = QColor(0, 255, 100, int(80 * self.pulse_val))
        elif "SPEAKING" in status_text:
            pulse_color = QColor(0, 180, 255, int(90 * self.pulse_val))
        else:
            pulse_color = QColor(0, 140, 200, int(40 * self.pulse_val))

        painter.setBrush(QBrush(pulse_color))
        painter.setPen(Qt.NoPen)
        painter.drawRect(0, 0, self.width(), 3)
        painter.drawRect(0, self.height() - 3, self.width(), 3)

        if not self.landmarks:
            return

        for lx, ly in self.landmarks:
            px = int(lx * self.width())
            py = int(ly * self.height())
            grad = QRadialGradient(px, py, 35)
            grad.setColorAt(0, QColor(0, 255, 255, 90))
            grad.setColorAt(1, QColor(0, 255, 255, 0))
            painter.setBrush(QBrush(grad))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(px - 35, py - 35, 70, 70)
            painter.setPen(QPen(QColor(0, 255, 255, 200), 1.5))
            painter.setBrush(Qt.NoBrush)
            painter.drawEllipse(px - 10, py - 10, 20, 20)
            painter.drawLine(px - 18, py, px + 18, py)
            painter.drawLine(px, py - 18, px, py + 18)
            painter.setFont(QFont("Consolas", 8))
            painter.drawText(px + 14, py - 12, f"PTR {int(lx * 100)},{int(ly * 100)}")

    # ── Vision Frame & Stats ───────────────────────────────────────────────────

    def _update_vision(self, frame):
        from PyQt5.QtGui import QImage, QPixmap
        h, w, ch = frame.shape
        bytes_per_line = ch * w
        q_img = QImage(frame.data, w, h, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
        self.vision_label.setPixmap(
            QPixmap.fromImage(q_img).scaled(
                self.vision_label.width(), self.vision_label.height(), Qt.KeepAspectRatio
            )
        )

    def update_stats(self):
        import datetime
        self.clock_label.setText(datetime.datetime.now().strftime("%H:%M:%S"))
        self.cpu_gauge.set_value(psutil.cpu_percent())
        self.ram_gauge.set_value(psutil.virtual_memory().percent)
        try:
            self.disk_gauge.set_value(psutil.disk_usage('/').percent)
        except Exception:
            pass
        self.hardware_info.update_sensors()
        self.process_monitor.update_processes()
        self.net_chart.update_data()

        try:
            from jarvis.core.world_state import world
            explain = world.get_snapshot().get("temporal_context", {}).get("last_fusion_explanation")
            if explain:
                self.explain_label.setText(f"C4: {explain}")
        except ImportError:
            pass



def start_hui():
    app = QApplication(sys.argv)
    window = HUIDashboard()
    window.show()
    return app, window
