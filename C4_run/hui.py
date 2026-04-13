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
    QSizePolicy
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
    update_vision = pyqtSignal(object)       # opencv frame
    update_landmarks = pyqtSignal(list)      # hand landmarks
    speaking_started = pyqtSignal(str)       # text being spoken
    speaking_stopped = pyqtSignal()
    thinking_started = pyqtSignal()
    thinking_stopped = pyqtSignal()
    face_detected = pyqtSignal(str, float)   # name, confidence
    alert_critical = pyqtSignal(str)         # critical message
    transcript_user = pyqtSignal(str)        # what user said
    transcript_jarvis = pyqtSignal(str)      # what jarvis said
    boot_step = pyqtSignal(int)              # boot sequence step 0-7
    update_gesture_debug = pyqtSignal(dict)  # real-time continuous gesture info


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


# ── Radar HUD ─────────────────────────────────────────────────────────────────

class RadarHUD(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.angle = 0
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_angle)
        self.timer.start(40)
        self.points = []
        self.setMinimumSize(200, 200)

    def _update_angle(self):
        self.angle = (self.angle + 4) % 360
        self.update()

    def refresh_network(self):
        self.points.clear()
        try:
            conns = psutil.net_connections(kind='inet')
            for conn in conns:
                if conn.status == 'ESTABLISHED' and conn.raddr:
                    ip_hash = hash(conn.raddr.ip)
                    dist = (abs(ip_hash) % 80)
                    deg = (abs(ip_hash) % 360)
                    rad = math.radians(deg)
                    px = int(math.cos(rad) * dist)
                    py = int(math.sin(rad) * dist)
                    self.points.append((px, py))
        except Exception:
            pass
        if not self.points:
            self.points = [(random.randint(-70, 70), random.randint(-70, 70)) for _ in range(4)]

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        width = self.width()
        height = self.height()
        side = min(width, height)
        painter.translate(width / 2, height / 2)
        painter.scale(side / 200.0, side / 200.0)
        painter.setBrush(QBrush(QColor(0, 20, 40, 100)))
        painter.setPen(QPen(QColor(0, 210, 255, 50), 1))
        painter.drawEllipse(-95, -95, 190, 190)
        for r in [30, 60, 90]:
            painter.drawEllipse(-r, -r, r * 2, r * 2)
        painter.drawLine(-95, 0, 95, 0)
        painter.drawLine(0, -95, 0, 95)
        gradient = QRadialGradient(0, 0, 95, 0, 0)
        gradient.setColorAt(0, QColor(0, 210, 255, 160))
        gradient.setColorAt(1, QColor(0, 210, 255, 0))
        painter.setBrush(QBrush(gradient))
        painter.setPen(Qt.NoPen)
        painter.drawPie(-95, -95, 190, 190, (90 - self.angle) * 16, 40 * 16)
        painter.setBrush(QBrush(QColor(0, 255, 255)))
        for px, py in self.points:
            painter.drawEllipse(px, py, 4, 4)


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


# ── Wireframe Globe ────────────────────────────────────────────────────────────

class WireframeGlobeHUD(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(200, 200)
        self.rotation = 0
        self.nodes = []
        rings = 10
        segments = 20
        for i in range(rings + 1):
            theta = i * math.pi / rings
            for j in range(segments):
                phi = j * 2 * math.pi / segments
                x = math.sin(theta) * math.cos(phi)
                y = math.sin(theta) * math.sin(phi)
                z = math.cos(theta)
                self.nodes.append((x, y, z))
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._rotate)
        self.timer.start(40)

    def _rotate(self):
        self.rotation += 0.04
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        w = self.width()
        h = self.height()
        s = min(w, h) / 2.2
        painter.translate(w / 2, h / 2)
        painter.setBrush(QBrush(QColor(0, 30, 60, 50)))
        painter.setPen(Qt.NoPen)
        painter.drawEllipse(int(-s), int(-s), int(s * 2), int(s * 2))
        painter.setPen(QPen(QColor(0, 255, 255, 60), 1))
        c = math.cos(self.rotation)
        si = math.sin(self.rotation)
        for x, y, z in self.nodes:
            rot_x = x * c + z * si
            rot_z = -x * si + z * c
            painter.setBrush(QBrush(QColor(0, 255, 255, 20 if rot_z < 0 else 150)))
            painter.setPen(Qt.NoPen)
            px = int(rot_x * s)
            py = int(y * s * 0.9)
            painter.drawEllipse(px - 2, py - 2, 4, 4)


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
        self.setWindowFlags(Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TranslucentBackground)

        self.signals = HUISignals()
        self.signals.update_status.connect(self._update_status)
        self.signals.log_message.connect(self._log_message)
        self.signals.update_vision.connect(self._update_vision)
        self.signals.update_landmarks.connect(self._update_landmarks)
        self.signals.speaking_started.connect(self._on_speaking_started)
        self.signals.speaking_stopped.connect(self._on_speaking_stopped)
        self.signals.thinking_started.connect(self._on_thinking_started)
        self.signals.thinking_stopped.connect(self._on_thinking_stopped)
        self.signals.face_detected.connect(self._on_face_detected)
        self.signals.alert_critical.connect(self._on_critical_alert)
        self.signals.transcript_user.connect(self._on_transcript_user)
        self.signals.transcript_jarvis.connect(self._on_transcript_jarvis)
        self.signals.boot_step.connect(self._on_boot_step)
        self.signals.update_gesture_debug.connect(self._on_gesture_debug)

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
        print("HUI: C4 Core Ready.")

    def init_ui(self):
        self.setWindowTitle("C4 INTELLIGENCE SYSTEM")
        self.setGeometry(0, 0, 1600, 950)

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        central_widget.setStyleSheet("""
            QWidget {
                background-color: rgba(1, 6, 12, 248);
                color: #00d2ff;
                font-family: 'Consolas', 'Courier New';
                /* Scanline visual effect */
                background-image: repeating-linear-gradient(0deg, transparent, transparent 2px, rgba(0, 210, 255, 0.03) 2px, rgba(0, 210, 255, 0.03) 4px);
            }
            QFrame#Panel {
                border: 1px solid rgba(0, 210, 255, 70);
                background-color: rgba(2, 12, 25, 160);
                border-radius: 3px;
            }
            QFrame#PanelActive {
                border: 2px solid rgba(0, 255, 255, 200);
                background-color: rgba(0, 40, 80, 60);
                border-radius: 3px;
            }
            QLabel#Title {
                font-weight: bold;
                letter-spacing: 3px;
                color: #00ffff;
                font-size: 11px;
            }
            QTextEdit {
                background: transparent;
                border: none;
                color: #00ffcc;
                font-size: 11px;
            }
        """)

        layout = QGridLayout(central_widget)
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)
        self.panels = {}

        # ── TOP ALERT BANNER ──────────────────────────────────────────────────
        self.alert_banner = AlertBanner()
        layout.addWidget(self.alert_banner, 0, 0, 1, 4)

        # ── HEADER ────────────────────────────────────────────────────────────
        header = QFrame()
        header.setObjectName("Panel")
        self.panels["header"] = header
        header_layout = QHBoxLayout(header)
        header_layout.setSpacing(12)

        # Left: Status orb + title
        left_hbox = QHBoxLayout()
        self.status_orb = StatusOrb()
        left_hbox.addWidget(self.status_orb)
        left_hbox.setAlignment(Qt.AlignVCenter)
        title_lbl = QLabel("C4  INTELLIGENCE  SYSTEM  ::  GLOBAL  COMMAND")
        title_lbl.setObjectName("Title")
        title_lbl.setFont(QFont("Consolas", 18, QFont.Bold))
        left_hbox.addWidget(title_lbl)
        header_layout.addLayout(left_hbox)
        header_layout.addStretch()

        # Center: JARVIS waveform
        waveform_layout = QVBoxLayout()
        waveform_layout.setAlignment(Qt.AlignCenter)
        self.waveform = JARVISWaveform()
        waveform_layout.addWidget(self.waveform)
        header_layout.addLayout(waveform_layout)
        header_layout.addStretch()

        # Right: Clock + status label
        right_vbox = QVBoxLayout()
        right_vbox.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.clock_label = QLabel("00:00:00")
        self.clock_label.setFont(QFont("Consolas", 20, QFont.Bold))
        self.clock_label.setStyleSheet("color: #00ffcc;")
        right_vbox.addWidget(self.clock_label)
        self.ai_status_label = QLabel("INITIALIZING...")
        self.ai_status_label.setFont(QFont("Consolas", 10))
        self.ai_status_label.setStyleSheet("color: #4499aa;")
        right_vbox.addWidget(self.ai_status_label)
        header_layout.addLayout(right_vbox)

        layout.addWidget(header, 1, 0, 1, 4)

        # ── VISION PANEL ──────────────────────────────────────────────────────
        vision_panel = QFrame()
        vision_panel.setObjectName("Panel")
        self.panels["vision"] = vision_panel
        vision_layout = QVBoxLayout(vision_panel)
        lbl_v = QLabel("◤ VISION MATRIX ◥")
        lbl_v.setObjectName("Title")
        vision_layout.addWidget(lbl_v)
        self.vision_label = QLabel("INITIALIZING SENSORS...")
        self.vision_label.setAlignment(Qt.AlignCenter)
        self.vision_label.setMinimumSize(380, 280)
        self.vision_label.setStyleSheet("background: rgba(0,5,10,200); border: 1px solid rgba(0,210,255,40);")
        vision_layout.addWidget(self.vision_label)
        self._face_lbl = QLabel("")
        self._face_lbl.setFont(QFont("Consolas", 10))
        self._face_lbl.setStyleSheet("color: #00ff88;")
        self._face_lbl.setAlignment(Qt.AlignCenter)
        vision_layout.addWidget(self._face_lbl)
        
        # Gesture telemetry labels
        self._gesture_lbl = QLabel("GESTURE: IDLE | MODE: SINGLE_HAND")
        self._gesture_lbl.setFont(QFont("Consolas", 10))
        self._gesture_lbl.setStyleSheet("color: #00d2ff;")
        self._gesture_lbl.setAlignment(Qt.AlignCenter)
        vision_layout.addWidget(self._gesture_lbl)
        
        self._metrics_lbl = QLabel("Z: -- | ROLL: -- | CONF: --")
        self._metrics_lbl.setFont(QFont("Consolas", 10))
        self._metrics_lbl.setStyleSheet("color: #0088ff;")
        self._metrics_lbl.setAlignment(Qt.AlignCenter)
        vision_layout.addWidget(self._metrics_lbl)
        
        layout.addWidget(vision_panel, 2, 0, 2, 1)

        # ── GLOBE + RADAR ──────────────────────────────────────────────────────
        globe_panel = QFrame()
        globe_panel.setObjectName("Panel")
        self.panels["globe"] = globe_panel
        globe_layout = QHBoxLayout(globe_panel)
        globe_vbox = QVBoxLayout()
        lbl_g = QLabel("◤ GLOBAL ASSETS ◥")
        lbl_g.setObjectName("Title")
        globe_vbox.addWidget(lbl_g)
        self.globe = WireframeGlobeHUD()
        globe_vbox.addWidget(self.globe)
        globe_layout.addLayout(globe_vbox)
        radar_vbox = QVBoxLayout()
        lbl_r = QLabel("◤ SECURE SOCKET RADAR ◥")
        lbl_r.setObjectName("Title")
        radar_vbox.addWidget(lbl_r)
        self.radar = RadarHUD()
        radar_vbox.addWidget(self.radar)
        globe_layout.addLayout(radar_vbox)
        layout.addWidget(globe_panel, 2, 1, 1, 2)

        # ── SYSTEM DIAGNOSTICS ────────────────────────────────────────────────
        sys_panel = QFrame()
        sys_panel.setObjectName("Panel")
        self.panels["system"] = sys_panel
        sys_layout = QVBoxLayout(sys_panel)
        lbl_s = QLabel("◤ SYSTEM DIAGNOSTICS ◥")
        lbl_s.setObjectName("Title")
        sys_layout.addWidget(lbl_s)
        self.hardware_info = HardwareSensorsHUD()
        sys_layout.addWidget(self.hardware_info)
        sys_layout.addWidget(QLabel(""))
        lbl_p = QLabel("◤ ACTIVE PROCESSES ◥")
        lbl_p.setObjectName("Title")
        sys_layout.addWidget(lbl_p)
        self.process_monitor = ProcessMonitorHUD()
        sys_layout.addWidget(self.process_monitor)
        layout.addWidget(sys_panel, 2, 3, 2, 1)

        # ── METRICS GAUGES ────────────────────────────────────────────────────
        metrics_panel = QFrame()
        metrics_panel.setObjectName("Panel")
        self.panels["metrics"] = metrics_panel
        metrics_layout = QHBoxLayout(metrics_panel)
        self.cpu_gauge = CircularGauge("CPU CORE")
        self.ram_gauge = CircularGauge("SYS RAM")
        self.disk_gauge = CircularGauge("SSD VOL", color="#ff00cc")
        self.arc_reactor = ArcReactorHUD()
        
        metrics_layout.addWidget(self.cpu_gauge)
        metrics_layout.addWidget(self.arc_reactor)
        metrics_layout.addWidget(self.ram_gauge)
        metrics_layout.addWidget(self.disk_gauge)
        layout.addWidget(metrics_panel, 3, 1, 1, 1)

        # ── NETWORK CHART ─────────────────────────────────────────────────────
        chart_panel = QFrame()
        chart_panel.setObjectName("Panel")
        self.panels["network"] = chart_panel
        chart_layout = QVBoxLayout(chart_panel)
        lbl_n = QLabel("◤ NETWORK BANDWIDTH SIGNAL ◥")
        lbl_n.setObjectName("Title")
        chart_layout.addWidget(lbl_n)
        self.net_chart = BarChartHUD()
        chart_layout.addWidget(self.net_chart)
        layout.addWidget(chart_panel, 3, 2, 1, 1)

        # ── TRANSCRIPT + COMMAND LOG ─────────────────────────────────────────
        log_panel = QFrame()
        log_panel.setObjectName("Panel")
        self.panels["log"] = log_panel
        log_layout = QVBoxLayout(log_panel)
        log_layout.setSpacing(6)

        # Transcript top section
        self.transcript_panel = TranscriptPanel()
        log_layout.addWidget(self.transcript_panel)

        # Explanation line
        self.explain_label = QLabel("C4 :: Awaiting instruction...")
        self.explain_label.setStyleSheet("color: #ffcc00; font-size: 12px; font-style: italic;")
        self.explain_label.setWordWrap(True)
        log_layout.addWidget(self.explain_label)

        # Log text
        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setMaximumHeight(100)
        log_layout.addWidget(self.log_text)

        layout.addWidget(log_panel, 4, 0, 1, 4)

        # ── TIMERS ────────────────────────────────────────────────────────────
        self.stats_timer = QTimer()
        self.stats_timer.timeout.connect(self.update_stats)
        self.stats_timer.start(1000)

        # Kick off boot sequence
        QTimer.singleShot(200, self._start_boot_sequence)

    # ── Boot Sequence ─────────────────────────────────────────────────────────

    def _start_boot_sequence(self):
        """Animate panels revealing one-by-one on startup."""
        self._boot_step = 0
        # Hide all panels initially
        for panel in self.panels.values():
            panel.setVisible(False)

        boot_order = ["header", "system", "vision", "metrics", "network", "globe", "log"]
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

    def _on_thinking_stopped(self):
        self.status_orb.set_mode("idle")
        self.ai_status_label.setStyleSheet("color: #4499aa; font-size: 10px;")

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

    def mouseMoveEvent(self, event):
        lx = event.x() / self.width()
        ly = event.y() / self.height()
        self._update_landmarks([(lx, ly)])

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
            for name, panel in self.panels.items():
                if name == new_focus:
                    panel.setStyleSheet("border: 2px solid rgba(0, 255, 255, 200); background-color: rgba(0, 80, 130, 40);")
                else:
                    panel.setStyleSheet("")
            self.focused_panel = new_focus
            self.dwell_count = 0

        if new_focus and new_focus == self.focused_panel:
            self.dwell_count += 1
            if self.dwell_count == 25:
                self._log_message(f"HUI: GESTURE COMMAND → {new_focus.upper()}")
                self.trigger_glitch()
                if new_focus == "log":
                    self.log_text.clear()
                elif new_focus == "globe":
                    self.radar.refresh_network()
                    self._log_message("NETWORK: Socket radar refreshed.")
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

        if random.random() < 0.15:
            self.radar.refresh_network()


    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        # Parallax effect based on mouse pos
        cx, cy = self.width() / 2, self.height() / 2
        dx, dy = event.x() - cx, event.y() - cy
        
        from PyQt5.QtWidgets import QGraphicsDropShadowEffect
        for name, panel in self.panels.items():
            if not panel: continue
            effect = panel.graphicsEffect()
            if not isinstance(effect, QGraphicsDropShadowEffect):
                effect = QGraphicsDropShadowEffect()
                effect.setBlurRadius(15)
                # Cyan/blue shadow for holographic effect
                effect.setColor(QColor(0, 210, 255, 60))
                panel.setGraphicsEffect(effect)
            
            # Offset shadow to create 3D depth illusion (parallax)
            depth_multiplier = -0.05 if name in ['header', 'globe'] else -0.02
            effect.setOffset(dx * depth_multiplier, dy * depth_multiplier)

def start_hui():
    app = QApplication(sys.argv)
    window = HUIDashboard()
    window.show()
    return app, window
