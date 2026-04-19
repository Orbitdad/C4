import sys
import math
import time
from PyQt5.QtWidgets import QWidget, QApplication
from PyQt5.QtCore import Qt, QTimer, QPointF
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush

class HandsOverlay(QWidget):
    """
    Global OS-level transparent overlay that renders the user's hand positions
    and gestures over the desktop natively.
    """
    def __init__(self, signals=None, parent=None):
        super().__init__(parent)
        
        # Transparent, borderless, always on top, and ignores mouse events
        self.setWindowFlags(
            Qt.WindowStaysOnTopHint | 
            Qt.FramelessWindowHint | 
            Qt.WindowTransparentForInput | 
            Qt.Tool
        )
        self.setAttribute(Qt.WA_TranslucentBackground)
        
        self.signals = signals
        if self.signals:
            if hasattr(self.signals, 'update_gesture_debug'):
                self.signals.update_gesture_debug.connect(self.on_gesture_update, Qt.QueuedConnection)
            if hasattr(self.signals, 'update_landmarks'):
                self.signals.update_landmarks.connect(self.on_landmarks_update, Qt.QueuedConnection)
                
        self.primary_pos = None # (x_norm, y_norm)
        self.current_gesture = "NONE"
        self.all_landmarks = [] # secondary dots
        self.last_update_time = 0.0
        self.pulse = 0.0
        
        # Build UI animation loops
        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self._animate)
        self.anim_timer.start(30)
        
    def showEvent(self, event):
        super().showEvent(event)
        # Ensure it covers all screens
        geometry = QApplication.desktop().geometry()
        self.setGeometry(geometry)

    def on_gesture_update(self, debug_dict):
        """Called when VisionManager emits gesture debug data."""
        if not debug_dict:
            return
            
        self.last_update_time = time.time()
        
        if "pos" in debug_dict:
            self.primary_pos = debug_dict["pos"] # (x, y, z)
        
        if "gesture" in debug_dict:
            self.current_gesture = debug_dict["gesture"]
            
        # Transform overrides
        if debug_dict.get("type") == "TRANSFORM":
            pass
            
        self.update()
        
    def on_landmarks_update(self, landmarks):
        """Called when VisionManager emits all hand tips (for multi-hand)."""
        self.last_update_time = time.time()
        self.all_landmarks = landmarks
        self.update()

    def _animate(self):
        self.pulse = (self.pulse + 0.1) % (math.pi * 2)
        # Auto hide after 2 seconds of inactivity
        if time.time() - self.last_update_time > 2.0:
            if self.primary_pos or self.all_landmarks:
                self.primary_pos = None
                self.all_landmarks = []
                self.current_gesture = "NONE"
                self.update()
        else:
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        w = self.width()
        h = self.height()
        
        # Draw secondary fingertips if available
        if self.all_landmarks:
            for (x_norm, y_norm) in self.all_landmarks:
                px = int(x_norm * w)
                py = int(y_norm * h)
                
                # Small trace dot
                painter.setPen(Qt.NoPen)
                painter.setBrush(QBrush(QColor(0, 200, 255, 60)))
                painter.drawEllipse(px - 10, py - 10, 20, 20)
                painter.setBrush(QBrush(QColor(0, 255, 255, 120)))
                painter.drawEllipse(px - 4, py - 4, 8, 8)
                
        # Draw primary hand crosshair
        if self.primary_pos and len(self.primary_pos) >= 2:
            x_norm, y_norm = self.primary_pos[0], self.primary_pos[1]
            cx = int(x_norm * w)
            cy = int(y_norm * h)
            
            is_pinching = self.current_gesture in ["SELECT", "DRAG"]
            is_fist = self.current_gesture == "CANCEL"
            is_nav = self.current_gesture == "NAVIGATE"
            
            # Colors: DRAG=Orange, SELECT=Green, CANCEL=Red, NAV=Purple, default TARGET=Cyan
            if self.current_gesture == "DRAG":
                 base_color = QColor(255, 140, 0)
            elif self.current_gesture == "SELECT":
                 base_color = QColor(0, 255, 80)
            elif is_fist:
                 base_color = QColor(255, 40, 40)
            elif is_nav:
                 base_color = QColor(140, 0, 255)
            else:
                 base_color = QColor(0, 255, 200)

            glow_color = QColor(base_color.red(), base_color.green(), base_color.blue(), 100)
            
            # Dynamic radii based on intent
            if is_pinching:
                r_outer = 25
                r_inner = 10
            elif is_nav:
                r_outer = 45 + (math.sin(self.pulse * 3) * 5) # Fast pulse for nav
                r_inner = 20
            else:
                r_outer = 45
                r_inner = 25 + math.sin(self.pulse) * 2
            
            # Glow
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(glow_color))
            painter.drawEllipse(cx - int(r_outer), cy - int(r_outer), int(r_outer * 2), int(r_outer * 2))
            
            # Reticle
            painter.setBrush(Qt.NoBrush)
            pen = QPen(base_color, 2)
            painter.setPen(pen)
            painter.drawEllipse(cx - int(r_inner), cy - int(r_inner), int(r_inner * 2), int(r_inner * 2))
            
            # Crosshairs
            length = 8
            painter.drawLine(cx, cy - int(r_inner), cx, cy - int(r_inner) + length)
            painter.drawLine(cx, cy + int(r_inner), cx, cy + int(r_inner) - length)
            painter.drawLine(cx - int(r_inner), cy, cx - int(r_inner) + length, cy)
            painter.drawLine(cx + int(r_inner), cy, cx + int(r_inner) - length, cy)
            
            # Center dot
            painter.setPen(Qt.NoPen)
            painter.setBrush(QBrush(base_color))
            painter.drawEllipse(cx - 3, cy - 3, 6, 6)
            
            # Optional tag
            if is_pinching:
                painter.setPen(base_color)
                painter.setFont(painter.font())
                painter.drawText(cx + 35, cy + 5, self.current_gesture)
            elif is_nav or is_fist:
                painter.setPen(base_color)
                painter.setFont(painter.font())
                painter.drawText(cx + 40, cy + 5, self.current_gesture)
