"""The animated orb that represents Lura's current state."""

from __future__ import annotations

import math

from PySide6.QtCore import QEvent, QPointF, QRectF, QTimer, Qt, Signal
from PySide6.QtGui import QColor, QLinearGradient, QPainter, QPen, QRadialGradient
from PySide6.QtWidgets import QApplication, QSizePolicy, QWidget


class CoreWidget(QWidget):
    """A quiet, interactive AI orb with distinct listening/thinking states."""

    pressed = Signal()
    released = Signal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(220, 220)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAttribute(Qt.WidgetAttribute.WA_AcceptTouchEvents, True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setAccessibleName("Lura voice orb")
        self.setToolTip("Hold to speak")
        self._state = "idle"
        self._phase = 0.0
        self._pressed = False
        self._mouse_pressed = False
        self._hovered = False
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(32)
        app = QApplication.instance()
        if app is not None:
            app.installEventFilter(self)

    @property
    def state(self) -> str:
        return self._state

    def set_state(self, state: str) -> None:
        if state not in {"idle", "listening", "thinking", "speaking", "error"}:
            state = "idle"
        if state == self._state:
            self.update()
            return
        self._state = state
        self.update()

    def _tick(self) -> None:
        if (
            self._mouse_pressed
            and not (QApplication.mouseButtons() & Qt.MouseButton.LeftButton)
        ):
            self._finish_press()
        speeds = {
            "idle": 0.018,
            "listening": 0.042,
            "thinking": 0.06,
            "speaking": 0.085,
            "error": 0.025,
        }
        self._phase = (self._phase + speeds[self._state]) % (math.pi * 2)
        self.update()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self._pressed = True
            self._mouse_pressed = True
            self.grabMouse()
            self.set_state("listening")
            self.pressed.emit()
            event.accept()
            return
        super().mousePressEvent(event)

    def mouseReleaseEvent(self, event) -> None:
        if self._pressed and event.button() in {
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.NoButton,
        }:
            self._finish_press()
            event.accept()
            return
        super().mouseReleaseEvent(event)

    def focusOutEvent(self, event) -> None:
        # If the window loses focus while the Orb is held, do not leave voice
        # recording stuck in the listening state.
        self._finish_press()
        super().focusOutEvent(event)

    def eventFilter(self, watched, event) -> bool:
        if self._pressed and event.type() in {
            QEvent.Type.MouseButtonRelease,
            QEvent.Type.ApplicationDeactivate,
        }:
            button = getattr(event, "button", lambda: Qt.MouseButton.NoButton)()
            if event.type() == QEvent.Type.ApplicationDeactivate or button in {
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.NoButton,
            }:
                self._finish_press()
        return super().eventFilter(watched, event)

    def _finish_press(self) -> None:
        if not self._pressed:
            return
        self._pressed = False
        self._mouse_pressed = False
        if self.mouseGrabber() is self:
            self.releaseMouse()
        self.released.emit()

    def event(self, event) -> bool:
        if event.type() == QEvent.Type.TouchBegin:
            if not self._pressed:
                self._pressed = True
                self._mouse_pressed = False
                self.set_state("listening")
                self.pressed.emit()
            event.accept()
            return True
        if event.type() in {
            QEvent.Type.TouchEnd,
            QEvent.Type.TouchCancel,
        }:
            self._finish_press()
            event.accept()
            return True
        return super().event(event)

    def enterEvent(self, event) -> None:
        self._hovered = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event) -> None:
        self._hovered = False
        self.update()
        super().leaveEvent(event)

    def keyPressEvent(self, event) -> None:
        if event.key() in {Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space}:
            if not self._pressed:
                self._pressed = True
                self.set_state("listening")
                self.pressed.emit()
            event.accept()
            return
        super().keyPressEvent(event)

    def keyReleaseEvent(self, event) -> None:
        if event.key() in {Qt.Key.Key_Return, Qt.Key.Key_Enter, Qt.Key.Key_Space}:
            if self._pressed:
                self._pressed = False
                self.released.emit()
            event.accept()
            return
        super().keyReleaseEvent(event)

    def paintEvent(self, event) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        side = min(self.width(), self.height())
        center = QPointF(self.width() / 2, self.height() / 2)
        base_radius = side * 0.27
        state_strength = {
            "idle": 0.0,
            "listening": 0.18,
            "thinking": 0.12,
            "speaking": 0.24,
            "error": 0.08,
        }[self._state]
        breath = math.sin(self._phase) * (0.018 + state_strength * 0.04)
        radius = base_radius * (1.0 + breath + (0.035 if self._pressed else 0.0))
        glow_alpha = {
            "idle": 78,
            "listening": 132,
            "thinking": 116,
            "speaking": 158,
            "error": 96,
        }[self._state]
        if self._hovered:
            glow_alpha += 18

        self._paint_glow(painter, center, radius, glow_alpha)
        self._paint_orbit(painter, center, radius, state_strength)
        self._paint_orb(painter, center, radius, state_strength)
        self._paint_state_ripples(painter, center, radius, state_strength)
        painter.end()

    @staticmethod
    def _paint_glow(
        painter: QPainter,
        center: QPointF,
        radius: float,
        alpha: int,
    ) -> None:
        gradient = QRadialGradient(center, radius * 2.15)
        gradient.setColorAt(0.0, QColor(44, 154, 221, alpha))
        gradient.setColorAt(0.34, QColor(20, 91, 145, int(alpha * 0.72)))
        gradient.setColorAt(0.62, QColor(8, 43, 79, int(alpha * 0.38)))
        gradient.setColorAt(1.0, QColor(0, 8, 18, 0))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(gradient)
        painter.drawEllipse(center, radius * 2.15, radius * 2.15)

    def _paint_orbit(
        self,
        painter: QPainter,
        center: QPointF,
        radius: float,
        strength: float,
    ) -> None:
        orbit_radius = radius * (1.42 + strength * 0.35)
        color = QColor(83, 183, 235, 180 if self._state != "idle" else 128)
        painter.setPen(QPen(color, 1.25))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.save()
        painter.translate(center)
        painter.rotate(math.degrees(self._phase * 0.48))
        painter.drawArc(
            QRectF(-orbit_radius, -orbit_radius * 0.42, orbit_radius * 2, orbit_radius * 0.84),
            24 * 16,
            116 * 16,
        )
        painter.rotate(156)
        painter.setPen(QPen(QColor(46, 125, 186, 120), 1.0))
        painter.drawArc(
            QRectF(-orbit_radius * 0.92, -orbit_radius * 0.3, orbit_radius * 1.84, orbit_radius * 0.6),
            194 * 16,
            102 * 16,
        )
        painter.restore()

    def _paint_orb(
        self,
        painter: QPainter,
        center: QPointF,
        radius: float,
        strength: float,
    ) -> None:
        orb_gradient = QRadialGradient(
            center - QPointF(radius * 0.25, radius * 0.35),
            radius * 1.25,
        )
        orb_gradient.setColorAt(0.0, QColor(63, 145, 202, 255))
        orb_gradient.setColorAt(0.32, QColor(19, 61, 94, 255))
        orb_gradient.setColorAt(0.58, QColor(6, 23, 42, 255))
        orb_gradient.setColorAt(0.82, QColor(2, 10, 21, 255))
        orb_gradient.setColorAt(1.0, QColor(0, 3, 10, 255))
        painter.setPen(QPen(QColor(91, 190, 237, 220), 1.25))
        painter.setBrush(orb_gradient)
        painter.drawEllipse(center, radius, radius)

        inner_radius = radius * (0.66 + strength * 0.08)
        inner = QRadialGradient(center + QPointF(radius * 0.12, radius * 0.1), inner_radius)
        inner.setColorAt(0.0, QColor(61, 167, 224, 180 + int(strength * 65)))
        inner.setColorAt(0.52, QColor(12, 61, 101, 145))
        inner.setColorAt(0.78, QColor(5, 28, 52, 80))
        inner.setColorAt(1.0, QColor(3, 12, 24, 0))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(inner)
        painter.drawEllipse(center, inner_radius, inner_radius)

        painter.save()
        painter.translate(center)
        painter.rotate(math.degrees(self._phase * 0.7))
        for index in range(3):
            angle = index * 2.1
            point = QPointF(
                math.cos(angle) * radius * (0.38 + strength * 0.22),
                math.sin(angle) * radius * (0.3 + strength * 0.14),
            )
            painter.setPen(Qt.PenStyle.NoPen)
            painter.setBrush(QColor(119, 215, 250, 105 + int(strength * 90)))
            painter.drawEllipse(point, radius * 0.065, radius * 0.065)
        painter.restore()

        rim = QLinearGradient(
            center - QPointF(radius, radius),
            center + QPointF(radius, radius),
        )
        rim.setColorAt(0.1, QColor(150, 226, 255, 180))
        rim.setColorAt(0.5, QColor(54, 140, 201, 110))
        rim.setColorAt(0.9, QColor(5, 30, 56, 180))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(rim, 1.5))
        painter.drawEllipse(center, radius * 0.96, radius * 0.96)

    def _paint_state_ripples(
        self,
        painter: QPainter,
        center: QPointF,
        radius: float,
        strength: float,
    ) -> None:
        if self._state == "idle":
            return
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for index in range(2 if self._state == "listening" else 3):
            cycle = (self._phase * (0.6 + index * 0.14) + index * 1.9) % (math.pi * 2)
            pulse = (math.sin(cycle) + 1) / 2
            ripple_radius = radius * (1.12 + pulse * (0.28 + strength * 0.28))
            alpha = max(0, int((1 - pulse) * (58 + strength * 115)))
            painter.setPen(QPen(QColor(81, 180, 232, alpha), 1.25))
            painter.drawEllipse(center, ripple_radius, ripple_radius)
