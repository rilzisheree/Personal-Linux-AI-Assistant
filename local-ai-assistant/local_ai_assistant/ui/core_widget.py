"""The compact animated-console style core used by Lura's dashboard."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QBrush, QColor, QPainter, QPen
from PySide6.QtWidgets import QSizePolicy, QWidget


class CoreWidget(QWidget):
    """Draw a small concentric local-AI core without external UI assets."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setMinimumSize(170, 170)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def paintEvent(self, event) -> None:
        del event
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        center = self.rect().center()
        radius = min(self.width(), self.height()) * 0.32

        painter.setPen(QPen(QColor("#102e40"), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        for scale in (1.42, 1.2, 1.0, 0.78):
            painter.drawEllipse(center, radius * scale, radius * scale)

        painter.setPen(QPen(QColor("#1c5b76"), 1))
        for index in range(16):
            angle = index * 22.5
            painter.save()
            painter.translate(center)
            painter.rotate(angle)
            painter.drawLine(0, int(-radius * 1.33), 0, int(-radius * 1.25))
            painter.restore()

        painter.setPen(QPen(QColor("#24566d"), 1))
        painter.setBrush(QBrush(QColor("#172f3c")))
        painter.drawEllipse(center, radius * 0.72, radius * 0.72)
        painter.setPen(QPen(QColor("#37758d"), 1))
        painter.drawEllipse(center, radius * 0.56, radius * 0.56)
        painter.setPen(QPen(QColor("#43879a"), 1))
        painter.drawEllipse(center, radius * 0.23, radius * 0.23)

        painter.setPen(QPen(QColor("#51b4c7"), 1))
        for offset in range(-3, 4):
            painter.drawPoint(int(center.x() + offset * 3), center.y())