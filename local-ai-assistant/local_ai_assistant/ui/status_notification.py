"""Non-focus-stealing status notification for background voice activity."""

from __future__ import annotations

from PySide6.QtCore import QPoint, Qt
from PySide6.QtGui import QCursor, QGuiApplication
from PySide6.QtWidgets import QHBoxLayout, QLabel, QWidget


class StatusNotification(QWidget):
    """A small top-of-screen notification for background voice states."""

    STATE_LABELS = {
        "idle": "Idle",
        "listening": "Listening…",
        "processing": "Thinking…",
        "speaking": "Speaking…",
        "error": "Error",
    }
    STATE_COLORS = {
        "idle": "#75b9e6",
        "listening": "#6dd9b4",
        "processing": "#b189ff",
        "speaking": "#4aaeff",
        "error": "#e07f91",
    }

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("statusNotification")
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.ToolTip
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating)
        self.setAttribute(Qt.WidgetAttribute.WA_X11DoNotAcceptFocus)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)
        self.setFixedHeight(42)

        self._state = "idle"
        self._dot = QLabel("●")
        self._dot.setObjectName("statusNotificationDot")
        self._label = QLabel()
        self._label.setObjectName("statusNotificationText")

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 0, 16, 0)
        layout.setSpacing(9)
        layout.addWidget(self._dot)
        layout.addWidget(self._label)

        self.set_state("idle")
        self.hide()

    def set_state(self, state: str) -> None:
        normalized = "processing" if state == "thinking" else state
        self._state = normalized if normalized in self.STATE_LABELS else "idle"
        self._label.setText(self.STATE_LABELS[self._state])
        self.setProperty("state", self._state)
        self._dot.setStyleSheet(
            "QLabel#statusNotificationDot {"
            f"color: {self.STATE_COLORS[self._state]};"
            "font-size: 14px;"
            "}"
        )
        self.style().unpolish(self)
        self.style().polish(self)
        self.adjustSize()

    def show_for_desktop(self) -> None:
        screen = QGuiApplication.screenAt(QCursor.pos())
        if screen is None:
            screen = QGuiApplication.primaryScreen()
        if screen is None:
            return
        geometry = screen.availableGeometry()
        self.adjustSize()
        x = geometry.x() + (geometry.width() - self.width()) // 2
        self.move(QPoint(x, geometry.y() + 18))
        self.show()
        self.raise_()