"""A small, non-focus-stealing orb for interactions outside Lura's window."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QCursor, QGuiApplication
from PySide6.QtWidgets import QSizePolicy

from .core_widget import CoreWidget


class GlobalOrb(CoreWidget):
    """Top-level orb that floats above other applications on Linux desktops."""

    STATE_LABELS = {
        "idle": "Idle",
        "listening": "Listening",
        "processing": "Processing",
        "speaking": "Speaking",
        "error": "Error",
    }

    def __init__(self, owner=None) -> None:
        # Keep the owner alive through MainWindow's reference, but do not make
        # this widget a child: it must remain a top-level desktop window.
        super().__init__(None)
        self._owner = owner
        self.setObjectName("globalOrb")
        self.setMinimumSize(0, 0)
        self.setFixedSize(112, 112)
        self.setSizePolicy(
            QSizePolicy.Policy.Fixed,
            QSizePolicy.Policy.Fixed,
        )
        self.setWindowFlags(
            Qt.WindowType.Tool
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_ShowWithoutActivating, True)
        # This flag is supported by X11 and ignored safely by other Qt
        # backends. It prevents the overlay from stealing keyboard focus.
        self.setAttribute(Qt.WidgetAttribute.WA_X11DoNotAcceptFocus, True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setToolTip("Lura — Idle · hold to speak")
        self.setAccessibleName("Lura global voice orb")
        self.setAccessibleDescription(
            "Floating Lura voice control. Hold to speak without opening the app."
        )

    def set_state(self, state: str) -> None:
        super().set_state(state)
        normalized = "processing" if state == "thinking" else state
        label = self.STATE_LABELS.get(normalized, "Idle")
        action = "hold to speak" if normalized in {"idle", "error"} else "Lura is active"
        self.setToolTip(f"Lura — {label} · {action}")

    def show_for_desktop(self) -> None:
        """Place the orb on the active monitor without activating it."""
        screen = QGuiApplication.screenAt(QCursor.pos())
        if screen is None:
            screen = QGuiApplication.primaryScreen()
        if screen is not None:
            available = screen.availableGeometry()
            margin = 24
            self.move(
                available.right() - self.width() - margin,
                available.bottom() - self.height() - margin,
            )
        self.show()
        self.raise_()
