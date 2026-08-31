"""Minimal transcript rendering for the orb-centered experience."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QFont, QPixmap
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QScrollArea,
    QSizePolicy,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from ..ollama import ChatMessage


def _make_empty_state() -> QWidget:
    state = QLabel("A quiet channel.\n\nAsk Lura anything.")
    state.setObjectName("emptyState")
    state.setAlignment(Qt.AlignmentFlag.AlignCenter)
    return state


class MessageBubble(QFrame):
    """A lightweight transcript row, intentionally not a chat card."""

    def __init__(
        self,
        role: str,
        content: str = "",
        images: tuple[str, ...] = (),
    ) -> None:
        super().__init__()
        self.role = role
        self.setObjectName("messageRow")
        self.setProperty("role", role)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        self.content_layout = QVBoxLayout(self)
        self.content_layout.setContentsMargins(8, 7, 8, 7)
        self.content_layout.setSpacing(3)

        label = QLabel({"user": "YOU", "tool": "SYSTEM"}.get(role, "LURA"))
        label.setObjectName("messageRole")
        self.content_layout.addWidget(label)

        self.body = QTextBrowser()
        self.body.setOpenExternalLinks(True)
        self.body.setFrameShape(QFrame.Shape.NoFrame)
        self.body.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.body.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.body.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.body.setFont(QFont("Noto Sans", 13))
        self.body.setMarkdown(content)
        self.body.document().contentsChanged.connect(self._resize_to_content)
        self.content_layout.addWidget(self.body)
        self.add_images(images)
        self._resize_to_content()

    def set_content(self, content: str) -> None:
        self.body.setMarkdown(content)
        self._resize_to_content()

    def append_content(self, content: str) -> None:
        self.set_content(self.body.toPlainText() + content)

    def add_images(self, image_paths: tuple[str, ...] | list[str]) -> None:
        for image_path in image_paths:
            path = Path(image_path).expanduser()
            pixmap = QPixmap(str(path))
            if pixmap.isNull():
                continue
            image = QLabel()
            image.setObjectName("messageImage")
            image.setAlignment(Qt.AlignmentFlag.AlignLeft)
            image.setPixmap(
                pixmap.scaled(
                    700,
                    440,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
            )
            image.setToolTip(str(path))
            self.content_layout.addWidget(image)
        self._resize_to_content()

    def _resize_to_content(self) -> None:
        height = int(self.body.document().size().height()) + 10
        self.body.setMinimumHeight(max(28, height))
        self.body.setMaximumHeight(max(28, height))


class ChatView(QScrollArea):
    def __init__(self) -> None:
        super().__init__()
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self.container = QWidget()
        self.layout = QVBoxLayout(self.container)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.layout.setSpacing(2)
        self.layout.addStretch(1)
        self.setWidget(self.container)

        self.empty_state = _make_empty_state()
        self.layout.insertWidget(0, self.empty_state)

    def add_message(
        self,
        role: str,
        content: str = "",
        images: tuple[str, ...] = (),
    ) -> MessageBubble:
        if self.empty_state is not None:
            self.empty_state.deleteLater()
            self.empty_state = None
        bubble = MessageBubble(role, content, images)
        self.layout.insertWidget(self.layout.count() - 1, bubble)
        self._scroll_to_bottom()
        return bubble

    def clear_messages(self) -> None:
        while self.layout.count() > 1:
            item = self.layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.empty_state = _make_empty_state()
        self.layout.insertWidget(0, self.empty_state)

    def set_messages(self, messages: list[ChatMessage]) -> None:
        self.clear_messages()
        for message in messages:
            self.add_message(message.role, message.content, message.images)

    def _scroll_to_bottom(self) -> None:
        scrollbar = self.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())