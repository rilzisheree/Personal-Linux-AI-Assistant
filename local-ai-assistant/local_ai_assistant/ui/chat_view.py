"""Conversation rendering widgets."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
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


class MessageBubble(QFrame):
    def __init__(self, role: str, content: str = "") -> None:
        super().__init__()
        self.role = role
        self.setObjectName("assistantBubble" if role == "assistant" else "userBubble")
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(6)

        label = QLabel("You" if role == "user" else "Assistant")
        label.setStyleSheet("color: #7fced0; font-size: 11px; font-weight: 700;")
        layout.addWidget(label)

        self.body = QTextBrowser()
        self.body.setOpenExternalLinks(True)
        self.body.setFrameShape(QFrame.Shape.NoFrame)
        self.body.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.body.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.body.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.body.setFont(QFont("Noto Sans", 14))
        self.body.setMarkdown(content)
        self.body.document().contentsChanged.connect(self._resize_to_content)
        layout.addWidget(self.body)

        if role == "user":
            self.setStyleSheet(
                "#userBubble { background: #1b343f; border: 1px solid #2d5967; border-radius: 12px; }"
            )
        else:
            self.setStyleSheet(
                "#assistantBubble { background: #161d24; border: 1px solid #293641; border-radius: 12px; }"
            )
        self._resize_to_content()

    def set_content(self, content: str) -> None:
        self.body.setMarkdown(content)
        self._resize_to_content()

    def append_content(self, content: str) -> None:
        self.set_content(self.body.toPlainText() + content)

    def _resize_to_content(self) -> None:
        height = int(self.body.document().size().height()) + 12
        self.body.setMinimumHeight(max(32, height))
        self.body.setMaximumHeight(max(32, height))


class ChatView(QScrollArea):
    def __init__(self) -> None:
        super().__init__()
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.Shape.NoFrame)

        self.container = QWidget()
        self.layout = QVBoxLayout(self.container)
        self.layout.setContentsMargins(28, 24, 28, 28)
        self.layout.setSpacing(14)
        self.layout.addStretch(1)
        self.setWidget(self.container)

        self.empty_state = QLabel(
            "Start a conversation\n\nYour messages stay local and go directly to Ollama."
        )
        self.empty_state.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_state.setStyleSheet("color: #778492; font-size: 15px; line-height: 1.5;")
        self.layout.insertWidget(0, self.empty_state)

    def add_message(self, role: str, content: str = "") -> MessageBubble:
        if self.empty_state is not None:
            self.empty_state.deleteLater()
            self.empty_state = None
        bubble = MessageBubble(role, content)
        self.layout.insertWidget(self.layout.count() - 1, bubble)
        self._scroll_to_bottom()
        return bubble

    def clear_messages(self) -> None:
        while self.layout.count() > 1:
            item = self.layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self.empty_state = QLabel(
            "Start a conversation\n\nYour messages stay local and go directly to Ollama."
        )
        self.empty_state.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.empty_state.setStyleSheet("color: #778492; font-size: 15px;")
        self.layout.insertWidget(0, self.empty_state)

    def set_messages(self, messages: list[ChatMessage]) -> None:
        self.clear_messages()
        for message in messages:
            self.add_message(message.role, message.content)

    def _scroll_to_bottom(self) -> None:
        scrollbar = self.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
