"""Main Phase 1 chat window."""

from __future__ import annotations

from PySide6.QtCore import QObject, QThread, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from ..assistant_core import AssistantService
from ..config import AppConfig
from ..conversations import Conversation, ConversationStore
from ..ollama import ChatMessage, OllamaClient
from ..workers import ChatWorker, ConnectionWorker
from .chat_view import ChatView, MessageBubble
from .settings_dialog import SettingsDialog


class MainWindow(QMainWindow):
    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.config = config
        self.client = OllamaClient(config.ollama_url)
        self.service = AssistantService(self.client)
        self.conversation_store = ConversationStore()
        self.conversations = self.conversation_store.load()
        if not self.conversations:
            self.conversations = [Conversation.create()]
        self.current_conversation = self.conversations[0]
        self.messages: list[ChatMessage] = list(self.current_conversation.messages)
        self.chat_thread: QThread | None = None
        self.chat_worker: ChatWorker | None = None
        self.connection_thread: QThread | None = None
        self.connection_worker: ConnectionWorker | None = None
        self.active_assistant_bubble: MessageBubble | None = None
        self.active_response = ""
        self.available_models: list[str] = []

        self.setWindowTitle("Local AI Assistant")
        self.resize(1040, 760)
        self.setMinimumSize(700, 540)
        self._build_ui()
        self._populate_conversations()
        self.chat_view.set_messages(self.messages)
        self._persist_conversations()
        self._refresh_connection()

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        top_bar = QFrame()
        top_bar.setObjectName("topBar")
        top_layout = QHBoxLayout(top_bar)
        top_layout.setContentsMargins(24, 16, 24, 16)

        identity = QVBoxLayout()
        identity.setSpacing(2)
        mark = QLabel("Local AI Assistant")
        mark.setObjectName("appMark")
        subtitle = QLabel("Private chat through Ollama")
        subtitle.setObjectName("appSubtitle")
        identity.addWidget(mark)
        identity.addWidget(subtitle)
        top_layout.addLayout(identity)
        new_chat_button = QPushButton("New chat")
        new_chat_button.setObjectName("newChatButton")
        new_chat_button.clicked.connect(self._new_chat)
        top_layout.addWidget(new_chat_button)
        top_layout.addStretch(1)

        self.status_label = QLabel("Checking Ollama")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setProperty("status", "checking")
        top_layout.addWidget(self.status_label)

        self.model_selector = QComboBox()
        self.model_selector.setMinimumWidth(190)
        self.model_selector.setToolTip("Model used for the next message")
        self.model_selector.addItem(self.config.model)
        top_layout.addWidget(self.model_selector)

        settings_button = QPushButton("Settings")
        settings_button.setObjectName("settingsButton")
        settings_button.clicked.connect(self._open_settings)
        top_layout.addWidget(settings_button)
        root_layout.addWidget(top_bar)

        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)

        history_panel = QFrame()
        history_panel.setObjectName("historyPanel")
        history_panel.setMinimumWidth(210)
        history_panel.setMaximumWidth(280)
        history_layout = QVBoxLayout(history_panel)
        history_layout.setContentsMargins(14, 18, 10, 14)
        history_layout.setSpacing(10)
        history_label = QLabel("Chats")
        history_label.setObjectName("sectionLabel")
        history_layout.addWidget(history_label)
        self.history_list = QListWidget()
        self.history_list.setObjectName("historyList")
        self.history_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.history_list.currentItemChanged.connect(self._conversation_selected)
        history_layout.addWidget(self.history_list, 1)
        body_layout.addWidget(history_panel)

        chat_column = QWidget()
        chat_column_layout = QVBoxLayout(chat_column)
        chat_column_layout.setContentsMargins(0, 0, 0, 0)
        chat_column_layout.setSpacing(0)
        self.chat_view = ChatView()
        chat_column_layout.addWidget(self.chat_view, 1)

        composer = QFrame()
        composer_layout = QVBoxLayout(composer)
        composer_layout.setContentsMargins(24, 10, 24, 22)
        composer_layout.setSpacing(10)
        hint = QLabel("Messages are sent directly to your configured Ollama server.")
        hint.setStyleSheet("color: #72808d; font-size: 12px;")
        composer_layout.addWidget(hint)

        input_row = QHBoxLayout()
        input_row.setSpacing(10)
        self.message_input = QLineEdit()
        self.message_input.setPlaceholderText("Message your local assistant…")
        self.message_input.setClearButtonEnabled(True)
        self.message_input.returnPressed.connect(self._send_message)
        input_row.addWidget(self.message_input, 1)

        self.stop_button = QPushButton("Stop")
        self.stop_button.setObjectName("stopButton")
        self.stop_button.setEnabled(False)
        self.stop_button.clicked.connect(self._stop_generation)
        input_row.addWidget(self.stop_button)

        self.send_button = QPushButton("Send")
        self.send_button.setDefault(True)
        self.send_button.clicked.connect(self._send_message)
        input_row.addWidget(self.send_button)
        composer_layout.addLayout(input_row)
        chat_column_layout.addWidget(composer)
        body_layout.addWidget(chat_column, 1)
        root_layout.addWidget(body, 1)

        self.setCentralWidget(root)

    @Slot()
    def _send_message(self) -> None:
        prompt = self.message_input.text().strip()
        if not prompt or self.chat_worker is not None:
            return
        model = self.model_selector.currentText().strip() or self.config.model
        self.messages.append(ChatMessage("user", prompt))
        self._persist_current_conversation()
        self.chat_view.add_message("user", prompt)
        self.message_input.clear()
        self.active_response = ""
        self.active_assistant_bubble = self.chat_view.add_message("assistant", "")
        self._set_generating(True)

        self.chat_thread = QThread(self)
        self.chat_worker = ChatWorker(self.service, list(self.messages), model)
        self.chat_worker.moveToThread(self.chat_thread)
        self.chat_thread.started.connect(self.chat_worker.run)
        self.chat_worker.chunk.connect(self._append_assistant_chunk)
        self.chat_worker.finished.connect(self._chat_finished)
        self.chat_worker.failed.connect(self._chat_failed)
        self.chat_worker.finished.connect(self.chat_thread.quit)
        self.chat_worker.failed.connect(self.chat_thread.quit)
        self.chat_thread.finished.connect(self._chat_thread_finished)
        self.chat_thread.start()
        self.status_label.setText(f"Generating with {model}")
        self._set_status("connected")

    @Slot(str)
    def _append_assistant_chunk(self, chunk: str) -> None:
        self.active_response += chunk
        if self.active_assistant_bubble:
            self.active_assistant_bubble.set_content(self.active_response)
            self.chat_view.verticalScrollBar().setValue(self.chat_view.verticalScrollBar().maximum())

    @Slot(str)
    def _chat_finished(self, response: str) -> None:
        self.messages.append(ChatMessage("assistant", response))
        self._persist_current_conversation()
        self.status_label.setText(f"Connected · {self.model_selector.currentText()}")
        self._set_generating(False)

    @Slot(str, str)
    def _chat_failed(self, message: str, kind: str) -> None:
        if kind == "cancelled":
            if self.active_assistant_bubble and self.active_response:
                self.active_assistant_bubble.set_content(self.active_response + "\n\n*Generation stopped.*")
            elif self.active_assistant_bubble:
                self.active_assistant_bubble.set_content("*Generation stopped.*")
        else:
            if self.active_assistant_bubble:
                self.active_assistant_bubble.set_content(message)
            self._set_status("error")
            self.status_label.setText("Ollama error")
        self._persist_current_conversation()
        self._set_generating(False)

    def _chat_thread_finished(self) -> None:
        if self.chat_worker:
            self.chat_worker.deleteLater()
        if self.chat_thread:
            self.chat_thread.deleteLater()
        self.chat_worker = None
        self.chat_thread = None
        self.active_assistant_bubble = None
        self.active_response = ""

    def _set_generating(self, generating: bool) -> None:
        self.send_button.setEnabled(not generating)
        self.stop_button.setEnabled(generating)
        self.message_input.setEnabled(not generating)
        self.model_selector.setEnabled(not generating)
        self.history_list.setEnabled(not generating)

    @Slot()
    def _stop_generation(self) -> None:
        if self.chat_worker:
            self.chat_worker.cancel()

    def _refresh_connection(self) -> None:
        self.status_label.setText("Checking Ollama")
        self._set_status("checking")
        self.connection_thread = QThread(self)
        self.connection_worker = ConnectionWorker(self.client)
        self.connection_worker.moveToThread(self.connection_thread)
        self.connection_thread.started.connect(self.connection_worker.run)
        self.connection_worker.succeeded.connect(self._connection_succeeded)
        self.connection_worker.failed.connect(self._connection_failed)
        self.connection_worker.succeeded.connect(self.connection_thread.quit)
        self.connection_worker.failed.connect(self.connection_thread.quit)
        self.connection_thread.finished.connect(self._connection_thread_finished)
        self.connection_thread.start()

    @Slot(list)
    def _connection_succeeded(self, models: list[str]) -> None:
        self.available_models = models
        current = self.model_selector.currentText() or self.config.model
        self.model_selector.blockSignals(True)
        self.model_selector.clear()
        self.model_selector.addItems(models)
        if current not in models:
            self.model_selector.insertItem(0, current)
        self.model_selector.setCurrentText(current)
        self.model_selector.blockSignals(False)
        self.status_label.setText(f"Connected · {current}")
        self._set_status("connected")

    @Slot(str)
    def _connection_failed(self, message: str) -> None:
        self.status_label.setText("Ollama unavailable")
        self.status_label.setToolTip(message)
        self._set_status("error")

    def _connection_thread_finished(self) -> None:
        if self.connection_worker:
            self.connection_worker.deleteLater()
        if self.connection_thread:
            self.connection_thread.deleteLater()
        self.connection_worker = None
        self.connection_thread = None

    @Slot()
    def _open_settings(self) -> None:
        if self.chat_worker is not None:
            return
        dialog = SettingsDialog(self.config, self)
        if dialog.exec() != SettingsDialog.DialogCode.Accepted:
            return
        self.config = dialog.config()
        self.config.save()
        self.client = OllamaClient(self.config.ollama_url)
        self.service = AssistantService(self.client)
        self.model_selector.clear()
        self.model_selector.addItem(self.config.model)
        self._refresh_connection()

    @Slot()
    def _new_chat(self) -> None:
        if self.chat_worker is not None:
            return
        if not self.messages and self.current_conversation.title == "New chat":
            self.message_input.setFocus()
            return
        self.current_conversation = Conversation.create()
        self.conversations.insert(0, self.current_conversation)
        self.messages = []
        self.chat_view.clear_messages()
        self._persist_conversations()
        self._populate_conversations()
        self.message_input.setFocus()

    @Slot(QListWidgetItem, QListWidgetItem)
    def _conversation_selected(
        self, current: QListWidgetItem | None, previous: QListWidgetItem | None
    ) -> None:
        if current is None:
            return
        conversation_id = current.data(Qt.ItemDataRole.UserRole)
        if conversation_id == self.current_conversation.id:
            return
        selected = next(
            (conversation for conversation in self.conversations if conversation.id == conversation_id),
            None,
        )
        if selected is None:
            return
        self.current_conversation = selected
        self.messages = list(selected.messages)
        self._persist_conversations()
        self.chat_view.set_messages(self.messages)
        self.status_label.setText(f"Connected · {self.model_selector.currentText()}")
        self.message_input.setFocus()

    def _persist_current_conversation(self) -> None:
        self.current_conversation.update_messages(self.messages)
        self._persist_conversations()
        self._populate_conversations()

    def _persist_conversations(self) -> None:
        if self.current_conversation in self.conversations:
            self.conversations.remove(self.current_conversation)
        self.conversations.insert(0, self.current_conversation)
        self.conversation_store.save(self.conversations)

    def _populate_conversations(self) -> None:
        selected_id = self.current_conversation.id
        self.history_list.blockSignals(True)
        self.history_list.clear()
        for conversation in self.conversations:
            item = QListWidgetItem(conversation.title)
            item.setData(Qt.ItemDataRole.UserRole, conversation.id)
            item.setToolTip(
                f"{len(conversation.messages)} messages"
                if conversation.messages
                else "No messages yet"
            )
            self.history_list.addItem(item)
            if conversation.id == selected_id:
                self.history_list.setCurrentItem(item)
        self.history_list.blockSignals(False)

    def _set_status(self, status: str) -> None:
        self.status_label.setProperty("status", status)
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)

    def closeEvent(self, event) -> None:
        if self.chat_worker:
            self.chat_worker.cancel()
        event.accept()
