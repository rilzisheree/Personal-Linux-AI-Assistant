"""Main Lura chat window."""

from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QDateTime, QObject, QThread, QTimer, Qt, Signal, Slot
from PySide6.QtGui import QAction, QColor, QIcon, QKeySequence, QPainter, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFileDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMenu,
    QMessageBox,
    QPushButton,
    QSystemTrayIcon,
    QVBoxLayout,
    QWidget,
)

from ..assistant_core import AssistantService
from ..api import ApiServer, start_background_server
from ..config import AppConfig
from ..conversations import Conversation, ConversationStore
from ..desktop_integration import set_autostart_enabled
from ..ollama import ChatMessage, OllamaClient
from ..telegram_bot import (
    TelegramConfig,
    load_telegram_token,
    save_telegram_token,
)
from ..tools import PermissionLevel, ToolManager
from ..voice import VoiceService
from ..workers import (
    ChatWorker,
    ConnectionWorker,
    SpeechWorker,
    TelegramBotWorker,
    VoiceRecordWorker,
    VoiceTranscriptionWorker,
)
from .chat_view import ChatView, MessageBubble
from .core_widget import CoreWidget
from .settings_dialog import SettingsDialog


class MainWindow(QMainWindow):
    def __init__(self, config: AppConfig) -> None:
        super().__init__()
        self.config = config
        self.client = OllamaClient(config.ollama_url)
        self.service = AssistantService(self.client)
        self.voice_service = VoiceService(config)
        self.tool_manager = ToolManager()
        self.api_server: ApiServer | None = None
        self.api_thread = None
        self.telegram_thread: QThread | None = None
        self.telegram_worker: TelegramBotWorker | None = None
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
        self.voice_record_thread: QThread | None = None
        self.voice_record_worker: VoiceRecordWorker | None = None
        self.voice_transcription_thread: QThread | None = None
        self.voice_transcription_worker: VoiceTranscriptionWorker | None = None
        self.speech_thread: QThread | None = None
        self.speech_worker: SpeechWorker | None = None
        self.last_voice_error: str | None = None
        self.active_assistant_bubble: MessageBubble | None = None
        self.active_response = ""
        self.active_tool_bubbles: dict[str, MessageBubble] = {}
        self.available_models: list[str] = []
        self._quitting = False
        self._focus_mode = False
        self.tray_icon: QSystemTrayIcon | None = None

        self.setWindowTitle("Lura")
        self.resize(1280, 760)
        self.setMinimumSize(900, 580)
        self._build_ui()
        self._setup_tray()
        self._sync_quit_behavior()
        self.api_server, self.api_thread = start_background_server(
            config.ollama_url,
            config.model,
            config.ollama_context_size,
        )
        self._populate_conversations()
        self.chat_view.set_messages(self.messages)
        self._persist_conversations()
        self._refresh_connection()
        self._start_telegram_bot()

    def _setup_tray(self) -> None:
        if not QSystemTrayIcon.isSystemTrayAvailable():
            return

        self.tray_icon = QSystemTrayIcon(self._tray_icon(), self)
        self.tray_icon.setToolTip("Lura — local AI assistant")
        menu = QMenu(self)

        show_action = QAction("Show Lura", self)
        show_action.triggered.connect(self._show_from_tray)
        menu.addAction(show_action)

        hide_action = QAction("Hide Lura", self)
        hide_action.triggered.connect(self.hide_to_tray)
        menu.addAction(hide_action)
        menu.addSeparator()

        quit_action = QAction("Quit Lura", self)
        quit_action.triggered.connect(self._quit_from_tray)
        menu.addAction(quit_action)

        self.tray_icon.setContextMenu(menu)
        self.tray_icon.activated.connect(self._tray_activated)
        self.tray_icon.show()

    def _sync_quit_behavior(self) -> None:
        app = QApplication.instance()
        if app is not None:
            app.setQuitOnLastWindowClosed(
                not (self.config.background_mode_enabled and self.tray_icon is not None)
            )

    @staticmethod
    def _tray_icon() -> QIcon:
        pixmap = QPixmap(32, 32)
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setBrush(QColor("#a974ff"))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(2, 2, 28, 28)
        painter.setPen(QColor("#0a0714"))
        painter.setFont(painter.font())
        painter.drawText(pixmap.rect(), Qt.AlignmentFlag.AlignCenter, "L")
        painter.end()
        return QIcon(pixmap)

    @Slot()
    def _show_from_tray(self) -> None:
        self.show()
        self.raise_()
        self.activateWindow()

    @Slot()
    def hide_to_tray(self, silent: bool = False) -> None:
        if self.tray_icon is None:
            self.show()
            if not silent:
                self.status_label.setText("System tray unavailable")
            return
        self.hide()
        if not silent:
            self.tray_icon.showMessage(
                "Lura is still running",
                "The assistant is available from the system tray.",
                QSystemTrayIcon.MessageIcon.Information,
                2500,
            )

    @Slot()
    def _quit_from_tray(self) -> None:
        self._quitting = True
        self.close()
        app = QApplication.instance()
        if app is not None:
            app.quit()

    @Slot(QSystemTrayIcon.ActivationReason)
    def _tray_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        if reason in {
            QSystemTrayIcon.ActivationReason.Trigger,
            QSystemTrayIcon.ActivationReason.DoubleClick,
        }:
            if self.isVisible():
                self.hide_to_tray()
            else:
                self._show_from_tray()

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("appRoot")
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        self.top_bar = QFrame()
        self.top_bar.setObjectName("topBar")
        top_layout = QHBoxLayout(self.top_bar)
        top_layout.setContentsMargins(24, 16, 24, 16)
        top_layout.setSpacing(14)

        identity = QVBoxLayout()
        identity.setSpacing(1)
        mark = QLabel("LURA")
        mark.setObjectName("appMark")
        subtitle = QLabel("LOCAL INTELLIGENCE")
        subtitle.setObjectName("appSubtitle")
        identity.addWidget(mark)
        identity.addWidget(subtitle)
        top_layout.addLayout(identity)
        top_layout.addStretch(1)

        self.status_label = QLabel("Checking Ollama")
        self.status_label.setObjectName("statusLabel")
        self.status_label.setProperty("status", "checking")
        top_layout.addWidget(self.status_label)

        self.telegram_status_label = QLabel("TELEGRAM OFF")
        self.telegram_status_label.setObjectName("telegramStatusLabel")
        self.telegram_status_label.setProperty("status", "disabled")
        self.telegram_status_label.setToolTip(
            "Telegram phone control is disabled in Settings."
        )
        top_layout.addWidget(self.telegram_status_label)

        self.model_selector = QComboBox()
        self.model_selector.setObjectName("modelSelector")
        self.model_selector.setMinimumWidth(150)
        self.model_selector.setToolTip("Model used for the next message")
        self.model_selector.addItem(self.config.model)
        top_layout.addWidget(self.model_selector)

        self.history_list = QComboBox()
        self.history_list.setObjectName("sessionSelector")
        self.history_list.setMinimumWidth(160)
        self.history_list.setToolTip("Switch conversation")
        self.history_list.currentIndexChanged.connect(self._conversation_selected)
        top_layout.addWidget(self.history_list)

        new_chat_button = QPushButton("+")
        new_chat_button.setObjectName("topAction")
        new_chat_button.setToolTip("New conversation")
        new_chat_button.clicked.connect(self._new_chat)
        top_layout.addWidget(new_chat_button)

        settings_button = QPushButton("···")
        settings_button.setObjectName("settingsButton")
        settings_button.setToolTip("Settings")
        settings_button.clicked.connect(self._open_settings)
        top_layout.addWidget(settings_button)
        self.focus_button = QPushButton("◌")
        self.focus_button.setObjectName("focusButton")
        self.focus_button.setToolTip("Focus mode — show only the orb")
        self.focus_button.clicked.connect(self._toggle_focus_mode)
        top_layout.insertWidget(top_layout.count() - 1, self.focus_button)
        root_layout.addWidget(self.top_bar)

        stage = QFrame()
        stage.setObjectName("mainStage")
        stage_layout = QVBoxLayout(stage)
        stage_layout.setContentsMargins(24, 20, 24, 18)
        stage_layout.setSpacing(10)
        stage_layout.addStretch(1)

        self.core_widget = CoreWidget()
        self.core_widget.setFixedSize(310, 310)
        self.core_widget.pressed.connect(self._start_recording)
        self.core_widget.released.connect(self._stop_recording)
        stage_layout.addWidget(self.core_widget, 0, Qt.AlignmentFlag.AlignCenter)

        self.core_name = QLabel("LURA")
        self.core_name.setObjectName("coreName")
        self.core_name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        stage_layout.addWidget(self.core_name)

        self.core_status = QLabel("READY // HOLD TO SPEAK")
        self.core_status.setObjectName("coreStatus")
        self.core_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        stage_layout.addWidget(self.core_status)

        self.core_quote = QLabel("A quiet channel.\nAsk Lura anything.")
        self.core_quote.setObjectName("coreQuote")
        self.core_quote.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.core_quote.setWordWrap(True)
        self.core_quote.hide()
        stage_layout.addWidget(self.core_quote)

        self.transcript_header = QWidget()
        transcript_header = QHBoxLayout(self.transcript_header)
        transcript_header.setContentsMargins(0, 14, 0, 0)
        transcript_header.setSpacing(12)
        transcript_label = QLabel("CONVERSATION")
        transcript_label.setObjectName("sectionLabel")
        transcript_header.addWidget(transcript_label)
        transcript_header.addStretch(1)
        clear_button = QPushButton("Clear")
        clear_button.setObjectName("quietButton")
        clear_button.clicked.connect(self._clear_current_conversation)
        transcript_header.addWidget(clear_button)
        export_button = QPushButton("Export")
        export_button.setObjectName("quietButton")
        export_button.clicked.connect(self._export_current_conversation)
        transcript_header.addWidget(export_button)
        stage_layout.addWidget(self.transcript_header)

        self.chat_view = ChatView()
        self.chat_view.setObjectName("transcript")
        self.chat_view.setMinimumHeight(150)
        self.chat_view.setMaximumWidth(820)
        stage_layout.addWidget(self.chat_view, 1, Qt.AlignmentFlag.AlignHCenter)

        self.composer = QFrame()
        self.composer.setObjectName("composer")
        composer_layout = QVBoxLayout(self.composer)
        composer_layout.setContentsMargins(0, 10, 0, 0)
        composer_layout.setSpacing(7)
        self.composer.setMaximumWidth(820)

        input_row = QHBoxLayout()
        input_row.setSpacing(10)
        self.message_input = QLineEdit()
        self.message_input.setObjectName("messageInput")
        self.message_input.setPlaceholderText("Ask Lura anything…")
        self.message_input.setClearButtonEnabled(True)
        self.message_input.returnPressed.connect(self._send_message)
        input_row.addWidget(self.message_input, 1)

        # Keep the existing worker-facing attribute, but make the old control
        # invisible. The orb now owns the same press/release recording signals.
        self.mic_button = QPushButton(root)
        self.mic_button.setVisible(False)
        self.mic_button.setObjectName("micButton")

        self.stop_button = QPushButton("×")
        self.stop_button.setObjectName("stopButton")
        self.stop_button.setToolTip("Stop generation")
        self.stop_button.setEnabled(False)
        self.stop_button.setVisible(False)
        self.stop_button.clicked.connect(self._stop_generation)
        input_row.addWidget(self.stop_button)

        self.send_button = QPushButton("↗")
        self.send_button.setObjectName("sendButton")
        self.send_button.setToolTip("Send message")
        self.send_button.setDefault(True)
        self.send_button.clicked.connect(self._send_message)
        input_row.addWidget(self.send_button)
        composer_layout.addLayout(input_row)

        self.composer_hint = QLabel("DIRECT LOCAL CHANNEL // NO CLOUD ROUTING")
        self.composer_hint.setObjectName("composerHint")
        composer_layout.addWidget(self.composer_hint)
        stage_layout.addWidget(self.composer, 0, Qt.AlignmentFlag.AlignHCenter)
        stage_layout.addStretch(1)

        root_layout.addWidget(stage, 1)

        self.setCentralWidget(root)

        self.clock_label = QLabel()
        self.clock_label.setObjectName("topMeta")
        self.clock_timer = QTimer(self)
        self.clock_timer.timeout.connect(self._update_clock)
        self.clock_timer.start(1000)
        self._update_clock()

        self.runtime_status_value = QLabel("CHECKING")
        self.runtime_status_value.setObjectName("runtimeStatusValue")
        self.runtime_status_value.hide()
        self._focus_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Escape), self)
        self._focus_shortcut.setContext(Qt.ShortcutContext.WindowShortcut)
        self._focus_shortcut.activated.connect(self._exit_focus_mode)

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
        self._set_orb_state("thinking")
        self._set_generating(True)

        self.chat_thread = QThread(self)
        self.chat_worker = ChatWorker(
            self.service,
            list(self.messages),
            model,
            self.tool_manager,
            self.config.ollama_context_size,
        )
        self.chat_worker.moveToThread(self.chat_thread)
        self.chat_thread.started.connect(self.chat_worker.run)
        self.chat_worker.chunk.connect(self._append_assistant_chunk)
        self.chat_worker.conversation_ready.connect(self._conversation_ready)
        self.chat_worker.tool_started.connect(self._tool_started)
        self.chat_worker.tool_requested.connect(self._tool_confirmation_requested)
        self.chat_worker.tool_completed.connect(self._tool_completed)
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

    @Slot(object)
    def _conversation_ready(self, messages: object) -> None:
        if isinstance(messages, list) and all(isinstance(message, ChatMessage) for message in messages):
            self.messages = list(messages)
            self._persist_current_conversation()

    @Slot(str, str, object, str)
    def _tool_started(self, call_id: str, name: str, arguments: object, permission: str) -> None:
        details = json.dumps(arguments, indent=2, sort_keys=True) if isinstance(arguments, dict) else str(arguments)
        if permission == PermissionLevel.SAFE.value:
            content = f"Running {name}…"
        else:
            content = f"⚠ Approval required for {name}\n\n{details}"
        self.active_tool_bubbles[call_id] = self.chat_view.add_message("tool", content)

    @Slot(str, str, object, str)
    def _tool_confirmation_requested(
        self, call_id: str, name: str, arguments: object, permission: str
    ) -> None:
        if not self.chat_worker:
            return
        details = json.dumps(arguments, indent=2, sort_keys=True) if isinstance(arguments, dict) else str(arguments)
        label = "dangerous" if permission == PermissionLevel.DANGEROUS.value else "confirmation-required"
        dialog = QMessageBox(self)
        dialog.setIcon(QMessageBox.Icon.Warning)
        dialog.setWindowTitle("Permission required")
        dialog.setText(f"Lura wants to run a {label} tool: {name}")
        dialog.setInformativeText(f"Arguments:\n{details}\n\nAllow this action?")
        allow_button = dialog.addButton("Allow", QMessageBox.ButtonRole.AcceptRole)
        dialog.addButton("Cancel", QMessageBox.ButtonRole.RejectRole)
        dialog.exec()
        approved = dialog.clickedButton() is allow_button
        if self.chat_worker:
            self.chat_worker.resolve_tool_call(call_id, approved)

    @Slot(str, str, str, bool, object)
    def _tool_completed(
        self,
        call_id: str,
        name: str,
        result: str,
        success: bool,
        images: object,
    ) -> None:
        bubble = self.active_tool_bubbles.pop(call_id, None)
        prefix = "✓" if success else "✕"
        content = f"{prefix} {name}\n\n{result}"
        image_paths = (
            tuple(path for path in images if isinstance(path, str))
            if isinstance(images, (tuple, list))
            else ()
        )
        if bubble:
            bubble.set_content(content)
            bubble.add_images(image_paths)
        else:
            self.chat_view.add_message("tool", content, image_paths)

    @Slot()
    def _start_recording(self) -> None:
        if (
            not self.config.voice_input_enabled
            or self.chat_worker is not None
            or self.voice_record_worker is not None
            or self.voice_transcription_worker is not None
        ):
            self._set_orb_state("idle")
            return
        self.last_voice_error = None
        self._set_orb_state("listening")
        destination = self.voice_service.new_recording_path()
        self.voice_record_thread = QThread(self)
        self.voice_record_worker = VoiceRecordWorker(self.voice_service, destination)
        self.voice_record_worker.moveToThread(self.voice_record_thread)
        self.voice_record_thread.started.connect(self.voice_record_worker.run)
        self.voice_record_worker.started.connect(self._recording_started)
        self.voice_record_worker.finished.connect(self._recording_finished)
        self.voice_record_worker.failed.connect(self._recording_failed)
        self.voice_record_worker.finished.connect(self.voice_record_thread.quit)
        self.voice_record_worker.failed.connect(self.voice_record_thread.quit)
        self.voice_record_thread.finished.connect(self._recording_thread_finished)
        self.voice_record_thread.start()
        self._set_voice_status("STARTING MICROPHONE…")
        self.message_input.setEnabled(False)
        self.send_button.setEnabled(False)

    @Slot()
    def _recording_started(self) -> None:
        self._set_orb_state("listening")
        self.mic_button.setText("STOP")
        self.mic_button.setEnabled(True)
        self.mic_button.setProperty("recording", True)
        self.mic_button.style().unpolish(self.mic_button)
        self.mic_button.style().polish(self.mic_button)
        self._set_voice_status("RECORDING // RELEASE TO TRANSCRIBE")

    @Slot()
    def _stop_recording(self) -> None:
        if self.voice_record_worker is not None:
            self._set_orb_state("thinking")
            self.mic_button.setEnabled(False)
            self.voice_record_worker.stop()
            self._set_voice_status("PROCESSING LOCAL AUDIO…")
        else:
            self._set_orb_state("idle")

    @Slot(str)
    def _recording_finished(self, audio_path: str) -> None:
        self._set_orb_state("thinking")
        self._set_voice_status("TRANSCRIBING WITH LOCAL WHISPER…")
        self.voice_transcription_thread = QThread(self)
        self.voice_transcription_worker = VoiceTranscriptionWorker(
            self.voice_service, Path(audio_path)
        )
        self.voice_transcription_worker.moveToThread(self.voice_transcription_thread)
        self.voice_transcription_thread.started.connect(
            self.voice_transcription_worker.run
        )
        self.voice_transcription_worker.finished.connect(self._transcription_finished)
        self.voice_transcription_worker.failed.connect(self._transcription_failed)
        self.voice_transcription_worker.finished.connect(
            self.voice_transcription_thread.quit
        )
        self.voice_transcription_worker.failed.connect(
            self.voice_transcription_thread.quit
        )
        self.voice_transcription_thread.finished.connect(
            self._transcription_thread_finished
        )
        self.voice_transcription_thread.start()

    @Slot(str)
    def _recording_failed(self, message: str) -> None:
        self.last_voice_error = message
        self._set_orb_state("error")
        self._set_voice_idle()
        self._set_voice_status(f"VOICE ERROR // {message[:180]}")
        self._set_status("error")
        self.status_label.setText("Voice input unavailable")
        self.status_label.setToolTip(message)

    def _recording_thread_finished(self) -> None:
        if self.voice_record_worker:
            self.voice_record_worker.deleteLater()
        if self.voice_record_thread:
            self.voice_record_thread.deleteLater()
        self.voice_record_worker = None
        self.voice_record_thread = None
        if self.voice_transcription_worker is None:
            self._set_voice_idle()
            if self.last_voice_error:
                self._set_voice_status(f"VOICE ERROR // {self.last_voice_error[:180]}")

    @Slot(str, str)
    def _transcription_finished(self, text: str, audio_path: str) -> None:
        self._remove_recording(audio_path)
        self._set_voice_idle()
        self.message_input.setText(text)
        self._send_message()

    @Slot(str, str)
    def _transcription_failed(self, message: str, audio_path: str) -> None:
        self._remove_recording(audio_path)
        self._set_voice_idle()
        self._set_status("error")
        self.status_label.setText("Voice transcription failed")
        self.status_label.setToolTip(message)

    def _transcription_thread_finished(self) -> None:
        if self.voice_transcription_worker:
            self.voice_transcription_worker.deleteLater()
        if self.voice_transcription_thread:
            self.voice_transcription_thread.deleteLater()
        self.voice_transcription_worker = None
        self.voice_transcription_thread = None

    @staticmethod
    def _remove_recording(audio_path: str) -> None:
        try:
            Path(audio_path).unlink()
        except OSError:
            pass

    def _set_voice_idle(self) -> None:
        self.mic_button.setText("MIC")
        self.mic_button.setProperty("recording", False)
        self.mic_button.style().unpolish(self.mic_button)
        self.mic_button.style().polish(self.mic_button)
        self.mic_button.setEnabled(
            self.config.voice_input_enabled
            and self.chat_worker is None
            and self.voice_record_worker is None
            and self.voice_transcription_worker is None
        )
        self.message_input.setEnabled(self.chat_worker is None)
        self.send_button.setEnabled(self.chat_worker is None)
        self._set_voice_status("DIRECT LOCAL CHANNEL // NO CLOUD ROUTING")
        if self.chat_worker is None and self.speech_worker is None:
            self._set_orb_state("idle")

    def _set_voice_status(self, message: str) -> None:
        self.composer_hint.setText(message)

    def _speak_response(self, response: str) -> None:
        if (
            not self.config.voice_responses_enabled
            or self.config.tts_engine == "disabled"
            or not response.strip()
            or self.speech_worker is not None
        ):
            self._set_orb_state("idle")
            return
        self._set_orb_state("speaking")
        self.speech_thread = QThread(self)
        self.speech_worker = SpeechWorker(self.voice_service, response)
        self.speech_worker.moveToThread(self.speech_thread)
        self.speech_thread.started.connect(self.speech_worker.run)
        self.speech_worker.finished.connect(self._speech_finished)
        self.speech_worker.failed.connect(self._speech_failed)
        self.speech_worker.finished.connect(self.speech_thread.quit)
        self.speech_worker.failed.connect(self.speech_thread.quit)
        self.speech_thread.finished.connect(self._speech_thread_finished)
        self.speech_thread.start()
        self._set_voice_status("SPEAKING LOCAL RESPONSE…")

    @Slot()
    def _speech_finished(self) -> None:
        self._set_orb_state("idle")
        self._set_voice_status("DIRECT LOCAL CHANNEL // NO CLOUD ROUTING")

    @Slot(str)
    def _speech_failed(self, message: str) -> None:
        self._set_orb_state("error")
        self._set_voice_status(f"VOICE ERROR // {message[:220]}")
        self.status_label.setText("Voice output unavailable")
        self.status_label.setToolTip(message)

    def _speech_thread_finished(self) -> None:
        if self.speech_worker:
            self.speech_worker.deleteLater()
        if self.speech_thread:
            self.speech_thread.deleteLater()
        self.speech_worker = None
        self.speech_thread = None

    @Slot(str)
    def _chat_finished(self, response: str) -> None:
        self.status_label.setText(f"Connected · {self.model_selector.currentText()}")
        self._set_generating(False)
        self._speak_response(response)

    @Slot(str, str)
    def _chat_failed(self, message: str, kind: str) -> None:
        self._set_orb_state("idle" if kind == "cancelled" else "error")
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
        self._set_voice_idle()

    def _set_generating(self, generating: bool) -> None:
        if generating:
            self._set_orb_state("thinking")
        self.send_button.setEnabled(not generating)
        self.stop_button.setEnabled(generating)
        self.stop_button.setVisible(generating)
        self.message_input.setEnabled(not generating)
        self.model_selector.setEnabled(not generating)
        self.history_list.setEnabled(not generating)
        self.mic_button.setEnabled(
            self.config.voice_input_enabled
            and not generating
            and self.chat_worker is None
            and self.voice_record_worker is None
            and self.voice_transcription_worker is None
        )

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
        self.runtime_status_value.setText("ONLINE")
        self._set_status("connected")

    @Slot(str)
    def _connection_failed(self, message: str) -> None:
        self.status_label.setText("Ollama unavailable")
        self.status_label.setToolTip(message)
        self.runtime_status_value.setText("OFFLINE")
        self._set_status("error")
        if self.chat_worker is None:
            self._set_orb_state("error")

    def _connection_thread_finished(self) -> None:
        if self.connection_worker:
            self.connection_worker.deleteLater()
        if self.connection_thread:
            self.connection_thread.deleteLater()
        self.connection_worker = None
        self.connection_thread = None

    def _start_telegram_bot(self) -> None:
        if not self.config.telegram_enabled:
            self._set_telegram_badge(
                "TELEGRAM OFF",
                "disabled",
                "Telegram phone control is disabled in Settings.",
            )
            return
        if self.telegram_worker is not None:
            return
        token = load_telegram_token()
        if not token:
            self._set_telegram_badge(
                "TELEGRAM SETUP",
                "error",
                "Open Settings and add your BotFather token.",
            )
            return
        try:
            telegram_config = TelegramConfig(
                token=token,
                allowed_user_id=int(self.config.telegram_allowed_user_id),
                ollama_url=self.config.ollama_url,
                model=self.config.model,
                context_size=self.config.ollama_context_size,
            )
        except (TypeError, ValueError) as error:
            self._set_telegram_badge(
                "TELEGRAM ERROR",
                "error",
                f"Telegram configuration error: {error}",
            )
            return

        self._set_telegram_badge(
            "TELEGRAM CONNECTING",
            "checking",
            "Connecting to Telegram…",
        )
        self.telegram_thread = QThread(self)
        self.telegram_worker = TelegramBotWorker(telegram_config)
        self.telegram_worker.moveToThread(self.telegram_thread)
        self.telegram_thread.started.connect(self.telegram_worker.run)
        self.telegram_worker.connected.connect(self._telegram_connected)
        self.telegram_worker.status.connect(self._telegram_status)
        self.telegram_worker.failed.connect(self._telegram_failed)
        self.telegram_worker.stopped.connect(self.telegram_thread.quit)
        self.telegram_thread.finished.connect(self._telegram_thread_finished)
        self.telegram_thread.start()

    @Slot(str)
    def _telegram_connected(self, username: str) -> None:
        self._set_telegram_badge(
            "TELEGRAM ON",
            "connected",
            f"Connected to Telegram as @{username}.",
        )

    @Slot(str)
    def _telegram_status(self, message: str) -> None:
        self._set_telegram_badge("TELEGRAM ON", "connected", f"Telegram: {message}")

    @Slot(str)
    def _telegram_failed(self, message: str) -> None:
        self._set_telegram_badge(
            "TELEGRAM ERROR",
            "error",
            f"Telegram error: {message}",
        )

    def _set_telegram_badge(
        self,
        text: str,
        status: str,
        tooltip: str,
    ) -> None:
        self.telegram_status_label.setText(text)
        self.telegram_status_label.setProperty("status", status)
        self.telegram_status_label.setToolTip(tooltip)
        self.telegram_status_label.style().unpolish(self.telegram_status_label)
        self.telegram_status_label.style().polish(self.telegram_status_label)

    def _telegram_thread_finished(self) -> None:
        if self.telegram_worker:
            self.telegram_worker.deleteLater()
        if self.telegram_thread:
            self.telegram_thread.deleteLater()
        self.telegram_worker = None
        self.telegram_thread = None

    def _stop_telegram_bot(self) -> None:
        worker = self.telegram_worker
        thread = self.telegram_thread
        if worker is not None:
            worker.cancel()
        if thread is not None and thread.isRunning():
            if not thread.wait(2500):
                thread.terminate()
                thread.wait(1000)
        self.telegram_worker = None
        self.telegram_thread = None

    @Slot()
    def _open_settings(self) -> None:
        if (
            self.chat_worker is not None
            or self.voice_record_worker is not None
            or self.voice_transcription_worker is not None
            or self.speech_worker is not None
        ):
            return
        dialog = SettingsDialog(
            self.config,
            self,
            telegram_token_present=bool(load_telegram_token()),
        )
        if dialog.exec() != SettingsDialog.DialogCode.Accepted:
            return
        previous_autostart = self.config.autostart_enabled
        next_config = dialog.config()
        token = dialog.telegram_token()
        if next_config.telegram_enabled and not token and not load_telegram_token():
            QMessageBox.warning(
                self,
                "Telegram token required",
                "Add the Telegram bot token before enabling Telegram.",
            )
            return
        if token:
            try:
                save_telegram_token(token)
            except (OSError, ValueError) as error:
                QMessageBox.warning(
                    self,
                    "Telegram token not saved",
                    f"Lura could not save the Telegram token:\n{error}",
                )
                return
        try:
            set_autostart_enabled(next_config.autostart_enabled)
        except OSError as error:
            QMessageBox.warning(
                self,
                "Autostart not updated",
                f"Lura could not update the Linux autostart entry:\n{error}",
            )
            next_config.autostart_enabled = previous_autostart
        self._stop_telegram_bot()
        self.config = next_config
        self.config.save()
        self._sync_quit_behavior()
        self.client = OllamaClient(self.config.ollama_url)
        self.service = AssistantService(self.client)
        self.voice_service = VoiceService(self.config)
        self._set_voice_idle()
        self.model_selector.clear()
        self.model_selector.addItem(self.config.model)
        self.runtime_status_value.setText("CHECKING")
        self._refresh_connection()
        self._start_telegram_bot()

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

    @Slot(int)
    def _conversation_selected(self, index: int) -> None:
        if index < 0:
            return
        conversation_id = self.history_list.itemData(index)
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

    @Slot()
    def _clear_current_conversation(self) -> None:
        if self.chat_worker is not None:
            return
        self.messages = []
        self.chat_view.clear_messages()
        self._persist_current_conversation()
        self.message_input.setFocus()

    @Slot()
    def _export_current_conversation(self) -> None:
        if not self.messages:
            return
        default_name = f"{self.current_conversation.title.replace('/', '-')}.md"
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            "Export conversation",
            str(Path.home() / default_name),
            "Markdown files (*.md)",
        )
        if not file_name:
            return
        lines = [f"# {self.current_conversation.title}", ""]
        for message in self.messages:
            speaker = "You" if message.role == "user" else "Lura"
            lines.extend([f"## {speaker}", "", message.content, ""])
        try:
            Path(file_name).write_text("\n".join(lines), encoding="utf-8")
        except OSError as error:
            self.status_label.setToolTip(f"Could not export conversation: {error}")

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
            self.history_list.addItem(conversation.title, conversation.id)
            if conversation.id == selected_id:
                self.history_list.setCurrentIndex(self.history_list.count() - 1)
        self.history_list.blockSignals(False)

    @Slot()
    def _update_clock(self) -> None:
        self.clock_label.setText(
            QDateTime.currentDateTime().toString("HH:mm:ss  |  MMM d, yyyy")
        )

    def _set_status(self, status: str) -> None:
        self.status_label.setProperty("status", status)
        self.status_label.style().unpolish(self.status_label)
        self.status_label.style().polish(self.status_label)

    def _set_orb_state(self, state: str) -> None:
        self.core_widget.set_state(state)
        labels = {
            "idle": "READY // HOLD TO SPEAK"
            if self.config.voice_input_enabled
            else "READY // TYPE TO SPEAK",
            "listening": "LISTENING // RELEASE TO SEND",
            "thinking": "THINKING // LOCAL MODEL",
            "speaking": "SPEAKING // LOCAL VOICE",
            "error": "CHANNEL ERROR // CHECK STATUS",
        }
        self.core_status.setText(labels.get(state, labels["idle"]))

    @Slot()
    def _toggle_focus_mode(self) -> None:
        self._set_focus_mode(not self._focus_mode)

    @Slot()
    def _exit_focus_mode(self) -> None:
        if self._focus_mode:
            self._set_focus_mode(False)

    def _set_focus_mode(self, enabled: bool) -> None:
        self._focus_mode = enabled
        self.top_bar.setVisible(not enabled)
        self.transcript_header.setVisible(not enabled)
        self.chat_view.setVisible(not enabled)
        self.composer.setVisible(not enabled)
        self.core_status.setVisible(not enabled)
        self.core_quote.setVisible(enabled)
        self.focus_button.setToolTip(
            "Exit focus mode (Escape)" if enabled else "Focus mode — show only the orb"
        )
        self.core_widget.setFocus()
        self.centralWidget().layout().invalidate()
        self.centralWidget().update()

    def closeEvent(self, event) -> None:
        if (
            not self._quitting
            and self.config.background_mode_enabled
            and self.tray_icon is not None
        ):
            event.ignore()
            self.hide_to_tray()
            return

        threads = (
            (self.chat_thread, self.chat_worker),
            (self.connection_thread, self.connection_worker),
            (self.voice_record_thread, self.voice_record_worker),
            (self.voice_transcription_thread, self.voice_transcription_worker),
            (self.speech_thread, self.speech_worker),
            (self.telegram_thread, self.telegram_worker),
        )
        for thread, worker in threads:
            if worker is not None:
                cancel = getattr(worker, "cancel", None)
                if cancel is not None:
                    cancel()
            if thread is not None and thread.isRunning():
                thread.quit()
                if not thread.wait(2000):
                    thread.terminate()
                    thread.wait(1000)
        if self.api_server is not None:
            self.api_server.shutdown()
            self.api_server.server_close()
            if self.api_thread is not None:
                self.api_thread.join(timeout=2)
        if self.tray_icon is not None:
            self.tray_icon.hide()
        event.accept()
