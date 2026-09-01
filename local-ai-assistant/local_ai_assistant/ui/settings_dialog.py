"""Settings dialog for Lura's local services and visual preferences."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSpinBox,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from ..config import (
    AppConfig,
    TTS_ENGINES,
    TTS_VOICE_PRESETS,
)
from ..voice import VoiceService


class SettingsDialog(QDialog):
    """A grouped dark settings surface while preserving the existing config API."""

    def __init__(
        self,
        config: AppConfig,
        parent=None,
        *,
        telegram_token_present: bool = False,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("settingsDialog")
        self.setWindowTitle("Lura // Settings")
        self.setMinimumSize(680, 540)
        self.resize(780, 700)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 22)
        layout.setSpacing(12)

        eyebrow = QLabel("LURA / CONFIGURATION")
        eyebrow.setObjectName("sectionLabel")
        layout.addWidget(eyebrow)
        intro = QLabel("Shape the local assistant without leaving your machine.")
        intro.setObjectName("settingsIntro")
        layout.addWidget(intro)

        tabs = QTabWidget()
        tabs.setObjectName("settingsTabs")
        layout.addWidget(tabs, 1)
        tabs.addTab(self._scroll_page(self._assistant_page(config)), "Assistant")
        tabs.addTab(
            self._scroll_page(
                self._ai_page(config)
            ),
            "AI & Models",
        )
        tabs.addTab(self._scroll_page(self._voice_page(config)), "Voice")
        tabs.addTab(
            self._scroll_page(
                self._integrations_page(
                    config,
                    telegram_token_present,
                )
            ),
            "Integrations",
        )
        tabs.addTab(self._scroll_page(self._security_page()), "Security")

        self.error_label = QLabel()
        self.error_label.setObjectName("settingsError")
        self.error_label.setWordWrap(True)
        self.error_label.hide()
        layout.addWidget(self.error_label)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save
            | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @staticmethod
    def _scroll_page(page: QWidget) -> QScrollArea:
        """Keep every settings page usable on short or scaled displays."""
        page.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Preferred,
        )
        scroll_area = QScrollArea()
        scroll_area.setObjectName("settingsScrollArea")
        scroll_area.setFrameShape(QFrame.Shape.NoFrame)
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(
            Qt.ScrollBarPolicy.ScrollBarAlwaysOff
        )
        scroll_area.setWidget(page)
        return scroll_area

    @staticmethod
    def _page_layout(title: str, description: str) -> tuple[QWidget, QVBoxLayout]:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(12, 16, 12, 16)
        layout.setSpacing(12)
        title_label = QLabel(title)
        title_label.setObjectName("settingsPageTitle")
        layout.addWidget(title_label)
        description_label = QLabel(description)
        description_label.setObjectName("settingsPageDescription")
        description_label.setWordWrap(True)
        layout.addWidget(description_label)
        return page, layout

    @staticmethod
    def _group(title: str) -> tuple[QGroupBox, QFormLayout]:
        group = QGroupBox(title)
        form = QFormLayout(group)
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setFormAlignment(Qt.AlignmentFlag.AlignTop)
        form.setHorizontalSpacing(18)
        form.setVerticalSpacing(10)
        return group, form

    def _assistant_page(self, config: AppConfig) -> QWidget:
        page, layout = self._page_layout(
            "Assistant presence",
            "These presentation controls prepare the orb and wake-word experience.",
        )
        group, form = self._group("ASSISTANT")
        self.assistant_name_input = QLineEdit(config.assistant_name)
        self.assistant_name_input.setPlaceholderText("Lura")
        self.wake_word_enabled = QCheckBox("Enable wake-word listening")
        self.wake_word_enabled.setChecked(config.wake_word_enabled)
        self.wake_word_inputs: list[QLineEdit] = []
        self.wake_word_rows: list[QWidget] = []
        self.wake_words_container = QWidget()
        self.wake_words_layout = QVBoxLayout(self.wake_words_container)
        self.wake_words_layout.setContentsMargins(0, 0, 0, 0)
        self.wake_words_layout.setSpacing(6)
        for wake_word in config.wake_words:
            self._add_wake_word_row(wake_word)
        if not self.wake_word_inputs:
            self._add_wake_word_row(config.wake_word)
        # Compatibility alias for existing callers and the primary wake word.
        self.wake_word_input = self.wake_word_inputs[0]
        self.add_wake_word_button = QPushButton("+ Add wake word")
        self.add_wake_word_button.setObjectName("quietButton")
        self.add_wake_word_button.clicked.connect(
            lambda _checked=False: self._add_wake_word_row()
        )
        self.active_listening_duration = QSpinBox()
        self.active_listening_duration.setRange(1, 60)
        self.active_listening_duration.setValue(config.active_listening_duration)
        self.active_listening_duration.setSuffix(" seconds")
        self.continuous_conversation_enabled = QCheckBox(
            "Continue listening after the wake word"
        )
        self.continuous_conversation_enabled.setChecked(
            config.continuous_conversation_enabled
        )
        self.conversation_timeout = QSpinBox()
        self.conversation_timeout.setRange(3, 120)
        self.conversation_timeout.setValue(config.conversation_timeout)
        self.conversation_timeout.setSuffix(" seconds")
        self.conversation_transition_delay = QDoubleSpinBox()
        self.conversation_transition_delay.setRange(0.1, 2.0)
        self.conversation_transition_delay.setDecimals(1)
        self.conversation_transition_delay.setSingleStep(0.1)
        self.conversation_transition_delay.setValue(
            config.conversation_transition_delay
        )
        self.conversation_transition_delay.setSuffix(" seconds")
        form.addRow("Assistant name", self.assistant_name_input)
        form.addRow("", self.wake_word_enabled)
        form.addRow("Wake words", self.wake_words_container)
        form.addRow("", self.add_wake_word_button)
        form.addRow("Active listening", self.active_listening_duration)
        form.addRow("", self.continuous_conversation_enabled)
        form.addRow("Conversation timeout", self.conversation_timeout)
        form.addRow("Voice transition delay", self.conversation_transition_delay)
        layout.addWidget(group)

        appearance, appearance_form = self._group("APPEARANCE")
        self.theme_input = QComboBox()
        self.theme_input.addItem("Obsidian / blue glow", "obsidian")
        self.theme_input.setCurrentIndex(
            max(0, self.theme_input.findData(config.theme))
        )
        self.orb_intensity_input = QSpinBox()
        self.orb_intensity_input.setRange(1, 100)
        self.orb_intensity_input.setValue(config.orb_intensity)
        self.orb_intensity_input.setSuffix("%")
        self.animation_intensity_input = QSpinBox()
        self.animation_intensity_input.setRange(1, 100)
        self.animation_intensity_input.setValue(config.animation_intensity)
        self.animation_intensity_input.setSuffix("%")
        appearance_form.addRow("Theme", self.theme_input)
        appearance_form.addRow("Orb intensity", self.orb_intensity_input)
        appearance_form.addRow("Motion intensity", self.animation_intensity_input)
        layout.addWidget(appearance)

        note = QLabel(
            "Wake-word listening stays local and automatically retries after "
            "temporary microphone or transcription errors."
        )
        note.setObjectName("settingsHint")
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addStretch(1)
        return page

    def _voice_page(self, config: AppConfig) -> QWidget:
        page, layout = self._page_layout(
            "Local voice",
            "Audio is processed on this host through your selected local tools.",
        )
        input_group, input_form = self._group("VOICE INPUT")
        self.voice_input_enabled = QCheckBox("Enable hold-to-talk orb input")
        self.voice_input_enabled.setChecked(config.voice_input_enabled)
        self.microphone_input = QComboBox()
        self.microphone_input.setEditable(False)
        self.microphone_input.setMinimumWidth(320)
        self.microphone_input.setToolTip(
            "Choose the exact PipeWire/PulseAudio input source used for recording."
        )
        self.refresh_microphones_button = QPushButton("Refresh")
        self.refresh_microphones_button.setObjectName("quietButton")
        self.refresh_microphones_button.setToolTip("Rescan available microphones")
        self.refresh_microphones_button.clicked.connect(self._populate_microphones)
        microphone_row = QWidget()
        microphone_layout = QHBoxLayout(microphone_row)
        microphone_layout.setContentsMargins(0, 0, 0, 0)
        microphone_layout.setSpacing(8)
        microphone_layout.addWidget(self.microphone_input, 1)
        microphone_layout.addWidget(self.refresh_microphones_button)
        self._populate_microphones(config.microphone_device)
        self.whisper_model_input = QLineEdit(config.whisper_model)
        self.whisper_model_input.setPlaceholderText("base")
        self.whisper_language_input = QLineEdit(config.whisper_language)
        self.whisper_language_input.setPlaceholderText("auto")
        self.voice_silence_duration_input = QDoubleSpinBox()
        self.voice_silence_duration_input.setRange(0.5, 3.0)
        self.voice_silence_duration_input.setSingleStep(0.1)
        self.voice_silence_duration_input.setDecimals(1)
        self.voice_silence_duration_input.setValue(config.voice_silence_duration)
        self.voice_silence_duration_input.setSuffix(" seconds")
        self.voice_min_speech_duration_input = QDoubleSpinBox()
        self.voice_min_speech_duration_input.setRange(0.1, 1.0)
        self.voice_min_speech_duration_input.setSingleStep(0.1)
        self.voice_min_speech_duration_input.setDecimals(1)
        self.voice_min_speech_duration_input.setValue(
            config.voice_min_speech_duration
        )
        self.voice_min_speech_duration_input.setSuffix(" seconds")
        self.voice_vad_threshold_input = QSpinBox()
        self.voice_vad_threshold_input.setRange(100, 4000)
        self.voice_vad_threshold_input.setSingleStep(50)
        self.voice_vad_threshold_input.setValue(config.voice_vad_threshold)
        self.voice_vad_threshold_input.setSuffix(" level")
        input_form.addRow("", self.voice_input_enabled)
        input_form.addRow("Microphone", microphone_row)
        input_form.addRow("Whisper model", self.whisper_model_input)
        input_form.addRow("Language", self.whisper_language_input)
        input_form.addRow("End-of-speech silence", self.voice_silence_duration_input)
        input_form.addRow(
            "Minimum speech", self.voice_min_speech_duration_input
        )
        input_form.addRow("Voice activity threshold", self.voice_vad_threshold_input)
        layout.addWidget(input_group)

        output_group, output_form = self._group("VOICE OUTPUT")
        self.voice_responses_enabled = QCheckBox("Speak assistant responses aloud")
        self.voice_responses_enabled.setChecked(config.voice_responses_enabled)
        self.tts_engine_input = QComboBox()
        engine_labels = {
            "disabled": "Disabled",
            "piper": "Piper",
        }
        for engine in TTS_ENGINES:
            self.tts_engine_input.addItem(engine_labels[engine], engine)
        self.tts_engine_input.setCurrentIndex(
            max(0, self.tts_engine_input.findData(config.tts_engine))
        )
        self.tts_voice_input = QComboBox()
        for label, voice in TTS_VOICE_PRESETS:
            self.tts_voice_input.addItem(label, voice)
        preset_index = next(
            (
                index
                for index, (_, voice) in enumerate(TTS_VOICE_PRESETS)
                if voice == config.tts_voice
            ),
            0,
        )
        self.tts_voice_input.setCurrentIndex(preset_index)
        self.tts_voice_input.currentIndexChanged.connect(self._voice_changed)
        output_form.addRow("", self.voice_responses_enabled)
        output_form.addRow("TTS engine", self.tts_engine_input)
        output_form.addRow("TTS voice", self.tts_voice_input)
        layout.addWidget(output_group)
        hint = QLabel(
            "Hold the central orb to record. Whisper handles transcription and "
            "Piper handles speech output locally."
        )
        hint.setObjectName("settingsHint")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addStretch(1)
        self._voice_changed(self.tts_voice_input.currentIndex())
        return page

    def _ai_page(self, config: AppConfig) -> QWidget:
        page, layout = self._page_layout(
            "Local AI runtime",
            "Lura runs entirely through Ollama. No cloud provider or API key is required.",
        )
        ai_group, ai_form = self._group("AI RUNTIME")
        self.url_input = QLineEdit(config.ollama_url)
        self.url_input.setPlaceholderText("http://localhost:11434")
        self.model_input = QLineEdit(config.model)
        self.model_input.setPlaceholderText("qwen3.5:4b")
        self.context_size_input = QSpinBox()
        self.context_size_input.setRange(2048, 131072)
        self.context_size_input.setSingleStep(1024)
        self.context_size_input.setValue(config.ollama_context_size)
        self.context_size_input.setSuffix(" tokens")
        ai_form.addRow("Ollama URL", self.url_input)
        ai_form.addRow("Ollama model", self.model_input)
        ai_form.addRow("Context size", self.context_size_input)
        layout.addWidget(ai_group)
        hint = QLabel(
            "Only the Ollama URL, model, and context size are configurable here. "
            "Chat, tools, and model discovery stay on your local Ollama server."
        )
        hint.setObjectName("settingsHint")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addStretch(1)
        return page

    def _integrations_page(
        self,
        config: AppConfig,
        telegram_token_present: bool,
    ) -> QWidget:
        page, layout = self._page_layout(
            "Desktop integrations",
            "Control how Lura stays available in the background and how Telegram "
            "can reach this machine.",
        )
        desktop_group, desktop_form = self._group("DESKTOP")
        self.background_mode_enabled = QCheckBox(
            "Keep running in the system tray when the window is closed"
        )
        self.background_mode_enabled.setChecked(config.background_mode_enabled)
        self.autostart_enabled = QCheckBox("Start Lura automatically when I sign in")
        self.autostart_enabled.setChecked(config.autostart_enabled)
        desktop_form.addRow("", self.background_mode_enabled)
        desktop_form.addRow("", self.autostart_enabled)
        layout.addWidget(desktop_group)

        telegram_group, telegram_form = self._group("TELEGRAM PHONE BRIDGE")
        self.telegram_enabled = QCheckBox("Enable private Telegram control")
        self.telegram_enabled.setChecked(config.telegram_enabled)
        self.telegram_token_input = QLineEdit()
        self.telegram_token_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.telegram_token_input.setPlaceholderText(
            "Saved securely — leave blank to keep it"
            if telegram_token_present
            else "BotFather token"
        )
        self.telegram_user_id_input = QLineEdit(config.telegram_allowed_user_id)
        self.telegram_user_id_input.setPlaceholderText("Numeric Telegram user ID")
        telegram_form.addRow("", self.telegram_enabled)
        telegram_form.addRow("Bot token", self.telegram_token_input)
        telegram_form.addRow("Your user ID", self.telegram_user_id_input)
        layout.addWidget(telegram_group)
        hint = QLabel(
            "Telegram accepts private messages from the configured ID only. "
            "The token stays in its local permission-restricted file."
        )
        hint.setObjectName("settingsHint")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addStretch(1)
        return page

    def _security_page(self) -> QWidget:
        page, layout = self._page_layout(
            "Private by default",
            "Lura keeps the desktop conversation store and assistant services on your machine.",
        )
        group, form = self._group("ACCESS")
        password_status = QLabel("LOCAL PASSWORD ACTIVE")
        password_status.setObjectName("settingsStatus")
        session_status = QLabel("API SESSION // LOCAL ONLY")
        session_status.setObjectName("settingsStatus")
        telegram_status = QLabel("TELEGRAM // PRIVATE USER ALLOWLIST")
        telegram_status.setObjectName("settingsStatus")
        form.addRow("Desktop unlock", password_status)
        form.addRow("API sessions", session_status)
        form.addRow("Phone bridge", telegram_status)
        layout.addWidget(group)
        note = QLabel(
            "Password verification continues to use the existing ApiStore. "
            "This redesign does not change authentication, permissions, or storage."
        )
        note.setObjectName("settingsHint")
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addStretch(1)
        return page

    def config(self) -> AppConfig:
        microphone_device = self.microphone_input.currentData()
        if not isinstance(microphone_device, str):
            microphone_device = self.microphone_input.currentText()
        return AppConfig(
            ollama_url=self.url_input.text(),
            model=self.model_input.text(),
            ollama_context_size=self.context_size_input.value(),
            voice_input_enabled=self.voice_input_enabled.isChecked(),
            voice_responses_enabled=self.voice_responses_enabled.isChecked(),
            microphone_device=microphone_device,
            whisper_model=self.whisper_model_input.text(),
            whisper_language=self.whisper_language_input.text(),
            tts_engine=str(self.tts_engine_input.currentData()),
            tts_voice=self._selected_voice(),
            background_mode_enabled=self.background_mode_enabled.isChecked(),
            autostart_enabled=self.autostart_enabled.isChecked(),
            telegram_enabled=self.telegram_enabled.isChecked(),
            telegram_allowed_user_id=self.telegram_user_id_input.text(),
            assistant_name=self.assistant_name_input.text(),
            wake_word_enabled=self.wake_word_enabled.isChecked(),
            wake_word=self.wake_word_inputs[0].text(),
            wake_words=tuple(
                wake_word_input.text() for wake_word_input in self.wake_word_inputs
            ),
            active_listening_duration=self.active_listening_duration.value(),
            continuous_conversation_enabled=self.continuous_conversation_enabled.isChecked(),
            conversation_timeout=self.conversation_timeout.value(),
            conversation_transition_delay=self.conversation_transition_delay.value(),
            voice_silence_duration=self.voice_silence_duration_input.value(),
            voice_min_speech_duration=self.voice_min_speech_duration_input.value(),
            voice_vad_threshold=self.voice_vad_threshold_input.value(),
            theme=str(self.theme_input.currentData()),
            orb_intensity=self.orb_intensity_input.value(),
            animation_intensity=self.animation_intensity_input.value(),
        )

    def telegram_token(self) -> str:
        return self.telegram_token_input.text().strip()

    def _selected_voice(self) -> str:
        return str(self.tts_voice_input.currentData())

    def _add_wake_word_row(self, value: str = "") -> None:
        row = QWidget()
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.setSpacing(8)
        wake_word_input = QLineEdit(value)
        wake_word_input.setPlaceholderText("Lura")
        remove_button = QPushButton("Remove")
        remove_button.setObjectName("quietButton")
        remove_button.setToolTip("Remove this wake-word alias")
        remove_button.clicked.connect(
            lambda _checked=False, row=row: self._remove_wake_word_row(row)
        )
        row_layout.addWidget(wake_word_input, 1)
        row_layout.addWidget(remove_button)
        self.wake_words_layout.addWidget(row)
        self.wake_word_inputs.append(wake_word_input)
        self.wake_word_rows.append(row)
        self._sync_wake_word_remove_buttons()

    def _remove_wake_word_row(self, row: QWidget) -> None:
        if row not in self.wake_word_rows or len(self.wake_word_rows) <= 1:
            return
        index = self.wake_word_rows.index(row)
        self.wake_words_layout.removeWidget(row)
        row.deleteLater()
        self.wake_word_rows.pop(index)
        self.wake_word_inputs.pop(index)
        self._sync_wake_word_remove_buttons()

    def _sync_wake_word_remove_buttons(self) -> None:
        for row in self.wake_word_rows:
            remove_button = row.findChild(QPushButton)
            if remove_button is not None:
                remove_button.setEnabled(len(self.wake_word_rows) > 1)

    def _voice_changed(self, index: int) -> None:
        voice = self.tts_voice_input.itemData(index)
        if isinstance(voice, str) and voice.lower().endswith(".onnx"):
            self.tts_engine_input.setCurrentIndex(self.tts_engine_input.findData("piper"))

    def _populate_microphones(self, selected: str | None = None) -> None:
        if selected is None:
            selected = self.microphone_input.currentData()
        if not isinstance(selected, str):
            selected = ""
        selected = selected.strip()
        microphones = VoiceService.list_microphones()
        self.microphone_input.blockSignals(True)
        self.microphone_input.clear()
        self.microphone_input.addItem("System default microphone", "")
        for name, label in microphones:
            display = label if label == name else f"{label}  ({name})"
            self.microphone_input.addItem(display, name)
        if selected and self.microphone_input.findData(selected) < 0:
            self.microphone_input.addItem(
                f"Saved microphone unavailable  ({selected})",
                selected,
            )
        index = self.microphone_input.findData(selected)
        self.microphone_input.setCurrentIndex(max(0, index))
        self.microphone_input.blockSignals(False)

    def _accept(self) -> None:
        try:
            self.config()
        except ValueError as error:
            self.error_label.setText(str(error))
            self.error_label.show()
            return
        self.accept()