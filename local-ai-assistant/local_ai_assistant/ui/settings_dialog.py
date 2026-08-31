"""Settings dialog for the Ollama endpoint and selected model."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
)

from ..config import AppConfig, TTS_ENGINES, TTS_VOICE_PRESETS


class SettingsDialog(QDialog):
    def __init__(self, config: AppConfig, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle("Settings")
        self.setMinimumWidth(460)

        layout = QVBoxLayout(self)
        intro = QLabel("Configure the local Ollama connection.")
        intro.setStyleSheet("color: #9ba9b5;")
        layout.addWidget(intro)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self.url_input = QLineEdit(config.ollama_url)
        self.url_input.setPlaceholderText("http://localhost:11434")
        self.model_input = QLineEdit(config.model)
        self.model_input.setPlaceholderText("qwen3.5:4b")
        self.context_size_input = QSpinBox()
        self.context_size_input.setRange(2048, 131072)
        self.context_size_input.setSingleStep(1024)
        self.context_size_input.setValue(config.ollama_context_size)
        self.context_size_input.setSuffix(" tokens")
        self.context_size_input.setToolTip(
            "Larger context windows support longer chats but use more RAM/VRAM."
        )
        form.addRow("Ollama URL", self.url_input)
        form.addRow("Default model", self.model_input)
        form.addRow("Context size", self.context_size_input)
        layout.addLayout(form)

        voice_label = QLabel("Voice")
        voice_label.setStyleSheet("color: #9ba9b5; font-weight: 700;")
        layout.addWidget(voice_label)

        voice_form = QFormLayout()
        voice_form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self.voice_input_enabled = QCheckBox("Enable push-to-talk microphone")
        self.voice_input_enabled.setChecked(config.voice_input_enabled)
        self.voice_responses_enabled = QCheckBox("Speak assistant responses aloud")
        self.voice_responses_enabled.setChecked(config.voice_responses_enabled)
        self.microphone_input = QLineEdit(config.microphone_device)
        self.microphone_input.setPlaceholderText("Default microphone")
        self.microphone_input.setToolTip("Optional PipeWire node name or ALSA device.")
        self.whisper_model_input = QLineEdit(config.whisper_model)
        self.whisper_model_input.setPlaceholderText("base")
        self.whisper_language_input = QLineEdit(config.whisper_language)
        self.whisper_language_input.setPlaceholderText("auto")
        self.tts_engine_input = QComboBox()
        engine_labels = {
            "disabled": "Disabled",
            "espeak-ng": "eSpeak-NG",
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
        self.tts_voice_input.addItem("Custom voice / Piper model path", None)
        self.tts_voice_input.setToolTip(
            "Choose a local eSpeak-NG voice or select custom for a Piper model path."
        )
        self.custom_voice_input = QLineEdit()
        self.custom_voice_input.setPlaceholderText("e.g. en-us or /path/to/piper.onnx")
        self.custom_voice_input.setToolTip(
            "Used only when Custom voice is selected."
        )
        preset_index = next(
            (
                index
                for index, (_, voice) in enumerate(TTS_VOICE_PRESETS)
                if voice == config.tts_voice
            ),
            len(TTS_VOICE_PRESETS),
        )
        self.tts_voice_input.setCurrentIndex(preset_index)
        if preset_index == len(TTS_VOICE_PRESETS):
            self.custom_voice_input.setText(config.tts_voice)
        self.tts_voice_input.currentIndexChanged.connect(self._voice_changed)
        voice_form.addRow("", self.voice_input_enabled)
        voice_form.addRow("", self.voice_responses_enabled)
        voice_form.addRow("Microphone", self.microphone_input)
        voice_form.addRow("Whisper model", self.whisper_model_input)
        voice_form.addRow("Language", self.whisper_language_input)
        voice_form.addRow("TTS engine", self.tts_engine_input)
        voice_form.addRow("TTS voice", self.tts_voice_input)
        voice_form.addRow("Custom voice", self.custom_voice_input)
        self._voice_changed(self.tts_voice_input.currentIndex())
        layout.addLayout(voice_form)

        voice_hint = QLabel(
            "Push-to-talk uses pw-record/arecord. Whisper, Piper, eSpeak-NG, "
            "and audio playback remain local optional dependencies."
        )
        voice_hint.setWordWrap(True)
        voice_hint.setStyleSheet("color: #6f8593;")
        layout.addWidget(voice_hint)

        desktop_label = QLabel("Desktop")
        desktop_label.setStyleSheet("color: #9ba9b5; font-weight: 700;")
        layout.addWidget(desktop_label)

        desktop_form = QFormLayout()
        desktop_form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        self.background_mode_enabled = QCheckBox(
            "Keep running in the system tray when the window is closed"
        )
        self.background_mode_enabled.setChecked(config.background_mode_enabled)
        self.background_mode_enabled.setToolTip(
            "Closing the window hides Lura instead of stopping its local API. "
            "Use the tray menu to show it again or quit completely."
        )
        self.autostart_enabled = QCheckBox(
            "Start Lura automatically when I sign in"
        )
        self.autostart_enabled.setChecked(config.autostart_enabled)
        self.autostart_enabled.setToolTip(
            "Creates a user-level Linux autostart entry. Autostart launches "
            "Lura hidden in background mode."
        )
        desktop_form.addRow("", self.background_mode_enabled)
        desktop_form.addRow("", self.autostart_enabled)
        layout.addLayout(desktop_form)

        desktop_hint = QLabel(
            "Autostart uses your user-level ~/.config/autostart entry and "
            "does not require administrator access."
        )
        desktop_hint.setWordWrap(True)
        desktop_hint.setStyleSheet("color: #6f8593;")
        layout.addWidget(desktop_hint)

        self.error_label = QLabel()
        self.error_label.setStyleSheet("color: #ffaaa7;")
        self.error_label.hide()
        layout.addWidget(self.error_label)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self._accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def config(self) -> AppConfig:
        return AppConfig(
            ollama_url=self.url_input.text(),
            model=self.model_input.text(),
            ollama_context_size=self.context_size_input.value(),
            voice_input_enabled=self.voice_input_enabled.isChecked(),
            voice_responses_enabled=self.voice_responses_enabled.isChecked(),
            microphone_device=self.microphone_input.text(),
            whisper_model=self.whisper_model_input.text(),
            whisper_language=self.whisper_language_input.text(),
            tts_engine=str(self.tts_engine_input.currentData()),
            tts_voice=self._selected_voice(),
            background_mode_enabled=self.background_mode_enabled.isChecked(),
            autostart_enabled=self.autostart_enabled.isChecked(),
        )

    def _selected_voice(self) -> str:
        preset = self.tts_voice_input.currentData()
        if isinstance(preset, str):
            return preset
        return self.custom_voice_input.text()

    def _voice_changed(self, index: int) -> None:
        voice = self.tts_voice_input.itemData(index)
        is_custom = voice is None
        self.custom_voice_input.setEnabled(is_custom)
        if isinstance(voice, str) and voice.lower().endswith(".onnx"):
            self.tts_engine_input.setCurrentIndex(self.tts_engine_input.findData("piper"))

    def _accept(self) -> None:
        try:
            self.config()
        except ValueError as error:
            self.error_label.setText(str(error))
            self.error_label.show()
            return
        self.accept()
