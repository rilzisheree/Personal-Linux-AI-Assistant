"""Settings dialog for the Ollama endpoint and selected model."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QLabel,
    QLineEdit,
    QVBoxLayout,
)

from ..config import AppConfig


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
        form.addRow("Ollama URL", self.url_input)
        form.addRow("Default model", self.model_input)
        layout.addLayout(form)

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
        return AppConfig(self.url_input.text(), self.model_input.text())

    def _accept(self) -> None:
        try:
            self.config()
        except ValueError as error:
            self.error_label.setText(str(error))
            self.error_label.show()
            return
        self.accept()
