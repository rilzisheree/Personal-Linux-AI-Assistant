"""Cinematic local unlock dialog."""

from __future__ import annotations

from PySide6.QtCore import QEasingCurve, QPropertyAnimation, Qt, QTimer
from PySide6.QtWidgets import (
    QDialog,
    QGraphicsOpacityEffect,
    QLabel,
    QLineEdit,
    QPushButton,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from ..api_store import ApiStore


class UnlockDialog(QDialog):
    """A visual wrapper around the existing local password flow."""

    def __init__(self, store: ApiStore, parent=None) -> None:
        super().__init__(parent)
        self.store = store
        self.first_run = not store.has_password()
        self.setObjectName("unlockDialog")
        self.setWindowTitle("Lura")
        self.setModal(True)
        self.setFixedSize(520, 420)
        self.setWindowFlag(Qt.WindowType.WindowContextHelpButtonHint, False)
        self._build_ui()
        self._welcome_animation.start()
        self._welcome_seconds_remaining = 2
        self._welcome_timer = QTimer(self)
        self._welcome_timer.setInterval(1000)
        self._welcome_timer.timeout.connect(self._tick_welcome_countdown)
        self._welcome_timer.start()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(52, 42, 52, 42)
        root.setSpacing(0)

        self.pages = QStackedWidget()
        root.addWidget(self.pages)

        welcome = QVBoxLayout()
        welcome.setAlignment(Qt.AlignmentFlag.AlignCenter)
        welcome_mark = QLabel("LURA")
        welcome_mark.setObjectName("unlockMark")
        welcome_mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        welcome.addWidget(welcome_mark)
        welcome_title = QLabel("Welcome, Sir.")
        welcome_title.setObjectName("welcomeTitle")
        welcome_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        welcome.addWidget(welcome_title)
        welcome_hint = QLabel("LOCAL INTELLIGENCE // PRIVATE CHANNEL")
        welcome_hint.setObjectName("welcomeHint")
        welcome_hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        welcome.addWidget(welcome_hint)
        self.welcome_countdown = QLabel("OPENING IN 2")
        self.welcome_countdown.setObjectName("welcomeCountdown")
        self.welcome_countdown.setAlignment(Qt.AlignmentFlag.AlignCenter)
        welcome.addWidget(self.welcome_countdown)
        welcome_page = QWidget()
        welcome_page.setLayout(welcome)
        welcome_effect = QGraphicsOpacityEffect(welcome_page)
        welcome_page.setGraphicsEffect(welcome_effect)
        self._welcome_animation = QPropertyAnimation(welcome_effect, b"opacity", self)
        self._welcome_animation.setDuration(560)
        self._welcome_animation.setStartValue(0.0)
        self._welcome_animation.setEndValue(1.0)
        self._welcome_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
        self.pages.addWidget(welcome_page)

        password_page = QVBoxLayout()
        password_page.setAlignment(Qt.AlignmentFlag.AlignCenter)
        password_page.setSpacing(12)
        label = QLabel("Create access" if self.first_run else "Enter password")
        label.setObjectName("unlockTitle")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        password_page.addWidget(label)
        subtitle = QLabel(
            "Set the local key for Lura and its private API."
            if self.first_run
            else "Your conversations stay on this machine."
        )
        subtitle.setObjectName("unlockSubtitle")
        subtitle.setWordWrap(True)
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        password_page.addWidget(subtitle)

        self.password_input = QLineEdit()
        self.password_input.setObjectName("unlockInput")
        self.password_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.password_input.setPlaceholderText("••••••••")
        self.password_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.password_input.returnPressed.connect(self._submit)
        password_page.addWidget(self.password_input)

        self.confirm_input = QLineEdit()
        self.confirm_input.setObjectName("unlockInput")
        self.confirm_input.setEchoMode(QLineEdit.EchoMode.Password)
        self.confirm_input.setPlaceholderText("Confirm password")
        self.confirm_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.confirm_input.returnPressed.connect(self._submit)
        self.confirm_input.setVisible(self.first_run)
        password_page.addWidget(self.confirm_input)

        self.error_label = QLabel()
        self.error_label.setObjectName("unlockError")
        self.error_label.setWordWrap(True)
        self.error_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.error_label.hide()
        password_page.addWidget(self.error_label)

        self.submit_button = QPushButton("Create local key" if self.first_run else "Unlock Lura")
        self.submit_button.setObjectName("unlockButton")
        self.submit_button.clicked.connect(self._submit)
        password_page.addWidget(self.submit_button)
        password_page.addStretch(1)

        password_page_widget = QWidget()
        password_page_widget.setLayout(password_page)
        self.pages.addWidget(password_page_widget)

    def _tick_welcome_countdown(self) -> None:
        self._welcome_seconds_remaining -= 1
        if self._welcome_seconds_remaining <= 0:
            self._welcome_timer.stop()
            self._show_password_page()
            return
        self.welcome_countdown.setText(
            f"OPENING IN {self._welcome_seconds_remaining}"
        )

    def _show_password_page(self) -> None:
        self._welcome_timer.stop()
        self.pages.setCurrentIndex(1)
        password_page = self.pages.currentWidget()
        if password_page is not None:
            password_effect = QGraphicsOpacityEffect(password_page)
            password_page.setGraphicsEffect(password_effect)
            self._password_animation = QPropertyAnimation(
                password_effect,
                b"opacity",
                self,
            )
            self._password_animation.setDuration(420)
            self._password_animation.setStartValue(0.0)
            self._password_animation.setEndValue(1.0)
            self._password_animation.setEasingCurve(QEasingCurve.Type.OutCubic)
            self._password_animation.start()
        self.password_input.setFocus()

    def _submit(self) -> None:
        password = self.password_input.text()
        if self.first_run:
            confirmation = self.confirm_input.text()
            if password != confirmation:
                self._show_error("The passwords do not match.")
                return
            try:
                self.store.set_password(password)
            except ValueError as error:
                self._show_error(str(error))
                return
            self.accept()
            return

        if not self.store.verify_password(password):
            self._show_error("That password is incorrect.")
            self.password_input.selectAll()
            return
        self.accept()

    def _show_error(self, message: str) -> None:
        self.error_label.setText(message)
        self.error_label.show()
