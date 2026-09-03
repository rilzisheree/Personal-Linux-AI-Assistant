"""Reminder timeline and safe accountability overlay."""

from __future__ import annotations

import time

from PySide6.QtCore import QDate, QDateTime, QTime, QTimer, Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDateEdit,
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTextEdit,
    QTimeEdit,
    QVBoxLayout,
    QWidget,
)

from ..reminders import (
    MAX_LOCK_DURATION_SECONDS,
    PRIORITY_VALUES,
    REPEAT_VALUES,
    Reminder,
    ReminderService,
)


def _countdown(timestamp: float) -> str:
    seconds = max(0, int(timestamp - time.time()))
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, seconds = divmod(remainder, 60)
    if days:
        return f"{days}d {hours:02d}h"
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def _date_time(timestamp: float) -> str:
    return QDateTime.fromSecsSinceEpoch(int(timestamp)).toString("ddd, dd MMM yyyy  •  hh:mm AP")


class ReminderView(QWidget):
    """A functional reminder editor, timeline, and completion history."""

    changed = Signal()

    def __init__(self, service: ReminderService, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.service = service
        self._countdown_timer = QTimer(self)
        self._countdown_timer.timeout.connect(self.refresh)
        self._countdown_timer.start(1000)
        self._build_ui()
        self.refresh()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        root.setContentsMargins(28, 22, 28, 24)
        root.setSpacing(14)

        header = QHBoxLayout()
        title = QLabel("REMINDERS")
        title.setObjectName("reminderTitle")
        header.addWidget(title)
        header.addStretch(1)
        subtitle = QLabel("ACCOUNTABILITY SYSTEM")
        subtitle.setObjectName("reminderSubtitle")
        header.addWidget(subtitle)
        root.addLayout(header)

        form = QFrame()
        form.setObjectName("reminderComposer")
        form_layout = QGridLayout(form)
        form_layout.setContentsMargins(18, 16, 18, 16)
        form_layout.setHorizontalSpacing(12)
        form_layout.setVerticalSpacing(9)

        self.task_input = QLineEdit()
        self.task_input.setPlaceholderText("Task name  ·  e.g. Do my homework")
        self.description_input = QTextEdit()
        self.description_input.setPlaceholderText("Optional instructions or context")
        self.description_input.setFixedHeight(54)
        self.date_input = QDateEdit(QDate.currentDate())
        self.date_input.setCalendarPopup(True)
        self.date_input.setMinimumDate(QDate.currentDate())
        self.time_input = QTimeEdit(QTime.currentTime().addSecs(300))
        self.time_input.setDisplayFormat("hh:mm AP")
        self.repeat_input = QComboBox()
        self.repeat_input.addItems(sorted(REPEAT_VALUES, key=lambda value: ["None", "Daily", "Weekly", "Custom"].index(value)))
        self.priority_input = QComboBox()
        self.priority_input.addItems(["Low", "Normal", "High", "Critical"])
        self.lock_input = QSpinBox()
        self.lock_input.setRange(1, MAX_LOCK_DURATION_SECONDS)
        self.lock_input.setValue(60)
        self.lock_input.setSuffix(" sec")
        self.custom_days_input = QSpinBox()
        self.custom_days_input.setRange(1, 365)
        self.custom_days_input.setValue(1)
        self.custom_days_input.setSuffix(" days")
        self.strict_input = QCheckBox("Strict mode")
        self.strict_input.setToolTip("Show the safe temporary accountability overlay when this reminder triggers.")
        self.add_button = QPushButton("+  ADD REMINDER")
        self.add_button.setObjectName("primaryReminderButton")
        self.add_button.clicked.connect(self._create_reminder)

        form_layout.addWidget(QLabel("TASK"), 0, 0)
        form_layout.addWidget(self.task_input, 0, 1, 1, 3)
        form_layout.addWidget(QLabel("DATE"), 1, 0)
        form_layout.addWidget(self.date_input, 1, 1)
        form_layout.addWidget(QLabel("TIME"), 1, 2)
        form_layout.addWidget(self.time_input, 1, 3)
        form_layout.addWidget(QLabel("DESCRIPTION"), 2, 0)
        form_layout.addWidget(self.description_input, 2, 1, 1, 3)
        form_layout.addWidget(QLabel("REPEAT"), 3, 0)
        form_layout.addWidget(self.repeat_input, 3, 1)
        form_layout.addWidget(QLabel("PRIORITY"), 3, 2)
        form_layout.addWidget(self.priority_input, 3, 3)
        form_layout.addWidget(QLabel("LOCK"), 4, 0)
        form_layout.addWidget(self.lock_input, 4, 1)
        form_layout.addWidget(QLabel("CUSTOM"), 4, 2)
        form_layout.addWidget(self.custom_days_input, 4, 3)
        form_layout.addWidget(self.strict_input, 5, 0, 1, 2)
        form_layout.addWidget(self.add_button, 5, 2, 1, 2)
        root.addWidget(form)

        self.upcoming_label = QLabel("UPCOMING")
        self.upcoming_label.setObjectName("reminderSectionLabel")
        root.addWidget(self.upcoming_label)
        self.upcoming_area, self.upcoming_layout = self._scroll_section()
        root.addWidget(self.upcoming_area, 1)

        self.history_label = QLabel("HISTORY")
        self.history_label.setObjectName("reminderSectionLabel")
        root.addWidget(self.history_label)
        self.history_area, self.history_layout = self._scroll_section()
        self.history_area.setMaximumHeight(180)
        root.addWidget(self.history_area)

    def _scroll_section(self) -> tuple[QScrollArea, QVBoxLayout]:
        content = QWidget()
        layout = QVBoxLayout(content)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(8)
        layout.addStretch(1)
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setWidget(content)
        area.setObjectName("reminderScroll")
        return area, layout

    def refresh(self) -> None:
        reminders = sorted(self.service.store.list(), key=lambda reminder: reminder.due_at)
        active = [reminder for reminder in reminders if reminder.status in {"upcoming", "triggered"}]
        history = [reminder for reminder in reminders if reminder.status in {"completed", "missed", "cancelled"}]
        self._render_cards(self.upcoming_layout, active, history=False)
        self._render_cards(self.history_layout, sorted(history, key=lambda reminder: reminder.completed_at or reminder.due_at, reverse=True), history=True)
        self.upcoming_label.setText(f"UPCOMING  ·  {len(active)}")
        self.history_label.setText(f"HISTORY  ·  {len(history)}")

    def _render_cards(self, layout: QVBoxLayout, reminders: list[Reminder], *, history: bool) -> None:
        while layout.count() > 1:
            item = layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        for reminder in reminders:
            card = QFrame()
            card.setObjectName("reminderCard")
            row = QHBoxLayout(card)
            row.setContentsMargins(14, 11, 12, 11)
            details = QVBoxLayout()
            details.setSpacing(3)
            task = QLabel(reminder.message)
            task.setObjectName("reminderTask")
            details.addWidget(task)
            schedule = QLabel(_date_time(reminder.due_at))
            schedule.setObjectName("reminderMeta")
            details.addWidget(schedule)
            if reminder.description:
                description = QLabel(reminder.description)
                description.setObjectName("reminderDescription")
                description.setWordWrap(True)
                details.addWidget(description)
            row.addLayout(details, 1)
            badges = QVBoxLayout()
            badges.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            badge = QLabel(f"{reminder.priority.upper()}  ·  {reminder.repeat.upper()}")
            badge.setObjectName(f"priority{reminder.priority}")
            badges.addWidget(badge, 0, Qt.AlignmentFlag.AlignRight)
            status = reminder.status.upper()
            if not history and reminder.status == "upcoming":
                status = _countdown(reminder.due_at)
            countdown = QLabel(status)
            countdown.setObjectName("reminderCountdown")
            badges.addWidget(countdown, 0, Qt.AlignmentFlag.AlignRight)
            if history:
                when = reminder.completed_at or reminder.due_at
                badges.addWidget(QLabel(f"{reminder.status.title()}  ·  {_date_time(when)}"), 0, Qt.AlignmentFlag.AlignRight)
            else:
                cancel = QPushButton("CANCEL")
                cancel.setObjectName("quietButton")
                cancel.clicked.connect(lambda _checked=False, rid=reminder.reminder_id: self._cancel(rid))
                badges.addWidget(cancel, 0, Qt.AlignmentFlag.AlignRight)
            row.addLayout(badges)
            layout.insertWidget(layout.count() - 1, card)

        if not reminders:
            empty = QLabel("No reminders here yet.")
            empty.setObjectName("reminderEmpty")
            layout.insertWidget(0, empty)

    def _create_reminder(self) -> None:
        task = self.task_input.text().strip()
        if not task:
            QMessageBox.warning(self, "Add reminder", "Enter a task name first.")
            return
        date_time = QDateTime(self.date_input.date(), self.time_input.time())
        due_at = float(date_time.toSecsSinceEpoch())
        if due_at <= time.time():
            QMessageBox.warning(self, "Add reminder", "Choose a future date and time.")
            return
        try:
            self.service.schedule_at(
                task,
                due_at,
                description=self.description_input.toPlainText(),
                repeat=self.repeat_input.currentText(),
                priority=self.priority_input.currentText(),
                lock_duration_seconds=self.lock_input.value(),
                strict_mode=self.strict_input.isChecked(),
                custom_repeat_days=self.custom_days_input.value(),
            )
        except ValueError as error:
            QMessageBox.warning(self, "Add reminder", str(error))
            return
        self.task_input.clear()
        self.description_input.clear()
        self.refresh()
        self.changed.emit()

    def _cancel(self, reminder_id: str) -> None:
        self.service.cancel(reminder_id)
        self.refresh()
        self.changed.emit()


class AccountabilityOverlay(QDialog):
    """Safe, bounded in-app accountability mode."""

    resolved = Signal(str, str)

    def __init__(self, reminder: Reminder, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.reminder = reminder
        self._started_at = time.monotonic()
        self._resolved = False
        self._escape_timer = QTimer(self)
        self._escape_timer.setSingleShot(True)
        self._escape_timer.setInterval(5000)
        self._escape_timer.timeout.connect(self._emergency_exit)
        self._countdown_timer = QTimer(self)
        self._countdown_timer.timeout.connect(self._update_countdown)
        self._countdown_timer.start(250)
        self.setWindowTitle("Lura Accountability Mode")
        self.setModal(True)
        self.setWindowModality(Qt.WindowModality.ApplicationModal)
        self.setWindowFlags(
            Qt.WindowType.Dialog
            | Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
        )
        self._build_ui()
        self._update_countdown()

    def _build_ui(self) -> None:
        self.setObjectName("accountabilityOverlay")
        root = QVBoxLayout(self)
        root.setContentsMargins(48, 44, 48, 44)
        root.setSpacing(16)
        root.addStretch(1)
        mark = QLabel("LURA  //  ACCOUNTABILITY MODE")
        mark.setObjectName("accountabilityMark")
        mark.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(mark)
        title = QLabel("TIME TO FOLLOW THROUGH")
        title.setObjectName("accountabilityTitle")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(title)
        task = QLabel(self.reminder.message)
        task.setObjectName("accountabilityTask")
        task.setWordWrap(True)
        task.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(task)
        if self.reminder.description:
            description = QLabel(self.reminder.description)
            description.setObjectName("accountabilityDescription")
            description.setWordWrap(True)
            description.setAlignment(Qt.AlignmentFlag.AlignCenter)
            root.addWidget(description)
        self.countdown = QLabel()
        self.countdown.setObjectName("accountabilityCountdown")
        self.countdown.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(self.countdown)
        hint = QLabel("You scheduled this reminder. Take care of it now.")
        hint.setObjectName("accountabilityHint")
        hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
        root.addWidget(hint)
        self.done_button = QPushButton("DONE")
        self.done_button.setObjectName("accountabilityDone")
        self.done_button.clicked.connect(lambda: self._resolve("completed"))
        root.addWidget(self.done_button, 0, Qt.AlignmentFlag.AlignCenter)
        self.emergency_hint = QLabel("Emergency exit: hold Escape for 5 seconds")
        self.emergency_hint.setObjectName("accountabilityEmergencyHint")
        self.emergency_hint.setVisible(False)
        root.addWidget(self.emergency_hint, 0, Qt.AlignmentFlag.AlignCenter)
        root.addStretch(1)

    def _update_countdown(self) -> None:
        remaining = max(0, self.reminder.lock_duration_seconds - int(time.monotonic() - self._started_at))
        self.countdown.setText(f"{remaining // 60:02d}:{remaining % 60:02d}")
        if remaining <= 0:
            self._resolve("missed")

    def _emergency_exit(self) -> None:
        self._resolve("cancelled")

    def keyPressEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape and not event.isAutoRepeat():
            self._escape_timer.start()
        else:
            event.ignore()

    def keyReleaseEvent(self, event) -> None:
        if event.key() == Qt.Key.Key_Escape:
            self._escape_timer.stop()
        else:
            event.ignore()

    def _resolve(self, status: str) -> None:
        if self._resolved:
            return
        self._resolved = True
        self._countdown_timer.stop()
        self._escape_timer.stop()
        self.resolved.emit(self.reminder.reminder_id, status)
        self.accept()

    def closeEvent(self, event) -> None:
        if self._resolved:
            event.accept()
        else:
            event.ignore()