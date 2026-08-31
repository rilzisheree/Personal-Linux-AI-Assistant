"""Application stylesheet for Lura's quiet cinematic interface."""

from __future__ import annotations


APP_STYLE = """
QMainWindow, QWidget {
    background: #03070d;
    color: #d8e9f5;
    font-family: "Noto Sans", "Inter", sans-serif;
    font-size: 12px;
}

QMainWindow {
    background: #03070d;
}

QFrame#topBar {
    background: #050b13;
    border-bottom: 1px solid #101d2b;
}

QLabel#appMark {
    color: #d8efff;
    font-size: 16px;
    font-weight: 700;
    letter-spacing: 3px;
}

QLabel#appSubtitle, QLabel#topMeta, QLabel#composerHint, QLabel#sectionLabel {
    color: #55718a;
    font-size: 8px;
    letter-spacing: 1px;
}

QLabel#statusLabel, QLabel#telegramStatusLabel {
    background: transparent;
    border: 1px solid #1d354b;
    border-radius: 10px;
    color: #76a7c9;
    padding: 4px 9px;
    font-size: 9px;
}

QLabel#statusLabel[status="connected"], QLabel#telegramStatusLabel[status="connected"] {
    border-color: #225478;
    color: #8cc8ed;
}

QLabel#statusLabel[status="error"], QLabel#telegramStatusLabel[status="error"] {
    border-color: #6e4351;
    color: #d28c9c;
}

QComboBox, QLineEdit, QTextEdit {
    background: #070f19;
    border: 1px solid #15283b;
    border-radius: 8px;
    color: #d8e9f5;
    padding: 9px 11px;
    selection-background-color: #153b5a;
}

QComboBox:hover, QComboBox:focus, QLineEdit:focus, QTextEdit:focus {
    border-color: #326b98;
}

QComboBox::drop-down {
    border: 0;
    width: 22px;
}

QComboBox#modelSelector, QComboBox#sessionSelector {
    background: #050b13;
    border-color: #102235;
    border-radius: 7px;
    color: #80a9c4;
    font-size: 9px;
    padding: 7px 9px;
}

QPushButton {
    background: #08121e;
    border: 1px solid #172f45;
    border-radius: 8px;
    color: #8db5d0;
    padding: 7px 10px;
}

QPushButton:hover {
    background: #0b1b2c;
    border-color: #3977a4;
    color: #c4e7ff;
}

QPushButton:disabled {
    background: #050b12;
    border-color: #0e1b29;
    color: #3d566b;
}

QPushButton#topAction, QPushButton#settingsButton, QPushButton#focusButton {
    min-width: 30px;
    max-width: 42px;
    min-height: 30px;
    max-height: 30px;
    padding: 0;
    background: transparent;
    border-color: #162d42;
    color: #7da4bf;
    font-size: 14px;
}

QPushButton#quietButton {
    background: transparent;
    border-color: transparent;
    color: #587891;
    font-size: 9px;
    padding: 4px 5px;
}

QPushButton#quietButton:hover {
    background: #091421;
    border-color: #19364e;
    color: #a7c9df;
}

QFrame#mainStage {
    background: #03070d;
}

QLabel#coreName {
    color: #d5eaff;
    font-size: 15px;
    font-weight: 700;
    letter-spacing: 5px;
}

QLabel#coreStatus {
    color: #6289a6;
    font-size: 9px;
    letter-spacing: 1px;
    padding: 3px 7px;
}

QLabel#coreQuote {
    color: #6e91aa;
    font-size: 10px;
    letter-spacing: 1px;
    line-height: 1.5;
    padding: 6px 12px;
}

QScrollArea#transcript {
    background: transparent;
}

QFrame#messageRow {
    background: transparent;
    border: 0;
    border-bottom: 1px solid #0c1723;
}

QFrame#messageRow[role="user"] {
    border-bottom-color: #11273a;
}

QFrame#messageRow[role="tool"] {
    border-bottom-color: #172338;
}

QFrame#messageRow QTextBrowser {
    background: transparent;
    border: 0;
    color: #c9dce9;
    padding: 0;
}

QFrame#messageRow[role="user"] QTextBrowser {
    color: #88b5d1;
}

QFrame#messageRow[role="tool"] QTextBrowser {
    color: #738da3;
}

QLabel#messageRole {
    color: #7eabc8;
    font-size: 8px;
    font-weight: 700;
    letter-spacing: 2px;
}

QFrame#messageRow[role="user"] QLabel#messageRole {
    color: #5486a8;
}

QFrame#messageRow[role="tool"] QLabel#messageRole {
    color: #596c86;
}

QLabel#messageImage {
    border: 1px solid #1a3b55;
    border-radius: 8px;
    padding: 4px;
    background: #050b13;
}

QLabel#emptyState {
    color: #46647d;
    font-size: 11px;
    line-height: 1.5;
    padding: 30px;
}

QFrame#composer {
    background: transparent;
}

QLineEdit#messageInput {
    min-height: 24px;
    background: #050b13;
    border: 1px solid #152b40;
    border-radius: 16px;
    padding: 11px 16px;
    color: #d8e9f5;
}

QPushButton#sendButton {
    min-width: 44px;
    max-width: 44px;
    min-height: 44px;
    max-height: 44px;
    padding: 0;
    background: #0b2439;
    border: 1px solid #2c6895;
    border-radius: 22px;
    color: #b8dcf5;
    font-size: 17px;
}

QPushButton#sendButton:hover {
    background: #103554;
    border-color: #5794c2;
}

QPushButton#stopButton {
    min-width: 44px;
    max-width: 44px;
    min-height: 44px;
    max-height: 44px;
    padding: 0;
    background: #1a1118;
    border-color: #6e4351;
    border-radius: 22px;
    color: #d28c9c;
    font-size: 18px;
}

QPushButton#stopButton:hover {
    background: #2b1721;
}

QPushButton#micButton {
    min-width: 0;
    max-width: 0;
    min-height: 0;
    max-height: 0;
}

QDialog, QDialog#unlockDialog {
    background: #03070d;
    color: #d8e9f5;
}

QDialog QLabel {
    color: #adc9dc;
}

QDialog QGroupBox {
    background: #050b13;
    border: 1px solid #112438;
    border-radius: 10px;
    margin-top: 16px;
    padding: 18px 12px 12px;
}

QDialog QGroupBox::title {
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 5px;
    color: #75a7c7;
    font-size: 9px;
    letter-spacing: 1px;
}

QDialog QCheckBox {
    color: #9ebbcf;
    spacing: 8px;
}

QDialog QCheckBox::indicator {
    width: 15px;
    height: 15px;
    border: 1px solid #23435c;
    border-radius: 4px;
    background: #050b13;
}

QDialog QCheckBox::indicator:checked {
    background: #1d5c86;
    border-color: #4b94c5;
}

QDialogButtonBox QPushButton {
    min-width: 90px;
}

QLabel#unlockMark {
    color: #84bce0;
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 7px;
}

QLabel#welcomeTitle {
    color: #e1f2ff;
    font-size: 28px;
    font-weight: 500;
    padding: 18px 0 10px;
}

QLabel#welcomeHint, QLabel#unlockSubtitle {
    color: #54738c;
    font-size: 8px;
    letter-spacing: 2px;
}

QLabel#unlockTitle {
    color: #d9efff;
    font-size: 22px;
    font-weight: 500;
}

QLineEdit#unlockInput {
    min-height: 42px;
    background: #050b13;
    border: 1px solid #1a3851;
    border-radius: 13px;
    color: #dff3ff;
    font-size: 17px;
    letter-spacing: 4px;
}

QLineEdit#unlockInput:focus {
    border-color: #4386b5;
}

QPushButton#unlockButton {
    min-height: 42px;
    background: #0a2942;
    border-color: #2b6895;
    border-radius: 13px;
    color: #cae7fa;
    font-weight: 600;
}

QLabel#unlockError {
    color: #d28c9c;
    font-size: 10px;
}

QScrollBar:vertical {
    background: transparent;
    width: 7px;
    margin: 2px;
}

QScrollBar::handle:vertical {
    background: #17344c;
    border-radius: 3px;
    min-height: 24px;
}

QScrollBar::add-line, QScrollBar::sub-line {
    height: 0;
}

QToolTip {
    background: #071321;
    border: 1px solid #23435e;
    color: #d2e9f8;
    padding: 5px;
}
"""