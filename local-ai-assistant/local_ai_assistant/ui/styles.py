"""Application stylesheet."""

from __future__ import annotations


APP_STYLE = """
QMainWindow, QWidget {
    background: #061019;
    color: #d6e9f4;
    font-family: "Noto Sans", "Inter", sans-serif;
    font-size: 12px;
}
QMainWindow {
    background: #061019;
}
QFrame#topBar {
    background: #08131e;
    border-bottom: 1px solid #183646;
}
QLabel#appMark {
    color: #b9ecff;
    font-size: 16px;
    font-weight: 700;
}
QLabel#appSubtitle {
    color: #5c8297;
    font-size: 8px;
}
QLabel#topMeta {
    color: #66899b;
    font-size: 9px;
    padding: 5px 8px;
    border: 1px solid #173344;
}
QLabel#statusLabel {
    border: 1px solid #20566d;
    border-radius: 10px;
    color: #78bfd4;
    padding: 4px 8px;
    font-size: 10px;
}
QLabel#statusLabel[status="connected"] {
    border-color: #26795f;
    color: #8edcb7;
}
QLabel#statusLabel[status="error"] {
    border-color: #8b5060;
    color: #ff9db5;
}
QWidget#leftRail {
    background: transparent;
}
QFrame#panelCard, QFrame#conversationPanel {
    background: #091722;
    border: 1px solid #1a3b4d;
    border-radius: 5px;
}
QLabel#cardTitle {
    color: #8bd7ed;
    font-size: 10px;
    font-weight: 700;
}
QLabel#cardEyebrow {
    color: #4c8298;
    font-size: 8px;
}
QLabel#cardKey {
    color: #5a8094;
    font-size: 9px;
}
QLabel#cardValue {
    color: #c2eaf4;
    font-size: 9px;
}
QLabel#cardNote {
    color: #517487;
    font-size: 8px;
}
QFrame#cameraPreview {
    background: #061019;
    border: 1px solid #19394a;
    border-radius: 3px;
}
QLabel#cameraStatus {
    color: #3d6677;
    font-size: 9px;
}
QFrame#colorSwatch {
    min-height: 10px;
    max-height: 10px;
    border-radius: 2px;
}
QFrame#centerStage {
    background: #061019;
    border: 1px solid #102b3a;
    border-radius: 5px;
}
QLabel#coreName {
    color: #e0f6fc;
    font-size: 15px;
    font-weight: 700;
}
QLabel#coreStatus {
    color: #83d5b1;
    font-size: 9px;
    padding: 3px 7px;
}
QPushButton#coreAction {
    min-width: 30px;
    max-width: 30px;
    min-height: 26px;
    max-height: 26px;
    padding: 0;
    background: #081b28;
    border: 1px solid #1b5e77;
    border-radius: 4px;
    color: #74d7f0;
    font-size: 14px;
}
QPushButton#coreAction:hover {
    background: #0d3042;
    border-color: #35c3e2;
}
QLabel#conversationTitle {
    color: #caeff8;
    font-size: 13px;
    font-weight: 700;
}
QLabel#conversationSubtitle, QLabel#composerHint {
    color: #4f7c8e;
    font-size: 8px;
}
QPushButton#panelButton {
    background: #0b2230;
    border: 1px solid #1a5067;
    border-radius: 3px;
    color: #7fc7dc;
    padding: 5px 7px;
    font-size: 9px;
}
QPushButton#panelButton:hover {
    background: #10384b;
    border-color: #35b9d8;
}
QComboBox#sessionSelector {
    background: #07141e;
    border: 1px solid #1a4355;
    border-radius: 3px;
    color: #a5d4e2;
    padding: 5px 7px;
    font-size: 9px;
}
QComboBox#sessionSelector:hover, QComboBox#sessionSelector:focus {
    border-color: #35b9d8;
}
QComboBox, QLineEdit, QTextEdit {
    background: #07141e;
    border: 1px solid #1a4355;
    border-radius: 3px;
    color: #d6edf5;
    padding: 8px 9px;
    selection-background-color: #126580;
}
QComboBox:hover, QLineEdit:focus, QTextEdit:focus {
    border-color: #35b9d8;
}
QComboBox::drop-down {
    border: 0;
    width: 20px;
}
QFrame#assistantBubble, QFrame#userBubble {
    border-radius: 5px;
}
QFrame#assistantBubble {
    background: #0b202c;
    border: 1px solid #1c5062;
}
QFrame#toolBubble {
    background: #1b1d24;
    border: 1px solid #75652b;
}
QFrame#userBubble {
    background: #0e3040;
    border: 1px solid #237593;
}
QFrame#assistantBubble QTextBrowser, QFrame#toolBubble QTextBrowser, QFrame#userBubble QTextBrowser {
    background: transparent;
    border: 0;
    color: #cde5ed;
}
QLabel#messageRole {
    color: #70c7d8;
    font-size: 9px;
    font-weight: 700;
}
QWidget#emptyState {
    background: transparent;
    color: #426679;
    font-size: 10px;
}
QFrame#composer {
    background: transparent;
}
QPushButton#sendButton {
    min-width: 34px;
    max-width: 34px;
    padding: 0;
    background: #0c4b60;
    border: 1px solid #28afd0;
    border-radius: 4px;
    color: #c9f5ff;
    font-size: 16px;
}
QPushButton#sendButton:hover {
    background: #126d85;
}
QPushButton#stopButton {
    min-width: 28px;
    max-width: 28px;
    padding: 0;
    background: transparent;
    border-color: #8b5060;
    color: #ff9db5;
    font-size: 16px;
}
QPushButton#stopButton:hover {
    background: #301827;
}
QPushButton#settingsButton {
    background: #0b2230;
    border-color: #1a5067;
    color: #82cce0;
    padding: 6px 9px;
}
QPushButton:disabled {
    background: #07131d;
    border-color: #193344;
    color: #456276;
}
QScrollArea {
    border: 0;
    background: transparent;
}
QScrollBar:vertical {
    background: #061019;
    width: 8px;
    margin: 2px;
}
QScrollBar::handle:vertical {
    background: #1c5062;
    border-radius: 3px;
    min-height: 24px;
}
QScrollBar::add-line, QScrollBar::sub-line {
    height: 0;
}
QDialog {
    background: #08131e;
}
QDialogButtonBox QPushButton {
    min-width: 80px;
}
"""
