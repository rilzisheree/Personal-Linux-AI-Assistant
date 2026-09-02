---
name: Qt reminder bridge
description: Safe handoff from the background reminder scheduler into the PySide6 UI
---

The reminder scheduler runs outside Qt, so due callbacks must emit a signal owned by
the main window and let Qt deliver the UI work on the main thread.

**Why:** Direct widget access from the scheduler thread can race with Qt event
processing and cause intermittent crashes or corrupted UI state.

**How to apply:** Keep scheduler callbacks limited to emitting data; start,
cancel, and clean up reminder alarm workers only from Qt slots.