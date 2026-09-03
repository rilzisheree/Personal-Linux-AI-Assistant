---
name: Reminder accountability safety
description: Safety constraints for strict reminders and temporary accountability mode
---

Strict reminders use a bounded Lura overlay rather than attempting to disable or
lock the operating system. The overlay has a hard maximum duration and a
reliable emergency Escape hold so the user can always recover the session.

**Why:** An OS-level input lock could interfere with emergency access,
shutdown, recovery, or other applications. The product requirement explicitly
forbids permanent or unbreakable lockouts.

**How to apply:** Keep future accountability changes inside the Qt app,
enforce the duration cap at the data/service boundary, and preserve the
emergency escape path even if the visual design changes.