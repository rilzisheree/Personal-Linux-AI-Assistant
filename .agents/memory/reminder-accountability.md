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

Reminder confirmation and timing must remain application-authoritative: bind
approval to the exact pending tool call, accept only explicit confirmation
phrases, and report relative delays from the scheduler rather than letting the
model reinterpret them.

**Why:** A short confirmation timeout and model-generated timing paraphrase
caused valid reminder approvals to become unrelated memory actions and made a
five-minute reminder appear to be scheduled for tomorrow.

**How to apply:** Keep confirmation state outside the conversation transcript;
generate final reminder timing from the same clock used by ReminderService.