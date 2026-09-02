---
name: Native Firefox launcher session
description: Native Firefox command forwarding depends on inheriting the active desktop session.
---

Custom native Firefox launchers must inherit the active desktop session and must not carry `MOZ_NO_REMOTE` when they are expected to forward URLs to an already-running Firefox instance.

**Why:** The same `firefox --new-tab URL` command reused the existing browser from a terminal but opened a second instance when launched detached by Lura.

**How to apply:** Keep custom browser launchers attached to the current session; do not switch to Flatpak solely to get existing-tab behavior.