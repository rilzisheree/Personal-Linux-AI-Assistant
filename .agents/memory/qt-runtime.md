---
name: Headless Qt runtime
description: Native libraries needed to import and smoke-test PySide6 applications in this workspace.
---

PySide6 applications in this workspace require native runtime packages for
headless startup: `zstd`, `libglvnd`, `libxkbcommon`, `fontconfig`, and `dbus`.
The current Replit Python runtime does not have the PySide6 wheel installed, so
Qt visual smoke tests must run on the target Linux host or a provisioned Qt
environment.

**Why:** Qt imports have previously failed one native library at a time, and
the current dependency-light workspace does not provide PySide6 at all.

**How to apply:** Keep these packages in the workspace’s validated `.replit` Nix package list when running or smoke-testing the desktop client.