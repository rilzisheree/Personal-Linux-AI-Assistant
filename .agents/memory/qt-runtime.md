---
name: Headless Qt runtime
description: Native libraries needed to import and smoke-test PySide6 applications in this workspace.
---

PySide6 applications in this workspace require native runtime packages for headless startup: `zstd`, `libglvnd`, `libxkbcommon`, `fontconfig`, and `dbus`.

**Why:** The Python wheel installed successfully, but Qt imports failed one library at a time until these Nix packages were available.

**How to apply:** Keep these packages in the workspace’s validated `.replit` Nix package list when running or smoke-testing the desktop client.