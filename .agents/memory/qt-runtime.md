---
name: Headless Qt runtime
description: Native libraries needed to import and smoke-test PySide6 applications in this workspace.
---

PySide6 applications in this workspace require native runtime packages for
headless startup: `zstd`, `libglvnd`, `libxkbcommon`, `fontconfig`, and `dbus`.
The current Replit Python runtime does not have the PySide6 wheel installed, so
Qt visual smoke tests must run on the target Linux host or a provisioned Qt
environment. The repository's current uv lock also prevents adding PySide6
through the workspace package helper because its pyside6 entry lacks a source
field while multiple matching packages exist.

**Why:** Qt imports have previously failed one native library at a time, and
the current dependency-light workspace does not provide PySide6 at all.

**How to apply:** Keep these packages in the workspace’s validated `.replit` Nix package list when running or smoke-testing the desktop client. Do not rewrite
the lockfile just to validate an isolated desktop UI change; use the target
Linux environment or provision the Qt runtime separately.