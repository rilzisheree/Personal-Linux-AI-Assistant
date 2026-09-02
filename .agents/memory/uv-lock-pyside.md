---
name: Python lock repair
description: Recovery guidance for an imported uv lockfile missing the root PySide6 package record
---

An imported workspace may have a valid PySide6 addon and essentials package
records but no `pyside6` package record, even though the root project depends on
PySide6. In that state, uv reports that the dependency has a missing `source`
field and refuses both locking and installation.

**Why:** GUI tests and the desktop app cannot import PySide6, and the package
installer cannot repair the environment until the lockfile parses.

**How to apply:** Generate a temporary lock from the existing root
`pyproject.toml`, copy only the missing `pyside6` package block into the
workspace lockfile, validate with `uv lock --check`, and then use the normal
workspace package installer.