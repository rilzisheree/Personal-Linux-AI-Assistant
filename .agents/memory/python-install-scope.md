---
name: Workspace Python installs
description: Temporary Python dependency setup can alter imported workspace metadata.
---

When installing Python dependencies for tests in an imported workspace, inspect the
root package metadata and lockfile afterward and revert unrelated changes before
finishing.

**Why:** The workspace package manager may resolve dependencies from the repository
root even when the application lives in a nested Python package, causing version
constraints or lockfile records unrelated to the feature to appear in the diff.

**How to apply:** Prefer the existing project environment when available; otherwise
use the managed Python environment for verification, then check `git diff` and
retain only intentional application or test changes.