---
name: Qt worker shutdown
description: Safe lifecycle behavior for Lura's background QThreads during Settings changes and application close.
---

Background workers that perform Python network or polling calls must be cancelled asynchronously from the GUI thread. The GUI must not call an unbounded `QThread.wait()` while applying Settings, and must not clear the thread reference until its `finished` signal has fired.

**Why:** A live QThread wrapper destroyed while its Python worker is still executing causes Qt to abort with “QThread: Destroyed while thread is still running”; an unbounded wait freezes the desktop UI.

**How to apply:** Have Settings mark a refresh/restart pending, request worker cancellation, and let the thread-finished callback clean up and start the replacement worker.