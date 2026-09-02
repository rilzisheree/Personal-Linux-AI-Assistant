---
name: Remote API tool boundary
description: Safe information tools must be explicitly exposed through the remote API
---

The desktop tool registry and the remote API tool allowlist are separate boundaries.
Adding a safe capability to the registry does not make it available to web or phone
clients.

**Why:** The API intentionally restricts remote desktop control, so a registered
search tool can still appear unavailable remotely unless the API allowlist and
round-trip coverage are updated.

**How to apply:** When exposing a new remote-safe tool, update the API allowlist,
keep mutation/input/shell tools local-only, and add an API test that verifies the
schema is advertised and the tool result is returned.