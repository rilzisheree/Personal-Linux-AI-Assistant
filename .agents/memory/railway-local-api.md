---
name: Railway local API bridge
description: Deployment constraint and session strategy for hosting Lura’s web companion separately from its local API.
---

The web companion can be hosted on Railway while the Lura API and Ollama stay on the user’s Linux machine, but the browser must reach the local API through an HTTPS tunnel or private network gateway. A Railway-hosted HTTPS page cannot reliably call an unexposed local HTTP port.

**Why:** A remote static host has no network path to the user’s `localhost`, and cross-origin cookies are restrictive. Lura’s API therefore supports an exact allowed origin and returns an expiring session token that the web client keeps in session storage for cross-origin requests.

**How to apply:** Keep same-origin relative API requests as the default for local use. For Railway, build with the API origin configured, set the API’s exact Railway origin in CORS configuration, and never expose the raw local API port without TLS and a strong password/session secret.