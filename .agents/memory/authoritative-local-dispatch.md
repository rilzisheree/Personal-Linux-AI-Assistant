---
name: Authoritative local dispatch
description: Boundary between Lura's local tools and the conversational model
---

Explicit desktop actions and live machine facts must be selected and executed by
the application layer before the LLM formats the result. The model may phrase a
successful result, but it must not decide whether an unambiguous local action
should happen or fill missing system data from memory.

**Why:** Tool schemas alone allowed the response model to refuse application
launches or invent hardware details even when the local tool implementation was
available.

**How to apply:** Keep deterministic dispatch narrow and conservative. Use the
existing permission gate and discovered application registry, return structured
tool results, and leave ambiguous requests on the normal model/tool path.