---
name: User response length policy
description: Output-token limits must distinguish routing from user-facing generation.
---

Do not apply a small fixed output-token limit to normal user-facing generations.
Keep strict limits only on bounded internal calls such as request classification, or
make a user-facing limit an explicit product setting with clear truncation telemetry.

**Why:** A short-request heuristic can still lead to a long answer after tool
execution, so a blanket cap can cut news summaries, status reports, and reminders
mid-sentence while looking like a random stream failure.

**How to apply:** Inspect the final generation payload, tool-result follow-up
payloads, and the model's completion reason before changing token limits. Treat
normal completion, token-limit completion, transport errors, and cancellation as
different states.