# Changelog

Newest first. Each entry uses STAR (Situation, Task, Action, Result) so it can be told as a story.

## [8.0.1] — 2026-10-09 — Neutral wording in deployment docs
**Files:** docs/lessons-learned.md, docs/adr/ADR-002-mongodb-alert-layer.md
- **Situation:** Blocker 5 (database ports blocked on a corporate network) named the specific network owner.
- **Task:** Keep the lesson — restricted networks only allow 80/443, so use HTTPS web consoles — without naming any organisation.
- **Action:** Reworded 4 lines to "the corporate network used during development". Root cause, fix and lesson are unchanged.
- **Result:** Documentation-only change; no code or deployment impact.
- **Talking point:** "When outbound DB ports were blocked, I switched to HTTPS-based managed consoles instead of waiting on firewall changes."
