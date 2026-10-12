# Changelog

Newest first. Each entry uses STAR (Situation, Task, Action, Result) so it can be told as a story.

## [8.0.2] — 2026-10-12 — MIT license
**Files:** LICENSE
- **Situation:** The repository had no license file, so by default nobody else could legally reuse the code.
- **Task:** Make the portfolio code reusable with credit, consistently across all portfolio repositories.
- **Action:** Added the standard MIT license (© 2026 Sabrina Pribadi). Data sources keep their own terms; the license covers code only.
- **Result:** GitHub now detects the license as MIT; no code changed.
- **Talking point:** "All my portfolio code is MIT-licensed, so anyone can reuse it with attribution."

## [8.0.1] — 2026-10-09 — Neutral wording in deployment docs
**Files:** docs/lessons-learned.md, docs/adr/ADR-002-mongodb-alert-layer.md
- **Situation:** Blocker 5 (database ports blocked on a corporate network) named the specific network owner.
- **Task:** Keep the lesson — restricted networks only allow 80/443, so use HTTPS web consoles — without naming any organisation.
- **Action:** Reworded 4 lines to "the corporate network used during development". Root cause, fix and lesson are unchanged.
- **Result:** Documentation-only change; no code or deployment impact.
- **Talking point:** "When outbound DB ports were blocked, I switched to HTTPS-based managed consoles instead of waiting on firewall changes."
