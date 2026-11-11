# Changelog

Newest first. Each entry uses STAR (Situation, Task, Action, Result) so it can be told as a story.

## [9.0.0] — 2026-11-11 — Label fix, measured evaluation, no paid hosting
**Files:** src/data/loader.py, tests/test_labels.py, scripts/evaluate_models.py, scripts/build_static_demo.py, src/utils/measured.py, src/ui/dashboard.py, src/agent/rag_agent.py, docs/index.html, docs/data/demo.json, data/eval_results.json, docs/adr/ADR-004, README.md, PRD.md, .github/workflows/ci.yml; render.yaml removed
- **Situation:** An audit found the loader mapped Elliptic class "1" (illicit) to licit and "2" (licit) to illicit; the model, risk numbers and "top suspicious" lists described licit transactions as fraud. Headline metrics were typed, not measured, and two paid hosting services billed while idle.
- **Task:** Fix the labels and prove the fix against the published benchmark, measure every shown number, and run at no cost.
- **Action:**
  - Fixed the mapping with a test pinned to the published counts (4,545 illicit / 42,019 licit).
  - Pre-registered ADR-004, then reproduced Weber et al. (2019): random forest on the temporal split.
  - Re-measured the dashboard's GraphSAGE and labelled its balanced-sample caveat.
  - Every metric in the app now reads `data/eval_results.json`; dropped an insight that read anonymised features as amounts.
  - Replaced the live API demo (which sent 166 zeros for every transaction) with a static GitHub Pages demo of real held-out transactions; removed the Render blueprint; added CI.
- **Result:** Illicit F1 0.815 vs the published 0.788 (within ±0.05): labels and pipeline confirmed. F1 0.90 before and 0.03 after the step-43 dark-market shutdown. Hosting cost: $0.
- **Talking point:** I found my own model was trained on inverted labels, fixed it, and proved the fix by reproducing the dataset authors' published benchmark within 0.03.

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
