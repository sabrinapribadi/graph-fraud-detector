# ADR-004: Label Fix and a Pre-Registered Evaluation

**Status:** Accepted — written and pushed before any model was retrained or scored; results appended
**Date:** 2026-11-11
**Deciders:** Sabrina Pribadi

---

## Context and Problem Statement

- **The bug:** the data loader mapped Elliptic class "1" to licit and class "2" to illicit. The dataset defines the opposite: class "1" = illicit (4,545 transactions, 2% of nodes), class "2" = licit (42,019, 21%) (Weber et al. 2019).
- **Effect:** every model, risk figure, "top suspicious" list and the AUC 0.955 described licit transactions as fraud. The README also listed the counts swapped.
- **Also found:** AUC 0.955 and F1 0.898 were typed into the README and the knowledge base, not produced by an evaluation script.
- **Question:** after fixing the labels, how good is the detector at finding illicit transactions, measured in a way the published benchmark makes comparable?

## Decision Drivers

- The protocol is fixed before any model is retrained (portfolio rule: gates before results).
- Numbers must come from a script and a result file, never typed (R17).
- Use the dataset authors' own protocol where possible, so results compare with published ones (R2, R16).

## Decision (pre-registered)

### 1. Label fix

- Class "1" → 1 (illicit), class "2" → 0 (licit), "unknown" → −1. A test pins the published counts (4,545 / 42,019 / 157,205).

### 2. Pipeline check against the paper (temporal split)

- **Protocol (Weber et al. 2019, section 4):** train on labelled transactions of time steps 1–34, test on time steps 35–49.
- **Model:** random forest, scikit-learn, 50 trees, `max_features = 50`, `random_state = 42`, all 166 features (the paper's "AF").
- **Reference:** the paper reports illicit precision 0.956, recall 0.670, F1 0.788.
- **Passes if** our illicit F1 is within ±0.05 of 0.788. Our own tolerance (library versions and forest randomness); a wrong label mapping would land far outside it.

### 3. The app's GraphSAGE, as built

- Exactly the dashboard's configuration (`src/ui/dashboard.py`: hidden size 64, 2 layers, dropout 0.3, learning rate 0.001, 3,000-node class-balanced sample, its own random split with seed 42, 100 epochs with early stopping after 20), with the fixed labels.
- **Reported, not gated:** AUC and illicit precision, recall and F1 on its test set.
- **Caveat stated wherever shown:** a balanced sample with a random split flatters a model; it is not comparable with the temporal benchmark.
- The app's dense adjacency matrix cannot score all 46,564 labelled transactions on an 8 GB laptop (≈ 8 GB matrix), so the temporal protocol is applied to the random forest only.

### 4. Published comparison (cited, not re-run)

| Method (Weber et al. 2019, temporal split) | Illicit F1 |
|---|---|
| Random forest, all features | 0.788 |
| EvolveGCN | 0.720 |
| Skip-GCN | 0.705 |
| GCN | 0.628 |

- After time step 43 (a dark-market shutdown) every method in the paper fails; per-step F1 is reported.

### 5. What changes downstream

- README, knowledge-base text, dashboard fallbacks and the static API demo use only the measured numbers, each with a one-line meaning (R17).
- The live API host is retired; the demo page reads precomputed responses from GitHub Pages (no server cost).

## Correction before any run (2026-11-11)

- §3 first named the API cache's setup (2,000 nodes, 50 epochs). The dashboard, which shows the headline numbers, uses the setup now stated in §3. Corrected and pushed before any model was retrained.

## Empirical results (2026-11-11)

| Check | Target | Result |
|---|---|---|
| Label counts after the fix | 4,545 / 42,019 / 157,205 | equal (tested) |
| Random forest, temporal split, illicit F1 | 0.788 ± 0.05 | **0.815: passed** (precision 0.932, recall 0.724, AUC 0.902) |
| F1 before / after the step-43 shutdown | report | 0.904 / 0.031 (the paper reports the same collapse) |
| Dashboard GraphSAGE (balanced 3,000-node sample) | report | AUC 0.956, illicit F1 0.877, precision 0.918, recall 0.839; test half illicit |

- **Downstream:** README, PRD, knowledge-base text and dashboard now read `data/eval_results.json`; a "structuring" insight was dropped because Elliptic features are anonymised, not amounts.
- **Hosting:** the paid API and dashboard services are retired; the demo is static on GitHub Pages and the dashboard stays on Streamlit Community Cloud.
- **Also found:** the old demo's prediction button sent 166 zeros for every transaction; the static demo shows real held-out transactions instead.

## Considered Options

| Option | Status | Why |
|---|---|---|
| **Fix labels; random-forest check on the paper's temporal split; app model reported with caveat** | **Chosen** | Proves the fix against a published number; honest about the app's flattering setup |
| Rebuild the GNN on sparse adjacency for the full temporal protocol | Deferred | A larger project; the fix comes first |
| Keep the old numbers with a footnote | Rejected | They describe licit transactions as fraud |

## Falsification

| Result | Diagnosis | Next action |
|---|---|---|
| RF illicit F1 within 0.788 ± 0.05 | Labels and features match the paper | Use the fixed pipeline everywhere |
| RF F1 far below 0.738 | Labels, features or split still differ from the paper | Find the difference before publishing anything |
| GraphSAGE far below the RF | Small dense GNN on a 2,000-node sample is weaker than trees on features, as the paper found for GCN | Report; recommend the forest for screening |

## Related Decisions

- ADR-001 (RL threshold bandit): its rewards used the inverted labels; it is retrained on the fixed labels.

## Evidence (OpenAlex, retrieved 2026-11-11; generated by `python scripts/evidence.py`)

| Reference | Venue | Peer-reviewed | Year | Citations | Per year | Used for |
|---|---|---|---|---|---|---|
| Anti-Money Laundering in Bitcoin: Experimenting with Graph Convolutional Networks for Financial Forensics | KDD 2019 Workshop on Anomaly Detection in Finance (arXiv 1908.02591) | yes | 2019 | 752 | 94.0 | Elliptic dataset, time split and published results |

