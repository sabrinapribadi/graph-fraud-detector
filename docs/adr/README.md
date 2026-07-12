# Architecture Decision Records

Major design decisions for graph-fraud-detector are documented here.

> For implementation blockers, root-cause diagnoses, and lessons learned across all sprints,
> see [docs/lessons-learned.md](../lessons-learned.md).

| ADR | Decision | Status |
|-----|----------|--------|
| [ADR-001](ADR-001-rl-threshold-bandit.md) | RL: LinUCB contextual bandit for adaptive fraud detection threshold selection | Accepted |
| [ADR-002](ADR-002-mongodb-alert-layer.md) | Persistence: MongoDB Atlas M0 Free for fraud alert logging (bronze layer) | Accepted |
| [ADR-003](ADR-003-postgresql-gold-layer.md) | Analytics: PostgreSQL (Supabase) gold layer for aggregated fraud metrics | Accepted |

## Decision Dependency Graph

```
ADR-001 (RL threshold bandit)
    └── depends on Elliptic 49 time steps + class labels (data layer)
    └── future: online updates driven by ADR-002 logged alerts
    └── future: rl_threshold_decisions table (ADR-003) stores bandit run history

ADR-002 (MongoDB alert layer — bronze)
    └── persists /predict HIGH/MEDIUM outputs as raw documents
    └── feeds ADR-003 gold layer via upsert on every prediction

ADR-003 (PostgreSQL gold layer)
    └── aggregates ADR-002 raw events into daily analytics rows
    └── provides /stats/daily endpoint for time-series reporting
    └── ready to log ADR-001 bandit decisions for longitudinal evaluation
```
