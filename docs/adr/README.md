# Architecture Decision Records

Major design decisions for graph-fraud-detector are documented here.

| ADR | Decision | Status |
|-----|----------|--------|
| [ADR-001](ADR-001-rl-threshold-bandit.md) | RL: LinUCB contextual bandit for adaptive fraud detection threshold selection | Accepted |
| [ADR-002](ADR-002-mongodb-alert-layer.md) | Persistence: MongoDB Atlas M0 Free for fraud alert logging | Accepted |

## Decision Dependency Graph

```
ADR-001 (RL threshold bandit)
    └── depends on Elliptic 49 time steps + class labels (data layer)
    └── future: online updates driven by ADR-002 logged alerts

ADR-002 (MongoDB alert layer)
    └── persists /predict HIGH/MEDIUM outputs
    └── foundation for ADR-001 online RL iteration
```
