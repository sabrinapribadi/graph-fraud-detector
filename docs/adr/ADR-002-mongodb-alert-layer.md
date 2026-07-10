# ADR-002: MongoDB Atlas Alert Persistence Layer

**Status:** Accepted  
**Date:** 2026-07-10  
**Deciders:** Sabrina Pribadi

---

## Context and Problem Statement

The FastAPI `/predict` endpoint classifies transactions and returns results synchronously,
but does not persist any decisions. For a production fraud detection system, compliance
and audit requirements demand:

1. A durable record of every HIGH/MEDIUM classification with timestamp
2. The ability to query recent alerts without re-running inference
3. A foundation for future online RL updates driven by logged alert outcomes

## Decision Drivers

- **Operational persistence**: fraud alerts must survive API restarts
- **Fail-safe**: a MongoDB outage must not break core prediction functionality
- **Portfolio stack**: MongoDB Atlas (bronze layer) + future PostgreSQL (gold layer)
  demonstrates medallion architecture competence
- **Zero cost**: MongoDB Atlas M0 Free tier (512 MB) is sufficient for alert logs
  from portfolio demonstration traffic

## Considered Options

### Option A — SQLite file (local)
Log alerts to a local `alerts.db` SQLite file in the container.

- **Pros**: Zero external dependency; no network latency
- **Cons**: Not durable across Render deployments (ephemeral filesystem); not
  queryable from outside the container; does not demonstrate cloud DB skills
- **Why not chosen**: Ephemeral filesystem on Render means alerts are lost on
  every deploy

### Option B — PostgreSQL (Supabase)
Store alerts in a structured PostgreSQL table.

- **Pros**: Relational; queryable with SQL; Supabase free tier
- **Cons**: Structured schema for unstructured event logs is overengineered;
  PostgreSQL is better suited for the gold/aggregated layer; adds SQLAlchemy
  dependency to the API container
- **Why not chosen**: MongoDB is the correct bronze/raw layer for event logs;
  PostgreSQL reserved for structured analytical queries

### Option C — MongoDB Atlas M0 Free (chosen)
Store alerts as documents in `fraud_alerts` collection; query via pymongo.

- **Pros**: Schema-flexible (easy to add fields later); Atlas M0 free forever;
  demonstrates NoSQL + cloud DB skills; pymongo is synchronous (works cleanly
  with FastAPI thread pool); fail-safe wrapper means no API breakage if Atlas
  is unreachable
- **Why chosen**: Best fit for event log storage pattern; zero cost; demonstrates
  medallion architecture (MongoDB bronze → future PostgreSQL gold)

## Decision Outcome

**Chosen: Option C — MongoDB Atlas M0 Free, `fraud_detector` database, `fraud_alerts` collection.**

### Document Schema

```json
{
  "node_id": "tx_00142",
  "fraud_probability": 0.923,
  "risk_level": "HIGH",
  "prediction": "FRAUD",
  "timestamp": "2026-07-10T07:31:20.672Z"
}
```

Only HIGH and MEDIUM risk classifications are logged — LOW predictions (p < 0.5)
are expected background traffic and do not require audit trails.

### Fail-safe Design

`src/db/mongo.py` wraps all operations in try/except with module-level
`_db = None` fallback:

```python
def connect():
    try:
        _client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        _client.admin.command("ping")   # verify reachability
        ...
    except Exception:
        _client = None    # all subsequent calls are no-ops

def log_alert(...):
    if _db is None:
        return           # silent skip — prediction still returned to caller
```

This means the `/predict` endpoint never fails due to MongoDB unavailability.

### New API Endpoints

| Endpoint | Method | Description |
|---------|--------|-------------|
| `/alerts` | GET | Last N alerts (default 50), sorted newest-first, `_id` excluded |
| `/alerts/summary` | GET | Count by risk_level, total count, latest alert timestamp |

### Index

`fraud_alerts.timestamp` descending index created on startup — ensures
efficient retrieval of recent alerts without full collection scan.

## Positive Consequences

- Fraud alerts are now durable across API restarts and Render deployments
- `GET /alerts/summary` provides a one-call compliance dashboard endpoint
- Foundation for online RL: future iteration can read logged alerts to compute
  actual FP/FN outcomes and update the LinUCB bandit online

## Negative Consequences

- Port 27017 is blocked on IOH corporate network — local dev requires mobile
  hotspot or a VPN; Render deployment is unaffected (Render → Atlas unrestricted)
- All writes are synchronous (pymongo, not motor); for high-traffic production use
  Motor (async pymongo) would be recommended to avoid blocking the event loop
- Atlas M0 has 500 concurrent connection limit; sufficient for portfolio demo,
  not for production load

## Implementation Notes

- `src/db/mongo.py`: `connect()`, `get_db()`, `close()`, `log_alert()`, `ALERTS_COL`
- `src/db/__init__.py`: empty
- `.env`: `MONGODB_URI` (URL-encoded password: `@` → `%40`) + `MONGODB_DB=fraud_detector`
- `.gitignore`: `.env` already present in all project `.gitignore` files
- `pyproject.toml`: `pymongo = {version = "^4.6.0", extras = ["srv"]}`
- `requirements.txt`: `pymongo==4.17.0`, `dnspython==2.8.0` added (SRV record resolution)
- Cluster: `portfolio-cluster.0dyprsm.mongodb.net` (AWS ap-southeast-1, M0 Free)
- Network access: `0.0.0.0/0` allowlist for Render deployment compatibility

## Related Decisions

- ADR-001: MongoDB alert logs provide the raw signal for future online LinUCB updates
- Future ADR: PostgreSQL (Supabase) gold layer for aggregated fraud analytics
