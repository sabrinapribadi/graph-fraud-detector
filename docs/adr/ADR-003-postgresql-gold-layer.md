# ADR-003: PostgreSQL (Supabase) Gold Layer — Aggregated Fraud Analytics

**Status:** Accepted  
**Date:** 2026-07-10  
**Deciders:** Sabrina Pribadi

---

## Context and Problem Statement

ADR-002 established MongoDB Atlas as the bronze layer for raw fraud alert events. For
analytics and reporting, individual documents must be aggregated into time-series metrics:

1. Daily alert counts (total, HIGH, MEDIUM), average fraud probability
2. Per-run RL threshold decisions (for evaluating bandit policy over time)
3. Model performance snapshots (AUC, F1) across retraining runs

MongoDB's aggregation pipeline can compute these on demand, but for the gold layer we
want pre-aggregated rows that can be queried with SQL, joined with external tables, and
served to BI tools without Map-Reduce overhead.

## Decision Drivers

- **Medallion architecture**: bronze (MongoDB raw events) → gold (PostgreSQL aggregates)
  demonstrates layered data pipeline design
- **SQL ergonomics**: relational tables are the natural home for structured analytical
  queries (GROUP BY date, rolling averages, LIMIT N)
- **Free tier availability**: Supabase offers a fully-managed PostgreSQL instance on a
  permanent free plan (500 MB, 2 CPUs, 1 GB RAM)
- **Fail-safe**: DB outage must never break `/predict`; same try/except pattern as mongo.py
- **Portfolio signal**: psycopg2 + Supabase demonstrates cloud-managed PostgreSQL skills

## Considered Options

### Option A — MongoDB aggregation on demand
Use `$group` and `$project` pipeline in `mongo.py` to compute daily totals at query time.

- **Pros**: No second database; single source of truth
- **Cons**: Every `/stats/daily` request runs a full collection scan; no SQL interface;
  can't easily join with external BI tools; doesn't demonstrate medallion architecture
- **Why not chosen**: Aggregation on demand doesn't constitute a gold layer; analytical
  queries should land on pre-aggregated structured data

### Option B — SQLite file in container
Write daily aggregates to a local SQLite file during each prediction.

- **Pros**: No external dependency; fast; standard SQL
- **Cons**: Ephemeral — lost on every Render deployment; not queryable from outside
  the container; doesn't demonstrate cloud skills
- **Why not chosen**: Ephemeral filesystem on Render makes SQLite unsuitable for a
  durable analytics store

### Option C — PostgreSQL via Supabase (chosen)
Use Supabase's free PostgreSQL instance as the gold layer; write from FastAPI via psycopg2.

- **Pros**: Durable across deployments; SQL-queryable from any client; free tier sufficient
  for portfolio demo; demonstrates medallion architecture; psycopg2 is sync (works cleanly
  with FastAPI thread pool); ON CONFLICT upsert keeps one row per day without application
  locking
- **Why chosen**: Correct fit for structured analytical data; demonstrates full medallion
  stack in a single project

## Decision Outcome

**Chosen: Option C — PostgreSQL on Supabase free tier, `portfolio_db` database.**

Connection: `aws-0-ap-southeast-1.pooler.supabase.com:6543` (transaction-mode pooler;
direct `db.*.supabase.co:5432` DNS had not propagated at time of setup).

### Schema

```sql
-- Aggregated daily fraud alert stats (gold layer)
CREATE TABLE fraud_alert_daily (
    id            SERIAL PRIMARY KEY,
    date          DATE UNIQUE NOT NULL,
    total_alerts  INT     NOT NULL DEFAULT 0,
    high_count    INT     NOT NULL DEFAULT 0,
    medium_count  INT     NOT NULL DEFAULT 0,
    avg_fraud_prob FLOAT  NOT NULL DEFAULT 0.0,
    created_at    TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Per-run RL threshold decisions (for bandit policy evaluation)
CREATE TABLE rl_threshold_decisions (
    id           SERIAL PRIMARY KEY,
    time_step    INT   NOT NULL,
    alpha        FLOAT NOT NULL,
    selected_tau FLOAT NOT NULL,
    oracle_tau   FLOAT NOT NULL,
    reward       FLOAT NOT NULL,
    created_at   TIMESTAMPTZ NOT NULL DEFAULT NOW()
);

-- Model performance across retraining runs
CREATE TABLE model_performance (
    id         SERIAL PRIMARY KEY,
    run_id     TEXT  NOT NULL,
    auc        FLOAT NOT NULL,
    f1         FLOAT NOT NULL,
    accuracy   FLOAT NOT NULL,
    hidden_dim INT   NOT NULL,
    num_layers INT   NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
);
```

### Upsert Pattern

Each `/predict` call for HIGH or MEDIUM risk increments today's row atomically:

```sql
INSERT INTO fraud_alert_daily (date, total_alerts, high_count, medium_count, avg_fraud_prob)
VALUES (%s, 1, %s, %s, %s)
ON CONFLICT (date) DO UPDATE SET
    total_alerts   = fraud_alert_daily.total_alerts + 1,
    high_count     = fraud_alert_daily.high_count   + EXCLUDED.high_count,
    medium_count   = fraud_alert_daily.medium_count + EXCLUDED.medium_count,
    avg_fraud_prob = (
        fraud_alert_daily.avg_fraud_prob * fraud_alert_daily.total_alerts + %s
    ) / (fraud_alert_daily.total_alerts + 1)
```

The rolling average formula `(old_avg × old_count + new_prob) / (old_count + 1)` is
computed in SQL to avoid race conditions from concurrent requests.

### Connection Pool

`src/db/postgres.py` uses `psycopg2.pool.ThreadedConnectionPool(min=1, max=5)`:
- Each write borrows a connection, executes with `autocommit=True`, and returns it
- `autocommit=True` is required by Supabase's transaction-mode pooler (port 6543),
  which does not support multi-statement transactions across pool slots
- Pool is initialised on FastAPI startup; smoke-tested with `SELECT 1`

### Fail-safe Design

```python
def connect():
    try:
        _pool = ThreadedConnectionPool(1, 5, uri, connect_timeout=8)
        ...
    except Exception:
        _pool = None  # all subsequent calls are no-ops

def upsert_daily_alert(...):
    if _pool is None:
        return          # silent skip — prediction still returned to caller
```

### New API Endpoint

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/stats/daily` | GET | Last N days of aggregated alert stats from PostgreSQL gold layer; `?days=30` (default) |

## Positive Consequences

- Daily fraud aggregates are now durable across Render deployments
- `GET /stats/daily` can power time-series dashboards without any application-level
  aggregation at query time
- The medallion architecture (MongoDB bronze → PostgreSQL gold) is now complete and
  demonstrable end-to-end
- `rl_threshold_decisions` table is ready to log bandit runs, enabling longitudinal
  evaluation of threshold policy quality over time

## Negative Consequences

- Supabase free tier pauses projects after 1 week of inactivity — first request after
  pause will fail until the project resumes (takes ~30 seconds); not an issue for
  portfolio demo traffic
- All writes are synchronous; for high-throughput production use async drivers
  (asyncpg or databases library) would be preferable
- The Supabase pooler (port 6543) is transaction-mode: cannot use session-level features
  (SET search_path, LISTEN/NOTIFY, advisory locks) — sufficient for this use case

## Implementation Notes

- `src/db/postgres.py`: `connect()`, `close()`, `upsert_daily_alert()`, `get_daily_summary()`
- `.env`: `SUPABASE_URI` (URL-encoded password: `@` → `%40`) + `POSTGRES_DB=portfolio_db`
- `pyproject.toml`: `psycopg2-binary = "^2.9.0"` added
- `requirements.txt`: `psycopg2-binary==2.9.10` added
- Project: `portfolio-db` on Supabase (ap-southeast-1, free tier)
- Network: Supabase free tier uses shared pooler — no IP allowlist required

## Related Decisions

- ADR-002: MongoDB Atlas receives raw alert events (bronze); PostgreSQL receives daily
  aggregates (gold) — these layers are complementary
- ADR-001: `rl_threshold_decisions` table provides a future landing zone for online
  LinUCB update logs, closing the loop between bandit decisions and observable outcomes
