"""
PostgreSQL (Supabase) gold layer — aggregated fraud analytics.

Uses a ThreadedConnectionPool (1–5 connections) against Supabase's
transaction-mode pooler (port 6543). Each borrowed connection is
returned immediately after the query — the intended usage pattern
for serverless/short-lived callers on the transaction pooler.

Fail-safe: if SUPABASE_URI is missing or Supabase is unreachable,
all writes and reads are silently skipped — core prediction endpoints
remain unaffected.
"""
import os
import logging
from datetime import date
from typing import List, Dict, Any, Optional

import psycopg2
import psycopg2.pool
from psycopg2.extras import RealDictCursor

logger = logging.getLogger(__name__)

_pool: Optional[psycopg2.pool.ThreadedConnectionPool] = None


def connect() -> None:
    global _pool
    uri = os.getenv("SUPABASE_URI")
    if not uri:
        logger.warning("SUPABASE_URI not set — PostgreSQL logging disabled")
        return
    try:
        _pool = psycopg2.pool.ThreadedConnectionPool(1, 5, uri, connect_timeout=8)
        # smoke-test
        conn = _pool.getconn()
        with conn.cursor() as cur:
            cur.execute("SELECT 1")
        _pool.putconn(conn)
        logger.info("PostgreSQL (Supabase) pool ready")
    except Exception as exc:
        logger.warning("PostgreSQL unavailable (%s) — logging disabled", exc)
        _pool = None


def close() -> None:
    global _pool
    if _pool:
        _pool.closeall()
    _pool = None


def upsert_daily_alert(fraud_probability: float, risk_level: str) -> None:
    """Increment today's aggregated row — rolling average for avg_fraud_prob."""
    if _pool is None:
        return
    try:
        conn = _pool.getconn()
        try:
            conn.autocommit = True
            high   = 1 if risk_level == "HIGH"   else 0
            medium = 1 if risk_level == "MEDIUM" else 0
            today  = date.today().isoformat()
            sql = """
                INSERT INTO fraud_alert_daily
                    (date, total_alerts, high_count, medium_count, avg_fraud_prob)
                VALUES (%s, 1, %s, %s, %s)
                ON CONFLICT (date) DO UPDATE SET
                    total_alerts   = fraud_alert_daily.total_alerts + 1,
                    high_count     = fraud_alert_daily.high_count   + %s,
                    medium_count   = fraud_alert_daily.medium_count + %s,
                    avg_fraud_prob = (
                        fraud_alert_daily.avg_fraud_prob
                        * fraud_alert_daily.total_alerts
                        + %s
                    ) / (fraud_alert_daily.total_alerts + 1)
            """
            with conn.cursor() as cur:
                cur.execute(sql, (
                    today, high, medium, fraud_probability,
                    high, medium, fraud_probability,
                ))
        finally:
            _pool.putconn(conn)
    except Exception as exc:
        logger.warning("PostgreSQL upsert failed: %s", exc)


def get_daily_summary(days: int = 30) -> List[Dict[str, Any]]:
    """Return last N days of aggregated alert stats from the gold layer."""
    if _pool is None:
        return []
    try:
        conn = _pool.getconn()
        try:
            with conn.cursor(cursor_factory=RealDictCursor) as cur:
                cur.execute("""
                    SELECT
                        date::text,
                        total_alerts,
                        high_count,
                        medium_count,
                        ROUND(avg_fraud_prob::numeric, 4) AS avg_fraud_prob
                    FROM fraud_alert_daily
                    ORDER BY date DESC
                    LIMIT %s
                """, (days,))
                rows = [dict(r) for r in cur.fetchall()]
        finally:
            _pool.putconn(conn)
        return rows
    except Exception as exc:
        logger.warning("PostgreSQL query failed: %s", exc)
        return []
