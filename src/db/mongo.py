"""
MongoDB connection singleton for fraud_detector database.

Fail-safe: if MONGODB_URI is missing or Atlas is unreachable, all DB
operations are silently skipped — core prediction endpoints still work.
"""
import os
import logging
from datetime import datetime, timezone
from typing import Optional

from pymongo import MongoClient, DESCENDING
from pymongo.database import Database

logger = logging.getLogger(__name__)

_client: Optional[MongoClient] = None
_db: Optional[Database] = None

ALERTS_COL = "fraud_alerts"


def connect() -> None:
    global _client, _db
    uri = os.getenv("MONGODB_URI")
    db_name = os.getenv("MONGODB_DB", "fraud_detector")
    if not uri:
        logger.warning("MONGODB_URI not set — alert persistence disabled")
        return
    try:
        _client = MongoClient(uri, serverSelectionTimeoutMS=5000)
        _client.admin.command("ping")
        _db = _client[db_name]
        _db[ALERTS_COL].create_index([("timestamp", DESCENDING)])
        logger.info("MongoDB connected: %s", db_name)
    except Exception as exc:
        logger.warning("MongoDB unavailable (%s) — alert persistence disabled", exc)
        _client = None
        _db = None


def get_db() -> Optional[Database]:
    return _db


def close() -> None:
    global _client, _db
    if _client:
        _client.close()
    _client = None
    _db = None


def log_alert(node_id: str, fraud_probability: float, risk_level: str, prediction: str) -> None:
    if _db is None:
        return
    try:
        _db[ALERTS_COL].insert_one({
            "node_id": node_id,
            "fraud_probability": round(fraud_probability, 6),
            "risk_level": risk_level,
            "prediction": prediction,
            "timestamp": datetime.now(timezone.utc),
        })
    except Exception as exc:
        logger.warning("Alert log failed: %s", exc)
