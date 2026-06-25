"""
One-time script: convert raw Elliptic CSVs to zstd-compressed parquet in data/processed/.

Run once before the first deploy (or after updating raw data):
    python scripts/preprocess_data.py

Output:
    data/processed/features.parquet   ~85 MB  (vs 658 MB CSV)
    data/processed/edgelist.parquet   ~1 MB
    data/processed/classes.parquet    ~0.5 MB
"""
import sys
import os
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

RAW_DIR = project_root / "data" / "raw" / "elliptic_bitcoin_dataset"
PROCESSED_DIR = project_root / "data" / "processed"


def _mb(path: Path) -> float:
    return os.path.getsize(path) / 1e6


def main() -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # ── Features ─────────────────────────────────────────────────────────────
    src = RAW_DIR / "elliptic_txs_features.csv"
    dst = PROCESSED_DIR / "features.parquet"
    logger.info(f"Reading {src.name}  ({_mb(src):.0f} MB) ...")
    features = pd.read_csv(src, header=None)
    features.to_parquet(dst, compression="zstd", index=False)
    logger.info(f"  -> {dst.name}  {_mb(dst):.1f} MB")

    # ── Classes ───────────────────────────────────────────────────────────────
    src = RAW_DIR / "elliptic_txs_classes.csv"
    dst = PROCESSED_DIR / "classes.parquet"
    logger.info(f"Reading {src.name}  ({_mb(src):.1f} MB) ...")
    classes = pd.read_csv(src)
    classes.to_parquet(dst, compression="zstd", index=False)
    logger.info(f"  -> {dst.name}  {_mb(dst):.1f} MB")

    # ── Edgelist ──────────────────────────────────────────────────────────────
    src = RAW_DIR / "elliptic_txs_edgelist.csv"
    dst = PROCESSED_DIR / "edgelist.parquet"
    logger.info(f"Reading {src.name}  ({_mb(src):.1f} MB) ...")
    edgelist = pd.read_csv(src, header=None)
    if edgelist.shape[1] == 3:
        edgelist.columns = ["source", "target", "timestamp"]
    else:
        edgelist.columns = ["source", "target"]
        edgelist["timestamp"] = 0
    edgelist["source"] = edgelist["source"].astype(str)
    edgelist["target"] = edgelist["target"].astype(str)
    edgelist.to_parquet(dst, compression="zstd", index=False)
    logger.info(f"  -> {dst.name}  {_mb(dst):.1f} MB")

    total = sum(
        _mb(PROCESSED_DIR / f)
        for f in ("features.parquet", "classes.parquet", "edgelist.parquet")
    )
    logger.info(f"Done. Total parquet size: {total:.1f} MB")


if __name__ == "__main__":
    main()
