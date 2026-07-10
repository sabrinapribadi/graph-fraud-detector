"""
Per-time-step context features for the threshold bandit.

6 features extracted from the 49 Elliptic time steps:
  illicit_rate       — illicit / (illicit + licit) this step
  labeled_fraction   — labeled nodes / total nodes this step
  tx_count_norm      — node count / max node count (volume proxy)
  illicit_velocity   — Δillicit_rate from previous step (momentum)
  cumul_illicit_rate — expanding mean illicit rate up to this step
  bias               — 1.0 (LinUCB intercept)
"""
import numpy as np
import pandas as pd
from typing import Tuple

FEATURE_NAMES = [
    "illicit_rate",
    "labeled_fraction",
    "tx_count_norm",
    "illicit_velocity",
    "cumul_illicit_rate",
    "bias",
]


def build_time_step_contexts(
    features_df: pd.DataFrame,
    classes_df: pd.DataFrame,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """
    Returns
    -------
    contexts : np.ndarray, shape (n_steps, 6)
    stats_df : pd.DataFrame with per-step statistics
    """
    tx = features_df[["0", "1"]].copy()
    tx.columns = ["txId", "time_step"]
    tx["txId"] = tx["txId"].astype(str)

    cls = classes_df.copy()
    cls["txId"] = cls["txId"].astype(str)

    merged = tx.merge(cls, on="txId", how="left")
    time_steps = sorted(merged["time_step"].unique())

    records = []
    for t in time_steps:
        sub = merged[merged["time_step"] == t]
        illicit = int((sub["class"] == "1").sum())
        licit = int((sub["class"] == "2").sum())
        total = len(sub)
        labeled = illicit + licit
        records.append({
            "time_step": int(t),
            "illicit": illicit,
            "licit": licit,
            "total": total,
            "labeled": labeled,
            "illicit_rate": illicit / labeled if labeled > 0 else 0.0,
            "labeled_fraction": labeled / total if total > 0 else 0.0,
            "tx_count": total,
        })

    stats = pd.DataFrame(records)
    stats["tx_count_norm"] = stats["tx_count"] / stats["tx_count"].max()
    stats["illicit_velocity"] = stats["illicit_rate"].diff().fillna(0.0)
    stats["cumul_illicit_rate"] = stats["illicit_rate"].expanding().mean()

    feature_cols = [
        "illicit_rate", "labeled_fraction", "tx_count_norm",
        "illicit_velocity", "cumul_illicit_rate",
    ]
    ctx_base = stats[feature_cols].values
    bias = np.ones((len(stats), 1))
    contexts = np.hstack([ctx_base, bias]).astype(np.float32)

    return contexts, stats
