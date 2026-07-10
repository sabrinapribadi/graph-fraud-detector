"""
ThresholdBanditSelector — LinUCB bandit for adaptive fraud detection threshold selection.

Problem
-------
The GNN outputs a fraud probability p ∈ [0,1] per node. A fixed threshold τ
classifies nodes as FRAUD (p ≥ τ) or LEGITIMATE (p < τ). Different temporal
regimes (varying illicit rates, transaction volumes, velocity shifts) favour
different thresholds because the cost trade-off between false positives
(analyst time wasted) and false negatives (missed fraud) shifts with regime.

Solution
--------
LinUCB contextual bandit with 5 threshold arms. At each of the 49 Elliptic
time steps the bandit observes a 6-dim regime context, selects a threshold,
and receives a normalised reward based on the cost-minimising performance
of that threshold on the labeled nodes at that step.

Offline training is valid because all arm rewards are observable from
pre-computed class labels — no simulator needed.

Fraud probability simulation
----------------------------
The trained GraphSAGE operates on a 3 000-node sampled subgraph; running
inference on all 203 769 nodes would require rebuilding the full adjacency
matrix. Instead we simulate per-node probabilities from Beta distributions
whose parameters match the GNN's empirical output distribution:
  illicit nodes : Beta(8, 2)  — mean 0.80, std ≈ 0.12
  licit nodes   : Beta(2, 8)  — mean 0.20, std ≈ 0.12
Seed 42 + time_step index for reproducibility.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List

from src.rl.bandit import LinUCBBandit
from src.rl.time_step_features import build_time_step_contexts, FEATURE_NAMES

THRESHOLDS: List[float] = [0.3, 0.5, 0.6, 0.7, 0.8]
ARM_LABELS: List[str] = [f"τ={t}" for t in THRESHOLDS]

FP_COST = 1.0    # false alarm: analyst time wasted
FN_COST = 10.0   # missed fraud: regulatory + financial loss (10× worse)

_ILLICIT_ALPHA, _ILLICIT_BETA = 8, 2
_LICIT_ALPHA, _LICIT_BETA = 2, 8


def _simulate_probs(
    n_illicit: int, n_licit: int, rng: np.random.Generator
) -> tuple:
    return (
        rng.beta(_ILLICIT_ALPHA, _ILLICIT_BETA, n_illicit),
        rng.beta(_LICIT_ALPHA, _LICIT_BETA, n_licit),
    )


def _arm_cost(
    illicit_probs: np.ndarray, licit_probs: np.ndarray, threshold: float
) -> float:
    tp = (illicit_probs >= threshold).sum()
    fn = (illicit_probs < threshold).sum()
    fp = (licit_probs >= threshold).sum()
    tn = (licit_probs < threshold).sum()
    fp_rate = fp / (fp + tn + 1e-9)
    fn_rate = fn / (fn + tp + 1e-9)
    return float(FP_COST * fp_rate + FN_COST * fn_rate)


class ThresholdBanditSelector:

    def __init__(self, data_dir: Path, alpha: float = 1.0):
        self.data_dir = Path(data_dir)
        self.alpha = alpha
        self._results: Dict[str, Any] = {}

    def _load_data(self):
        features = pd.read_parquet(self.data_dir / "data" / "processed" / "features.parquet")
        classes = pd.read_parquet(self.data_dir / "data" / "processed" / "classes.parquet")
        return features, classes

    def train(self) -> Dict[str, Any]:
        features_df, classes_df = self._load_data()
        contexts, stats_df = build_time_step_contexts(features_df, classes_df)

        n_steps, n_features = contexts.shape
        n_arms = len(THRESHOLDS)

        # Join txId → time_step + class for per-step labeled nodes
        tx = features_df[["0", "1"]].copy()
        tx.columns = ["txId", "time_step"]
        tx["txId"] = tx["txId"].astype(str)
        cls = classes_df.copy()
        cls["txId"] = cls["txId"].astype(str)
        merged = tx.merge(cls, on="txId", how="left")
        time_steps = stats_df["time_step"].tolist()

        # Build cost matrix (n_steps × n_arms)
        cost_matrix = np.zeros((n_steps, n_arms))
        for i, t in enumerate(time_steps):
            sub = merged[merged["time_step"] == t]
            n_illicit = int((sub["class"] == "1").sum())
            n_licit = int((sub["class"] == "2").sum())
            if n_illicit == 0 or n_licit == 0:
                continue
            rng = np.random.default_rng(42 + i)
            ill_p, lit_p = _simulate_probs(n_illicit, n_licit, rng)
            for j, tau in enumerate(THRESHOLDS):
                cost_matrix[i, j] = _arm_cost(ill_p, lit_p, tau)

        # Normalise per-step to [0, 1] reward: 1.0 = lowest cost arm
        reward_matrix = np.zeros_like(cost_matrix)
        for i in range(n_steps):
            row = cost_matrix[i]
            span = row.max() - row.min()
            if span > 1e-9:
                reward_matrix[i] = (row.max() - row) / span
            else:
                reward_matrix[i] = np.full(n_arms, 0.5)

        # LinUCB sequential training (select BEFORE update — no look-ahead bias)
        bandit = LinUCBBandit(n_arms=n_arms, n_features=n_features, alpha=self.alpha)
        selected_arms = np.zeros(n_steps, dtype=int)
        oracle_arms = reward_matrix.argmax(axis=1)

        for i in range(n_steps):
            ctx = contexts[i]
            selected_arms[i], _ = bandit.select_arm(ctx)
            for a in range(n_arms):
                bandit.update(a, ctx, reward_matrix[i, a])

        bandit_rewards = reward_matrix[np.arange(n_steps), selected_arms]
        oracle_rewards = reward_matrix[np.arange(n_steps), oracle_arms]
        static_best_arm = int(reward_matrix.sum(axis=0).argmax())
        static_rewards = reward_matrix[:, static_best_arm]
        random_rewards = reward_matrix.mean(axis=1)

        cumul_regret_bandit = np.cumsum(oracle_rewards - bandit_rewards)
        cumul_regret_static = np.cumsum(oracle_rewards - static_rewards)
        cumul_regret_random = np.cumsum(oracle_rewards - random_rewards)

        weights = [[float(v) for v in w] for w in bandit.get_weights()]
        ctx_no_bias = contexts[:, :-1]

        self._results = {
            "n_steps": n_steps,
            "n_arms": n_arms,
            "arm_labels": ARM_LABELS,
            "thresholds": THRESHOLDS,
            "time_steps": [int(t) for t in time_steps],
            "reward_matrix": reward_matrix.tolist(),
            "selected_arms": selected_arms.tolist(),
            "oracle_arms": oracle_arms.tolist(),
            "selected_thresholds": [THRESHOLDS[a] for a in selected_arms],
            "oracle_thresholds": [THRESHOLDS[a] for a in oracle_arms],
            "bandit_rewards": bandit_rewards.tolist(),
            "oracle_rewards": oracle_rewards.tolist(),
            "static_rewards": static_rewards.tolist(),
            "static_best_arm": static_best_arm,
            "static_best_label": ARM_LABELS[static_best_arm],
            "cumul_regret_bandit": cumul_regret_bandit.tolist(),
            "cumul_regret_static": cumul_regret_static.tolist(),
            "cumul_regret_random": cumul_regret_random.tolist(),
            "arm_counts": {ARM_LABELS[a]: int((selected_arms == a).sum()) for a in range(n_arms)},
            "weights": weights,
            "feat_names": FEATURE_NAMES[:-1],
            "feat_min": ctx_no_bias.min(axis=0).tolist(),
            "feat_max": ctx_no_bias.max(axis=0).tolist(),
            "feat_mean": ctx_no_bias.mean(axis=0).tolist(),
            "kpis": {
                "avg_bandit_reward": float(bandit_rewards.mean()),
                "avg_oracle_reward": float(oracle_rewards.mean()),
                "avg_static_reward": float(static_rewards.mean()),
                "avg_random_reward": float(random_rewards.mean()),
                "final_regret_bandit": float(cumul_regret_bandit[-1]),
                "final_regret_static": float(cumul_regret_static[-1]),
                "pct_vs_random": float(
                    (bandit_rewards.mean() - random_rewards.mean())
                    / (random_rewards.mean() + 1e-9) * 100
                ),
            },
            "alpha": self.alpha,
            "fp_cost": FP_COST,
            "fn_cost": FN_COST,
        }
        return self._results

    def get_results(self) -> Dict[str, Any]:
        if not self._results:
            self.train()
        return self._results

    def predict_from_features(self, context_5d: np.ndarray) -> Dict[str, Any]:
        """What-if: given 5 context features (no bias), return arm scores and recommendation."""
        if not self._results:
            self.train()
        ctx = np.append(context_5d, 1.0).astype(np.float64)
        scores = [float(np.array(w) @ ctx) for w in self._results["weights"]]
        best = int(np.argmax(scores))
        return {
            "recommended_threshold": THRESHOLDS[best],
            "recommended_label": ARM_LABELS[best],
            "scores": scores,
            "arm_labels": ARM_LABELS,
        }
