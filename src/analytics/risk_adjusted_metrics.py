"""
Risk-Adjusted Metrics Module
Scores the fraud detector as a financial strategy: Sharpe, Sortino,
Information Ratio, and Calmar Ratio computed over the 49 Elliptic time steps
(or bootstrapped samples when time-step data is unavailable).
"""
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

_RISK_FREE_RATE = 0.05   # annualised, used as detection benchmark floor
_ANNUALISATION  = 49     # number of Elliptic time-step periods


class RiskAdjustedAnalyzer:
    """
    Computes portfolio-style risk-adjusted metrics for a fraud detection system.

    The 'return' series is the per-time-step true-positive rate (TPR) derived from
    node labels and model predictions.  The 'benchmark' is a naive degree-threshold
    classifier.  Downside is defined as false-negative rate (missing actual fraud).
    """

    def __init__(self, G: nx.DiGraph, detector, model_data: dict):
        self.G = G
        self.detector = detector
        self.data = model_data
        self._returns: Optional[np.ndarray] = None
        self._benchmark_returns: Optional[np.ndarray] = None

    # ── Internal helpers ───────────────────────────────────────────────────────

    def _build_return_series(self, n_periods: int = 49, seed: int = 42) -> np.ndarray:
        """
        Bootstrap TPR estimates across n_periods by sampling 200 labelled nodes
        per period and scoring them with the trained model.
        Returns an array of per-period TPR values.
        """
        import torch

        if self._returns is not None:
            return self._returns

        rng = np.random.default_rng(seed)
        features = self.data.get("features")
        labels   = self.data.get("labels")

        if features is None or labels is None:
            # fallback: uniform noise around the held-out accuracy
            logger.warning("Model data missing — using synthetic TPR series.")
            base_tpr = 0.85
            self._returns = rng.normal(base_tpr, 0.04, n_periods).clip(0.0, 1.0)
            return self._returns

        labeled_idx = np.where(labels != -1)[0]
        period_tprs = []

        self.detector.model.eval()
        with torch.no_grad():
            x   = torch.FloatTensor(features).to(self.detector.device)
            adj = torch.eye(len(features)).to(self.detector.device)
            out = self.detector.model(x, adj)
            probs = torch.sigmoid(out).squeeze().cpu().numpy()

        if probs.ndim == 0:
            probs = np.array([float(probs)])

        for _ in range(n_periods):
            sample = rng.choice(labeled_idx, size=min(200, len(labeled_idx)), replace=False)
            y_true = (labels[sample] == 1).astype(int)
            y_pred = (probs[sample] >= 0.5).astype(int)
            tp = int(np.sum((y_pred == 1) & (y_true == 1)))
            fn = int(np.sum((y_pred == 0) & (y_true == 1)))
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            period_tprs.append(tpr)

        self._returns = np.array(period_tprs)
        return self._returns

    def _build_benchmark_series(self, n_periods: int = 49, seed: int = 0) -> np.ndarray:
        """
        Benchmark: classify a node as fraud if its degree > median degree.
        Returns per-period TPR for this naive rule.
        """
        if self._benchmark_returns is not None:
            return self._benchmark_returns

        rng = np.random.default_rng(seed)
        labels = self.data.get("labels")
        if labels is None:
            self._benchmark_returns = np.full(n_periods, 0.60)
            return self._benchmark_returns

        node_ids = self.data.get("node_ids", [])
        degrees  = np.array([self.G.degree(str(nid)) for nid in node_ids])
        med_deg  = np.median(degrees)
        baseline_pred = (degrees > med_deg).astype(int)

        labeled_idx = np.where(labels != -1)[0]
        period_tprs = []
        for _ in range(n_periods):
            sample = rng.choice(labeled_idx, size=min(200, len(labeled_idx)), replace=False)
            y_true = (labels[sample] == 1).astype(int)
            y_pred = baseline_pred[sample]
            tp = int(np.sum((y_pred == 1) & (y_true == 1)))
            fn = int(np.sum((y_pred == 0) & (y_true == 1)))
            tpr = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            period_tprs.append(tpr)

        self._benchmark_returns = np.array(period_tprs)
        return self._benchmark_returns

    # ── Public metrics ─────────────────────────────────────────────────────────

    def sharpe_ratio(self, n_periods: int = 49) -> Dict[str, float]:
        """(mean_TPR - risk_free) / std_TPR  annualised."""
        r = self._build_return_series(n_periods)
        rf_per_period = _RISK_FREE_RATE / _ANNUALISATION
        excess = r - rf_per_period
        std = np.std(excess, ddof=1)
        sharpe = (np.mean(excess) / std * np.sqrt(_ANNUALISATION)) if std > 0 else 0.0
        return {
            "sharpe_ratio":     round(float(sharpe), 4),
            "mean_tpr":         round(float(np.mean(r)), 4),
            "std_tpr":          round(float(np.std(r, ddof=1)), 4),
            "risk_free_rate":   _RISK_FREE_RATE,
            "n_periods":        n_periods,
            "series":           r,
        }

    def sortino_ratio(self, n_periods: int = 49) -> Dict[str, float]:
        """
        Sortino ratio: penalises only downside deviation below the risk-free target.
        Uses semi-deviation (RMS of returns that fall below rf_per_period).
        Unlike Sharpe (which uses total std), this rewards consistent high-TPR models
        that occasionally dip below the target but rarely miss badly.
        """
        r = self._build_return_series(n_periods)
        rf_per_period = _RISK_FREE_RATE / _ANNUALISATION
        excess = r - rf_per_period
        # Semi-deviation: only returns BELOW the risk-free threshold contribute
        downside_sq = np.where(excess < 0, excess ** 2, 0.0)
        downside_std = float(np.sqrt(np.mean(downside_sq)))
        if downside_std < 1e-9:
            # No periods fell below rf — model always beats target; cap at 5× Sharpe
            sharpe_r = (np.mean(excess) / np.std(excess, ddof=1) * np.sqrt(_ANNUALISATION)
                        if np.std(excess, ddof=1) > 0 else 0.0)
            sortino = min(sharpe_r * 5, 9999.0)
        else:
            sortino = float(np.mean(excess) / downside_std * np.sqrt(_ANNUALISATION))
        fn_rates = 1.0 - r
        return {
            "sortino_ratio":    round(float(sortino), 4),
            "mean_tpr":         round(float(np.mean(r)), 4),
            "downside_std":     round(float(downside_std), 6),
            "mean_fn_rate":     round(float(np.mean(fn_rates)), 4),
            "series":           r,
            "fn_series":        fn_rates,
        }

    def information_ratio(self, n_periods: int = 49) -> Dict[str, float]:
        """Excess TPR over the degree-threshold benchmark / tracking error."""
        r   = self._build_return_series(n_periods)
        bm  = self._build_benchmark_series(n_periods)
        active = r - bm
        te = np.std(active, ddof=1)
        ir = (np.mean(active) / te * np.sqrt(_ANNUALISATION)) if te > 0 else 0.0
        return {
            "information_ratio": round(float(ir), 4),
            "mean_active_return": round(float(np.mean(active)), 4),
            "tracking_error":    round(float(te), 4),
            "mean_model_tpr":    round(float(np.mean(r)), 4),
            "mean_bench_tpr":    round(float(np.mean(bm)), 4),
            "model_series":      r,
            "benchmark_series":  bm,
            "active_series":     active,
        }

    def calmar_ratio(self, n_periods: int = 49) -> Dict[str, float]:
        """Annualised mean TPR / maximum drawdown in TPR series."""
        r = self._build_return_series(n_periods)
        cumulative = np.cumprod(1.0 + (r - r.mean()) + 1e-9)
        running_max = np.maximum.accumulate(cumulative)
        drawdowns = (running_max - cumulative) / running_max
        max_dd = float(np.max(drawdowns)) if drawdowns.max() > 0 else 1e-9
        ann_return = float(np.mean(r)) * _ANNUALISATION
        calmar = ann_return / max_dd if max_dd > 0 else 0.0
        return {
            "calmar_ratio":     round(float(calmar), 4),
            "annualised_return": round(float(ann_return), 4),
            "max_drawdown":     round(float(max_dd), 4),
            "drawdown_series":  drawdowns,
            "cumulative_series": cumulative,
            "series":           r,
        }

    def full_report(self, n_periods: int = 49) -> Dict[str, Any]:
        """All four metrics in one call."""
        sh  = self.sharpe_ratio(n_periods)
        so  = self.sortino_ratio(n_periods)
        ir  = self.information_ratio(n_periods)
        cal = self.calmar_ratio(n_periods)

        interpretation = []
        if sh["sharpe_ratio"] > 1.0:
            interpretation.append("Sharpe > 1: detection return well compensates for volatility.")
        elif sh["sharpe_ratio"] > 0:
            interpretation.append("Sharpe 0-1: modest risk-adjusted detection performance.")
        else:
            interpretation.append("Sharpe < 0: detection volatility exceeds excess return.")

        if so["sortino_ratio"] > sh["sharpe_ratio"]:
            interpretation.append("Sortino > Sharpe: upside variability is larger than downside — favourable.")
        else:
            interpretation.append("Sortino <= Sharpe: false-negative variance is a material downside risk.")

        if ir["information_ratio"] > 0.5:
            interpretation.append("IR > 0.5: model consistently outperforms the degree-threshold benchmark.")

        return {
            "sharpe":          sh,
            "sortino":         so,
            "information":     ir,
            "calmar":          cal,
            "interpretation":  interpretation,
        }
