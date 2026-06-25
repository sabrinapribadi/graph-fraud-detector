"""
Loss Forecasting Module
Forecasts future fraud loss over the 49 Elliptic time steps using Prophet
(with an exponential-smoothing fallback when Prophet is not installed).
"""
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, Any, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    logger.info("Prophet not installed — using exponential-smoothing fallback.")


# ── Exponential smoothing (Holt-Winters, no external dep) ─────────────────────

def _holt_winters(series: np.ndarray, n_forecast: int,
                  alpha: float = 0.3, beta: float = 0.1) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Double exponential smoothing (trend-aware)."""
    n = len(series)
    level = np.zeros(n)
    trend = np.zeros(n)
    level[0] = series[0]
    trend[0] = series[1] - series[0] if n > 1 else 0.0

    for t in range(1, n):
        prev_l = level[t - 1]
        prev_t = trend[t - 1]
        level[t] = alpha * series[t] + (1 - alpha) * (prev_l + prev_t)
        trend[t] = beta * (level[t] - prev_l) + (1 - beta) * prev_t

    fc   = np.array([level[-1] + i * trend[-1] for i in range(1, n_forecast + 1)])
    resid_std = np.std(series - (level + trend), ddof=1) if n > 2 else series.std()
    lo = fc - 1.96 * resid_std
    hi = fc + 1.96 * resid_std
    return np.maximum(fc, 0), np.maximum(lo, 0), np.maximum(hi, 0)


class LossForecaster:
    """
    Derives a fraud-loss time series from the NetworkX graph and model predictions,
    then forecasts forward using Prophet (or Holt-Winters fallback).

    Time axis: node degrees are used as a proxy for transaction activity per
    time step (the Elliptic dataset doesn't expose an explicit timestamp per node,
    but nodes were added across 49 steps which maps to degree quintiles here).
    """

    N_STEPS   = 49    # Elliptic dataset time steps
    LOSS_PER_FRAUD = 10_000  # default USD loss per illicit node

    def __init__(
        self,
        G: nx.DiGraph,
        detector,
        model_data: dict,
        loss_per_fraud: float = LOSS_PER_FRAUD,
    ):
        self.G = G
        self.detector = detector
        self.data = model_data
        self.loss_per_fraud = loss_per_fraud
        self._ts: Optional[pd.DataFrame] = None

    # ── Time-series construction ───────────────────────────────────────────────

    def build_time_series(self) -> pd.DataFrame:
        """
        Splits nodes into N_STEPS buckets by degree percentile and computes
        estimated fraud loss per bucket:
            loss_t = illicit_count_t × fraud_prob_mean_t × loss_per_fraud
        """
        if self._ts is not None:
            return self._ts

        import torch

        nodes      = list(self.G.nodes())
        degrees    = np.array([self.G.degree(n) for n in nodes])
        labels     = np.array([self.G.nodes[n].get("label", -1) for n in nodes])
        n_nodes    = len(nodes)

        # Get model fraud probabilities if available
        features = self.data.get("features")
        node_ids = self.data.get("node_ids")
        if features is not None and self.detector is not None:
            try:
                self.detector.model.eval()
                with torch.no_grad():
                    x = torch.FloatTensor(features).to(self.detector.device)
                    adj = torch.eye(len(features)).to(self.detector.device)
                    out = self.detector.model(x, adj)
                    probs_model = torch.sigmoid(out).squeeze().cpu().numpy()
                if probs_model.ndim == 0:
                    probs_model = np.array([float(probs_model)])
                node_id_to_prob = {
                    str(nid): float(p)
                    for nid, p in zip(node_ids, probs_model)
                }
            except Exception as e:
                logger.warning(f"Could not get model probabilities: {e}")
                node_id_to_prob = {}
        else:
            node_id_to_prob = {}

        fraud_probs = np.array([node_id_to_prob.get(str(n), 0.21) for n in nodes])

        # Assign each node to a time-step bucket by degree percentile
        buckets = pd.qcut(degrees, q=self.N_STEPS, labels=False, duplicates="drop")
        n_actual = buckets.nunique() if hasattr(buckets, "nunique") else self.N_STEPS

        # Elliptic dataset: 49 time steps ≈ bi-weekly snapshots, Jan 2011 – Jan 2013
        _STEP_ORIGIN = pd.Timestamp("2011-01-01")
        _STEP_FREQ   = pd.Timedelta(weeks=2)

        rows = []
        for step in range(n_actual):
            mask = (buckets == step)
            if mask.sum() == 0:
                continue
            illicit_in_step  = int(np.sum(labels[mask] == 1))
            mean_fraud_prob  = float(np.mean(fraud_probs[mask]))
            est_loss = illicit_in_step * mean_fraud_prob * self.loss_per_fraud
            rows.append({"ds": _STEP_ORIGIN + _STEP_FREQ * step,
                         "y": est_loss,
                         "step": step + 1,
                         "illicit_count": illicit_in_step,
                         "mean_fraud_prob": mean_fraud_prob})

        df = pd.DataFrame(rows)
        # smooth to reduce noise from bucket imbalance
        df["y"] = df["y"].rolling(window=3, min_periods=1, center=True).mean()
        self._ts = df
        return df

    # ── Prophet forecast ───────────────────────────────────────────────────────

    def forecast_prophet(self, n_periods: int = 10) -> Dict[str, Any]:
        df = self.build_time_series()
        if not PROPHET_AVAILABLE:
            logger.info("Prophet not available — delegating to Holt-Winters.")
            return self.forecast_holt_winters(n_periods)

        m = Prophet(
            yearly_seasonality=False,
            weekly_seasonality=False,
            daily_seasonality=False,
            changepoint_prior_scale=0.3,
            interval_width=0.80,
        )
        m.fit(df[["ds", "y"]])
        future = m.make_future_dataframe(periods=n_periods, freq="W")
        fc     = m.predict(future)

        return {
            "method":     "Prophet",
            "history":    df,
            "forecast":   fc[["ds", "yhat", "yhat_lower", "yhat_upper"]].tail(n_periods),
            "full":       fc[["ds", "yhat", "yhat_lower", "yhat_upper"]],
            "components": fc[["ds", "trend"]],
            "n_periods":  n_periods,
        }

    # ── Holt-Winters fallback ──────────────────────────────────────────────────

    def forecast_holt_winters(self, n_periods: int = 10) -> Dict[str, Any]:
        df   = self.build_time_series()
        vals = df["y"].values

        fc, lo, hi = _holt_winters(vals, n_periods)
        last_date = df["ds"].iloc[-1]
        future_dates = pd.date_range(
            start=last_date + pd.Timedelta(weeks=2),
            periods=n_periods, freq="2W",
        )

        history_fc = pd.DataFrame({
            "ds": df["ds"],
            "yhat": vals,
            "yhat_lower": vals * 0.85,
            "yhat_upper": vals * 1.15,
        })
        future_df = pd.DataFrame({
            "ds": future_dates,
            "yhat": fc, "yhat_lower": lo, "yhat_upper": hi,
        })
        full = pd.concat([history_fc, future_df], ignore_index=True)

        return {
            "method":    "Holt-Winters (fallback)",
            "history":   df,
            "forecast":  future_df,
            "full":      full,
            "n_periods": n_periods,
        }

    # ── Unified entry point ────────────────────────────────────────────────────

    def forecast(self, n_periods: int = 10) -> Dict[str, Any]:
        if PROPHET_AVAILABLE:
            return self.forecast_prophet(n_periods)
        return self.forecast_holt_winters(n_periods)

    # ── Summary stats ──────────────────────────────────────────────────────────

    def summary_stats(self, result: Dict[str, Any]) -> Dict[str, float]:
        hist = result["history"]["y"]
        fc   = result["forecast"]["yhat"]
        return {
            "historical_mean":    round(float(hist.mean()), 2),
            "historical_peak":    round(float(hist.max()), 2),
            "forecast_mean":      round(float(fc.mean()), 2),
            "forecast_peak":      round(float(fc.max()), 2),
            "forecast_vs_hist":   round(float(fc.mean() / hist.mean()), 3) if hist.mean() > 0 else 0.0,
            "trend_direction":    "increasing" if fc.mean() > hist.mean() else "decreasing",
        }
