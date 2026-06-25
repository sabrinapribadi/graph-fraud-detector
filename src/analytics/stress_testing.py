"""
Stress Testing Module
Applies named macroeconomic crisis scenarios to the Monte Carlo risk parameters
and compares outcomes side-by-side.
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Any
import logging

from src.analytics.risk_analysis import QuantitativeRiskAnalyzer

logger = logging.getLogger(__name__)


@dataclass
class StressScenario:
    name: str
    description: str
    pd_multiplier: float    # multiplier on base fraud probability
    lgd_delta: float        # additive shift on LGD  (e.g. +0.20 = worse recovery)
    ead_multiplier: float   # multiplier on average loss per fraud
    volume_shock: float     # multiplier on transaction count
    delay_delta: int        # additive days added to detection time


# Pre-defined scenario library
SCENARIOS: List[StressScenario] = [
    StressScenario(
        name="Baseline",
        description="Current model conditions — no shocks applied.",
        pd_multiplier=1.0, lgd_delta=0.0, ead_multiplier=1.0,
        volume_shock=1.0, delay_delta=0,
    ),
    StressScenario(
        name="2008 Financial Crisis",
        description=(
            "Credit crunch: fraud triples as controls break down, "
            "recovery on seized assets deteriorates, transaction volume collapses 40%, "
            "and enforcement response is delayed by 30 days."
        ),
        pd_multiplier=3.0, lgd_delta=0.20, ead_multiplier=0.8,
        volume_shock=0.6, delay_delta=30,
    ),
    StressScenario(
        name="COVID-19 Pandemic",
        description=(
            "Digital fraud surge (1.8× baseline) as remote activity rises; "
            "on-chain volume drops 30%; detection capacity constrained by 15 days."
        ),
        pd_multiplier=1.8, lgd_delta=0.10, ead_multiplier=1.0,
        volume_shock=0.7, delay_delta=15,
    ),
    StressScenario(
        name="Crypto Winter",
        description=(
            "Market collapse reduces honest activity 60%; illicit hubs concentrate "
            "transaction flows; LGD worsens as seized assets lose value."
        ),
        pd_multiplier=2.2, lgd_delta=0.15, ead_multiplier=0.5,
        volume_shock=0.4, delay_delta=20,
    ),
    StressScenario(
        name="Regulatory Crackdown",
        description=(
            "Enhanced monitoring halves the fraud rate; faster enforcement "
            "cuts detection delay by 10 days; slight LGD improvement."
        ),
        pd_multiplier=0.5, lgd_delta=-0.10, ead_multiplier=1.0,
        volume_shock=1.0, delay_delta=-10,
    ),
]


class StressTester:
    """
    Runs each StressScenario against base risk parameters and collects
    Monte Carlo + Expected Loss + TVM results for comparison.
    """

    def __init__(
        self,
        n_transactions: int,
        fraud_probability: float,
        avg_loss_per_fraud: float,
        lgd: float = 0.45,
        detection_time: float = 30.0,
        exposure_per_transaction: float = 1.0,
        n_simulations: int = 5000,
    ):
        self.base = dict(
            n_transactions=n_transactions,
            fraud_probability=fraud_probability,
            avg_loss_per_fraud=avg_loss_per_fraud,
            lgd=lgd,
            detection_time=detection_time,
            exposure_per_transaction=exposure_per_transaction,
        )
        self.n_simulations = n_simulations
        self.analyzer = QuantitativeRiskAnalyzer()
        self.results: Dict[str, Any] = {}

    # ── Single scenario ────────────────────────────────────────────────────────

    def run_scenario(self, scenario: StressScenario) -> Dict[str, Any]:
        b = self.base

        pd_ = float(np.clip(b["fraud_probability"] * scenario.pd_multiplier, 0.0, 1.0))
        lgd = float(np.clip(b["lgd"] + scenario.lgd_delta, 0.0, 1.0))
        n   = max(1, int(b["n_transactions"] * scenario.volume_shock))
        ead = float(b["avg_loss_per_fraud"] * scenario.ead_multiplier)
        delay = max(1.0, b["detection_time"] + scenario.delay_delta)

        mc = self.analyzer.monte_carlo_simulation(
            n_transactions=n,
            fraud_probability=pd_,
            avg_loss_per_fraud=ead,
            n_simulations=self.n_simulations,
        )
        el = self.analyzer.calculate_expected_loss(
            exposure=n * b["exposure_per_transaction"] * ead,
            probability_default=pd_,
            loss_given_default=lgd,
        )
        tvm = self.analyzer.time_value_money_adjustment(
            expected_loss=mc["mean_loss"],
            detection_time=delay,
        )

        # severity: ratio of total_loss to baseline (computed after all scenarios run)
        return {
            "scenario":        scenario,
            "stressed_pd":     pd_,
            "stressed_lgd":    lgd,
            "stressed_n":      n,
            "stressed_ead":    ead,
            "stressed_delay":  delay,
            "mean_loss":       mc["mean_loss"],
            "var_95":          np.percentile(mc["simulations"], 95),
            "expected_shortfall": mc["expected_shortfall"],
            "expected_loss":   el["expected_loss"],
            "tvm_cost":        tvm["time_value_cost"],
            "total_loss":      mc["mean_loss"] + tvm["time_value_cost"],
            "simulations":     mc["simulations"],
        }

    # ── All scenarios ──────────────────────────────────────────────────────────

    def run_all(self, scenarios: List[StressScenario] = SCENARIOS) -> Dict[str, Any]:
        logger.info("Running %d stress scenarios...", len(scenarios))
        self.results = {s.name: self.run_scenario(s) for s in scenarios}

        # add severity relative to baseline
        baseline_loss = self.results["Baseline"]["total_loss"] or 1.0
        for r in self.results.values():
            r["severity_ratio"] = r["total_loss"] / baseline_loss
        return self.results

    # ── Summary DataFrame ─────────────────────────────────────────────────────

    def summary_df(self) -> pd.DataFrame:
        if not self.results:
            self.run_all()
        rows = []
        for name, r in self.results.items():
            rows.append({
                "Scenario":          name,
                "PD":                round(r["stressed_pd"], 4),
                "LGD":               round(r["stressed_lgd"], 4),
                "Transactions":      r["stressed_n"],
                "Mean Loss":         round(r["mean_loss"], 2),
                "VaR (95%)":         round(r["var_95"], 2),
                "Exp. Shortfall":    round(r["expected_shortfall"], 2),
                "Expected Loss":     round(r["expected_loss"], 2),
                "TVM Cost":          round(r["tvm_cost"], 2),
                "Total Loss":        round(r["total_loss"], 2),
                "Severity vs Base":  round(r["severity_ratio"], 2),
            })
        return pd.DataFrame(rows)
