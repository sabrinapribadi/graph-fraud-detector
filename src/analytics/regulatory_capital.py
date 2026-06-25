"""
Regulatory Capital Module
Basel III / IRB-inspired capital requirement calculations for fraud risk.

Two approaches:
  1. Standardised Approach (SA): flat risk-weight tables from Basel III.
  2. Internal Ratings-Based (IRB): Vasicek single-factor model formula —
     the same formula banks use for credit capital, adapted for fraud PD/LGD.
"""
import numpy as np
import pandas as pd
from scipy.stats import norm
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)

# Basel III standardised risk weights for operational risk (fraud) by exposure band
_SA_RISK_WEIGHTS = {
    "retail":          0.75,   # 75% — standard retail
    "corporate":       1.00,   # 100%
    "high_risk":       1.50,   # 150%
    "crypto_exchange": 1.00,   # treated as unrated corporate
}

# Minimum capital ratio under Basel III
_MIN_CAPITAL_RATIO = 0.08   # 8% (Pillar 1 minimum)
_CAPITAL_BUFFER    = 0.025  # 2.5% conservation buffer
_TOTAL_MIN         = _MIN_CAPITAL_RATIO + _CAPITAL_BUFFER   # 10.5%

# Asset correlation bounds (Basel IRB for retail)
_RHO_MIN = 0.03
_RHO_MAX = 0.16


def _vasicek_capital(
    pd_: float,
    lgd: float,
    ead: float,
    maturity: float = 1.0,
    rho: float = 0.12,
    confidence: float = 0.999,
) -> Dict[str, float]:
    """
    Basel II / III IRB formula (Vasicek single-factor model).
    Returns capital requirement and risk-weighted asset.
    """
    if pd_ <= 0 or pd_ >= 1:
        pd_ = np.clip(pd_, 1e-6, 1 - 1e-6)

    G_pd  = norm.ppf(pd_)
    G_con = norm.ppf(confidence)

    # Conditional PD under the worst-case systematic factor
    pd_stress = norm.cdf(
        (G_pd + np.sqrt(rho) * G_con) / np.sqrt(1 - rho)
    )

    # Maturity adjustment (simplified — 1-year instruments)
    b = (0.11852 - 0.05478 * np.log(pd_)) ** 2
    ma = (1 + (maturity - 2.5) * b) / (1 - 1.5 * b)

    # Unexpected loss (capital charge per unit of EAD)
    ul = (lgd * pd_stress - lgd * pd_) * ma

    # Capital requirement
    capital = ul * ead
    rwa = capital / _MIN_CAPITAL_RATIO

    return {
        "pd_stress":   float(pd_stress),
        "ul":          float(ul),
        "capital":     float(capital),
        "rwa":         float(rwa),
        "el":          float(pd_ * lgd * ead),
    }


class RegulatoryCapitalCalculator:
    """
    Computes regulatory capital requirements under both Standardised and IRB approaches
    for a portfolio of Bitcoin transaction fraud exposures.
    """

    def __init__(
        self,
        total_exposure: float,
        fraud_probability: float,
        lgd: float = 0.45,
        rho: float = 0.12,
        exposure_class: str = "retail",
    ):
        self.total_exposure    = total_exposure
        self.fraud_probability = np.clip(fraud_probability, 1e-6, 1 - 1e-6)
        self.lgd               = lgd
        self.rho               = rho
        self.exposure_class    = exposure_class

    # ── Standardised Approach ─────────────────────────────────────────────────

    def standardised_approach(self) -> Dict[str, float]:
        """Basel III SA: RWA = EAD × risk weight; Capital = RWA × 8%."""
        rw  = _SA_RISK_WEIGHTS.get(self.exposure_class, 1.0)
        rwa = self.total_exposure * rw
        capital = rwa * _MIN_CAPITAL_RATIO
        total_capital = rwa * _TOTAL_MIN
        return {
            "approach":            "Standardised (SA)",
            "risk_weight":         rw,
            "rwa":                 float(rwa),
            "min_capital":         float(capital),
            "total_capital":       float(total_capital),
            "tier1_ratio_at_min":  _MIN_CAPITAL_RATIO,
            "el":                  float(self.fraud_probability * self.lgd * self.total_exposure),
        }

    # ── IRB Approach ──────────────────────────────────────────────────────────

    def irb_approach(self, confidence: float = 0.999) -> Dict[str, float]:
        """Basel III IRB: Vasicek single-factor model at 99.9% confidence."""
        v = _vasicek_capital(
            pd_=self.fraud_probability,
            lgd=self.lgd,
            ead=self.total_exposure,
            rho=self.rho,
            confidence=confidence,
        )
        total_capital = v["capital"] * (_TOTAL_MIN / _MIN_CAPITAL_RATIO)
        return {
            "approach":        "IRB (Vasicek)",
            "pd_input":        self.fraud_probability,
            "pd_stressed":     v["pd_stress"],
            "lgd":             self.lgd,
            "rho":             self.rho,
            "ul":              v["ul"],
            "rwa":             v["rwa"],
            "min_capital":     v["capital"],
            "total_capital":   float(total_capital),
            "el":              v["el"],
            "ul_over_el":      v["ul"] / (v["el"] / self.total_exposure + 1e-9),
        }

    # ── Sensitivity: rho sweep ────────────────────────────────────────────────

    def rho_sensitivity(self, rho_range: np.ndarray | None = None) -> pd.DataFrame:
        """Capital requirement across a range of asset correlations."""
        if rho_range is None:
            rho_range = np.linspace(_RHO_MIN, _RHO_MAX, 30)
        rows = []
        for rho in rho_range:
            v = _vasicek_capital(
                pd_=self.fraud_probability, lgd=self.lgd,
                ead=self.total_exposure, rho=float(rho),
            )
            rows.append({"rho": round(float(rho), 4),
                         "capital": round(v["capital"], 2),
                         "rwa": round(v["rwa"], 2)})
        return pd.DataFrame(rows)

    # ── Sensitivity: PD sweep ─────────────────────────────────────────────────

    def pd_sensitivity(self, pd_range: np.ndarray | None = None) -> pd.DataFrame:
        """Capital requirement across a range of fraud probabilities."""
        if pd_range is None:
            pd_range = np.linspace(0.01, 0.60, 50)
        rows = []
        for pd_ in pd_range:
            v = _vasicek_capital(
                pd_=float(pd_), lgd=self.lgd,
                ead=self.total_exposure, rho=self.rho,
            )
            rows.append({"pd": round(float(pd_), 4),
                         "capital": round(v["capital"], 2),
                         "el": round(v["el"], 2)})
        return pd.DataFrame(rows)

    # ── Comparison ────────────────────────────────────────────────────────────

    def compare(self) -> Dict[str, Any]:
        sa  = self.standardised_approach()
        irb = self.irb_approach()
        diff = irb["min_capital"] - sa["min_capital"]
        return {
            "sa":           sa,
            "irb":          irb,
            "capital_diff": float(diff),
            "irb_is_lower": diff < 0,
            "saving_pct":   float(abs(diff) / sa["min_capital"] * 100) if sa["min_capital"] > 0 else 0.0,
        }
