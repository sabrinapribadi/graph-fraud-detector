"""
Quantitative Risk Analysis for Fraud Detection
Monte Carlo simulations, TVM calculations, Expected Loss
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, Tuple, Optional
import logging
from scipy import stats

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QuantitativeRiskAnalyzer:
    """
    Quantitative risk analysis for fraud detection
    """
    def __init__(self, discount_rate: float = 0.10):
        self.discount_rate = discount_rate
        
    def monte_carlo_simulation(
        self,
        n_transactions: int,
        fraud_probability: float,
        avg_loss_per_fraud: float,
        n_simulations: int = 10000,
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """Run Monte Carlo simulation"""
        logger.info(f"Running Monte Carlo simulation with {n_simulations} iterations...")
        
        fraud_counts = np.random.binomial(n_transactions, fraud_probability, n_simulations)
        
        losses = fraud_counts * avg_loss_per_fraud
        loss_variation = np.random.normal(1.0, 0.2, n_simulations)
        losses_varied = losses * loss_variation
        losses_varied = np.maximum(losses_varied, 0)
        
        lower_percentile = (1 - confidence_level) / 2 * 100
        upper_percentile = (1 + confidence_level) / 2 * 100
        
        results = {
            'mean_loss': np.mean(losses_varied),
            'median_loss': np.median(losses_varied),
            'std_loss': np.std(losses_varied),
            'lower_bound': np.percentile(losses_varied, lower_percentile),
            'upper_bound': np.percentile(losses_varied, upper_percentile),
            'max_loss': np.max(losses_varied),
            'min_loss': np.min(losses_varied),
            'value_at_risk': np.percentile(losses_varied, 5),
            'expected_shortfall': np.mean(losses_varied[losses_varied > np.percentile(losses_varied, 95)]),
            'simulations': losses_varied,
            'fraud_counts': fraud_counts
        }
        
        return results
    
    def calculate_expected_loss(
        self,
        exposure: float,
        probability_default: float,
        loss_given_default: float
    ) -> Dict[str, float]:
        """Calculate Expected Loss"""
        expected_loss = exposure * probability_default * loss_given_default
        
        return {
            'expected_loss': expected_loss,
            'exposure': exposure,
            'probability_default': probability_default,
            'loss_given_default': loss_given_default,
            'risk_weighted_asset': exposure * 0.5,
            'capital_requirement': expected_loss * 0.08
        }
    
    def time_value_money_adjustment(
        self,
        expected_loss: float,
        detection_time: float,
        discount_rate: Optional[float] = None
    ) -> Dict[str, float]:
        """Calculate TVM-adjusted fraud cost"""
        if discount_rate is None:
            discount_rate = self.discount_rate
        
        daily_rate = discount_rate / 365
        present_value = expected_loss / ((1 + daily_rate) ** detection_time)
        time_value_cost = expected_loss - present_value
        
        return {
            'present_value': present_value,
            'future_value': expected_loss,
            'time_value_cost': time_value_cost,
            'detection_time_days': detection_time,
            'discount_rate': discount_rate,
            'cost_per_day': time_value_cost / detection_time
        }
    
    def full_risk_assessment(
        self,
        n_transactions: int,
        fraud_probability: float,
        avg_loss_per_fraud: float,
        exposure_per_transaction: float,
        detection_time: float = 30,
        n_simulations: int = 10000
    ) -> Dict[str, Any]:
        """Complete risk assessment combining all methods"""
        monte_carlo = self.monte_carlo_simulation(
            n_transactions=n_transactions,
            fraud_probability=fraud_probability,
            avg_loss_per_fraud=avg_loss_per_fraud,
            n_simulations=n_simulations
        )
        
        expected_loss = self.calculate_expected_loss(
            exposure=n_transactions * exposure_per_transaction,
            probability_default=fraud_probability,
            loss_given_default=0.5
        )
        
        tvm = self.time_value_money_adjustment(
            expected_loss=monte_carlo['mean_loss'],
            detection_time=detection_time
        )
        
        return {
            'monte_carlo': monte_carlo,
            'expected_loss': expected_loss,
            'tvm_adjusted': tvm,
            'total_risk_score': monte_carlo['mean_loss'] + tvm['time_value_cost'],
            'risk_metrics': {
                'value_at_risk_95': monte_carlo['value_at_risk'],
                'expected_shortfall': monte_carlo['expected_shortfall'],
                'capital_requirement': expected_loss['capital_requirement'],
                'cost_of_delay': tvm['time_value_cost']
            }
        }