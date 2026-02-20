"""
Risk and performance metrics calculator.

Implements metrics from Sections 4.1-4.3 of the thesis:
- Volatility and standard deviation (4.1)
- Sharpe ratio (4.2)
- Value at Risk (4.3)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd


@dataclass
class VaRResult:
    """Container for Value at Risk calculation results."""
    level: float
    var: float


class MetricsCalculator:
    """
    Computes risk-adjusted performance metrics for portfolio analysis.
    
    Based on classical portfolio theory metrics discussed in Chapter 4
    of the thesis.[file:2]
    """

    def __init__(self, periods_per_year: int = 252) -> None:
        """
        Initialize calculator with annualization parameter.
        
        :param periods_per_year: Number of trading periods per year (252 for daily data)
        """
        self.periods_per_year = periods_per_year

    # ---------- Helper methods ----------

    def _convert_annual_rf_to_periodic(self, risk_free_rate: float) -> float:
        """
        Convert annual risk-free rate to periodic rate.
        
        Uses simple approximation: r_periodic ≈ r_annual / periods_per_year.
        """
        return risk_free_rate / float(self.periods_per_year)

    # ---------- Risk-adjusted return metrics ----------

    def sharpe_ratio(
        self,
        returns: pd.Series,
        risk_free_rate_annual: float,
    ) -> float:
        """
        Compute annualized Sharpe ratio (Section 4.2, Formula 6).[file:2]
        
        S = (E[R_p] - R_f) / σ_p
        
        The Sharpe ratio shows how many units of surplus return (above the 
        risk-free rate) the portfolio generates for each unit of risk.
        
        :param returns: Periodic returns (e.g. daily) of the asset or portfolio
        :param risk_free_rate_annual: Annual risk-free rate (e.g. 0.02 for 2%)
        :return: Annualized Sharpe ratio
        """
        rf_periodic = self._convert_annual_rf_to_periodic(risk_free_rate_annual)

        excess_periodic = returns - rf_periodic
        mean_excess_periodic = excess_periodic.mean()
        std_periodic = returns.std()

        if std_periodic == 0:
            return float("nan")

        # Annualize: mean * periods_per_year and std * sqrt(periods_per_year)
        mean_excess_annual = mean_excess_periodic * self.periods_per_year
        std_annual = std_periodic * np.sqrt(self.periods_per_year)

        return float(mean_excess_annual / std_annual)

    # ---------- Value at Risk ----------

    def value_at_risk(
        self,
        returns: pd.Series,
        level: float = 0.95,
        method: Literal["historical", "parametric"] = "historical",
    ) -> float:
        """
        Compute Value at Risk (VaR) at the given confidence level (Section 4.3).[file:2]
        
        VaR answers: "What is the worst loss we can expect with a given probability?"
        For example, a 95% VaR of 2% means there is only a 5% chance that 
        losses will exceed 2% in a single period.
        
        :param returns: Periodic returns
        :param level: Confidence level (e.g. 0.95 for 95%)
        :param method: 'historical' (empirical quantile) or 'parametric' (normal assumption)
        :return: VaR as positive number representing loss (e.g. 0.02 = 2%)
        """
        if method == "historical":
            # Historical quantile of losses
            sorted_returns = returns.sort_values()
            var_quantile = sorted_returns.quantile(1.0 - level)
            return float(-var_quantile)
        
        elif method == "parametric":
            # Assuming normal distribution: VaR = -(mu + z * sigma)
            mu = returns.mean()
            sigma = returns.std()
            z = self._z_score(level)
            var_value = -(mu + z * sigma)
            return float(max(var_value, 0.0))
        
        else:
            raise ValueError(f"Unsupported VaR method: {method}")

    def var_result(
        self,
        returns: pd.Series,
        level: float = 0.95,
        method: Literal["historical", "parametric"] = "historical",
    ) -> VaRResult:
        """
        Convenience method to compute VaR and return structured result.
        """
        var_value = self.value_at_risk(returns, level=level, method=method)
        return VaRResult(level=level, var=var_value)

    # ---------- Internal helpers ----------

    @staticmethod
    def _z_score(level: float) -> float:
        """
        Approximate z-score for the standard normal quantile.
        
        Used in parametric VaR calculation.
        """
        # Common values
        if np.isclose(level, 0.90):
            return 1.2816
        if np.isclose(level, 0.95):
            return 1.6449
        if np.isclose(level, 0.99):
            return 2.3263
        
        # Fallback: use scipy if available
        try:
            from scipy.stats import norm
            return float(norm.ppf(level))
        except ImportError:
            # Simple approximation if scipy not available
            if level > 0.5:
                return float(np.sqrt(2) * np.sqrt(-np.log(2 * (1 - level))))
            else:
                return 0.0
