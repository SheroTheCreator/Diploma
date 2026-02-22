"""
Risk and performance metrics calculator.

Implements risk-adjusted performance metrics.
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
    """

    def __init__(self, periods_per_year: int = 252) -> None:
        """
        Initialize calculator with annualization parameter.
        
        :param periods_per_year: Number of trading periods per year
        """
        self.periods_per_year = periods_per_year

    def _convert_annual_rf_to_periodic(self, risk_free_rate: float) -> float:
        """Convert annual risk-free rate to periodic rate."""
        return risk_free_rate / float(self.periods_per_year)

    def sharpe_ratio(
        self,
        returns: pd.Series,
        risk_free_rate_annual: float,
    ) -> float:
        """
        Compute annualized Sharpe ratio.
        
        :param returns: Periodic returns of the asset or portfolio
        :param risk_free_rate_annual: Annual risk-free rate
        :return: Annualized Sharpe ratio
        """
        rf_periodic = self._convert_annual_rf_to_periodic(risk_free_rate_annual)

        excess_periodic = returns - rf_periodic
        mean_excess_periodic = excess_periodic.mean()
        std_periodic = returns.std()

        if std_periodic == 0:
            return float("nan")

        mean_excess_annual = mean_excess_periodic * self.periods_per_year
        std_annual = std_periodic * np.sqrt(self.periods_per_year)

        return float(mean_excess_annual / std_annual)

    def value_at_risk(
        self,
        returns: pd.Series,
        level: float = 0.95,
        method: Literal["historical"] = "historical",
    ) -> float:
        """
        Compute Value at Risk (VaR) at the given confidence level.
        
        :param returns: Periodic returns
        :param level: Confidence level
        :param method: 'historical' (empirical quantile)
        :return: VaR as positive number representing loss
        """
        if method == "historical":
            sorted_returns = returns.sort_values()
            var_quantile = sorted_returns.quantile(1.0 - level)
            return float(-var_quantile)
        
        else:
            raise ValueError(f"Unsupported VaR method: {method}")

    def var_result(
        self,
        returns: pd.Series,
        level: float = 0.95,
        method: Literal["historical"] = "historical",
    ) -> VaRResult:
        """Convenience method to compute VaR and return structured result."""
        var_value = self.value_at_risk(returns, level=level, method=method)
        return VaRResult(level=level, var=var_value)
