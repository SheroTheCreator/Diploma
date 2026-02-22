"""
Risk and performance metrics calculator.
"""

import numpy as np
import pandas as pd


def calculate_sharpe_ratio(
    returns: pd.Series, 
    risk_free_rate_annual: float, 
    periods_per_year: int = 252
) -> float:
    """Compute annualized Sharpe ratio."""
    rf_periodic = risk_free_rate_annual / float(periods_per_year)
    excess_periodic = returns - rf_periodic
    
    std_periodic = returns.std()
    if std_periodic == 0:
        return float("nan")

    mean_excess_annual = excess_periodic.mean() * periods_per_year
    std_annual = std_periodic * np.sqrt(periods_per_year)

    return float(mean_excess_annual / std_annual)


def calculate_value_at_risk(returns: pd.Series, level: float = 0.95) -> float:
    """
    Compute historical Value at Risk (VaR) at the given confidence level.
    Returns VaR as a positive number representing loss.
    """
    sorted_returns = returns.sort_values()
    var_quantile = sorted_returns.quantile(1.0 - level)
    return float(-var_quantile)
