"""
Risk and return statistics calculator.
"""

import numpy as np
import pandas as pd


def calculate_annualized_return(returns: pd.DataFrame, periods_per_year: int = 252) -> pd.Series:
    """Calculate annualized expected returns."""
    return returns.mean() * periods_per_year


def calculate_annualized_volatility(returns: pd.DataFrame, periods_per_year: int = 252) -> pd.Series:
    """Calculate annualized volatility (standard deviation)."""
    return returns.std() * np.sqrt(periods_per_year)


def calculate_covariance_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Calculate covariance matrix of returns."""
    return returns.cov()


def calculate_correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """Calculate correlation matrix of returns with values in [-1, 1]."""
    return returns.corr()
